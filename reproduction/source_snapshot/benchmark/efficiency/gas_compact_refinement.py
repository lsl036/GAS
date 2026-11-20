"""Experimental group-max refinement from compact Twilight candidates.

Inputs are padded per-head token indices from an already executed head top-p,
plus the original head mask and estimator logits.  No dense GAS mask is built.
The caller must provide capacities large enough for every selected group token.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _groups(IDS, MASK, SCORES, HEAD_COUNTS, GROUP_IDS, GROUP_COUNTS, FLAGS,
            L: tl.constexpr, K: tl.constexpr, OUT: tl.constexpr,
            P: tl.constexpr, BLOCK: tl.constexpr, LIMIT: tl.constexpr,
            LOW: tl.constexpr, BUCKETED: tl.constexpr, RAGGED: tl.constexpr):
    row = tl.program_id(0)
    batch, group = row // 8, row % 8
    if BUCKETED:
        heads = tl.arange(0, 4)
        head_counts = tl.load(HEAD_COUNTS + batch * 32 + group * 4 + heads)
        maximum_count = tl.sum(head_counts, axis=0) if RAGGED else tl.max(head_counts, axis=0)
        if maximum_count <= LOW or maximum_count > LIMIT:
            return
    i = tl.arange(0, BLOCK)
    if RAGGED:
        base = HEAD_COUNTS + batch * 32 + group * 4
        n0, n1, n2, n3 = tl.load(base), tl.load(base + 1), tl.load(base + 2), tl.load(base + 3)
        head = tl.where(i < n0, 0, tl.where(i < n0 + n1, 1, tl.where(i < n0 + n1 + n2, 2, 3)))
        pos = i - tl.where(head == 0, 0, tl.where(head == 1, n0, tl.where(head == 2, n0 + n1, n0 + n1 + n2)))
        in_range = i < n0 + n1 + n2 + n3
    else:
        head = i // LIMIT
        pos = i % LIMIT
        in_range = head < 4
    token = tl.load(IDS + (batch * 32 + group * 4 + head) * K + pos,
                    mask=in_range & (pos < K), other=-1)
    valid = in_range & (pos < K) & (token >= 0) & (token < L)
    # Triton 3.1's bitonic sort multiplies values by zero internally; an
    # infinite sentinel would therefore generate NaNs during compare-and-swap.
    score = tl.full((BLOCK,), -1.0e30, tl.float32)
    duplicate = tl.full((BLOCK,), False, tl.int1)
    for h in range(4):
        offset = (batch * 32 + group * 4 + h) * L + token
        present = tl.load(MASK + offset, mask=valid, other=0) != 0
        value = tl.load(SCORES + offset, mask=valid & present, other=-1.0e30).to(tl.float32)
        score = tl.maximum(score, value)
        duplicate = duplicate | (present & (h < head))
    unique = valid & ~duplicate
    score = tl.where(unique, score, -1.0e30)
    maximum = tl.max(score, axis=0)
    safe_max = tl.where(maximum == -1.0e30, 0.0, maximum)
    mass = tl.where(unique, tl.exp(score - safe_max), 0.0)
    total = tl.sum(mass, axis=0)
    prob = mass / tl.where(total > 0, total, 1.0)
    # Match Twilight sampling.cuh: the binary-search upper bound is the
    # attention-sink probability at token 0, not the maximum probability.
    low = tl.full((), 0.0, tl.float32)
    high = tl.sum(tl.where(unique & (token == 0), prob, 0.0), axis=0)
    delta = tl.full((), 1.0, tl.float32)
    while delta > 1.0e-5:
        mid = (low + high) * 0.5
        min_gt_low = tl.min(tl.where(prob > low, prob, high), axis=0)
        max_le_high = tl.max(tl.where(prob <= high, prob, low), axis=0)
        above = tl.sum(tl.where(prob > mid, prob, 0.0), axis=0)
        low = tl.where(above >= P, mid, low)
        high = tl.where(above >= P, high, tl.minimum(mid, max_le_high))
        delta = max_le_high - min_gt_low
    keep = unique & (prob > low)
    rank = tl.cumsum(keep.to(tl.int32), axis=0) - 1
    count = tl.sum(keep.to(tl.int32), axis=0)
    tl.store(GROUP_IDS + row * OUT + rank, token, mask=keep & (rank < OUT))
    tl.store(FLAGS + row * L + token, 1, mask=keep)
    tl.store(GROUP_COUNTS + row, count)


@triton.jit
def _score_token_max(score_a, token_a, score_b, token_b):
    take_b = (score_b > score_a) | ((score_b == score_a) & (token_b < token_a))
    return tl.where(take_b, score_b, score_a), tl.where(take_b, token_b, token_a)


@triton.jit
def _exceptions(IDS, SCORES, HEAD_COUNTS, FLAGS, EPOCHS, EX_IDS, EX_COUNTS,
                L: tl.constexpr, K: tl.constexpr, E: tl.constexpr,
                BLOCK: tl.constexpr, LIMIT: tl.constexpr, LOW: tl.constexpr,
                BUCKETED: tl.constexpr, TAGGED: tl.constexpr, PAIR_REDUCE: tl.constexpr,
                SORT_REDUCE: tl.constexpr):
    row = tl.program_id(0)
    batch, head = row // 32, row % 32
    group = head // 4
    if BUCKETED:
        count = tl.load(HEAD_COUNTS + row)
        if count <= LOW or count > LIMIT:
            return
    i = tl.arange(0, BLOCK)
    token = tl.load(IDS + row * K + i, mask=(i < K) & (i < LIMIT), other=-1)
    valid = (i < K) & (i < LIMIT) & (token >= 0) & (token < L)
    if BUCKETED:
        valid = valid & (i < count)
    flag = tl.load(FLAGS + (batch * 8 + group) * L + token, mask=valid, other=0)
    if TAGGED:
        in_group = flag == tl.load(EPOCHS + batch * 8 + group)
    else:
        in_group = flag != 0
    residual = valid & ~in_group
    residual_count = tl.sum(residual.to(tl.int32), axis=0)
    count = residual_count
    if residual_count <= E:
        positions = tl.cumsum(residual.to(tl.int32), axis=0) - 1
        tl.store(EX_IDS + row * E + positions, token, mask=residual)
    elif SORT_REDUCE:
        score = tl.load(SCORES + row * L + token, mask=residual, other=-1.0e30).to(tl.float32)
        score = tl.where(score == 0.0, 0.0, score)
        bits = score.to(tl.uint32, bitcast=True)
        ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
        key = (ordered.to(tl.uint64) << 32) | (0xffffffff - token.to(tl.uint32)).to(tl.uint64)
        key = tl.where(residual & (score > -1.0e30), key, tl.full((BLOCK,), 0, tl.uint64))
        ordered_keys = tl.sort(key, descending=True)
        sorted_tokens = (0xffffffff - (ordered_keys & 0xffffffff).to(tl.uint32)).to(tl.int32)
        selected = (i < E) & (ordered_keys != 0)
        tl.store(EX_IDS + row * E + i, sorted_tokens, mask=selected)
        count = tl.sum(selected.to(tl.int32), axis=0)
    else:
        score = tl.load(SCORES + row * L + token, mask=residual, other=-1.0e30).to(tl.float32)
        count = tl.full((), 0, tl.int32)
        for rank in range(E):
            if PAIR_REDUCE:
                best, selected = tl.reduce((score, tl.where(residual, token, 2147483647)),
                                           axis=0, combine_fn=_score_token_max)
            else:
                best = tl.max(score, axis=0)
                selected = tl.min(tl.where(score == best, token, 2147483647), axis=0)
            found = best > -1.0e30
            tl.store(EX_IDS + row * E + rank, selected, mask=found)
            count += found.to(tl.int32)
            score = tl.where(token == selected, -1.0e30, score)
    tl.store(EX_COUNTS + row, count)


@triton.jit
def _exception_tile_sort(IDS, SCORES, COUNTS, FLAGS, EPOCHS, TMP,
                         L: tl.constexpr, K: tl.constexpr, TILES: tl.constexpr,
                         E: tl.constexpr, TILE: tl.constexpr, TAGGED: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    batch, head = row // 32, row % 32
    group = head // 4
    i = tl.arange(0, TILE)
    position = tile * TILE + i
    count = tl.load(COUNTS + row)
    if tile * TILE >= count:
        tl.store(TMP + (row * TILES + tile) * E + i, tl.full((TILE,), 0, tl.uint64), mask=i < E)
        return
    token = tl.load(IDS + row * K + position, mask=(position < K) & (position < count), other=-1)
    valid = (position < K) & (position < count) & (token >= 0) & (token < L)
    flag = tl.load(FLAGS + (batch * 8 + group) * L + token, mask=valid, other=0)
    if TAGGED:
        in_group = flag == tl.load(EPOCHS + batch * 8 + group)
    else:
        in_group = flag != 0
    residual = valid & ~in_group
    score = tl.load(SCORES + row * L + token, mask=residual, other=0).to(tl.float32)
    score = tl.where(score == 0.0, 0.0, score)
    bits = score.to(tl.uint32, bitcast=True)
    ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
    key = (ordered.to(tl.uint64) << 32) | (0xffffffff - token.to(tl.uint32)).to(tl.uint64)
    key = tl.where(residual, key, tl.full((TILE,), 0, tl.uint64))
    sorted_keys = tl.sort(key, descending=True)
    tl.store(TMP + (row * TILES + tile) * E + i, sorted_keys, mask=i < E)


@triton.jit
def _exception_merge_sort(TMP, EX_IDS, EX_COUNTS, TILES: tl.constexpr,
                          E: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    keys = tl.load(TMP + row * TILES * E + i, mask=i < TILES * E, other=0).to(tl.uint64)
    keys = tl.sort(keys, descending=True)
    valid = (i < E) & (keys != 0)
    token = (0xffffffff - (keys & 0xffffffff).to(tl.uint32)).to(tl.int32)
    tl.store(EX_IDS + row * E + i, token, mask=valid)
    tl.store(EX_COUNTS + row, tl.sum(valid.to(tl.int32), axis=0))


def exception_stage_tiled_sort(indices, scores, output, counts, scratch, epochs=None, tile=256):
    """Experimental finite-score Top-16; scratch preparation is caller-owned."""
    batch, _, k = indices.shape
    _, _, flags, ex_ids, ex_counts = output
    if tile not in (128, 256, 512, 1024) or ex_ids.shape[-1] != 16:
        raise ValueError('supported tile sizes are 128..1024 and exception budget is 16')
    tiles = triton.cdiv(k, tile)
    if scratch.shape != (batch, 32, tiles, 16) or scratch.dtype != torch.uint64:
        raise ValueError('incorrect tiled-sort scratch shape or dtype')
    _exception_tile_sort[(batch * 32, tiles)](indices, scores, counts, flags,
                                            epochs if epochs is not None else ex_counts,
                                            scratch, scores.shape[-1], k, tiles, 16, tile,
                                            epochs is not None, num_warps=4)
    _exception_merge_sort[(batch * 32,)](scratch, ex_ids, ex_counts, tiles, 16,
                                         triton.next_power_of_2(tiles * 16), num_warps=4)


def allocate(batch, k, length, device, exception_budget=16):
    return (torch.empty((batch, 8, 4 * k), dtype=torch.int32, device=device),
            torch.empty((batch, 8), dtype=torch.int32, device=device),
            torch.empty((batch, 8, length), dtype=torch.uint8, device=device),
            torch.empty((batch, 32, exception_budget), dtype=torch.int32, device=device),
            torch.empty((batch, 32), dtype=torch.int32, device=device))


def refine(indices, mask, scores, output, threshold=0.8, exception_budget=16, counts=None, ragged=False):
    if (indices.ndim != 3 or indices.shape[1] != 32 or indices.dtype != torch.int32
            or mask.shape != scores.shape or mask.shape[:2] != indices.shape[:2]
            or mask.dtype != torch.bool or not 0 < threshold < 1
            or not all(x.is_cuda and x.is_contiguous() for x in (indices, mask, scores))):
        raise ValueError('expected contiguous CUDA [B,32,K] int32 indices and matching [B,32,L] mask/logits')
    batch, _, k = indices.shape
    if k < exception_budget or exception_budget <= 0:
        raise ValueError('candidate capacity must cover a positive exception budget')
    group_ids, group_counts, flags, ex_ids, ex_counts = output
    length = mask.shape[-1]
    flags.zero_()
    group_stage(indices, mask, scores, output, threshold, counts, ragged)
    exception_stage(indices, scores, output, exception_budget, counts)
    return group_ids, group_counts, ex_ids, ex_counts


def buckets(k):
    result, low = [], -1
    for limit in (128, 512, 2048, 8192, 32768, k):
        limit = min(k, limit)
        if limit > low:
            result.append((low, limit))
            low = limit
        if limit == k:
            break
    return result


def group_stage(indices, mask, scores, output, threshold=0.8, counts=None, ragged=False):
    batch, _, k = indices.shape
    group_ids, group_counts, flags, _, _ = output
    length = mask.shape[-1]
    if ragged and counts is None:
        raise ValueError('ragged grouping requires compact head counts')
    for low, limit in buckets(4 * k if ragged else k) if counts is not None else [(-1, k)]:
        _groups[(batch * 8,)](indices, mask, scores, counts if counts is not None else group_counts,
                              group_ids, group_counts, flags, length, k, 4 * k, threshold,
                              triton.next_power_of_2(limit if ragged else 4 * limit), limit,
                              low, counts is not None, ragged, num_warps=8)


def exception_stage(indices, scores, output, exception_budget=16, counts=None, epochs=None,
                    pair_reduce=False, active_buckets=None, sort_reduce=False):
    batch, _, k = indices.shape
    _, _, flags, ex_ids, ex_counts = output
    length = scores.shape[-1]
    if active_buckets is not None and counts is None:
        raise ValueError('active buckets require compact head counts')
    if pair_reduce and sort_reduce:
        raise ValueError('choose only one experimental reduction strategy')
    ranges = active_buckets if active_buckets is not None else (buckets(k) if counts is not None else [(-1, k)])
    for low, limit in ranges:
        _exceptions[(batch * 32,)](indices, scores, counts if counts is not None else ex_counts,
                                   flags, epochs if epochs is not None else ex_counts, ex_ids, ex_counts,
                                   length, k, exception_budget, triton.next_power_of_2(limit), limit, low,
                                   counts is not None, epochs is not None, pair_reduce, sort_reduce, num_warps=4)


@triton.jit
def _exceptions_unified(IDS, SCORES, COUNTS, FLAGS, EPOCHS, EX_IDS, EX_COUNTS,
                        L: tl.constexpr, K: tl.constexpr, E: tl.constexpr, TAGGED: tl.constexpr):
    count = tl.load(COUNTS + tl.program_id(0))
    if count <= 128:
        _exceptions(IDS, SCORES, COUNTS, FLAGS, EPOCHS, EX_IDS, EX_COUNTS,
                    L, K, E, 128, 128, -1, True, TAGGED, False, False)
    elif count <= 512:
        _exceptions(IDS, SCORES, COUNTS, FLAGS, EPOCHS, EX_IDS, EX_COUNTS,
                    L, K, E, 512, 512, 128, True, TAGGED, False, False)
    elif count <= 2048:
        _exceptions(IDS, SCORES, COUNTS, FLAGS, EPOCHS, EX_IDS, EX_COUNTS,
                    L, K, E, 2048, 2048, 512, True, TAGGED, False, False)
    else:
        _exceptions(IDS, SCORES, COUNTS, FLAGS, EPOCHS, EX_IDS, EX_COUNTS,
                    L, K, E, 8192, 8192, 2048, True, TAGGED, False, False)


def exception_stage_unified(indices, scores, output, counts, epochs=None):
    """Experimental single-launch bucket dispatch for capacities up to 8192."""
    batch, _, k = indices.shape
    if k > 8192:
        raise ValueError('unified exception kernel supports capacity <= 8192')
    _, _, flags, ex_ids, ex_counts = output
    _exceptions_unified[(batch * 32,)](indices, scores, counts, flags,
                                      epochs if epochs is not None else ex_counts, ex_ids, ex_counts,
                                      scores.shape[-1], k, ex_ids.shape[-1], epochs is not None, num_warps=4)
