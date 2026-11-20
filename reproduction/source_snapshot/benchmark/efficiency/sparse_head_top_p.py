"""Experimental compact-candidate head top-p, not integrated into inference."""
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice

from gas_compact_refinement import buckets


@triton.jit
def _head_top_p(IDS, COUNTS, SCORES, OUT_MASK, OUT_IDS, OUT_COUNTS, AMBIGUOUS, HEAD_EPOCHS,
                L: tl.constexpr, K: tl.constexpr, P: tl.constexpr,
                BLOCK: tl.constexpr, LOW: tl.constexpr, LIMIT: tl.constexpr,
                DOUBLE_SUM: tl.constexpr, ADAPTIVE: tl.constexpr, GUARD: tl.constexpr,
                TAGGED_OUTPUT: tl.constexpr):
    row = tl.program_id(0)
    count = tl.load(COUNTS + row)
    if count <= LOW or count > LIMIT:
        return
    i = tl.arange(0, BLOCK)
    token = tl.load(IDS + row * K + i, mask=(i < count) & (i < K), other=-1)
    valid = (i < count) & (i < K) & (token >= 0) & (token < L)
    value = tl.load(SCORES + row * L + token, mask=valid, other=-float('inf')).to(tl.float32)
    maximum = tl.max(value, axis=0)
    safe_maximum = tl.where(count > 0, maximum, 0.0)
    mass = tl.where(valid, libdevice.exp(value - safe_maximum), 0.0)
    if DOUBLE_SUM:
        mass = mass.to(tl.float64)
    total = tl.sum(mass, axis=0)
    if DOUBLE_SUM:
        prob = libdevice.div_rn(mass, tl.where(total > 0, total, 1.0))
    else:
        prob = tl.div_rn(mass, tl.where(total > 0, total, 1.0))
    if DOUBLE_SUM:
        low = tl.full((), 0.0, tl.float64)
        threshold = tl.full((), P, tl.float32).to(tl.float64)
        delta = tl.full((), 1.0, tl.float64)
    else:
        low = tl.full((), 0.0, tl.float32)
        threshold = tl.full((), P, tl.float32)
        delta = tl.full((), 1.0, tl.float32)
    high = tl.sum(tl.where(valid & (token == 0), prob, 0.0), axis=0)
    ambiguous = tl.full((), False, tl.int1)
    while delta > 1.0e-5:
        mid = (low + high) * 0.5
        min_gt_low = tl.min(tl.where(prob > low, prob, high), axis=0)
        max_le_high = tl.max(tl.where(prob <= high, prob, low), axis=0)
        above = tl.sum(tl.where(prob > mid, prob, 0.0), axis=0)
        if ADAPTIVE:
            ambiguous = ambiguous | (tl.abs(above - threshold) <= GUARD)
        low = tl.where(above >= threshold, mid, low)
        high = tl.where(above >= threshold, high, tl.minimum(mid, max_le_high))
        delta = max_le_high - min_gt_low
    if ADAPTIVE:
        tl.store(AMBIGUOUS + row, ambiguous.to(tl.int32))
        if ambiguous:
            mass64 = mass.to(tl.float64)
            total64 = tl.sum(mass64, axis=0)
            prob64 = libdevice.div_rn(mass64, tl.where(total64 > 0, total64, 1.0))
            low64 = tl.full((), 0.0, tl.float64)
            high64 = tl.sum(tl.where(valid & (token == 0), prob64, 0.0), axis=0)
            delta64 = tl.full((), 1.0, tl.float64)
            threshold64 = tl.full((), P, tl.float32).to(tl.float64)
            while delta64 > 1.0e-5:
                mid64 = (low64 + high64) * 0.5
                min64 = tl.min(tl.where(prob64 > low64, prob64, high64), axis=0)
                max64 = tl.max(tl.where(prob64 <= high64, prob64, low64), axis=0)
                above64 = tl.sum(tl.where(prob64 > mid64, prob64, 0.0), axis=0)
                low64 = tl.where(above64 >= threshold64, mid64, low64)
                high64 = tl.where(above64 >= threshold64, high64, tl.minimum(mid64, max64))
                delta64 = max64 - min64
            keep = valid & (prob64 > low64)
        else:
            keep = valid & (prob > low)
    else:
        keep = valid & (prob > low)
    rank = tl.cumsum(keep.to(tl.int32), axis=0) - 1
    kept = tl.sum(keep.to(tl.int32), axis=0)
    if TAGGED_OUTPUT:
        epoch = tl.load(HEAD_EPOCHS + row) + 1
        tl.store(HEAD_EPOCHS + row, epoch)
        tl.store(OUT_MASK + row * L + token, epoch, mask=keep)
    else:
        tl.store(OUT_MASK + row * L + token, True, mask=keep)
    tl.store(OUT_IDS + row * K + rank, token, mask=keep)
    tl.store(OUT_COUNTS + row, kept)


def sparse_head_top_p(indices, counts, scores, output_mask, output_ids, output_counts, threshold=0.85,
                      double_sum=False, adaptive_sum=False, ambiguous_out=None, guard=1.0e-6,
                      head_epochs=None):
    """Top-p on compact candidates; inputs must contain unique, valid indices.

    Scores must be finite and candidates/counts must describe the full input
    selection. Extremely negative logits near the dtype minimum can differ
    from the dense implementation's finite masked-value semantics. Output mask
    clearing and all bucket launches are included.
    """
    batch, heads, capacity = indices.shape
    if (indices.dtype != torch.int32 or counts.shape != (batch, heads) or
            scores.shape[:2] != (batch, heads) or output_mask.shape != scores.shape or
            output_ids.shape != indices.shape or output_counts.shape != counts.shape):
        raise ValueError('incompatible compact top-p tensors')
    if double_sum and adaptive_sum:
        raise ValueError('choose double_sum or adaptive_sum, not both')
    if adaptive_sum and (ambiguous_out is None or ambiguous_out.shape != counts.shape or
                         ambiguous_out.dtype != torch.int32 or guard <= 0):
        raise ValueError('adaptive mode requires matching int32 ambiguity output and a positive guard')
    tagged = head_epochs is not None
    if tagged:
        if output_mask.dtype != torch.int32 or head_epochs.dtype != torch.int32 or head_epochs.shape != counts.shape:
            raise ValueError('tagged output requires int32 mask and per-head epochs')
    else:
        if output_mask.dtype != torch.bool:
            raise ValueError('untagged output requires boolean mask')
        output_mask.zero_()
    for low, limit in buckets(capacity):
        _head_top_p[(batch * heads,)](indices, counts, scores, output_mask, output_ids, output_counts,
                                      ambiguous_out if adaptive_sum else output_counts,
                                      head_epochs if tagged else output_counts,
                                      scores.shape[-1], capacity, threshold, triton.next_power_of_2(limit),
                                      low, limit, double_sum, adaptive_sum, guard, tagged, num_warps=4)
    return output_mask, output_ids, output_counts
