"""Experimental split-K GQA attention on a shared token-major FP16 KV pool."""
import torch
import triton
import triton.language as tl


@triton.jit
def _partial(Q, K, V, SHARED, EX, SC, EC, PARTIAL,
             LENGTH: tl.constexpr, GROUPS: tl.constexpr, SW: tl.constexpr, EW: tl.constexpr,
             PARTS: tl.constexpr, BLOCK: tl.constexpr):
    bg = tl.program_id(0)
    part = tl.program_id(1)
    batch = bg // GROUPS
    group = bg % GROUPS
    h = tl.arange(0, 16)
    d = tl.arange(0, 128)
    n = tl.arange(0, BLOCK)
    valid_h = h < 4
    q = tl.load(Q + (batch * GROUPS * 4 + group * 4 + h[:, None]) * 128 + d[None, :],
                mask=valid_h[:, None], other=0)
    count = tl.load(SC + bg)
    span = tl.cdiv(count, PARTS)
    pos = part * span + n
    valid_token = (n < span) & (pos < count)
    token = tl.load(SHARED + bg * SW + pos, mask=valid_token, other=0)
    key = tl.load(K + (token[None, :] * GROUPS + group) * 128 + d[:, None],
                  mask=valid_token[None, :], other=0)
    logits = tl.dot(q, key) * 0.08838834764831845
    logits = tl.where(valid_h[:, None] & valid_token[None, :], logits, float('-inf'))
    ex_count = tl.load(EC + batch * GROUPS * 4 + group * 4 + h, mask=valid_h, other=0)
    has_ex = valid_h & (part < ex_count) & (part < EW)
    ex_token = tl.load(EX + (batch * GROUPS * 4 + group * 4 + h) * EW + part,
                       mask=has_ex, other=0)
    ex_k = tl.load(K + (ex_token[:, None] * GROUPS + group) * 128 + d[None, :],
                   mask=has_ex[:, None], other=0).to(tl.float32)
    ex_logits = tl.sum(q.to(tl.float32) * ex_k, 1) * 0.08838834764831845
    ex_logits = tl.where(has_ex, ex_logits, float('-inf'))
    maximum = tl.maximum(tl.max(logits, 1), ex_logits)
    stable_max = tl.where(maximum == float('-inf'), 0.0, maximum)
    prob = tl.exp(logits - stable_max[:, None])
    ex_prob = tl.exp(ex_logits - stable_max)
    val = tl.load(V + (token[:, None] * GROUPS + group) * 128 + d[None, :],
                  mask=valid_token[:, None], other=0)
    acc = tl.dot(prob.to(tl.float16), val)
    ex_v = tl.load(V + (ex_token[:, None] * GROUPS + group) * 128 + d[None, :],
                   mask=has_ex[:, None], other=0).to(tl.float32)
    acc += ex_prob[:, None] * ex_v
    denom = tl.sum(prob, 1) + ex_prob
    base = ((bg * PARTS + part) * 4 + h) * 130
    tl.store(PARTIAL + base[:, None] + d[None, :], acc, mask=valid_h[:, None])
    tl.store(PARTIAL + base + 128, maximum, mask=valid_h)
    tl.store(PARTIAL + base + 129, denom, mask=valid_h)


@triton.jit
def _merge(PARTIAL, OUT, GROUPS: tl.constexpr, PARTS: tl.constexpr):
    bh = tl.program_id(0)
    batch = bh // (GROUPS * 4)
    head = bh % (GROUPS * 4)
    group = head // 4
    h = head % 4
    parts = tl.arange(0, PARTS)
    dims = tl.arange(0, 128)
    base = (((batch * GROUPS + group) * PARTS + parts) * 4 + h) * 130
    maximum = tl.load(PARTIAL + base + 128)
    denom = tl.load(PARTIAL + base + 129)
    total_max = tl.max(maximum, 0)
    weights = tl.where(denom > 0, tl.exp(maximum - total_max), 0.0)
    result = tl.sum(tl.load(PARTIAL + base[:, None] + dims[None, :]) * weights[:, None], 0)
    total = tl.sum(denom * weights, 0)
    tl.store(OUT + bh * 128 + dims, result / total)


def decode(q, k, v, shared, exceptions, shared_counts, exception_counts, parts=16, block=128):
    if q.dtype != torch.float16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError('FP16 Q/K/V required')
    if not all(x.is_cuda and x.is_contiguous() for x in (q, k, v, shared, exceptions, shared_counts, exception_counts)):
        raise ValueError('contiguous CUDA inputs required')
    if q.ndim != 3 or k.ndim != 3 or k.shape != v.shape or q.shape[1] != 4 * k.shape[1] or q.shape[2] != 128 or k.shape[2] != 128:
        raise ValueError('expected Q[B,4G,128] and K/V[L,G,128]')
    batch, heads, _ = q.shape
    groups = k.shape[1]
    if shared.shape[:2] != (batch, groups) or exceptions.shape[:2] != (batch, heads):
        raise ValueError('invalid support shape')
    if shared_counts.shape != (batch, groups) or exception_counts.shape != (batch, heads):
        raise ValueError('invalid count shape')
    if any(x.dtype != torch.int32 for x in (shared, exceptions, shared_counts, exception_counts)):
        raise ValueError('int32 support metadata required')
    if parts not in (8, 16, 32) or block not in (64, 128, 256) or exceptions.shape[-1] > parts:
        raise ValueError('unsupported parts, block or exception width')
    # Span must fit one tile for every group; width is a conservative bound.
    if triton.cdiv(shared.shape[-1], parts) > block:
        raise ValueError('shared support requires a larger tile or more parts')
    partial = torch.empty((batch, groups, parts, 4, 130), device=q.device, dtype=torch.float32)
    output = torch.empty_like(q)
    _partial[(batch * groups, parts)](q, k, v, shared, exceptions, shared_counts,
                                      exception_counts, partial, k.shape[0], groups,
                                      shared.shape[-1], exceptions.shape[-1], parts, block,
                                      num_warps=4)
    _merge[(batch * heads,)](partial, output, groups, parts, num_warps=4)
    return output
