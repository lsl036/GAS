"""Two-pass grouped mask compaction for a fixed-capacity FI paged plan."""
import torch
import triton
import triton.language as tl


@triton.jit
def _counts(MASK, COUNTS, LENGTH: tl.constexpr, TILES: tl.constexpr, TILE: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    batch, group = row // 8, row % 8
    tokens = tile * TILE + tl.arange(0, TILE)
    valid = tokens < LENGTH
    union = tl.full((TILE,), False, tl.int1)
    for head in range(4):
        union |= tl.load(MASK + (batch * 32 + group * 4 + head) * LENGTH + tokens,
                         mask=valid, other=False) != 0
    tl.store(COUNTS + row * TILES + tile, tl.sum(union.to(tl.int32), 0))


@triton.jit
def _scatter(MASK, PREFIX, CAP, KVPTR, IDS, BITS,
             LENGTH: tl.constexpr, TILES: tl.constexpr, TILE: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    batch, group = row // 8, row % 8
    tokens = tile * TILE + tl.arange(0, TILE)
    valid = tokens < LENGTH
    m0 = tl.load(MASK + (batch * 32 + group * 4) * LENGTH + tokens, mask=valid, other=False)
    m1 = tl.load(MASK + (batch * 32 + group * 4 + 1) * LENGTH + tokens, mask=valid, other=False)
    m2 = tl.load(MASK + (batch * 32 + group * 4 + 2) * LENGTH + tokens, mask=valid, other=False)
    m3 = tl.load(MASK + (batch * 32 + group * 4 + 3) * LENGTH + tokens, mask=valid, other=False)
    union = (m0 | m1 | m2 | m3) != 0
    prior = tl.load(PREFIX + row * TILES + tile - 1, mask=tile > 0, other=0)
    rank = prior + tl.cumsum(union.to(tl.int32), 0) - 1
    base = tl.load(KVPTR + row)
    capacity = tl.load(CAP + row)
    tl.store(IDS + base + rank, tokens * 8 + group, mask=union)
    tl.store(BITS + base * 4 + rank, m0, mask=union)
    tl.store(BITS + base * 4 + capacity + rank, m1, mask=union)
    tl.store(BITS + base * 4 + capacity * 2 + rank, m2, mask=union)
    tl.store(BITS + base * 4 + capacity * 3 + rank, m3, mask=union)


@triton.jit
def _pack(BITS, PACKED, NBYTES: tl.constexpr, BLOCK: tl.constexpr):
    byte = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    bit = tl.arange(0, 8)
    values = tl.load(BITS + byte[:, None] * 8 + bit[None, :],
                     mask=byte[:, None] < NBYTES, other=False).to(tl.int32)
    packed = tl.sum(values << bit[None, :], 1)
    tl.store(PACKED + byte, packed.to(tl.uint8), mask=byte < NBYTES)


def update(plan, mask, tile=512):
    if mask.shape != (plan.batch, 32, plan.length) or mask.device != plan.device or mask.dtype != torch.bool or not mask.is_contiguous():
        raise ValueError('mask must be contiguous CUDA bool [B,32,L] on the planned device')
    rows = plan.batch * 8
    tiles = triton.cdiv(plan.length, tile)
    counts = torch.empty((rows, tiles), device=plan.device, dtype=torch.int32)
    _counts[(rows, tiles)](mask, counts, plan.length, tiles, tile, num_warps=4)
    prefix = counts.cumsum(1, dtype=torch.int32)
    if bool((prefix[:, -1] > plan.capacities).any().item()):
        raise ValueError('group support exceeds fixed plan capacity; replan required')
    plan.mask_bits.zero_()
    _scatter[(rows, tiles)](mask, prefix, plan.capacities, plan.kvptr, plan.indices,
                            plan.mask_bits, plan.length, tiles, tile, num_warps=4)
    _pack[(triton.cdiv(plan.packed.numel(), 256),)](plan.mask_bits, plan.packed,
                                                    plan.packed.numel(), 256, num_warps=4)
