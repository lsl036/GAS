"""Build a validated, fixed-shape research plan for counted GAS decoding.

This planner runs on the host. Rebuild it whenever counts or batch change;
CPU synchronization and transfer are intentionally not part of decode timing.
"""
import torch


def build_plan(shared_counts, exception_counts, shared_width, exception_width, chunk_size, max_parts, device):
    if shared_counts.ndim != 2 or exception_counts.ndim != 2:
        raise ValueError("counts must be rank two")
    if shared_counts.shape[0] != exception_counts.shape[0] or exception_counts.shape[1] != 4 * shared_counts.shape[1]:
        raise ValueError("expected GQA group size four")
    if chunk_size <= 0 or max_parts < 1:
        raise ValueError("chunk_size and max_parts must be positive")
    sc, ec = shared_counts.detach().cpu(), exception_counts.detach().cpu()
    if sc.dtype != torch.int32 or ec.dtype != torch.int32:
        raise ValueError("counts must be int32")
    if bool((sc < 0).any()) or bool((sc > shared_width).any()) or bool((ec < 0).any()) or bool((ec > exception_width).any()):
        raise ValueError("counts out of range")
    batch, groups = sc.shape
    heads = groups * 4
    sc_rows, ec_rows = sc.tolist(), ec.tolist()
    jobs, parts, offsets, long_rows = [], [], [], []
    partial_count = 0
    for b in range(batch):
        for h in range(heads):
            ns, ne = sc_rows[b][h // 4], ec_rows[b][h]
            if ns + ne == 0:
                raise ValueError("every head must have nonempty support")
            n = min(max_parts, max(1, (max(ns, ne) + chunk_size - 1) // chunk_size))
            row = b * heads + h
            parts.append(n)
            offsets.append(partial_count if n > 1 else -1)
            if n > 1:
                long_rows.append(row)
                partial_count += n
            jobs.extend((row, p) for p in range(n))
    return {
        "jobs": torch.tensor(jobs, dtype=torch.int32, device=device).reshape(-1, 2),
        "parts": torch.tensor(parts, dtype=torch.int32, device=device).reshape(batch, heads),
        "offsets": torch.tensor(offsets, dtype=torch.int32, device=device).reshape(batch, heads),
        "long_rows": torch.tensor(long_rows, dtype=torch.int32, device=device),
        "partial_count": partial_count,
    }


def build_plan_gpu(shared_counts, exception_counts, chunk_size, max_parts):
    """Construct ragged jobs on GPU; values must already satisfy count bounds.

    `nonzero` has data-dependent output shape and may synchronize with the host.
    The resulting fixed plan must be rebuilt when workload shape or counts change.
    """
    if not shared_counts.is_cuda or not exception_counts.is_cuda or shared_counts.device != exception_counts.device:
        raise ValueError("counts must be on the same CUDA device")
    if shared_counts.ndim != 2 or exception_counts.ndim != 2:
        raise ValueError("counts must be rank two")
    if shared_counts.dtype != torch.int32 or exception_counts.dtype != torch.int32:
        raise ValueError("counts must be int32")
    if shared_counts.shape[0] != exception_counts.shape[0] or exception_counts.shape[1] != 4 * shared_counts.shape[1]:
        raise ValueError("expected GQA group size four")
    if chunk_size <= 0 or max_parts < 1:
        raise ValueError("chunk_size and max_parts must be positive")
    shared_for_head = shared_counts.repeat_interleave(4, dim=1)
    max_count = torch.maximum(shared_for_head, exception_counts)
    parts = torch.div(max_count + chunk_size - 1, chunk_size, rounding_mode="floor").clamp_(1, max_parts)
    flat = parts.reshape(-1)
    part_ids = torch.arange(max_parts, dtype=torch.int32, device=parts.device)
    rows, part = torch.nonzero(part_ids.unsqueeze(0) < flat.unsqueeze(1), as_tuple=True)
    jobs = torch.stack((rows, part), dim=1).to(torch.int32).contiguous()
    long_mask = flat > 1
    long_rows = torch.nonzero(long_mask, as_tuple=False).reshape(-1).to(torch.int32).contiguous()
    sizes = torch.where(long_mask, flat, 0)
    starts = torch.cumsum(sizes, dim=0, dtype=torch.int32) - sizes
    offsets = torch.where(long_mask, starts, -1).reshape_as(parts).contiguous()
    partial_count = jobs.shape[0] - (flat.numel() - long_rows.numel())
    return {"jobs": jobs, "parts": parts.contiguous(), "offsets": offsets,
            "long_rows": long_rows, "partial_count": partial_count}
