#!/usr/bin/env python3
"""Audit compact head top-p and GAS on each supplied captured layer."""
import argparse
import csv
import math
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from gas_compact_refinement import allocate, exception_stage
from sparse_head_top_p import sparse_head_top_p
from test_gas_compact_refinement import materialize, pack
from twilight.pyimpl.gqa_refine import build_group_refined_mask
from twilight.pyimpl.quantize import min_max_per_token_quant_kv
from twilight.pyimpl.top_p import top_p_unnormalized

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ['trace', 'seq_len', 'quest_capacity', 'head_mask_diff', 'head_trace_diff',
          'head_index_diff', 'gas_mask_diff', 'gas_trace_diff', 'ambiguous_heads', 'status', 'error']


@torch.inference_mode()
def audit(path, cub, double_sum, adaptive_sum, guard, fused_mass):
    trace = torch.load(path, map_location='cpu', weights_only=True)
    q, k = trace['q'].cuda(), trace['k'].cuda().repeat_interleave(4, dim=1)
    scores = (torch.matmul(q.unsqueeze(2), min_max_per_token_quant_kv(k, 4).transpose(-1, -2))
              / math.sqrt(q.shape[-1])).squeeze(2).contiguous()
    candidate = trace['candidate'].cuda().bool().contiguous()
    candidate_ids = pack(candidate)
    candidate_counts = candidate.sum(-1, dtype=torch.int32).contiguous()
    batch, heads, length = candidate.shape
    capacity = candidate_ids.shape[-1]
    mask = torch.empty_like(candidate)
    ids = torch.empty_like(candidate_ids)
    counts = torch.empty_like(candidate_counts)
    ambiguous = torch.empty_like(candidate_counts) if adaptive_sum else None
    sparse_head_top_p(candidate_ids, candidate_counts, scores, mask, ids, counts, double_sum=double_sum,
                      adaptive_sum=adaptive_sum, ambiguous_out=ambiguous, guard=guard)
    masked_scores = scores.masked_fill(~candidate, torch.finfo(scores.dtype).min)
    reference_mask = top_p_unnormalized(masked_scores.unsqueeze(2), .85).squeeze(2) & candidate
    head_diff = int((mask != reference_mask).sum())
    trace_head_diff = int((mask != trace['twilight'].cuda().bool()).sum())
    index_diff = 0
    for head in range(heads):
        n = int(counts[0, head])
        actual = ids[0, head, :n].long().sort().values
        expected = mask[0, head].nonzero().flatten()
        if not torch.equal(actual, expected):
            index_diff += 1
    out = allocate(batch, capacity, length, scores.device)
    out = (out[0], out[1], torch.zeros((batch, 8, length), dtype=torch.int32, device=scores.device), out[3], out[4])
    epochs = torch.zeros((batch, 8), dtype=torch.int32, device=scores.device)
    scratch = torch.empty((batch, 8, 4 * capacity), dtype=torch.float32, device=scores.device)
    fn = cub.group_tagged_fused_mass if fused_mass else cub.group_tagged_fused_reduce
    fn(ids, counts, mask, scores, out[0], out[1], out[2], epochs, scratch, .8)
    exception_stage(ids, scores, out, counts=counts, epochs=epochs)
    actual = materialize((out[0], out[1], out[3], out[4]), length)
    expected = build_group_refined_mask(scores.unsqueeze(2), reference_mask.unsqueeze(2), 8, .8, 'group_max', 16)
    gas_diff = int((actual != expected.squeeze(2).cpu()).sum())
    gas_trace_diff = int((actual != trace['gas'].bool()).sum())
    passed = not any((head_diff, trace_head_diff, index_diff, gas_diff, gas_trace_diff))
    return dict(seq_len=length, quest_capacity=capacity, head_mask_diff=head_diff,
                head_trace_diff=trace_head_diff, head_index_diff=index_diff,
                gas_mask_diff=gas_diff, gas_trace_diff=gas_trace_diff,
                ambiguous_heads=int(ambiguous.sum()) if adaptive_sum else '',
                status='pass' if passed else 'mismatch', error='')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace-dir', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--double-sum', action='store_true')
    p.add_argument('--adaptive-sum', action='store_true')
    p.add_argument('--guard', type=float, default=1.0e-6)
    p.add_argument('--fused-mass', action='store_true')
    args = p.parse_args()
    paths = []
    for directory in args.trace_dir:
        found = sorted(directory.glob('step001_layer*.pt'))
        if not found:
            p.error(f'no traces in {directory}')
        paths.extend(found)
    if args.output.exists():
        raise FileExistsError(args.output)
    cub = load(name='gas_compact_refine', sources=[str(ROOT / 'csrc/src/gas_compact_refine.cu')],
               extra_cuda_cflags=['-O3'], verbose=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    with args.output.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        f.flush()
        for i, path in enumerate(paths, 1):
            row = {'trace': str(path)}
            try:
                row.update(audit(path, cub, args.double_sum, args.adaptive_sum, args.guard, args.fused_mass))
            except Exception as exc:
                row.update(status='error', error=f'{type(exc).__name__}: {exc}')
            if row['status'] != 'pass':
                failures += 1
            writer.writerow(row)
            f.flush()
            print(f"{i}/{len(paths)} {row['status']} {path.parent.name}/{path.name} "
                  f"head={row.get('head_mask_diff', '')} gas={row.get('gas_mask_diff', '')} {row.get('error', '')}",
                  flush=True)
    print(f'wrote {len(paths)} rows, {failures} failures: {args.output}', flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
