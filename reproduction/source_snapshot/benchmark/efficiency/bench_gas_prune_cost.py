#!/usr/bin/env python3
"""Measure Twilight top-p and experimental GAS pruning on a fixed Qasper trace.

Candidate indices and counts are constructed outside the timed region. This is
an isolated pruning microbenchmark, not a connected inference implementation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from bench_gas_compact_pipeline import graph_timing
from gas_compact_refinement import allocate, buckets, exception_stage, exception_stage_tiled_sort
from test_gas_compact_refinement import materialize, pack
from twilight.pyimpl.gqa_refine import build_group_refined_mask
from twilight.pyimpl.quantize import min_max_per_token_quant_kv
from twilight.pyimpl.top_p import top_p_unnormalized


ROOT = Path(__file__).resolve().parents[2]


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario', choices=('rate08', 'fixed8192'), required=True)
    p.add_argument('--context', choices=('16k', '32k'), default='16k')
    p.add_argument('--sample', type=int, choices=(15, 54), required=True)
    p.add_argument('--layer', type=int, required=True)
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--warmups', type=int, default=3)
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--graph-iterations', type=int, default=4)
    p.add_argument('--pair-reduce', action='store_true')
    p.add_argument('--active-buckets', action='store_true')
    p.add_argument('--fused-exception', action='store_true')
    p.add_argument('--fast-group-scores', action='store_true')
    p.add_argument('--fused-mass', action='store_true')
    p.add_argument('--sort-reduce', action='store_true')
    p.add_argument('--tile-sort', type=int, choices=(128, 256, 512, 1024))
    args = p.parse_args()
    if args.fused_exception and (args.pair_reduce or args.active_buckets or args.fast_group_scores or args.fused_mass or args.sort_reduce or args.tile_sort):
        p.error('--fused-exception cannot be combined with separate exception options')
    if args.fused_mass and args.fast_group_scores:
        p.error('--fused-mass cannot be combined with --fast-group-scores')
    if sum(bool(v) for v in (args.pair_reduce, args.sort_reduce, args.tile_sort)) > 1:
        p.error('choose only one exception implementation')
    if args.tile_sort and args.active_buckets:
        p.error('--tile-sort does not use bucket dispatch')
    if args.output.exists():
        raise FileExistsError(args.output)

    context = 'qasper16k' if args.context == '16k' else 'qasper32k_repeated'
    trace_path = (ROOT / 'benchmark/results_efficiency/fig6_traces' /
                  f'{context}_{args.scenario}_s{args.sample}' / f'step001_layer{args.layer:02d}.pt')
    trace = torch.load(trace_path, map_location='cpu', weights_only=True)
    q, k = trace['q'].cuda(), trace['k'].cuda().repeat_interleave(4, dim=1)
    scores0 = (torch.matmul(q.unsqueeze(2), min_max_per_token_quant_kv(k, 4).transpose(-1, -2))
               / math.sqrt(q.shape[-1])).squeeze(2).contiguous()
    candidate0 = trace['candidate'].cuda().bool().contiguous()
    masked0 = scores0.masked_fill(~candidate0, torch.finfo(scores0.dtype).min)
    head0 = (top_p_unnormalized(masked0.unsqueeze(2), .85).squeeze(2) & candidate0).contiguous()
    head_trace_diff = int((head0 != trace['twilight'].cuda().bool()).sum())
    if head_trace_diff:
        raise ValueError(f'recomputed Twilight mask differs from captured trace: {head_trace_diff} entries')

    length = head0.shape[-1]
    base_ids = pack(head0)
    indices = base_ids.repeat(args.batch, 1, 1).contiguous()
    counts = head0.sum(-1, dtype=torch.int32).repeat(args.batch, 1).contiguous()
    counts_cpu = head0.sum(-1).cpu()
    active_buckets = [(low, high) for low, high in buckets(indices.shape[-1])
                      if bool(((counts_cpu > low) & (counts_cpu <= high)).any())] if args.active_buckets else None
    mask = head0.repeat(args.batch, 1, 1).contiguous()
    candidate = candidate0.repeat(args.batch, 1, 1).contiguous()
    scores = scores0.repeat(args.batch, 1, 1).contiguous()
    masked_scores = masked0.repeat(args.batch, 1, 1).contiguous()
    out = allocate(args.batch, indices.shape[-1], length, scores.device)
    out = (out[0], out[1], torch.zeros((args.batch, 8, length), dtype=torch.int32, device=scores.device),
           out[3], out[4])
    epochs = torch.zeros((args.batch, 8), dtype=torch.int32, device=scores.device)
    scratch = torch.empty((args.batch, 8, 4 * indices.shape[-1]), dtype=torch.float32, device=scores.device)
    tile_scratch = (torch.empty((args.batch, 32, math.ceil(indices.shape[-1] / args.tile_sort), 16),
                                dtype=torch.uint64, device=scores.device) if args.tile_sort else None)
    source = ROOT / 'csrc/src/gas_compact_refine.cu'
    cub = load(name='gas_compact_refine', sources=[str(source)], extra_cuda_cflags=['-O3'], verbose=False)

    def head_top_p():
        return (top_p_unnormalized(masked_scores.unsqueeze(2), .85).squeeze(2) & candidate).contiguous()

    def group(input_mask=mask):
        if args.fused_exception:
            return cub.group_tagged_fused_exception(indices, counts, input_mask, scores, out[0], out[1],
                                                      out[2], epochs, scratch, out[3], out[4], .8)
        fn = (cub.group_tagged_fused_mass if args.fused_mass else
              cub.group_tagged_fast_scores if args.fast_group_scores else cub.group_tagged_fused_reduce)
        return fn(indices, counts, input_mask, scores, out[0], out[1], out[2], epochs, scratch, .8)

    def exception():
        if args.tile_sort:
            return exception_stage_tiled_sort(indices, scores, out, counts, tile_scratch, epochs=epochs,
                                              tile=args.tile_sort)
        return exception_stage(indices, scores, out, counts=counts, epochs=epochs,
                               pair_reduce=args.pair_reduce, active_buckets=active_buckets,
                               sort_reduce=args.sort_reduce)

    def refinement():
        group()
        if not args.fused_exception:
            exception()
        return out[0], out[1], out[3], out[4]

    def combined_fixed_indices():
        group(head_top_p())
        if not args.fused_exception:
            exception()
        return out[0], out[1], out[3], out[4]

    output = refinement()
    actual = materialize(tuple(x[:1] for x in output), length)
    expected = build_group_refined_mask(scores0.unsqueeze(2), head0.unsqueeze(2), 8, .8, 'group_max', 16)
    difference = int((actual != expected.squeeze(2).cpu()).sum())
    if difference:
        raise ValueError(f'GAS compact output differs from reference: {difference} entries')
    trace_diff = int((actual != trace['gas'].bool()).sum())
    calls = {'twilight_top_p': head_top_p}
    if not args.fused_exception:
        calls.update({'gas_group': group, 'gas_exception': exception})
    calls.update({'gas_refinement': refinement, 'top_p_plus_gas_fixed_indices': combined_fixed_indices})
    measurements = {}
    for name, fn in calls.items():
        median, samples, _ = graph_timing(fn, args.warmups, args.repeats, args.rounds, args.graph_iterations)
        measurements[name] = {'median_us': median, 'samples_us': samples}
        print(f'{name}: {median:.3f} us', flush=True)
    output = combined_fixed_indices()
    combined_diff = int((materialize(tuple(x[:1] for x in output), length) != expected.squeeze(2).cpu()).sum())
    if combined_diff:
        raise ValueError(f'combined output differs from reference: {combined_diff} entries')
    result = {
        'scenario': args.scenario, 'context': args.context, 'sample': args.sample,
        'layer': args.layer, 'batch': args.batch, 'seq_len': length,
        'trace': str(trace_path), 'trace_sha256': hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        'cuda_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'gpu': torch.cuda.get_device_name(), 'torch_version': torch.__version__,
        'score_dtype': str(scores.dtype), 'head_threshold': .85, 'group_threshold': .8,
        'exception_budget': 16, 'candidate_capacity': indices.shape[-1], 'pair_reduce': args.pair_reduce,
        'fused_exception': args.fused_exception, 'fast_group_scores': args.fast_group_scores, 'fused_mass': args.fused_mass,
        'sort_reduce': args.sort_reduce, 'tile_sort': args.tile_sort,
        'active_buckets': active_buckets,
        'triton_source_sha256': hashlib.sha256((ROOT / 'benchmark/efficiency/gas_compact_refinement.py').read_bytes()).hexdigest(),
        'head_trace_diff': head_trace_diff, 'gas_reference_diff': difference,
        'gas_trace_diff': trace_diff, 'combined_reference_diff': combined_diff,
        'timing': 'CUDA graph device event medians; fixed trace repeated across batch',
        'excludes': ['selector', 'estimator', 'candidate masking', 'compact-index/count construction',
                     'attention', 'model computation', 'CPU and non-graph dispatch overhead'],
        'connected_caveat': 'combined consumes fresh head mask but uses prebuilt indices, counts and optional active buckets for fixed input',
        'measurements': measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
