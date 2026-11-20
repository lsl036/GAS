#!/usr/bin/env python3
"""Paired pruning-only benchmark for dynamic compact Top-p and GAS."""
import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from bench_e4_kernel_evidence import cuda_event_samples
from gas_compact_refinement import allocate, exception_stage
from sparse_head_top_p import sparse_head_top_p
from test_gas_compact_refinement import materialize, pack
from gas_attention.pyimpl.gqa_refine import build_group_refined_mask
from gas_attention.pyimpl.quantize import min_max_per_token_quant_kv
from gas_attention.pyimpl.top_p import top_p_unnormalized

ROOT = Path(__file__).resolve().parents[2]


def interleaved_graph_timing(calls, warmups, repeats, rounds, graph_iterations, seed):
    graphs, outputs, values, round_medians, orders = {}, {}, {}, {}, []
    for name, fn in calls.items():
        for _ in range(warmups):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(graph_iterations):
                output = fn()
        graphs[name], outputs[name] = graph, output
        graph.replay()
        values[name], round_medians[name] = [], []
    torch.cuda.synchronize()
    rng = random.Random(seed)
    for _ in range(rounds):
        order = list(calls)
        rng.shuffle(order)
        orders.append(order)
        for name in order:
            samples = [sample / graph_iterations for sample in
                       cuda_event_samples(graphs[name].replay, torch.device('cuda'), warmups, repeats)]
            values[name].extend(samples)
            round_medians[name].append(statistics.median(samples))
    return ({name: {'median_us': statistics.median(samples), 'samples_us': samples,
                    'round_medians_us': round_medians[name]} for name, samples in values.items()}, orders)


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
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--double-sum', action='store_true')
    p.add_argument('--adaptive-sum', action='store_true')
    p.add_argument('--guard', type=float, default=1.0e-6)
    p.add_argument('--tagged-head', action='store_true')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    context = 'qasper16k' if args.context == '16k' else 'qasper32k_repeated'
    path = ROOT / 'benchmark/results_efficiency/fig6_traces' / f'{context}_{args.scenario}_s{args.sample}' / f'step001_layer{args.layer:02d}.pt'
    trace = torch.load(path, map_location='cpu', weights_only=True)
    q, k = trace['q'].cuda(), trace['k'].cuda().repeat_interleave(4, dim=1)
    scores0 = (torch.matmul(q.unsqueeze(2), min_max_per_token_quant_kv(k, 4).transpose(-1, -2))
               / math.sqrt(q.shape[-1])).squeeze(2).contiguous()
    candidate0 = trace['candidate'].cuda().bool().contiguous()
    candidate_ids = pack(candidate0).repeat(args.batch, 1, 1).contiguous()
    candidate_counts = candidate0.sum(-1, dtype=torch.int32).repeat(args.batch, 1).contiguous()
    candidate = candidate0.repeat(args.batch, 1, 1).contiguous()
    scores = scores0.repeat(args.batch, 1, 1).contiguous()
    masked_scores = scores.masked_fill(~candidate, torch.finfo(scores.dtype).min)
    length, capacity = scores.shape[-1], candidate_ids.shape[-1]
    head_mask = (torch.zeros_like(candidate, dtype=torch.int32) if args.tagged_head else torch.empty_like(candidate))
    head_epochs = torch.zeros_like(candidate_counts) if args.tagged_head else None
    head_ids = torch.empty_like(candidate_ids)
    head_counts = torch.empty_like(candidate_counts)
    ambiguous = torch.empty_like(candidate_counts) if args.adaptive_sum else None
    out = allocate(args.batch, capacity, length, scores.device)
    out = (out[0], out[1], torch.zeros((args.batch, 8, length), dtype=torch.int32, device=scores.device), out[3], out[4])
    epochs = torch.zeros((args.batch, 8), dtype=torch.int32, device=scores.device)
    scratch = torch.empty((args.batch, 8, 4 * capacity), dtype=torch.float32, device=scores.device)
    source = ROOT / 'csrc/src/gas_compact_refine.cu'
    cub = load(name='gas_compact_refine', sources=[str(source)], extra_cuda_cflags=['-O3'], verbose=False)

    def dense_top_p():
        return (top_p_unnormalized(masked_scores.unsqueeze(2), .85).squeeze(2) & candidate).contiguous()

    def compact_top_p():
        return sparse_head_top_p(candidate_ids, candidate_counts, scores, head_mask, head_ids, head_counts,
                                 double_sum=args.double_sum, adaptive_sum=args.adaptive_sum,
                                 ambiguous_out=ambiguous, guard=args.guard, head_epochs=head_epochs)

    def compact_gas():
        compact_top_p()
        if args.tagged_head:
            cub.group_head_tagged(head_ids, head_counts, head_mask, head_epochs, scores, out[0], out[1],
                                  out[2], epochs, scratch, .8)
        else:
            cub.group_tagged_fused_reduce(head_ids, head_counts, head_mask, scores, out[0], out[1], out[2], epochs, scratch, .8)
        exception_stage(head_ids, scores, out, counts=head_counts, epochs=epochs)
        return out[0], out[1], out[3], out[4]

    dense = dense_top_p()
    compact_top_p()
    current_mask = head_mask == head_epochs.unsqueeze(-1) if args.tagged_head else head_mask
    head_diff = int((dense != current_mask).sum())
    trace_head_diff = int((current_mask[:1] != trace['twilight'].cuda().bool()).sum())
    if head_diff or trace_head_diff:
        raise ValueError(f'head mask mismatch: dense={head_diff}, captured={trace_head_diff}')
    reference = build_group_refined_mask(scores0.unsqueeze(2), dense[:1].unsqueeze(2), 8, .8, 'group_max', 16)
    actual = materialize(tuple(x[:1] for x in compact_gas()), length)
    gas_diff = int((actual != reference.squeeze(2).cpu()).sum())
    trace_gas_diff = int((actual != trace['gas'].bool()).sum())
    if gas_diff or trace_gas_diff:
        raise ValueError(f'GAS mask mismatch: reference={gas_diff}, captured={trace_gas_diff}')
    calls = {'dense_top_p': dense_top_p, 'compact_top_p': compact_top_p, 'compact_top_p_plus_gas': compact_gas}
    if not args.tagged_head:
        calls['mask_clear'] = lambda: head_mask.zero_()
    measurements, orders = interleaved_graph_timing(calls, args.warmups, args.repeats, args.rounds,
                                                    args.graph_iterations, args.seed)
    for name, measurement in measurements.items():
        print(f"{name}: {measurement['median_us']:.3f} us", flush=True)
    compact_gas()
    final_diff = int((materialize(tuple(x[:1] for x in (out[0], out[1], out[3], out[4])), length)
                      != reference.squeeze(2).cpu()).sum())
    if final_diff:
        raise ValueError(f'GAS mismatch after graph timings: {final_diff}')
    result = {
        'scenario': args.scenario, 'context': args.context, 'sample': args.sample, 'layer': args.layer,
        'batch': args.batch, 'seq_len': length, 'candidate_capacity': capacity, 'trace': str(path),
        'trace_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'cuda_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'compact_top_p_source_sha256': hashlib.sha256((ROOT / 'benchmark/efficiency/sparse_head_top_p.py').read_bytes()).hexdigest(),
        'refinement_source_sha256': hashlib.sha256((ROOT / 'benchmark/efficiency/gas_compact_refinement.py').read_bytes()).hexdigest(),
        'gpu': torch.cuda.get_device_name(), 'torch_version': torch.__version__, 'score_dtype': str(scores.dtype),
        'head_diff_all_batch': head_diff, 'trace_head_diff_first': trace_head_diff,
        'gas_reference_diff_first': gas_diff, 'gas_trace_diff_first': trace_gas_diff,
        'gas_reference_diff_after_timing_first': final_diff,
        'timing': 'interleaved CUDA graph device event medians; same trace replicated across batch',
        'round_orders': orders, 'rounds': args.rounds, 'repeats': args.repeats,
        'graph_iterations': args.graph_iterations, 'timing_seed': args.seed,
        'double_sum': args.double_sum, 'adaptive_sum': args.adaptive_sum, 'tagged_head': args.tagged_head,
        'guard': args.guard if args.adaptive_sum else None,
        'ambiguous_heads': int(ambiguous.sum()) if args.adaptive_sum else None,
        'benchmark_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'includes': ['head top-p', 'head mask scatter (and zeroing for untagged path)', 'head compact-index/count generation',
                     'GAS group and exception for combined case'],
        'excludes': ['selector', 'estimator', 'candidate mask/indices/counts generation', 'attention',
                     'scratch allocation', 'CPU and non-graph dispatch overhead'],
        'measurements': measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
