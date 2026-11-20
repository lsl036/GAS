#!/usr/bin/env python3
"""Single-layer estimator/pruning pilot; no selector or sparse attention timing."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from bench_e4_kernel_evidence import cuda_event_samples
from bench_compact_head_pruning import interleaved_graph_timing
from gas_compact_refinement import allocate, exception_stage
from sparse_head_top_p import sparse_head_top_p
from test_gas_compact_refinement import materialize, pack
from gas_attention.pyimpl.gqa_refine import build_group_refined_mask
from gas_attention.pyimpl.quantize import min_max_per_token_quant_kv
from gas_attention.pyimpl.top_p import top_p_unnormalized

ROOT = Path(__file__).resolve().parents[2]


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--scenario', choices=['fixed8192', 'rate08'], default='fixed8192')
    p.add_argument('--sample', type=int, default=54)
    p.add_argument('--layer', type=int, default=2)
    p.add_argument('--warmups', type=int, default=2)
    p.add_argument('--repeats', type=int, default=10)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--batch-tile', type=int, default=0,
                   help='Process the expanded-key estimator serially in batch tiles; 0 uses the full batch.')
    p.add_argument('--compact-cache', action='store_true',
                   help='Cache fake-quantized K at KV-head granularity and use grouped QK.')
    p.add_argument('--pruning-only', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    path = ROOT / 'benchmark/results_efficiency/fig6_traces' / f'qasper16k_{a.scenario}_s{a.sample}' / f'step001_layer{a.layer:02d}.pt'
    trace = torch.load(path, map_location='cpu', weights_only=True)
    q = trace['q'].cuda().repeat(a.batch, 1, 1).unsqueeze(2).contiguous()
    compact_k = trace['k'].cuda().repeat(a.batch, 1, 1, 1).contiguous()
    k = compact_k.repeat_interleave(4, dim=1).contiguous()
    candidate0 = trace['candidate'].cuda().bool().contiguous()
    candidates = candidate0.repeat(a.batch, 1, 1).contiguous()
    candidate_ids = pack(candidate0).repeat(a.batch, 1, 1).contiguous()
    candidate_counts = candidate0.sum(-1, dtype=torch.int32).repeat(a.batch, 1).contiguous()
    length, capacity = k.shape[-2], candidate_ids.shape[-1]
    # Full expanded, dequantized BF16 cache, not a packed INT4 cache.
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    cache = min_max_per_token_quant_kv(compact_k if a.compact_cache else k, 4).contiguous()
    end.record()
    end.synchronize()
    init_us = start.elapsed_time(end) * 1000

    head_mask = torch.empty_like(candidates)
    head_ids, head_counts = torch.empty_like(candidate_ids), torch.empty_like(candidate_counts)
    ambiguous = torch.empty_like(candidate_counts)
    out = allocate(a.batch, capacity, length, q.device)
    out = (out[0], out[1], torch.zeros((a.batch, 8, length), dtype=torch.int32, device=q.device), out[3], out[4])
    epochs = torch.zeros((a.batch, 8), dtype=torch.int32, device=q.device)
    scratch = torch.empty((a.batch, 8, 4 * capacity), dtype=torch.float32, device=q.device)
    source = ROOT / 'csrc/src/gas_compact_refine.cu'
    cub = load(name='gas_compact_refine', sources=[str(source)], extra_cuda_cflags=['-O3'], verbose=False)
    slow_out = torch.empty((a.batch, 32, length), dtype=q.dtype, device=q.device)
    tile = a.batch_tile or a.batch
    if tile <= 0:
        raise ValueError('batch tile must be positive')

    def slow():
        for start in range(0, a.batch, tile):
            end = min(start + tile, a.batch)
            quant = min_max_per_token_quant_kv(k[start:end], 4)
            values = (torch.matmul(q[start:end], quant.transpose(-1, -2)) / math.sqrt(q.shape[-1])).squeeze(2)
            slow_out[start:end].copy_(values)
        return slow_out

    def fast():
        source_k = compact_k if a.compact_cache else k
        cache[:, :, -1:, :].copy_(min_max_per_token_quant_kv(source_k[:, :, -1:, :], 4))
        if a.compact_cache:
            return (torch.matmul(q.reshape(a.batch, 8, 4, -1), cache.transpose(-1, -2)) /
                    math.sqrt(q.shape[-1])).reshape(a.batch, 32, length).contiguous()
        return (torch.matmul(q, cache.transpose(-1, -2)) / math.sqrt(q.shape[-1])).squeeze(2).contiguous()

    def top_p(scores):
        return sparse_head_top_p(candidate_ids, candidate_counts, scores, head_mask, head_ids, head_counts,
                                 adaptive_sum=True, ambiguous_out=ambiguous)

    def refine(scores):
        cub.group_tagged_fused_reduce(head_ids, head_counts, head_mask, scores, out[0], out[1], out[2], epochs, scratch, .8)
        exception_stage(head_ids, scores, out, counts=head_counts, epochs=epochs)
        return out[0], out[1], out[3], out[4]

    def pipeline(estimator, gas):
        scores = estimator()
        top_p(scores)
        return refine(scores) if gas else head_mask

    slow_scores, fast_scores = slow(), fast()
    score_diff = int((slow_scores != fast_scores).sum().item())
    max_diff = float((slow_scores.float() - fast_scores.float()).abs().max().item())
    dense = top_p_unnormalized(slow_scores.masked_fill(~candidates, torch.finfo(slow_scores.dtype).min).unsqueeze(2), .85).squeeze(2) & candidates
    top_p(slow_scores)
    head_diff = int((dense != head_mask).sum().item())
    trace_head_diff = int((head_mask[:1] != trace['twilight'].cuda().bool()).sum().item())
    reference = build_group_refined_mask(slow_scores[:1].unsqueeze(2), dense[:1].unsqueeze(2), 8, .8, 'group_max', 16)
    actual = materialize(tuple(x[:1] for x in refine(slow_scores)), length)
    group_diff = int((actual != reference.squeeze(2).cpu()).sum().item())
    trace_group_diff = int((actual != trace['gas'].bool()).sum().item())
    top_p(fast_scores)
    fast_head_diff = int((dense != head_mask).sum().item())
    fast_actual = materialize(tuple(x[:1] for x in refine(fast_scores)), length)
    fast_group_diff = int((fast_actual != reference.squeeze(2).cpu()).sum().item())
    if any((head_diff, trace_head_diff, group_diff, trace_group_diff, fast_head_diff, fast_group_diff)):
        raise ValueError(f'correctness failed: scores={score_diff}, head={head_diff}, trace_head={trace_head_diff}, group={group_diff}, trace_group={trace_group_diff}, fast_head={fast_head_diff}, fast_group={fast_group_diff}, max_score_error={max_diff}')
    del slow_scores, fast_scores, dense, reference, actual, fast_actual
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    fixed_scores = fast()
    top_p(fixed_scores)

    # Capturing each graph separately bounds graph-pool memory for the expanded K.
    calls = {
        'head_top_p': lambda: top_p(fixed_scores),
        'gas_refinement': lambda: refine(fixed_scores),
        'estimator_requantize': slow,
        'estimator_cached': fast,
        'twilight_requantize': lambda: pipeline(slow, False),
        'gas_requantize': lambda: pipeline(slow, True),
        'twilight_cached': lambda: pipeline(fast, False),
        'gas_cached': lambda: pipeline(fast, True),
    }
    if a.pruning_only:
        calls = {name: calls[name] for name in ('head_top_p', 'gas_refinement')}
    results = {}
    for name, fn in calls.items():
        for _ in range(a.warmups):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            value = fn()
        samples, medians = [], []
        for _ in range(a.rounds):
            vals = cuda_event_samples(graph.replay, torch.device('cuda'), a.warmups, a.repeats)
            samples.extend(vals)
            medians.append(statistics.median(vals))
        results[name] = {'median_us': statistics.median(samples), 'round_medians_us': medians, 'samples_us': samples}
        print(f'{name}: {results[name]["median_us"]:.3f} us', flush=True)
        del value, graph
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    paired, orders = ({}, []) if a.pruning_only else interleaved_graph_timing(
        {'twilight_cached': lambda: pipeline(fast, False),
         'gas_cached': lambda: pipeline(fast, True)},
        a.warmups, a.repeats, a.rounds, 1, 42)
    for name, measurement in paired.items():
        print(f'paired_{name}: {measurement["median_us"]:.3f} us', flush=True)

    result = {
        'trace': str(path), 'trace_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'scenario': a.scenario, 'sample': a.sample, 'layer': a.layer, 'batch': a.batch,
        'seq_len': length, 'candidate_capacity': capacity, 'dtype': str(q.dtype),
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
        'timing': 'CUDA graph replay, GPU events, sequential variants; each batch duplicates one trace',
        'correctness': {'score_differences': score_diff, 'max_score_difference': max_diff,
                        'head_mask_differences': head_diff, 'trace_head_differences': trace_head_diff,
                        'gas_mask_differences': group_diff, 'trace_gas_differences': trace_group_diff,
                        'fast_head_differences': fast_head_diff, 'fast_gas_differences': fast_group_diff},
        'cache_initialization_us_one_layer': init_us, 'cache_extra_bytes_one_layer': cache.numel() * cache.element_size(),
        'batch_tile': tile, 'compact_kv_head_cache': a.compact_cache,
        'includes': ['quantization or incremental cache update', 'estimated QK matmul', 'compact head top-p', 'GAS group and exceptions where applicable'],
        'excludes': ['selector', 'candidate construction', 'prefill cache initialization', 'QKV projections', 'sparse attention', 'CPU dispatch', 'other layers'],
        'warning': 'cached estimator uses extra dequantized floating K, not packed int4; experimental prototype, not integrated in model',
        'measurements': results,
        'paired_cached_measurements': paired, 'paired_round_orders': orders,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'wrote {a.output}', flush=True)


if __name__ == '__main__':
    main()
