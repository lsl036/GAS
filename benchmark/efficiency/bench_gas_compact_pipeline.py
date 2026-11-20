#!/usr/bin/env python3
"""Paired Qasper per-layer FI vs compact-refinement + native GAS pilot."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import flashinfer
import torch
from torch.utils.cpp_extension import load

from bench_e4_kernel_evidence import cuda_event_samples
from bench_compact_head_pruning import interleaved_graph_timing
from bench_e6_qasper_trace import reference
from gas_compact_refinement import allocate, exception_stage, exception_stage_unified, group_stage, refine
from test_gas_compact_refinement import materialize, pack
from gas_attention.pyimpl.gqa_refine import build_group_refined_mask
from gas_attention.pyimpl.quantize import min_max_per_token_quant_kv

ROOT = Path(__file__).resolve().parents[2]


def grouped_fi(q, k, v, mask):
    batch, _, dim = q.shape
    length = k.shape[0]
    m = mask.view(8, 4, length).cpu()
    counts, ids, custom = [], [], []
    for group in range(8):
        tokens = m[group].any(0).nonzero().flatten()
        counts.append(tokens.numel())
        ids.append(tokens * 8 + group)
        custom.append(m[group, :, tokens].reshape(-1))
    if min(counts) <= 0:
        raise ValueError('FI baseline requires nonempty groups')
    device = q.device
    counts = torch.tensor(counts * batch, dtype=torch.int32, device=device)
    indices = torch.cat(ids).int().repeat(batch).to(device)
    custom = torch.cat(custom).repeat(batch).to(device)
    ptr = torch.nn.functional.pad(counts.cumsum(0, dtype=torch.int32), (1, 0))
    qptr = torch.arange(batch * 8 + 1, device=device, dtype=torch.int32) * 4
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, 'NHD')
    wrapper.plan(qptr, ptr, indices, torch.ones_like(counts), 1, 1, dim, 1,
                 custom_mask=custom, causal=False, pos_encoding_mode='NONE',
                 q_data_type=q.dtype, kv_data_type=k.dtype, sm_scale=1 / math.sqrt(dim))
    pages = (k.view(-1, 1, 1, dim), v.view(-1, 1, 1, dim))
    return lambda: wrapper.run(q.view(-1, 1, dim), pages).view_as(q), (wrapper, pages, workspace)


def graph_timing(fn, warmups, repeats, rounds, graph_iterations):
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_iterations):
            result = fn()
    graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        samples.extend(x / graph_iterations for x in
                       cuda_event_samples(graph.replay, torch.device('cuda'), warmups, repeats))
    return statistics.median(samples), samples, result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=('rate08', 'fixed8192'), required=True)
    parser.add_argument('--context', choices=('16k', '32k'), default='16k')
    parser.add_argument('--sample', type=int, choices=(15, 54), required=True)
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trace-file', type=Path, help='Use an explicit captured or synthetic trace.')
    parser.add_argument('--use-trace-scores', action='store_true', help='Use separately supplied synthetic estimator logits.')
    parser.add_argument('--warmups', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--graph-iterations', type=int, default=4)
    parser.add_argument('--attention-dtype', choices=('float16', 'bfloat16'), default='float16')
    parser.add_argument('--include-quest', action='store_true')
    parser.add_argument('--paired', action='store_true', help='Also interleave connected GAS and attention replays.')
    parser.add_argument('--bucketed', action='store_true')
    parser.add_argument('--ragged', action='store_true')
    parser.add_argument('--cuda-cub', action='store_true')
    parser.add_argument('--tiled-exception', action='store_true')
    parser.add_argument('--triton-exception', action='store_true')
    parser.add_argument('--tagged-flags', action='store_true')
    parser.add_argument('--blocksort-exception', action='store_true')
    parser.add_argument('--shared-scratch', action='store_true')
    parser.add_argument('--hist-exception', action='store_true')
    parser.add_argument('--dense-group-scan', action='store_true')
    parser.add_argument('--fused-group-reductions', action='store_true')
    parser.add_argument('--warp-group-compaction', action='store_true')
    parser.add_argument('--normalized-group', action='store_true')
    parser.add_argument('--normalized-shared-group', action='store_true')
    parser.add_argument('--pair-exception', action='store_true')
    parser.add_argument('--normalized-fast-scores', action='store_true')
    parser.add_argument('--normalized-prepared', action='store_true')
    parser.add_argument('--unified-exception', action='store_true')
    parser.add_argument('--decode-mode', choices=('tiled8', 'split8p8', 'tiled8p8', 'split8p4',
                                                  'perhead_counted8', 'coop8p8', 'coop8p4'), default='tiled8')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    trace_context = 'qasper16k' if args.context == '16k' else 'qasper32k_repeated'
    trace_path = (ROOT / 'benchmark/results_efficiency/fig6_traces' /
                  f'{trace_context}_{args.scenario}_s{args.sample}' / f'step001_layer{args.layer:02d}.pt')
    if args.trace_file is not None:
        trace_path = args.trace_file.resolve()
    trace = torch.load(trace_path, map_location='cpu', weights_only=True)
    q_bf16, k_bf16 = trace['q'].cuda(), trace['k'].cuda()
    attention_dtype = getattr(torch, args.attention_dtype)
    q = q_bf16.to(attention_dtype).repeat(args.batch, 1, 1).contiguous()
    k, v = (trace[x][0].permute(1, 0, 2).to('cuda', dtype=attention_dtype).contiguous() for x in ('k', 'v'))
    length = k.shape[0]
    base_mask = trace['twilight'].cuda().bool().contiguous()
    expanded_k = k_bf16.repeat_interleave(4, dim=1)
    if args.use_trace_scores:
        scores0 = trace['scores'].cuda().contiguous()
        if scores0.shape != (1, 32, k.shape[0]) or not torch.isfinite(scores0).all():
            raise ValueError('synthetic scores must be finite and have shape [1,32,length]')
    else:
        scores0 = (torch.matmul(q_bf16.unsqueeze(2), min_max_per_token_quant_kv(expanded_k, 4).transpose(-1, -2))
                   / math.sqrt(q_bf16.shape[-1])).squeeze(2).contiguous()
    indices = pack(base_mask).repeat(args.batch, 1, 1).contiguous()
    mask = base_mask.repeat(args.batch, 1, 1).contiguous()
    if args.tiled_exception and not args.cuda_cub:
        parser.error('--tiled-exception requires --cuda-cub')
    if args.triton_exception and not args.cuda_cub:
        parser.error('--triton-exception requires --cuda-cub')
    if args.triton_exception and args.tiled_exception:
        parser.error('--triton-exception excludes --tiled-exception')
    if args.tagged_flags and not (args.cuda_cub and (args.tiled_exception or args.triton_exception)):
        parser.error('--tagged-flags requires --cuda-cub and a tagged-compatible exception path')
    if args.blocksort_exception and not args.tagged_flags:
        parser.error('--blocksort-exception requires --tagged-flags')
    if args.hist_exception and (not args.tagged_flags or args.blocksort_exception):
        parser.error('--hist-exception requires --tagged-flags and excludes --blocksort-exception')
    if args.shared_scratch and not args.tagged_flags:
        parser.error('--shared-scratch requires --tagged-flags')
    if args.dense_group_scan and (not args.tagged_flags or args.shared_scratch):
        parser.error('--dense-group-scan requires --tagged-flags and excludes --shared-scratch')
    if args.fused_group_reductions and (not args.tagged_flags or args.dense_group_scan or args.shared_scratch):
        parser.error('--fused-group-reductions requires --tagged-flags and excludes other group variants')
    if args.warp_group_compaction and not args.fused_group_reductions:
        parser.error('--warp-group-compaction requires --fused-group-reductions')
    if args.normalized_group and (not args.fused_group_reductions or args.warp_group_compaction):
        parser.error('--normalized-group requires --fused-group-reductions and excludes warp compaction')
    if args.normalized_shared_group and not args.normalized_group:
        parser.error('--normalized-shared-group requires --normalized-group')
    if args.normalized_fast_scores and (not args.normalized_group or args.normalized_shared_group):
        parser.error('--normalized-fast-scores requires --normalized-group and excludes shared scratch')
    if args.normalized_prepared and (not args.normalized_group or args.normalized_shared_group or args.normalized_fast_scores):
        parser.error('--normalized-prepared requires --normalized-group and excludes shared/fast-score options')
    if args.unified_exception and (not args.triton_exception or args.pair_exception):
        parser.error('--unified-exception requires --triton-exception and excludes pair reduction')
    counts = mask.sum(-1, dtype=torch.int32).contiguous() if args.bucketed or args.ragged or args.cuda_cub else None
    scores = scores0.repeat(args.batch, 1, 1).contiguous()
    out = allocate(args.batch, indices.shape[-1], length, q.device)
    epochs = None
    if args.tagged_flags:
        out = (out[0], out[1], torch.zeros((args.batch, 8, length), dtype=torch.int32, device=q.device), out[3], out[4])
        epochs = torch.zeros((args.batch, 8), dtype=torch.int32, device=q.device)
    cub = None
    scratch = None
    tmp_scores = tmp_ids = None
    if args.cuda_cub:
        cub = load(name='gas_compact_refine', sources=[str(ROOT / 'csrc/src/gas_compact_refine.cu')],
                   extra_cuda_cflags=['-O3'], verbose=False)
        scratch_size = length if args.dense_group_scan else 4 * indices.shape[-1]
        scratch = torch.empty((args.batch, 8, scratch_size), dtype=torch.float32, device=q.device)
        if args.tiled_exception:
            shape = (args.batch, 32, math.ceil(indices.shape[-1] / 1024), 16)
            tmp_scores = torch.empty(shape, dtype=torch.float32, device=q.device)
            tmp_ids = torch.empty(shape, dtype=torch.int32, device=q.device)
    def group_call():
        if cub is not None:
            if args.tagged_flags:
                if args.fused_group_reductions:
                    if args.normalized_group:
                        fn = (cub.group_tagged_normalized_shared if args.normalized_shared_group else
                              cub.group_tagged_normalized_prepared if args.normalized_prepared else
                              cub.group_tagged_normalized_fast_scores if args.normalized_fast_scores else cub.group_tagged_normalized)
                        return fn(indices, counts, mask, scores, out[0], out[1], out[2], epochs, scratch, 0.8)
                    if args.warp_group_compaction:
                        return cub.group_tagged_warp_reduce(indices, counts, mask, scores, out[0], out[1], out[2], epochs, scratch, 0.8)
                    return cub.group_tagged_fused_reduce(indices, counts, mask, scores, out[0], out[1], out[2], epochs, scratch, 0.8)
                if args.dense_group_scan:
                    return cub.group_tagged_dense(indices, counts, mask, scores, out[0], out[1], out[2], epochs, scratch, 0.8)
                if args.shared_scratch:
                    return cub.group_tagged_shared(indices, counts, mask, scores, out[0], out[1], out[2], epochs, scratch, 0.8)
                return cub.group_tagged(indices, counts, mask, scores, out[0], out[1], out[2], epochs, scratch, 0.8)
            return cub.group(indices, counts, mask, scores, out[0], out[1], out[2], scratch, 0.8)
        return group_stage(indices, mask, scores, out, counts=counts, ragged=args.ragged)
    def exception_call():
        if cub is not None:
            if args.triton_exception:
                if args.unified_exception:
                    return exception_stage_unified(indices, scores, out, counts, epochs)
                return exception_stage(indices, scores, out, counts=counts, epochs=epochs, pair_reduce=args.pair_exception)
            if args.tagged_flags:
                if args.hist_exception:
                    return cub.exception_histogram_tagged(indices, counts, scores, out[2], epochs, out[3], out[4])
                if args.blocksort_exception:
                    return cub.exception_blocksort_tagged(indices, counts, scores, out[2], epochs, out[3], out[4])
                return cub.exception_tiled_tagged(indices, counts, scores, out[2], epochs, out[3], out[4], tmp_scores, tmp_ids)
            if args.tiled_exception:
                return cub.exception_tiled(indices, counts, scores, out[2], out[3], out[4], tmp_scores, tmp_ids)
            return cub.exception(indices, counts, scores, out[2], out[3], out[4])
        return exception_stage(indices, scores, out, counts=counts)
    def selection():
        if cub is not None:
            if not args.tagged_flags:
                out[2].zero_()
            group_call()
            exception_call()
            return out[0], out[1], out[3], out[4]
        return refine(indices, mask, scores, out, counts=counts, ragged=args.ragged)
    group_ids, group_counts, ex_ids, ex_counts = selection()
    actual = materialize((group_ids[:1], group_counts[:1], ex_ids[:1], ex_counts[:1]), length)
    expected = build_group_refined_mask(scores0.unsqueeze(2), base_mask.unsqueeze(2), 8, 0.8, 'group_max', 16)
    diff = int((actual != expected.squeeze(2).cpu()).sum())
    if diff:
        locations = (actual != expected.squeeze(2).cpu()).nonzero()[:20]
        print('first mask differences (batch, head, token, score, actual, expected):', flush=True)
        for b, h, token in locations.tolist():
            print(b, h, token, float(scores0[b, h, token]), bool(actual[b, h, token]),
                  bool(expected[b, h, 0, token]), flush=True)
        raise ValueError(f'compact selection differs from current Python reference at {diff} entries')
    trace_diff = int((actual != trace['gas'].bool()).sum())
    kernel = load(name='gas_group_warp_shared_pool', sources=[str(ROOT / 'csrc/src/gas_group_warp_shared_pool.cu')],
                  extra_cuda_cflags=['-O3'], verbose=False)
    modes = {
        'tiled8': (kernel.decode_tiled, 8, 1, 64),
        'split8p8': (kernel.decode, 8, 8, 2),
        'tiled8p8': (kernel.decode_tiled, 8, 8, 2),
        'split8p4': (kernel.decode, 8, 4, 2),
    }
    counted = None
    cooperative = None
    k_batch = v_batch = None
    if args.decode_mode == 'perhead_counted8':
        counted = load(name='gas_counted_decode', sources=[str(ROOT / 'csrc/src/gas_counted_decode.cu')],
                       extra_cuda_cflags=['-O3'], verbose=False)
        k_batch = k.transpose(0, 1).unsqueeze(0).repeat(args.batch, 1, 1, 1).contiguous()
        v_batch = v.transpose(0, 1).unsqueeze(0).repeat(args.batch, 1, 1, 1).contiguous()
    elif args.decode_mode in ('coop8p8', 'coop8p4'):
        cooperative = load(name='gas_cooperative_shared_pool',
                           sources=[str(ROOT / 'csrc/src/gas_cooperative_shared_pool.cu')],
                           extra_cuda_cflags=['-O3'], verbose=False)
    def gas_attention():
        if counted is not None:
            return counted.decode(q, k_batch, v_batch, group_ids, ex_ids, group_counts, ex_counts, 8)
        if cooperative is not None:
            parts = 8 if args.decode_mode == 'coop8p8' else 4
            return cooperative.decode(q, k, v, group_ids, ex_ids, group_counts, ex_counts, 8, parts, 2)
        fn, segments, parts, chunk = modes[args.decode_mode]
        return fn(q, k, v, group_ids, ex_ids, group_counts, ex_counts, segments, parts, chunk)
    def gas_path():
        selection()
        return gas_attention()
    twi_attention, keepalive = grouped_fi(q, k, v, base_mask[0])
    gas_reference = reference(q[:1], k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0), actual.cuda())
    twi_reference = reference(q[:1], k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0), base_mask)
    for name, fn, ref in [('gas', gas_path, gas_reference), ('twilight', twi_attention, twi_reference)]:
        value = fn()
        if not torch.isfinite(value).all() or not torch.allclose(value[:1].float(), ref, atol=.02, rtol=.02):
            raise ValueError(f'{name} attention differs from FP32 masked reference')
    calls = {'twilight_attention': twi_attention,
             'gas_group_stage': group_call,
             'gas_exception_stage': exception_call,
             'gas_refinement': selection,
             'gas_attention': gas_attention, 'gas_connected': gas_path}
    if args.include_quest:
        quest_mask = trace['candidate'].cuda().bool().contiguous()
        quest_attention, quest_keepalive = grouped_fi(q, k, v, quest_mask[0])
        quest_reference = reference(q[:1], k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0), quest_mask)
        quest_result = quest_attention()
        if not torch.isfinite(quest_result).all() or not torch.allclose(quest_result[:1].float(), quest_reference, atol=.02, rtol=.02):
            raise ValueError('quest attention differs from FP32 masked reference')
        calls['quest_attention'] = quest_attention
    if not args.tagged_flags:
        calls['gas_flags_zero'] = lambda: out[2].zero_()
    results = {}
    for name, fn in calls.items():
        median, samples, _ = graph_timing(fn, args.warmups, args.repeats, args.rounds, args.graph_iterations)
        results[name] = {'median_us': median, 'samples_us': samples}
        print(f'{name}: {median:.3f} us', flush=True)
    paired, paired_orders = {}, []
    if args.paired:
        paired_calls = {'twilight_attention': twi_attention, 'gas_connected': gas_path}
        if args.include_quest:
            paired_calls['quest_attention'] = quest_attention
        paired, paired_orders = interleaved_graph_timing(
            paired_calls, args.warmups, args.repeats, args.rounds, args.graph_iterations, 42)
        for name, timing in paired.items():
            print(f"paired_{name}: {timing['median_us']:.3f} us", flush=True)
        after = gas_path()
        after_mask = materialize((group_ids[:1], group_counts[:1], ex_ids[:1], ex_counts[:1]), length)
        if not torch.equal(after_mask, actual) or not torch.allclose(after[:1].float(), gas_reference, atol=.02, rtol=.02):
            raise ValueError('GAS output changed after interleaved graph timings')
    sources = [Path(__file__), ROOT / 'csrc/src/gas_compact_refine.cu',
               ROOT / 'csrc/src/gas_group_warp_shared_pool.cu',
               ROOT / 'csrc/src/gas_counted_decode.cu',
               ROOT / 'csrc/src/gas_cooperative_shared_pool.cu',
               ROOT / 'benchmark/efficiency/gas_compact_refinement.py']
    result = {'scenario': args.scenario, 'context': args.context, 'sample': args.sample, 'layer': args.layer,
              'batch': args.batch, 'length': length, 'candidate_capacity': indices.shape[-1], 'attention_dtype': args.attention_dtype,
              'use_trace_scores': args.use_trace_scores,
              'bucketed': args.bucketed or args.ragged, 'ragged': args.ragged, 'cuda_cub': args.cuda_cub,
              'tiled_exception': args.tiled_exception, 'triton_exception': args.triton_exception,
              'tagged_flags': args.tagged_flags,
              'blocksort_exception': args.blocksort_exception,
              'shared_scratch': args.shared_scratch,
              'hist_exception': args.hist_exception,
              'dense_group_scan': args.dense_group_scan,
              'fused_group_reductions': args.fused_group_reductions,
              'warp_group_compaction': args.warp_group_compaction,
              'normalized_group': args.normalized_group,
              'normalized_shared_group': args.normalized_shared_group,
              'pair_exception': args.pair_exception,
              'normalized_fast_scores': args.normalized_fast_scores,
              'normalized_prepared': args.normalized_prepared,
              'unified_exception': args.unified_exception,
              'decode_mode': args.decode_mode,
              'twilight_group_union_mean': float(base_mask.view(1, 8, 4, length).any(2).sum(-1).float().mean()),
              'gas_group_union_mean': float(actual.view(1, 8, 4, length).any(2).sum(-1).float().mean()),
              'shared_group_mean': float(group_counts.float().mean()),
              'exception_head_mean': float(ex_counts.float().mean()),
              'reconstructed_vs_captured_gas_mask_differences': trace_diff,
              'prototype_vs_python_mask_differences': diff, 'attention_reference_pass': True,
              'timings': results, 'trace': str(trace_path), 'device': torch.cuda.get_device_name(),
              'paired_timings': paired, 'paired_round_orders': paired_orders,
              'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              'timing_method': 'CUDA graph event median; independent stages and connected GAS path',
              'limitations': ['one captured request repeated within batch; FI always uses one physical KV pool',
                              '32k Qasper traces repeat context rather than using naturally 32k samples',
                              'Twilight uses preplanned FI grouped masked prefill, not author native kernel',
                              'GAS uses research kernels, not production serving; perhead_counted8 uses physically duplicated KV with per-head CTAs',
                              'common selector, estimator and head top-p are excluded for both methods',
                              'compact head ids and dense head mask prebuilt, their construction cost excluded',
                              'FI metadata and plan prebuilt; GAS flags zeroing included unless tagged_flags',
                              'tagged flags use mutable per-group epochs, require overflow and concurrency handling in production',
                              'synthetic trace scores are independent of Q/K' if args.use_trace_scores else 'BF16 estimator logits reconstructed from captured Q/K; not captured estimator output',
                              'attention dtype specified in result; mask/trajectory comes from GAS inference']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
