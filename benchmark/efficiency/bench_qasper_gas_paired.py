#!/usr/bin/env python3
"""Paired GAS attention replay on the two-request Qasper rate08 traces.

FI's connected case replays refinement but consumes a *static* preplanned mask;
it is not an implementation of dynamic refinement-to-FI metadata construction.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

from bench_compact_head_pruning import interleaved_graph_timing
from bench_qasper_rate08_grouped import LAYERS, load_trace, plan_sparse
from bench_twilight_upstream_six import reference
from gas_compact_refinement import allocate, exception_stage_unified
from test_gas_compact_refinement import materialize, pack
from gas_attention.pyimpl.gqa_refine import build_group_refined_mask
from gas_attention.pyimpl.quantize import min_max_per_token_quant_kv

ROOT = Path(__file__).resolve().parents[2]


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--context', choices=('16k', '32k'), required=True)
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--layers', type=int, nargs='+', default=list(LAYERS))
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--warmups', type=int, default=3)
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--graph-iterations', type=int, default=4)
    p.add_argument('--timer', choices=('graph', 'author'), default='graph')
    p.add_argument('--attention-only', action='store_true', help='Only time the three attention backends.')
    args = p.parse_args()
    if args.batch <= 0 or args.batch % 2 or any(x not in LAYERS for x in args.layers):
        p.error('batch must be positive and even; layers must be in the original 28 layers')
    if len(set(args.layers)) != len(args.layers) or min(args.repeats, args.rounds, args.graph_iterations) < 1 or args.warmups < 0:
        p.error('layers must be distinct and timing parameters positive')
    prefix = 'qasper16k' if args.context == '16k' else 'qasper32k_repeated'
    dirs = [ROOT / 'benchmark/results_efficiency/fig6_traces' / f'{prefix}_rate08_s{x}' for x in (15, 54)]
    manifests = [json.loads((d / 'manifest.json').read_text()) for d in dirs]
    if any(m['task'] != 'qasper' or m['config'] != manifests[0]['config'] for m in manifests):
        raise ValueError('incompatible Qasper manifests')
    if manifests[0]['config']['selector'].get('sparsity_rate') != .08 or len({m['sample_sha256'] for m in manifests}) != 2:
        raise ValueError('expected two distinct Quest 8% samples')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {'context': args.context, 'batch': args.batch, 'layers': args.layers,
              'trace_dirs': [str(d) for d in dirs],
              'manifest_sha256': [hashlib.sha256((d / 'manifest.json').read_bytes()).hexdigest() for d in dirs],
              'warmups': args.warmups, 'repeats': args.repeats, 'rounds': args.rounds,
              'graph_iterations': args.graph_iterations, 'device': torch.cuda.get_device_name(),
              'timer': args.timer, 'attention_only': args.attention_only,
              'timing': ('randomized interleaved CUDA graph event median' if args.timer == 'graph' else
                         'author torch.utils.benchmark helper; randomized round order, median of round means'),
              'limitations': ['two Qasper requests alternated across batch with two physical KV pools',
                  '32K traces use repeated context', 'BF16 scores reconstructed from captured Q/K, not saved estimator logits',
                  'prepacked Twilight indices and FI plan excluded',
                  'FI connected replay runs refinement but FI consumes a static preplanned mask; dynamic mask conversion and replanning excluded',
                  'custom connected replay includes adding pool offsets after refinement; already shifted offsets are prebuilt for attention-only',
                  'candidate, estimator and Twilight top-p excluded', 'experimental shared-pool kernel; not full model inference']}
    config_path = args.output_dir / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('output directory already contains a different configuration')
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    cub = load(name='gas_compact_refine', sources=[str(ROOT / 'csrc/src/gas_compact_refine.cu')], extra_cuda_cflags=['-O3'], verbose=False)
    kernel = load(name='gas_cooperative_shared_pool', sources=[str(ROOT / 'csrc/src/gas_cooperative_shared_pool.cu')], extra_cuda_cflags=['-O3'], verbose=False)
    author = None
    if args.timer == 'author':
        bench_path = ROOT.parent / 'Twilight_Upstream_20261113/benchmark/efficiency/bench_utils.py'
        spec = importlib.util.spec_from_file_location('qasper_author_bench_utils', bench_path)
        author = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(author)
    for layer in args.layers:
        result_path = args.output_dir / f'layer{layer:02d}.json'
        if result_path.exists():
            print(f'SKIP layer={layer}', flush=True)
            continue
        traces = [load_trace(d / f'step001_layer{layer:02d}.pt') for d in dirs]
        lengths = [t['k'].shape[2] for t in traces]
        length = max(lengths)
        q0 = torch.cat([t['q'].to('cuda', dtype=torch.float16) for t in traces])
        order = torch.arange(args.batch, device='cuda') % 2
        q = q0[order].contiguous()
        k = torch.zeros((2, length, 8, 128), device='cuda', dtype=torch.float16)
        v = torch.zeros_like(k)
        masks, scores0, captured_gas = [], [], []
        for i, t in enumerate(traces):
            n = lengths[i]
            k[i, :n] = t['k'][0].permute(1, 0, 2).to('cuda', dtype=torch.float16)
            v[i, :n] = t['v'][0].permute(1, 0, 2).to('cuda', dtype=torch.float16)
            masks.append(F.pad(t['twilight'][0].bool(), (0, length - n)))
            captured_gas.append(t['gas'][0].bool())
            qbf, kbf = t['q'].cuda(), t['k'].cuda().repeat_interleave(4, dim=1)
            s = (torch.matmul(qbf.unsqueeze(2), min_max_per_token_quant_kv(kbf, 4).transpose(-1, -2)) / math.sqrt(128)).squeeze(2)
            scores0.append(F.pad(s, (0, length - n)))
        masks0 = torch.stack(masks).cuda()
        mask = masks0[order].contiguous()
        scores = torch.cat(scores0)[order].contiguous()
        indices = pack(mask).contiguous()
        counts = mask.sum(-1, dtype=torch.int32).contiguous()
        out = allocate(args.batch, indices.shape[-1], length, q.device)
        out = (out[0], out[1], torch.zeros((args.batch, 8, length), dtype=torch.int32, device=q.device), out[3], out[4])
        epochs = torch.zeros((args.batch, 8), dtype=torch.int32, device=q.device)
        scratch = torch.empty((args.batch, 8, 4 * indices.shape[-1]), dtype=torch.float32, device=q.device)
        offsets = (order * length).to(torch.int32).view(-1, 1, 1)

        def refine():
            cub.group_tagged_normalized_prepared(indices, counts, mask, scores, out[0], out[1], out[2], epochs, scratch, .8)
            exception_stage_unified(indices, scores, out, counts, epochs)

        def shift():
            # Current kernel uses one flat KV pool; shift local IDs to the appropriate pool.
            out[0].add_(offsets)
            out[3].add_(offsets)

        refine()
        actual = materialize((out[0][:2], out[1][:2], out[3][:2], out[4][:2]), length)
        expected = build_group_refined_mask(scores[:2].unsqueeze(2), mask[:2].unsqueeze(2), 8, .8, 'group_max', 16).squeeze(2).cpu()
        captured = torch.stack([F.pad(x, (0, length - x.shape[-1])) for x in captured_gas])
        ref_diff = int((actual != expected).sum())
        trace_diff = int((actual != captured).sum())
        if ref_diff or trace_diff:
            raise ValueError(f'layer={layer}: reconstructed GAS differs: python={ref_diff}, captured={trace_diff}')
        fi, keepalive = plan_sparse(q, k, v, captured_gas, True)
        fi_head, head_keepalive = plan_sparse(q, k, v, captured_gas, False)
        shift()
        def custom():
            return kernel.decode(q, k.view(-1, 8, 128), v.view(-1, 8, 128), out[0], out[3], out[1], out[4], 8, 4, 2)
        fi_output, head_output, custom_output = fi(), fi_head(), custom()
        head_diff = float((fi_output.float() - head_output.float()).abs().max())
        if not torch.allclose(fi_output.float(), head_output.float(), atol=.003, rtol=.003):
            raise ValueError(f'layer={layer}: FI group/head max error {head_diff}')
        max_diff = float((fi_output.float() - custom_output.float()).abs().max())
        if not torch.allclose(fi_output.float(), custom_output.float(), atol=.02, rtol=.02):
            raise ValueError(f'layer={layer}: FI/custom max error {max_diff}')
        fp32_error = []
        for i in range(2):
            ref = reference(q[i:i+1], k[i, :lengths[i]], v[i, :lengths[i]], captured_gas[i])
            fp32_error.append(float((custom_output[i].float() - ref).abs().max()))
            if not torch.allclose(custom_output[i].float(), ref, atol=.02, rtol=.02):
                raise ValueError(f'layer={layer}, request={i}: FP32 reference mismatch')

        # Independent prebuilt IDs keep attention-only measurement free of shifts.
        shared, ex = out[0].clone(), out[3].clone()
        sc, ec = out[1].clone(), out[4].clone()
        def custom_static():
            return kernel.decode(q, k.view(-1, 8, 128), v.view(-1, 8, 128), shared, ex, sc, ec, 8, 4, 2)
        def custom_connected():
            refine()
            shift()
            return custom()
        def fi_static_refinement():
            refine()
            return fi()
        calls = {'fi_attention': fi, 'fi_head_attention': fi_head, 'custom_attention': custom_static}
        if not args.attention_only:
            calls.update(fi_refine_static_attention=fi_static_refinement, custom_refine_attention=custom_connected)
        if args.timer == 'graph':
            times, orders = interleaved_graph_timing(calls, args.warmups, args.repeats, args.rounds, args.graph_iterations,
                                                     seed=42 + layer + args.batch)
        else:
            samples = {name: [] for name in calls}
            orders = []
            rng = random.Random(42 + layer + args.batch)
            for _ in range(args.rounds):
                names = list(calls)
                rng.shuffle(names)
                orders.append(names)
                for name in names:
                    _, timing = author.benchmark_forward_with_warmup(calls[name], warmups=args.warmups, repeats=args.repeats)
                    samples[name].append(timing.mean * 1e6)
            times = {name: {'median_us': statistics.median(values), 'round_means_us': values} for name, values in samples.items()}
        # Refinement updates are checked once more after graph replay.
        refine()
        final = materialize((out[0][:2], out[1][:2], out[3][:2], out[4][:2]), length)
        if not torch.equal(final, captured):
            raise ValueError(f'layer={layer}: mask changed after replay')
        result = {'layer': layer, 'lengths': lengths, 'batch': args.batch,
                  'captured_mask_differences': trace_diff, 'python_mask_differences': ref_diff,
                  'fi_custom_max_abs_error_all_batch': max_diff, 'custom_fp32_max_abs_error_first_two': fp32_error,
                  'fi_head_group_max_abs_error_all_batch': head_diff,
                  'gas_group_union_mean': sum(float(x.view(8, 4, -1).any(1).sum(-1).float().mean()) for x in captured_gas) / 2,
                  'timings': times, 'round_orders': orders}
        tmp = result_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(result, indent=2) + '\n')
        tmp.replace(result_path)
        print(f'layer={layer} batch={args.batch} complete: FI={times["fi_attention"]["median_us"]:.2f} custom={times["custom_attention"]["median_us"]:.2f}', flush=True)
        del traces, q, k, v, masks, scores0, indices, mask, scores, out, epochs, scratch, keepalive, head_keepalive, fi, fi_head, calls, times
        torch.cuda.empty_cache()
    rows = []
    for layer in args.layers:
        result = json.loads((args.output_dir / f'layer{layer:02d}.json').read_text())
        row = {key: result[key] for key in ('layer', 'batch', 'lengths', 'gas_group_union_mean', 'fi_custom_max_abs_error_all_batch', 'captured_mask_differences')}
        row.update({name + '_us': val['median_us'] for name, val in result['timings'].items()})
        rows.append(row)
    with (args.output_dir / 'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'COMPLETE {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
