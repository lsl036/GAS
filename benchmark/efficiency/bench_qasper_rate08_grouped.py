#!/usr/bin/env python3
"""Qasper trace replay with per-head and grouped FI, Less-style K=8%."""
import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import random
import statistics
from pathlib import Path

import flashinfer
import torch

from bench_e6_qasper_official_b1 import less_mask
from bench_twilight_upstream_six import reference

LAYERS = tuple(x for x in range(3, 32) if x != 13)


def plan_sparse(q, k, v, masks, grouped):
    """Plan FI using two real per-request KV pools repeated across the batch."""
    batch, heads, dim = q.shape
    padded_len = k.shape[1]
    counts_by_sample, ids_by_sample, custom_by_sample = [], [], []
    for sample, mask in enumerate(masks):
        length = mask.shape[-1]
        m = mask.view(8, 4, length)
        counts, ids, custom = [], [], []
        for group in range(8):
            if grouped:
                tokens = m[group].any(0).nonzero().flatten()
                counts.append(tokens.numel())
                ids.append((sample * padded_len + tokens) * 8 + group)
                custom.append(m[group, :, tokens].reshape(-1))
            else:
                for head in range(4):
                    tokens = m[group, head].nonzero().flatten()
                    counts.append(tokens.numel())
                    ids.append((sample * padded_len + tokens) * 8 + group)
        if min(counts) == 0:
            raise ValueError('empty head/group support')
        counts_by_sample.append(torch.tensor(counts, dtype=torch.int32))
        ids_by_sample.append(torch.cat(ids).int())
        if grouped:
            custom_by_sample.append(torch.cat(custom))
    counts = torch.cat([counts_by_sample[b % 2] for b in range(batch)]).to(q.device)
    ids = torch.cat([ids_by_sample[b % 2] for b in range(batch)]).to(q.device)
    ptr = torch.nn.functional.pad(counts.cumsum(0, dtype=torch.int32), (1, 0))
    last = torch.ones_like(counts)
    pages = (k.view(-1, 1, 1, dim), v.view(-1, 1, 1, dim))
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    if grouped:
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, 'NHD')
        qptr = torch.arange(batch * 8 + 1, device=q.device, dtype=torch.int32) * 4
        custom = torch.cat([custom_by_sample[b % 2] for b in range(batch)]).to(q.device)
        wrapper.plan(qptr, ptr, ids, last, 1, 1, dim, 1, custom_mask=custom,
                     causal=False, pos_encoding_mode='NONE', q_data_type=q.dtype,
                     kv_data_type=k.dtype, sm_scale=1 / math.sqrt(dim))
    else:
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, 'NHD', use_tensor_cores=False)
        wrapper.plan(ptr, ids, last, 1, 1, dim, 1, pos_encoding_mode='NONE',
                     q_data_type=q.dtype, kv_data_type=k.dtype, sm_scale=1 / math.sqrt(dim))
    return lambda: wrapper.run(q.view(-1, 1, dim), pages).view_as(q), (wrapper, pages, workspace)


def load_trace(path):
    return torch.load(path, map_location='cpu', weights_only=True, mmap=True)


def less_search(trace, rate):
    length = trace['k'].shape[2]
    search = {key: trace[key].to('cuda') for key in ('q', 'k')}
    result = less_mask(search, length, int(length * rate))[0].cpu()
    return result


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--trace-dirs', type=Path, nargs=2, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--layers', type=int, nargs='+', default=list(LAYERS))
    p.add_argument('--warmups', type=int, default=5)
    p.add_argument('--repeats', type=int, default=50)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.batch <= 0 or args.batch % 2 or args.rounds <= 0 or args.repeats <= 0 or args.warmups < 0:
        p.error('batch must be positive and even; rounds/repeats positive; warmups nonnegative')
    if not args.layers or len(set(args.layers)) != len(args.layers) or any(x not in LAYERS for x in args.layers):
        p.error('layers must be distinct and in 3..31 excluding search layer 13')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f'refusing to overwrite {args.output_dir}')
    manifests = [json.loads((d / 'manifest.json').read_text()) for d in args.trace_dirs]
    if any(m['task'] != 'qasper' or m['config'] != manifests[0]['config'] for m in manifests):
        raise ValueError('expected Qasper traces with matching configurations')
    selector = manifests[0]['config']['selector']
    if selector.get('type') != 'quest' or selector.get('sparsity_rate') != 0.08:
        raise ValueError('expected Quest 8% traces')
    if len({m['sample_sha256'] for m in manifests}) != 2:
        raise ValueError('expected two distinct Qasper samples')
    bench_path = args.upstream / 'benchmark/efficiency/bench_utils.py'
    spec = importlib.util.spec_from_file_location('author_bench_utils', bench_path)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), Path(__file__).with_name('bench_e6_qasper_official_b1.py'), bench_path]
    config = {'trace_dirs': [str(d.resolve()) for d in args.trace_dirs],
              'manifest_sha256': [hashlib.sha256((d / 'manifest.json').read_bytes()).hexdigest() for d in args.trace_dirs],
              'batch': args.batch, 'layers': args.layers, 'warmups': args.warmups,
              'repeats': args.repeats, 'rounds': args.rounds, 'seed': args.seed,
              'input': 'two real Qasper GAS-trajectory traces, repeated round-robin; two physical KV pools; BF16 source cast FP16',
              'supports': 'Quest 8% captured, Twilight p=.85 captured, GAS p=.80 exception16 captured; Less-style search-layer mask K=floor(8% length), 25% recent',
              'less': 'search layer 2 for layers 3-12 and search layer 13 for layers 14-31; reconstructed from GAS states, not independent LessIsMore inference',
              'backend': 'per-head FI paged decode and group FI paged prefill with exact per-head mask; not native method kernels',
              'timing': 'author torch.utils.benchmark helper, median of round means; no selector, estimator, top-p, refinement, packing, planning, or model costs',
              'device': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
              'flashinfer': importlib.metadata.version('flashinfer-python'),
              'source_sha256': {str(x.resolve()): hashlib.sha256(x.read_bytes()).hexdigest() for x in sources}}
    (args.output_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    less = {layer: [less_search(load_trace(d / f'step001_layer{layer:02d}.pt'), .08)
                    for d in args.trace_dirs] for layer in (2, 13)}
    rows = []
    for layer in args.layers:
        traces = [load_trace(d / f'step001_layer{layer:02d}.pt') for d in args.trace_dirs]
        lengths = [t['k'].shape[2] for t in traces]
        max_len = max(lengths)
        q0 = torch.cat([t['q'].to('cuda', dtype=torch.float16) for t in traces])
        q = q0[torch.arange(args.batch, device='cuda') % 2].contiguous()
        k = torch.zeros(2, max_len, 8, 128, device='cuda', dtype=torch.float16)
        v = torch.zeros_like(k)
        for i, t in enumerate(traces):
            k[i, :lengths[i]] = t['k'][0].permute(1, 0, 2).to('cuda', dtype=torch.float16)
            v[i, :lengths[i]] = t['v'][0].permute(1, 0, 2).to('cuda', dtype=torch.float16)
        masks = {name: [t[key][0].bool() for t in traces] for name, key in
                 (('quest', 'candidate'), ('twilight', 'twilight'), ('gas', 'gas'))}
        masks['less_style_8pct'] = less[2 if layer < 13 else 13]
        calls, keepalive, error = {}, [], {}
        for name, mask in masks.items():
            flat, hold1 = plan_sparse(q, k, v, mask, False)
            grouped, hold2 = plan_sparse(q, k, v, mask, True)
            out, out_group = flat(), grouped()
            error[name] = float((out.float() - out_group.float()).abs().max())
            if not torch.allclose(out, out_group, atol=.003, rtol=.003):
                raise ValueError(f'layer {layer} {name}: grouped/flat max error {error[name]}')
            for sample in range(2):
                ref = reference(q[sample:sample + 1], k[sample, :lengths[sample]], v[sample, :lengths[sample]], mask[sample])
                if not torch.allclose(out[sample].float(), ref, atol=.02, rtol=.02):
                    raise ValueError(f'layer {layer} {name}: FP32 reference mismatch sample {sample}')
            calls[name + '_per_head'] = flat
            calls[name + '_grouped'] = grouped
            keepalive.extend((hold1, hold2))
        timings = {name: [] for name in calls}
        order = list(calls)
        rng = random.Random(f'{args.seed}:{layer}:{args.batch}')
        for _ in range(args.rounds):
            rng.shuffle(order)
            for name in order:
                _, result = bench.benchmark_forward_with_warmup(calls[name], warmups=args.warmups, repeats=args.repeats)
                timings[name].append(result.mean * 1e6)
        for method, mask in masks.items():
            per_head = statistics.fmean(float(m.sum(-1).float().mean()) for m in mask)
            union = statistics.fmean(float(m.view(8, 4, -1).any(1).sum(-1).float().mean()) for m in mask)
            for backend in ('per_head', 'grouped'):
                samples = timings[method + '_' + backend]
                rows.append({'layer': layer, 'batch': args.batch, 'lengths': json.dumps(lengths),
                             'method': method, 'backend': backend, 'median_us': statistics.median(samples),
                             'round_means_us': json.dumps(samples), 'per_head_mean': per_head,
                             'group_union_mean': union, 'max_abs_error': error[method]})
        tmp = args.output_dir / 'results.tmp'
        with tmp.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(args.output_dir / 'results.csv')
        print(f'layer={layer} batch={args.batch} complete', flush=True)
        del traces, q, k, v, calls, keepalive, out, out_group
        torch.cuda.empty_cache()
    print(f'COMPLETE {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
