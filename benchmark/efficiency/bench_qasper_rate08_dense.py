#!/usr/bin/env python3
"""Dense FA2 and FI replay on the same two-pool Qasper trace layout."""
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
from flash_attn import flash_attn_with_kvcache

from bench_qasper_rate08_grouped import load_trace
from bench_twilight_upstream_six import reference


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--trace-dirs', type=Path, nargs=2, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--layer', type=int, default=15)
    p.add_argument('--warmups', type=int, default=5)
    p.add_argument('--repeats', type=int, default=50)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.batch <= 0 or args.batch % 2 or min(args.repeats, args.rounds) <= 0 or args.warmups < 0:
        p.error('batch must be positive and even; repeats/rounds positive; warmups nonnegative')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f'refusing to overwrite {args.output_dir}')
    manifests = [json.loads((d / 'manifest.json').read_text()) for d in args.trace_dirs]
    if any(m['task'] != 'qasper' or m['config'] != manifests[0]['config'] for m in manifests):
        raise ValueError('expected Qasper traces with matching configurations')
    if manifests[0]['config']['selector'].get('sparsity_rate') != 0.08:
        raise ValueError('expected Quest 8% traces')
    if len({m['sample_sha256'] for m in manifests}) != 2:
        raise ValueError('expected two distinct samples')
    bench_path = args.upstream / 'benchmark/efficiency/bench_utils.py'
    spec = importlib.util.spec_from_file_location('author_bench_utils', bench_path)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    traces = [load_trace(d / f'step001_layer{args.layer:02d}.pt') for d in args.trace_dirs]
    lengths = [t['k'].shape[2] for t in traces]
    max_len = max(lengths)
    q0 = torch.cat([t['q'].to('cuda', dtype=torch.float16) for t in traces])
    q = q0[torch.arange(args.batch, device='cuda') % 2].contiguous()
    k = torch.zeros(2, max_len, 8, 128, device='cuda', dtype=torch.float16)
    v = torch.zeros_like(k)
    for i, t in enumerate(traces):
        k[i, :lengths[i]] = t['k'][0].permute(1, 0, 2).to('cuda', dtype=torch.float16)
        v[i, :lengths[i]] = t['v'][0].permute(1, 0, 2).to('cuda', dtype=torch.float16)
    sample_ids = torch.arange(args.batch, device='cuda', dtype=torch.int32) % 2
    seq_lens = torch.tensor([lengths[i % 2] for i in range(args.batch)], device='cuda', dtype=torch.int32)
    fa2 = lambda: flash_attn_with_kvcache(q.unsqueeze(1), k, v, cache_seqlens=seq_lens,
                                         cache_batch_idx=sample_ids, causal=False,
                                         softmax_scale=1 / math.sqrt(128)).squeeze(1)
    ptr = torch.nn.functional.pad(seq_lens.cumsum(0, dtype=torch.int32), (1, 0))
    page_ids = torch.cat([torch.arange(i % 2 * max_len, i % 2 * max_len + lengths[i % 2],
                                      device='cuda', dtype=torch.int32) for i in range(args.batch)])
    last = torch.ones(args.batch, device='cuda', dtype=torch.int32)
    workspace = torch.empty(128 * 1024 * 1024, device='cuda', dtype=torch.uint8)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, 'NHD', use_tensor_cores=False)
    wrapper.plan(ptr, page_ids, last, 32, 8, 128, 1, pos_encoding_mode='NONE',
                 q_data_type=q.dtype, kv_data_type=k.dtype, sm_scale=1 / math.sqrt(128))
    pages = (k.view(-1, 1, 8, 128), v.view(-1, 1, 8, 128))
    fi = lambda: wrapper.run(q, pages)
    output_fa, output_fi = fa2(), fi()
    max_error = float((output_fa.float() - output_fi.float()).abs().max())
    if not torch.allclose(output_fa, output_fi, atol=.003, rtol=.003):
        raise ValueError(f'FA2 differs from dense FI: {max_error}')
    for sample in range(2):
        ref = reference(q[sample:sample+1], k[sample, :lengths[sample]], v[sample, :lengths[sample]], None)
        if not torch.allclose(output_fa[sample].float(), ref, atol=.02, rtol=.02):
            raise ValueError(f'FP32 reference mismatch sample {sample}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = {'batch': args.batch, 'layer': args.layer, 'lengths': lengths, 'seed': args.seed,
            'warmups': args.warmups, 'repeats': args.repeats, 'rounds': args.rounds,
            'trace_dirs': [str(d.resolve()) for d in args.trace_dirs],
            'manifest_sha256': [hashlib.sha256((d / 'manifest.json').read_bytes()).hexdigest() for d in args.trace_dirs],
            'input': 'two real Qasper GAS-trajectory traces, repeated round-robin; two physical KV pools; BF16 source cast FP16',
            'timing': 'author torch.utils.benchmark helper; dense FA2 cache_batch_idx and dense FI page-1; plans and model costs excluded',
            'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'flashinfer': importlib.metadata.version('flashinfer-python'),
            'source_sha256': {str(x.resolve()): hashlib.sha256(x.read_bytes()).hexdigest() for x in (Path(__file__), bench_path)}}
    (args.output_dir / 'config.json').write_text(json.dumps(meta, indent=2) + '\n')
    calls = {'fa2': fa2, 'flashinfer_dense': fi}
    timings = {name: [] for name in calls}
    order = list(calls)
    rng = random.Random(args.seed)
    for _ in range(args.rounds):
        rng.shuffle(order)
        for name in order:
            _, result = bench.benchmark_forward_with_warmup(calls[name], warmups=args.warmups, repeats=args.repeats)
            timings[name].append(result.mean * 1e6)
    rows = [{'method': name, 'batch': args.batch, 'layer': args.layer,
             'median_us': statistics.median(values), 'round_means_us': json.dumps(values),
             'max_abs_error_vs_fi': max_error} for name, values in timings.items()]
    with (args.output_dir / 'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['method']}: {row['median_us']:.3f} us", flush=True)
    print(f'COMPLETE {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
