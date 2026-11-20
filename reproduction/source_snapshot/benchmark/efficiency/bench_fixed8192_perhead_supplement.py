#!/usr/bin/env python3
"""Per-head FI replay for the fixed-8192 grouped-FI synthetic fixture."""
import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import random
import statistics
from pathlib import Path

import torch

from bench_fixed8192_group_supplement import make_supplement_masks
from bench_twilight_group_fi import grouped_mask_fi
from bench_twilight_upstream_six import author_budgets, fi_sparse, reference


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--length', type=int, required=True)
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--warmups', type=int, default=10)
    p.add_argument('--repeats', type=int, default=100)
    p.add_argument('--rounds', type=int, default=5)
    args = p.parse_args()
    if args.length < 12514 or min(args.batch, args.repeats, args.rounds) <= 0 or args.warmups < 0:
        p.error('length must be >=12514, batch/repeats/rounds positive and warmups nonnegative')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f'refusing to overwrite {args.output_dir}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    author = args.upstream / 'benchmark/efficiency/bench_breakdown.py'
    bench_path = args.upstream / 'benchmark/efficiency/bench_utils.py'
    spec = importlib.util.spec_from_file_location('author_bench_utils', bench_path)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    budgets = author_budgets(author, 256)
    extra, original = make_supplement_masks(args.length, budgets, args.seed + args.length)
    masks = {'quest_group12514': extra['quest_group12514'],
             'twilight': original['twilight'], 'gas': original['gas'],
             'less_twi_matched': original['less_style'], 'less_group8192': extra['less_group8192']}
    torch.manual_seed(args.seed + args.length)
    kv = torch.randn(args.length, 2, 1, 8, 128, dtype=torch.float16, device='cuda')
    k, v = kv[:, 0, 0].contiguous(), kv[:, 1, 0].contiguous()
    torch.manual_seed(args.seed + args.length + args.batch)
    q = torch.randn(args.batch, 32, 128, dtype=torch.float16, device='cuda')
    calls, keepalive, errors = {}, [], {}
    for name, mask in masks.items():
        flat, flat_wrapper, flat_pages = fi_sparse(q, k, v, mask)
        grouped, wrapper, pages = grouped_mask_fi(q, k, v, mask)
        actual, expected = flat(), grouped()
        errors[name] = float((actual.float() - expected.float()).abs().max())
        if not torch.allclose(actual, expected, atol=.003, rtol=.003):
            raise ValueError(f'{name}: per-head FI differs from grouped FI: {errors[name]}')
        if not torch.allclose(actual[0].float(), reference(q, k, v, mask), atol=.02, rtol=.02):
            raise ValueError(f'{name}: per-head FI differs from FP32 reference')
        calls[name] = flat
        keepalive.extend((wrapper, pages, flat_wrapper, flat_pages))
    sources = [Path(__file__), Path(__file__).with_name('bench_fixed8192_group_supplement.py'),
               Path(__file__).with_name('bench_twilight_fixed8192_grouped.py'),
               Path(__file__).with_name('bench_twilight_group_fi.py'),
               Path(__file__).with_name('bench_twilight_upstream_six.py'), author, bench_path]
    meta = {'length': args.length, 'batch': args.batch, 'seed': args.seed,
            'warmups': args.warmups, 'repeats': args.repeats, 'rounds': args.rounds,
            'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'flashinfer': importlib.metadata.version('flashinfer-python'),
            'fixture': 'synthetic shared physical KV and support across batch; Q differs per request',
            'timing': 'author torch.utils.benchmark helper; per-head FI paged decode; excludes plan and selection',
            'backend': 'FI per-head adapter, not native Quest, Twilight, GAS or LessIsMore kernel',
            'support_sha256': {n: hashlib.sha256(m.numpy().tobytes()).hexdigest() for n, m in masks.items()},
            'source_sha256': {str(x.resolve()): hashlib.sha256(x.read_bytes()).hexdigest() for x in sources}}
    (args.output_dir / 'config.json').write_text(json.dumps(meta, indent=2) + '\n')
    times = {name: [] for name in calls}
    order = list(calls)
    rng = random.Random(args.seed)
    for round_id in range(args.rounds):
        rng.shuffle(order)
        for name in order:
            _, result = bench.benchmark_forward_with_warmup(calls[name], warmups=args.warmups, repeats=args.repeats)
            times[name].append(result.mean * 1e6)
        print(f'finished round {round_id + 1}/{args.rounds}', flush=True)
    rows = []
    for name, mask in masks.items():
        rows.append({'method': name, 'length': args.length, 'batch': args.batch,
                     'median_us': statistics.median(times[name]), 'round_means_us': json.dumps(times[name]),
                     'tokens_per_head_mean': float(mask.sum(-1).float().mean()),
                     'union_per_group_mean': float(mask.view(8, 4, args.length).any(1).sum(-1).float().mean()),
                     'max_abs_error_vs_grouped_fi': errors[name]})
    with (args.output_dir / 'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['method']}: {row['median_us']:.2f} us; head={row['tokens_per_head_mean']:.2f}; union={row['union_per_group_mean']:.2f}")
    print(f'COMPLETE {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
