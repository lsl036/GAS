#!/usr/bin/env python3
"""Group-FI supplemental synthetic supports for fixed-8192 attention replay."""
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

from bench_twilight_fixed8192_grouped import make_masks
from bench_twilight_group_fi import grouped_mask_fi
from bench_twilight_upstream_six import author_budgets, fi_sparse, reference


def make_supplement_masks(length, budgets, seed):
    """Keep the original Twilight/GAS masks inside a controlled Quest union."""
    if length < 12514:
        raise ValueError('length must be at least 12514')
    original, _ = make_masks(length, budgets, seed)
    generator = torch.Generator().manual_seed(seed + 200000)
    quest = torch.zeros_like(original['twilight'])
    less = torch.zeros_like(quest)
    for group in range(8):
        heads = slice(group * 4, (group + 1) * 4)
        twi = original['twilight'][heads]
        union = twi.any(0)
        union_size = int(union.sum())
        if union_size > 12514:
            raise ValueError('Twilight union does not fit in the target Quest union')
        spare = (~union).nonzero().flatten()
        spare = spare[torch.randperm(spare.numel(), generator=generator)]
        universe = torch.cat((union.nonzero().flatten(), spare[:12514 - union_size]))
        universe = universe[torch.randperm(12514, generator=generator)]
        quest[heads] = twi
        for head in range(4):
            candidate = quest[group * 4 + head]
            candidate[universe[head::4]] = True
            remaining = 8192 - int(candidate.sum())
            if remaining < 0:
                raise ValueError('initial candidate assignment exceeds head budget')
            available = universe[~candidate[universe]]
            order = torch.randperm(available.numel(), generator=generator)
            candidate[available[order[:remaining]]] = True
        ids = torch.randperm(length, generator=generator)[:8192]
        less[heads, ids] = True
    if not torch.all(quest.sum(-1) == 8192):
        raise ValueError('incorrect Quest per-head count')
    if not torch.all(quest.view(8, 4, length).any(1).sum(-1) == 12514):
        raise ValueError('incorrect Quest group union')
    if (original['twilight'] & ~quest).any():
        raise ValueError('Twilight supports must remain within Quest candidates')
    return {'quest_group12514': quest, 'less_group8192': less}, original


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
    masks, original = make_supplement_masks(args.length, budgets, args.seed + args.length)
    torch.manual_seed(args.seed + args.length)
    kv = torch.randn(args.length, 2, 1, 8, 128, dtype=torch.float16, device='cuda')
    k, v = kv[:, 0, 0].contiguous(), kv[:, 1, 0].contiguous()
    torch.manual_seed(args.seed + args.length + args.batch)
    q = torch.randn(args.batch, 32, 128, dtype=torch.float16, device='cuda')
    calls, keepalive, errors = {}, [], {}
    for name, mask in masks.items():
        grouped, wrapper, pages = grouped_mask_fi(q, k, v, mask)
        flat, flat_wrapper, flat_pages = fi_sparse(q, k, v, mask)
        actual, expected = grouped(), flat()
        errors[name] = float((actual.float() - expected.float()).abs().max())
        if not torch.allclose(actual, expected, atol=.003, rtol=.003):
            raise ValueError(f'{name}: grouped FI differs from per-head FI: {errors[name]}')
        if not torch.allclose(actual[0].float(), reference(q, k, v, mask), atol=.02, rtol=.02):
            raise ValueError(f'{name}: grouped FI differs from FP32 reference')
        calls[name] = grouped
        keepalive.extend((wrapper, pages, flat_wrapper, flat_pages))
    sources = [Path(__file__), Path(__file__).with_name('bench_twilight_fixed8192_grouped.py'),
               Path(__file__).with_name('bench_twilight_group_fi.py'),
               Path(__file__).with_name('bench_twilight_upstream_six.py'), author, bench_path]
    meta = {'length': args.length, 'batch': args.batch, 'seed': args.seed,
            'warmups': args.warmups, 'repeats': args.repeats, 'rounds': args.rounds,
            'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'flashinfer': importlib.metadata.version('flashinfer-python'),
            'fixture': 'synthetic shared physical KV and support across batch; Q differs per batch request',
            'quest': '8192 tokens per head, exactly 12514 union per group; constructed to contain prior Twilight supports',
            'less': '8192 random shared tokens per group; synthetic budget-only proxy, not official LessIsMore',
            'timing': 'author torch.utils.benchmark helper; grouped FI paged prefill with per-head mask; plan and selection excluded',
            'support_sha256': {n: hashlib.sha256(m.numpy().tobytes()).hexdigest() for n, m in {**masks, 'twilight': original['twilight'], 'gas': original['gas']}.items()},
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
                     'max_abs_error_vs_flat_fi': errors[name]})
    with (args.output_dir / 'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['method']}: {row['median_us']:.2f} us; head={row['tokens_per_head_mean']:.2f}; union={row['union_per_group_mean']:.2f}")
    print(f'COMPLETE {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
