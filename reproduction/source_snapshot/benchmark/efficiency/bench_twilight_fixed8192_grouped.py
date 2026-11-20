#!/usr/bin/env python3
"""Fixed-budget synthetic attention replay with comparable grouped FI paths."""
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

import torch
from flash_attn import flash_attn_with_kvcache

from bench_twilight_group_fi import grouped_mask_fi
from bench_twilight_upstream_six import author_budgets, fi_dense, fi_sparse, reference, synthetic_masks


def make_masks(length, budgets, seed):
    masks, _, shared = synthetic_masks(length, 8192, budgets, seed)
    groups = masks['twilight'].view(8, 4, length).any(1)
    generator = torch.Generator().manual_seed(seed + 100000)
    less = torch.zeros_like(masks['twilight'])
    for group in range(8):
        count = int(groups[group].sum())
        ids = torch.randperm(length, generator=generator)[:count]
        less[group * 4:(group + 1) * 4, ids] = True
    masks['less_style'] = less
    del masks['lessismore_style']
    assert torch.equal(less.view(8, 4, length).any(1).sum(-1), groups.sum(-1))
    return masks, shared


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--length', type=int, default=16384)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--warmups', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--rounds', type=int, default=5)
    args = parser.parse_args()
    if args.length < 8192 or min(args.batch, args.repeats, args.rounds) <= 0 or args.warmups < 0:
        parser.error('length must be >=8192, batch/repeats/rounds positive and warmups nonnegative')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f'refusing to overwrite {args.output_dir}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    author_script = args.upstream / 'benchmark/efficiency/bench_breakdown.py'
    bench_path = args.upstream / 'benchmark/efficiency/bench_utils.py'
    spec = importlib.util.spec_from_file_location('author_bench_utils', bench_path)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    budgets = author_budgets(author_script, 256)
    masks, shared = make_masks(args.length, budgets, args.seed + args.length)
    torch.manual_seed(args.seed + args.length)
    kv = torch.randn(args.length, 2, 1, 8, 128, dtype=torch.float16, device='cuda')
    k, v = kv[:, 0, 0].contiguous(), kv[:, 1, 0].contiguous()
    torch.manual_seed(args.seed + args.length + args.batch)
    q = torch.randn(args.batch, 32, 128, dtype=torch.float16, device='cuda')
    kcache = k.unsqueeze(0).expand(args.batch, -1, -1, -1)
    vcache = v.unsqueeze(0).expand(args.batch, -1, -1, -1)
    dense, dense_wrapper = fi_dense(q, kv)
    calls = {
        'fa2': lambda: flash_attn_with_kvcache(q.unsqueeze(1), kcache, vcache,
                                              causal=False, softmax_scale=1 / math.sqrt(128)).squeeze(1),
        'flashinfer_dense': dense,
    }
    keepalive = [dense_wrapper, kcache, vcache]
    errors = {}
    for name, mask in masks.items():
        if name == 'quest':
            call, wrapper, pages = fi_sparse(q, k, v, mask)
            actual = call()
            expected = reference(q, k, v, mask)
            errors[name] = float((actual[0].float() - expected).abs().max())
            if not torch.allclose(actual[0].float(), expected, atol=.02, rtol=.02):
                raise ValueError(f'{name}: per-head FI differs from FP32 reference')
            calls['quest_per_head_fi'] = call
            keepalive.extend([wrapper, pages])
            continue
        call, wrapper, pages = grouped_mask_fi(q, k, v, mask)
        flat, flat_wrapper, flat_pages = fi_sparse(q, k, v, mask)
        actual, expected = call(), flat()
        errors[name] = float((actual.float() - expected.float()).abs().max())
        if not torch.allclose(actual, expected, atol=.003, rtol=.003):
            raise ValueError(f'{name}: grouped FI differs from flat FI, max error {errors[name]}')
        if not torch.allclose(actual[0].float(), reference(q, k, v, mask), atol=.02, rtol=.02):
            raise ValueError(f'{name}: grouped FI differs from FP32 reference')
        calls[name + '_grouped_fi'] = call
        keepalive.extend([wrapper, pages, flat_wrapper, flat_pages])
    for name in ('fa2', 'flashinfer_dense'):
        actual = calls[name]()
        expected = reference(q, k, v, None)
        errors[name] = float((actual[0].float() - expected).abs().max())
        if not torch.allclose(actual[0].float(), expected, atol=.02, rtol=.02):
            raise ValueError(f'{name}: dense output differs from FP32 reference')

    sources = [Path(__file__), Path(__file__).with_name('bench_twilight_group_fi.py'),
               Path(__file__).with_name('bench_twilight_upstream_six.py'), author_script, bench_path]
    meta = {'length': args.length, 'batch': args.batch, 'seed': args.seed,
            'candidate_budget_per_head': 8192, 'twilight_profile_head_counts': budgets,
            'group_shared_counts': shared.sum(-1).tolist(),
            'group_union_counts': {name: mask.view(8, 4, args.length).any(1).sum(-1).tolist()
                                   for name, mask in masks.items()},
            'warmups': args.warmups, 'repeats_per_round': args.repeats, 'rounds': args.rounds,
            'input': 'synthetic shared physical KV and identical support masks across batch; Q varies',
            'support': 'random per-head Quest 8192; scaled author layer4 Twilight counts; synthetic logits GAS p=0.8 plus <=16 exceptions; Less-style random group shared support with per-group Twilight union sizes',
            'timing': 'author torch.utils.benchmark helper; fixed plans; excludes selection, estimation, top-p, GAS refinement, packing and model computation',
            'backend': 'Quest uses per-head FI decode; Twilight, GAS and Less-style use grouped FI paged prefill with exact per-head mask; not native Quest, Twilight or LessIsMore kernel',
            'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'flashinfer': importlib.metadata.version('flashinfer-python'),
            'source_sha256': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (args.output_dir / 'config.json').write_text(json.dumps(meta, indent=2) + '\n')
    times = {name: [] for name in calls}
    order = list(calls)
    rng = random.Random(args.seed)
    for round_id in range(args.rounds):
        rng.shuffle(order)
        for name in order:
            _, measurement = bench.benchmark_forward_with_warmup(calls[name], warmups=args.warmups, repeats=args.repeats)
            times[name].append(measurement.mean * 1e6)
        print(f'finished round {round_id + 1}/{args.rounds}', flush=True)
    rows = []
    for name in calls:
        method = 'quest' if name == 'quest_per_head_fi' else name.removesuffix('_grouped_fi')
        mask = masks.get(method)
        rows.append({'method': name, 'length': args.length, 'batch': args.batch,
                     'median_us': statistics.median(times[name]), 'round_means_us': json.dumps(times[name]),
                     'tokens_per_head_mean': float(mask.sum(-1).float().mean()) if mask is not None else args.length,
                     'union_per_group_mean': float(mask.view(8, 4, args.length).any(1).sum(-1).float().mean()) if mask is not None else args.length,
                     'max_abs_error': errors[method]})
    with (args.output_dir / 'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['method']}: {row['median_us']:.2f} us; head={row['tokens_per_head_mean']:.2f}; union={row['union_per_group_mean']:.2f}")
    print(f'COMPLETE {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
