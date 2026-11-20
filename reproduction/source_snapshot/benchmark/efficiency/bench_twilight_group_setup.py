#!/usr/bin/env python3
"""Diagnostic wall time for CPU-mask-to-FI metadata/plan; not selection or decode."""
import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch

from bench_twilight_group_fi import grouped_mask_fi
from bench_twilight_upstream_six import author_budgets, fi_sparse, synthetic_masks


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--length', type=int, default=16384)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--warmups', type=int, default=2)
    p.add_argument('--repeats', type=int, default=5)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    budgets = author_budgets(args.upstream / 'benchmark/efficiency/bench_breakdown.py', 256)
    masks, _, _ = synthetic_masks(args.length, int(args.length * .08), budgets, args.seed + args.length)
    torch.manual_seed(args.seed + args.length)
    k = torch.randn(args.length, 8, 128, dtype=torch.float16, device='cuda')
    v = torch.randn_like(k)
    q = torch.randn(args.batch, 32, 128, dtype=torch.float16, device='cuda')
    calls = {f'{method}_{layout}': (lambda method=method, factory=factory: factory(q, k, v, masks[method]))
             for method in ('twilight', 'gas')
             for layout, factory in (('flat', fi_sparse), ('grouped_cpu', grouped_mask_fi),
                                     ('grouped_gpu', lambda q, k, v, m: grouped_mask_fi(q, k, v, m, True)))}
    samples = {name: [] for name in calls}
    for iteration in range(args.warmups + args.repeats):
        order = list(calls)
        random.Random(f'{args.seed}:{iteration}').shuffle(order)
        for name in order:
            torch.cuda.synchronize()
            start = time.perf_counter()
            created = calls[name]()
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            if iteration >= args.warmups:
                samples[name].append(elapsed)
            del created
    result = {'scope': 'synchronized host wall time from CPU boolean mask through FI metadata and plan; excludes selector, pruning and attention',
              'warning': 'CPU mask input; actual GPU mask transfer and complete decode are not measured',
              'length': args.length, 'batch': args.batch, 'seed': args.seed,
              'warmups': args.warmups, 'repeats': args.repeats,
              'samples_ms': samples, 'median_ms': {k: statistics.median(v) for k, v in samples.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['median_ms'], indent=2))


if __name__ == '__main__':
    main()
