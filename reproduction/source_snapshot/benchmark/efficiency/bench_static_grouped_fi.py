#!/usr/bin/env python3
"""Diagnostic: fixed-capacity reusable FI plan including per-step mask updates."""
import argparse
import hashlib
import importlib.metadata
import json
import random
import statistics
import time
from pathlib import Path

import torch

from bench_twilight_upstream_six import author_budgets, fi_sparse, synthetic_masks
from gas_grouped_static_fi import StaticGroupedAttention


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--length', type=int, default=16384)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--warmups', type=int, default=2)
    p.add_argument('--repeats', type=int, default=10)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    budgets = author_budgets(args.upstream / 'benchmark/efficiency/bench_breakdown.py', 256)
    masks, _, _ = synthetic_masks(args.length, int(args.length * .08), budgets, args.seed + args.length)
    torch.manual_seed(args.seed + args.length)
    kv = torch.randn(args.length, 2, 1, 8, 128, dtype=torch.float16, device='cuda')
    k, v = kv[:, 0, 0].contiguous(), kv[:, 1, 0].contiguous()
    torch.manual_seed(args.seed + args.length + args.batch)
    q = torch.randn(args.batch, 32, 128, dtype=torch.float16, device='cuda')
    gpu_masks = {name: masks[name].to('cuda').unsqueeze(0).expand(args.batch, -1, -1).contiguous()
                 for name in ('twilight', 'gas')}
    plans, plan_times = {}, {}
    for name in gpu_masks:
        torch.cuda.synchronize()
        start = time.perf_counter()
        plans[name] = StaticGroupedAttention(q, k, v, gpu_masks[name])
        torch.cuda.synchronize()
        plan_times[name] = (time.perf_counter() - start) * 1000
        plans[name].update(gpu_masks[name])
        flat, wrapper, pages = fi_sparse(q, k, v, gpu_masks[name])
        if not torch.allclose(plans[name].run(q), flat(), atol=.003, rtol=.003):
            raise ValueError(f'{name}: static plan output mismatch')
        plans[name].update_triton(gpu_masks[name])
        if not torch.allclose(plans[name].run(q), flat(), atol=.003, rtol=.003):
            raise ValueError(f'{name}: Triton metadata output mismatch')
    calls = {}
    for name in plans:
        calls[f'{name}_update'] = lambda name=name: plans[name].update(gpu_masks[name])
        calls[f'{name}_attention'] = lambda name=name: plans[name].run(q)
        calls[f'{name}_update_triton'] = lambda name=name: plans[name].update_triton(gpu_masks[name])
        def connected(name=name):
            plans[name].update(gpu_masks[name])
            return plans[name].run(q)
        calls[f'{name}_update_attention'] = connected
        def triton_connected(name=name):
            plans[name].update_triton(gpu_masks[name])
            return plans[name].run(q)
        calls[f'{name}_triton_update_attention'] = triton_connected
    samples = {name: [] for name in calls}
    for i in range(args.warmups + args.repeats):
        order = list(calls)
        random.Random(f'{args.seed}:{i}').shuffle(order)
        for name in order:
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = calls[name]()
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            if i >= args.warmups:
                samples[name].append(elapsed)
            del output
    sources = [Path(__file__).resolve(), Path(__file__).with_name('gas_grouped_static_fi.py'),
               Path(__file__).with_name('gas_grouped_compact_triton.py'), Path(__file__).with_name('bench_twilight_upstream_six.py'),
               args.upstream / 'benchmark/efficiency/bench_breakdown.py']
    result = {'scope': 'GPU mask already computed; synchronized host wall time for static capacity plan updates and attention; selection/refinement and model excluded',
              'warning': 'per-group capacities determined from the test mask; overflow raises and requires replan; .item/nonzero sync means update is not graph capturable',
              'device': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
              'flashinfer': importlib.metadata.version('flashinfer-python'),
              'source_sha256': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              'length': args.length, 'batch': args.batch, 'seed': args.seed, 'warmups': args.warmups,
              'repeats': args.repeats, 'one_time_plan_ms': plan_times,
              'capacity_mean': {name: float(plan.capacities.float().mean()) for name, plan in plans.items()},
              'median_ms': {name: statistics.median(values) for name, values in samples.items()}, 'samples_ms': samples}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['median_ms'], indent=2))


if __name__ == '__main__':
    main()
