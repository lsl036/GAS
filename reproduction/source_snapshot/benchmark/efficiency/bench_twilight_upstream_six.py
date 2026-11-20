#!/usr/bin/env python3
"""Author-style synthetic GQA attention replay; sparse supports use one FI adapter."""
import argparse
import ast
import csv
import importlib.util
import json
import math
import statistics
from pathlib import Path

import flashinfer
import torch
from flash_attn import flash_attn_with_kvcache


def author_budgets(path, avg):
    tree = ast.parse(path.read_text())
    values = next(ast.literal_eval(n.value) for n in tree.body
                  if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'LAYER4_HEAD_BUDGETS' for t in n.targets))
    return [max(1, round(x * avg / statistics.fmean(values))) for x in values]


def top_p_mask(scores, p):
    probabilities = scores.softmax(-1)
    ordered, ids = probabilities.sort(-1, descending=True)
    chosen = (ordered.cumsum(-1) - ordered) < p
    output = torch.zeros_like(chosen)
    return output.scatter(-1, ids, chosen)


def synthetic_masks(length, budget, head_budgets, seed):
    generator = torch.Generator().manual_seed(seed)
    if max(head_budgets) > budget:
        raise ValueError('author profile requires more tokens than Quest candidate budget')
    quest = torch.zeros(32, length, dtype=torch.bool)
    twilight = torch.zeros_like(quest)
    scores = torch.full((32, length), torch.finfo(torch.float32).min)
    for head, count in enumerate(head_budgets):
        candidates = torch.randperm(length, generator=generator)[:budget]
        selected = candidates[:count]
        quest[head, candidates] = True
        twilight[head, selected] = True
        scores[head, selected] = torch.randn(count, generator=generator)
    group_mask = twilight.view(8, 4, length)
    union = group_mask.any(1)
    group_scores = scores.view(8, 4, length).max(1).values.masked_fill(~union, torch.finfo(torch.float32).min)
    group_shared = top_p_mask(group_scores, 0.80) & union
    shared = group_shared.repeat_interleave(4, 0)
    residual = twilight & ~shared
    exception_indices = scores.masked_fill(~residual, torch.finfo(torch.float32).min).topk(16, dim=-1).indices
    exception_mask = torch.zeros_like(residual).scatter(-1, exception_indices, True) & residual
    gas = shared | exception_mask
    less_k = round(union.sum(-1).float().mean().item())
    less_indices = torch.randperm(length, generator=generator)[:less_k]
    less = torch.zeros_like(quest)
    less[:, less_indices] = True
    assert not (twilight & ~quest).any() and not (gas & ~union.repeat_interleave(4, 0)).any()
    assert int((gas & ~shared).sum(-1).max()) <= 16
    return {'quest': quest, 'twilight': twilight, 'gas': gas, 'lessismore_style': less}, less_k, group_shared


def fi_sparse(q, k, v, mask):
    batch, heads, dim = q.shape
    length = k.shape[0]
    mask = mask.to(device=q.device).expand(batch, -1, -1).reshape(batch * heads, length)
    rows, positions = mask.nonzero(as_tuple=True)
    counts = mask.sum(-1, dtype=torch.int32)
    ptr = torch.nn.functional.pad(counts.cumsum(0, dtype=torch.int32), (1, 0))
    ids = (positions * 8 + (rows % heads) // 4).int()
    last = torch.ones(batch * heads, dtype=torch.int32, device=q.device)
    pages = (k.contiguous().view(-1, 1, 1, dim), v.contiguous().view(-1, 1, 1, dim))
    work = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(work, 'NHD', use_tensor_cores=False)
    wrapper.plan(ptr, ids, last, 1, 1, dim, 1, pos_encoding_mode='NONE',
                 q_data_type=q.dtype, kv_data_type=k.dtype, sm_scale=1 / math.sqrt(dim))
    return lambda: wrapper.run(q.view(-1, 1, dim), pages).view_as(q), wrapper, pages


def fi_dense(q, kv):
    batch, heads, dim = q.shape
    length = kv.shape[0]
    ptr = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * length
    ids = torch.arange(length, device=q.device, dtype=torch.int32).repeat(batch)
    last = torch.ones(batch, device=q.device, dtype=torch.int32)
    work = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(work, 'NHD', use_tensor_cores=False)
    wrapper.plan(ptr, ids, last, heads, 8, dim, 1, pos_encoding_mode='NONE',
                 q_data_type=q.dtype, kv_data_type=kv.dtype, sm_scale=1 / math.sqrt(dim))
    return lambda: wrapper.run(q, kv), wrapper


def reference(q, k, v, mask):
    repeated_k = k.float().transpose(0, 1).repeat_interleave(4, 0)
    repeated_v = v.float().transpose(0, 1).repeat_interleave(4, 0)
    logits = torch.matmul(q[0].float().unsqueeze(1), repeated_k.transpose(-1, -2)).squeeze(1) / math.sqrt(q.shape[-1])
    if mask is not None:
        logits.masked_fill_(~mask.to(q.device), float('-inf'))
    return torch.matmul(logits.softmax(-1).unsqueeze(1), repeated_v).squeeze(1)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--lengths', type=int, nargs='+', default=[16384, 32768])
    p.add_argument('--batch-sizes', type=int, nargs='+', default=[32, 64, 96, 128])
    p.add_argument('--warmups', type=int, default=10)
    p.add_argument('--repeats', type=int, default=100)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--gas-native-e6', action='store_true', help='also benchmark old E6 fused GAS with batch-replicated physical KV')
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f'refusing to overwrite {args.output_dir}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.upstream / 'benchmark/efficiency/bench_utils.py'
    spec = importlib.util.spec_from_file_location('author_bench_utils', path)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    budgets = author_budgets(args.upstream / 'benchmark/efficiency/bench_breakdown.py', 256)
    cfg = {'input': 'synthetic shared KV pool, same Q/K/V across methods; not Qasper',
           'supports': 'Quest random 8% per head; Twilight author layer4 per-head counts sampled from Quest; GAS group_max p=.8 and 16 exceptions on synthetic logits; Less-style shared K matched to Twilight union',
           'backend': 'FA2 dense; FI dense and same FI page-1 flattened per-head adapter for all sparse methods',
           'excluded': 'selector, top-p, refinement, compact metadata, FI planning, model and KV updates',
           'upstream': str(args.upstream.resolve()), 'warmups': args.warmups, 'repeats': args.repeats,
           'rounds': args.rounds, 'lengths': args.lengths, 'batch_sizes': args.batch_sizes, 'seed': args.seed,
           'gas_native_e6': 'old E6 fused kernel, separate physically replicated batch KV; incomparable physical cache layout' if args.gas_native_e6 else 'not requested',
           'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cuda': torch.version.cuda}
    (args.output_dir / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    rows = []
    for length in args.lengths:
        mask, less_k, shared_mask = synthetic_masks(length, int(length * .08), budgets, args.seed + length)
        torch.manual_seed(args.seed + length)
        kv = torch.randn(length, 2, 1, 8, 128, dtype=torch.float16, device='cuda')
        k, v = kv[:, 0, 0].contiguous(), kv[:, 1, 0].contiguous()
        for batch in args.batch_sizes:
            torch.manual_seed(args.seed + length + batch)
            q = torch.randn(batch, 32, 128, dtype=torch.float16, device='cuda')
            dense, dense_wrapper = fi_dense(q, kv)
            kcache = k.unsqueeze(0).expand(batch, -1, -1, -1)
            vcache = v.unsqueeze(0).expand(batch, -1, -1, -1)
            calls = {'fa2': lambda: flash_attn_with_kvcache(q.unsqueeze(1), kcache, vcache,
                                                            causal=False, softmax_scale=1 / math.sqrt(128)).squeeze(1),
                     'flashinfer': dense}
            keepalive = [dense_wrapper, kcache, vcache]
            for name, support in mask.items():
                calls[name], wrapper, pages = fi_sparse(q, k, v, support)
                keepalive.extend([wrapper, pages])
            if args.gas_native_e6:
                from bench_e4_kernel_evidence import padded_tensor
                from bench_e6_kernel_baselines import get_gas_module
                module = get_gas_module()
                group_lists = [row.nonzero().flatten().tolist() for row in shared_mask]
                exceptions = mask['gas'] & ~shared_mask.repeat_interleave(4, 0)
                ex_lists = [row.nonzero().flatten().tolist() for row in exceptions]
                group_ids = padded_tensor(group_lists * batch, (batch, 8), q.device)
                ex_ids = padded_tensor(ex_lists * batch, (batch, 32), q.device, width=16)
                native_k = k.transpose(0, 1).unsqueeze(0).repeat(batch, 1, 1, 1).contiguous()
                native_v = v.transpose(0, 1).unsqueeze(0).repeat(batch, 1, 1, 1).contiguous()
                calls['gas_native_e6'] = lambda: module.batched_gqa_group_exception_decode(q, native_k, native_v, group_ids, ex_ids)
                keepalive.extend([group_ids, ex_ids, native_k, native_v])
            for name, call in calls.items():
                support = mask['gas'] if name == 'gas_native_e6' else mask.get(name)
                expected = reference(q, k, v, support)
                actual = call()[0].float()
                max_error = (actual - expected).abs().max().item()
                if not torch.allclose(actual, expected, atol=.02, rtol=.02):
                    raise ValueError(f'correctness failed: {length} {batch} {name}, max {max_error}')
                times = []
                for _ in range(args.rounds):
                    _, timing = bench.benchmark_forward_with_warmup(call, warmups=args.warmups, repeats=args.repeats)
                    times.append(timing.mean * 1e6)
                rows.append({'length': length, 'batch': batch, 'method': name,
                             'median_us': statistics.median(times), 'round_means_us': json.dumps(times),
                             'per_head_mean': float(support.sum(-1).float().mean()) if support is not None else length,
                             'group_union_mean': float(support.view(8, 4, length).any(1).sum(-1).float().mean()) if support is not None else length,
                             'less_k': less_k, 'max_abs_error': max_error})
                print(f'L={length} B={batch} {name}: {rows[-1]["median_us"]:.3f} us', flush=True)
            with (args.output_dir / 'results.csv').open('w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            del q, calls, keepalive, dense_wrapper
            if args.gas_native_e6:
                del native_k, native_v, group_ids, ex_ids
            torch.cuda.empty_cache()
        del kv, k, v
    print(f'COMPLETE {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
