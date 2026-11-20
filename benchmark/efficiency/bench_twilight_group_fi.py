#!/usr/bin/env python3
"""Exact FI GQA-shared plus per-head-exception replay on the upstream synthetic fixture."""
import argparse
import csv
import hashlib
import importlib.util
import importlib.metadata
import json
import math
import random
import statistics
from pathlib import Path

import flashinfer
import torch

from bench_twilight_upstream_six import author_budgets, fi_sparse, reference, synthetic_masks


def group_fi(q, k, v, shared):
    batch, heads, dim = q.shape
    length = k.shape[0]
    groups = 8
    rows, positions = shared.to(q.device).unsqueeze(0).expand(batch, -1, -1).reshape(batch * groups, length).nonzero(as_tuple=True)
    counts = shared.sum(-1, dtype=torch.int32).repeat(batch).to(q.device)
    ptr = torch.nn.functional.pad(counts.cumsum(0, dtype=torch.int32), (1, 0))
    ids = (positions * groups + rows % groups).int()
    last = (counts > 0).to(torch.int32)
    pages = (k.contiguous().view(-1, 1, 1, dim), v.contiguous().view(-1, 1, 1, dim))
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, 'NHD', use_tensor_cores=False)
    wrapper.plan(ptr, ids, last, 4, 1, dim, 1, pos_encoding_mode='NONE',
                 q_data_type=q.dtype, kv_data_type=k.dtype, sm_scale=1 / math.sqrt(dim))
    return lambda: wrapper.run(q.view(-1, 4, dim), pages, return_lse=True), wrapper, pages


def exception_fi(q, k, v, exceptions):
    batch, heads, dim = q.shape
    length = k.shape[0]
    rows, positions = exceptions.to(q.device).unsqueeze(0).expand(batch, -1, -1).reshape(batch * heads, length).nonzero(as_tuple=True)
    counts = exceptions.sum(-1, dtype=torch.int32).repeat(batch).to(q.device)
    ptr = torch.nn.functional.pad(counts.cumsum(0, dtype=torch.int32), (1, 0))
    ids = (positions * 8 + (rows % heads) // 4).int()
    last = (counts > 0).to(torch.int32)
    pages = (k.contiguous().view(-1, 1, 1, dim), v.contiguous().view(-1, 1, 1, dim))
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, 'NHD', use_tensor_cores=False)
    wrapper.plan(ptr, ids, last, 1, 1, dim, 1, pos_encoding_mode='NONE',
                 q_data_type=q.dtype, kv_data_type=k.dtype, sm_scale=1 / math.sqrt(dim))
    return lambda: wrapper.run(q.view(-1, 1, dim), pages, return_lse=True), wrapper, pages


def grouped_mask_fi(q, k, v, mask, use_gpu_metadata=False):
    """Four query positions share each group's union pages, with exact head masks."""
    batch, heads, dim = q.shape
    length = k.shape[0]
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).expand(batch, -1, -1)
    if mask.shape != (batch, heads, length):
        raise ValueError('mask must be [heads,length] or [batch,heads,length]')
    if use_gpu_metadata:
        grouped = mask.to(q.device).reshape(batch * 8, 4, length)
        union = grouped.any(1)
        rows, tokens = union.nonzero(as_tuple=True)
        counts = union.sum(-1, dtype=torch.int32)
        ids = (tokens * 8 + rows % 8).to(torch.int32)
        custom_mask = grouped[union.unsqueeze(1).expand(-1, 4, -1)].contiguous()
    else:
        grouped = mask.cpu().reshape(batch, 8, 4, length)
        group_ids, group_masks = [], []
        for b in range(batch):
            for group in range(8):
                tokens = grouped[b, group].any(0).nonzero().flatten()
                group_ids.append(tokens * 8 + group)
                group_masks.append(grouped[b, group, :, tokens].reshape(-1))
        counts = torch.tensor([x.numel() for x in group_ids], dtype=torch.int32).to(q.device)
        ids = torch.cat(group_ids).to(device=q.device, dtype=torch.int32)
        custom_mask = torch.cat(group_masks).to(device=q.device)
    ptr = torch.nn.functional.pad(counts.cumsum(0, dtype=torch.int32), (1, 0))
    qptr = torch.arange(batch * 8 + 1, device=q.device, dtype=torch.int32) * 4
    last = (counts > 0).to(torch.int32)
    pages = (k.contiguous().view(-1, 1, 1, dim), v.contiguous().view(-1, 1, 1, dim))
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, 'NHD')
    wrapper.plan(qptr, ptr, ids, last, 1, 1, dim, 1, custom_mask=custom_mask,
                 causal=False, pos_encoding_mode='NONE', q_data_type=q.dtype,
                 kv_data_type=k.dtype, sm_scale=1 / math.sqrt(dim))
    return lambda: wrapper.run(q.view(-1, 1, dim), pages).view_as(q), wrapper, pages


def shared_batch_grouped_fi(q, k, v, mask):
    """Synthetic-only: identical support and KV are shared across all batch rows."""
    batch, heads, dim = q.shape
    length = k.shape[0]
    if mask.shape != (heads, length):
        raise ValueError('shared batch path requires one identical [heads,length] mask')
    grouped = mask.cpu().reshape(8, 4, length)
    group_ids, group_masks = [], []
    for group in range(8):
        tokens = grouped[group].any(0).nonzero().flatten()
        group_ids.append(tokens * 8 + group)
        group_masks.append(grouped[group, :, tokens].repeat(batch, 1).reshape(-1))
    counts = torch.tensor([x.numel() for x in group_ids], dtype=torch.int32, device=q.device)
    ptr = torch.nn.functional.pad(counts.cumsum(0, dtype=torch.int32), (1, 0))
    ids = torch.cat(group_ids).to(device=q.device, dtype=torch.int32)
    custom_mask = torch.cat(group_masks).to(device=q.device)
    qptr = torch.arange(9, device=q.device, dtype=torch.int32) * (batch * 4)
    last = (counts > 0).to(torch.int32)
    pages = (k.contiguous().view(-1, 1, 1, dim), v.contiguous().view(-1, 1, 1, dim))
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, 'NHD')
    wrapper.plan(qptr, ptr, ids, last, 1, 1, dim, 1, custom_mask=custom_mask,
                 causal=False, pos_encoding_mode='NONE', q_data_type=q.dtype,
                 kv_data_type=k.dtype, sm_scale=1 / math.sqrt(dim))
    def run():
        grouped_q = q.view(batch, 8, 4, dim).permute(1, 0, 2, 3).contiguous().view(-1, 1, dim)
        grouped_out = wrapper.run(grouped_q, pages).view(8, batch, 4, dim)
        return grouped_out.permute(1, 0, 2, 3).contiguous().view_as(q)
    return run, wrapper, pages


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
    p.add_argument('--native-shared-pool', action='store_true')
    p.add_argument('--triton-shared-pool', action='store_true')
    p.add_argument('--coop-shared-pool', action='store_true')
    p.add_argument('--breakdown', action='store_true', help='also time disconnected group, exception, and merge components')
    p.add_argument('--grouped-fi-mask', action='store_true')
    p.add_argument('--batch-shared-fi', action='store_true')
    p.add_argument('--variants', nargs='+', default=['s4_p16_c64', 's8_p8_c128', 's8_p1_c2048'])
    args = p.parse_args()
    supported = {f'{prefix}s{seg}_p{parts}_c{chunk}'
                 for seg, parts, chunk in ((4, 8, 128), (8, 8, 128), (4, 16, 64), (4, 4, 256), (4, 1, 2048), (8, 1, 2048), (8, 16, 64), (8, 32, 32))
                 for prefix in ('', 'tiled_')}
    if not set(args.variants) <= supported:
        p.error(f'unknown variants: {sorted(set(args.variants) - supported)}')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f'refusing to overwrite {args.output_dir}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('author_bench_utils', args.upstream / 'benchmark/efficiency/bench_utils.py')
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    native = None
    cooperative = None
    if args.native_shared_pool:
        from torch.utils.cpp_extension import load
        source = Path(__file__).resolve().parents[2] / 'csrc/src/gas_group_warp_shared_pool.cu'
        native = load(name='gas_group_warp_shared_pool', sources=[str(source)], extra_cuda_cflags=['-O3'], verbose=False)
    if args.coop_shared_pool:
        from torch.utils.cpp_extension import load
        source = Path(__file__).resolve().parents[2] / 'csrc/src/gas_cooperative_shared_pool.cu'
        cooperative = load(name='gas_cooperative_shared_pool', sources=[str(source)], extra_cuda_cflags=['-O3'], verbose=False)
    budgets = author_budgets(args.upstream / 'benchmark/efficiency/bench_breakdown.py', 256)
    root = Path(__file__).resolve().parents[2]
    sources = [Path(__file__).resolve(), Path(__file__).with_name('bench_twilight_upstream_six.py'),
               args.upstream / 'benchmark/efficiency/bench_utils.py', args.upstream / 'benchmark/efficiency/bench_breakdown.py']
    if args.native_shared_pool:
        sources.append(root / 'csrc/src/gas_group_warp_shared_pool.cu')
    if args.coop_shared_pool:
        sources.append(root / 'csrc/src/gas_cooperative_shared_pool.cu')
    if args.triton_shared_pool:
        sources.append(Path(__file__).with_name('gas_shared_triton.py'))
    cfg = {'input': 'identical to bench_twilight_upstream_six synthetic shared-KV fixture',
           'timing': 'upstream torch.utils.benchmark helper; planning, compaction, selection excluded',
           'methods': 'per-head FI Twilight and GAS; group FI plus per-head FI exceptions plus merge_state',
           'warmups': args.warmups, 'repeats': args.repeats, 'rounds': args.rounds, 'variants': args.variants,
           'grouped_fi_mask': args.grouped_fi_mask, 'native_shared_pool': args.native_shared_pool,
           'batch_shared_fi': args.batch_shared_fi,
           'coop_shared_pool': args.coop_shared_pool, 'triton_shared_pool': args.triton_shared_pool,
           'lengths': args.lengths, 'batch_sizes': args.batch_sizes, 'seed': args.seed,
           'device': torch.cuda.get_device_name(), 'torch': torch.__version__,
           'cuda': torch.version.cuda, 'flashinfer': importlib.metadata.version('flashinfer-python'),
           'source_sha256': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (args.output_dir / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    records = []
    for length in args.lengths:
        masks, less_k, shared = synthetic_masks(length, int(length * .08), budgets, args.seed + length)
        exceptions = masks['gas'] & ~shared.repeat_interleave(4, 0)
        torch.manual_seed(args.seed + length)
        kv = torch.randn(length, 2, 1, 8, 128, dtype=torch.float16, device='cuda')
        k, v = kv[:, 0, 0].contiguous(), kv[:, 1, 0].contiguous()
        for batch in args.batch_sizes:
            torch.manual_seed(args.seed + length + batch)
            q = torch.randn(batch, 32, 128, dtype=torch.float16, device='cuda')
            twi, twi_w, twi_pages = fi_sparse(q, k, v, masks['twilight'])
            gas, gas_w, gas_pages = fi_sparse(q, k, v, masks['gas'])
            group, group_w, group_pages = group_fi(q, k, v, shared)
            ex, ex_w, ex_pages = exception_fi(q, k, v, exceptions)
            def group_merge():
                out_g, lse_g = group()
                out_e, lse_e = ex()
                merged, _ = flashinfer.merge_state(out_g, lse_g, out_e.view(-1, 4, 128), lse_e.view(-1, 4))
                return merged.view(batch, 32, 128)
            expected = gas()
            merged = group_merge()
            if not torch.allclose(merged, expected, atol=.02, rtol=.02):
                diff = (merged.float() - expected.float()).abs().max().item()
                raise ValueError(f'group FI output differs from full GAS: L={length} B={batch}, max={diff}')
            fp32 = reference(q, k, v, masks['gas'])
            if not torch.allclose(merged[0].float(), fp32, atol=.02, rtol=.02):
                raise ValueError(f'group FI output differs from FP32 reference: L={length} B={batch}')
            funcs = {'twilight_fi': twi, 'gas_flat_fi': gas, 'gas_group_merge_fi': group_merge}
            errors = {'gas_group_merge_fi': (merged.float() - expected.float()).abs().max().item(), 'gas_flat_fi': 0.0}
            if args.grouped_fi_mask:
                grouped_twi, grouped_twi_w, grouped_twi_pages = grouped_mask_fi(q, k, v, masks['twilight'])
                grouped_gas, grouped_gas_w, grouped_gas_pages = grouped_mask_fi(q, k, v, masks['gas'])
                for name, function, target in (('twilight_grouped_mask_fi', grouped_twi, twi()),
                                               ('gas_grouped_mask_fi', grouped_gas, expected)):
                    out = function()
                    errors[name] = (out.float() - target.float()).abs().max().item()
                    if not torch.allclose(out, target, atol=.003, rtol=.003):
                        raise ValueError(f'{name} does not match per-head reference: {errors[name]}')
                    funcs[name] = function
            if args.batch_shared_fi:
                batch_twi, batch_twi_w, batch_twi_pages = shared_batch_grouped_fi(q, k, v, masks['twilight'])
                batch_gas, batch_gas_w, batch_gas_pages = shared_batch_grouped_fi(q, k, v, masks['gas'])
                for name, function, target in (('twilight_batch_shared_fi', batch_twi, twi()),
                                               ('gas_batch_shared_fi', batch_gas, expected)):
                    out = function()
                    errors[name] = (out.float() - target.float()).abs().max().item()
                    if not torch.allclose(out, target, atol=.003, rtol=.003):
                        raise ValueError(f'{name} does not match per-head reference: {errors[name]}')
                    funcs[name] = function
            if args.breakdown:
                group_out, group_lse = group()
                ex_out, ex_lse = ex()
                funcs['diagnostic_group_fi'] = lambda: group()[0]
                funcs['diagnostic_exception_fi'] = lambda: ex()[0]
                funcs['diagnostic_merge'] = lambda: flashinfer.merge_state(group_out, group_lse, ex_out.view(-1, 4, 128), ex_lse.view(-1, 4))[0]
            if native is not None or cooperative is not None or args.triton_shared_pool:
                from bench_e4_kernel_evidence import padded_tensor
                group_lists = [row.nonzero().flatten().tolist() for row in shared]
                ex_lists = [row.nonzero().flatten().tolist() for row in exceptions]
                group_ids = padded_tensor(group_lists * batch, (batch, 8), q.device)
                ex_ids = padded_tensor(ex_lists * batch, (batch, 32), q.device, width=16)
                sc = shared.sum(-1, dtype=torch.int32).repeat(batch, 1).to(q.device)
                ec = exceptions.sum(-1, dtype=torch.int32).repeat(batch, 1).to(q.device)
                if native is not None:
                    for seg, parts, chunk in ((4, 8, 128), (8, 8, 128), (4, 16, 64), (4, 4, 256), (4, 1, 2048), (8, 1, 2048), (8, 16, 64), (8, 32, 32)):
                        key = f's{seg}_p{parts}_c{chunk}'
                        if key in args.variants:
                            funcs[f'gas_native_{key}'] = lambda seg=seg, parts=parts, chunk=chunk: native.decode(q, k, v, group_ids, ex_ids, sc, ec, seg, parts, chunk)
                        if f'tiled_{key}' in args.variants:
                            funcs[f'gas_native_tiled_{key}'] = lambda seg=seg, parts=parts, chunk=chunk: native.decode_tiled(q, k, v, group_ids, ex_ids, sc, ec, seg, parts, chunk)
                if args.triton_shared_pool:
                    from gas_shared_triton import decode as triton_decode
                    for parts, block in ((16, 128), (16, 256), (32, 64)):
                        if (group_ids.shape[-1] + parts - 1) // parts <= block:
                            funcs[f'gas_triton_p{parts}_b{block}'] = lambda parts=parts, block=block: triton_decode(q, k, v, group_ids, ex_ids, sc, ec, parts, block)
                if cooperative is not None:
                    for seg, parts, chunk in ((4, 16, 64), (8, 16, 64), (8, 8, 128), (16, 8, 128)):
                        funcs[f'gas_coop_s{seg}_p{parts}_c{chunk}'] = lambda seg=seg, parts=parts, chunk=chunk: cooperative.decode(q, k, v, group_ids, ex_ids, sc, ec, seg, parts, chunk)
                for name, function in funcs.items():
                    if not name.startswith(('gas_native_', 'gas_triton_', 'gas_coop_')):
                        continue
                    native_out = function()
                    errors[name] = (native_out.float()-expected.float()).abs().max().item()
                    if not torch.allclose(native_out, expected, atol=.003, rtol=.003):
                        raise ValueError(f'{name} differs from flat GAS: L={length} B={batch}, max={(native_out.float()-expected.float()).abs().max().item()}')
            samples = {name: [] for name in funcs}
            for rnd in range(args.rounds):
                order = list(funcs)
                random.Random(f'{length}:{batch}:{rnd}:{args.seed}').shuffle(order)
                for name in order:
                    _, timing = bench.benchmark_forward_with_warmup(funcs[name], warmups=args.warmups, repeats=args.repeats)
                    samples[name].append(timing.mean * 1e6)
            for name in funcs:
                records.append({'length': length, 'batch': batch, 'method': name,
                                'median_us': statistics.median(samples[name]), 'round_means_us': json.dumps(samples[name]),
                                'comparison_target': 'twilight_fi' if name in ('twilight_grouped_mask_fi', 'twilight_batch_shared_fi') else ('gas_flat_fi' if name in errors else ''),
                                'max_error_vs_target': errors.get(name),
                                'max_error_vs_fp32_gas_first': (merged[0].float() - fp32).abs().max().item(),
                                'shared_counts': json.dumps(shared.sum(-1).tolist()),
                                'exception_counts': json.dumps(exceptions.sum(-1).tolist())})
            with (args.output_dir / 'results.csv').open('w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=records[0].keys())
                writer.writeheader()
                writer.writerows(records)
            print(f'L={length} B={batch} ' + ' '.join(f'{name}={statistics.median(samples[name]):.3f}us' for name in funcs), flush=True)
            del q, twi, gas, group, ex, group_merge, funcs, merged, expected
            del twi_w, gas_w, group_w, ex_w, twi_pages, gas_pages, group_pages, ex_pages
            if args.grouped_fi_mask:
                del grouped_twi, grouped_gas, grouped_twi_w, grouped_gas_w, grouped_twi_pages, grouped_gas_pages
            if args.batch_shared_fi:
                del batch_twi, batch_gas, batch_twi_w, batch_gas_w, batch_twi_pages, batch_gas_pages
            if args.breakdown:
                del group_out, group_lse, ex_out, ex_lse
            if native is not None or cooperative is not None or args.triton_shared_pool:
                del group_ids, ex_ids, sc, ec, native_out
            torch.cuda.empty_cache()
        del kv, k, v
    print(f'COMPLETE {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
