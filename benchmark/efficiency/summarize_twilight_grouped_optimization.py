#!/usr/bin/env python3
"""Summarize paired synthetic grouped-FI kernel and fixed-plan experiments."""
import argparse
import csv
import json
import statistics
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    methods = ('twilight_fi', 'gas_flat_fi', 'twilight_grouped_mask_fi',
               'gas_grouped_mask_fi', 'gas_native_s4_p16_c64', 'gas_group_merge_fi')
    for seed in (42, 2026):
        directory = args.results_root / f'twilight_grouped_audited_seed{seed}'
        cfg = json.loads((directory / 'config.json').read_text())
        if cfg['seed'] != seed or not cfg.get('source_sha256'):
            raise ValueError(f'invalid kernel provenance: {directory}')
        with (directory / 'results.csv').open(newline='') as f:
            raw = list(csv.DictReader(f))
        for length in (16384, 32768):
            for batch in (32, 64, 96, 128):
                rows = {r['method']: r for r in raw if int(r['length']) == length and int(r['batch']) == batch}
                if set(methods) - rows.keys():
                    raise ValueError(f'missing methods: seed={seed} length={length} batch={batch}')
                static_path = args.results_root / f'twilight_static_grouped_audited_seed{seed}' / f'L{length}_B{batch}.json'
                static = json.loads(static_path.read_text())
                if static['seed'] != seed or static['length'] != length or static['batch'] != batch or not static.get('source_sha256'):
                    raise ValueError(f'invalid static provenance: {static_path}')
                wall = static['median_ms']
                values = {name: float(rows[name]['median_us']) for name in methods}
                records.append({'seed': seed, 'length': length, 'batch': batch,
                    'twilight_flat_us': values['twilight_fi'], 'gas_flat_us': values['gas_flat_fi'],
                    'twilight_grouped_us': values['twilight_grouped_mask_fi'],
                    'gas_grouped_us': values['gas_grouped_mask_fi'],
                    'gas_warp_us': values['gas_native_s4_p16_c64'],
                    'gas_two_pass_us': values['gas_group_merge_fi'],
                    'grouped_speedup': values['twilight_grouped_mask_fi'] / values['gas_grouped_mask_fi'],
                    'gas_vs_flat_twilight': values['twilight_fi'] / values['gas_grouped_mask_fi'],
                    'twilight_static_update_attention_ms': wall['twilight_triton_update_attention'],
                    'gas_static_update_attention_ms': wall['gas_triton_update_attention'],
                    'static_speedup': wall['twilight_triton_update_attention'] / wall['gas_triton_update_attention'],
                    'gas_torch_update_ms': wall['gas_update'], 'gas_triton_update_ms': wall['gas_update_triton'],
                    'gas_update_speedup': wall['gas_update'] / wall['gas_update_triton'],
                    'twilight_capacity_mean': static['capacity_mean']['twilight'],
                    'gas_capacity_mean': static['capacity_mean']['gas'],
                    'twilight_grouped_max_error': float(rows['twilight_grouped_mask_fi']['max_error_vs_target']),
                    'gas_grouped_max_error': float(rows['gas_grouped_mask_fi']['max_error_vs_target']),
                    'gas_warp_max_error': float(rows['gas_native_s4_p16_c64']['max_error_vs_target'])})
    with (args.output_dir / 'summary_per_seed.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    lines = [
        '# GAS Grouped Synthetic Optimization Handoff', '',
        '## Scope', '',
        'A100 40GB, FP16, 32 Q heads / 8 KV heads, head dimension 128. Two synthetic seeds (42 and 2026), 16K/32K and B32/64/96/128. All batch rows share one physical KV pool and the same support mask, although queries differ. This is the upstream-style synthetic fixture, not Qasper or independent serving requests.',
        'Quest supports are random 8% subsets; Twilight uses the upstream layer4 per-head budget profile, not actual top-p=0.85 output. GAS uses synthetic group-max p=0.80 and 16 exceptions. Thus this is neither a quality comparison nor the official Twilight kernel.',
        'Kernel-only timing uses upstream torch.utils.benchmark: each result is the median of five round means, 100 iterations per round and 10 warmups. Planning, selector, estimator, top-p, refinement and metadata construction are excluded. Methods use the same Q/K/V and support masks in each paired comparison.',
        'Grouped FI maps each GQA group to four query positions in a FlashInfer paged-prefill wrapper, with union pages and an exact per-head custom mask. It is the same FI grouped path for Twilight and GAS, but it is not the author\'s native GQA decode kernel. Flat FI is the independent per-head decode adapter. Warp GAS is our custom fused CUDA path on the same shared KV pool.',
        '', '## Kernel Comparison', '',
        'Seed 2026 results, microseconds per invocation:', '',
        '| Length | B | Twilight flat FI | Twilight grouped FI | GAS grouped FI | Grouped speedup | GAS warp CUDA |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for row in (r for r in records if r['seed'] == 2026):
        lines.append(f"| {row['length']//1024}K | {row['batch']} | {row['twilight_flat_us']:.2f} | {row['twilight_grouped_us']:.2f} | {row['gas_grouped_us']:.2f} | {row['grouped_speedup']:.2f}x | {row['gas_warp_us']:.2f} |")
    grouped = [r['grouped_speedup'] for r in records]
    best_flat = [r['gas_vs_flat_twilight'] for r in records]
    lines += ['', f'Across both seeds and eight shapes, GAS grouped is faster than Twilight grouped in {sum(x > 1 for x in grouped)}/16 cases; ratio range {min(grouped):.2f}–{max(grouped):.2f}x.',
              f'However, GAS grouped is faster than the stronger Twilight flat FI adapter in only {sum(x > 1 for x in best_flat)}/16 cases. Grouped-vs-grouped wins must not be described as a general win over the best Twilight implementation.',
              'The warp CUDA prototype wins versus flat Twilight at B32, but is slower at B64–B128. Separate two-pass FI, cooperative CUDA and Triton fused experiments did not improve the larger-batch result; exploratory raw results are archived in benchmark/results_efficiency/twilight_grouped_exploratory_20261113/.',
              '', '## Metadata Plus Attention', '',
              'Fixed per-group capacity is rounded from the observed test mask; overflow is checked and raises instead of silently truncating. FI plan is reused. The wall-time measurement includes GPU-mask compaction, host-synchronized overflow check, bit packing and grouped attention; it excludes mask generation, model work, one-time planning and overflow replans. Updates are not CUDA-graph capturable due to dynamic sizes and host synchronization. Values are medians of 10 synchronized wall-time samples, not end-to-end TPOT.', '',
              '| Length | B | Twilight update + attention ms | GAS update + attention ms | Speedup | GAS metadata update: Torch / Triton |',
              '|---:|---:|---:|---:|---:|---:|']
    for row in (r for r in records if r['seed'] == 2026):
        lines.append(f"| {row['length']//1024}K | {row['batch']} | {row['twilight_static_update_attention_ms']:.3f} | {row['gas_static_update_attention_ms']:.3f} | {row['static_speedup']:.2f}x | {row['gas_torch_update_ms']:.3f} / {row['gas_triton_update_ms']:.3f} ms |")
    static_speed = [r['static_speedup'] for r in records]
    metadata_speed = [r['gas_update_speedup'] for r in records]
    lines += ['', f'The static-plan GAS path is faster than the same static-plan Twilight path in {sum(x > 1 for x in static_speed)}/16 seed-shape cases; range {min(static_speed):.2f}–{max(static_speed):.2f}x. The isolated GAS metadata update improves {min(metadata_speed):.2f}–{max(metadata_speed):.2f}x over the Torch prototype.',
              'The dynamic per-token grouped CPU-metadata+FI-plan diagnostic was 73–82 ms at B32/16K; its GPU-vectorized version was approximately 12.5 ms. Those diagnostic timings have different scopes from the fixed-plan rows and are not to be summed with kernel-only times.',
              '', '## Verification And Limitations', '',
              f"Max absolute error versus flat FI across both seeds: grouped Twilight {max(r['twilight_grouped_max_error'] for r in records):.6f}, grouped GAS {max(r['gas_grouped_max_error'] for r in records):.6f}, warp GAS {max(r['gas_warp_max_error'] for r in records):.6f}. The benchmark checks all batch outputs; a first-request FP32 reference is checked separately.",
              'GPU unit tests covered FP16/BF16 CUDA warp/cooperative kernels, different masks across batch, empty shared/exception cases, CUDA Graph replay, grouped FI masks, exact packed-mask agreement, multi-tile/tail compaction and overflow errors. Six tests are intentionally skipped where a cooperative kernel does not implement a requested variant.',
              'No model score, full model inference, real request batching or serving latency was measured. Static capacities are picked in-sample and replan/overflow costs are absent. The shared synthetic KV/support across batch favors reuse and should not be generalized to independent requests. The native paths are experimental and are not integrated into attention.py.',
              '', '## Files', '',
              '- `summary_per_seed.csv`: all paired measurements and error statistics.',
              '- `benchmark/results_efficiency/twilight_grouped_audited_seed{42,2026}/`: kernel raw rounds and source-hashed config.',
              '- `benchmark/results_efficiency/twilight_static_grouped_audited_seed{42,2026}/`: per-shape wall samples, capacities and source hashes.',
              '- `benchmark/efficiency/bench_twilight_group_fi.py`: grouped and flattened FI replay.',
              '- `benchmark/efficiency/gas_grouped_static_fi.py` and `gas_grouped_compact_triton.py`: experimental static plan and metadata update.',
              '- `csrc/src/gas_group_warp_shared_pool.cu`: experimental fused warp kernel.', '']
    (args.output_dir / 'handoff.md').write_text('\n'.join(lines))
    print(f'wrote {len(records)} paired rows to {args.output_dir}')


if __name__ == '__main__':
    main()
