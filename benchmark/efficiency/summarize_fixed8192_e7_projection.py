#!/usr/bin/env python3
"""Combine fixed-8192 attention replays with historical E7 common components.

This is an attention-path-only component projection, not end-to-end TPOT.
"""
import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METHODS = ('fa2', 'flashinfer_dense', 'quest_per_head_fi', 'twilight_grouped_fi',
           'gas_grouped_fi', 'less_style_grouped_fi')


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--results-root', type=Path, default=ROOT / 'benchmark/results_efficiency')
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f'refusing to overwrite {args.output_dir}')
    old_path = args.results_root / 'e7_estimated_tpot/E7_components.csv'
    old = read_csv(old_path)
    rows = []
    for length in (16384, 32768):
        for batch in (32, 64, 96, 128):
            path = args.results_root / f'fixed8192_quest_perhead_L{length}_B{batch}_20261113/results.csv'
            current = read_csv(path)
            kernels = {r['method']: float(r['median_us']) for r in current}
            if len(current) != len(METHODS) or set(kernels) != set(METHODS):
                raise ValueError(f'incomplete or duplicate kernel methods: {path}')
            available = [r for r in old if int(r['length']) == length and int(r['batch_size']) == batch
                         and r['status'] == 'available' and r['common_ms'] and r['output_ms']]
            if not available:
                raise ValueError(f'no historical common components: {length}, {batch}')
            chosen = next((r for r in available if r['method'] == 'fa2'), available[0])
            common, output = float(chosen['common_ms']), float(chosen['output_ms'])
            if any(abs(float(r['common_ms']) - common) > 1e-6 or
                   abs(float(r['output_ms']) - output) > 1e-6 for r in available):
                raise ValueError(f'inconsistent historical common components: {length}, {batch}')
            for method in METHODS:
                attn_us = (32 * kernels[method] if method in ('fa2', 'flashinfer_dense')
                           else 2 * kernels['flashinfer_dense'] + 30 * kernels[method])
                rows.append({'length': length, 'batch': batch, 'method': method,
                             'common_ms': common, 'lm_head_ms': output,
                             'attention_ms': attn_us / 1000,
                             'projected_step_ms': common + output + attn_us / 1000,
                             'attention_one_layer_us': kernels[method],
                             'dense_fi_one_layer_us': kernels['flashinfer_dense'],
                             'historical_common_source_method': chosen['method'],
                             'kernel_source': str(path), 'common_source': str(old_path)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'projection.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    meta = {'scope': 'disconnected attention-path component projection; not measured end-to-end TPOT',
            'formula_dense': 'old 32-layer common + old LM head + 32 * new dense attention',
            'formula_sparse': 'old 32-layer common + old LM head + 2 * new dense FI + 30 * new sparse attention',
            'excluded': 'selector, estimator, top-p, GAS refinement, metadata, other unmeasured operations',
            'caveat': 'historical random FP16 common components timed with CUDA events; new synthetic attention timed with torch.utils.benchmark; separate executions and input layouts',
            'old_components': str(old_path), 'kernel_sources': sorted({r['kernel_source'] for r in rows})}
    (args.output_dir / 'config.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    print('| Context | Batch | FA2 | FlashInfer | Quest | Twilight | GAS | Less-style |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|')
    for length in (16384, 32768):
        for batch in (32, 64, 96, 128):
            subset = {r['method']: r for r in rows if r['length'] == length and r['batch'] == batch}
            print(f"| {length // 1024}K | {batch} | " + ' | '.join(f"{subset[m]['projected_step_ms']:.3f}" for m in METHODS) + ' |')
    print(f'wrote {len(rows)} rows to {args.output_dir}')


if __name__ == '__main__':
    main()
