#!/usr/bin/env python3
"""Summarize paired GAS replay results without hiding incomplete layers."""
import argparse
import csv
import json
import statistics
from pathlib import Path

LAYERS = tuple(x for x in range(3, 32) if x != 13)

METHODS = ('fi_attention', 'custom_attention', 'fi_refine_static_attention', 'custom_refine_attention')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    rows = []
    for context in ('16k', '32k'):
        for batch in (32, 64, 96, 128):
            directory = args.root / f'{context}_b{batch}'
            missing = [x for x in LAYERS if not (directory / f'layer{x:02d}.json').exists()]
            records = [json.loads((directory / f'layer{x:02d}.json').read_text()) for x in LAYERS if x not in missing]
            row = {'context': context, 'batch': batch, 'completed_layers': len(records), 'expected_layers': len(LAYERS),
                   'missing_layers': json.dumps(missing)}
            for method in METHODS:
                row[method + '_us'] = statistics.fmean(r['timings'][method]['median_us'] for r in records) if records and not missing else ''
            row['fi_over_custom_attention'] = row['fi_attention_us'] / row['custom_attention_us'] if not missing else ''
            row['fi_static_over_custom_connected'] = row['fi_refine_static_attention_us'] / row['custom_refine_attention_us'] if not missing else ''
            row['max_fi_custom_abs_error'] = max((r['fi_custom_max_abs_error_all_batch'] for r in records), default='')
            row['max_fp32_abs_error'] = max((max(r['custom_fp32_max_abs_error_first_two']) for r in records), default='')
            row['mask_differences'] = sum(r['captured_mask_differences'] for r in records)
            rows.append(row)
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / 'summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = ['# Qasper GAS Paired Replay', '',
             'Arithmetic means of per-layer CUDA graph event medians, microseconds per layer. Blank cells indicate incomplete configurations.', '',
             '| Context | Batch | Layers | FI attention | Custom attention | FI refinement + static-plan attention | Custom refinement + attention | FI/custom attention | Max output error |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        fmt = lambda x: f'{x:.2f}' if isinstance(x, float) else str(x or '-')
        lines.append(f'| {r["context"]} | {r["batch"]} | {r["completed_layers"]}/{r["expected_layers"]} | '
                     + ' | '.join(fmt(r[k]) for k in ('fi_attention_us', 'custom_attention_us', 'fi_refine_static_attention_us', 'custom_refine_attention_us', 'fi_over_custom_attention'))
                     + f' | {r["max_fi_custom_abs_error"]:.6g} |')
    lines += ['', 'FI connected timing is an idealized static-mask replay: refinement output is checked but the FI plan does not consume it dynamically. Index/plan construction is excluded. Custom connected timing includes converting local token indices to flat two-pool indices. Both paths exclude Quest selection, estimator and Twilight top-p.',
              '', 'The two captured Qasper requests alternate in the batch and use two physical KV pools. The 32K traces repeat context. All completed cases require exact captured GAS masks, an all-batch FI/custom comparison and FP32 checks for both requests.',
              '', 'The prior FI summary uses a different timer and aggregation. Do not merge old and new latencies into a purported paired comparison.', '']
    (args.root / 'summary.md').write_text('\n'.join(lines))
    print(f'wrote {args.root / "summary.csv"} and summary.md')


if __name__ == '__main__':
    main()
