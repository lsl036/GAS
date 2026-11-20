#!/usr/bin/env python3
"""Summarize the three-path author-timer Qasper attention replay."""
import argparse
import csv
import json
import statistics
from pathlib import Path

LAYERS = tuple(x for x in range(3, 32) if x != 13)
METHODS = ('fi_head_attention', 'fi_attention', 'custom_attention')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    rows = []
    for context in ('16k', '32k'):
        for batch in (32, 64, 96, 128):
            directory = args.root / f'{context}_b{batch}'
            missing = [n for n in LAYERS if not (directory / f'layer{n:02d}.json').exists()]
            records = [json.loads((directory / f'layer{n:02d}.json').read_text()) for n in LAYERS if n not in missing]
            row = {'context': context, 'batch': batch, 'completed_layers': len(records), 'expected_layers': len(LAYERS),
                   'missing_layers': json.dumps(missing)}
            for name in METHODS:
                row[name + '_us'] = statistics.fmean(r['timings'][name]['median_us'] for r in records) if records and not missing else ''
            row['head_over_custom'] = row['fi_head_attention_us'] / row['custom_attention_us'] if not missing else ''
            row['group_over_custom'] = row['fi_attention_us'] / row['custom_attention_us'] if not missing else ''
            row['max_head_group_error'] = max((r['fi_head_group_max_abs_error_all_batch'] for r in records), default='')
            row['max_fi_custom_error'] = max((r['fi_custom_max_abs_error_all_batch'] for r in records), default='')
            row['max_custom_fp32_error'] = max((max(r['custom_fp32_max_abs_error_first_two']) for r in records), default='')
            row['mask_differences'] = sum(r['captured_mask_differences'] for r in records)
            rows.append(row)
    with (args.root / 'summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = ['# Qasper GAS: Author Timer Replay', '',
             'Microseconds per layer; arithmetic mean of 28 per-layer medians of three torch.utils.benchmark round means (5 warmups, 50 repeats). Blank values are incomplete.', '',
             '| Context | Batch | Layers | GAS-FI head | GAS-FI group | GAS-custom | Head/custom | Group/custom | Max FI/custom error |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        fmt = lambda x: f'{x:.2f}' if isinstance(x, float) else str(x or '-')
        vals = ' | '.join(fmt(r[k]) for k in ('fi_head_attention_us', 'fi_attention_us', 'custom_attention_us', 'head_over_custom', 'group_over_custom'))
        err = f'{r["max_fi_custom_error"]:.6g}' if r['max_fi_custom_error'] != '' else '-'
        lines.append(f'| {r["context"]} | {r["batch"]} | {r["completed_layers"]}/{r["expected_layers"]} | {vals} | {err} |')
    lines += ['', 'All three use the same two captured Qasper requests, alternating across each batch and sharing two physical KV pools. The 32K trace is repeated context. The same prebuilt GAS mask is used for all methods; selection, refinement, metadata preparation, FI plan construction, and model work are excluded.',
              '', 'FI head is a paged decode adapter, FI group a paged prefill adapter; custom uses the experimental cooperative shared-pool kernel and pre-offset indices. This is an attention-only adapter replay, not native model TPOT. torch.utils.benchmark includes host/framework dispatch; do not label the results GPU-kernel-only or mix them with CUDA Graph event numbers.',
              '', 'Each completed layer verifies matching captured masks, all-batch FI-head/group/custom outputs, and FP32 references on both requests. See summary.csv and per-layer JSON for errors and round means.', '']
    (args.root / 'summary.md').write_text('\n'.join(lines))
    print(f'wrote {args.root / "summary.csv"} and summary.md')


if __name__ == '__main__':
    main()
