#!/usr/bin/env python3
"""Validate and summarize eight Qasper attention-only replay shapes."""
import argparse
import csv
import json
import statistics
from pathlib import Path

METHODS = ('quest', 'twilight', 'gas', 'less_style_8pct')
BACKENDS = ('per_head', 'grouped')
LAYERS = {x for x in range(3, 32) if x != 13}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f'refusing to overwrite {args.output_dir}')
    result = []
    source_files = []
    for length in (16, 32):
        for batch in (32, 64, 96, 128):
            base = args.results_root / f'qasper_rate08_grouped_{length}k_b{batch}_20261113'
            config = json.loads((base / 'config.json').read_text())
            with (base / 'results.csv').open(newline='') as f:
                rows = list(csv.DictReader(f))
            expected = {(layer, method, backend) for layer in LAYERS for method in METHODS for backend in BACKENDS}
            actual = {(int(r['layer']), r['method'], r['backend']) for r in rows}
            if actual != expected or len(rows) != len(expected) or set(config['layers']) != LAYERS:
                raise ValueError(f'incomplete or duplicate rows: {base}')
            if int(config['batch']) != batch:
                raise ValueError(f'batch mismatch: {base}')
            source_files.append(str((base / 'results.csv').resolve()))
            for method in METHODS:
                for backend in BACKENDS:
                    subset = [r for r in rows if r['method'] == method and r['backend'] == backend]
                    result.append({'nominal_context_k': length, 'batch': batch, 'method': method,
                                   'backend': backend, 'layers': len(subset),
                                   'mean_layer_median_us': statistics.fmean(float(r['median_us']) for r in subset),
                                   'mean_tokens_per_head': statistics.fmean(float(r['per_head_mean']) for r in subset),
                                   'mean_group_union': statistics.fmean(float(r['group_union_mean']) for r in subset),
                                   'max_abs_error': max(float(r['max_abs_error']) for r in subset),
                                   'source': str((base / 'results.csv').resolve())})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    lookup = {(r['nominal_context_k'], r['batch'], r['method'], r['backend']): r for r in result}
    lines = ['# Qasper Quest 8%: Per-Head Versus Grouped FI', '',
             'A100 40GB, two Qasper samples (15 and 54), 28 non-search layers 3-12 and 14-31, teacher-forced question token. '
             'The 32K input is repeated-context Qasper, not a natural 32K task. Source BF16 Q/K/V cast to FP16 before timing. '
             'Each batch alternates the two requests and shares two physical KV pools; it is not an independent-request batch.', '',
             'Quest 8%, Twilight p=0.85 and GAS p=0.8 + exception16 use saved masks on the GAS trajectory; '
             'Less-style K=floor(0.08 * actual length), 25% recent, is reconstructed from search-layer GAS Q/K. '
             'It is not official LessIsMore model inference or kernel.', '',
             'Values are arithmetic means across 28 layer medians; each median is from three round means '
             'of 50 calls, five warmups, using the author torch.utils.benchmark helper. '
             'Selection, mask construction, FI plan, layout and model work are excluded.', '',
             'Per-head FI is a paged decode adapter; grouped FI is a paged prefill adapter with four query positions '
             'and exact per-head custom masks. They are different backend paths. They are not native Quest, Twilight, '
             'GAS or LessIsMore kernels and these data are not TPOT.', '',
             '| Context | Batch | Quest head / group | Twilight head / group | GAS head / group | Less 8% head / group |',
             '|---|---:|---:|---:|---:|---:|']
    for length in (16, 32):
        for batch in (32, 64, 96, 128):
            vals = [f"{lookup[length,batch,m,'per_head']['mean_layer_median_us']:.2f} / {lookup[length,batch,m,'grouped']['mean_layer_median_us']:.2f}" for m in METHODS]
            lines.append(f"| {length}K{' repeated' if length == 32 else ''} | {batch} | " + ' | '.join(vals) + ' |')
    lines += ['', '## Logical Supports', '',
              '| Context | Method | Tokens/head | Union/group |', '|---|---|---:|---:|']
    for length in (16, 32):
        for method in METHODS:
            row = lookup[length, 32, method, 'grouped']
            lines.append(f"| {length}K{' repeated' if length == 32 else ''} | {method} | {row['mean_tokens_per_head']:.2f} | {row['mean_group_union']:.2f} |")
    lines += ['', 'Exact-mask per-head and grouped outputs were compared for all batch rows; FP32 references '
              'were checked on both distinct requests. Group union is a logical support count, not measured HBM traffic.',
              'These eight shapes use only two samples and one position; 28 layer observations are correlated. '
              'Previous Fig.6 data used physically copied KV and CUDA Graph timing and cannot be directly merged.',
              '', 'Per-layer results and per-round means:'] + [f'- `{path}`' for path in source_files]
    (args.output_dir / 'Qasper_Rate08_FI_Summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[:14]))
    print(f'wrote {len(result)} aggregate rows to {args.output_dir}')


if __name__ == '__main__':
    main()
