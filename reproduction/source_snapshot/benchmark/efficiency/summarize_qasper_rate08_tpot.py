#!/usr/bin/env python3
"""Build attention-only E7 projections from Qasper 8% replay components."""
import argparse
import csv
import json
from pathlib import Path

METHODS = ('fa2', 'flashinfer_dense', 'quest', 'twilight', 'gas', 'less_style_8pct')


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f'refusing to overwrite {args.output_dir}')
    root = args.results_root
    old_path = root / 'e7_estimated_tpot/E7_components.csv'
    sparse_path = root / 'qasper_rate08_grouped_summary_20261113/summary.csv'
    old, sparse_rows = read_csv(old_path), read_csv(sparse_path)
    sparse = {(int(r['nominal_context_k']), int(r['batch']), r['method'], r['backend']): r for r in sparse_rows}
    if len(sparse_rows) != 64 or len(sparse) != 64:
        raise ValueError('expected 64 unique sparse summary rows')
    rows = []
    for length in (16, 32):
        for batch in (32, 64, 96, 128):
            dense_dir = root / f'qasper_rate08_dense_{length}k_b{batch}_20261113'
            sparse_dir = root / f'qasper_rate08_grouped_{length}k_b{batch}_20261113'
            dense_config = json.loads((dense_dir / 'config.json').read_text())
            sparse_config = json.loads((sparse_dir / 'config.json').read_text())
            if dense_config['manifest_sha256'] != sparse_config['manifest_sha256'] or dense_config['batch'] != batch:
                raise ValueError(f'mismatched trace manifests: {length}K B{batch}')
            if dense_config['layer'] not in sparse_config['layers']:
                raise ValueError('dense representative layer missing from sparse layers')
            dense_rows = read_csv(dense_dir / 'results.csv')
            dense = {r['method']: float(r['median_us']) for r in dense_rows}
            if len(dense_rows) != 2 or set(dense) != {'fa2', 'flashinfer_dense'}:
                raise ValueError(f'incomplete dense timings: {dense_dir}')
            common_candidates = [r for r in old if int(r['length']) == length * 1024 and int(r['batch_size']) == batch
                                 and r['status'] == 'available' and r['common_ms'] and r['output_ms']]
            if not common_candidates:
                raise ValueError(f'missing common components: {length}K B{batch}')
            source = next((r for r in common_candidates if r['method'] == 'fa2'), common_candidates[0])
            common, lm = float(source['common_ms']), float(source['output_ms'])
            if any(abs(float(r['common_ms']) - common) > 1e-6 or abs(float(r['output_ms']) - lm) > 1e-6
                   for r in common_candidates):
                raise ValueError(f'inconsistent common costs: {length}K B{batch}')
            for backend in ('per_head', 'grouped'):
                for method in METHODS:
                    if method in dense:
                        per_layer = dense[method]
                        attention_ms = 32 * per_layer / 1000
                    else:
                        record = sparse[(length, batch, method, backend)]
                        if record['layers'] != '28':
                            raise ValueError('expected 28 common sparse layers')
                        per_layer = float(record['mean_layer_median_us'])
                        attention_ms = (2 * dense['flashinfer_dense'] + 30 * per_layer) / 1000
                    less_search_aware = (common + lm + (4 * dense['flashinfer_dense'] + 28 * per_layer) / 1000
                                         if method == 'less_style_8pct' else '')
                    rows.append({'context_k': length, 'batch': batch, 'sparse_backend': backend,
                                 'method': method, 'common_ms': common, 'lm_head_ms': lm,
                                 'attention_ms': attention_ms, 'projected_step_ms': common + lm + attention_ms,
                                 'sparse_or_dense_one_layer_us': per_layer,
                                 'dense_fi_one_layer_us': dense['flashinfer_dense'],
                                 'less_search_aware_ms': less_search_aware,
                                 'historical_common_source_method': source['method'],
                                 'dense_source': str((dense_dir / 'results.csv').resolve()),
                                 'sparse_source': str(sparse_path.resolve()), 'common_source': str(old_path.resolve())})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'projection.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lookup = {(r['context_k'], r['batch'], r['sparse_backend'], r['method']): r for r in rows}
    lines = ['# Qasper 8% Attention-Only E7 Projection', '',
             'This is a disconnected component projection, not measured model or serving TPOT. '
             'It uses two saved Qasper GAS-trajectory samples repeated across the batch; 32K uses repeated context. '
             'The FA2 and FI dense measurements use representative layer 15 from the same two-pool input layout; '
             'the sparse input uses the arithmetic mean of 28 layer medians. Q/K/V were captured as BF16 and cast to FP16.', '',
             'Dense formula: historical 32-layer common + historical LM head + 32 * new dense attention. '
             'Sparse formula: historical common + historical LM head + 2 * new dense FI + 30 * new sparse attention. '
             'The 28 measured sparse layers are treated as representative of all 30; layers 2 and 13 were not measured '
             'as sparse in the trace benchmark.', '',
             'The 30-sparse approximation for Less ignores its two dense search layers. The CSV separately provides '
             '`less_search_aware_ms`, using 4 * dense FI + 28 * sparse Less; it is still a proxy and excludes search/selection cost.', '',
             'All methods exclude selector, estimator, top-p, GAS refinement, Less search, metadata, packing and planning. '
             'Historical random-matrix common/LM-head values use CUDA-event measurement; attention uses '
             'torch.utils.benchmark in separate runs. Their sum is neither a true end-to-end latency nor a strict bound.', '',
             'Sparse supports and FI wrappers are as documented in qasper_rate08_grouped_summary_20261113: '
             'per-head paged decode or grouped paged prefill, not official method kernels. The batch repeats two physical '
             'KV pools rather than independently resident requests. Less is a reconstructed 8% proxy.', '']
    for backend in ('per_head', 'grouped'):
        lines += [f'## Sparse backend: {backend}', '', '| Context | Batch | FA2 | FI dense | Quest | Twilight | GAS | Less-style 8% |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for length in (16, 32):
            for batch in (32, 64, 96, 128):
                values = [f"{lookup[length,batch,backend,m]['projected_step_ms']:.3f}" for m in METHODS]
                lines.append(f"| {length}K{' repeated' if length == 32 else ''} | {batch} | " + ' | '.join(values) + ' |')
        lines.append('')
    lines += ['All values are ms per batch step and are not divided by batch size.',
              'For 32K/B128 historical common and LM head are sourced from the available Quest row, '
              'because the historical dense rows were unavailable; all available historical rows have matching common values.',
              '', f'Historical common source: `{old_path.resolve()}`.', f'Sparse source: `{sparse_path.resolve()}`.',
              'Dense per-shape sources and all component values are recorded in `projection.csv`.']
    (args.output_dir / 'Qasper_Rate08_TPOT_Projection.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'wrote {len(rows)} rows to {args.output_dir}')


if __name__ == '__main__':
    main()
