#!/usr/bin/env python3
"""Combine tuned dense measurements with the existing Qasper 8% replay."""
import argparse
import csv
import json
from pathlib import Path

METHODS = ('fa2', 'flashinfer', 'quest', 'twilight', 'gas', 'less_style_8pct')
DENSE_METHODS = {'fa2', 'fi_page1_tc0', 'fi_page1_tc1', 'fi_page16_tc0', 'fi_page16_tc1'}


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
    common_path = root / 'e7_estimated_tpot/E7_components.csv'
    sparse_path = root / 'qasper_rate08_grouped_summary_20261113/summary.csv'
    historical, sparse_rows = read_csv(common_path), read_csv(sparse_path)
    sparse = {(int(r['nominal_context_k']), int(r['batch']), r['method'], r['backend']): r for r in sparse_rows}
    if len(sparse_rows) != 64 or len(sparse) != 64:
        raise ValueError('expected 64 unique sparse summary rows')
    rows, dense_audit = [], []
    for length in (16, 32):
        for batch in (32, 64, 96, 128):
            dense_dir = root / f'qasper_rate08_dense_tuned_{length}k_b{batch}_20261113'
            sparse_dir = root / f'qasper_rate08_grouped_{length}k_b{batch}_20261113'
            dc = json.loads((dense_dir / 'config.json').read_text())
            sc = json.loads((sparse_dir / 'config.json').read_text())
            if dc['manifest_sha256'] != sc['manifest_sha256'] or dc['batch'] != batch or dc['layer'] not in sc['layers']:
                raise ValueError(f'dense/sparse fixture mismatch: {length}K B{batch}')
            if dc['backends']['fi_page1_tc1']['resolved'] != 'fa2' or dc['backends']['fi_page16_tc1']['resolved'] != 'fa2':
                raise ValueError('unexpected FI tensor-core backend')
            dense_rows = read_csv(dense_dir / 'results.csv')
            dense = {r['method']: float(r['median_us']) for r in dense_rows}
            if len(dense_rows) != len(DENSE_METHODS) or set(dense) != DENSE_METHODS:
                raise ValueError(f'incomplete dense timings: {dense_dir}')
            for r in dense_rows:
                dense_audit.append({'context_k': length, 'batch': batch, **r,
                                    'source': str((dense_dir / 'results.csv').resolve())})
            candidates = [r for r in historical if int(r['length']) == length * 1024 and int(r['batch_size']) == batch
                          and r['status'] == 'available' and r['common_ms'] and r['output_ms']]
            if not candidates:
                raise ValueError(f'missing common components: {length}K B{batch}')
            source = next((r for r in candidates if r['method'] == 'fa2'), candidates[0])
            common, lm = float(source['common_ms']), float(source['output_ms'])
            if any(abs(float(r['common_ms']) - common) > 1e-6 or abs(float(r['output_ms']) - lm) > 1e-6
                   for r in candidates):
                raise ValueError(f'inconsistent historical components: {length}K B{batch}')
            for dense_key in ('fi_page16_tc1', 'fi_page1_tc1'):
                for backend in ('per_head', 'grouped'):
                    for method in METHODS:
                        if method == 'fa2':
                            per_layer, attn_ms = dense['fa2'], 32 * dense['fa2'] / 1000
                        elif method == 'flashinfer':
                            per_layer, attn_ms = dense[dense_key], 32 * dense[dense_key] / 1000
                        else:
                            item = sparse[(length, batch, method, backend)]
                            if item['layers'] != '28':
                                raise ValueError('expected 28 sparse layers')
                            per_layer = float(item['mean_layer_median_us'])
                            attn_ms = (2 * dense[dense_key] + 30 * per_layer) / 1000
                        less_aware = (common + lm + (4 * dense[dense_key] + 28 * per_layer) / 1000
                                      if method == 'less_style_8pct' else '')
                        rows.append({'context_k': length, 'batch': batch, 'dense_fi_config': dense_key,
                                     'sparse_backend': backend, 'method': method, 'common_ms': common, 'lm_head_ms': lm,
                                     'attention_ms': attn_ms, 'projected_step_ms': common + lm + attn_ms,
                                     'less_search_aware_ms': less_aware, 'one_layer_us': per_layer,
                                     'dense_fi_one_layer_us': dense[dense_key], 'historical_common_source_method': source['method'],
                                     'dense_source': str((dense_dir / 'results.csv').resolve()),
                                     'sparse_source': str(sparse_path.resolve()), 'common_source': str(common_path.resolve())})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, data in (('projection.csv', rows), ('dense_audit.csv', dense_audit)):
        with (args.output_dir / filename).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    lookup = {(r['context_k'], r['batch'], r['dense_fi_config'], r['sparse_backend'], r['method']): r for r in rows}
    lines = ['# Qasper 8% TPOT Component Projection: Tuned Dense FI', '',
             'Disconnected component projection, not measured model or serving TPOT. Two Qasper samples '
             'are repeated across each batch with two shared physical KV pools; the 32K workload repeats context.', '',
             'Dense FA2 and FI are remeasured at representative layer 15 on the same Q/K/V layout. '
             'The main view consistently uses FI page_size=16 and use_tensor_cores=True across all shapes; '
             'page_size=1 with tensor cores is a fixed alternative, not a per-shape fastest selection. '
             'The local FI 0.6.11.post1 resolves its FP16 tensor-core path to its `fa2` backend; '
             'this is not necessarily an independent algorithm from the separately timed FA2 interface.', '',
             'Dense formula: historical 32-layer common + LM head + 32 * measured dense attention. '
             'Sparse formula: historical common + LM head + 2 * tuned dense FI + 30 * sparse attention. '
             'Sparse attention is the average of 28 non-search layer medians and is extrapolated to 30 layers. '
             'For Less the table uses 4 * dense FI + 28 * sparse attention to account for two additional search layers; '
             'the older uniform 2+30 approximation remains in the CSV.', '',
             'Selection, estimators, top-p, GAS refinement, Less search, metadata, packing and FI plan are excluded. '
             'Common components are old random FP16 matrix CUDA-event timings; attention is measured in separate '
             'torch.utils.benchmark runs. This sum is not a measured TPOT or a guaranteed bound.', '',
             'All sparse paths are FI adapters, not official native Quest, Twilight, GAS or LessIsMore kernels. '
             'Less-style K is floor(8% of each actual context) and its support is reconstructed from GAS search states.', '']
    for backend in ('grouped', 'per_head'):
        lines += [f'## Sparse FI: {backend}; dense FI: page16 + TC', '',
                  '| Context | Batch | FA2 | FI dense | Quest | Twilight | GAS | Less, 28 sparse + 4 dense |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for length in (16, 32):
            for batch in (32, 64, 96, 128):
                cells = []
                for method in METHODS:
                    r = lookup[length, batch, 'fi_page16_tc1', backend, method]
                    cells.append(f"{(r['less_search_aware_ms'] if method == 'less_style_8pct' else r['projected_step_ms']):.3f}")
                lines.append(f"| {length}K{' repeated' if length == 32 else ''} | {batch} | " + ' | '.join(cells) + ' |')
        lines.append('')
    lines += ['Units are ms per batch step, not divided by batch size. All per-round dense measurements, '
              'page/tensor-core settings and errors are in `dense_audit.csv`; the full alternative projection '
              'is in `projection.csv`. 32K/B128 common costs come from the available Quest historical row '
              'because the historical dense rows were unavailable.']
    (args.output_dir / 'Qasper_Rate08_Tuned_TPOT_Projection.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'wrote {len(rows)} projections and {len(dense_audit)} dense timing rows to {args.output_dir}')


if __name__ == '__main__':
    main()
