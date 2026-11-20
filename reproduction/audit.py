#!/usr/bin/env python3
"""Compare the preserved paper figure inputs with available experiment records."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = Path(__file__).resolve().parent / 'figure_inputs'


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def notebook_arrays(path, variable):
    notebook = json.loads(path.read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(line for line in cell['source'] if not line.lstrip().startswith('%'))
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == variable for t in node.targets):
                if not isinstance(node.value, ast.Dict):
                    raise ValueError(f'{variable} is not a literal dictionary')
                return {ast.literal_eval(k): ast.literal_eval(v.args[0]) for k, v in zip(node.value.keys, node.value.values)}
    raise ValueError(f'{variable} missing from {path}')


def first_markdown_table(path):
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.startswith('##'):
            break
        if line.strip().startswith('|'):
            rows.append([v.strip() for v in next(csv.reader([line.strip().strip('|')], delimiter='|'))])
    return {row[0]: dict(zip(rows[0], row)) for row in rows[2:]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--live', action='store_true', help='Compare with live repository results instead of frozen reference summaries.')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f'refusing to overwrite {args.output}')
    reference = ROOT if args.live else Path(__file__).resolve().parent / 'reference_results'
    if not args.live:
        manifest = json.loads((reference / 'manifest.json').read_text())
        for entry in manifest['files']:
            path = reference / entry['path']
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError(f'frozen reference missing or changed: {entry["path"]}')
    report = ['# Figure Data Audit', '',
              'This audit compares the immutable figure-input snapshot with available saved summaries. '
              'A matching number is not proof of a connected, fair or representative experiment. '
              f'The snapshot is not modified. Reference: {"live repository" if args.live else "frozen reference_results"}.', '']
    md = first_markdown_table(SNAPSHOT / 'fig4-KVvsScore/LLaMa_KV_Score.md')
    formal = {r['method_id']: r for r in read_csv(reference / 'benchmark/RULER/fig4a_llama31/fig4a_summary.csv')}
    mapping = {'Full attention': 'full', 'GAS': 'gas', 'Quest 8%': 'quest_8pct', 'DS 8%': 'ds_8pct',
               'Twilight default': 'twilight_default', 'Quest KV-calibrated': 'quest_cal',
               'DS KV-calibrated': 'ds_cal', 'Twilight KV-calibrated': 'twilight_cal'}
    report += ['## Fig. 4: Snapshot Versus 32K Formal Summary', '',
               '| Method | Snapshot score | Formal score | Snapshot KV | Formal KV |', '|---|---:|---:|---:|---:|']
    mismatches = 0
    for label, key in mapping.items():
        x, y = md[label], formal[key]
        ps, fs = float(x['Macro score']), float(y['macro_score'])
        pk, fk = float(x['Mean loaded KV']), float(y['mean_loaded_KV'])
        if abs(ps - fs) > .011 or abs(pk - fk) > .51:
            mismatches += 1
            report.append(f'| {label} | {ps:.2f} | {fs:.2f} | {pk:.2f} | {fk:.2f} |')
    if not mismatches:
        report.append('| No mismatch above rounding tolerance | | | | |')
    report += ['', 'Comparison source: `benchmark/RULER/fig4a_llama31/fig4a_summary.csv`. '
               'Other curve points are not verified by this comparison.', '']
    source = read_csv(reference / 'benchmark/results_efficiency/qasper_rate08_grouped_summary_20261113/summary.csv')
    index = {(int(x['nominal_context_k']), int(x['batch']), x['method'], x['backend']): float(x['mean_layer_median_us']) for x in source}
    arrays = notebook_arrays(SNAPSHOT / 'fig6-kernel speedup/SparseAttenKernel/sparse_attention_kernel.ipynb', 'KERNEL_LATENCY_US')
    report += ['## Fig. 6: GAS-FI Series Provenance', '',
               '| Context | Batch | Snapshot GAS | Earlier FI head | Earlier FI group | Matches |',
               '|---|---:|---:|---:|---:|---|']
    for i, batch in enumerate((32, 64, 96, 128)):
        for j, context in enumerate((16, 32)):
            val = float(arrays['gas'][i][j])
            head, group = index[(context, batch, 'gas', 'per_head')], index[(context, batch, 'gas', 'grouped')]
            matches = [name for name, actual in (('head', head), ('group', group)) if abs(val - actual) <= .011]
            report.append(f'| {context}K | {batch} | {val:.2f} | {head:.2f} | {group:.2f} | {", ".join(matches) or "neither"} |')
    report += ['', 'Source: `benchmark/results_efficiency/qasper_rate08_grouped_summary_20261113/summary.csv`. '
               'The notebook also includes GAS-custom from a later paired run. '
               'Different source runs must not be silently presented as one paired measurement.', '']
    fig7 = notebook_arrays(SNAPSHOT / 'fig7-TPOT result/NewEnv/TPOT_Speedup.ipynb', 'TPOT_EXCLUDED_MS')
    old = read_csv(reference / 'benchmark/results_efficiency/qasper_rate08_tpot_projection_20261113/projection.csv')
    tuned = read_csv(reference / 'benchmark/results_efficiency/qasper_rate08_tpot_tuned_projection_20261113/projection.csv')
    custom = read_csv(reference / 'benchmark/results_efficiency/qasper_rate08_tpot_tuned_projection_20261113/gas_custom_projection.csv')
    report += ['## Fig. 7: Projection Sources', '',
               'Compare each displayed array with old grouped projection, tuned grouped projection, and custom projection at 0.011 ms tolerance.', '',
               '| Array | Matches old | Matches tuned | Matches custom |', '|---|---:|---:|---:|']
    names = {'fa2': 'fa2', 'flashinfer': 'flashinfer', 'quest': 'quest', 'lessismore': 'less_style_8pct', 'quest_twi': 'twilight', 'gas': 'gas'}
    for name, method in names.items():
        counts = [0, 0, 0]
        for i, batch in enumerate((32, 64, 96, 128)):
            for j, context in enumerate((16, 32)):
                value = float(fig7[name][i][j])
                old_method = 'flashinfer_dense' if method == 'flashinfer' else method
                o = next(x for x in old if int(x['context_k']) == context and int(x['batch']) == batch and x['sparse_backend'] == 'grouped' and x['method'] == old_method)
                t = next(x for x in tuned if int(x['context_k']) == context and int(x['batch']) == batch and x['sparse_backend'] == 'grouped' and x['dense_fi_config'] == 'fi_page16_tc1' and x['method'] == method)
                c = next(x for x in custom if int(x['context_k']) == context and int(x['batch']) == batch)
                for n, number in enumerate((float(o['projected_step_ms']), float(t['projected_step_ms']), float(c['custom_projection_ms']) if name == 'gas' else None)):
                    if number is not None and abs(value - number) <= .011:
                        counts[n] += 1
        report.append(f'| {name} | {counts[0]}/8 | {counts[1]}/8 | {counts[2]}/8 |')
    report += ['', 'Counts do not establish provenance by themselves; some values can coincide. '
               'Less-style uses `projected_step_ms` (30 sparse), not the separate search-aware column. '
               'The custom projection mixes separately measured common/dense and sparse components; it is not measured TPOT.', '']
    nb = json.loads((SNAPSHOT / 'fig8-TimeBreakdown/TimeBreakdown.ipynb').read_text(encoding='utf-8'))
    src = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
    report += ['## Fig. 8: Explicit Scaling And Labels', '',
               f'- Snapshot contains `BREAKDOWN_MS[method][:, [0, 1, 2, 4]] *= 0.9`: **{str("BREAKDOWN_MS[method][:, [0, 1, 2, 4]] *= 0.9" in src).lower()}**.',
               '- The notebook labels `common_ms` as `Top-k Selector`; its adjacent README describes common transformer-layer computation, not a selector measurement.',
               '- The backup notebook uses a different active synthetic component table than its `REPORTED_TOTAL_US` values. Its source comments alone do not prove those active values were measured.',
               '- Treat these as manuscript-input snapshots only; do not present them as an audited measured pipeline or overwrite them without an explicit provenance decision.', '']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(report), encoding='utf-8')
    print(f'wrote {args.output}; Fig.4 mismatched rows: {mismatches}')


if __name__ == '__main__':
    main()
