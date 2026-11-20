#!/usr/bin/env python3
"""Summarize a mixed upstream-component / captured-mask pilot, not TPOT."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'benchmark/results_efficiency/gas_compact_pilot_20261113'


def main():
    official_path = RESULTS / 'upstream32k_official_components.json'
    official = json.loads(official_path.read_text())['results']
    rows = []
    for batch in (32, 64, 96, 128):
        captured_path = RESULTS / f'upstream32k_group_b{batch}.json'
        captured = json.loads(captured_path.read_text())
        if captured['prototype_vs_python_mask_differences'] or captured['reconstructed_vs_captured_gas_mask_differences']:
            raise ValueError(f'GAS masks do not match: {captured_path}')
        q = official['quest']['components_us'][str(batch)]
        t = official['quest-twi']['components_us'][str(batch)]
        timing = captured['timings']
        measure = lambda key: timing[key]['median_us']
        settings = (
            ('quest_grouped_fi', q['token selector'], 0, 0, 0, measure('quest_attention'), None),
            ('twilight_grouped_fi', t['token selector'], t['spgemv'], t['top p'], 0, measure('twilight_attention'), None),
            ('gas_prototype', t['token selector'], t['spgemv'], t['top p'], measure('gas_refinement'),
             measure('gas_attention'), measure('gas_connected')),
        )
        for method, selector, estimator, top_p, refine, attention, connected in settings:
            rows.append({'batch': batch, 'method': method, 'selector_us': selector,
                         'int4_spgemv_us': estimator, 'top_p_us': top_p,
                         'refinement_us': refine, 'attention_us': attention,
                         'sum_independent_components_us': selector + estimator + top_p + refine + attention,
                         'refinement_plus_attention_connected_us': connected if connected is not None else '',
                         'sum_with_connected_tail_us': selector + estimator + top_p + connected if connected is not None else '',
                         'upstream_component_json': str(official_path), 'attention_json': str(captured_path)})
    output = RESULTS / 'upstream32k_trace_breakdown_components.csv'
    with output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print('WROTE', output)


if __name__ == '__main__':
    main()
