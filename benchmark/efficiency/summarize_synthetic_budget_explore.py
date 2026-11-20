#!/usr/bin/env python3
"""Summarize synthetic-budget component projections with explicit provenance."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'benchmark/results_efficiency/gas_compact_pilot_20261113'


def main():
    upstream_path = RESULTS / 'synthetic_budget_32768_upstream_components.json'
    upstream = json.loads(upstream_path.read_text())['results']
    rows = []
    for batch in (32, 64, 96, 128):
        replay_path = RESULTS / f'synthetic_budget_offset8_peak12_seed43_b{batch}.json'
        replay = json.loads(replay_path.read_text())
        if not replay['use_trace_scores'] or replay['prototype_vs_python_mask_differences'] or replay['reconstructed_vs_captured_gas_mask_differences']:
            raise ValueError(f'input or mask validation failed: {replay_path}')
        q = upstream['quest']['components_us'][str(batch)]
        t = upstream['quest-twi']['components_us'][str(batch)]
        paired = replay['paired_timings']
        for method, selector, estimator, top_p, tail in (
            ('quest_grouped_fi', q['token selector'], 0, 0, paired['quest_attention']['median_us']),
            ('twilight_grouped_fi', t['token selector'], t['spgemv'], t['top p'], paired['twilight_attention']['median_us']),
            ('gas_prototype', t['token selector'], t['spgemv'], t['top p'], paired['gas_connected']['median_us']),
        ):
            rows.append({'batch': batch, 'method': method, 'selector_us': selector,
                         'estimator_us': estimator, 'top_p_us': top_p, 'tail_us': tail,
                         'projected_sum_us': selector + estimator + top_p + tail,
                         'tail_definition': 'refinement_plus_attention' if method == 'gas_prototype' else 'attention',
                         'upstream_json': str(upstream_path), 'replay_json': str(replay_path)})
    output = RESULTS / 'synthetic_budget_offset8_peak12_projected_components.csv'
    with output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print('WROTE', output)


if __name__ == '__main__':
    main()
