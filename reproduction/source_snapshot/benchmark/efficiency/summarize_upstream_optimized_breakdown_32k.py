#!/usr/bin/env python3
"""Combine upstream component timings with paired 32K trace tail measurements."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'benchmark/results_efficiency/gas_compact_pilot_20261113'


def main():
    upstream_path = RESULTS / 'upstream32k_official_components.json'
    upstream = json.loads(upstream_path.read_text())['results']
    rows = []
    for batch in (32, 64, 96, 128):
        replay_path = RESULTS / f'upstream32k_prepared_unified_coop4_repeat500_b{batch}.json'
        replay = json.loads(replay_path.read_text())
        if not (replay['normalized_group'] and replay['normalized_prepared'] and
                replay['unified_exception'] and replay['decode_mode'] == 'coop8p4'):
            raise ValueError(f'unexpected kernel configuration: {replay_path}')
        if replay['prototype_vs_python_mask_differences'] or replay['reconstructed_vs_captured_gas_mask_differences']:
            raise ValueError(f'mask mismatch: {replay_path}')
        q = upstream['quest']['components_us'][str(batch)]
        t = upstream['quest-twi']['components_us'][str(batch)]
        paired = replay['paired_timings']
        for method, selector, estimator, top_p, tail in (
            ('quest_grouped_fi', q['token selector'], 0, 0, paired['quest_attention']['median_us']),
            ('twilight_grouped_fi', t['token selector'], t['spgemv'], t['top p'], paired['twilight_attention']['median_us']),
            ('gas_prototype', t['token selector'], t['spgemv'], t['top p'], paired['gas_connected']['median_us']),
        ):
            rows.append({'batch': batch, 'method': method, 'selector_us': selector,
                         'int4_spgemv_us': estimator, 'top_p_us': top_p, 'tail_us': tail,
                         'projected_sum_us': selector + estimator + top_p + tail,
                         'tail_definition': 'refinement_plus_attention' if method == 'gas_prototype' else 'attention',
                         'upstream_json': str(upstream_path), 'paired_replay_json': str(replay_path)})
    output = RESULTS / 'upstream32k_trace_breakdown_optimized_components.csv'
    with output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print('WROTE', output)


if __name__ == '__main__':
    main()
