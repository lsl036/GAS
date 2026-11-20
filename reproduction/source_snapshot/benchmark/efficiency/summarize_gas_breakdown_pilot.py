#!/usr/bin/env python3
"""Create a clearly labelled sum-of-components pilot from measured JSON files."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'benchmark/results_efficiency/gas_compact_pilot_20261113'


def main():
    rows = []
    for batch in (32, 64, 96, 128):
        def read(name):
            return json.loads((RESULTS / f'breakdown_{name}_b{batch}.json').read_text())

        estimator, pruning, attention = read('compact_cache'), read('pruning'), read('attention')
        ms = lambda value: value['median_us'] / 1000
        slow = ms(estimator['measurements']['estimator_requantize'])
        fast = ms(estimator['measurements']['estimator_cached'])
        top = ms(pruning['measurements']['head_top_p'])
        refinement = ms(pruning['measurements']['gas_refinement'])
        qa = ms(attention['timings']['quest_attention'])
        ta = ms(attention['timings']['twilight_attention'])
        ga = ms(attention['timings']['gas_attention'])
        for method, est, pruning_ms, refine_ms, attn in (
            ('quest', 0, 0, 0, qa),
            ('twilight_requantize', slow, top, 0, ta),
            ('twilight_cached_control', fast, top, 0, ta),
            ('gas_cached', fast, top, refinement, ga),
        ):
            rows.append({'batch': batch, 'method': method, 'estimator_ms': est,
                         'top_p_ms': pruning_ms, 'refinement_ms': refine_ms,
                         'attention_ms': attn, 'observed_components_sum_ms': est + pruning_ms + refine_ms + attn,
                         'selector_ms': '', 'pipeline_total_ms': '', 'selector_measured': False,
                         'attention_kv_pool': 'one shared physical KV pool across batch'})
    output = RESULTS / 'breakdown_qasper16k_fixed8192_projected.csv'
    with output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {output}')


if __name__ == '__main__':
    main()
