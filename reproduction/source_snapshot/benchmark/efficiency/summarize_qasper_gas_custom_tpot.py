#!/usr/bin/env python3
"""Project paired GAS attention replays using the existing Qasper E7 components."""
import argparse
import csv
from pathlib import Path


def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def main():
    root = Path(__file__).resolve().parents[2] / 'benchmark/results_efficiency'
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=root / 'qasper_rate08_tpot_tuned_projection_20261113/gas_custom_projection.csv')
    args = p.parse_args()
    prior_path = root / 'qasper_rate08_tpot_tuned_projection_20261113/projection.csv'
    paired_path = root / 'qasper_gas_author_20261114/summary.csv'
    prior = read(prior_path)
    paired = read(paired_path)
    rows = []
    for context in (16, 32):
        for batch in (32, 64, 96, 128):
            matches = [r for r in prior if int(r['context_k']) == context and int(r['batch']) == batch
                       and r['dense_fi_config'] == 'fi_page16_tc1' and r['sparse_backend'] == 'grouped' and r['method'] == 'gas']
            samples = [r for r in paired if r['context'] == f'{context}k' and int(r['batch']) == batch]
            if len(matches) != 1 or len(samples) != 1 or samples[0]['completed_layers'] != '28' or samples[0]['expected_layers'] != '28':
                raise ValueError(f'incomplete or duplicate source for {context}K B{batch}')
            old, new = matches[0], samples[0]
            common, lm = float(old['common_ms']), float(old['lm_head_ms'])
            dense_us = float(old['dense_fi_one_layer_us'])
            base = common + lm + 2 * dense_us / 1000
            head_us, group_us, custom_us = (float(new[k]) for k in ('fi_head_attention_us', 'fi_attention_us', 'custom_attention_us'))
            head, group, custom = (base + 30 * x / 1000 for x in (head_us, group_us, custom_us))
            rows.append({'context_k': context, 'batch': batch, 'common_ms': common, 'lm_head_ms': lm,
                         'dense_fi_one_layer_us': dense_us, 'sparse_layers': 30,
                         'head_one_layer_us': head_us, 'group_one_layer_us': group_us, 'custom_one_layer_us': custom_us,
                         'old_group_projection_ms': float(old['projected_step_ms']),
                         'paired_head_projection_ms': head, 'paired_group_projection_ms': group,
                         'custom_projection_ms': custom, 'group_minus_custom_ms': group - custom,
                         'group_over_custom': group / custom, 'head_over_custom': head / custom,
                         'historical_common_source_method': old['historical_common_source_method'],
                         'prior_source': str(prior_path), 'paired_source': str(paired_path),
                         'scope': 'disconnected component sum; selection/refinement/metadata excluded; mixed timing methods; not measured TPOT'})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {len(rows)} rows to {args.output}')


if __name__ == '__main__':
    main()
