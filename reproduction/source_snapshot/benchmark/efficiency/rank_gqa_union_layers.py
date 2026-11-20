#!/usr/bin/env python3
"""Rank captured Qasper layers by Twilight/GAS group-union reduction."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'benchmark/results_efficiency'
OUT = RESULTS / 'gas_compact_pilot_20261113/union_layer_ranking.csv'


def rate08(path):
    data = {}
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['backend'] == 'grouped' and row['method'] in ('twilight', 'gas'):
                data.setdefault(int(row['layer']), {})[row['method']] = float(row['group_union_mean'])
    return [(layer, values['twilight'], values['gas']) for layer, values in sorted(data.items())
            if 'twilight' in values and 'gas' in values]


def fixed(path):
    data = {}
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['step'] == '1' and row['batch'] == '32':
                pair = (float(row['twi_group_union_mean']), float(row['gas_group_union_mean']))
                layer = int(row['layer'])
                if layer in data and data[layer] != pair:
                    raise ValueError(f'inconsistent masks for layer {layer}')
                data[layer] = pair
    return [(layer, pair[0], pair[1]) for layer, pair in sorted(data.items())]


def main():
    cases = [
        ('rate08', '16k', 'two original Qasper samples', rate08,
         RESULTS / 'qasper_rate08_grouped_16k_b32_20261113/results.csv'),
        ('rate08', '32k', 'same samples with repeated context', rate08,
         RESULTS / 'qasper_rate08_grouped_32k_b32_20261113/results.csv'),
        ('fixed8192', '16k', 'two original Qasper samples', fixed,
         RESULTS / 'fig6_qasper16k_fixed8192/main/results.csv'),
        ('fixed8192', '32k', 'same samples with repeated context', fixed,
         RESULTS / 'fig6_qasper32k_repeated_fixed8192/main/results.csv'),
    ]
    rows = []
    for scenario, length, workload, loader, path in cases:
        for layer, twi, gas in loader(path):
            rows.append({'scenario': scenario, 'length': length, 'workload': workload,
                         'layer': layer, 'twilight_union_mean': twi, 'gas_union_mean': gas,
                         'ratio_of_means': twi / gas if gas else '', 'absolute_reduction': twi - gas,
                         'source': str(path)})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {len(rows)} layer records to {OUT}')


if __name__ == '__main__':
    main()
