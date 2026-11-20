#!/usr/bin/env python3
"""Freeze small experiment summaries used to audit the figure inputs."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = Path(__file__).resolve().parent / 'reference_results'
FILES = (
    'benchmark/RULER/fig4a_llama31/fig4a_summary.csv',
    'benchmark/RULER/fig4a_llama31/Fig4a_Plot_Data.md',
    'benchmark/RULER/fig4a_llama31/calibration.csv',
    'benchmark/RULER/fig4a_llama31/frozen_selection.json',
    'benchmark/RULER/fig4a_baselines/results/formal_paired_curve_expanded.csv',
    'benchmark/results_passkey_ours/fig4b_generality_8192/fig4b_summary.csv',
    'benchmark/results_efficiency/qasper_rate08_grouped_summary_20261113/summary.csv',
    'benchmark/results_efficiency/qasper_rate08_tpot_projection_20261113/projection.csv',
    'benchmark/results_efficiency/qasper_rate08_tpot_tuned_projection_20261113/projection.csv',
    'benchmark/results_efficiency/qasper_rate08_tpot_tuned_projection_20261113/dense_audit.csv',
    'benchmark/results_efficiency/qasper_rate08_tpot_tuned_projection_20261113/gas_custom_projection.csv',
    'benchmark/results_efficiency/qasper_gas_author_20261114/summary.csv',
    'benchmark/results_efficiency/qasper_gas_paired_20261114/summary.csv',
    'benchmark/results_efficiency/e7_estimated_tpot/VALIDITY_NOTICE.md',
    'benchmark/results_efficiency/e7_v1_simplified_projection/E7_v1_frontend_inclusive.csv',
    'benchmark/results_longbench/e5_longbench_completion/E5_LongBench_Task_Scores.csv',
    'benchmark/results_longbench/e5_longbench_completion/E5_LongBench_Summary.csv',
    'benchmark/results_longbench/e5_selector_matrix_ds50/E5_Selector_Matrix_Task_Scores.csv',
    'benchmark/results_longbench/e5_selector_matrix_ds50/E5_Selector_Matrix_Summary.csv',
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if DEST.exists():
        raise FileExistsError(f'{DEST} already exists; refusing to overwrite')
    missing = [name for name in FILES if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f'missing reference summaries: {missing}')
    entries = []
    for name in FILES:
        src, dst = ROOT / name, DEST / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        entries.append({'path': name, 'sha256': sha(dst), 'bytes': dst.stat().st_size})
    (DEST / 'manifest.json').write_text(json.dumps({'files': entries}, indent=2) + '\n')
    print(f'Snapshotted {len(entries)} reference summaries into {DEST}')


if __name__ == '__main__':
    main()
