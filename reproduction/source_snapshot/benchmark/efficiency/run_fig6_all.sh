#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

if [[ $# -eq 0 ]]; then
    set -- qasper16k_fixed8192 qasper16k_rate08 ruler32k_fixed8192 ruler32k_rate08
fi
for label in "$@"; do
    case "$label" in
        qasper16k_fixed8192) left=qasper16k_fixed8192_s15; right=qasper16k_fixed8192_s54 ;;
        qasper16k_rate08) left=qasper16k_rate08_s15; right=qasper16k_rate08_s54 ;;
        ruler32k_fixed8192) left=ruler32k_fixed8192_s0; right=ruler32k_fixed8192_s1 ;;
        ruler32k_rate08) left=ruler32k_rate08_s0; right=ruler32k_rate08_s1 ;;
        *) printf 'Unknown condition: %s\n' "$label" >&2; exit 2 ;;
    esac
    bash benchmark/efficiency/run_fig6_condition.sh "$label" \
        "benchmark/results_efficiency/fig6_traces/$left" \
        "benchmark/results_efficiency/fig6_traces/$right"
done
"${FIG6_PYTHON:-/data/home/lsl/Software/miniconda3/envs/moba/bin/python}" benchmark/efficiency/summarize_fig6.py
