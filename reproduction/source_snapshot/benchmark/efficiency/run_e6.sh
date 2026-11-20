#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
PYTHON=${E6_PYTHON:-/data/home/lsl/Software/miniconda3/envs/moba/bin/python}
OUTPUT_DIR=${E6_OUTPUT_DIR:-$REPO_ROOT/benchmark/results_efficiency/e6_kernel_baselines}

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT_DIR/logs"

exec "$PYTHON" benchmark/efficiency/bench_e6_kernel_baselines.py \
  --output-dir "$OUTPUT_DIR"
