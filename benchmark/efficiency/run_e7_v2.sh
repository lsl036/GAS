#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
PYTHON=${E7_PYTHON:-/data/home/lsl/Software/miniconda3/envs/moba/bin/python}
OUTPUT=${E7_OUTPUT_DIR:-$REPO_ROOT/benchmark/results_efficiency/e7_v2_connected}
TRACES=${E7_TRACE_DIR:-$OUTPUT/traces}

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/data/tmp/torch_ext_ours_e7_v2}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUTPUT/logs"

"$PYTHON" benchmark/efficiency/prepare_e7_v2_traces.py --output-dir "$TRACES"
"$PYTHON" benchmark/efficiency/bench_e7_v2_estimated_tpot.py \
    --trace-dir "$TRACES" --output-dir "$OUTPUT" "$@"
"$PYTHON" benchmark/efficiency/summarize_e7_v2.py --output-dir "$OUTPUT"
printf 'E7 v2 measurement complete; check all 40 statuses before plotting: %s\n' "$OUTPUT"
