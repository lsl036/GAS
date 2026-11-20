#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
PYTHON=${QUEST_AUDIT_PYTHON:-/data/home/lsl/Software/miniconda3/envs/moba/bin/python}
OUTPUT_ROOT=${QUEST_AUDIT_OUTPUT_DIR:-$REPO_ROOT/benchmark/results_efficiency/quest_budget_audit_v2_formal}

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/data/tmp/torch_ext_ours_quest_audit}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Sequential shapes, independent buffers, and the unchanged 8192 upper budget.
for shape in 16384:32 32768:128; do
    length=${shape%:*}
    batch=${shape#*:}
    "$PYTHON" benchmark/efficiency/bench_quest_budget_sweep.py \
        --output-dir "$OUTPUT_ROOT/L${length}_B${batch}" \
        --length "$length" --batch-size "$batch" \
        --budgets 512 1024 2048 4096 8192 \
        --warmups 20 --repeats 100 --rounds 5 --profile-kernels
done
printf 'Quest fixed-support audit complete: %s\n' "$OUTPUT_ROOT"
