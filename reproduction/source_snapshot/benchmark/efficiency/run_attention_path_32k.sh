#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
PYTHON=${ATTN_PYTHON:-/data/home/lsl/Software/miniconda3/envs/moba/bin/python}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/data/tmp/torch_ext_ours_fig6}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
OUT=${ATTN_OUTPUT_DIR:-benchmark/results_efficiency/attention_path_32k_b64}
LABEL=attention_path_32k
mkdir -p "$OUT/logs" "$OUT/measurements"
source "$ROOT/benchmark/efficiency/fig6_gpu_guard.sh"
trace=benchmark/results_efficiency/fig6_traces/qasper32k_repeated_rate08_s54
for method in ${ATTN_METHODS:-sdpa_flash fi_dense quest quest_twi gas less}; do
    result="$OUT/measurements/$method.json"
    if [[ -e "$result" ]]; then
        echo "Existing result, refusing implicit reuse: $result" >&2
        exit 1
    fi
    echo "RUN $method"
    if run_guarded "$method" "$PYTHON" benchmark/efficiency/bench_attention_path_32k.py \
        --trace-dir "$trace" --output "$result" --method "$method" \
        --batch "${ATTN_BATCH:-64}" --repeats "${ATTN_REPEATS:-3}" --warmups "${ATTN_WARMUPS:-1}"; then
        echo "DONE $method"
    else
        echo "FAILED $method; see $OUT/logs/$method.log" >&2
        exit 1
    fi
done
echo "COMPLETE $OUT"
