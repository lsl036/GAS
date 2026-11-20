#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
PYTHON=${SDPA_PYTHON:-/data/home/lsl/Software/miniconda3/envs/moba/bin/python}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/data/tmp/torch_ext_ours_fig6}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
OUT=${SDPA_OUTPUT_DIR:-benchmark/results_efficiency/sdpa_fig6_audit}
LABEL=sdpa_fig6_audit
mkdir -p "$OUT/logs" "$OUT/measurements"
source "$ROOT/benchmark/efficiency/fig6_gpu_guard.sh"
for length in ${SDPA_LENGTHS:-16k 32k}; do
    case "$length" in
        16k) trace_prefix=benchmark/results_efficiency/fig6_traces/qasper16k_rate08_s ;;
        32k) trace_prefix=benchmark/results_efficiency/fig6_traces/qasper32k_repeated_rate08_s ;;
        *) echo "Unknown length: $length" >&2; exit 2 ;;
    esac
    for sample in ${SDPA_SAMPLES:-15 54}; do
      trace="${trace_prefix}${sample}"
      for batch in ${SDPA_BATCHES:-32 64}; do
        for method in ${SDPA_METHODS:-sdpa_flash fa2_kvcache gas_fi sdpa_efficient_expanded}; do
            label="${length}_s${sample}_b${batch}_${method}"
            result="$OUT/measurements/$label.json"
            if [[ -e "$result" ]]; then
                echo "SKIP existing $label; verify protocol before reusing"
                continue
            fi
            echo "RUN $label"
            if run_guarded "$label" "$PYTHON" benchmark/efficiency/bench_sdpa_fig6_audit.py \
                --trace-dir "$trace" --output "$result" --batch "$batch" --method "$method"; then
                echo "DONE $label"
            else
                if grep -q 'torch.OutOfMemoryError: CUDA out of memory' "$OUT/logs/$label.log"; then
                    echo "OOM $label; see $OUT/logs/$label.log" >&2
                    continue
                fi
                echo "FAILED $label; see $OUT/logs/$label.log" >&2
                exit 1
            fi
        done
      done
    done
done
echo "COMPLETE $OUT"
