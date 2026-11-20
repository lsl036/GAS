#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
PYTHON=${SDPA_PYTHON:-/data/home/lsl/Software/miniconda3/envs/moba/bin/python}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/data/tmp/torch_ext_ours_fig6}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
OUT=${SDPA_B1_OUTPUT_DIR:-benchmark/results_efficiency/sdpa_official_b1_audit}
LABEL=sdpa_official_b1_audit
mkdir -p "$OUT/logs" "$OUT/measurements"
source "$ROOT/benchmark/efficiency/fig6_gpu_guard.sh"
for length in ${SDPA_LENGTHS:-16k 32k}; do
    case "$length" in
        16k) prefix=benchmark/results_efficiency/fig6_traces/qasper16k_rate08_s; less_k=277 ;;
        32k) prefix=benchmark/results_efficiency/fig6_traces/qasper32k_repeated_rate08_s; less_k=468 ;;
        *) echo "Unknown length: $length" >&2; exit 2 ;;
    esac
    for sample in ${SDPA_SAMPLES:-15 54}; do
        trace="${prefix}${sample}"
        for method in ${SDPA_B1_METHODS:-sdpa_flash fa2_kvcache gas_fi sdpa_efficient_expanded quest_official_b1 less_official_b1}; do
            label="${length}_s${sample}_b1_${method}"
            result="$OUT/measurements/$label.json"
            if [[ -e "$result" ]]; then
                echo "SKIP existing $label; verify protocol before reusing"
                continue
            fi
            echo "RUN $label"
            case "$method" in
                quest_official_b1|less_official_b1)
                    command=("$PYTHON" benchmark/efficiency/bench_official_sdpa_b1_audit.py --trace-dir "$trace" --output "$result" --less-k "$less_k" --method "$method") ;;
                sdpa_flash|fa2_kvcache|gas_fi|sdpa_efficient_expanded)
                    command=("$PYTHON" benchmark/efficiency/bench_sdpa_fig6_audit.py --trace-dir "$trace" --output "$result" --batch 1 --timing direct --method "$method") ;;
                *) echo "Unknown method: $method" >&2; exit 2 ;;
            esac
            if run_guarded "$label" "${command[@]}"; then
                echo "DONE $label"
            else
                echo "FAILED $label; see $OUT/logs/$label.log" >&2
                exit 1
            fi
        done
    done
done
echo "COMPLETE $OUT"
