#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    printf 'Usage: %s LABEL TRACE_DIR_1 TRACE_DIR_2\n' "$0" >&2
    exit 2
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
PYTHON=${FIG6_PYTHON:-/data/home/lsl/Software/miniconda3/envs/moba/bin/python}
LABEL=$1
TRACE1=$2
TRACE2=$3
OUT="benchmark/results_efficiency/fig6_${LABEL}"
mkdir -p "$OUT/logs"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/data/tmp/torch_ext_ours_fig6}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
source "$ROOT/benchmark/efficiency/fig6_gpu_guard.sh"

"$PYTHON" benchmark/efficiency/calibrate_less_qasper_budget.py \
    --trace-dirs "$TRACE1" "$TRACE2" --output "$OUT/less_calibration.json"
K=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["K"])' "$OUT/less_calibration.json")
echo "RUN $LABEL K=$K"

COMMON=(--trace-dirs "$TRACE1" "$TRACE2" --steps 1 --batch-sizes 32 64 96 128 \
    --warmups 3 --repeats 10 --rounds 2 --graph-iterations 8 --reference-batch-chunk 2)
LAYERS=($(seq 2 31))

if [[ -f "$OUT/main/results.csv" ]]; then
    "$PYTHON" benchmark/efficiency/verify_fig6_main.py --output-dir "$OUT/main" \
        --trace-dirs "$TRACE1" "$TRACE2"
    echo "SKIP verified $LABEL dense-fa2-and-sparse"
else
    echo "RUN $LABEL dense-fa2-and-sparse"
    run_guarded main "$PYTHON" benchmark/efficiency/bench_gas_mixed_trace.py "${COMMON[@]}" \
        --output-dir "$OUT/main" --layers "${LAYERS[@]}" --dtype float16 \
        --methods fa2_dense fi_quest_adapter fi_twilight_adapter fi_gas_adapter inline8
    echo "DONE $LABEL dense-fa2-and-sparse"
fi

if [[ -f "$OUT/fi_dense/results.csv" ]]; then
    "$PYTHON" benchmark/efficiency/verify_fig6_phase.py --output-dir "$OUT/fi_dense" \
        --trace-dirs "$TRACE1" "$TRACE2" --method fi_dense
    echo "SKIP verified $LABEL dense-flashinfer"
else
    echo "RUN $LABEL dense-flashinfer"
    run_guarded fi_dense "$PYTHON" benchmark/efficiency/bench_gas_mixed_trace.py "${COMMON[@]}" \
        --output-dir "$OUT/fi_dense" --layers "${LAYERS[@]}" --dtype float16 --methods fi_dense
    echo "DONE $LABEL dense-flashinfer"
fi

for method in less_fi_adapter less_patched_headpacked; do
    if [[ -f "$OUT/$method/results.csv" ]]; then
        "$PYTHON" benchmark/efficiency/verify_fig6_phase.py --output-dir "$OUT/$method" \
            --trace-dirs "$TRACE1" "$TRACE2" --method "$method"
        echo "SKIP verified $LABEL $method"
        continue
    fi
    script=benchmark/efficiency/bench_less_qasper_layers.py
    if [[ "$method" == less_patched_headpacked ]]; then
        script=benchmark/efficiency/bench_less_qasper_layers_v2.py
    fi
    echo "RUN $LABEL $method"
    run_guarded "$method" "$PYTHON" "$script" "${COMMON[@]}" \
        --output-dir "$OUT/$method" --budget "$K" --budget-calibration "$OUT/less_calibration.json" \
        --methods "$method"
    echo "DONE $LABEL $method"
done
echo "COMPLETE $LABEL $OUT"
