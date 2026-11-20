#!/usr/bin/env bash
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${REAL_DECODE_PYTHON:-/data/home/lsl/Software/miniconda3/envs/moba/bin/python}
MODEL=${REAL_DECODE_MODEL:-/data2/lsl_data/models/Meta-Llama-3.1-8B-Instruct}
PROMPT=${REAL_DECODE_PROMPT:-$REPO/benchmark/RULER/scripts/dataset/llama3-8b-chat-128k/synthetic/16384/data/niah_single_1/validation.jsonl}
OUTPUT=${REAL_DECODE_OUTPUT:-$REPO/benchmark/results_efficiency/real_decode_ruler_one}
SAMPLE=${REAL_DECODE_SAMPLE:-0}
MAX_STEPS=${REAL_DECODE_MAX_STEPS:-126}

export TWILIGHT_PROFILE=0
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT/logs"
if [[ ! -f "$PROMPT" ]]; then
    printf 'Missing RULER input: %s\n' "$PROMPT" >&2
    exit 1
fi
status=0
for method in native_fa2 quest_twi gas gas_fused; do
    if [[ -f "$OUTPUT/$method/result.json" ]]; then
        printf 'SKIP existing result: %s\n' "$method"
        continue
    fi
    printf 'RUN %s\n' "$method"
    if "$PYTHON" "$REPO/benchmark/efficiency/bench_real_decode.py" \
        --model "$MODEL" --method "$method" --prompt-jsonl "$PROMPT" \
        --sample-index "$SAMPLE" --batch-size 1 --dtype fp16 --rounds 1 \
        --warmup-steps 1 --steps "$MAX_STEPS" --feeding greedy --stop-on-eos \
        --output-dir "$OUTPUT/$method" > "$OUTPUT/logs/$method.log" 2>&1; then
        printf 'DONE %s\n' "$method"
    else
        printf 'FAILED %s; see %s/logs/%s.log\n' "$method" "$OUTPUT" "$method"
        status=1
    fi
done
printf 'Finished all methods; result directory: %s\n' "$OUTPUT"
exit "$status"
