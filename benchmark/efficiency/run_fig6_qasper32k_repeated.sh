#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
PYTHON=${FIG6_PYTHON:-/data/home/lsl/Software/miniconda3/envs/moba/bin/python}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/data/tmp/torch_ext_ours_fig6}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
source "$ROOT/benchmark/efficiency/fig6_gpu_guard.sh"

DATA=benchmark/results_efficiency/fig6_traces/qasper32k_repeated.jsonl
[[ -f "$DATA" && -f "$DATA.meta.json" ]] || { echo "Missing prepared repeated-context dataset" >&2; exit 1; }
"$PYTHON" -c '
import hashlib, json, sys
from pathlib import Path
data = Path(sys.argv[1])
meta = json.loads(Path(str(data) + ".meta.json").read_text())
if hashlib.sha256(data.read_bytes()).hexdigest() != meta["output_sha256"]:
    raise ValueError("Repeated Qasper dataset does not match its metadata")
' "$DATA"
for budget in fixed8192 rate08; do
    case "$budget" in
        fixed8192) config=benchmark/configs/config_quest_twi_gqa_group_max_p080_exception16.json ;;
        rate08) config=benchmark/configs/config_quest_twi_gas_ratio8_llama31.json ;;
    esac
    LABEL="qasper32k_repeated_${budget}"
    OUT="benchmark/results_efficiency/fig6_${LABEL}"
    mkdir -p "$OUT/logs"
    for sample in 15 54; do
        trace="benchmark/results_efficiency/fig6_traces/${LABEL}_s${sample}"
        if [[ -e "$trace" ]]; then
            "$PYTHON" -c '
import hashlib, json, sys
from pathlib import Path
trace, source, config, sample = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
manifest = json.loads((trace / "manifest.json").read_text())
row = [json.loads(line) for line in source.open(encoding="utf-8") if line.strip()][sample]
payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
expected_files = [f"step001_layer{i:02d}.pt" for i in range(2, 32)]
assert manifest["task"] == "qasper" and manifest["sample_index"] == sample
assert manifest["sample_sha256"] == hashlib.sha256(payload).hexdigest()
assert manifest["max_prompt_length_before_chat"] == 32768
assert manifest["config"] == json.loads(config.read_text())
assert manifest["files"] == expected_files and all((trace / f).is_file() for f in expected_files)
' "$trace" "$DATA" "$config" "$sample" || { echo "Incomplete or incompatible trace: $trace" >&2; exit 1; }
            echo "SKIP verified capture $LABEL sample=$sample"
            continue
        fi
        echo "RUN capture $LABEL sample=$sample"
        run_guarded "capture_s${sample}" "$PYTHON" benchmark/efficiency/capture_e6_qasper_trace.py \
            --dataset "$DATA" --sample-index "$sample" --max-prompt-length 32768 \
            --config "$config" --output-dir "$trace"
        echo "DONE capture $LABEL sample=$sample"
    done
    bash benchmark/efficiency/run_fig6_condition.sh "$LABEL" \
        "benchmark/results_efficiency/fig6_traces/${LABEL}_s15" \
        "benchmark/results_efficiency/fig6_traces/${LABEL}_s54"
done
"$PYTHON" benchmark/efficiency/summarize_fig6_qasper32k.py
