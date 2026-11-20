#!/usr/bin/env bash
set -u

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
PYTHON=${FIG4B_PYTHON:-/data/home/lsl/Software/miniconda3/envs/moba/bin/python}
OUTPUT_DIR=${FIG4B_OUTPUT_DIR:-$REPO_ROOT/benchmark/results_passkey_ours/fig4b_generality}
MANIFEST=$OUTPUT_DIR/case_manifest.json
FAILURE_LOG=$OUTPUT_DIR/failed_cells.tsv
ONLY=${FIG4B_ONLY:-}

mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/prompts"
if [[ ! -f "$MANIFEST" ]]; then
  "$PYTHON" benchmark/fig4b/generate_manifest.py --output "$MANIFEST" --seed 1337
fi
if [[ ! -f "$FAILURE_LOG" ]]; then
  printf 'cell\tmodel\tcontext\tstrategy\tlog\n' > "$FAILURE_LOG"
fi

declare -a CELLS=(
  "llama31_quest_8k|/data2/lsl_data/models/Meta-Llama-3.1-8B-Instruct|8192|headwise|benchmark/configs/config_quest_twi_ratio8_llama31.json"
  "llama31_quest_8k|/data2/lsl_data/models/Meta-Llama-3.1-8B-Instruct|8192|gas|benchmark/configs/config_quest_twi_gas_ratio8_llama31.json"
  "llama31_quest_16k|/data2/lsl_data/models/Meta-Llama-3.1-8B-Instruct|16384|headwise|benchmark/configs/config_quest_twi_ratio8_llama31.json"
  "llama31_quest_16k|/data2/lsl_data/models/Meta-Llama-3.1-8B-Instruct|16384|gas|benchmark/configs/config_quest_twi_gas_ratio8_llama31.json"
  "llama31_ds_8k|/data2/lsl_data/models/Meta-Llama-3.1-8B-Instruct|8192|headwise|benchmark/configs/config_ds_twi_ratio8_llama31.json"
  "llama31_ds_8k|/data2/lsl_data/models/Meta-Llama-3.1-8B-Instruct|8192|gas|benchmark/configs/config_ds_gas_ratio8_llama31.json"
  "llama31_ds_16k|/data2/lsl_data/models/Meta-Llama-3.1-8B-Instruct|16384|headwise|benchmark/configs/config_ds_twi_ratio8_llama31.json"
  "llama31_ds_16k|/data2/lsl_data/models/Meta-Llama-3.1-8B-Instruct|16384|gas|benchmark/configs/config_ds_gas_ratio8_llama31.json"
  "qwen3_quest_8k|/data2/lsl_data/models/Qwen3-8B|8192|headwise|benchmark/configs/config_quest_twi_ratio8_qwen3.json"
  "qwen3_quest_8k|/data2/lsl_data/models/Qwen3-8B|8192|gas|benchmark/configs/config_quest_twi_gas_ratio8_qwen3.json"
  "qwen3_quest_16k|/data2/lsl_data/models/Qwen3-8B|16384|headwise|benchmark/configs/config_quest_twi_ratio8_qwen3.json"
  "qwen3_quest_16k|/data2/lsl_data/models/Qwen3-8B|16384|gas|benchmark/configs/config_quest_twi_gas_ratio8_qwen3.json"
  "qwen3_ds_8k|/data2/lsl_data/models/Qwen3-8B|8192|headwise|benchmark/configs/config_ds_twi_ratio8_qwen3.json"
  "qwen3_ds_8k|/data2/lsl_data/models/Qwen3-8B|8192|gas|benchmark/configs/config_ds_gas_ratio8_qwen3.json"
  "qwen3_ds_16k|/data2/lsl_data/models/Qwen3-8B|16384|headwise|benchmark/configs/config_ds_twi_ratio8_qwen3.json"
  "qwen3_ds_16k|/data2/lsl_data/models/Qwen3-8B|16384|gas|benchmark/configs/config_ds_gas_ratio8_qwen3.json"
)

for cell in "${CELLS[@]}"; do
  IFS='|' read -r label model context strategy config <<< "$cell"
  cell_id="${label}_${strategy}"
  if [[ -n "$ONLY" && " $ONLY " != *" $cell_id "* ]]; then
    continue
  fi
  cell_dir="$OUTPUT_DIR/$cell_id"
  log_path="$OUTPUT_DIR/logs/${cell_id}.log"
  if [[ -f "$cell_dir/predictions.jsonl" && -f "$cell_dir/profile.jsonl" && -f "$cell_dir/metadata.json" ]]; then
    echo "Skip complete $cell_id"
    continue
  fi
  echo "Running $cell_id; log=$log_path"
  if ! (
    cd "$REPO_ROOT"
    export TWILIGHT_PROFILE=1
    export TWILIGHT_PROFILE_RAW=0
    export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    "$PYTHON" benchmark/fig4b/run_passkey_case.py \
      --model "$model" \
      --model-label "$label" \
      --config "$REPO_ROOT/$config" \
      --manifest "$MANIFEST" \
      --prompt-cache "$OUTPUT_DIR/prompts/${label}.json" \
      --target-tokens "$context" \
      --output-dir "$cell_dir" \
      --max-new-tokens 64 \
      --seed 1337
  ) > "$log_path" 2>&1; then
    printf '%s\t%s\t%s\t%s\t%s\n' "$cell_id" "$model" "$context" "$strategy" "$log_path" >> "$FAILURE_LOG"
    echo "Failed $cell_id; continuing"
  fi
done

echo "Fig. 4(b) matrix runner finished. Run summarize_fig4b.py after all cells complete."
