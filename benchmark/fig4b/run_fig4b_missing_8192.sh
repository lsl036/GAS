#!/usr/bin/env bash
set -u

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
PYTHON=${FIG4B_PYTHON:-/data/home/lsl/Software/miniconda3/envs/moba/bin/python}
OUTPUT_ROOT=${FIG4B_OUTPUT_DIR:-$REPO_ROOT/benchmark/results_passkey_ours/fig4b_generality_8192}
EXT_DIR=${TORCH_EXTENSIONS_DIR:-/data/tmp/torch_ext_ours_fig4b_8192}
FAILURE_LOG="$OUTPUT_ROOT/failed_cells.tsv"

mkdir -p "$OUTPUT_ROOT/logs"
if [[ ! -f "$FAILURE_LOG" ]]; then
  printf 'cell\tmodel\tfixed_length\tconfig\tlog\n' > "$FAILURE_LOG"
fi

declare -a CELLS=(
  "llama31_ds_16k|/data2/lsl_data/models/Meta-Llama-3.1-8B-Instruct|16000|$REPO_ROOT/benchmark/configs/config_ds_twi_gqa_group_max_p080_exception16_llama31.json"
  "qwen3_quest_16k|/data2/lsl_data/models/Qwen3-8B|16000|$REPO_ROOT/benchmark/configs/config_quest_twi_gqa_group_max_p080_exception16.json"
  "qwen3_ds_8k|/data2/lsl_data/models/Qwen3-8B|8000|$REPO_ROOT/benchmark/configs/config_ds_twi_gqa_group_max_p080_exception16_qwen3.json"
  "qwen3_ds_16k|/data2/lsl_data/models/Qwen3-8B|16000|$REPO_ROOT/benchmark/configs/config_ds_twi_gqa_group_max_p080_exception16_qwen3.json"
)

for cell in "${CELLS[@]}"; do
  IFS='|' read -r label model fixed_length config <<< "$cell"
  output_dir="$OUTPUT_ROOT/$label"
  profile_output="$output_dir/profile.jsonl"
  summary_output="$output_dir/profile_summary.json"
  log_path="$OUTPUT_ROOT/logs/${label}.log"
  if [[ -f "$profile_output" && -f "$summary_output" ]]; then
    echo "Skip complete $label"
    continue
  fi
  echo "Running $label; log=$log_path"
  if ! (
    cd "$REPO_ROOT"
    export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    export TORCH_EXTENSIONS_DIR="$EXT_DIR"
    "$PYTHON" benchmark/efficiency/profile_passkey_sparse_attention.py \
      --model "$model" \
      --fixed-length "$fixed_length" \
      --iterations 5 \
      --config "$config" \
      --output-dir "$output_dir" \
      --profile-output "$profile_output" \
      --summary-output "$summary_output" \
      --torch-extensions-dir "$EXT_DIR" \
      --skip-first-records 64
  ) > "$log_path" 2>&1; then
    printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$model" "$fixed_length" "$config" "$log_path" >> "$FAILURE_LOG"
    echo "Failed $label; continuing"
  fi
done

echo "Fig. 4(b) missing 8192-budget GAS cells finished."
