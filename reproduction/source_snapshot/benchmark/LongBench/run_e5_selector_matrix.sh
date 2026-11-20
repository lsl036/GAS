#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
PYTHON=${E5_PYTHON:-/data/home/lsl/Software/miniconda3/envs/moba/bin/python}
DATA_DIR=${E5_DATA_DIR:-/data/home/lsl/SparseAtt/data/LongBench/data}
MODEL_PATH=${E5_MODEL_PATH:-/data2/lsl_data/models/Meta-Llama-3.1-8B-Instruct}
OUTPUT_DIR=${E5_OUTPUT_DIR:-$REPO_ROOT/benchmark/results_longbench/e5_selector_matrix}
CONFIG_DIR=$REPO_ROOT/benchmark/configs
LONGBENCH_CONFIG=$REPO_ROOT/benchmark/LongBench/config
FAIL_FAST=${E5_FAIL_FAST:-0}
PREP_SAMPLE_ARGS=()
PRED_SAMPLE_ARGS=()
if [[ -n "${E5_SAMPLE_LIMIT:-}" ]]; then
  PREP_SAMPLE_ARGS=(--samples-per-task "$E5_SAMPLE_LIMIT")
  PRED_SAMPLE_ARGS=(--sample-limit "$E5_SAMPLE_LIMIT")
fi

TASKS=(
  narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique
  gov_report qmsum multi_news triviaqa passage_retrieval_en lcc repobench
)
METHODS=(full_twi full_gas ds_4pct ds_8pct ds_twi ds_gas)
if [[ -n "${E5_METHODS:-}" ]]; then
  read -r -a METHODS <<< "$E5_METHODS"
fi
if [[ -n "${E5_TASKS:-}" ]]; then
  read -r -a TASKS <<< "$E5_TASKS"
fi

declare -A CONFIGS=(
  [full_twi]="$CONFIG_DIR/config_full_twi.json"
  [full_gas]="$CONFIG_DIR/config_full_gas.json"
  [ds_4pct]="$CONFIG_DIR/config_ds_4pct_llama31.json"
  [ds_8pct]="$CONFIG_DIR/config_ds_8pct_llama31.json"
  [ds_twi]="$CONFIG_DIR/config_ds_twi_llama31.json"
  [ds_gas]="$CONFIG_DIR/config_ds_twi_gqa_group_max_p080_exception16_llama31.json"
)

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/work" "$OUTPUT_DIR/checkpoints"
FAILURE_LOG="$OUTPUT_DIR/failed_tasks.tsv"
if [[ ! -f "$FAILURE_LOG" ]]; then
  printf 'method\ttask\treason\tlog\n' > "$FAILURE_LOG"
fi
cd "$REPO_ROOT"

"$PYTHON" benchmark/LongBench/prepare_e5_selector_matrix.py \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --model-path "$MODEL_PATH" \
  "${PREP_SAMPLE_ARGS[@]}"

for method in "${METHODS[@]}"; do
  if [[ -z "${CONFIGS[$method]+x}" ]]; then
    echo "Unknown E5 selector-matrix method: $method" >&2
    exit 2
  fi
  for task in "${TASKS[@]}"; do
    dataset_key=$task
    [[ "$task" == "repobench" ]] && dataset_key=repobench-p
    final_path="$OUTPUT_DIR/pred/$method/$task/predictions.jsonl"
    if "$PYTHON" benchmark/LongBench/verify_e5_prediction.py \
      --prediction "$final_path" \
      --manifest "$OUTPUT_DIR/manifest.jsonl" \
      --task "$task"; then
      echo "Skip complete $method/$task"
      continue
    fi

    work_dir="$OUTPUT_DIR/work/$method/$task"
    checkpoint="$OUTPUT_DIR/checkpoints/$method/$task.jsonl"
    log_path="$OUTPUT_DIR/logs/${method}_${task}.log"
    mkdir -p "$work_dir" "$(dirname "$final_path")"
    echo "Running $method/$task; log=$log_path"
    if ! "$PYTHON" benchmark/LongBench/pred.py \
        --model Meta-Llama-3.1-8B-Instruct \
        --model_path "$MODEL_PATH" \
        --task "$dataset_key" \
        --algo-config-path "${CONFIGS[$method]}" \
        --e 0 \
        --t "e5_matrix_${method}_${task}" \
        --dataset-path "$DATA_DIR" \
        --output-dir "$work_dir" \
        --config-path "$LONGBENCH_CONFIG" \
        --checkpoint-path "$checkpoint" \
        --checkpoint-task-name "$task" \
        "${PRED_SAMPLE_ARGS[@]}" \
        > "$log_path" 2>&1; then
      printf '%s\t%s\tprediction_failed\t%s\n' "$method" "$task" "$log_path" >> "$FAILURE_LOG"
      echo "Failed $method/$task; recorded in $FAILURE_LOG"
      if [[ "$FAIL_FAST" == "1" ]]; then
        exit 1
      fi
      continue
    fi

    mapfile -t candidates < <(find "$work_dir/pred" -maxdepth 1 -type f -name "${dataset_key%%-*}-*.jsonl" -print)
    if [[ ${#candidates[@]} -ne 1 ]]; then
      printf '%s\t%s\tprediction_file_count_%s\t%s\n' "$method" "$task" "${#candidates[@]}" "$log_path" >> "$FAILURE_LOG"
      echo "Expected one generated prediction for $method/$task, found ${#candidates[@]}" >&2
      if [[ "$FAIL_FAST" == "1" ]]; then
        exit 1
      fi
      continue
    fi
    cp "${candidates[0]}" "$final_path"
    if ! "$PYTHON" benchmark/LongBench/verify_e5_prediction.py \
        --prediction "$final_path" \
        --manifest "$OUTPUT_DIR/manifest.jsonl" \
        --task "$task"; then
      printf '%s\t%s\tprediction_verification_failed\t%s\n' "$method" "$task" "$log_path" >> "$FAILURE_LOG"
      echo "Verification failed for $method/$task; recorded in $FAILURE_LOG"
      if [[ "$FAIL_FAST" == "1" ]]; then
        exit 1
      fi
      continue
    fi
    echo "Completed $method/$task"
  done
done

"$PYTHON" benchmark/LongBench/summarize_e5_selector_matrix.py \
  --output-dir "$OUTPUT_DIR"
echo "E5 selector matrix complete: $OUTPUT_DIR"
