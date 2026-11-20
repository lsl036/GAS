#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
PYTHON=${E5_PYTHON:-/data/home/lsl/Software/miniconda3/envs/moba/bin/python}
DATA_DIR=${E5_DATA_DIR:-/data/home/lsl/SparseAtt/data/LongBench/data}
MODEL_PATH=${E5_MODEL_PATH:-/data2/lsl_data/models/Meta-Llama-3.1-8B-Instruct}
OUTPUT_DIR=${E5_OUTPUT_DIR:-$REPO_ROOT/benchmark/results_longbench/e5_longbench_completion}
CONFIG_DIR=$REPO_ROOT/benchmark/configs
LONGBENCH_CONFIG=$REPO_ROOT/benchmark/LongBench/config
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  cat <<EOF
E5 LongBench completion runner

Runs the four missing methods sequentially on the frozen 13-task manifest,
while reusing the normalized Quest+Twilight and GAS group-max predictions.
The default output is:
  $OUTPUT_DIR

Environment overrides: E5_PYTHON, E5_DATA_DIR, E5_MODEL_PATH, E5_OUTPUT_DIR.
EOF
  exit 0
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "E5 Python is not executable: $PYTHON" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/work" "$OUTPUT_DIR/checkpoints"
cd "$REPO_ROOT"

"$PYTHON" benchmark/LongBench/prepare_e5_longbench.py \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --model-path "$MODEL_PATH"

TASKS=(
  narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique
  gov_report qmsum multi_news triviaqa passage_retrieval_en lcc repobench
)
METHODS=(full quest_4pct quest_8pct prob_sum)

declare -A CONFIGS=(
  [full]="$CONFIG_DIR/config_full.json"
  [quest_4pct]="$CONFIG_DIR/config_quest_4pct.json"
  [quest_8pct]="$CONFIG_DIR/config_quest_8pct.json"
  [prob_sum]="$CONFIG_DIR/config_quest_twi_gqa_prob_sum_p080_exception16.json"
)

for method in "${METHODS[@]}"; do
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

    "$PYTHON" benchmark/LongBench/pred.py \
      --model Meta-Llama-3.1-8B-Instruct \
      --model_path "$MODEL_PATH" \
      --task "$dataset_key" \
      --algo-config-path "${CONFIGS[$method]}" \
      --e 0 \
      --t "e5_${method}_${task}" \
      --dataset-path "$DATA_DIR" \
      --output-dir "$work_dir" \
      --config-path "$LONGBENCH_CONFIG" \
      --checkpoint-path "$checkpoint" \
      --checkpoint-task-name "$task" \
      > "$log_path" 2>&1

    mapfile -t candidates < <(find "$work_dir/pred" -maxdepth 1 -type f -name "${dataset_key%%-*}-*.jsonl" -print)
    if [[ ${#candidates[@]} -ne 1 ]]; then
      echo "Expected one generated prediction for $method/$task, found ${#candidates[@]}" >&2
      exit 1
    fi
    cp "${candidates[0]}" "$final_path"
    "$PYTHON" benchmark/LongBench/verify_e5_prediction.py \
      --prediction "$final_path" \
      --manifest "$OUTPUT_DIR/manifest.jsonl" \
      --task "$task"
    echo "Completed $method/$task"
  done
done

"$PYTHON" benchmark/LongBench/summarize_e5_longbench.py \
  --output-dir "$OUTPUT_DIR"

echo "E5 complete: $OUTPUT_DIR"
