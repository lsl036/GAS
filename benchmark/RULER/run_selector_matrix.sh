#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
RULER_SCRIPTS="$SCRIPT_DIR/scripts"
PYTHON=${E5_PYTHON:-/data/home/lsl/Software/miniconda3/envs/moba/bin/python}
ROOT_DIR=${E5_RULER_ROOT_DIR:-$RULER_SCRIPTS/dataset}
LENGTHS=${E5_RULER_LENGTHS:-"16384 32768 65536"}
NUM_SAMPLES=${NUM_SAMPLES:-50}
FAIL_FAST=${E5_FAIL_FAST:-0}
SUMMARY_OUTPUT=${E5_RULER_SUMMARY_OUTPUT:-$REPO_ROOT/benchmark/RULER/results_selector_matrix/RULER_Selector_Matrix_Summary.csv}

METHODS=(full_twi full_gas ds_4pct ds_8pct ds_twi ds_gas)
if [[ -n "${E5_METHODS:-}" ]]; then
  read -r -a METHODS <<< "$E5_METHODS"
fi

TASKS=(
  niah_single_1 niah_single_2 niah_single_3
  niah_multikey_1 niah_multikey_2 niah_multikey_3
  niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2
)
FAILURE_LOG=${E5_RULER_FAILURE_LOG:-$REPO_ROOT/benchmark/RULER/results_selector_matrix/failed_tasks.tsv}
mkdir -p "$(dirname "$FAILURE_LOG")" "$(dirname "$SUMMARY_OUTPUT")"
if [[ ! -f "$FAILURE_LOG" ]]; then
  printf 'method\tlength\ttask\treason\tlog\n' > "$FAILURE_LOG"
fi

declare -A CONFIGS=(
  [full_twi]="$REPO_ROOT/benchmark/configs/config_full_twi.json"
  [full_gas]="$REPO_ROOT/benchmark/configs/config_full_gas.json"
  [ds_4pct]="$REPO_ROOT/benchmark/configs/config_ds_4pct_llama31.json"
  [ds_8pct]="$REPO_ROOT/benchmark/configs/config_ds_8pct_llama31.json"
  [ds_twi]="$REPO_ROOT/benchmark/configs/config_ds_twi_llama31.json"
  [ds_gas]="$REPO_ROOT/benchmark/configs/config_ds_twi_gqa_group_max_p080_exception16_llama31.json"
)

for method in "${METHODS[@]}"; do
  if [[ -z "${CONFIGS[$method]+x}" ]]; then
    echo "Unknown RULER selector-matrix method: $method" >&2
    exit 2
  fi
  for length in $LENGTHS; do
    for task in "${TASKS[@]}"; do
      log_path="$REPO_ROOT/benchmark/RULER/results_selector_matrix/logs/${method}_${length}_${task}.log"
      mkdir -p "$(dirname "$log_path")"
      echo "Running RULER $method length=$length task=$task; log=$log_path"
      if ! (
        cd "$RULER_SCRIPTS"
        TWILIGHT_RULER_ROOT_DIR="$ROOT_DIR" \
        TWILIGHT_SEQ_LENGTHS="$length" \
        TWILIGHT_RULER_TASKS="$task" \
        NUM_SAMPLES="$NUM_SAMPLES" \
        bash run.sh "${CONFIGS[$method]}" "$method" synthetic
      ) > "$log_path" 2>&1; then
        printf '%s\t%s\t%s\trun_failed\t%s\n' "$method" "$length" "$task" "$log_path" >> "$FAILURE_LOG"
        echo "Failed RULER $method length=$length task=$task; recorded in $FAILURE_LOG"
        if [[ "$FAIL_FAST" == "1" ]]; then
          exit 1
        fi
        continue
      fi
      echo "Completed RULER $method length=$length task=$task"
    done
  done
done

"$PYTHON" "$REPO_ROOT/benchmark/scripts/summarize_ruler_selector_matrix.py" \
  --ruler-root "$ROOT_DIR" \
  --lengths $LENGTHS \
  --output "$SUMMARY_OUTPUT"
echo "RULER selector matrix complete: root=$ROOT_DIR"
