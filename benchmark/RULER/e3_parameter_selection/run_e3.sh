#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/data/home/lsl/SparseAtt/Twilight_Ours"
E3_ROOT="$REPO_ROOT/benchmark/RULER/e3_parameter_selection"
RULER_SCRIPTS="$REPO_ROOT/benchmark/RULER/scripts"
DATA_DIR="$REPO_ROOT/benchmark/RULER/runs_e2_20260921/llama3-8b-chat-128k/synthetic_e2/65536/data"
E2_A3_DIR="$REPO_ROOT/benchmark/RULER/runs_e2_20260921/llama3-8b-chat-128k/synthetic_e2/65536/pred/e2_A3"
MODEL_PATH="/data2/lsl_data/models/Meta-Llama-3.1-8B-Instruct"
PYTHON="${E3_PYTHON:-/data/home/lsl/Software/miniconda3/envs/moba/bin/python}"
EXPECTED_SAMPLES=50
DRY_RUN="${E3_DRY_RUN:-0}"
TASKS=(niah_multikey_3 niah_multivalue vt)
CONFIG_IDS=(P0-L P0-H)
CONFIG_PATHS=(
  "$E3_ROOT/configs/p0_l_pt080_pg080_m16.json"
  "$E3_ROOT/configs/p0_h_pt090_pg080_m16.json"
)

mkdir -p "$E3_ROOT/pred" "$E3_ROOT/logs" "$E3_ROOT/metadata" "$E3_ROOT/analysis"
if [[ "$DRY_RUN" == "1" ]]; then
  COMMAND_LOG="$E3_ROOT/metadata/dry_run_commands.log"
  : > "$COMMAND_LOG"
else
  COMMAND_LOG="$E3_ROOT/metadata/commands.log"
  printf '# run started %s\n' "$(date --iso-8601=seconds)" >> "$COMMAND_LOG"
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable is unavailable: $PYTHON" >&2
  exit 1
fi
if [[ ! -d "$DATA_DIR" || ! -d "$E2_A3_DIR" ]]; then
  echo "The frozen E2 manifest or A3 baseline is missing." >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

prediction_is_complete() {
  local manifest="$1"
  local prediction="$2"
  "$PYTHON" -c '
import json, sys
manifest_path, prediction_path, expected = sys.argv[1], sys.argv[2], int(sys.argv[3])
try:
    manifest = [json.loads(x) for x in open(manifest_path, encoding="utf-8") if x.strip()]
    prediction = [json.loads(x) for x in open(prediction_path, encoding="utf-8") if x.strip()]
except (FileNotFoundError, json.JSONDecodeError):
    raise SystemExit(1)
if len(manifest) != expected or len(prediction) != expected:
    raise SystemExit(1)
for source, result in zip(manifest, prediction):
    if source.get("index") != result.get("index") or source.get("input") != result.get("input"):
        raise SystemExit(1)
    if not isinstance(result.get("kv_metrics", {}).get("final_loaded_kv_tokens_mean"), (int, float)):
        raise SystemExit(1)
' "$manifest" "$prediction" "$EXPECTED_SAMPLES"
}

validate_method() {
  local method_dir="$1"
  local task
  for task in "${TASKS[@]}"; do
    prediction_is_complete \
      "$DATA_DIR/$task/validation.jsonl" \
      "$method_dir/$task.jsonl" || return 1
  done
}

# P0-D is the frozen E2 A3 result. Keep the original files read-only and expose
# them in the E3 layout through a symlink for paired analysis.
if ! validate_method "$E2_A3_DIR"; then
  echo "E2 A3 is incomplete or lacks synchronized KV metrics; refusing to run E3." >&2
  exit 1
fi
if [[ -e "$E3_ROOT/pred/P0-D" && ! -L "$E3_ROOT/pred/P0-D" ]]; then
  echo "$E3_ROOT/pred/P0-D exists and is not a symlink." >&2
  exit 1
fi
ln -sfn "$E2_A3_DIR" "$E3_ROOT/pred/P0-D"

# Verify that the P0-D snapshot is semantically identical to E2 A3.
"$PYTHON" -c '
import json, sys
left, right = (json.load(open(path, encoding="utf-8")) for path in sys.argv[1:])
if left != right:
    raise SystemExit("P0-D config differs from frozen E2 A3")
' \
  "$E3_ROOT/configs/p0_d_pt085_pg080_m16.json" \
  "$REPO_ROOT/benchmark/configs/e2_20260921/a3_group_max_p080_m16.json"

{
  echo "created_at=$(date --iso-8601=seconds)"
  echo "hostname=$(hostname)"
  echo "repo_root=$REPO_ROOT"
  echo "model_path=$MODEL_PATH"
  echo "data_dir=$DATA_DIR"
  echo "context_length=65536"
  echo "tasks=${TASKS[*]}"
  echo "expected_samples_per_task=$EXPECTED_SAMPLES"
  echo "temperature=0.0"
  echo "top_k=32"
  echo "top_p=1.0"
  echo "batch_size=1"
  echo "hf_internal_seed=43"
  echo "bootstrap_iterations=10000"
  echo "bootstrap_seed=20260929"
  echo "python=$PYTHON"
  echo "git_commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
} > "$E3_ROOT/metadata/run_metadata.txt"

git -C "$REPO_ROOT" status --short > "$E3_ROOT/metadata/git_status.txt"
if ! nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free \
  --format=csv,noheader > "$E3_ROOT/metadata/gpu.txt" 2>&1; then
  echo "nvidia-smi unavailable during metadata capture" \
    >> "$E3_ROOT/metadata/gpu.txt"
fi
"$PYTHON" -c '
import torch, transformers
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"transformers={transformers.__version__}")
' > "$E3_ROOT/metadata/python_packages.txt"

sha256sum \
  "$E3_ROOT"/configs/*.json \
  "$REPO_ROOT/benchmark/configs/e2_20260921/a3_group_max_p080_m16.json" \
  "$REPO_ROOT/gas_attention/pyimpl/attention.py" \
  "$REPO_ROOT/gas_attention/pyimpl/gqa_refine.py" \
  "$REPO_ROOT/benchmark/scripts/analyze_e3_results.py" \
  "$RULER_SCRIPTS/pred/call_api.py" \
  "$RULER_SCRIPTS/pred/model_wrappers.py" \
  "$RULER_SCRIPTS/eval/evaluate.py" \
  "$E3_ROOT/experiment_plan.json" \
  "$E3_ROOT/run_e3.sh" \
  "$MODEL_PATH/config.json" \
  > "$E3_ROOT/metadata/code_and_config_sha256.txt"

: > "$E3_ROOT/metadata/manifest_sha256.txt"
for task in "${TASKS[@]}"; do
  sha256sum "$DATA_DIR/$task/validation.jsonl" \
    >> "$E3_ROOT/metadata/manifest_sha256.txt"
done

cd "$RULER_SCRIPTS"
for index in "${!CONFIG_IDS[@]}"; do
  config_id="${CONFIG_IDS[$index]}"
  config_path="${CONFIG_PATHS[$index]}"
  method_dir="$E3_ROOT/pred/$config_id"
  mkdir -p "$method_dir"

  for task in "${TASKS[@]}"; do
    manifest="$DATA_DIR/$task/validation.jsonl"
    prediction="$method_dir/$task.jsonl"
    if prediction_is_complete "$manifest" "$prediction"; then
      echo "[$config_id] skip completed task: $task"
      continue
    fi

    export TWILIGHT_PROFILE=1
    export TWILIGHT_PROFILE_MODEL="$MODEL_PATH"
    export TWILIGHT_PROFILE_CONFIG="$config_path"
    export TWILIGHT_PROFILE_OUTPUT="$method_dir/$task.profile.jsonl"
    export TWILIGHT_PROFILE_RAW=0

    command=(
      "$PYTHON" pred/call_api.py
      --data_dir "$DATA_DIR"
      --save_dir "$method_dir"
      --benchmark synthetic
      --task "$task"
      --server_type hf
      --model_name_or_path "$MODEL_PATH"
      --temperature 0.0
      --top_k 32
      --top_p 1.0
      --random_seed 0
      --batch_size 1
      --algo_config_path "$config_path"
    )
    printf '%q ' "${command[@]}" >> "$COMMAND_LOG"
    printf '\n' >> "$COMMAND_LOG"
    if [[ "$DRY_RUN" == "1" ]]; then
      printf '[dry-run] '
      printf '%q ' "${command[@]}"
      printf '\n'
      continue
    fi
    "${command[@]}" 2>&1 | tee -a "$E3_ROOT/logs/${config_id}_${task}.log"
  done

  if [[ "$DRY_RUN" == "1" ]]; then
    continue
  fi
  if ! validate_method "$method_dir"; then
    echo "$config_id is incomplete after inference; analysis was not started." >&2
    exit 1
  fi
  "$PYTHON" eval/evaluate.py --data_dir "$method_dir" --benchmark synthetic \
    2>&1 | tee -a "$E3_ROOT/logs/${config_id}_evaluate.log"
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "E3 dry-run completed; no inference or analysis was executed."
  exit 0
fi

"$PYTHON" "$REPO_ROOT/benchmark/scripts/analyze_e3_results.py" \
  --experiment-dir "$E3_ROOT"

echo "E3-P0 completed. Summary: $E3_ROOT/analysis/E3_Result_Summary.md"
