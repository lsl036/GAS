#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
PYTHON=${FIG4A_PYTHON:-/data/home/lsl/Software/miniconda3/envs/moba/bin/python}
exec "$PYTHON" "$ROOT/benchmark/RULER/fig4a/run_fig4a.py" \
  --model-path /data2/lsl_data/models/Meta-Llama-3.1-8B-Instruct \
  --model-name llama3-8b-chat-128k \
  --selector quest \
  --channel-config "$ROOT/benchmark/configs/ds_channel_llama31_q.json" \
  --calibration-data-root "$ROOT/benchmark/RULER/runs_e1_calibration_v2/llama3-8b-chat-128k/synthetic_e1/32768/data" \
  --formal-data-root "$ROOT/benchmark/RULER/runs_e1_main/llama3-8b-chat-128k/synthetic_e1/32768/data" \
  --output-dir "${FIG4A_OUTPUT_DIR:-$ROOT/benchmark/RULER/fig4a_llama31}"
