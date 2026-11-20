#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
PYTHON=${FIG4A_PYTHON:-/data/home/lsl/Software/miniconda3/envs/moba/bin/python}
exec "$PYTHON" "$ROOT/benchmark/RULER/fig4a/run_fig4a.py" \
  --model-path /data2/lsl_data/models/Qwen3-8B \
  --model-name qwen3-8b \
  --selector quest \
  --channel-config "$ROOT/benchmark/configs/ds_channel_qwen3_q.json" \
  --skip-calibration \
  --output-dir "${FIG4A_OUTPUT_DIR:-$ROOT/benchmark/RULER/fig4a_qwen3}"
