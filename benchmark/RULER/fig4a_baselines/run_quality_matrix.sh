#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/../../.." && pwd)
PYTHON=${LIM_PYTHON:-/data/home/lsl/Software/miniconda3/envs/moba/bin/python}
OUTPUT=${LIM_OUTPUT:-$ROOT/benchmark/results_quality/lessismore_13task_10}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT/logs"

"$PYTHON" "$HERE/run_quality_matrix.py" --suite preflight --output "$OUTPUT"
failed=0
for suite in ruler longbench; do
  for budget in 128 512; do
    log="$OUTPUT/logs/${suite}_k${budget}.log"
    printf 'RUN %s k=%s log=%s\n' "$suite" "$budget" "$log"
    if "$PYTHON" -u "$HERE/run_quality_matrix.py" --suite "$suite" --budget "$budget" --output "$OUTPUT" > "$log" 2>&1; then
      printf 'DONE %s k=%s\n' "$suite" "$budget"
    else
      printf 'INCOMPLETE %s k=%s; see %s\n' "$suite" "$budget" "$log" >&2
      failed=1
    fi
  done
done
"$PYTHON" "$HERE/run_quality_matrix.py" --suite summarize --output "$OUTPUT"
if (( failed )); then
  echo "INCOMPLETE: resolve errors, then rerun this command to resume." >&2
  exit 1
fi
echo "COMPLETE quality matrix: $OUTPUT"
