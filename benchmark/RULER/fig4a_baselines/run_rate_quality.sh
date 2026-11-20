#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/../../.." && pwd)
PYTHON=${LIM_PYTHON:-/data/home/lsl/Software/miniconda3/envs/moba/bin/python}
OUTPUT=${LIM_RATE_OUTPUT:-$ROOT/benchmark/results_quality/lessismore_rate_13task_10}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT/logs"

"$PYTHON" "$HERE/run_rate_quality.py" --suite preflight --output "$OUTPUT"
failed=0
for suite in ruler longbench; do
  for rate in 4pct 8pct; do
    log="$OUTPUT/logs/${suite}_${rate}.log"
    printf 'RUN %s rate=%s log=%s\n' "$suite" "$rate" "$log"
    if "$PYTHON" -u "$HERE/run_rate_quality.py" --suite "$suite" --rate "$rate" --output "$OUTPUT" > "$log" 2>&1; then
      printf 'DONE %s rate=%s\n' "$suite" "$rate"
    else
      printf 'INCOMPLETE %s rate=%s; see %s\n' "$suite" "$rate" "$log" >&2
      failed=1
    fi
  done
done
"$PYTHON" "$HERE/run_rate_quality.py" --suite summarize --output "$OUTPUT"
if (( failed )); then
  echo "INCOMPLETE: resolve errors, then rerun this command to resume." >&2
  exit 1
fi
echo "COMPLETE dynamic-rate quality matrix: $OUTPUT"
