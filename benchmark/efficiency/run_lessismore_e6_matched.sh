#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/../.." && pwd)
PYTHON=${LIM_E6_PYTHON:-/data/home/lsl/Software/miniconda3/envs/moba/bin/python}
OUTPUT=${LIM_E6_OUTPUT:-$ROOT/benchmark/results_efficiency/lessismore_e6_matched}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -u "$HERE/bench_lessismore_e6_matched.py" --output-dir "$OUTPUT"
