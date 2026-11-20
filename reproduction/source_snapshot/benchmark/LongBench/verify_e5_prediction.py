#!/usr/bin/env python3
"""Return success only when one E5 method/task prediction is complete."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()

    expected = {}
    with args.manifest.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["task"] == args.task:
                expected[row["sample_id"]] = row["sample_hash"]
    if not expected or not args.prediction.is_file():
        raise SystemExit(1)

    seen = set()
    with args.prediction.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row.get("_e5_sample_id")
            if sample_id in seen or sample_id not in expected:
                raise SystemExit(1)
            if row.get("_e5_sample_hash") != expected[sample_id]:
                raise SystemExit(1)
            seen.add(sample_id)
    if seen != set(expected):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
