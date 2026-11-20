#!/usr/bin/env python3
"""Extend the existing paired LessIsMore calibration without changing selection."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import run_lessismore as runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args_cli = parser.parse_args()
    results = Path(__file__).resolve().parent / "results"
    manifest = json.loads((results / "manifest.json").read_text())
    if (manifest["cal_samples"] != 3 or manifest["length"] != 32768
            or manifest["recent_ratio"] != 0.25 or manifest["selection_layers"] != [2, 13]
            or manifest["budgets"] != [128, 512]):
        raise ValueError("Existing experiment manifest does not match the expected pilot")
    args = SimpleNamespace(model=manifest["model"], reference=Path(manifest["reference"]),
                           output=results, cal_samples=manifest["cal_samples"],
                           formal_samples=manifest["formal_samples"])
    for budget in manifest["budgets"]:
        runner.method_stats(args, "calibration", budget)
    extras = (168, 256, 384)
    for budget in extras:
        for task in runner.TASKS:
            refs = runner.reference(args.reference, "calibration", task, args.cal_samples)
            path = results / "calibration" / f"lim_k{budget}" / f"{task}.jsonl"
            done = runner.validate(path, refs)
            count = len(runner.read_jsonl(path)) if path.exists() else 0
            print(f"{'COMPLETE' if done else 'PENDING'} k={budget} {task} {count}/{len(refs)}", flush=True)
    if args_cli.dry_run:
        print("Preflight passed; no GPU work was started.", flush=True)
        return
    for budget in extras:
        runner.execute(args, "calibration", budget)
    summarize = Path(__file__).resolve().parent / "summarize_paired_curve.py"
    subprocess.run([sys.executable, str(summarize), "--extra-budgets", *map(str, extras)], check=True)
    print("COMPLETE expanded calibration curves", flush=True)


if __name__ == "__main__":
    main()
