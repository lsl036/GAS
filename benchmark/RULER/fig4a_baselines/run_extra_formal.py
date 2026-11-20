#!/usr/bin/env python3
"""Evaluate additional LessIsMore budgets on the existing paired formal set."""

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import run_lessismore as runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args()
    results = Path(__file__).resolve().parent / "results"
    manifest = json.loads((results / "manifest.json").read_text())
    if (manifest["formal_samples"] != 10 or manifest["length"] != 32768
            or manifest["recent_ratio"] != 0.25 or manifest["selection_layers"] != [2, 13]
            or manifest["budgets"] != [128, 512]):
        raise ValueError("Existing experiment manifest does not match the expected pilot")
    args = SimpleNamespace(model=manifest["model"], reference=Path(manifest["reference"]),
                           output=results, cal_samples=manifest["cal_samples"],
                           formal_samples=manifest["formal_samples"])
    runner.method_stats(args, "formal")
    runner.method_stats(args, "formal", 128)
    extras = (168, 256, 384, 512)
    for budget in extras:
        for task in runner.TASKS:
            refs = runner.reference(args.reference, "formal", task, args.formal_samples)
            path = results / "formal" / f"lim_k{budget}" / f"{task}.jsonl"
            done = runner.validate(path, refs)
            count = len(runner.read_jsonl(path)) if path.exists() else 0
            print(f"{'COMPLETE' if done else 'PENDING'} k={budget} {task} {count}/{len(refs)}", flush=True)
    if options.dry_run:
        print("Preflight passed; no GPU work was started.", flush=True)
        return

    for budget in extras:
        runner.execute(args, "formal", budget)

    gas = runner.method_stats(args, "formal")
    gas_by_task = {row["task"]: row for row in gas}
    gas_score = runner.mean([row["score"] for row in gas])
    gas_kv = runner.mean([row["loaded_kv"] for row in gas])
    output = results / "formal_paired_curve_expanded.csv"
    temporary = output.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        fields = ["method", "budget", "split", "task", "samples", "score", "mean_loaded_kv",
                  "kv_ratio_to_gas", "score_delta_vs_gas", "sparse_only_kv"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for budget in (None, 128, *extras):
            rows = gas if budget is None else runner.method_stats(args, "formal", budget)
            method = "gas" if budget is None else "lessismore"
            for row in rows:
                base = gas_by_task[row["task"]]
                writer.writerow({"method": method, "budget": budget, "split": "formal",
                                 "task": row["task"], "samples": row["samples"],
                                 "score": row["score"], "mean_loaded_kv": row["loaded_kv"],
                                 "kv_ratio_to_gas": row["loaded_kv"] / base["loaded_kv"],
                                 "score_delta_vs_gas": row["score"] - base["score"],
                                 "sparse_only_kv": row["sparse_only_kv"]})
            score = runner.mean([row["score"] for row in rows])
            loaded = runner.mean([row["loaded_kv"] for row in rows])
            writer.writerow({"method": method, "budget": budget, "split": "formal", "task": "macro",
                             "samples": sum(row["samples"] for row in rows), "score": score,
                             "mean_loaded_kv": loaded, "kv_ratio_to_gas": loaded / gas_kv,
                             "score_delta_vs_gas": score - gas_score,
                             "sparse_only_kv": runner.mean([row["sparse_only_kv"] for row in rows]) if budget else ""})
    temporary.replace(output)
    print(f"COMPLETE expanded formal curves: {output}", flush=True)


if __name__ == "__main__":
    main()
