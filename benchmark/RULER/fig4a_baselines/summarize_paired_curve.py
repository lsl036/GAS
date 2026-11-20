#!/usr/bin/env python3
"""Export Fig4a calibration curves on the exact LessIsMore pilot samples."""

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import run_lessismore as runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extra-budgets", type=int, nargs="*", default=[])
    args_cli = parser.parse_args()
    results = Path(__file__).resolve().parent / "results"
    manifest = json.loads((results / "manifest.json").read_text())
    args = SimpleNamespace(reference=Path(manifest["reference"]), output=results,
                           cal_samples=manifest["cal_samples"],
                           formal_samples=manifest["formal_samples"])
    pred_root = args.reference / "calibration" / runner.RELATIVE_PRED.parent
    refs = {task: runner.reference(args.reference, "calibration", task, args.cal_samples)
            for task in runner.TASKS}
    rows = []
    for directory in sorted(pred_root.iterdir()):
        if not directory.is_dir():
            continue
        task_results = []
        for task in runner.TASKS:
            records = runner.read_jsonl(directory / f"{task}.jsonl")[:args.cal_samples]
            if len(records) != args.cal_samples:
                raise ValueError(f"Not enough records: {directory}/{task}")
            for record, ref in zip(records, refs[task]):
                if (record["index"], record["input"], record["outputs"]) != (
                    ref["index"], ref["input"], ref["outputs"]
                ):
                    raise ValueError(f"Unpaired record: {directory}/{task}")
            task_results.append({
                "score": runner.mean([runner.score(record) for record in records]),
                "loaded": runner.mean([record["kv_metrics"]["final_loaded_kv_tokens_mean"]
                                       for record in records]),
            })
        rows.append({"method": directory.name,
                     "macro_score": runner.mean([r["score"] for r in task_results]),
                     "mean_loaded_kv": runner.mean([r["loaded"] for r in task_results]),
                     "sparse_only_kv": ""})

    for budget in sorted(set(manifest["budgets"] + args_cli.extra_budgets)):
        task_results = runner.method_stats(args, "calibration", budget)
        rows.append({"method": f"lessismore_k{budget}",
                     "macro_score": runner.mean([r["score"] for r in task_results]),
                     "mean_loaded_kv": runner.mean([r["loaded_kv"] for r in task_results]),
                     "sparse_only_kv": runner.mean([r["sparse_only_kv"] for r in task_results])})

    gas = next(row for row in rows if row["method"] == "gas")
    name = "calibration_15_paired_curve_expanded.csv" if args_cli.extra_budgets else "calibration_15_paired_curve.csv"
    output = results / name
    with output.open("w", newline="", encoding="utf-8") as handle:
        fields = ["method", "split", "samples_per_task", "total_samples", "macro_score",
                  "mean_loaded_kv", "kv_ratio_to_gas", "score_delta_vs_gas", "sparse_only_kv"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "split": "calibration", "samples_per_task": args.cal_samples,
                             "total_samples": args.cal_samples * len(runner.TASKS),
                             "kv_ratio_to_gas": row["mean_loaded_kv"] / gas["mean_loaded_kv"],
                             "score_delta_vs_gas": row["macro_score"] - gas["macro_score"]})
    print(f"wrote {len(rows)} paired configurations to {output}")


if __name__ == "__main__":
    main()
