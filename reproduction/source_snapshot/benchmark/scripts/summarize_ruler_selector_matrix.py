#!/usr/bin/env python3
"""Summarize RULER selector/downstream runs across 16K/32K/64K."""

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean


TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2",
]
METHODS = [
    "full_paper50", "quest_4pct_paper50", "quest_8pct_paper50",
    "quest_twi_full", "group_max_p080_exception16_extra", "prob_sum_p080_exception16_full",
    "full_twi", "full_gas", "ds_4pct", "ds_8pct", "ds_twi", "ds_gas",
]
METRICS = (
    "final_loaded_kv_tokens_mean", "final_union_amplification_mean",
    "final_per_head_selected_mean", "pre_loaded_kv_tokens_mean",
    "post_loaded_kv_tokens_mean", "post_group_shared_count_mean",
)


def read_scores(method_dir: Path) -> dict[str, float]:
    paths = []
    combined = method_dir / "summary.csv"
    if combined.is_file():
        paths.append(combined)
    paths.extend(sorted(method_dir.glob("summary-*.csv")))
    scores = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        task_row = next((row for row in rows if row and row[0] == "Tasks"), None)
        score_row = next((row for row in rows if row and row[0] == "Score"), None)
        if task_row is None or score_row is None:
            continue
        for task, score in zip(task_row[1:], score_row[1:]):
            if not task or not score:
                continue
            try:
                scores[task] = float(score)
            except ValueError:
                continue
    return scores


def read_records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ruler-root", type=Path, required=True)
    parser.add_argument("--model-name", default="llama3-8b-chat-128k")
    parser.add_argument("--lengths", nargs="+", type=int, default=[16384, 32768, 65536])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for length in args.lengths:
        pred_root = args.ruler_root / args.model_name / "synthetic" / str(length) / "pred"
        for method in METHODS:
            method_dir = pred_root / method
            scores = read_scores(method_dir)
            task_rows = []
            for task in TASKS:
                records = read_records(method_dir / f"{task}.jsonl")
                metric_values = {name: [] for name in METRICS}
                for record in records:
                    metrics = record.get("kv_metrics", {})
                    for name in METRICS:
                        value = metrics.get(name)
                        if isinstance(value, (int, float)):
                            metric_values[name].append(float(value))
                row = {
                    "length": length,
                    "method": method,
                    "task": task,
                    "samples": len(records),
                    "score": scores.get(task, ""),
                    "missing_kv_metrics": sum(
                        not isinstance(record.get("kv_metrics", {}).get(METRICS[0]), (int, float))
                        for record in records
                    ),
                }
                for name, values in metric_values.items():
                    row[name] = fmean(values) if values else ""
                task_rows.append(row)
                rows.append(row)
            scored = [row["score"] for row in task_rows if row["score"] != ""]
            macro = {
                "length": length,
                "method": method,
                "task": "__macro_task_mean__",
                "samples": sum(row["samples"] for row in task_rows),
                "score": fmean(scored) if scored else "",
                "missing_kv_metrics": sum(row["missing_kv_metrics"] for row in task_rows),
            }
            for name in METRICS:
                values = [row[name] for row in task_rows if row[name] != ""]
                macro[name] = fmean(values) if values else ""
            rows.append(macro)

    fieldnames = ["length", "method", "task", "samples", "score", "missing_kv_metrics", *METRICS]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
