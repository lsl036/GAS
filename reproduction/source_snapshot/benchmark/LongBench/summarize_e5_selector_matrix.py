#!/usr/bin/env python3
"""Summarize the original E5 rows together with selector-matrix rows."""

import argparse
import csv
import json
from pathlib import Path

from summarize_e5_longbench import (
    TASKS,
    load_manifest,
    metric_modules,
    read_predictions,
    score_record,
    write_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    manifest = load_manifest(root / "manifest.jsonl")
    methods = list(protocol["methods"])
    metrics = metric_modules()
    task_rows = []
    summary_rows = []

    for method in methods:
        method_task_rows = []
        method_scores = []
        for task in TASKS:
            source = root / "pred" / method / task / "predictions.jsonl"
            expected = {
                row["sample_id"]: row["sample_hash"] for row in manifest[task]
            }
            metric_name, metric_fn = metrics[task]
            if not source.is_file():
                row = {
                    "method": method,
                    "task": task,
                    "metric": metric_name,
                    "n_expected": len(expected),
                    "n_predictions": 0,
                    "n_scored": 0,
                    "n_failed": len(expected),
                    "score": "",
                    "source_path": str(source),
                    "reused": str(protocol["methods"][method].get("reused", False)).lower(),
                }
                task_rows.append(row)
                method_task_rows.append(row)
                continue

            predictions = read_predictions(source)
            seen = set()
            scores = []
            failed = 0
            for record in predictions:
                sample_id = record.get("_e5_sample_id")
                if sample_id not in expected or sample_id in seen:
                    raise ValueError(f"{method}/{task}: invalid or duplicate sample ID {sample_id}")
                if record.get("_e5_sample_hash") != expected[sample_id]:
                    raise ValueError(f"{method}/{task}: sample hash mismatch for {sample_id}")
                seen.add(sample_id)
                score, is_failed = score_record(task, record, metric_fn)
                scores.append(score)
                failed += int(is_failed)
            if seen != set(expected):
                raise ValueError(f"{method}/{task}: incomplete prediction set")
            n_scored = len(scores) - failed
            score = 100.0 * sum(scores) / n_scored if n_scored else 0.0
            row = {
                "method": method,
                "task": task,
                "metric": metric_name,
                "n_expected": len(expected),
                "n_predictions": len(predictions),
                "n_scored": n_scored,
                "n_failed": failed,
                "score": repr(score),
                "source_path": str(source),
                "reused": str(protocol["methods"][method].get("reused", False)).lower(),
            }
            task_rows.append(row)
            method_task_rows.append(row)
            method_scores.append(score)

        summary_rows.append(
            {
                "method": method,
                "n_tasks": len(TASKS),
                "n_samples": sum(int(row["n_scored"]) for row in method_task_rows),
                "macro_score": repr(sum(method_scores) / len(method_scores)) if method_scores else "",
                "reused": str(protocol["methods"][method].get("reused", False)).lower(),
            }
        )

    task_output = root / "E5_Selector_Matrix_Task_Scores.csv"
    write_csv(
        task_output,
        task_rows,
        [
            "method",
            "task",
            "metric",
            "n_expected",
            "n_predictions",
            "n_scored",
            "n_failed",
            "score",
            "source_path",
            "reused",
        ],
    )
    summary_output = root / "E5_Selector_Matrix_Summary.csv"
    write_csv(
        summary_output,
        summary_rows,
        ["method", "n_tasks", "n_samples", "macro_score", "reused"],
    )
    print(f"wrote {len(task_rows)} task rows to {task_output}")
    print(f"wrote {len(summary_rows)} method rows to {summary_output}")


if __name__ == "__main__":
    main()
