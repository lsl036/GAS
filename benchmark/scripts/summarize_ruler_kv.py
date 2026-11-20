"""Summarize per-sample RULER quality and synchronized GQA KV metrics."""

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean


METRICS = (
    "final_loaded_kv_tokens_mean",
    "final_union_amplification_mean",
    "final_per_head_selected_mean",
    "pre_loaded_kv_tokens_mean",
    "post_loaded_kv_tokens_mean",
    "post_group_shared_count_mean",
)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_scores(method_dir: Path):
    summary_path = method_dir / "summary.csv"
    if not summary_path.exists():
        return {}
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    task_row = next((row for row in rows if row and row[0] == "Tasks"), None)
    score_row = next((row for row in rows if row and row[0] == "Score"), None)
    if task_row is None or score_row is None:
        return {}
    return {
        task: float(score)
        for task, score in zip(task_row[1:], score_row[1:])
        if task and score
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for method_dir in sorted(path for path in args.pred_root.iterdir() if path.is_dir()):
        scores = read_scores(method_dir)
        for pred_path in sorted(method_dir.glob("*.jsonl")):
            if pred_path.name.endswith(".profile.jsonl"):
                continue
            records = read_jsonl(pred_path)
            metric_values = {name: [] for name in METRICS}
            for record in records:
                metrics = record.get("kv_metrics", {})
                for name in METRICS:
                    value = metrics.get(name)
                    if isinstance(value, (int, float)):
                        metric_values[name].append(float(value))
            row = {
                "method": method_dir.name,
                "task": pred_path.stem,
                "samples": len(records),
                "score": scores.get(pred_path.stem, ""),
                "missing_kv_metrics": sum(
                    not isinstance(record.get("kv_metrics", {}).get(METRICS[0]), (int, float))
                    for record in records
                ),
            }
            for name, values in metric_values.items():
                row[name] = fmean(values) if values else ""
            rows.append(row)

        method_rows = [row for row in rows if row["method"] == method_dir.name]
        scored = [row["score"] for row in method_rows if row["score"] != ""]
        macro_row = {
            "method": method_dir.name,
            "task": "__macro_task_mean__",
            "samples": sum(row["samples"] for row in method_rows),
            "score": fmean(scored) if scored else "",
            "missing_kv_metrics": sum(row["missing_kv_metrics"] for row in method_rows),
        }
        for name in METRICS:
            values = [row[name] for row in method_rows if row[name] != ""]
            macro_row[name] = fmean(values) if values else ""
        rows.append(macro_row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method", "task", "samples", "score", "missing_kv_metrics", *METRICS]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
