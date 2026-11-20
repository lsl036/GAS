#!/usr/bin/env python3
"""Create comparable full-protocol and paired first-50 LongBench tables."""

import argparse
import csv
import json
from pathlib import Path

from summarize_e5_longbench import TASKS, metric_modules, read_predictions, score_record


ROOT = Path(__file__).resolve().parents[2]
METHOD_SOURCES = {
    "full": (ROOT / "benchmark/results_longbench/e5_longbench_completion/pred/full", "complete_3150"),
    "quest_4pct": (ROOT / "benchmark/results_longbench/e5_longbench_completion/pred/quest_4pct", "complete_3150"),
    "quest_8pct": (ROOT / "benchmark/results_longbench/e5_longbench_completion/pred/quest_8pct", "complete_3150"),
    "prob_sum": (ROOT / "benchmark/results_longbench/e5_longbench_completion/pred/prob_sum", "complete_3150"),
    "quest_twi": (ROOT / "benchmark/results_longbench/e5_longbench_completion/pred/quest_twi", "complete_3150"),
    "group_max": (ROOT / "benchmark/results_longbench/e5_longbench_completion/pred/group_max", "complete_3150"),
    "full_twi": (ROOT / "benchmark/results_longbench/e5_selector_matrix/pred/full_twi", "incomplete_matrix"),
    "full_gas": (ROOT / "benchmark/results_longbench/e5_selector_matrix/pred/full_gas", "incomplete_matrix"),
    "ds_4pct": (ROOT / "benchmark/results_longbench/e5_selector_matrix_ds50/pred/ds_4pct", "paired_first50"),
    "ds_8pct": (ROOT / "benchmark/results_longbench/e5_selector_matrix_ds50/pred/ds_8pct", "paired_first50"),
    "ds_twi": (ROOT / "benchmark/results_longbench/e5_selector_matrix_ds50/pred/ds_twi", "paired_first50"),
    "ds_gas": (ROOT / "benchmark/results_longbench/e5_selector_matrix_ds50/pred/ds_gas", "paired_first50"),
}


def score_task(task: str, records: list[dict], limit: int | None = None) -> tuple[float, int]:
    records = records if limit is None else records[:limit]
    _, metric_fn = metric_modules()[task]
    scores = [score_record(task, record, metric_fn)[0] for record in records]
    return (100.0 * sum(scores) / len(scores), len(records)) if scores else (float("nan"), 0)


def load_existing_macro(method: str) -> str:
    path = ROOT / "benchmark/results_longbench/e5_longbench_completion/E5_LongBench_Summary.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["method"] == method:
                return row["macro_score"]
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "thoughts/Paper",
    )
    parser.add_argument("--first-n", type=int, default=50)
    args = parser.parse_args()
    if args.first_n <= 0:
        raise ValueError("--first-n must be positive")

    task_rows = []
    summary_rows = []
    for method, (method_dir, protocol) in METHOD_SOURCES.items():
        scores = []
        total_samples = 0
        completed_tasks = []
        for task in TASKS:
            path = method_dir / task / "predictions.jsonl"
            records = read_predictions(path) if path.is_file() else []
            score, samples = score_task(task, records, args.first_n)
            if samples == args.first_n:
                scores.append(score)
                completed_tasks.append(task)
            total_samples += samples
            task_rows.append(
                {
                    "method": method,
                    "protocol": protocol,
                    "task": task,
                    "samples_used": samples,
                    "score_first50": "" if samples == 0 else repr(score),
                    "complete_task": str(samples == args.first_n).lower(),
                    "source": str(path),
                }
            )

        summary_rows.append(
            {
                "method": method,
                "protocol": protocol,
                "samples_per_task": args.first_n,
                "total_samples_first50": total_samples,
                "completed_tasks": len(completed_tasks),
                "total_tasks": len(TASKS),
                "macro_score_first50": "" if len(completed_tasks) != len(TASKS) else repr(sum(scores) / len(scores)),
                "available_task_mean_first50": "" if not scores else repr(sum(scores) / len(scores)),
                "full_3150_macro_score": load_existing_macro(method) if protocol == "complete_3150" else "",
                "missing_tasks": ",".join(task for task in TASKS if task not in completed_tasks),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_csv = args.output_dir / "LongBench_First50_Task_Scores.csv"
    with task_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(task_rows[0]))
        writer.writeheader()
        writer.writerows(task_rows)

    summary_csv = args.output_dir / "LongBench_Comparison_Summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    complete = [row for row in summary_rows if row["protocol"] == "complete_3150"]
    paired = [row for row in summary_rows if row["completed_tasks"] == len(TASKS)]
    incomplete = [row for row in summary_rows if row["completed_tasks"] != len(TASKS)]
    lines = [
        "# LongBench Comparison Table",
        "",
        "Model: `Meta-Llama-3.1-8B-Instruct`; tasks: 13; evaluator: original task-specific LongBench metrics.",
        "",
        "## Complete 3150-Sample E5 Results",
        "",
        "| Method | Samples | Macro score |",
        "|---|---:|---:|",
    ]
    for row in complete:
        lines.append(f"| {row['method']} | 3150 | {float(row['full_3150_macro_score']):.2f} |")
    lines.extend(
        [
            "",
            "## Paired First-50 Results",
            "",
            "Each method uses the same first 50 samples of every task. The macro is the equal mean of the 13 task scores.",
            "",
            "| Method | Samples | Macro score | Protocol |",
            "|---|---:|---:|---|",
        ]
    )
    for row in paired:
        lines.append(
            f"| {row['method']} | {row['total_samples_first50']} | "
            f"{float(row['macro_score_first50']):.2f} | {row['protocol']} |"
        )
    lines.extend(
        [
            "",
            "## Incomplete Runs",
            "",
            "These rows are reported for audit only. Their available-task mean is not a valid 13-task LongBench macro.",
            "",
            "| Method | Completed tasks | Available samples | Available-task mean | Missing tasks |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in incomplete:
        available = row["available_task_mean_first50"]
        available_text = "" if not available else f"{float(available):.2f}"
        lines.append(
            f"| {row['method']} | {row['completed_tasks']}/{row['total_tasks']} | "
            f"{row['total_samples_first50']} | {available_text} | {row['missing_tasks']} |"
        )
    lines.extend(
        [
            "",
            "The complete E5 and paired first-50 scores use different sample counts and should be presented as separate protocol rows.",
            "The first-50 DS results are an ablation/pilot comparison and should not replace the complete 3150-sample E5 table.",
            "",
        ]
    )
    report = args.output_dir / "LongBench_Comparison_Table.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {task_csv}")
    print(f"wrote {summary_csv}")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
