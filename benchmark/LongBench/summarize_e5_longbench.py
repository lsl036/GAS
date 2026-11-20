#!/usr/bin/env python3
"""Score and audit the six-method E5 LongBench completion."""

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LONGBENCH = ROOT / "benchmark" / "LongBench"
DEFAULT_OUTPUT = ROOT / "benchmark" / "results_longbench" / "e5_longbench_completion"
TASKS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "triviaqa",
    "passage_retrieval_en",
    "lcc",
    "repobench",
]
DATASET_KEY = {"repobench": "repobench-p"}
METHODS = ["full", "quest_4pct", "quest_8pct", "prob_sum", "quest_twi", "group_max"]


def metric_modules():
    sys.path.insert(0, str(LONGBENCH))
    from metrics import (
        code_sim_score,
        count_score,
        qa_f1_score,
        retrieval_score,
        rouge_score,
    )

    return {
        "narrativeqa": ("qa_f1", qa_f1_score),
        "qasper": ("qa_f1", qa_f1_score),
        "multifieldqa_en": ("qa_f1", qa_f1_score),
        "hotpotqa": ("qa_f1", qa_f1_score),
        "2wikimqa": ("qa_f1", qa_f1_score),
        "musique": ("qa_f1", qa_f1_score),
        "gov_report": ("rouge_l", rouge_score),
        "qmsum": ("rouge_l", rouge_score),
        "multi_news": ("rouge_l", rouge_score),
        "triviaqa": ("qa_f1", qa_f1_score),
        "passage_retrieval_en": ("retrieval", retrieval_score),
        "lcc": ("code_sim", code_sim_score),
        "repobench": ("code_sim", code_sim_score),
    }


def load_manifest(path: Path) -> dict[str, list[dict]]:
    result = {task: [] for task in TASKS}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["task"] not in result:
                raise ValueError(f"Unexpected task in manifest: {row['task']}")
            result[row["task"]].append(row)
    if any(not rows for rows in result.values()):
        raise ValueError("Manifest is missing one or more required tasks")
    return result


def prediction_path(root: Path, method: str, task: str) -> Path:
    path = root / "pred" / method / task / "predictions.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing E5 prediction: {path}")
    return path


def read_predictions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score_record(task: str, record: dict, metric_fn) -> tuple[float, bool]:
    if "pred" not in record:
        return 0.0, True
    prediction = record["pred"]
    if task in {"triviaqa"}:
        prediction = prediction.lstrip("\n").split("\n")[0]
    answers = record.get("answers", [])
    if not answers:
        return 0.0, False
    sample_score = max(
        metric_fn(prediction, answer, all_classes=record.get("all_classes"))
        for answer in answers
    )
    return float(sample_score), False


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    manifest = load_manifest(root / "manifest.jsonl")
    metrics = metric_modules()
    reused = {method: bool(protocol["methods"][method]["reused"]) for method in METHODS}

    task_rows = []
    method_scores = {method: {} for method in METHODS}
    method_samples = {method: 0 for method in METHODS}
    for method in METHODS:
        for task in TASKS:
            expected_rows = manifest[task]
            expected_ids = {row["sample_id"]: row["sample_hash"] for row in expected_rows}
            source = prediction_path(root, method, task)
            predictions = read_predictions(source)
            if len(predictions) != len(expected_rows):
                raise ValueError(
                    f"{method}/{task}: {len(predictions)} predictions, expected {len(expected_rows)}"
                )
            seen = set()
            scores = []
            failed = 0
            metric_name, metric_fn = metrics[task]
            for record in predictions:
                sample_id = record.get("_e5_sample_id")
                if sample_id not in expected_ids or sample_id in seen:
                    raise ValueError(f"{method}/{task}: invalid or duplicate sample ID {sample_id}")
                if record.get("_e5_sample_hash") != expected_ids[sample_id]:
                    raise ValueError(f"{method}/{task}: sample hash mismatch for {sample_id}")
                seen.add(sample_id)
                score, is_failed = score_record(task, record, metric_fn)
                scores.append(score)
                failed += int(is_failed)
            if seen != set(expected_ids):
                raise ValueError(f"{method}/{task}: manifest/prediction sample sets differ")
            n_scored = len(scores) - failed
            task_score = 100.0 * sum(scores) / n_scored if n_scored else 0.0
            method_scores[method][task] = task_score
            method_samples[method] += n_scored
            task_rows.append(
                {
                    "method": method,
                    "task": task,
                    "metric": metric_name,
                    "n_expected": len(expected_rows),
                    "n_predictions": len(predictions),
                    "n_scored": n_scored,
                    "n_failed": failed,
                    "score": repr(task_score),
                    "source_path": str(source),
                    "reused": str(reused[method]).lower(),
                }
            )

        method_result_path = root / "pred" / method / "result.json"
        method_result_path.write_text(
            json.dumps(method_scores[method], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    task_output = root / "E5_LongBench_Task_Scores.csv"
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

    macro = {
        method: sum(method_scores[method].values()) / len(TASKS) for method in METHODS
    }
    summary_rows = []
    for method in METHODS:
        summary_rows.append(
            {
                "method": method,
                "n_tasks": len(TASKS),
                "n_samples": method_samples[method],
                "macro_score": repr(macro[method]),
                "delta_vs_full": repr(macro[method] - macro["full"]),
                "delta_vs_twilight": repr(macro[method] - macro["quest_twi"]),
                "delta_vs_group_max": repr(macro[method] - macro["group_max"]),
            }
        )
    summary_output = root / "E5_LongBench_Summary.csv"
    write_csv(
        summary_output,
        summary_rows,
        [
            "method",
            "n_tasks",
            "n_samples",
            "macro_score",
            "delta_vs_full",
            "delta_vs_twilight",
            "delta_vs_group_max",
        ],
    )

    report = [
        "# E5 LongBench Completion",
        "",
        "## Main Results",
        "",
        "| method | tasks | samples | macro score | vs Full | vs Quest+Twilight | vs GAS group-max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        report.append(
            f"| {row['method']} | {row['n_tasks']} | {row['n_samples']} | "
            f"{float(row['macro_score']):.2f} | {float(row['delta_vs_full']):+.2f} | "
            f"{float(row['delta_vs_twilight']):+.2f} | {float(row['delta_vs_group_max']):+.2f} |"
        )
    report.extend(
        [
            "",
            "All rows use the same 13 tasks and the same manifest. Task scores are computed first",
            "with the original task-specific LongBench evaluators; the macro score is the equal mean",
            "of the 13 unrounded task scores. Empty predictions remain in the scoring denominator.",
            "",
            "See `E5_LongBench_Task_Scores.csv` for task-level scores and `protocol.json` for the",
            "dataset, sample hashes, method configs, generation protocol, and reused-source mapping.",
            "",
        ]
    )
    (root / "E5_LongBench_Result_Summary.md").write_text("\n".join(report), encoding="utf-8")
    print(f"wrote {len(task_rows)} task rows to {task_output}")
    print(f"wrote {len(summary_rows)} method rows to {summary_output}")
    print(f"wrote report to {root / 'E5_LongBench_Result_Summary.md'}")


if __name__ == "__main__":
    main()
