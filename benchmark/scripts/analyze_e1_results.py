"""Compute paired quality and KV-access statistics for the E1 RULER subset."""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


CONTROL_CHARS = re.compile(r"[\x00-\x1f]")
METRIC = "final_loaded_kv_tokens_mean"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sample_score(record: dict) -> float:
    prediction = CONTROL_CHARS.sub("\n", record["pred"].strip()).strip().lower()
    references = record.get("outputs", [record.get("output", "")])
    if not references:
        return 0.0
    return 100.0 * sum(str(ref).lower() in prediction for ref in references) / len(
        references
    )


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def stratified_bootstrap(
    task_values: dict[str, np.ndarray], rng: np.random.Generator, iterations: int
) -> np.ndarray:
    samples = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        task_means = []
        for values in task_values.values():
            indices = rng.integers(0, len(values), size=len(values))
            task_means.append(float(values[indices].mean()))
        samples[iteration] = np.mean(task_means)
    return samples


def load_methods(pred_root: Path, methods: list[str]) -> dict[str, dict[str, list[dict]]]:
    loaded = {}
    for method in methods:
        method_dir = pred_root / method
        if not method_dir.is_dir():
            raise FileNotFoundError(f"Missing method directory: {method_dir}")
        tasks = {}
        for path in sorted(method_dir.glob("*.jsonl")):
            if path.name.endswith(".profile.jsonl"):
                continue
            tasks[path.stem] = read_jsonl(path)
        loaded[method] = tasks
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--ours", required=True)
    parser.add_argument("--baselines", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="e1")
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()

    methods = [args.ours, *args.baselines]
    data = load_methods(args.pred_root, methods)
    tasks = sorted(data[args.ours])
    for method in methods:
        if sorted(data[method]) != tasks:
            raise ValueError(f"Task set differs for {method}")

    sample_rows = []
    task_scores = {method: {} for method in methods}
    task_loaded = {method: {} for method in methods}
    for task in tasks:
        expected_records = data[args.ours][task]
        for method in methods:
            method_records = data[method][task]
            if len(method_records) != len(expected_records):
                raise ValueError(f"Sample count differs for {method}/{task}")
            expected_ids = [
                str(record.get("others", {}).get("id", record["index"]))
                for record in expected_records
            ]
            method_ids = [
                str(record.get("others", {}).get("id", record["index"]))
                for record in method_records
            ]
            if method_ids != expected_ids:
                raise ValueError(f"Sample order or IDs differ for {method}/{task}")

        for position, ours_record in enumerate(expected_records):
            sample_id = str(
                ours_record.get("others", {}).get("id", ours_record["index"])
            )
            references = ours_record.get("outputs", [ours_record.get("output", "")])
            row = {
                "task": task,
                "sample_position": position,
                "sample_id": sample_id,
            }
            for method in methods:
                record = data[method][task][position]
                method_refs = record.get("outputs", [record.get("output", "")])
                if method_refs != references:
                    raise ValueError(f"References differ for {method}/{task}/{sample_id}")
                score = sample_score(record)
                loaded = record.get("kv_metrics", {}).get(METRIC)
                if not isinstance(loaded, (int, float)):
                    raise ValueError(f"Missing {METRIC} for {method}/{task}/{sample_id}")
                row[f"{method}_score"] = score
                row[f"{method}_loaded_kv"] = float(loaded)
            sample_rows.append(row)

        for method in methods:
            records = data[method][task]
            task_scores[method][task] = np.asarray(
                [sample_score(record) for record in records], dtype=np.float64
            )
            task_loaded[method][task] = np.asarray(
                [record["kv_metrics"][METRIC] for record in records], dtype=np.float64
            )

    rng = np.random.default_rng(args.seed)
    pairwise_rows = []
    for baseline in args.baselines:
        score_diff = {
            task: task_scores[args.ours][task] - task_scores[baseline][task]
            for task in tasks
        }
        loaded_diff = {
            task: task_loaded[args.ours][task] - task_loaded[baseline][task]
            for task in tasks
        }
        score_bootstrap = stratified_bootstrap(
            score_diff, rng, args.bootstrap_iterations
        )
        loaded_bootstrap = stratified_bootstrap(
            loaded_diff, rng, args.bootstrap_iterations
        )
        score_ci = percentile_interval(score_bootstrap)
        loaded_ci = percentile_interval(loaded_bootstrap)

        ours_scores = np.concatenate(list(task_scores[args.ours].values()))
        baseline_scores = np.concatenate(list(task_scores[baseline].values()))
        ours_loaded = np.concatenate(list(task_loaded[args.ours].values()))
        baseline_loaded = np.concatenate(list(task_loaded[baseline].values()))
        pairwise_rows.append(
            {
                "ours": args.ours,
                "baseline": baseline,
                "samples": len(ours_scores),
                "ours_score": ours_scores.mean(),
                "baseline_score": baseline_scores.mean(),
                "score_diff": (ours_scores - baseline_scores).mean(),
                "score_diff_ci95_low": score_ci[0],
                "score_diff_ci95_high": score_ci[1],
                "score_wins": int((ours_scores > baseline_scores).sum()),
                "score_ties": int((ours_scores == baseline_scores).sum()),
                "score_losses": int((ours_scores < baseline_scores).sum()),
                "ours_loaded_kv": ours_loaded.mean(),
                "baseline_loaded_kv": baseline_loaded.mean(),
                "loaded_kv_diff": (ours_loaded - baseline_loaded).mean(),
                "loaded_kv_diff_ci95_low": loaded_ci[0],
                "loaded_kv_diff_ci95_high": loaded_ci[1],
                "loaded_kv_reduction_pct": 100.0
                * (1.0 - ours_loaded.mean() / baseline_loaded.mean()),
                "baseline_budget_excess_vs_ours_pct": 100.0
                * (baseline_loaded.mean() / ours_loaded.mean() - 1.0),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.output_dir / f"{args.prefix}_paired_samples.csv"
    with sample_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)

    summary_path = args.output_dir / f"{args.prefix}_pairwise_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairwise_rows[0]))
        writer.writeheader()
        writer.writerows(pairwise_rows)

    print(f"wrote {len(sample_rows)} paired samples to {sample_path}")
    print(f"wrote {len(pairwise_rows)} comparisons to {summary_path}")


if __name__ == "__main__":
    main()
