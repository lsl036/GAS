"""Validate and summarize the paired E3 p_t sensitivity experiment."""

import argparse
import csv
import json
import os
import re
from pathlib import Path

import numpy as np


CONTROL_CHARS = re.compile(r"[\x00-\x1f]")
PRIMARY_KV = "final_loaded_kv_tokens_mean"
DIAGNOSTICS = (
    "final_loaded_kv_tokens_mean",
    "final_union_amplification_mean",
    "final_per_head_selected_mean",
    "final_group_union_count_mean",
    "pre_loaded_kv_tokens_mean",
    "pre_union_amplification_mean",
    "post_loaded_kv_tokens_mean",
    "post_union_amplification_mean",
    "post_group_shared_count_mean",
    "profile_record_count",
    "profile_record_count_total",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sample_id(record: dict) -> str:
    return str(record.get("others", {}).get("id", record["index"]))


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


def paired_bootstrap(
    candidate: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    iterations: int,
    rng: np.random.Generator,
    relative: bool = False,
) -> tuple[float, float]:
    draws = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        candidate_means = []
        baseline_means = []
        for task in candidate:
            indices = rng.integers(0, len(candidate[task]), size=len(candidate[task]))
            candidate_means.append(float(candidate[task][indices].mean()))
            baseline_means.append(float(baseline[task][indices].mean()))
        candidate_macro = float(np.mean(candidate_means))
        baseline_macro = float(np.mean(baseline_means))
        if relative:
            draws[iteration] = 100.0 * (candidate_macro / baseline_macro - 1.0)
        else:
            draws[iteration] = candidate_macro - baseline_macro
    return percentile_interval(draws)


def task_bootstrap(
    candidate: np.ndarray,
    baseline: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
    relative: bool = False,
) -> tuple[float, float]:
    indices = rng.integers(0, len(candidate), size=(iterations, len(candidate)))
    candidate_means = candidate[indices].mean(axis=1)
    baseline_means = baseline[indices].mean(axis=1)
    if relative:
        draws = 100.0 * (candidate_means / baseline_means - 1.0)
    else:
        draws = candidate_means - baseline_means
    return percentile_interval(draws)


def load_and_validate(plan: dict, repo_root: Path) -> tuple[dict, list[dict]]:
    tasks = plan["tasks"]
    expected = plan["expected_samples_per_task"]
    data_dir = repo_root / plan["data_dir"]
    loaded = {}
    paired_rows = []

    manifests = {task: read_jsonl(data_dir / task / "validation.jsonl") for task in tasks}
    for task, records in manifests.items():
        if len(records) != expected:
            raise ValueError(f"Expected {expected} manifest rows for {task}, got {len(records)}")

    for method in plan["methods"]:
        config_id = method["config_id"]
        method_dir = repo_root / method["prediction_dir"]
        loaded[config_id] = {}
        for task in tasks:
            records = read_jsonl(method_dir / f"{task}.jsonl")
            manifest = manifests[task]
            if len(records) != expected:
                raise ValueError(f"Expected {expected} rows for {config_id}/{task}, got {len(records)}")
            for position, (source, record) in enumerate(zip(manifest, records)):
                if source.get("index") != record.get("index"):
                    raise ValueError(f"Index mismatch for {config_id}/{task}/{position}")
                if source.get("input") != record.get("input"):
                    raise ValueError(f"Input mismatch for {config_id}/{task}/{position}")
                if source.get("outputs") != record.get("outputs"):
                    raise ValueError(f"Reference mismatch for {config_id}/{task}/{position}")
                if not isinstance(record.get("kv_metrics", {}).get(PRIMARY_KV), (int, float)):
                    raise ValueError(f"Missing {PRIMARY_KV} for {config_id}/{task}/{position}")
            loaded[config_id][task] = records

    baseline_id = plan["default_config_id"]
    for task in tasks:
        baseline_records = loaded[baseline_id][task]
        baseline_ids = [sample_id(record) for record in baseline_records]
        for method in plan["methods"]:
            config_id = method["config_id"]
            records = loaded[config_id][task]
            if [sample_id(record) for record in records] != baseline_ids:
                raise ValueError(f"Sample ID order differs for {config_id}/{task}")
            for position, record in enumerate(records):
                metrics = record["kv_metrics"]
                row = {
                    "config_id": config_id,
                    "task": task,
                    "sample_position": position,
                    "sample_id": sample_id(record),
                    "score": sample_score(record),
                    "loaded_kv": metrics[PRIMARY_KV],
                }
                for metric in DIAGNOSTICS:
                    row[metric] = metrics.get(metric, "")
                row["complement_count_mean"] = metrics.get("complement_count_mean", "")
                paired_rows.append(row)
    return loaded, paired_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_summary(plan: dict, loaded: dict) -> list[dict]:
    methods = {method["config_id"]: method for method in plan["methods"]}
    tasks = plan["tasks"]
    default_id = plan["default_config_id"]
    expected = plan["expected_samples_per_task"]
    iterations = plan["bootstrap"]["iterations"]
    rng = np.random.default_rng(plan["bootstrap"]["seed"])

    scores = {
        config_id: {
            task: np.asarray([sample_score(record) for record in records], dtype=np.float64)
            for task, records in task_records.items()
        }
        for config_id, task_records in loaded.items()
    }
    kv = {
        config_id: {
            task: np.asarray(
                [record["kv_metrics"][PRIMARY_KV] for record in records], dtype=np.float64
            )
            for task, records in task_records.items()
        }
        for config_id, task_records in loaded.items()
    }

    rows = []
    for method in plan["methods"]:
        config_id = method["config_id"]
        for task in [*tasks, "__macro_task_mean__"]:
            if task == "__macro_task_mean__":
                score = float(np.mean([scores[config_id][name].mean() for name in tasks]))
                loaded_kv = float(np.mean([kv[config_id][name].mean() for name in tasks]))
                n_expected = expected * len(tasks)
                n_valid = sum(len(scores[config_id][name]) for name in tasks)
                if config_id == default_id:
                    score_ci = (0.0, 0.0)
                    kv_ci = (0.0, 0.0)
                else:
                    score_ci = paired_bootstrap(
                        scores[config_id], scores[default_id], iterations, rng
                    )
                    kv_ci = paired_bootstrap(
                        kv[config_id], kv[default_id], iterations, rng, relative=True
                    )
                default_score = float(np.mean([scores[default_id][name].mean() for name in tasks]))
                default_kv = float(np.mean([kv[default_id][name].mean() for name in tasks]))
            else:
                score = float(scores[config_id][task].mean())
                loaded_kv = float(kv[config_id][task].mean())
                n_expected = expected
                n_valid = len(scores[config_id][task])
                if config_id == default_id:
                    score_ci = (0.0, 0.0)
                    kv_ci = (0.0, 0.0)
                else:
                    score_ci = task_bootstrap(
                        scores[config_id][task], scores[default_id][task], iterations, rng
                    )
                    kv_ci = task_bootstrap(
                        kv[config_id][task],
                        kv[default_id][task],
                        iterations,
                        rng,
                        relative=True,
                    )
                default_score = float(scores[default_id][task].mean())
                default_kv = float(kv[default_id][task].mean())

            rows.append(
                {
                    "config_id": config_id,
                    "task": task,
                    "p_t": method["p_t"],
                    "p_g": method["p_g"],
                    "exception_budget": method["exception_budget"],
                    "n_expected": n_expected,
                    "n_valid": n_valid,
                    "score": score,
                    "loaded_kv": loaded_kv,
                    "delta_q_vs_default": score - default_score,
                    "delta_q_ci95_low": score_ci[0],
                    "delta_q_ci95_high": score_ci[1],
                    "delta_k_pct_vs_default": 100.0 * (loaded_kv / default_kv - 1.0),
                    "delta_k_pct_ci95_low": kv_ci[0],
                    "delta_k_pct_ci95_high": kv_ci[1],
                }
            )
    return rows


def write_plot(path: Path, macro_rows: list[dict]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-e3")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5.4, 4.0))
    for row in macro_rows:
        axis.scatter(row["loaded_kv"], row["score"], s=55)
        axis.annotate(
            f'{row["config_id"]} (p_t={float(row["p_t"]):.2f})',
            (row["loaded_kv"], row["score"]),
            xytext=(6, 5),
            textcoords="offset points",
        )
    axis.set_xlabel("Mean loaded KV token slots")
    axis.set_ylabel("Macro score")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_markdown(path: Path, plan: dict, macro_rows: list[dict]) -> None:
    by_id = {row["config_id"]: row for row in macro_rows}
    default = by_id[plan["default_config_id"]]
    low = by_id["P0-L"]
    high = by_id["P0-H"]
    default_dominated = any(
        row["score"] >= default["score"]
        and row["loaded_kv"] <= default["loaded_kv"]
        and (row["score"] > default["score"] or row["loaded_kv"] < default["loaded_kv"])
        for row in (low, high)
    )
    trigger_low = low["score"] > default["score"] and low["loaded_kv"] < default["loaded_kv"]
    trigger_high = (
        high["delta_q_ci95_low"] > 0.0 and high["loaded_kv"] > default["loaded_kv"]
    )

    lines = [
        "# E3 P0 Result Summary",
        "",
        "## Scope",
        "",
        "LLaMA-3.1-8B, 64K, three frozen E2 tasks, 50 paired samples per task. ",
        "P0-D reuses E2 A3; P0-L and P0-H reuse the identical E2 manifests.",
        "",
        "## Macro Results",
        "",
        "| ID | p_t | p_g | m | Score | Loaded KV | Delta Q | Delta K | Delta Q 95% CI | Delta K 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in macro_rows:
        lines.append(
            f'| {row["config_id"]} | {float(row["p_t"]):.2f} | '
            f'{float(row["p_g"]):.2f} | {row["exception_budget"]} | '
            f'{row["score"]:.2f} | {row["loaded_kv"]:.2f} | '
            f'{row["delta_q_vs_default"]:+.2f} | '
            f'{row["delta_k_pct_vs_default"]:+.2f}% | '
            f'[{row["delta_q_ci95_low"]:.2f}, {row["delta_q_ci95_high"]:.2f}] | '
            f'[{row["delta_k_pct_ci95_low"]:.2f}%, {row["delta_k_pct_ci95_high"]:.2f}%] |'
        )
    lines.extend(
        [
            "",
            "## Local Selection Conclusion",
            "",
            f'- The default is {"dominated by an observed local point" if default_dominated else "not dominated by either observed local point"} on measured means.',
            f'- P1 low-side `.75` trigger: {"yes" if trigger_low else "no"}. The trigger requires P0-L to have both higher quality and lower loaded KV than P0-D.',
            f'- P1 high-side `.95` trigger: {"yes" if trigger_high else "no"}. The frozen trigger requires a positive Delta Q confidence interval and increased loaded KV at P0-H.',
            "- A confidence interval covering zero is reported as inconclusive, not as equivalence.",
            "- Loaded KV is a group-level unique token-slot count, not HBM bytes, memory capacity, or latency.",
            "- `complement_count_mean` is unavailable in the current synchronized KV records and is retained as an empty field in `paired_samples.csv`; no value is estimated.",
            "",
            "![E3 quality-KV scatter](quality_kv_scatter.png)",
            "",
            "## Artifacts",
            "",
            "- `paired_samples.csv`: strict paired sample-level scores and KV diagnostics.",
            "- `summary.csv`: task and equal-weight macro results with paired bootstrap intervals.",
            "- `../metadata/`: commands, manifests, GPU, package, config, and code-version records.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()

    experiment_dir = args.experiment_dir.resolve()
    repo_root = Path(__file__).resolve().parents[2]
    plan = json.loads((experiment_dir / "experiment_plan.json").read_text(encoding="utf-8"))
    loaded, paired_rows = load_and_validate(plan, repo_root)
    summary_rows = build_summary(plan, loaded)
    analysis_dir = experiment_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_csv(analysis_dir / "paired_samples.csv", paired_rows)
    write_csv(analysis_dir / "summary.csv", summary_rows)
    macro_rows = [row for row in summary_rows if row["task"] == "__macro_task_mean__"]
    write_plot(analysis_dir / "quality_kv_scatter.png", macro_rows)
    write_markdown(analysis_dir / "E3_Result_Summary.md", plan, macro_rows)
    print(f"wrote {len(paired_rows)} paired rows to {analysis_dir / 'paired_samples.csv'}")
    print(f"wrote {len(summary_rows)} summary rows to {analysis_dir / 'summary.csv'}")
    print(f"wrote E3 report to {analysis_dir / 'E3_Result_Summary.md'}")


if __name__ == "__main__":
    main()
