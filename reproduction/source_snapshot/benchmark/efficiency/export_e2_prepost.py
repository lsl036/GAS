import argparse
import csv
import json
import math
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "thoughts" / "Suggests" / "e012" / "e2_ablation_64k.csv"
DEFAULT_CONFIG_DIR = ROOT / "benchmark" / "configs" / "e2_20260921"
DEFAULT_OUTPUT = (
    ROOT
    / "benchmark"
    / "results_efficiency"
    / "e4_kernel_evidence"
    / "E2_PrePost_Export.csv"
)


def method_number(method: str) -> int:
    return int(method.removeprefix("e2_A"))


def config_for(method: str, config_dir: Path) -> Path:
    number = method_number(method)
    matches = sorted(config_dir.glob(f"a{number}_*.json"))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one config for {method}, found {matches}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export E2 A1-A8 same-run pre/post loaded-KV statistics without inference."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        source_rows = list(csv.DictReader(f))
    selected = [
        row
        for row in source_rows
        if row["method"].startswith("e2_A") and 1 <= method_number(row["method"]) <= 8
    ]
    selected.sort(key=lambda row: (method_number(row["method"]), row["task"]))
    if len(selected) != 32:
        raise ValueError(f"Expected 8 configs x 4 rows, found {len(selected)} rows")
    for method_index in range(1, 9):
        method = f"e2_A{method_index}"
        rows = [row for row in selected if row["method"] == method]
        macro = next(row for row in rows if row["task"] == "__macro_task_mean__")
        tasks = [row for row in rows if row["task"] != "__macro_task_mean__"]
        for field in (
            "score",
            "pre_loaded_kv_tokens_mean",
            "post_loaded_kv_tokens_mean",
        ):
            expected = sum(float(row[field]) for row in tasks) / len(tasks)
            if not math.isclose(float(macro[field]), expected, rel_tol=1e-12, abs_tol=1e-9):
                raise ValueError(
                    f"{method} macro {field} is not the equal-task mean: "
                    f"recorded={macro[field]}, expected={expected}"
                )

    output_rows: List[dict] = []
    for row in selected:
        config_path = config_for(row["method"], args.config_dir)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        pre = float(row["pre_loaded_kv_tokens_mean"])
        post = float(row["post_loaded_kv_tokens_mean"])
        missing = int(row["missing_kv_metrics"])
        samples = int(row["samples"])
        expected_samples = 150 if row["task"] == "__macro_task_mean__" else 50
        if missing != 0 or samples != expected_samples:
            raise ValueError(
                f"Incomplete synchronized metrics for {row['method']} {row['task']}: "
                f"samples={samples}, missing={missing}"
            )
        output_rows.append(
            {
                "method": row["method"],
                "task": row["task"],
                "p_t": config["weight_pruner"]["threshold"],
                "p_g": config["gqa_refinement"]["threshold"],
                "exception_budget": config["gqa_refinement"].get("exception_budget", 0),
                "samples": samples,
                "score": row["score"],
                "missing_kv_metrics": missing,
                "pre_loaded_kv_tokens_mean": pre,
                "post_loaded_kv_tokens_mean": post,
                "kv_reduction_pct": 100.0 * (1.0 - post / pre),
                "post_union_amplification_mean": row["final_union_amplification_mean"],
                "post_group_shared_count_mean": row["post_group_shared_count_mean"],
                "source": str(args.input.resolve()),
                "config": str(config_path.resolve()),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output_rows[0])
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} E2 pre/post rows to {args.output}")


if __name__ == "__main__":
    main()
