#!/usr/bin/env python3
"""Prepare the frozen E5 manifest and add the selector matrix configs."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE_PREP = Path(__file__).with_name("prepare_e5_longbench.py")
BASE_OUTPUT = ROOT / "benchmark" / "results_longbench" / "e5_longbench_completion"
EXISTING_METHODS = ("full", "quest_4pct", "quest_8pct", "prob_sum")
EXTRA_CONFIGS = {
    "full_twi": ROOT / "benchmark" / "configs" / "config_full_twi.json",
    "full_gas": ROOT / "benchmark" / "configs" / "config_full_gas.json",
    "ds_4pct": ROOT / "benchmark" / "configs" / "config_ds_4pct_llama31.json",
    "ds_8pct": ROOT / "benchmark" / "configs" / "config_ds_8pct_llama31.json",
    "ds_twi": ROOT / "benchmark" / "configs" / "config_ds_twi_llama31.json",
    "ds_gas": ROOT
    / "benchmark"
    / "configs"
    / "config_ds_twi_gqa_group_max_p080_exception16_llama31.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--samples-per-task", type=int, default=None)
    args = parser.parse_args()

    if args.samples_per_task is not None and args.samples_per_task <= 0:
        raise ValueError("--samples-per-task must be positive")

    subprocess.run(
        [
            sys.executable,
            str(BASE_PREP),
            "--data-dir",
            str(args.data_dir),
            "--output-dir",
            str(args.output_dir),
            "--model-path",
            str(args.model_path),
        ],
        check=True,
    )

    # The base preparer normalizes the two legacy sources, while the other
    # four completed E5 methods already live in the frozen E5 output tree.
    # Copy their normalized predictions into this independent matrix tree so
    # the final summary covers all old and new methods in one manifest.
    for method in EXISTING_METHODS:
        source = BASE_OUTPUT / "pred" / method
        target = args.output_dir.resolve() / "pred" / method
        if not source.is_dir():
            raise FileNotFoundError(source)
        shutil.copytree(source, target, dirs_exist_ok=True)

    protocol_path = args.output_dir.resolve() / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    if args.samples_per_task is not None:
        manifest_path = args.output_dir.resolve() / "manifest.jsonl"
        manifest_rows = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        selected_rows = [
            row
            for task in protocol["tasks"]
            for row in [r for r in manifest_rows if r["task"] == task][
                : args.samples_per_task
            ]
        ]
        with manifest_path.open("w", encoding="utf-8") as handle:
            for row in selected_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        for method in protocol["methods"]:
            for task in protocol["tasks"]:
                prediction_path = (
                    args.output_dir.resolve()
                    / "pred"
                    / method
                    / task
                    / "predictions.jsonl"
                )
                records = [
                    json.loads(line)
                    for line in prediction_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                with prediction_path.open("w", encoding="utf-8") as handle:
                    for record in records[: args.samples_per_task]:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        counts = {
            task: min(
                args.samples_per_task,
                sum(1 for row in manifest_rows if row["task"] == task),
            )
            for task in protocol["tasks"]
        }
        protocol["sample_limit_per_task"] = args.samples_per_task
        protocol["sample_manifest"]["sha256"] = sha256(manifest_path)
        protocol["sample_manifest"]["total_samples"] = len(selected_rows)
        protocol["sample_manifest"]["counts"] = counts
    for method, config_path in EXTRA_CONFIGS.items():
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        protocol["methods"][method] = {
            "path": str(config_path.resolve()),
            "sha256": sha256(config_path),
            "config": json.loads(config_path.read_text(encoding="utf-8")),
            "reused": False,
        }

    protocol["matrix"] = {
        "selector_methods": list(EXTRA_CONFIGS),
        "existing_methods": [
            "full",
            "quest_4pct",
            "quest_8pct",
            "prob_sum",
            "quest_twi",
            "group_max",
        ],
        "ds_downstream_budget": "fixed token_budget=8192 for ds_twi and ds_gas",
        "ds_rate_baselines": "dynamic sparsity_rate=0.04 and 0.08 for ds_4pct and ds_8pct",
    }
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote selector matrix protocol to {protocol_path}")


if __name__ == "__main__":
    main()
