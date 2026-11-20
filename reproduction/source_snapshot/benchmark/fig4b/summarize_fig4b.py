#!/usr/bin/env python3
"""Aggregate Fig. 4(b) Passkey accuracy and GQA KV metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean


TASKS = ("quest", "ds")
MODELS = ("llama31", "qwen3")
LENGTHS = ("8k", "16k")


def records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def profile_means(path: Path, prefix: str) -> dict[str, float | None]:
    rows = records(path)
    values = {}
    for name in (
        "per_head_selected_mean",
        "union_amplification_mean",
        "loaded_kv_tokens_mean",
    ):
        samples = []
        key = f"{prefix}{name}" if prefix else name
        for row in rows:
            value = row.get("gqa", {}).get(key)
            if isinstance(value, (int, float)):
                samples.append(float(value))
        values[name] = fmean(samples) if samples else None
    values["n_profiled"] = len(rows)
    return values


def cell_summary(root: Path, label: str, strategy: str) -> dict:
    path = root / f"{label}_{strategy}"
    if not all((path / name).is_file() for name in ("metadata.json", "predictions.jsonl", "profile.jsonl")):
        return None
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    predictions = records(path / "predictions.jsonl")
    if len(predictions) != 50:
        return None
    correct = sum(bool(row.get("correct")) for row in predictions)
    return {
        "path": path,
        "metadata": metadata,
        "n_expected": len(predictions),
        "n_scored": len(predictions),
        "correct": correct,
        "accuracy": correct / len(predictions) if predictions else None,
    }


def fmt(value) -> str:
    if isinstance(value, str):
        return value
    return "" if value is None else f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results_passkey_ours/fig4b_generality",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    rows = []
    for model in MODELS:
        for selector in TASKS:
            for length in LENGTHS:
                label = f"{model}_{selector}_{length}"
                headwise = cell_summary(root, label, "headwise")
                gas = cell_summary(root, label, "gas")
                head_profile = (
                    profile_means(headwise["path"] / "profile.jsonl", "final_")
                    if headwise
                    else {"per_head_selected_mean": None, "n_profiled": None}
                )
                gas_pre = (
                    profile_means(gas["path"] / "profile.jsonl", "pre_")
                    if gas
                    else {"loaded_kv_tokens_mean": None, "union_amplification_mean": None, "n_profiled": None}
                )
                gas_post = (
                    profile_means(gas["path"] / "profile.jsonl", "post_")
                    if gas
                    else {"loaded_kv_tokens_mean": None, "union_amplification_mean": None}
                )
                before = gas_pre["loaded_kv_tokens_mean"]
                after = gas_post["loaded_kv_tokens_mean"]
                reduction = None if before in (None, 0) or after is None else 1.0 - after / before
                rows.append(
                    {
                        "model": model,
                        "selector": selector,
                        "context_tokens": {"8k": 8192, "16k": 16384}[length],
                        "candidate_budget_ratio": 0.08,
                        "effective_candidates_mean": head_profile["per_head_selected_mean"],
                        "p_t": 0.85,
                        "p_g": 0.80,
                        "m": 16,
                        "aggregation": "group_max",
                        "n_expected": 50,
                        "n_scored_headwise": headwise["n_scored"] if headwise else "-",
                        "n_scored_GAS": gas["n_scored"] if gas else "-",
                        "n_profiled": min(head_profile["n_profiled"], gas_pre["n_profiled"])
                        if headwise and gas
                        else "-",
                        "headwise_correct": headwise["correct"] if headwise else "-",
                        "GAS_correct": gas["correct"] if gas else "-",
                        "headwise_accuracy": headwise["accuracy"] if headwise else "-",
                        "GAS_accuracy": gas["accuracy"] if gas else "-",
                        "loaded_KV_before": before if gas else "-",
                        "loaded_KV_after": after if gas else "-",
                        "reduction": reduction if headwise and gas else "-",
                        "amplification_before": gas_pre["union_amplification_mean"] if gas else "-",
                        "amplification_after": gas_post["union_amplification_mean"] if gas else "-",
                        "source_path": str(root / label),
                    }
                )

    args.output = root / "fig4b_summary.csv"
    fields = list(rows[0]) if rows else ["model", "selector", "context_tokens"]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    report = root / "Fig4b_Result_Summary.md"
    lines = [
        "# Fig. 4(b) Cross-Model Selector Matrix",
        "",
        "Protocol: 50 shared cases, seed 1337, 8% candidate budget, head-wise p_t=0.85, GAS group-max p_g=0.80 and exception budget 16.",
        "",
        "| Model | Selector | Context | Head-wise accuracy | GAS accuracy | Before KV | After KV | Reduction |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        reduction_text = (
            row["reduction"]
            if isinstance(row["reduction"], str)
            else f"{row['reduction'] * 100:.2f}%"
        )
        headwise_accuracy = (
            row["headwise_accuracy"]
            if isinstance(row["headwise_accuracy"], str)
            else f"{row['headwise_accuracy'] * 100:.1f}%"
        )
        gas_accuracy = (
            row["GAS_accuracy"]
            if isinstance(row["GAS_accuracy"], str)
            else f"{row['GAS_accuracy'] * 100:.1f}%"
        )
        lines.append(
            f"| {row['model']} | {row['selector']} | {row['context_tokens']} | "
            f"{headwise_accuracy} | {gas_accuracy} | "
            f"{fmt(row['loaded_KV_before'])} | {fmt(row['loaded_KV_after'])} | "
            f"{reduction_text} |"
        )
    lines.extend(
        [
            "",
            "Before and After are computed from the same GAS attention calls; head-wise accuracy is a separate paired baseline.",
            "Missing cells are not treated as zero and must be rerun before making a complete 8-row claim.",
            "",
        ]
    )
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {args.output}")
    print(f"wrote report to {report}")


if __name__ == "__main__":
    main()
