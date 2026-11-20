#!/usr/bin/env python3
"""Summarize the four missing Fig. 4(b) GAS cells using fixed budget 8192."""

import argparse
import csv
import json
from pathlib import Path


CELLS = (
    ("llama31", "ds", "16k", "llama31_ds_16k", 16000),
    ("qwen3", "quest", "16k", "qwen3_quest_16k", 16000),
    ("qwen3", "ds", "8k", "qwen3_ds_8k", 8000),
    ("qwen3", "ds", "16k", "qwen3_ds_16k", 16000),
)


def accuracy(root: Path, label: str, selector: str) -> tuple[int | str, int | str, float | str]:
    path = root / label
    output = path / ("quest_top_p.jsonl" if selector == "quest" else "ds_top_p.jsonl")
    if not output.is_file():
        return "-", "-", "-"
    lines = [line.strip().split(",") for line in output.read_text().splitlines() if line.strip()]
    if len(lines) < 2:
        return "-", "-", "-"
    try:
        value = float(lines[1][-1])
    except ValueError:
        return "-", "-", "-"
    n = 5
    return int(round(value * n)), n, value


def metric(gqa: dict, key: str):
    item = gqa.get(key)
    return item.get("mean_us") if isinstance(item, dict) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results_passkey_ours/fig4b_generality_8192",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    rows = []
    for model, selector, length, label, fixed_length in CELLS:
        summary_path = root / label / "profile_summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            gqa = summary.get("gqa", {})
            before = metric(gqa, "pre_loaded_kv_tokens_mean")
            after = metric(gqa, "post_loaded_kv_tokens_mean")
            reduction = None if before in (None, 0) or after is None else 1.0 - after / before
            amp_before = metric(gqa, "pre_union_amplification_mean")
            amp_after = metric(gqa, "post_union_amplification_mean")
            profiled = gqa.get("pre_loaded_kv_tokens_mean", {}).get("count", "-")
        else:
            before = after = reduction = amp_before = amp_after = None
            profiled = "-"
        correct, scored, acc = accuracy(root, label, selector)
        rows.append(
            {
                "model": model,
                "selector": selector,
                "length_label": length,
                "fixed_length": fixed_length,
                "token_budget": 8192,
                "p_t": 0.85,
                "p_g": 0.80,
                "m": 16,
                "iterations": 5,
                "n_scored": scored,
                "correct": correct,
                "accuracy": acc,
                "loaded_KV_before": before if before is not None else "-",
                "loaded_KV_after": after if after is not None else "-",
                "reduction": reduction if reduction is not None else "-",
                "amplification_before": amp_before if amp_before is not None else "-",
                "amplification_after": amp_after if amp_after is not None else "-",
                "n_profiled": profiled,
                "source_path": str(root / label),
            }
        )

    output = root / "fig4b_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = root / "Fig4b_Result_Summary.md"
    lines = [
        "# Fig. 4(b) Missing GAS Cells, Fixed Budget 8192",
        "",
        "Historical Passkey protocol: fixed_length 8000/16000, iterations=5, p_t=0.85, group-max p_g=0.80, exception budget 16.",
        "",
        "| Model | Selector | Length | Correct | Accuracy | Before KV | After KV | Reduction |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        acc = row["accuracy"] if isinstance(row["accuracy"], str) else f"{row['accuracy'] * 100:.1f}%"
        red = row["reduction"] if isinstance(row["reduction"], str) else f"{row['reduction'] * 100:.2f}%"
        lines.append(
            f"| {row['model']} | {row['selector']} | {row['length_label']} | "
            f"{row['correct']}/{row['n_scored']} | {acc} | {row['loaded_KV_before']} | "
            f"{row['loaded_KV_after']} | {red} |"
        )
    lines.extend(
        [
            "",
            "Before and After are taken from the same GAS profile's pre/post refinement metrics.",
            "A dash means that the corresponding profile or score file is incomplete.",
            "",
        ]
    )
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {output}")
    print(f"wrote report to {report}")


if __name__ == "__main__":
    main()
