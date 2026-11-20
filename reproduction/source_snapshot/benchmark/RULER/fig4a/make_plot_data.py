#!/usr/bin/env python3
"""Create paper plotting data from the completed Fig. 4(a) LLaMA run."""

import argparse
import csv
from pathlib import Path


LABELS = {
    "full": ("Full attention", "full_reference", "reference"),
    "gas": ("GAS", "gas_highlight", "reference"),
    "quest_8pct": ("Quest (8%)", "selector_baseline", "formal"),
    "ds_8pct": ("DS (8%)", "selector_baseline", "formal"),
    "twilight_default": ("Twilight (default)", "twilight_default", "formal"),
    "quest_cal": ("Quest (KV-calibrated)", "kv_matched", "in_range"),
    "ds_cal": ("DS (KV-calibrated)", "kv_matched", "nearest_above"),
    "twilight_cal": ("Twilight (KV-calibrated)", "kv_matched", "nearest_above"),
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    summary = {row["method_id"]: row for row in read_csv(args.root / "fig4a_summary.csv")}
    pairwise = {row["method_id"]: row for row in read_csv(args.root / "pairwise_statistics.csv")}
    gas_score = float(summary["gas"]["macro_score"])
    gas_kv = float(summary["gas"]["mean_loaded_KV"])
    rows = []
    for method_id in LABELS:
        row = summary[method_id]
        label, point_kind, match_status = LABELS[method_id]
        pair = pairwise.get(method_id, {})
        mean_kv = row["mean_loaded_KV"]
        rows.append(
            {
                "method_id": method_id,
                "display_label": label,
                "selector": row["selector"],
                "strategy": row["strategy"],
                "point_kind": point_kind,
                "budget_type": row["budget_type"],
                "budget_value": row["budget_value"],
                "effective_candidates": row["effective_candidates"],
                "p_t": row["p_t"],
                "p_g": row["p_g"],
                "m": row["m"],
                "n_samples": row["n_samples"],
                "macro_score": row["macro_score"],
                "mean_loaded_KV": mean_kv,
                "KV_ratio_to_GAS": row["KV_ratio_to_GAS"],
                "KV_delta_pct_vs_GAS": 100.0 * (float(row["KV_ratio_to_GAS"]) - 1.0),
                "score_delta_vs_GAS": pair.get("score_delta_vs_GAS", "0.0"),
                "score_delta_ci95_low": pair.get("ci95_low", "0.0"),
                "score_delta_ci95_high": pair.get("ci95_high", "0.0"),
                "match_status": match_status,
                "source_path": row["source_path"],
            }
        )

    output = args.root / "Fig4a_Plot_Data.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Fig. 4(a) LLaMA Plot Data",
        "",
        "Model: Meta-Llama-3.1-8B-Instruct; context: 32K; five RULER tasks; 250 formal samples per method.",
        "The x-axis is mean group-level loaded KV and the y-axis is macro RULER score.",
        "Full attention is a quality reference; GAS is the highlighted method.",
        "",
        "| Method | Macro score | Mean loaded KV | KV ratio to GAS | Score delta vs GAS | Match status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f'| {row["display_label"]} | {float(row["macro_score"]):.2f} | '
            f'{float(row["mean_loaded_KV"]):.2f} | {float(row["KV_ratio_to_GAS"]):.3f} | '
            f'{float(row["score_delta_vs_GAS"]):+.2f} | {row["match_status"]} |'
        )
    lines.extend(
        [
            "",
            f"GAS reference: score={gas_score:.2f}, mean loaded KV={gas_kv:.2f}.",
            "`in_range` means the calibration candidate was within 1.00--1.05 of the calibration GAS target.",
            "`nearest_above` means no candidate was in range; the closest measured candidate above the target was selected.",
        ]
    )
    (args.root / "Fig4a_Plot_Data.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} plotting rows to {output}")


if __name__ == "__main__":
    main()
