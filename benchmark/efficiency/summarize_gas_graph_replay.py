#!/usr/bin/env python3
"""Validate and summarize paired GAS CUDA-graph replay results."""
import argparse
import csv
import json
import statistics
from pathlib import Path

from bench_e6_qasper_trace import write_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.input_dir / "config.json").read_text())
    with (args.input_dir / "results.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    expected = {(trace, str(batch), method) for trace in config["files"]
                for batch in config["batch_sizes"] for method in config["methods"]}
    actual = {(r["trace"], r["batch"], r["method"]) for r in rows}
    if len(rows) != len(expected) or actual != expected:
        raise ValueError(f"incomplete/duplicate results: rows={len(rows)} expected={len(expected)}")
    if any(int(r["samples"]) != config["repeats"] * config["rounds"] for r in rows):
        raise ValueError("unexpected number of timing samples")
    table = {(r["trace"], r["batch"], r["method"]): float(r["median_us"]) for r in rows}
    cohorts = {"all": config["files"], "nonpilot_81": [name for name in config["files"]
                if not (int(name[4:7]) in (1, 8, 14) and int(name[13:15]) in (3, 15, 29))]}
    if len(config["files"]) == 90 and len(cohorts["nonpilot_81"]) != 81:
        raise ValueError("unexpected non-pilot trace count")
    summary, pairs = [], []
    for cohort, traces in cohorts.items():
        if not traces:
            continue
        for batch in map(str, config["batch_sizes"]):
            for method in config["methods"]:
                values = [table[(trace, batch, method)] for trace in traces]
                summary.append({"cohort": cohort, "batch": batch, "method": method,
                                "trace_count": len(values), "median_trace_us": statistics.median(values),
                                "mean_trace_us": statistics.mean(values), "min_trace_us": min(values),
                                "max_trace_us": max(values)})
            for baseline, candidate in (("old_fused", "warp_fused"), ("old_fused", "split8"),
                                        ("old_fused", "vector16_adaptive"),
                                        ("old_fused", "gas_hybrid"),
                                        ("flashinfer_quest", "flashinfer_exact"),
                                        ("flashinfer_twilight", "flashinfer_exact"),
                                        ("flashinfer_quest", "vector16_adaptive"),
                                        ("flashinfer_twilight", "vector16_adaptive"),
                                        ("flashinfer_quest", "gas_hybrid"),
                                        ("flashinfer_twilight", "gas_hybrid"),
                                        ("flashinfer_exact", "gas_hybrid"),
                                        ("flashinfer_exact", "vector16_adaptive"),
                                        ("flashinfer_exact", "warp_fused"),
                                        ("flashinfer_exact", "split8")):
                if baseline not in config["methods"] or candidate not in config["methods"]:
                    continue
                ratio = [table[(t, batch, baseline)] / table[(t, batch, candidate)] for t in traces]
                pairs.append({"cohort": cohort, "batch": batch, "baseline": baseline, "candidate": candidate,
                              "trace_count": len(ratio), "median_baseline_over_candidate": statistics.median(ratio),
                              "candidate_faster_count": sum(r > 1 for r in ratio),
                              "candidate_faster_fraction": sum(r > 1 for r in ratio) / len(ratio)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "summary.csv", summary)
    write_rows(args.output_dir / "paired.csv", pairs)
    lookup = {(r["cohort"], r["batch"], r["method"]): r["median_trace_us"] for r in summary}
    lines = ["# GAS CUDA-graph replay", "", "All times are microseconds per attention call, "
             "the median across per-trace medians of graph-replay averages. "
             "Each batch physically repeats the same Qasper sample; all masks come from the GAS trajectory. "
             "The 81 non-pilot traces exclude the nine pilot layer/step combinations, but are not independent samples or a blind holdout: earlier complete-run results were inspected.", "",
             "Selection, metadata construction, planning, tensor layout conversion, cache updates, prefill and model layers are excluded. "
             "The FlashInfer rows use a per-head exact-mask adapter, not the official Quest/Twilight kernel or a group-shared kernel.", ""]
    for cohort in cohorts:
        if not cohorts[cohort]:
            continue
        lines += [f"## {cohort}", "", "| Batch | " + " | ".join(config["methods"]) + " |",
                  "|---:|" + "---:|" * len(config["methods"])]
        for batch in map(str, config["batch_sizes"]):
            lines.append("| " + batch + " | " + " | ".join(f"{lookup[(cohort, batch, m)]:.2f}" for m in config["methods"]) + " |")
        lines.append("")
    lines += ["Paired per-trace ratios and win counts: `paired.csv`. "
              "Do not equate logical support with measured DRAM traffic or cache capacity. "
              "This is neither real model TPOT nor a 16K/32K-context evaluation."]
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(summary)} summary rows and {len(pairs)} paired rows")


if __name__ == "__main__":
    main()
