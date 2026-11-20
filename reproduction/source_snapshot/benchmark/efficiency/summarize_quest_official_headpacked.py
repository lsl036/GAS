#!/usr/bin/env python3
"""Validate and summarize the official Quest B1 grouped replay."""
import argparse
import csv
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.input_dir / "config.json").read_text())
    with (args.input_dir / "results.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = {(str(s), str(l), str(b), m) for s in config["steps"] for l in config["layers"]
                for b in config["batch_sizes"] for m in config["methods"]}
    actual = {(r["step"], r["layer"], r["batch"], r["method"]) for r in rows}
    if len(rows) != len(expected) or actual != expected:
        raise ValueError("incomplete or duplicate results")
    if any(int(r["samples"]) != config["repeats"] * config["rounds"] for r in rows):
        raise ValueError("sample count mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    values = {(r["step"], r["layer"], r["batch"], r["method"]): float(r["median_us"]) for r in rows}
    summary = []
    for batch in map(str, config["batch_sizes"]):
        for method in config["methods"]:
            points = [values[(str(s), str(l), batch, method)] for s in config["steps"] for l in config["layers"]]
            calls = [int(r["quest_calls"]) for r in rows if r["batch"] == batch and r["method"] == method]
            summary.append({"batch": batch, "method": method, "trace_count": len(points),
                            "median_trace_us": statistics.median(points), "min_trace_us": min(points),
                            "max_trace_us": max(points), "min_quest_calls": min(calls), "max_quest_calls": max(calls)})
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    lookup = {(r["batch"], r["method"]): r["median_trace_us"] for r in summary}
    quest_name = config["methods"][0]
    lines = ["# Quest B1 grouped adapter pilot", "", f"Module {config['module_name']}; patched modules are not unmodified official kernels. "
             "FP16, four captured Qasper requests repeated within batch. "
             "CUDA Graph microseconds, median of layer/step medians. This is not native batch: Quest's B1 kernel is called "
             "multiple times after grouping by page count and final-page length, with timed output scatter. "
             "Query gathering, KV duplication/repacking, selection and plans are excluded. "
             "Other methods are per-head FlashInfer exact-mask adapters; all masks derive from GAS-run states.", "",
             "| Batch | " + " | ".join(config["methods"]) + " | Quest calls (min–max) |",
             "|---:|" + "---:|" * (len(config["methods"]) + 1)]
    for batch in map(str, config["batch_sizes"]):
        item = next(r for r in summary if r["batch"] == batch and r["method"] == quest_name)
        lines.append("| " + batch + " | " + " | ".join(f"{lookup[(batch, m)]:.2f}" for m in config["methods"]) +
                     f" | {item['min_quest_calls']}–{item['max_quest_calls']} |")
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(summary)} summary rows")


if __name__ == "__main__":
    main()
