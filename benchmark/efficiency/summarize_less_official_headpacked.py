#!/usr/bin/env python3
"""Summarize completeness-checked head-packed LessIsMore replay."""
import argparse
import csv
import json
import statistics
from pathlib import Path


def write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.input_dir / "config.json").read_text())
    with (args.input_dir / "results.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = {(str(step), str(batch), method) for step in config["steps"]
                for batch in config["batch_sizes"] for method in config["methods"]}
    actual = {(r["step"], r["batch"], r["method"]) for r in rows}
    if len(rows) != len(expected) or actual != expected:
        raise ValueError("incomplete or duplicate results")
    if any(int(r["samples"]) != config["repeats"] * config["rounds"] for r in rows):
        raise ValueError("sample count mismatch")
    values = {(r["step"], r["batch"], r["method"]): float(r["median_us"]) for r in rows}
    summaries = []
    for batch in map(str, config["batch_sizes"]):
        for method in config["methods"]:
            times = [values[(str(step), batch, method)] for step in config["steps"]]
            summaries.append({"batch": batch, "method": method, "step_count": len(times),
                              "median_step_us": statistics.median(times), "min_step_us": min(times), "max_step_us": max(times)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write(args.output_dir / "summary.csv", summaries)
    lookup = {(r["batch"], r["method"]): r["median_step_us"] for r in summaries}
    lines = ["# LessIsMore head-packed FP16 pilot", "", "Only layer 3, with layer 2 search, "
             f"at steps {config['steps']}. Four captured requests are repeated in each batch. "
             "CUDA Graph event times in microseconds, median of per-step medians. Selection, packing and plans are excluded.", "",
             f"LessIsMore module `{config['module_name']}` uses a B1 wrapper: groups of up to {config['pack_size']} requests are mapped into heads with a repacked KV layout; multiple calls are timed. Patched modules are not unmodified official kernels. "
             "This is not native batching or measured TPOT. Quest and Twilight are FlashInfer exact-mask adapters, "
             "not their official kernels. All methods use FP16; K is matched to the mean Twilight group union.", "",
             "| Batch | " + " | ".join(config["methods"]) + " |", "|---:|" + "---:|" * len(config["methods"])]
    for batch in map(str, config["batch_sizes"]):
        lines.append("| " + batch + " | " + " | ".join(f"{lookup[(batch, m)]:.2f}" for m in config["methods"]) + " |")
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(summaries)} summary rows")


if __name__ == "__main__":
    main()
