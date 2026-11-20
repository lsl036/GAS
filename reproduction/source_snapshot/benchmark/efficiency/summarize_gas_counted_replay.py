#!/usr/bin/env python3
"""Validate and summarize attention and prefix-counting pilot measurements."""
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
    expected = {(trace, str(batch), method) for trace in config["files"] for batch in config["batch_sizes"]
                for method in config["methods"]}
    actual = {(r["trace"], r["batch"], r["method"]) for r in rows}
    if len(rows) != len(expected) or actual != expected:
        raise ValueError("incomplete or duplicate results")
    if any(int(r["samples"]) != config["repeats"] * config["rounds"] for r in rows):
        raise ValueError("sample count mismatch")
    values = {(r["trace"], r["batch"], r["method"]): float(r["median_us"]) for r in rows}
    summary, pairs = [], []
    for batch in map(str, config["batch_sizes"]):
        for method in config["methods"]:
            times = [values[(trace, batch, method)] for trace in config["files"]]
            summary.append({"dtype": config["dtype"], "batch": batch, "method": method,
                            "trace_count": len(times), "median_trace_us": statistics.median(times),
                            "min_trace_us": min(times), "max_trace_us": max(times)})
        for baseline, candidate in (("fi_gas", "counted8"), ("fi_gas", "inline8"),
                                    ("fi_gas", "prefix_plus8"), ("fi_twilight", "inline8"),
                                    ("fi_quest", "inline8"), ("fi_gas", "split8p4"),
                                    ("fi_gas", "prefix_split8p4"), ("fi_twilight", "split8p4"),
                                    ("fi_gas", "persistent128"), ("fi_gas", "prefix_persistent128"),
                                    ("fi_twilight", "persistent128"), ("fi_twilight", "prefix_persistent128"),
                                    ("fi_gas", "adaptive128"), ("fi_gas", "prefix_adaptive128"),
                                    ("fi_twilight", "adaptive128"), ("fi_twilight", "prefix_adaptive128"),
                                    ("fi_gas", "cooperative8"), ("fi_gas", "prefix_cooperative8"),
                                    ("fi_twilight", "cooperative8"), ("fi_twilight", "prefix_cooperative8"),
                                    ("fi_gas", "groupsplit128"), ("fi_gas", "prefix_groupsplit128"),
                                    ("fi_twilight", "groupsplit128"), ("fi_twilight", "prefix_groupsplit128"),
                                    ("fi_gas", "twopass8"), ("fi_gas", "twopass_inline8"),
                                    ("fi_twilight", "twopass8"), ("fi_twilight", "twopass_inline8")):
            if baseline not in config["methods"] or candidate not in config["methods"]:
                continue
            ratio = [values[(trace, batch, baseline)] / values[(trace, batch, candidate)]
                     for trace in config["files"]]
            pairs.append({"batch": batch, "baseline": baseline, "candidate": candidate,
                          "trace_count": len(ratio), "median_baseline_over_candidate": statistics.median(ratio),
                          "candidate_faster_count": sum(x > 1 for x in ratio)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write(args.output_dir / "summary.csv", summary)
    if pairs:
        write(args.output_dir / "paired.csv", pairs)
    lookup = {(r["batch"], r["method"]): r["median_trace_us"] for r in summary}
    lines = ["# Counted GAS replay pilot", "", f"Dtype: {config['dtype']}; {len(config['files'])} traces from one Qasper sample. "
             "B>1 physically duplicates the same request. CUDA Graph event time divided by graph call count; "
             "median across per-trace medians. Repeated buffers are hot; selection and compaction excluded.", "",
             "`counted` uses precomputed counts; `inline` includes binary-search count inside attention; "
             "`prefix_plus` includes separate prefix counting plus attention. `split` includes a partial-state kernel and merge kernel; "
             "`prefix_split` additionally includes prefix counting. `adaptive` includes fixed-grid dynamic splitting and merge, "
             "and `prefix_adaptive` additionally includes prefix counting. `cooperative` reuses shared KV for four query heads in a CTA; "
             "`prefix_cooperative` additionally includes prefix counting. `twopass` stages logits in shared memory; "
             "`twopass_inline` additionally counts prefixes in the kernel. `groupsplit` includes adaptive cooperative partitioning and merge; "
             "`prefix_groupsplit` additionally includes prefix counting. `persistent` includes GPU queue construction, attention and merge; "
             "`prefix_persistent` additionally includes prefix counting. FI plan construction is excluded. `count_prefix` is metadata-only, not attention. "
             "FlashInfer baselines are per-head exact-mask adapters, not original author kernels.", "",
             "| Batch | " + " | ".join(config["methods"]) + " |", "|---:|" + "---:|" * len(config["methods"])]
    for batch in map(str, config["batch_sizes"]):
        lines.append("| " + batch + " | " + " | ".join(f"{lookup[(batch, m)]:.2f}" for m in config["methods"]) + " |")
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(summary)} summary rows and {len(pairs)} paired rows")


if __name__ == "__main__":
    main()
