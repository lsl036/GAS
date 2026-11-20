#!/usr/bin/env python3
"""Summarize correctness-gated Qasper kernel replays without hiding failures."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    low = int(index)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (index - low)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--union-dir", type=Path)
    parser.add_argument("--official-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.replay_dir / "config.json").read_text())
    rows = read_csv(args.replay_dir / "trace_results.csv")
    expected = {(name, str(batch), method) for name in config["trace_files"]
                for batch in config["batch_sizes"] for method in config["methods"]}
    actual = [(row["trace"], row["batch"], row["method"]) for row in rows]
    if len(set(actual)) != len(actual):
        raise ValueError("duplicate trace/batch/method rows")
    if set(actual) != expected:
        raise ValueError(f"incomplete or unexpected matrix: missing={len(expected-set(actual))}, extra={len(set(actual)-expected)}")
    trace_hashes = {}
    for row in rows:
        previous = trace_hashes.setdefault(row["trace"], row["trace_sha256"])
        if previous != row["trace_sha256"]:
            raise ValueError(f"base trace hash changed during replay: {row['trace']}")
    if args.union_dir:
        union_config = json.loads((args.union_dir / "config.json").read_text())
        if union_config["trace_manifest_sha256"] != config["trace_manifest_sha256"]:
            raise ValueError("union and base trace manifests differ")
        union_rows = read_csv(args.union_dir / "trace_results.csv")
        union_expected = {(name, str(batch), method) for name in union_config["trace_files"]
                          for batch in union_config["batch_sizes"] for method in union_config["methods"]}
        union_actual = [(row["trace"], row["batch"], row["method"]) for row in union_rows]
        if len(set(union_actual)) != len(union_actual) or set(union_actual) != union_expected:
            raise ValueError("union-prototype matrix is incomplete or duplicated")
        if set(union_config["trace_files"]) != set(config["trace_files"]) or set(union_config["batch_sizes"]) != set(config["batch_sizes"]):
            raise ValueError("union and base trace/batch sets differ")
        if any(row["trace_sha256"] != trace_hashes[row["trace"]] for row in union_rows):
            raise ValueError("union and base trace file hashes differ")
        rows.extend(union_rows)
    official_availability = []
    if args.official_dir:
        for method in ("quest", "less"):
            path = args.official_dir / f"{method}.results.csv"
            official_config = json.loads((args.official_dir / f"{method}.config.json").read_text())
            official_rows = read_csv(path)
            if official_config["trace_manifest_sha256"] != config["trace_manifest_sha256"]:
                raise ValueError(f"official {method} trace manifest differs")
            official_names = [row["trace"] for row in official_rows]
            if len(set(official_names)) != len(official_names) or set(official_names) != set(config["trace_files"]):
                raise ValueError(f"official {method} trace matrix is incomplete or duplicated")
            if any(row["trace_sha256"] != trace_hashes[row["trace"]] for row in official_rows):
                raise ValueError(f"official {method} trace file hashes differ")
            official_availability.append({"method": "official_" + method, "batch": 1,
                                          "status": "measured_with_trace_exclusions",
                                          "available": sum(r["status"] == "available" for r in official_rows),
                                          "unavailable": sum(r["status"] == "unavailable" for r in official_rows),
                                          "failed": sum(r["status"] == "failed" for r in official_rows),
                                          "reason": "see per-trace results"})
            for batch in config["batch_sizes"]:
                if batch != 1:
                    official_availability.append({"method": "official_" + method, "batch": batch,
                                                  "status": "unsupported_upstream_batch", "available": 0,
                                                  "unavailable": len(config["trace_files"]), "failed": 0,
                                                  "reason": "upstream decode wrapper hardcodes batch size 1"})
            for row in official_rows:
                row["method"] = "official_" + method
                rows.append(row)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], int(row["batch"]))].append(row)
    summary = []
    for (method, batch), group in sorted(grouped.items()):
        passed = [row for row in group if row["status"] == "available"]
        medians = [float(row["median_us"]) for row in passed]
        unions = [float(row["group_union_mean"]) for row in passed if row.get("group_union_mean")]
        widths = [float(row["padded_union_width"]) for row in passed if row.get("padded_union_width")]
        calls = [int(row["quest_call_groups"]) for row in passed if row.get("quest_call_groups")]
        summary.append({"method": method, "batch": batch, "traces_total": len(group), "traces_passed": len(passed),
                        "traces_failed": sum(row["status"] == "failed" for row in group),
                        "traces_unavailable": sum(row["status"] == "unavailable" for row in group),
                        "trace_median_us": statistics.median(medians) if medians else "",
                        "trace_mean_us": statistics.fmean(medians) if medians else "",
                        "trace_p10_us": percentile(medians, .1) if medians else "",
                        "trace_p90_us": percentile(medians, .9) if medians else "",
                        "mean_group_union": statistics.fmean(unions) if unions else "",
                        "mean_padded_union_width": statistics.fmean(widths) if widths else "",
                        "mean_quest_call_groups": statistics.fmean(calls) if calls else ""})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if official_availability:
        with (args.output_dir / "official_availability.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(official_availability[0]))
            writer.writeheader()
            writer.writerows(official_availability)
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    paired = []
    for batch in sorted({int(row["batch"]) for row in rows}):
        base = {row["trace"]: row for row in rows if row["method"] == "gas_fused" and int(row["batch"]) == batch and row["status"] == "available"}
        for method in sorted({row["method"] for row in rows if int(row["batch"]) == batch} - {"gas_fused"}):
            matches = [row for row in rows if row["method"] == method and int(row["batch"]) == batch
                       and row["status"] == "available" and row["trace"] in base]
            ratios = [float(row["median_us"]) / float(base[row["trace"]]["median_us"]) for row in matches]
            paired.append({"batch": batch, "baseline": method, "pair_count": len(ratios),
                           "median_baseline_over_gas": statistics.median(ratios) if ratios else "",
                           "p10": percentile(ratios, .1) if ratios else "", "p90": percentile(ratios, .9) if ratios else "",
                           "same_mask": method in ("gas_per_head", "gas_fi_exact")})
    with (args.output_dir / "paired_speedups.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    if args.union_dir:
        union_pairs = []
        for batch in config["batch_sizes"]:
            for baseline, prototype in (("quest_fi_exact", "quest_union_masked_proto"),
                                        ("twilight_fi_exact", "twilight_union_masked_proto")):
                refs = {r["trace"]: r for r in rows if r["method"] == baseline and int(r["batch"]) == batch and r["status"] == "available"}
                candidates = [r for r in rows if r["method"] == prototype and int(r["batch"]) == batch
                              and r["status"] == "available" and r["trace"] in refs]
                ratios = [float(refs[r["trace"]]["median_us"]) / float(r["median_us"]) for r in candidates]
                union_pairs.append({"batch": batch, "baseline": baseline, "prototype": prototype,
                                    "pair_count": len(ratios), "median_baseline_over_prototype": statistics.median(ratios) if ratios else "",
                                    "p10": percentile(ratios, .1) if ratios else "", "p90": percentile(ratios, .9) if ratios else ""})
        with (args.output_dir / "union_strategy_pairs.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(union_pairs[0]))
            writer.writeheader()
            writer.writerows(union_pairs)
    lines = ["# Qasper Real-Trace E6 Kernel Replay", "", 
             "One Qasper sample, 30 sparse layers, three question decode steps; B>1 independently copies the same request.",
             "Original model trace is BF16; Q/K/V are cast to FP16 outside timing. All times are attention-only CUDA-event microseconds.",
             "Selection, compaction, layout conversion, library planning, prefill, projections and service costs are excluded.",
             "", "| Method | Batch | Passed / total | Median trace latency (us) | Trace p10 / p90 (us) | Mean group union |",
             "|---|---:|---:|---:|---:|---:|"]
    def fmt(value):
        return "N/A" if value == "" else f"{float(value):.2f}"
    for row in summary:
        lines.append(f"| {row['method']} | {row['batch']} | {row['traces_passed']}/{row['traces_total']} | "
                     f"{fmt(row['trace_median_us'])} | {fmt(row['trace_p10_us'])} / {fmt(row['trace_p90_us'])} | {fmt(row['mean_group_union'])} |")
    lines += ["", "- Quest/Twilight FlashInfer exact-support rows use page size 1 and B*32 logical requests. They are compatible controls, not author kernels, and do not implement shared group-union loading or Quest's page-16 metadata.",
              "- Repeats reuse the same buffers and fixed indices; warm-cache effects are not measured or controlled, especially for B1.",
              "- Quest/Twilight union-masked prototype rows load group-union KV tiles with per-head membership, and are in-house implementations rather than author kernels.",
              "- GAS per-head and GAS FlashInfer exact-support rows use exactly the same final head masks as GAS fused; their paired timings help isolate execution strategy.",
              "- Twilight author-dense-ops replays the author's quality-path dense QK/mask/softmax/PV operations, with repeated KV layout prepared outside timing; it is not an optimized sparse kernel.",
              "- Official Quest B1 uses physically repeated KV heads; partial pages may require multiple official calls and timed output scatter. Packing/plan are excluded. It is not native GQA or a B32-B128 measurement.",
              "- Official LessIsMore B1 uses global shared supports calibrated to K=91 and a 25% recent window. Dense search layers are not sparse-attention rows.",
              "- Official B1 wrappers reuse preallocated outputs; other wrappers return allocated outputs. CUDA-event intervals may include GPU idle gaps from host dispatch and are not isolated profiler kernel durations.",
              "- Differences between methods combine support and implementation; do not infer task accuracy, measured KV residency or end-to-end latency.",
              "- Report failed or unavailable traces; method averages with different passed trace sets are not matched comparisons. See paired_speedups.csv for trace-matched ratios."]
    (args.output_dir / "E6_Qasper_Result_Summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(summary)} summary rows to {args.output_dir}")


if __name__ == "__main__":
    main()
