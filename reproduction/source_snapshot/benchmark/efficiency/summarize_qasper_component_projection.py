#!/usr/bin/env python3
"""Construct an explicitly frontend-excluded, per-layer component projection."""
import argparse
import csv
import json
import statistics
from pathlib import Path


SPARSE = ("fi_quest_adapter", "fi_twilight_adapter", "fi_gas_adapter", "inline8", "twopass_inline8", "prefix_groupsplit128")
METHODS = ("fa2_dense", "fi_dense") + SPARSE


def combine(attn, common, batches, steps, less=None):
    layers = list(range(2, 32))
    a = {(str(r["step"]), str(r["layer"]), str(r["batch"]), r["method"]): float(r["median_us"]) for r in attn}
    c = {(str(r["layer"]), str(r["batch"])): float(r["median_us"]) for r in common}
    less_values = {(str(r["step"]), str(r["layer"]), str(r["batch"]), r["method"]): float(r["median_us"]) for r in (less or [])}
    if len(a) != len(attn) or len(c) != len(common) or len(less_values) != len(less or []):
        raise ValueError("duplicate component rows")
    rows = []
    for step in map(str, steps):
        for batch in map(str, batches):
            total_common = sum(c[(str(layer), batch)] for layer in range(32))
            output = c[("output", batch)]
            dense_mean = {m: statistics.fmean(a[(step, str(layer), batch, m)] for layer in layers)
                          for m in ("fa2_dense", "fi_dense")}
            for method in METHODS:
                measured = sum(a[(step, str(layer), batch, method)] for layer in layers)
                first_two = 2 * dense_mean[method if method in dense_mean else "fi_dense"]
                total = total_common + output + measured + first_two
                rows.append({"step": step, "batch": batch, "method": method,
                             "common_32_layers_ms": total_common / 1000, "output_ms": output / 1000,
                             "attention_measured_30_layers_ms": measured / 1000,
                             "first_two_dense_estimate_ms": first_two / 1000,
                             "selection_cache_metadata_ms": "excluded",
                             "frontend_excluded_step_projection_ms": total / 1000})
            for method in (("less_patched_headpacked", "less_fi_adapter") if less is not None else ()):
                sparse_layers = [x for x in layers if x not in (2, 13)]
                measured = sum(less_values[(step, str(layer), batch, method)] for layer in sparse_layers)
                measured += a[(step, "2", batch, "fi_dense")] + a[(step, "13", batch, "fi_dense")]
                first_two = 2 * dense_mean["fi_dense"]
                total = total_common + output + measured + first_two
                rows.append({"step": step, "batch": batch, "method": method,
                             "common_32_layers_ms": total_common / 1000, "output_ms": output / 1000,
                             "attention_measured_30_layers_ms": measured / 1000,
                             "first_two_dense_estimate_ms": first_two / 1000,
                             "selection_cache_metadata_ms": "excluded",
                             "frontend_excluded_step_projection_ms": total / 1000})
    for row in rows:
        baseline = next(x for x in rows if x["step"] == row["step"] and x["batch"] == row["batch"] and x["method"] == "fi_dense")
        row["speedup_vs_fi_dense"] = baseline["frontend_excluded_step_projection_ms"] / row["frontend_excluded_step_projection_ms"]
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-dir", type=Path, required=True)
    parser.add_argument("--common-dir", type=Path, required=True)
    parser.add_argument("--less-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    ac = json.loads((args.attention_dir / "config.json").read_text())
    cc = json.loads((args.common_dir / "config.json").read_text())
    if ac["dtype"] != "float16" or cc["dtype"] != "float16" or ac["device"] != cc["device"]:
        raise ValueError("attention/common dtype or GPU mismatch")
    if sorted(ac["layers"]) != list(range(2, 32)) or sorted(cc["layers"]) != list(range(32)):
        raise ValueError("30 attention layers and 32 common layers required")
    if sorted(ac["batch_sizes"]) != sorted(cc["batch_sizes"]):
        raise ValueError("batch sizes differ")
    if not set(METHODS).issubset(ac["methods"]):
        raise ValueError("required attention methods missing")
    for path in ac["trace_dirs"]:
        manifest = json.loads((Path(path) / "manifest.json").read_text())
        if str(Path(manifest["model"]).resolve()) != cc["model"]:
            raise ValueError("trace model differs from common checkpoint")
    with (args.attention_dir / "results.csv").open(newline="", encoding="utf-8") as stream:
        attn = list(csv.DictReader(stream))
    with (args.common_dir / "results.csv").open(newline="", encoding="utf-8") as stream:
        common = list(csv.DictReader(stream))
    if len(attn) != len(ac["steps"]) * 30 * len(ac["batch_sizes"]) * len(ac["methods"]):
        raise ValueError("incomplete attention results")
    if len(common) != 33 * len(cc["batch_sizes"]):
        raise ValueError("incomplete common results")
    less = None
    if args.less_dir is not None:
        lc = json.loads((args.less_dir / "config.json").read_text())
        if (lc["dtype"] != "float16" or lc["device"] != ac["device"] or lc["model"] != cc["model"] or
                lc["manifest_sha256"] != ac["manifest_sha256"] or sorted(lc["steps"]) != sorted(ac["steps"]) or
                sorted(lc["batch_sizes"]) != sorted(ac["batch_sizes"]) or
                sorted(lc["layers"]) != [x for x in range(3, 32) if x != 13]):
            raise ValueError("LessIsMore protocol does not match attention traces")
        with (args.less_dir / "results.csv").open(newline="", encoding="utf-8") as stream:
            less = list(csv.DictReader(stream))
        if len(less) != len(lc["steps"]) * 28 * len(lc["batch_sizes"]) * 2:
            raise ValueError("incomplete LessIsMore results")
    rows = combine(attn, common, ac["batch_sizes"], ac["steps"], less)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "frontend_excluded_projection.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ["# Qasper frontend-excluded component projection", "", "Not measured end-to-end TPOT. "
             "Sum of 32 individually measured real-weight common layer medians on synthetic inputs, "
             "output-component median, and 30 attention medians from repeated four-request Qasper batches. "
             "First two dense attention layers are estimated as twice the measured 30-layer mean. "
             "Selection, estimator, compaction, cache update, layout conversion, plans, scheduling and communication are excluded; "
             "CUDA Graph and individual-layer residency are used.", "",
             "Quest/Twilight are exact-mask FI adapters, not official native kernels. " +
             ("LessIsMore uses an experimental patched B1 head-packed kernel and FI dense search layers 2 and 13. " if less is not None else "LessIsMore is unavailable in this projection. ") +
             "The source model trajectory is GAS; all sparse masks are counterfactual on those Q/K/V. "
             "These estimates cannot prove real serving latency, model fit or a total TPOT speedup.", "",
             "| Step | Batch | Method | Common ms | Output ms | 30-layer attention ms | Two dense est. ms | Projection ms | vs FI dense |",
             "|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['step']} | {r['batch']} | {r['method']} | {r['common_32_layers_ms']:.3f} | {r['output_ms']:.3f} | "
                     f"{r['attention_measured_30_layers_ms']:.3f} | {r['first_two_dense_estimate_ms']:.3f} | "
                     f"{r['frontend_excluded_step_projection_ms']:.3f} | {r['speedup_vs_fi_dense']:.2f}x |")
    (args.output_dir / "frontend_excluded_projection.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(rows)} component projection rows")


if __name__ == "__main__":
    main()
