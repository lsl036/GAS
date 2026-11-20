#!/usr/bin/env python3
"""Run and summarize an exploratory 16K Qasper sparse-path breakdown.

The connected prototype and the previously measured attention-only projection
are intentionally reported separately; they do not share a timing backend.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmark/efficiency/bench_attention_path_32k.py"
TRACE = ROOT / "benchmark/results_efficiency/fig6_traces/qasper16k_rate08_s15"
PROJECTION = ROOT / "benchmark/results_efficiency/qasper_rate08_tpot_tuned_projection_20261113/projection.csv"
DEFAULT_OUTPUT = ROOT / "benchmark/results_efficiency/qasper_rate08_breakdown_16k_20261113"
BATCHES = (32, 64, 96, 128)
METHODS = ("quest", "quest_twi", "gas", "less")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def static_inputs():
    rows = {}
    with PROJECTION.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["context_k"] != "16" or row["dense_fi_config"] != "fi_page16_tc1":
                continue
            method = row["method"]
            if method in ("quest", "twilight", "gas", "less_style_8pct"):
                rows[(int(row["batch"]), method, row["sparse_backend"])] = row
    return rows


def summarize(output):
    static = static_inputs()
    stage_rows, model_rows = [], []
    names = {"quest": "quest", "quest_twi": "twilight", "gas": "gas", "less": "less_style_8pct"}
    stage_keys = ("selector", "estimator_cached_fake_quant_qk", "top_p", "gas_refinement",
                  "less_search_qk_cross_head_selection", "mask_to_fi_metadata", "fi_plan", "attention")
    for batch in BATCHES:
        for method in METHODS:
            path = output / "measurements" / f"{method}_b{batch}_s15.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            backend = "per_head" if method == "gas" else "grouped"
            source = static[(batch, names[method], backend)]
            grouped_source = static[(batch, names[method], "grouped")]
            stages = data["stages"]
            row = {"batch": batch, "method": method, "length": data["length"],
                   "repeats": len(data["raw_runs"]),
                   "candidate_mask_diff_first": data["selector_vs_quality_token_differences_first_request"],
                   "final_mask_diff_first": data["final_mask_vs_quality_token_differences_first_request"],
                   "fp32_reference_pass": data["fp32_attention_reference_check_first_request"]["pass"],
                   "selected_per_head_mean": data["final_per_head_tokens_mean"],
                   "group_union_mean": data["final_group_union_mean"]}
            for stage in stage_keys:
                row[stage + "_wall_us"] = stages.get(stage, {}).get("wall_us")
                row[stage + "_event_us"] = stages.get(stage, {}).get("event_us")
            row["connected_wall_us"] = data["connected_wall_median_us"]
            row["measurement_json"] = str(path)
            stage_rows.append(row)

            dense = float(source["dense_fi_one_layer_us"])
            attn = float(source["one_layer_us"])
            # This is the user's attention-only projection, not a pipeline timing.
            attention_only = (2 * dense + 30 * attn) / 1000
            connected = data["connected_wall_median_us"]
            if method == "less":
                # Less computes its search support in two layers and reuses it.
                per_sparse = sum(stages[x]["wall_us"] for x in ("mask_to_fi_metadata", "fi_plan", "attention"))
                search = stages["less_search_qk_cross_head_selection"]["wall_us"]
                pipeline = (2 * dense + 30 * per_sparse + 2 * search) / 1000
                search_aware = (4 * dense + 28 * per_sparse + 2 * search) / 1000
            else:
                pipeline = (2 * dense + 30 * connected) / 1000
                search_aware = None
            model_rows.append({"batch": batch, "method": method,
                               "static_sparse_backend": backend,
                               "static_sparse_one_layer_us": attn, "static_dense_fi_one_layer_us": dense,
                               "attention_only_2dense_30sparse_ms": attention_only,
                               "grouped_attention_only_2dense_30sparse_ms":
                                   (2 * dense + 30 * float(grouped_source["one_layer_us"])) / 1000,
                               "prototype_pipeline_2dense_30sparse_ms": pipeline,
                               "less_search_aware_4dense_28sparse_ms": search_aware})
    write_csv(output / "stage_breakdown.csv", stage_rows)
    write_csv(output / "layer_projection.csv", model_rows)
    lines = ["# Qasper 16K 8% Breakdown: Exploratory Prototype", "",
             "Measured on the captured sample 15 layer 15, repeated across the whole batch. "
             "FP16 replay; one physical input replicated, not independent requests. "
             "Quest uses 8% and chunk 16; Twilight uses fake-quantized K and top-p 0.85; "
             "GAS adds group-max 0.80 and exception 16; Less-style uses K=1310 and the captured layer 13 search.", "",
             "The connected Python/Torch prototype uses the same per-head FI page-size-1 adapter for all four "
             "methods. It is not the upstream native CUDA Quest/Twilight implementation or a native Less/GAS kernel. "
             "The cached page bounds and fake-quantized prefix K are outside the timing. Metadata construction "
             "and FI plan are inside; projections, RoPE, cache updates, full-model work and prefill are not.", "",
             "Each stage below is a median of separately synchronized wall timings (microseconds). "
             "Connected is the median of separately timed complete paths; it need not equal the sum of stage medians. "
             "The Less prototype shares the first request's selection, valid here only because requests are repeated. "
             "Event timings, individual runs, exact support, reference checks and input hashes are in the CSV/JSON.", "",
             "| B | Method | Selector | Estimator | Top-p | Refinement | Less search | Metadata | Plan | Attention | Connected |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(value):
        return "-" if value is None else f"{value:.1f}"
    for row in stage_rows:
        values = [fmt(row[key + "_wall_us"]) for key in stage_keys]
        lines.append(f"| {row['batch']} | {row['method']} | " + " | ".join(values) + f" | {row['connected_wall_us']:.1f} |")
    lines += ["", "## Layer-count projections", "",
              "Attention-only uses the previously supplied 28-layer-mean attention numbers: Quest, Twilight and "
              "Less use group FI; GAS uses per-head FI. Dense FI is the earlier page16 tensor-core measurement. "
              "This is a mixed-backend disconnected projection, excludes all selection overhead, "
              "and is not a model TPOT.", "",
              "Prototype projection combines a separately measured dense FI result with the measured "
              "per-head connected prototype for 30 layers. It is a synthetic estimate, not a model measurement. "
              "For Less, it charges two measured searches and 30 sparse replay stages; the 4-dense/28-sparse "
              "column instead accounts for its two additional dense search layers. Less uses summed stage medians, "
              "not an independently timed 30-layer decode step.", "",
              "The grouped-only column uses group FI for all four sparse methods; it differs only for GAS "
              "from the supplied mixed-backend projection. Both columns are attention-only.", "",
              "| B | Method | Supplied attention 2+30 (ms) | Grouped attention 2+30 (ms) | Prototype 2+30 (ms) | Less 4+28 (ms) |",
              "|---:|---|---:|---:|---:|---:|"]
    for row in model_rows:
        less_value = row['less_search_aware_4dense_28sparse_ms']
        lines.append(f"| {row['batch']} | {row['method']} | {row['attention_only_2dense_30sparse_ms']:.3f} | "
                     f"{row['grouped_attention_only_2dense_30sparse_ms']:.3f} | "
                     f"{row['prototype_pipeline_2dense_30sparse_ms']:.3f} | "
                     f"{'-' if less_value is None else f'{less_value:.3f}'} |")
    lines += ["", "Not an end-to-end TPOT, serving benchmark, or official Figure 10 reproduction. "
              "Mask-to-page and plan timings in the prototype are per-head, whereas the static attention "
              "projection mixes group and per-head adapters; do not add those stage timings to the static "
              "attention numbers as if a connected grouped pipeline had been measured. "
              "The Less 2+30 projection is a requested hypothetical, not its actual layer schedule.", ""]
    (output / "Qasper_16K_Breakdown.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Summary: {len(stage_rows)}/16 method-shapes in {output}", flush=True)
    return len(stage_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    (output / "measurements").mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        for batch in BATCHES:
            for method in METHODS:
                path = output / "measurements" / f"{method}_b{batch}_s15.json"
                if path.exists():
                    print(f"SKIP {method} B{batch}: {path}", flush=True)
                    continue
                command = [sys.executable, str(BENCH), "--trace-dir", str(TRACE), "--output", str(path),
                           "--method", method, "--batch", str(batch), "--warmups", str(args.warmups),
                           "--repeats", str(args.repeats), "--less-k", "1310"]
                log = output / "logs" / f"{method}_b{batch}.log"
                print(f"RUN {method} B{batch}", flush=True)
                with log.open("w", encoding="utf-8") as stream:
                    result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                print(f"{'DONE' if result.returncode == 0 else 'FAILED'} {method} B{batch}; {log}", flush=True)
                summarize(output)
    if summarize(output) != len(BATCHES) * len(METHODS):
        sys.exit(1)


if __name__ == "__main__":
    main()
