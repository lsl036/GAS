"""Rebuild all E7 v2 deliverables from per-round JSON measurement shards."""

import argparse
import csv
import json
from pathlib import Path
import statistics

METHODS = ("fa2", "flashinfer", "quest", "quest_twi", "gas")


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def mean_time(unit, host=False):
    return statistics.fmean(row["host_wall_us" if host else "latency_us"] for row in unit["timings"])


def diagnostics(unit):
    by_component = {}
    for row in unit.get("diagnostic_timings", []):
        if row["timing_scope"] == "diagnostic_stage":
            by_component.setdefault(row["component"], []).append(row["latency_us"])
    return {name: statistics.fmean(values) for name, values in by_component.items()}


def summarize(output):
    config = json.loads((output / "config.json").read_text())
    manifest = config["trace_manifest"]
    units = [json.loads(file.read_text()) for file in sorted((output / "shards").glob("*.json"))]
    if any(row["protocol_hash"] != config["protocol_hash"] for row in units):
        raise ValueError("Mixed E7 protocols in the same directory")
    index = {(row["length"], row["batch_size"], row["sample"], row["step"], row["layer_id"], row["round"], row["method"]): row for row in units}
    raw, checks, supports, components, optimizations, dispatch_rows = [], [], [], [], [], []
    for unit in units:
        meta = {key: unit[key] for key in ("length", "batch_size", "sample", "step", "trace_id", "layer_id", "round", "method", "order", "status")}
        run_id = f"L{unit['length']}_B{unit['batch_size']}_s{unit['sample']}_d{unit['step']}_l{unit['layer_id']}_r{unit['round']}_{unit['method']}"
        if "correctness" in unit:
            checks.append({**meta, **unit["correctness"]})
        if "support" in unit and unit["support"]:
            stats = dict(unit["support"])
            if "candidate_tokens_mean" in stats:
                stats["candidate_per_head_mean"] = stats.pop("candidate_tokens_mean")
                stats["final_per_head_mean"] = unit["length"]
            supports.append({**meta, **stats, "dispatch": unit.get("dispatch", "")})
        for row in unit.get("timings", []) + unit.get("diagnostic_timings", []) + unit.get("optimization_timings", []):
            raw.append({"run_id": run_id, **meta, **row})
        if unit["status"] == "available":
            profile = diagnostics(unit)
            components.append({**meta, "full_path_us": mean_time(unit), "host_wall_us": mean_time(unit, True), "profile_attention_us": profile.get("attention", 0), "profile_other_stages_us": sum(v for k, v in profile.items() if k != "attention"), "profile_stages_json": json.dumps(profile, sort_keys=True), "peak_allocated_gib": unit["peak_allocated_gib"], "peak_reserved_gib": unit["peak_reserved_gib"], "dispatch": unit.get("dispatch", ""), "path_minus_profile_sum_us": mean_time(unit) - sum(profile.values()) if profile else ""})
        else:
            components.append({**meta, "reason": unit.get("reason", "")})
        if "optimization" in unit:
            optimizations.append({**meta, **unit["optimization"]})
        if unit["method"] == "gas":
            fused = unit.get("dispatch") == "gas_fused_group_exception"
            dispatch_rows.append({**meta, "dispatch": unit.get("dispatch", ""), "fused_only_available": fused, "fused_only_attention_us": diagnostics(unit).get("attention", "") if fused else "", "fallback_reason": "" if fused else "fixed 96KiB shared-memory capacity exceeded" if unit.get("dispatch") == "gas_fi_capacity_fallback" else unit.get("reason", "")})
    for name, rows in (("E7_timings_raw.csv", raw), ("E7_correctness.csv", checks), ("E7_support_stats.csv", supports), ("E7_components.csv", components), ("E7_optimization.csv", optimizations), ("E7_GAS_Dispatch.csv", dispatch_rows)):
        write_csv(output / name, rows)
    estimates = []
    failures = []
    for length in config["lengths"]:
        for batch in config["batch_sizes"]:
            for method in METHODS:
                for sample in range(len(manifest["sources"])):
                    for step in manifest["steps"]:
                        for round_id in range(1, config["rounds"] + 1):
                            selected = []
                            expected = [(layer, "common") for layer in manifest["layers"]] + [("output", "output")]
                            expected += [(layer, "fa2" if method not in ("fa2", "flashinfer") and layer < 2 else method) for layer in manifest["layers"]]
                            for layer, label in expected:
                                unit = index.get((length, batch, sample, step, layer, round_id, label))
                                if unit is None or unit["status"] != "available":
                                    failures.append({"length": length, "batch_size": batch, "method": method, "sample": sample, "step": step, "round": round_id, "layer_id": layer, "required_component": label, "status": "missing" if unit is None else unit["status"], "reason": "" if unit is None else unit.get("reason", "")})
                                else:
                                    selected.append(unit)
                            if len(selected) != len(expected):
                                continue
                            common = sum(mean_time(u) for u in selected if u["method"] == "common")
                            outside = next(mean_time(u) for u in selected if u["method"] == "output")
                            paths = [u for u in selected if u["method"] not in ("common", "output")]
                            dense = [u for u in paths if method not in ("fa2", "flashinfer") and u["layer_id"] < 2]
                            sparse = [u for u in paths if u not in dense]
                            dense_us = sum(mean_time(u) for u in dense)
                            attention_us = sum(diagnostics(u).get("attention", 0) for u in sparse)
                            other_us = sum(sum(v for k, v in diagnostics(u).items() if k != "attention") for u in sparse)
                            path_us = sum(mean_time(u) for u in sparse)
                            total = common + outside + dense_us + path_us
                            estimates.append({"method": method, "length": length, "batch_size": batch, "trace_id": f"sample{sample}_step{step}", "round": round_id, "common_ms": common / 1000, "output_ms": outside / 1000, "dense_layer_ms": dense_us / 1000, "selection_and_metadata_ms": other_us / 1000, "attention_ms": attention_us / 1000, "path_overhead_ms": (path_us - attention_us - other_us) / 1000, "estimated_tpot_ms": total / 1000, "host_replay_estimated_tpot_ms": sum(mean_time(u, True) for u in selected) / 1000, "formula": "sum(actual_common_layers)+output+sum(actual_attention_paths); dense sparse-method IDs use shared FA2", "representative_layer_weight": 1, "measured_layers": len(manifest["layers"]), "input_type": config["input_type"]})
    write_csv(output / "E7_step_estimates.csv", estimates)
    write_csv(output / "E7_missing_or_unavailable.csv", failures)
    summary = []
    count = len(manifest["sources"]) * len(manifest["steps"]) * config["rounds"]
    metrics = ("estimated_tpot_ms", "host_replay_estimated_tpot_ms", "common_ms", "output_ms", "dense_layer_ms", "selection_and_metadata_ms", "attention_ms", "path_overhead_ms")
    for length in config["lengths"]:
        for batch in config["batch_sizes"]:
            for method in METHODS:
                available = [u for u in estimates if (u["length"], u["batch_size"], u["method"]) == (length, batch, method)]
                status = "available" if len(available) == count and config["formal_matrix"] else "smoke_only" if len(available) == count else "incomplete_or_unavailable"
                row = {"method": method, "length": length, "batch_size": batch, "status": status, "completed_trace_rounds": len(available), "required_trace_rounds": count, "input_type": config["input_type"], "timing_scope": "component_sum_serialized_cuda_event_intervals_with_host_gaps"}
                for metric in metrics:
                    row[metric] = statistics.fmean(u[metric] for u in available) if len(available) == count else ""
                row["estimated_tpot_ms_std"] = statistics.stdev(u["estimated_tpot_ms"] for u in available) if len(available) == count and count > 1 else ""
                row["estimated_tokens_per_second"] = batch * 1000 / row["estimated_tpot_ms"] if row["estimated_tpot_ms"] else ""
                gas_units = [u for u in dispatch_rows if u["length"] == length and u["batch_size"] == batch and u["status"] == "available"] if method == "gas" else []
                row["gas_fused_units"] = sum(u["dispatch"] == "gas_fused_group_exception" for u in gas_units) if gas_units else ""
                row["gas_fi_fallback_units"] = sum(u["dispatch"] == "gas_fi_capacity_fallback" for u in gas_units) if gas_units else ""
                row["gas_fallback_unit_fraction"] = row["gas_fi_fallback_units"] / len(gas_units) if gas_units else ""
                summary.append(row)
    for row in summary:
        for baseline in ("fa2", "flashinfer", "quest", "quest_twi"):
            label = "twilight" if baseline == "quest_twi" else baseline
            base = next(x for x in summary if x["method"] == baseline and x["length"] == row["length"] and x["batch_size"] == row["batch_size"])
            row["speedup_vs_" + label] = base["estimated_tpot_ms"] / row["estimated_tpot_ms"] if row["status"] == base["status"] == "available" else ""
    write_csv(output / "E7_summary.csv", summary)
    report = ["# E7 v2 Connected Estimated TPOT", "", "Per-layer real checkpoint-weight and dense-model trace replay; every captured layer is measured, not extrapolated.", "CUDA-event intervals include serialized host dispatch/compaction/plan gaps; synchronized host wall times are also saved.", "This is a component sum, not a closed-loop serving TPOT, and assumes no offload/communication cost.", "Real source sequences are replicated into independent physical KV addresses for each batch row.", "No per-batch division; no cached query-dependent selection; prefix extrema and quantized-grid K are persistent metadata.", "Sparse methods retain FA2 at layers 0 and 1; no sparse latency is charged to those layers.", "Quest uses FI page-16; Twilight uses FI group-varlen page-1; GAS is fused with an explicitly counted FI capacity fallback.", "", "| Length | Batch | Method | Estimated TPOT ms | Host replay ms | vs FI | FI fallback units | Status |", "|---:|---:|---|---:|---:|---:|---:|---|"]
    for row in summary:
        fmt = lambda k: f"{row[k]:.4f}" if row.get(k) != "" else ""
        report.append(f"| {row['length']} | {row['batch_size']} | {row['method']} | {fmt('estimated_tpot_ms')} | {fmt('host_replay_estimated_tpot_ms')} | {fmt('speedup_vs_flashinfer')} | {row['gas_fi_fallback_units']} | {row['status']} |")
    report += ["", f"Formal matrix: {config['formal_matrix']}. Valid points: {sum(r['status'] == 'available' for r in summary)}/{len(summary)}.", "Incomplete or failed points have blank TPOT; see E7_missing_or_unavailable.csv. Smoke points are not paper results.", "Stage diagnostics use fewer samples; path_overhead_ms is the signed difference from the main connected-path timing, not a forced scaling correction.", "GAS capacity fallback is reported per layer in E7_components.csv. Performance conclusions must include slower/failed points.", "Use E7_step_estimates.csv for paired trace/round analysis; raw JSON shards and this script reproduce all CSVs."]
    (output / "E7_Result_Summary.md").write_text("\n".join(report) + "\n")
    print(f"E7 v2 summary: {output}, valid points={sum(r['status'] == 'available' for r in summary)}/{len(summary)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    summarize(parser.parse_args().output_dir)
