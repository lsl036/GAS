#!/usr/bin/env python3
"""Project LessIsMore onto the frozen selection-excluded E7 component view.

This consumes existing E6 kernel measurements and E7 components; it does not
measure a full model or rerun GPU kernels.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
E7 = ROOT / "benchmark/results_efficiency/e7_v1_simplified_projection/E7_v1_attention_only.csv"
E6 = ROOT / "benchmark/results_efficiency/lessismore_e6_matched"
OUTPUT = ROOT / "benchmark/results_efficiency/e7_lessismore_attention_projection"
LENGTHS = (16384, 32768)
BATCHES = (32, 64, 96, 128)
METHODS = ("fa2", "flashinfer", "quest", "quest_twi", "gas")
K_BY_LENGTH = {16384: 1101, 32768: 1059}
E6_METHODS = {"twilight_per_head", "twilight_union_proxy", "less_global_shared", "less_global_per_head", "gas_fused"}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_rows(rows, label):
    result = {}
    for row in rows:
        key = (row["method"], int(row["length"]), int(row["batch_size"]))
        if key in result:
            raise ValueError(f"Duplicate {label} row: {key}")
        result[key] = row
    return result


def projected_ms(common_ms, output_ms, dense_us, sparse_us):
    return common_ms + output_ms + (4 * dense_us + 28 * sparse_us) / 1000.0


def check_e6(root, length, batch, expected_k):
    stem = f"L{length}_B{batch}"
    correctness = read_csv(root / "shards" / f"{stem}_correctness.csv")
    if len(correctness) != 5 or {row["method"] for row in correctness} != E6_METHODS:
        raise ValueError(f"Expected five correctness rows for {stem}")
    for row in correctness:
        if (int(row["length"]) != length or int(row["batch_size"]) != batch
                or row["pass"].lower() != "true" or row["finite_all_batch"].lower() != "true"):
            raise ValueError(f"Failed E6 correctness check for {stem}: {row['method']}")
    support_path = root / "shards" / f"{stem}_support.json"
    support = json.loads(support_path.read_text(encoding="utf-8"))
    union = support["twilight_group_union"]
    if (support["length"] != length or support["batch"] != batch
            or support["less_K"] != expected_k or union["union_min"] != expected_k
            or union["union_max"] != expected_k
            or not support["same_support_less_kernel_equivalence"]["pass"]):
        raise ValueError(f"Unmatched support or failed same-support check: {stem}")
    return support_path, root / "shards" / f"{stem}_correctness.csv", root / "shards" / f"{stem}_timings.csv"


def write_csv(path, rows):
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e7", type=Path, default=E7)
    parser.add_argument("--e6", type=Path, default=E6)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    e7 = index_rows(read_csv(args.e7), "E7")
    e6_summary = args.e6 / "summary.csv"
    e6 = index_rows(read_csv(e6_summary), "E6")
    config_path = args.e6 / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (config["dtype"] != "float16" or config["heads"] != 32 or config["kv_heads"] != 8
            or config["head_dim"] != 128 or config["K_by_length"] != {str(k): v for k, v in K_BY_LENGTH.items()}
            or (config["warmups"], config["rounds"], config["repeats"]) != (20, 5, 100)):
        raise ValueError("E6 configuration does not match the frozen projection protocol")

    rows, sources = [], {}
    for batch in BATCHES:
        for length in LENGTHS:
            key = ("less_global_shared", length, batch)
            if key not in e6:
                raise ValueError(f"Missing LessIsMore kernel measurement: {key}")
            measurement = e6[key]
            expected_k = K_BY_LENGTH[length]
            if (int(measurement["K"]) != expected_k
                    or int(measurement["samples"]) != config["rounds"] * config["repeats"]):
                raise ValueError(f"Invalid K or timing count in E6 row {key}")
            sparse_us = float(measurement["median_us"])
            if not math.isfinite(sparse_us) or sparse_us <= 0:
                raise ValueError(f"Invalid E6 kernel latency: {key}")
            shape_files = check_e6(args.e6, length, batch, expected_k)
            sources[f"{length}_{batch}"] = [{"path": str(path), "sha256": sha256(path)} for path in shape_files]
            result = {"batch_size": batch, "context_length": length, "K": expected_k,
                      "status": "unavailable", "reason": "", "common_ms": "", "lm_head_ms": "",
                      "dense_backend": "flashinfer", "dense_one_layer_us": "",
                      "ordinary_dense_layers": 2, "search_dense_layers": 2, "sparse_layers": 28,
                      "less_sparse_one_layer_us": sparse_us, "less_attention_ms": "",
                      "less_projection_ms": "", "fa2_projection_ms": "", "flashinfer_projection_ms": "",
                      "quest_adapter_projection_ms": "", "quest_twi_proxy_projection_ms": "",
                      "gas_projection_ms": "", "fa2_over_less": "", "flashinfer_over_less": ""}
            if not all((method, length, batch) in e7 for method in METHODS):
                result["reason"] = "Frozen E7 baseline/components unavailable; no extrapolation"
                rows.append(result)
                continue
            base = e7[("flashinfer", length, batch)]
            common, output = float(base["common_ms"]), float(base["output_ms"])
            dense_us = float(base["dense_attention_one_layer_us"])
            for method in METHODS:
                baseline = e7[(method, length, batch)]
                if baseline["status"] != "available" or not math.isclose(float(baseline["common_ms"]), common, abs_tol=1e-8) or not math.isclose(float(baseline["output_ms"]), output, abs_tol=1e-8):
                    raise ValueError(f"Inconsistent E7 common components for {(method, length, batch)}")
            less = projected_ms(common, output, dense_us, sparse_us)
            fa2 = float(e7[("fa2", length, batch)]["attention_only_projection_ms"])
            fi = float(base["attention_only_projection_ms"])
            result.update({"status": "available", "common_ms": common, "lm_head_ms": output,
                           "dense_one_layer_us": dense_us,
                           "less_attention_ms": (4 * dense_us + 28 * sparse_us) / 1000,
                           "less_projection_ms": less, "fa2_projection_ms": fa2,
                           "flashinfer_projection_ms": fi,
                           "quest_adapter_projection_ms": float(e7[("quest", length, batch)]["attention_only_projection_ms"]),
                           "quest_twi_proxy_projection_ms": float(e7[("quest_twi", length, batch)]["attention_only_projection_ms"]),
                           "gas_projection_ms": float(e7[("gas", length, batch)]["attention_only_projection_ms"]),
                           "fa2_over_less": fa2 / less, "flashinfer_over_less": fi / less})
            rows.append(result)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "lessismore_attention_projection.csv", rows)
    provenance = {"e7": str(args.e7), "e7_sha256": sha256(args.e7),
                  "e6_summary": str(e6_summary), "e6_summary_sha256": sha256(e6_summary),
                  "e6_config": str(config_path), "e6_config_sha256": sha256(config_path),
                  "per_shape_support": sources,
                  "formula": "common_32_layers + lm_head + 4*dense_flashinfer + 28*less_sparse",
                  "input": "synthetic, fixed-index, FP16 component projection; not end-to-end TPOT",
                  "search_layer": "dense FlashInfer is a proxy; selection/logit emission is excluded"}
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    report = ["# LessIsMore: Selection-Excluded E7 Projection", "",
              "CPU recombination of existing FP16 E7 components and E6 kernel medians; no new GPU or model execution.",
              "", "| Batch | Context | K | FA2 ms | FlashInfer ms | Quest adapter ms | Quest+Twi proxy ms | GAS ms | LessIsMore ms |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    fields = ("fa2_projection_ms", "flashinfer_projection_ms", "quest_adapter_projection_ms",
              "quest_twi_proxy_projection_ms", "gas_projection_ms", "less_projection_ms")
    for row in rows:
        values = [f"{row[field]:.3f}" if row[field] != "" else "N/A" for field in fields]
        report.append(f"| {row['batch_size']} | {row['context_length']//1024}K | {row['K']} | " + " | ".join(values) + " |")
    report.extend(["", "LessIsMore: 2 ordinary dense layers + 2 full search layers + 28 sparse layers.",
                   "The full search layers use dense FlashInfer as an attention-only proxy; no selection,",
                   "logit extraction, union, metadata or host costs are included. The original E7 Quest",
                   "row uses an E6 profile-sized adapter, not the official Quest kernel. The Quest+Twi",
                   "union proxy is not numerically equivalent to a head-specific Twilight mask.",
                   "E6 LessIsMore uses synthetic shared supports and our kernel, not the authors' kernel.",
                   "32K/B128 is unavailable because the frozen E7 dense baseline is missing; no inference.",
                   "These are component projections, not measured end-to-end TPOT, serving speed, or GPU capacity."])
    (args.output_dir / "LessIsMore_Attention_Projection.md").write_text("\n".join(report) + "\n")
    print(f"WROTE {len(rows)} projection rows to {args.output_dir}")
    print(f"AVAILABLE {sum(row['status'] == 'available' for row in rows)}/{len(rows)}; missing shapes remain unavailable")


if __name__ == "__main__":
    main()
