"""Build honest two-view summaries from frozen E7 v1 and E6 measurements.

This does not rerun a model. It replaces the historical E7 attention rows with
same-shape E6 kernel medians, fixes dense-layer accounting, and emits:
  * attention_only: no query-dependent frontend for any method;
  * frontend_inclusive: the frozen v1 synthetic frontend cost for every sparse
    method, consistently included.
Both outputs are projections, not serving TPOT.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METHODS = ("fa2", "flashinfer", "quest", "quest_twi", "gas")
SPARSE = {"quest", "quest_twi", "gas"}
E6_METHOD = {"fa2": "fa2", "flashinfer": "flashinfer", "quest": "quest", "quest_twi": "quest_twi", "gas": "gas"}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-e7", type=Path, default=ROOT / "benchmark/results_efficiency/e7_estimated_tpot/E7_components.csv")
    parser.add_argument("--e6-kernel", type=Path, default=ROOT / "benchmark/results_efficiency/e6_kernel_baselines/E6_summary.csv")
    parser.add_argument("--e6-batch", type=Path, default=ROOT / "benchmark/results_efficiency/e6_batch_scaling/E6_summary.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmark/results_efficiency/e7_v1_simplified_projection")
    args = parser.parse_args()
    old = read_csv(args.old_e7)
    e6 = read_csv(args.e6_kernel) + read_csv(args.e6_batch)
    component = {(r["method"], int(r["length"]), int(r["batch_size"])): r for r in old if r.get("status") == "available"}
    kernel = {(r["method"], int(r["length"]), int(r["batch_size"])): float(r["pooled_median_us"]) for r in e6 if r.get("status") == "available" and r["method"] in E6_METHOD.values()}
    candidate_shapes = sorted({(int(r["length"]), int(r["batch_size"])) for r in old if r.get("status") == "available"})
    shapes = [
        shape for shape in candidate_shapes
        if all((method, *shape) in component for method in METHODS)
        and all((E6_METHOD[method], *shape) in kernel for method in METHODS)
    ]
    missing_shapes = [shape for shape in candidate_shapes if shape not in shapes]
    rows = []
    for length, batch in shapes:
        base = component[("fa2", length, batch)]
        common_ms, output_ms = float(base["common_ms"]), float(base["output_ms"])
        dense_us = kernel[("flashinfer", length, batch)]
        for method in METHODS:
            source = component[(method, length, batch)]
            sparse_us = kernel[(E6_METHOD[method], length, batch)]
            frontend_ms = 0.0 if method not in SPARSE else float(source["selection_and_metadata_ms"])
            if method in ("fa2", "flashinfer"):
                attention_ms = kernel[(E6_METHOD[method], length, batch)] * 32 / 1000.0
                dense_layers, sparse_layers = 32, 0
            else:
                # The first two layers remain dense and use one common dense
                # backend. Sparse latency is charged only to layers 2..31.
                attention_ms = (2 * dense_us + 30 * sparse_us) / 1000.0
                dense_layers, sparse_layers = 2, 30
            attention_only = common_ms + output_ms + attention_ms
            frontend_inclusive = attention_only + frontend_ms
            rows.append({
                "length": length, "batch_size": batch, "method": method,
                "common_ms": common_ms, "output_ms": output_ms,
                "dense_backend": "flashinfer" if method in SPARSE else method,
                "dense_attention_one_layer_us": dense_us if method in SPARSE else sparse_us,
                "sparse_attention_one_layer_us": sparse_us if method in SPARSE else "",
                "selection_frontend_ms": frontend_ms,
                "attention_ms": attention_ms,
                "dense_layers": dense_layers, "sparse_layers": sparse_layers,
                "attention_only_projection_ms": attention_only,
                "frontend_inclusive_projection_ms": frontend_inclusive,
                "input_type": "profile_calibrated_synthetic_components",
                "kernel_source": "E6 pooled median",
                "status": "available",
            })
    for view, field in (("attention_only", "attention_only_projection_ms"), ("frontend_inclusive", "frontend_inclusive_projection_ms")):
        for row in rows:
            base = next(x for x in rows if x["length"] == row["length"] and x["batch_size"] == row["batch_size"] and x["method"] == "flashinfer")
            row["speedup_vs_flashinfer_" + view] = float(base[field]) / float(row[field])
        write_csv(args.output_dir / f"E7_v1_{view}.csv", rows)
    report = [
        "# E7 v1 simplified projection",
        "",
        "This report uses E6 same-shape pooled-median kernel measurements and E7 v1 common/frontend components.",
        "It fixes sparse dense-layer accounting: 2 dense FlashInfer layers + 30 sparse method layers.",
        "It is a profile-calibrated projection, not measured end-to-end serving TPOT.",
        "",
        "## Attention-only projection",
        "",
        "Selection, estimator, union, refinement and metadata are excluded for every method.",
        "| Length | Batch | FA2 ms | FlashInfer ms | Quest ms | Quest+Twilight ms | GAS ms | GAS vs FI |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for length, batch in shapes:
        subset = {r["method"]: r for r in rows if r["length"] == length and r["batch_size"] == batch}
        report.append("| {} | {} | {} | {} | {} | {} | {} | {:.2f}x |".format(length, batch, *[f"{float(subset[m]['attention_only_projection_ms']):.3f}" for m in METHODS], float(subset["gas"]["speedup_vs_flashinfer_attention_only"])))
    report += ["", "## Frontend-inclusive projection", "", "The frozen v1 synthetic frontend is included for every sparse method.", "| Length | Batch | FA2 ms | FlashInfer ms | Quest ms | Quest+Twilight ms | GAS ms | GAS vs FI |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for length, batch in shapes:
        subset = {r["method"]: r for r in rows if r["length"] == length and r["batch_size"] == batch}
        report.append("| {} | {} | {} | {} | {} | {} | {} | {:.2f}x |".format(length, batch, *[f"{float(subset[m]['frontend_inclusive_projection_ms']):.3f}" for m in METHODS], float(subset["gas"]["speedup_vs_flashinfer_frontend_inclusive"])))
    report += [
        "",
        "Quest uses the E6 measured adapter rows; those rows are an efficient compatible adapter, not the official Quest implementation.",
        "The frontend-included rows use disconnected profile-calibrated synthetic frontend costs and must not be called connected end-to-end TPOT.",
        "The attention-only rows are the appropriate view for an algorithmic/kernel support-reduction figure.",
        "Do not merge these rows with the withdrawn E7 v1 totals or incomplete E7 v2 connected results.",
    ]
    (args.output_dir / "E7_v1_Simplified_Projection.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "E7_v1_missing_shapes.csv", [{"length": length, "batch_size": batch, "reason": "historical E7 component or E6 kernel baseline unavailable; no extrapolation"} for length, batch in missing_shapes])
    print(f"wrote E7 v1 simplified views to {args.output_dir}")


if __name__ == "__main__":
    main()
