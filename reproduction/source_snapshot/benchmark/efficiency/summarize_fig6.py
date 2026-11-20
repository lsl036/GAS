#!/usr/bin/env python3
"""Summarize completed Fig.6 per-layer replays with strict coverage checks."""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmark/results_efficiency"
OUTPUT = ROOT.parent / "thoughts/Paper/Fig6_handoff.md"
BATCHES = (32, 64, 96, 128)
LAYERS = tuple(range(2, 32))
LESS_LAYERS = tuple(layer for layer in LAYERS if layer not in (2, 13))
METHODS = ("FA2 dense", "FlashInfer dense", "Quest FI", "Twilight FI",
           "GAS FI", "GAS inline8", "Less FI", "Less patched")
CONDITIONS = {
    ("8192", "16K"): "qasper16k_fixed8192",
    ("8192", "32K"): "ruler32k_fixed8192",
    ("8%", "16K"): "qasper16k_rate08",
    ("8%", "32K"): "ruler32k_rate08",
}


def read_rows(path, method, layers):
    rows = [r for r in csv.DictReader(path.open(newline="", encoding="utf-8")) if r["method"] == method]
    index = {(int(r["layer"]), int(r["batch"])): r for r in rows}
    expected = {(layer, batch) for layer in layers for batch in BATCHES}
    if len(rows) != len(index) or set(index) != expected:
        raise ValueError(f"incomplete or duplicate rows for {method} in {path}: {len(index)}/{len(expected)}")
    if any(int(row["samples"]) != 20 for row in rows):
        raise ValueError(f"unexpected timing sample count for {method}: {path}")
    if any(not math.isfinite(float(row["median_us"])) or float(row["median_us"]) <= 0 for row in rows):
        raise ValueError(f"invalid kernel latency for {method}: {path}")
    if any("max_abs_error" in row and not math.isfinite(float(row["max_abs_error"])) for row in rows):
        raise ValueError(f"invalid correctness error for {method}: {path}")
    return index


def load_condition(label):
    base = RESULTS / f"fig6_{label}"
    main = base / "main/results.csv"
    dense = base / "fi_dense/results.csv"
    less_fi = base / "less_fi_adapter/results.csv"
    less_patched = base / "less_patched_headpacked/results.csv"
    columns = {
        "FA2 dense": read_rows(main, "fa2_dense", LAYERS),
        "FlashInfer dense": read_rows(dense, "fi_dense", LAYERS),
        "Quest FI": read_rows(main, "fi_quest_adapter", LAYERS),
        "Twilight FI": read_rows(main, "fi_twilight_adapter", LAYERS),
        "GAS FI": read_rows(main, "fi_gas_adapter", LAYERS),
        "GAS inline8": read_rows(main, "inline8", LAYERS),
        "Less FI": read_rows(less_fi, "less_fi_adapter", LESS_LAYERS),
        "Less patched": read_rows(less_patched, "less_patched_headpacked", LESS_LAYERS),
    }
    # Less search layers 2 and 13 use the measured FI dense attention proxy.
    for method in ("Less FI", "Less patched"):
        for layer in (2, 13):
            for batch in BATCHES:
                columns[method][layer, batch] = columns["FlashInfer dense"][layer, batch]
    calibration = json.loads((base / "less_calibration.json").read_text())
    config = json.loads((base / "main/config.json").read_text())
    if (config["steps"] != [1] or config["layers"] != list(LAYERS) or config["batch_sizes"] != list(BATCHES) or
            config["dtype"] != "float16" or config["repeats"] != 10 or config["rounds"] != 2 or
            config["graph_iterations"] != 8):
        raise ValueError(f"incompatible main experiment settings for {label}")
    if len(config["trace_dirs"]) != 2 or len(set(config["sample_hashes"])) != 2:
        raise ValueError(f"expected two distinct trace requests for {label}")
    for directory, expected in zip(config["trace_dirs"], config["manifest_sha256"], strict=True):
        manifest_path = Path(directory) / "manifest.json"
        raw = manifest_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"trace manifest changed: {manifest_path}")
        manifest = json.loads(raw)
        selector = manifest["config"]["selector"]
        expected_task = "qasper" if label.startswith("qasper") else "ruler_forced_prompt_decode"
        is_fixed = label.endswith("fixed8192")
        if (manifest["task"] != expected_task or selector["type"] != "quest" or selector["chunk_size"] != 16 or
                (is_fixed and selector.get("token_budget") != 8192) or
                (not is_fixed and selector.get("sparsity_rate") != 0.08) or
                manifest["config"]["weight_estimator"]["type"] != "min_max_quant" or
                manifest["config"]["weight_estimator"]["quant_bit"] != 4 or
                manifest["config"]["weight_pruner"]["threshold"] != 0.85 or
                manifest["config"]["gqa_refinement"]["mode"] != "group_max" or
                manifest["config"]["gqa_refinement"]["threshold"] != 0.8 or
                manifest["config"]["gqa_refinement"]["exception_budget"] != 16):
            raise ValueError(f"incorrect trace task or algorithm for {label}: {manifest_path}")
    for name in ("fi_dense", "less_fi_adapter", "less_patched_headpacked"):
        other = json.loads((base / name / "config.json").read_text())
        if (other["manifest_sha256"] != config["manifest_sha256"] or other["steps"] != [1] or
                other["batch_sizes"] != list(BATCHES) or other["dtype"] != "float16" or
                other["repeats"] != 10 or other["rounds"] != 2 or other["graph_iterations"] != 8):
            raise ValueError(f"incompatible experiment settings in {base / name / 'config.json'}")
        if name == "fi_dense" and (other["layers"] != list(LAYERS) or other["script_sha256"] != config["script_sha256"]):
            raise ValueError(f"incompatible dense replay source or layer coverage in {base / name / 'config.json'}")
        if name.startswith("less_") and other["budget"] != calibration["K"]:
            raise ValueError(f"mismatched Less budget in {base / name / 'config.json'}")
    if calibration["trace_manifests_sha256"] != config["manifest_sha256"]:
        raise ValueError(f"calibration trace mismatch for {label}")
    means = {(method, batch): statistics.fmean(float(columns[method][layer, batch]["median_us"]) for layer in LAYERS)
             for method in METHODS for batch in BATCHES}
    sparse_means = {(method, batch): statistics.fmean(float(columns[method][layer, batch]["median_us"]) for layer in LESS_LAYERS)
                    for method in METHODS for batch in BATCHES}
    return means, sparse_means, calibration, config, columns


def main():
    helper = ROOT / "benchmark/efficiency/bench_less_official_headpacked.py"
    if hashlib.sha256(helper.read_bytes()).hexdigest() != "3f3f1103530ea31f55f5aab4de338d5c4fca0308e2d90d81b6b59c4f7314aef5":
        raise ValueError("Less head-packed setup source changed since the Fig.6 experiment")
    data = {key: load_condition(value) for key, value in CONDITIONS.items()}
    lines = [
        "# Fig.6 Kernel Replay Handoff", "",
        "## Scope", "",
        "Attention-only replay on LLaMA-3.1-8B-Instruct, A100-PCIE-40GB; 32 query heads, 8 KV heads, head dimension 128, decode step 1. Captured BF16 Q/K/V are replayed in FP16.",
        "16K uses two real Qasper samples, middle-truncated before chat formatting; 32K uses two RULER niah_single_1 samples. These are different datasets, not a controlled length scaling experiment.",
        "Step 1 is a teacher-forced question token for Qasper and a teacher-forced final prompt token for RULER; it is not a generated-answer decoding workload.",
        "Each batch repeats its two requests round-robin. The reported arithmetic mean is over 30 layer latency medians (layers 2–31; Less uses two dense search-layer proxies); it is not an average over independent requests or model TPOT. Per-layer medians use 2 rounds × 10 CUDA-event samples, 8 graph calls per sample.",
        "Quest uses either fixed 8192 candidates or an 8% dynamic budget, with chunk size 16; Twilight applies min-max fake-quant4 and top-p 0.85, and GAS adds group-max 0.80 and at most 16 per-head exceptions. All supports use the same GAS trajectory within a condition, not separate end-to-end model trajectories.",
        "The LessIsMore-style support is reconstructed from search-layer Q/K on the GAS trajectory, using interleaved cross-head selection plus a 25% recent-token reserve; it is not a separate LessIsMore inference trajectory. One fixed K per condition is rounded from that condition's mean Twilight group union on layers 3–31 excluding search layer 13; this is an in-sample budget match, not a quality match. Its layers 2 and 13 are represented by measured FlashInfer dense attention as a proxy; the other 28 layers use the reconstructed Less support. The patched kernel is a modified B1/head-packed adapter, not native batch execution.",
        "FI sparse adapters use page size 1 and operate per query head, rather than a native group-union-aware Quest or Twilight kernel. Selection, estimation, pruning, packing, transfers, allocation, metadata and planning are excluded. Dense and sparse methods use different supports. No end-to-end, native official-kernel, quality or memory-capacity claim follows.", "",
        "Methods within the main replay are shuffled per layer and batch, but dense FI and the Less adapters run in separate sequential phases. The 20 timing samples are repeated kernel measurements, not independent inputs or independent experiment runs.", "",
        "Measurements include resumed phases. After GPU contention was observed, potentially affected dense FI shards were retained separately in `fig6_ruler32k_fixed8192/fi_dense/quarantine_20261112_203039` and remeasured. The subsequent GPU process guard polls every two seconds; short transient interference may escape detection, so strict GPU exclusivity cannot be guaranteed.", "",
        "Less head-packed setup source SHA-256: `3f3f1103530ea31f55f5aab4de338d5c4fca0308e2d90d81b6b59c4f7314aef5` ([source](/data/home/lsl/SparseAtt/Twilight_Ours/benchmark/efficiency/bench_less_official_headpacked.py)). The KV layout copy is outside timing.", "",
        "## Budget And Inputs", "",
        "| Quest budget | Context | Dataset | Samples | Actual KV lengths | Less K | Matched Twilight union mean |", "|---|---|---|---|---|---:|---:|",
    ]
    for (budget, length), (means, sparse, cal, cfg, cols) in data.items():
        lengths = json.loads(cols["Quest FI"][2, 32]["lengths"])
        lines.append(f"| {budget} | {length} | {'Qasper' if length == '16K' else 'RULER NIAH'} | {', '.join(map(str, cfg['sample_indices']))} | {', '.join(map(str, lengths))} | {cal['K']} | {cal['mean_union']:.3f} |")
    lines += ["", "Across the 30 layers, B32 support statistics are averaged over the two repeated requests and query heads or KV groups. These describe logical selected sets, not measured HBM traffic.", "", "| Quest budget | Context | Twilight tokens/head | Twilight group union | GAS tokens/head | GAS group union |", "|---|---|---:|---:|---:|---:|"]
    for (budget, length), (_, _, _, _, columns) in data.items():
        rows = [columns["Twilight FI"][layer, 32] for layer in LAYERS]
        fields = ("twi_per_head_tokens_mean", "twi_group_union_mean", "gas_per_head_tokens_mean", "gas_group_union_mean")
        values = [statistics.fmean(float(row[field]) for row in rows) for field in fields]
        lines.append(f"| {budget} | {length} | " + " | ".join(f"{value:.3f}" for value in values) + " |")
    for budget in ("8192", "8%"):
        lines += ["", f"## {budget}: 30-Layer Mean", "", "| Batch | Context | " + " | ".join(METHODS) + " |",
                  "|---:|---|" + "---:|" * len(METHODS)]
        for batch in BATCHES:
            for length in ("16K", "32K"):
                means = data[budget, length][0]
                lines.append(f"| {batch} | {length} | " + " | ".join(f"{means[m, batch]:.3f}" for m in METHODS) + " |")
        lines += ["", "### 28 Non-Search Layers", "", "| Batch | Context | " + " | ".join(METHODS) + " |",
                  "|---:|---|" + "---:|" * len(METHODS)]
        for batch in BATCHES:
            for length in ("16K", "32K"):
                means = data[budget, length][1]
                lines.append(f"| {batch} | {length} | " + " | ".join(f"{means[m, batch]:.3f}" for m in METHODS) + " |")
        lines += ["", "### Ratios Of 30-Layer Means", "", "Ratios greater than 1 mean that the denominator is faster; these are not means of per-layer speedups. The Less ratio includes two FI dense search-layer proxies, so use the 28-layer table for a sparse-kernel-only comparison.", "", "| Batch | Context | Quest FI / Twilight FI | Twilight FI / GAS FI | Twilight FI / inline8 | Less FI / GAS FI |", "|---:|---|---:|---:|---:|---:|"]
        for batch in BATCHES:
            for length in ("16K", "32K"):
                means = data[budget, length][0]
                q, t, g, i, less = (means[m, batch] for m in ("Quest FI", "Twilight FI", "GAS FI", "GAS inline8", "Less FI"))
                lines.append(f"| {batch} | {length} | {q / t:.3f}× | {t / g:.3f}× | {t / i:.3f}× | {less / g:.3f}× |")
        lines += ["", "### Layerwise Wins", "", "Counts compare measured per-layer medians; they do not treat layers as statistically independent requests.", "", "| Batch | Context | inline8 faster than Twilight FI | inline8 faster than GAS FI |", "|---:|---|---:|---:|"]
        for batch in BATCHES:
            for length in ("16K", "32K"):
                columns = data[budget, length][4]
                def latency(method, layer):
                    return float(columns[method][layer, batch]["median_us"])
                win_twi = sum(latency("GAS inline8", layer) < latency("Twilight FI", layer) for layer in LAYERS)
                win_gas = sum(latency("GAS inline8", layer) < latency("GAS FI", layer) for layer in LAYERS)
                lines.append(f"| {batch} | {length} | {win_twi}/30 | {win_gas}/30 |")
        lines += ["", "### Plot Arrays: 30 Layers", "", "Columns are 16K Qasper and 32K RULER; rows are B32, B64, B96 and B128. Values are μs.", "", "```python", "import numpy as np", "", "latency_us = {"]
        for method in METHODS:
            lines.append(f"    {method!r}: np.array([")
            for batch in BATCHES:
                values = [data[budget, length][0][method, batch] for length in ("16K", "32K")]
                lines.append(f"        [{values[0]:.6f}, {values[1]:.6f}],  # B{batch}")
            lines.append("    ]),")
        lines += ["}", "```"]
    lines += ["", "## Sources", ""]
    for key, label in CONDITIONS.items():
        base = RESULTS / f"fig6_{label}"
        for source in ("main/results.csv", "fi_dense/results.csv", "less_fi_adapter/results.csv",
                       "less_patched_headpacked/results.csv", "less_calibration.json"):
            lines.append(f"- {key[0]} / {key[1]}: [{source}]({base / source})")
    lines += ["", "All reported rows passed the benchmark's per-mask FP32 reference and CUDA Graph replay checks. See adjacent configs, shards and raw timing samples for reproducibility.", ""]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
