#!/usr/bin/env python3
"""Summarize same-trace dense SDPA and GAS FI kernel-only comparisons."""
import argparse
import csv
import json
from pathlib import Path

METHODS = ("sdpa_flash", "fa2_kvcache", "gas_fi", "sdpa_efficient_expanded")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    root = args.output_dir
    records = []
    for length in ("16k", "32k"):
        for sample in (15, 54):
            for batch in (32, 64):
                common_hash = None
                for method in METHODS:
                    label = f"{length}_s{sample}_b{batch}_{method}"
                    path = root / "measurements" / f"{label}.json"
                    log = root / "logs" / f"{label}.log"
                    if not path.exists():
                        status = "oom" if log.exists() and "torch.OutOfMemoryError: CUDA out of memory" in log.read_text() else "missing"
                        records.append({"length": length, "sample": sample, "batch": batch, "method": method, "status": status})
                        continue
                    r = json.loads(path.read_text())
                    if (r["method"] != method or r["batch"] != batch or r["sample_index"] != sample or
                            r["layer"] != 15 or r["dtype"] != "torch.float16" or
                            r["graph_iterations"] != 8 or r["rounds"] != 2 or r["repeats"] != 10 or
                            not r["check"]["pass"] or not r["graph_check"]["pass"]):
                        raise ValueError(f"incompatible or invalid result: {path}")
                    if common_hash is not None and r["trace_sha256"] != common_hash:
                        raise ValueError(f"Q/K/V trace mismatch: {path}")
                    common_hash = r["trace_sha256"]
                    records.append({"length": length, "sample": sample, "batch": batch, "method": method,
                                    "status": "ok", "kv_length": r["kv_length"], "median_us": r["median_us"],
                                    "gas_per_head_tokens_mean": r["gas_per_head_tokens_mean"],
                                    "setup_allocated_bytes": r["setup_allocated_bytes"],
                                    "peak_process_allocated_bytes": r["peak_process_allocated_bytes"],
                                    "trace_sha256": r["trace_sha256"]})
    fields = ("length", "sample", "batch", "method", "status", "kv_length", "median_us",
              "gas_per_head_tokens_mean", "setup_allocated_bytes", "peak_process_allocated_bytes", "trace_sha256")
    with (root / "sdpa_audit.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    index = {(r["length"], r["sample"], r["batch"], r["method"]): r for r in records}
    lines = ["# SDPA Fig.6 Baseline Audit", "",
             "Single Qasper sample repeated across the batch; layer 15, FP16, Quest 8% GAS trajectory. 32K uses artificially repeated context. Each cell is the median of 2 x 10 CUDA-event graph replays, divided by 8 graph calls. FA2/SDPA are dense, GAS FI uses the per-head sparse mask; outputs are individually checked against their own FP32 references, not against each other.",
             "SDPA flash is forced with native GQA. PyTorch 2.5.1 memory-efficient SDPA is forced after materializing K/V from 8 to 32 heads; that copy and the extra setup allocations are outside latency. FI plan, sparse mask construction, data layout conversion and input transfer are outside timing. None is model TPOT or a quality-equivalent end-to-end comparison.", "",
             "| Length | Sample | Batch | KV length | GAS tokens/head | SDPA Flash us | FA2 kvcache us | GAS FI us | SDPA efficient expanded us | SDPA Flash / GAS FI |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for length in ("16k", "32k"):
        for sample in (15, 54):
            for batch in (32, 64):
                r = {m: index[length, sample, batch, m] for m in METHODS}
                def value(method):
                    item = r[method]
                    return f"{item['median_us']:.3f}" if item["status"] == "ok" else item["status"].upper()
                kv = next((item["kv_length"] for item in r.values() if item["status"] == "ok"), "-")
                gas = r["gas_fi"]
                count = f"{gas['gas_per_head_tokens_mean']:.3f}" if gas["status"] == "ok" else "-"
                ratio = (f"{r['sdpa_flash']['median_us'] / gas['median_us']:.2f}x"
                         if r["sdpa_flash"]["status"] == gas["status"] == "ok" else "-")
                lines.append(f"| {length} | {sample} | {batch} | {kv} | {count} | {value('sdpa_flash')} | {value('fa2_kvcache')} | {value('gas_fi')} | {value('sdpa_efficient_expanded')} | {ratio} |")
    lines += ["", "## Setup Memory", "", "Incremental allocated memory after creating the backend is measured outside timing; peak is the process's total allocated memory after resetting the peak before backend setup. This is not total GPU memory or end-to-end inference peak.", "",
              "| Length | Sample | Batch | Method | Setup GiB | Peak process GiB |", "|---|---:|---:|---|---:|---:|"]
    for r in records:
        if r["status"] == "ok":
            lines.append(f"| {r['length']} | {r['sample']} | {r['batch']} | {r['method']} | {r['setup_allocated_bytes']/2**30:.3f} | {r['peak_process_allocated_bytes']/2**30:.3f} |")
    lines += ["", f"Raw JSON and failure logs: `{root}`. OOM and missing entries are not imputed.", ""]
    (root / "SDPA_Audit_Summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {root / 'SDPA_Audit_Summary.md'}")


if __name__ == "__main__":
    main()
