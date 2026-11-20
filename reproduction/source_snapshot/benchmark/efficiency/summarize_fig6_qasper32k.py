#!/usr/bin/env python3
"""Append the repeated-Qasper 32K comparison to the validated Fig.6 report."""

import json
import statistics

from summarize_fig6 import BATCHES, LAYERS, METHODS, OUTPUT, RESULTS, load_condition


def main():
    existing = OUTPUT.read_text(encoding="utf-8")
    marker = "## Additional: Qasper 16K vs Repeated-Context Qasper 32K"
    existing = existing.split(marker, 1)[0].rstrip() + "\n\n"
    notice = ("**Dataset navigation:** The initial 32K tables and arrays below use RULER, not Qasper. "
              "For Qasper 16K versus repeated-context Qasper 32K plotting arrays and the separate RULER comparison, "
              "see [Additional: Qasper 16K vs Repeated-Context Qasper 32K](#additional-qasper-16k-vs-repeated-context-qasper-32k).\n\n")
    if notice not in existing:
        existing = existing.replace("## Scope\n\n", notice + "## Scope\n\n", 1)
    metadata = json.loads((RESULTS / "fig6_traces/qasper32k_repeated.jsonl.meta.json").read_text())
    data = {}
    for budget, suffix in (("8192", "fixed8192"), ("8%", "rate08")):
        for length, label in (("16K", f"qasper16k_{suffix}"),
                              ("32K repeated", f"qasper32k_repeated_{suffix}"),
                              ("32K RULER", f"ruler32k_{suffix}")):
            data[budget, length] = load_condition(label)

    lines = [marker, "",
             "Qasper 32K repeats each original context three times before the existing middle truncation and chat formatting; the original question is retained. This is a synthetic repeated-context workload, not natural 32K Qasper or a quality result. Qasper 16K and repeated Qasper 32K use samples 15 and 54. RULER 32K uses different inputs and is shown as a separate alternative, not a controlled length comparison.",
             f"Repeated dataset source SHA-256: `{metadata['source_sha256']}`, output SHA-256: `{metadata['output_sha256']}`.",
             "All numbers are arithmetic means of the 30 per-layer latency medians in microseconds; Less uses FI dense proxies at search layers 2 and 13. Replay setup, model trajectory, adapter and GPU-contention caveats above also apply here.", ""]
    for budget in ("8192", "8%"):
        lines += [f"### Quest budget {budget}", "",
                  "| Batch | Input | " + " | ".join(METHODS) + " |",
                  "|---:|---|" + "---:|" * len(METHODS)]
        for batch in BATCHES:
            for length in ("16K", "32K repeated", "32K RULER"):
                means = data[budget, length][0]
                lines.append(f"| {batch} | {length} | " + " | ".join(f"{means[m, batch]:.3f}" for m in METHODS) + " |")
        lines += ["", "#### 28 Common Non-Search Layers", "",
                  "| Batch | Input | " + " | ".join(METHODS) + " |",
                  "|---:|---|" + "---:|" * len(METHODS)]
        for batch in BATCHES:
            for length in ("16K", "32K repeated", "32K RULER"):
                means = data[budget, length][1]
                lines.append(f"| {batch} | {length} | " + " | ".join(f"{means[m, batch]:.3f}" for m in METHODS) + " |")
        lines += ["", "Logical support counts, averaged across 30 layers and the two requests (not measured memory traffic):", "",
                  "| Input | Twilight tokens/head | Twilight group union | GAS tokens/head | GAS group union |",
                  "|---|---:|---:|---:|---:|"]
        fields = ("twi_per_head_tokens_mean", "twi_group_union_mean", "gas_per_head_tokens_mean", "gas_group_union_mean")
        for length in ("16K", "32K repeated", "32K RULER"):
            columns = data[budget, length][4]
            values = [statistics.fmean(float(columns["Twilight FI"][layer, 32][field]) for layer in LAYERS) for field in fields]
            lines.append(f"| {length} | " + " | ".join(f"{value:.3f}" for value in values) + " |")
        lines += ["", "Plot arrays below are in microseconds; columns are Qasper 16K and repeated-context Qasper 32K, rows B32/B64/B96/B128.", "", "```python", "import numpy as np", "", "latency_us = {"]
        for method in METHODS:
            values = [[data[budget, length][0][method, b] for length in ("16K", "32K repeated")] for b in BATCHES]
            lines.append(f"    {method!r}: np.array({values!r}),")
        lines += ["}", "```", ""]
        for length in ("16K", "32K repeated", "32K RULER"):
            _, _, cal, _, cols = data[budget, length]
            lengths = json.loads(cols["Quest FI"][2, 32]["lengths"])
            lines.append(f"- {length}: actual KV lengths {lengths}; Less K={cal['K']} matched to Twilight union mean {cal['mean_union']:.3f}.")
        base = RESULTS / f"fig6_qasper32k_repeated_{'fixed8192' if budget == '8192' else 'rate08'}"
        for source in ("main/results.csv", "fi_dense/results.csv", "less_fi_adapter/results.csv",
                       "less_patched_headpacked/results.csv", "less_calibration.json"):
            lines.append(f"- Repeated Qasper 32K: [{source}]({base / source})")
        lines.append("")
    OUTPUT.write_text(existing + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
