import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

from bench_batched_gqa_fused import METHOD_META, summary_path


OVERHEAD_STAGES = (
    "selector",
    "fake_quant_estimator",
    "top_p_prune",
    "gqa_refine",
    "final_mask",
)

STATIC_STAGES = (
    "qkv_proj_rope_cache",
    "budget_score_update",
    "o_proj",
)


def read_attention_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["length"] = int(row["length"])
        row["batch_size"] = int(row["batch_size"])
        row["attention_op_total_us"] = float(row["attention_op_total_us"])
        row["sparse_attention_us"] = float(row["sparse_attention_us"])
        row["o_proj_us"] = float(row["o_proj_us"])
    return rows


def profile_components(method: str, length: int) -> dict:
    data = json.loads(summary_path(method, length).read_text(encoding="utf-8"))
    stages = data.get("stages", {})
    _, variant, _ = METHOD_META[method]

    def stage_mean(name: str) -> float:
        return float(stages.get(name, {}).get("mean_us", 0.0))

    if variant == "selector_only":
        overhead_stages = ("selector", "final_mask")
    elif variant == "twi":
        overhead_stages = ("selector", "fake_quant_estimator", "top_p_prune", "final_mask")
    else:
        overhead_stages = OVERHEAD_STAGES

    overhead_us = sum(stage_mean(name) for name in overhead_stages)
    static_decode_us = sum(stage_mean(name) for name in STATIC_STAGES)
    total_profile_us = float(data.get("total_decode_attention", {}).get("mean_us", 0.0))
    python_unattributed_us = max(
        0.0,
        total_profile_us
        - overhead_us
        - static_decode_us
        - stage_mean("softmax_value")
        - stage_mean("full_attn_logits"),
    )
    return {
        "static_decode_us": static_decode_us,
        "selector_estimator_pruner_us": overhead_us,
        "overhead_stages": "+".join(overhead_stages),
        "profile_softmax_value_us": stage_mean("softmax_value"),
        "profile_total_decode_attention_us": total_profile_us,
        "python_unattributed_us": python_unattributed_us,
        "source_profile": str(summary_path(method, length).relative_to(Path(__file__).resolve().parents[3])),
    }


def nearest_attention_row(rows: List[dict], method: str, length: int, batch_size: int) -> dict:
    candidates = [row for row in rows if row["method"] == method and row["length"] == length]
    if not candidates:
        raise ValueError(f"No attention-op row for method={method}, length={length}")
    exact = [row for row in candidates if row["batch_size"] == batch_size]
    if exact:
        return exact[0]
    return min(candidates, key=lambda row: abs(row["batch_size"] - batch_size))


def add_speedups(rows: List[dict]) -> None:
    selector_baseline: Dict[tuple, float] = {}
    twi_baseline: Dict[tuple, float] = {}
    for row in rows:
        key = (row["selector"], row["length"], row["batch_size"])
        if row["variant"] == "selector_only":
            selector_baseline[key] = row["tpot_ms"]
        if row["variant"] == "twi":
            twi_baseline[key] = row["tpot_ms"]
    for row in rows:
        key = (row["selector"], row["length"], row["batch_size"])
        selector_base = selector_baseline.get(key)
        twi_base = twi_baseline.get(key)
        row["speedup_vs_selector"] = (selector_base / row["tpot_ms"]) if selector_base else 1.0
        row["speedup_vs_own_twi"] = (twi_base / row["tpot_ms"]) if twi_base else 1.0


def render_markdown(rows: Iterable[dict], attention_op_csv: Path) -> str:
    lines = [
        "# Simulated Decode TPOT Benchmark",
        "",
        "This benchmark composes profile-derived overhead with attention-op replay latency to estimate batched decode TPOT.",
        "",
        "Important: this is simulated TPOT, not real HF/vLLM serving TPOT.",
        "",
        f"- attention-op source: `{attention_op_csv}`",
        "",
        "| method | length | batch | static us | overhead us | sparse attention us | total decode us | TPOT ms | speedup vs selector | speedup vs Twi |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {length} | {batch_size} | {static_decode_us:.2f} | "
            "{selector_estimator_pruner_us:.2f} | {sparse_attention_us:.2f} | "
            "{total_decode_us:.2f} | {tpot_ms:.4f} | {speedup_vs_selector:.2f}x | "
            "{speedup_vs_own_twi:.2f}x |".format(**row)
        )
    lines.extend(["", "Notes:", ""])
    lines.append("- `total_decode_us = static_decode_us + selector_estimator_pruner_us + sparse_attention_us`.")
    lines.append("- `TPOT ms = total_decode_us / batch_size / 1000`.")
    lines.append("- `quest` and `ds` use only selector/final-mask overhead; Twi adds estimator/top-p overhead; group-max adds GQA refinement overhead.")
    lines.append("- Static and overhead terms are profile-derived per-layer decode attention costs; large-batch behavior is approximated.")
    lines.append("- Use this table as serving-style trend evidence, not as measured model.generate/vLLM TPOT.")
    return "\n".join(lines) + "\n"


def write_csv(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "selector",
        "variant",
        "length",
        "batch_size",
        "static_decode_us",
        "selector_estimator_pruner_us",
        "overhead_stages",
        "sparse_attention_us",
        "attention_op_batch_used",
        "total_decode_us",
        "tpot_ms",
        "speedup_vs_selector",
        "speedup_vs_own_twi",
        "profile_softmax_value_us",
        "profile_total_decode_attention_us",
        "python_unattributed_us",
        "source_profile",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention-op-csv", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["quest", "quest_twi", "quest_groupmax_ex16", "ds", "ds_twi", "ds_groupmax_ex16"],
        choices=sorted(METHOD_META),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    attention_rows = read_attention_rows(args.attention_op_csv)
    rows = []
    for length in args.lengths:
        for method in args.methods:
            selector, variant, _ = METHOD_META[method]
            components = profile_components(method, length)
            for batch_size in args.batch_sizes:
                attention_row = nearest_attention_row(attention_rows, method, length, batch_size)
                sparse_attention_us = float(attention_row["sparse_attention_us"])
                total_decode_us = (
                    components["static_decode_us"]
                    + components["selector_estimator_pruner_us"]
                    + sparse_attention_us
                )
                rows.append(
                    {
                        "method": method,
                        "selector": selector,
                        "variant": variant,
                        "length": length,
                        "batch_size": batch_size,
                        "static_decode_us": components["static_decode_us"],
                        "selector_estimator_pruner_us": components["selector_estimator_pruner_us"],
                        "sparse_attention_us": sparse_attention_us,
                        "attention_op_batch_used": attention_row["batch_size"],
                        "total_decode_us": total_decode_us,
                        "tpot_ms": total_decode_us / batch_size / 1000.0,
                        **components,
                    }
                )
    add_speedups(rows)
    markdown = render_markdown(rows, args.attention_op_csv)
    print(markdown)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    if args.csv_output:
        write_csv(rows, args.csv_output)


if __name__ == "__main__":
    main()
