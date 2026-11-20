import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, List

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from bench_batched_gqa_fused import (  # noqa: E402
    METHOD_META,
    benchmark,
    get_module,
    make_group_indices,
    make_head_indices,
    render_markdown as render_kernel_markdown,
    run_scenario,
    scenario_from_summary,
    synchronize,
)


def benchmark_o_proj(
    sparse_out: torch.Tensor,
    warmups: int,
    repeats: int,
    device: torch.device,
) -> float:
    batch_size, num_heads, head_dim = sparse_out.shape
    hidden_size = num_heads * head_dim
    flat = sparse_out.reshape(batch_size, hidden_size).contiguous()
    torch.manual_seed(1)
    weight = torch.randn(hidden_size, hidden_size, device=device, dtype=sparse_out.dtype)
    return benchmark(lambda: torch.matmul(flat, weight), device, warmups, repeats)


def make_sparse_output(
    module,
    scenario,
    batch_size: int,
    device: torch.device,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    torch.manual_seed(0)
    q = torch.randn(batch_size, num_heads, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch_size, num_kv_heads, scenario.length, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch_size, num_kv_heads, scenario.length, head_dim, device=device, dtype=torch.float16)
    if scenario.kernel == "per_head":
        indices = make_head_indices(batch_size, num_heads, scenario.selected, scenario.length, device)
        return module.batched_per_head_sparse_decode(q, k, v, indices)
    indices = make_group_indices(batch_size, num_kv_heads, scenario.selected, scenario.length, device)
    return module.batched_gqa_group_shared_decode(q, k, v, indices)


def add_speedups(rows: List[dict]) -> None:
    selector_baseline = {}
    twi_baseline = {}
    for row in rows:
        key = (row["selector"], row["length"], row["batch_size"])
        if row["variant"] == "selector_only":
            selector_baseline[key] = row["attention_op_total_us"]
        if row["variant"] == "twi":
            twi_baseline[key] = row["attention_op_total_us"]
    for row in rows:
        key = (row["selector"], row["length"], row["batch_size"])
        selector_base = selector_baseline.get(key)
        twi_base = twi_baseline.get(key)
        row["speedup_vs_selector"] = (selector_base / row["attention_op_total_us"]) if selector_base else 1.0
        row["speedup_vs_own_twi"] = (twi_base / row["attention_op_total_us"]) if twi_base else 1.0


def render_markdown(rows: Iterable[dict], warmups: int, repeats: int) -> str:
    lines = [
        "# Attention-op Replay Benchmark",
        "",
        "This benchmark replays sparse decode attention with random tensors, using profile-derived selected sizes.",
        "It measures fused sparse attention plus a mock dense `o_proj` matmul, without modifying `attention.py`.",
        "",
        "Settings:",
        "",
        f"- dtype: `torch.float16`",
        f"- warmups: `{warmups}`",
        f"- repeats: `{repeats}`",
        "- model shape: `num_heads=32`, `num_kv_heads=8`, `group_size=4`, `head_dim=128`, `hidden_size=4096`",
        "",
        "| method | length | batch | kernel | selected | loaded KV mean | sparse attention us | o_proj us | attention-op total us | speedup vs selector | speedup vs Twi |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {length} | {batch_size} | {kernel} | {selected} | {loaded_kv_mean:.2f} | "
            "{sparse_attention_us:.2f} | {o_proj_us:.2f} | {attention_op_total_us:.2f} | "
            "{speedup_vs_selector:.2f}x | {speedup_vs_own_twi:.2f}x |".format(**row)
        )
    lines.extend(["", "Sources:", ""])
    seen = set()
    for row in rows:
        if row["source"] in seen:
            continue
        seen.add(row["source"])
        lines.append(f"- `{row['source']}`")
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- `quest` and `ds` are selector-only baselines with no Twi estimator/top-p pruning.",
            "- `quest_twi` and `ds_twi` use fused per-head sparse decode.",
            "- `*_groupmax_ex16` means selector + Twi estimator/top-p + `group_max p=0.80` + `exception_budget=16`; it uses fused GQA group-shared sparse decode.",
            "- `o_proj` is a random dense `[hidden, hidden]` matmul replay, not the model's actual weights.",
            "- This is an attention-operator replay benchmark, not full model serving latency.",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "selector",
        "variant",
        "kernel",
        "length",
        "batch_size",
        "selected",
        "selected_source_value",
        "loaded_kv_mean",
        "sparse_attention_us",
        "o_proj_us",
        "attention_op_total_us",
        "speedup_vs_selector",
        "speedup_vs_own_twi",
        "source",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 16, 32, 64])
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["quest", "quest_twi", "quest_groupmax_ex16", "ds", "ds_twi", "ds_groupmax_ex16"],
        choices=sorted(METHOD_META),
    )
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--skip-reference-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.num_heads % args.num_kv_heads != 0:
        raise ValueError("--num-heads must be divisible by --num-kv-heads")

    module = get_module()
    rows = []
    for length in args.lengths:
        for method in args.methods:
            scenario = scenario_from_summary(method, length, args.num_heads, args.num_kv_heads)
            for batch_size in args.batch_sizes:
                kernel_row = run_scenario(
                    module,
                    scenario,
                    batch_size,
                    device,
                    args.warmups,
                    args.repeats,
                    args.num_heads,
                    args.num_kv_heads,
                    args.head_dim,
                    check=not args.skip_reference_check,
                )
                sparse_out = make_sparse_output(
                    module,
                    scenario,
                    batch_size,
                    device,
                    args.num_heads,
                    args.num_kv_heads,
                    args.head_dim,
                )
                synchronize(device)
                o_proj_us = benchmark_o_proj(sparse_out, args.warmups, args.repeats, device)
                rows.append(
                    {
                        **kernel_row,
                        "sparse_attention_us": kernel_row["latency_us"],
                        "o_proj_us": o_proj_us,
                        "attention_op_total_us": kernel_row["latency_us"] + o_proj_us,
                    }
                )
    add_speedups(rows)
    markdown = render_markdown(rows, args.warmups, args.repeats)
    print(markdown)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    if args.csv_output:
        write_csv(rows, args.csv_output)


if __name__ == "__main__":
    main()
