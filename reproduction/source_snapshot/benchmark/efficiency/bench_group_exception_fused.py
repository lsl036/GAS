import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from bench_batched_gqa_fused import (  # noqa: E402
    METHOD_META,
    benchmark,
    get_module,
    make_group_indices,
    scenario_from_summary,
    summary_path,
    synchronize,
)
from bench_simulated_decode_tpot import STATIC_STAGES  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def make_exception_indices(
    batch_size: int,
    num_heads: int,
    exception_budget: int,
    seq_len: int,
    shared_selected: int,
    device: torch.device,
) -> torch.Tensor:
    if exception_budget == 0:
        return torch.empty(batch_size, num_heads, 0, dtype=torch.int32, device=device)
    base = torch.linspace(
        min(seq_len - 1, shared_selected + 1),
        max(1, min(seq_len - 1, shared_selected + exception_budget)),
        exception_budget,
        device=device,
        dtype=torch.float32,
    ).round().to(torch.int32)
    return base.view(1, 1, exception_budget).expand(
        batch_size, num_heads, exception_budget
    ).contiguous()


def torch_group_exception_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    group_indices: torch.Tensor,
    exception_indices: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    outputs = []
    scale = 1.0 / (q.shape[-1] ** 0.5)
    for batch_idx in range(q.shape[0]):
        head_outputs = []
        for head_idx in range(q.shape[1]):
            kv_idx = head_idx // group_size
            merged = torch.cat(
                [
                    group_indices[batch_idx, kv_idx],
                    exception_indices[batch_idx, head_idx],
                ]
            ).to(torch.long)
            valid = merged[(merged >= 0) & (merged < k.shape[2])]
            valid = torch.unique(valid, sorted=False)
            k_sel = k[batch_idx, kv_idx].index_select(0, valid)
            v_sel = v[batch_idx, kv_idx].index_select(0, valid)
            probs = torch.softmax(torch.matmul(q[batch_idx, head_idx], k_sel.T) * scale, dim=-1)
            head_outputs.append(torch.matmul(probs, v_sel))
        outputs.append(torch.stack(head_outputs, dim=0))
    return torch.stack(outputs, dim=0)


def stage_mean(data: dict, name: str) -> float:
    return float(data.get("stages", {}).get(name, {}).get("mean_us", 0.0))


def profile_static_us(method: str, length: int) -> float:
    data = json.loads(summary_path(method, length).read_text(encoding="utf-8"))
    return sum(stage_mean(data, name) for name in STATIC_STAGES)


def profile_twi_overhead_us(method: str, length: int, include_final_mask: bool = True) -> float:
    data = json.loads(summary_path(method, length).read_text(encoding="utf-8"))
    names = ["selector", "fake_quant_estimator", "top_p_prune"]
    if include_final_mask:
        names.append("final_mask")
    return sum(stage_mean(data, name) for name in names)


def read_attention_sparse_us(
    attention_op_rows: List[dict],
    method: str,
    length: int,
    batch_size: int,
) -> float:
    candidates = [
        row
        for row in attention_op_rows
        if row["method"] == method and int(row["length"]) == length
    ]
    if not candidates:
        raise ValueError(f"No attention-op row for method={method}, length={length}")
    exact = [row for row in candidates if int(row["batch_size"]) == batch_size]
    row = exact[0] if exact else min(
        candidates, key=lambda item: abs(int(item["batch_size"]) - batch_size)
    )
    return float(row["sparse_attention_us"])


def read_attention_rows(path: Optional[Path]) -> List[dict]:
    if path is None:
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def profile_twi_total_us(selector: str, length: int, twi_sparse_us: float) -> float:
    method = f"{selector}_twi"
    return profile_static_us(method, length) + profile_twi_overhead_us(method, length) + twi_sparse_us


def run_group_exception(
    module,
    method: str,
    length: int,
    batch_size: int,
    device: torch.device,
    warmups: int,
    repeats: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    exception_budget: int,
    check: bool,
    attention_op_rows: List[dict],
) -> dict:
    scenario = scenario_from_summary(method, length, num_heads, num_kv_heads)
    selector, _, _ = METHOD_META[method]
    group_size = num_heads // num_kv_heads
    shared_selected = max(1, scenario.selected - exception_budget)

    torch.manual_seed(0)
    q = torch.randn(batch_size, num_heads, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch_size, num_kv_heads, length, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch_size, num_kv_heads, length, head_dim, device=device, dtype=torch.float16)
    group_indices = make_group_indices(batch_size, num_kv_heads, shared_selected, length, device)
    exception_indices = make_exception_indices(
        batch_size,
        num_heads,
        exception_budget,
        length,
        shared_selected,
        device,
    )

    fused = lambda: module.batched_gqa_group_exception_decode(
        q, k, v, group_indices, exception_indices
    )
    fused_out = fused()
    if check:
        torch.testing.assert_close(
            fused_out,
            torch_group_exception_decode(q, k, v, group_indices, exception_indices, group_size),
            rtol=2e-2,
            atol=2e-2,
        )
    if not torch.isfinite(fused_out).all():
        raise RuntimeError(f"{method} batch={batch_size} produced NaN or Inf")

    exception_decode_us = benchmark(fused, device, warmups, repeats)
    twi_sparse_us = read_attention_sparse_us(
        attention_op_rows,
        f"{selector}_twi",
        length,
        batch_size,
    )
    simulated_twi_total = profile_twi_total_us(selector, length, twi_sparse_us)

    static_us = profile_static_us(f"{selector}_twi", length)
    twi_overhead_us = profile_twi_overhead_us(
        f"{selector}_twi",
        length,
        include_final_mask=False,
    )
    estimated_compact_us = exception_decode_us * 0.10
    fused_total_us = static_us + twi_overhead_us + estimated_compact_us + exception_decode_us

    return {
        "method": method,
        "selector": selector,
        "length": length,
        "batch_size": batch_size,
        "group_selected": shared_selected,
        "profile_post_union_selected": scenario.selected,
        "exception_budget": exception_budget,
        "exception_decode_us": exception_decode_us,
        "estimated_compact_us": estimated_compact_us,
        "static_us": static_us,
        "twi_overhead_us": twi_overhead_us,
        "estimated_total_us": fused_total_us,
        "estimated_tpot_ms": fused_total_us / batch_size / 1000.0,
        "baseline_twi_total_us": simulated_twi_total,
        "speedup_vs_twi": simulated_twi_total / fused_total_us,
        "source": str(scenario.source.relative_to(PROJECT_ROOT)),
    }


def render_markdown(rows: Iterable[dict], warmups: int, repeats: int) -> str:
    lines = [
        "# Fused Group Exception Decode Benchmark",
        "",
        "This benchmark estimates the TPOT target after moving Method B's group/exception decode path into CUDA.",
        "",
        "Settings:",
        "",
        f"- warmups: `{warmups}`",
        f"- repeats: `{repeats}`",
        "- `estimated_compact_us` is a conservative 10% of the fused exception decode time.",
        "- `twi_overhead_us` keeps selector + estimator + top-p, but excludes full final-mask because fused decode consumes compact indices directly.",
        "- `estimated_total_us = static_us + twi_overhead_us + estimated_compact_us + exception_decode_us`.",
        "",
        "| method | length | batch | group selected | exception | decode us | compact est us | estimated total us | TPOT ms | speedup vs Twi |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {length} | {batch_size} | {group_selected} | {exception_budget} | "
            "{exception_decode_us:.2f} | {estimated_compact_us:.2f} | {estimated_total_us:.2f} | "
            "{estimated_tpot_ms:.4f} | {speedup_vs_twi:.2f}x |".format(**row)
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- This is not yet a model-runtime path; it is the CUDA target line for the next integration step.",
            "- If `speedup_vs_twi > 1`, then the remaining requirement is to implement compaction/refinement with similar low overhead in the real decode path.",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "selector",
        "length",
        "batch_size",
        "group_selected",
        "profile_post_union_selected",
        "exception_budget",
        "exception_decode_us",
        "estimated_compact_us",
        "static_us",
        "twi_overhead_us",
        "estimated_total_us",
        "estimated_tpot_ms",
        "baseline_twi_total_us",
        "speedup_vs_twi",
        "source",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["quest_groupmax_ex16", "ds_groupmax_ex16"],
        choices=["quest_groupmax_ex16", "ds_groupmax_ex16"],
    )
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--exception-budget", type=int, default=16)
    parser.add_argument(
        "--attention-op-csv",
        type=Path,
        default=PROJECT_ROOT / "thoughts" / "Efficiency" / "attention_op_replay_benchmark.csv",
    )
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
    attention_op_rows = read_attention_rows(args.attention_op_csv)
    rows: List[dict] = []
    for length in args.lengths:
        for method in args.methods:
            for batch_size in args.batch_sizes:
                rows.append(
                    run_group_exception(
                        module,
                        method,
                        length,
                        batch_size,
                        device,
                        args.warmups,
                        args.repeats,
                        args.num_heads,
                        args.num_kv_heads,
                        args.head_dim,
                        args.exception_budget,
                        check=not args.skip_reference_check,
                        attention_op_rows=attention_op_rows,
                    )
                )
    markdown = render_markdown(rows, args.warmups, args.repeats)
    print(markdown)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    if args.csv_output:
        write_csv(rows, args.csv_output)


if __name__ == "__main__":
    main()
