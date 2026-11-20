"""Fixed-support Quest adapter audit, not an end-to-end Quest benchmark.

Compare the legacy per-head kernel with FlashInfer on identical Q/K/V and
candidate IDs. Gather, index preparation, and random-score top-k are separate
diagnostics; none is silently added to either attention timing.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_e4_kernel_evidence import (  # noqa: E402
    affine_tokens, cuda_event_samples, error_metrics, fp32_reference, padded_tensor,
)
from bench_e6_kernel_baselines import get_gas_module  # noqa: E402

COMPONENTS = ("format", "gather", "topk", "attention", "paged_metadata", "flashinfer_attention")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def make_indices(batch: int, heads: int, length: int, budget: int, device: torch.device) -> torch.Tensor:
    lists = [affine_tokens(length, budget, 910001 + linear * 31) for linear in range(batch * heads)]
    return padded_tensor(lists, (batch, heads), device, width=budget)


def explicit_gather(
    k: torch.Tensor, v: torch.Tensor, indices_long: torch.Tensor,
    out_k: torch.Tensor, out_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand GQA heads as a view and gather directly into preallocated outputs."""
    batch, kv_heads, length, dim = k.shape
    group_size = indices_long.shape[1] // kv_heads
    budget = indices_long.shape[-1]
    selected = indices_long.view(batch, kv_heads, group_size, budget, 1).expand(-1, -1, -1, -1, dim)
    key = k.view(batch, kv_heads, 1, length, dim).expand(-1, -1, group_size, -1, -1)
    value = v.view(batch, kv_heads, 1, length, dim).expand_as(key)
    torch.gather(key, 3, selected, out=out_k.view(batch, kv_heads, group_size, budget, dim))
    torch.gather(value, 3, selected, out=out_v.view(batch, kv_heads, group_size, budget, dim))
    return out_k, out_v


def paged_metadata(indices: torch.Tensor, kv_heads: int, length: int) -> tuple[torch.Tensor, ...]:
    """Map each (batch, query head) to a logical single-head paged request.

    Physical pages are one token from one KV head. K/V can therefore be viewed
    as pages without duplicating any KV, even when GQA heads choose different IDs.
    """
    batch, heads, budget = indices.shape
    group_size = heads // kv_heads
    head = torch.arange(heads, device=indices.device, dtype=torch.int32) // group_size
    request = torch.arange(batch, device=indices.device, dtype=torch.int32)
    offsets = (request[:, None] * kv_heads + head[None, :]) * length
    page_ids = (indices + offsets[..., None]).reshape(-1)
    indptr = torch.arange(batch * heads + 1, device=indices.device, dtype=torch.int32) * budget
    last_page_len = torch.ones(batch * heads, device=indices.device, dtype=torch.int32)
    return indptr, page_ids, last_page_len


def flashinfer_adapter(q, k, v, indices):
    import flashinfer

    indptr, page_ids, last_page_len = paged_metadata(indices, k.shape[1], k.shape[2])
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
    torch.cuda.synchronize(q.device)
    start = time.perf_counter()
    wrapper.plan(
        indptr, page_ids, last_page_len, num_qo_heads=1, num_kv_heads=1,
        head_dim=q.shape[-1], page_size=1, pos_encoding_mode="NONE",
        q_data_type=q.dtype, kv_data_type=k.dtype, sm_scale=1.0 / math.sqrt(q.shape[-1]),
    )
    torch.cuda.synchronize(q.device)
    plan_ms = (time.perf_counter() - start) * 1000
    query = q.view(-1, 1, q.shape[-1])
    pages = (k.view(-1, 1, 1, q.shape[-1]), v.view(-1, 1, 1, q.shape[-1]))
    output = torch.empty_like(query)

    def run():
        wrapper.run(query, pages, out=output)
        return output.view_as(q)

    return run, plan_ms


def profile_kernels(function: Callable) -> list[dict]:
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
        function()
        torch.cuda.synchronize()
    kernels = {}
    for event in profile.events():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        item = kernels.setdefault(event.name, {"kernel_name": event.name, "count": 0, "total_gpu_us": 0.0})
        item["count"] += 1
        item["total_gpu_us"] += event.device_time_total
    if not kernels:
        raise RuntimeError("profiler returned no CUDA events; kernel counts are unavailable")
    return list(kernels.values())


def save_results(output_dir, args, rows, samples, correctness, statuses, kernel_rows):
    write_csv(output_dir / "quest_budget_timings_raw.csv", rows)
    write_csv(output_dir / "quest_budget_samples.csv", samples)
    write_csv(output_dir / "quest_budget_correctness.csv", correctness)
    write_csv(output_dir / "quest_budget_status.csv", statuses)
    write_csv(output_dir / "quest_budget_kernels.csv", kernel_rows)
    summary = []
    report = [
        "# Quest Budget Sweep v2", "",
        "Fixed Q/K/V and nested affine candidate supports. This is an adapter audit, not official Quest or TPOT.",
        "All timings are pooled CUDA-event medians in microseconds; allocation/JIT/plan are outside attention timing.",
        "Random-score top-k is a standalone diagnostic, not Quest page scoring or a connected selection pipeline.", "",
        "| Budget | Legacy attention us | FlashInfer same-support us | Legacy/FI | Gather us | Index format us | Paged metadata us | Top-k us |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for budget in args.budgets:
        values = {}
        for component in COMPONENTS:
            values[component] = [r["latency_us"] for r in samples if r["budget"] == budget and r["component"] == component]
        medians = {name: statistics.median(value) if value else None for name, value in values.items()}
        summary.append({
            "length": args.length, "batch_size": args.batch_size, "budget": budget,
            **{name + "_median_us": value if value is not None else "" for name, value in medians.items()},
            **{name + "_sample_count": len(value) for name, value in values.items()},
            "scope": "fixed_support_adapter_audit_v2",
        })
        a, fi = medians["attention"], medians["flashinfer_attention"]
        ratio = f"{a / fi:.2f}" if a is not None and fi is not None else "unavailable"
        fmt = lambda name: f"{medians[name]:.3f}" if medians[name] is not None else "unavailable"
        report.append(f"| {budget} | {fmt('attention')} | {fmt('flashinfer_attention')} | {ratio} | {fmt('gather')} | {fmt('format')} | {fmt('paged_metadata')} | {fmt('topk')} |")
    report.extend([
        "", "FlashInfer maps every query head to its own logical request and references the original GQA KV storage.",
        "This preserves per-head candidate IDs and shares physical KV without a head-expanded gather.",
        "The gather diagnostic uses two preallocated outputs; it is not consumed by either indirect attention kernel.",
        "Do not add gather to the indirect attention latency. Metadata and plan must be included when measuring a dynamic pipeline.",
        "Plan setup wall time is recorded only for diagnosis (may include first JIT); it is not a steady-state per-step measurement.",
        "Unavailable components and profiler failures are recorded in quest_budget_status.csv; completed samples are retained.",
        "No kernel call counts are inferred: --profile-kernels records actual CUDA events, separately from timing.",
        "A corrected Estimated TPOT additionally needs real connected selection/refinement, identical upstream candidates, and dense-layer accounting.",
    ])
    write_csv(output_dir / "quest_budget_summary.csv", summary)
    (output_dir / "Quest_Budget_Sweep.md").write_text("\n".join(report) + "\n", encoding="utf-8")


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmark/results_efficiency/quest_budget_sweep_v2")
    parser.add_argument("--length", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--budgets", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--components", nargs="+", choices=COMPONENTS, default=list(COMPONENTS))
    parser.add_argument("--profile-kernels", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.num_heads, args.num_kv_heads, args.head_dim, args.repeats, args.rounds) <= 0 or args.warmups < 0:
        parser.error("dimensions, repeats and rounds must be positive; warmups must be nonnegative")
    if args.num_heads % args.num_kv_heads or any(b <= 0 or b > args.length for b in args.budgets):
        parser.error("query heads must divide into KV groups; budgets must be in [1, length]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if any(args.output_dir.glob("quest_budget_*.csv")):
        parser.error("output directory contains earlier results; choose a new output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    q = torch.randn(args.batch_size, args.num_heads, args.head_dim, device=device, dtype=torch.float16)
    k = torch.randn(args.batch_size, args.num_kv_heads, args.length, args.head_dim, device=device, dtype=torch.float16)
    v = torch.randn_like(k)
    scores = torch.randn(args.batch_size, args.num_heads, args.length, device=device, dtype=torch.float16) if "topk" in args.components else None
    module = get_gas_module() if "attention" in args.components else None
    manifest = {**vars(args), "output_dir": str(args.output_dir), "protocol": "fixed_support_adapter_audit_v2", "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device), "full_kv_gib": 2 * k.numel() * k.element_size() / 2**30, "attention_scope": "kernel_only_static_support_no_plan_no_gather", "selection_scope": "random_score_topk_only_not_Quest_selection"}
    (args.output_dir / "config.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    rows, samples, correctness, statuses, kernel_rows = [], [], [], [], []
    for budget in args.budgets:
        print(f"start budget={budget}, length={args.length}, batch={args.batch_size}", flush=True)
        indices = make_indices(args.batch_size, args.num_heads, args.length, budget, device)
        count = min(args.batch_size, 4)
        reference = fp32_reference(q[:count], k[:count], v[:count], indices[:count], [budget] * (count * args.num_heads), args.num_heads // args.num_kv_heads)
        disabled = set()
        for round_idx in range(args.rounds):
            order = list(args.components)
            random.Random(args.seed + budget * 101 + round_idx).shuffle(order)
            for component in order:
                if component in disabled:
                    continue
                function = None
                out_k = out_v = indices_long = output = None
                try:
                    if component == "gather":
                        indices_long = indices.long()
                        out_k = torch.empty(args.batch_size, args.num_heads, budget, args.head_dim, device=device, dtype=k.dtype)
                        out_v = torch.empty_like(out_k)
                        function = lambda: explicit_gather(k, v, indices_long, out_k, out_v)
                    elif component == "format":
                        function = lambda: indices.long()
                    elif component == "topk":
                        function = lambda: torch.topk(scores, budget, dim=-1, sorted=False)
                    elif component == "paged_metadata":
                        function = lambda: paged_metadata(indices, args.num_kv_heads, args.length)
                    elif component == "attention":
                        output = torch.empty_like(q)
                        function = lambda: module.batched_per_head_sparse_decode_out(q, k, v, indices, output)
                    else:
                        function, plan_ms = flashinfer_adapter(q, k, v, indices)
                        statuses.append({"budget": budget, "round": round_idx + 1, "component": component, "status": "plan_setup_only", "reason": "outside kernel timing; may include JIT", "setup_wall_ms": plan_ms})
                    if round_idx == 0 and component in ("attention", "flashinfer_attention"):
                        actual = function()
                        metrics = error_metrics(actual[:count], reference, 2e-2, 2e-2)
                        metrics["finite"] = bool(torch.isfinite(actual).all().item())
                        del actual
                        correctness.append({"budget": budget, "component": component, "reference_batch": count, **metrics})
                        if not metrics["pass"] or not metrics["finite"]:
                            raise RuntimeError(f"{component} failed correctness: {metrics}")
                    values = cuda_event_samples(function, device, args.warmups, args.repeats)
                    samples.extend({"length": args.length, "batch_size": args.batch_size, "budget": budget, "round": round_idx + 1, "repeat": repeat_idx + 1, "component": component, "latency_us": value, "order": ">".join(order)} for repeat_idx, value in enumerate(values))
                    rows.append({"length": args.length, "batch_size": args.batch_size, "budget": budget, "round": round_idx + 1, "component": component, "repeat_count": len(values), "latency_us_mean": statistics.fmean(values), "latency_us_median": statistics.median(values), "order": ">".join(order), "aggregation": "per_round"})
                    statuses.append({"budget": budget, "round": round_idx + 1, "component": component, "status": "available", "reason": "", "setup_wall_ms": ""})
                    if args.profile_kernels and round_idx == 0:
                        try:
                            kernel_rows.extend({"budget": budget, "component": component, **item} for item in profile_kernels(function))
                        except Exception as exc:
                            statuses.append({"budget": budget, "round": 1, "component": component, "status": "profiler_unavailable", "reason": str(exc), "setup_wall_ms": ""})
                    print(f"finished budget={budget}, round={round_idx + 1}/{args.rounds}, {component} median={statistics.median(values):.3f} us", flush=True)
                except (torch.OutOfMemoryError, ImportError) as exc:
                    disabled.add(component)
                    statuses.append({"budget": budget, "round": round_idx + 1, "component": component, "status": "unavailable", "reason": str(exc), "setup_wall_ms": ""})
                    print(f"unavailable budget={budget}, {component}: {exc}", flush=True)
                finally:
                    function = out_k = out_v = indices_long = output = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    save_results(args.output_dir, args, rows, samples, correctness, statuses, kernel_rows)
        del indices, reference
        torch.cuda.empty_cache()
    print(f"wrote Quest budget sweep to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
