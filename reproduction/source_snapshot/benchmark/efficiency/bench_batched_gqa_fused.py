import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import torch
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent


@dataclass(frozen=True)
class Scenario:
    method: str
    selector: str
    variant: str
    length: int
    kernel: str
    source: Path
    selected: int
    loaded_kv_mean: float
    selected_source_value: float


SUMMARY_PATHS = {
    ("quest", 4096): "phase2_group_max_p080_llama31_4k/profile_summary.json",
    ("quest", 8192): "phase3_group_max_p080_llama31_8k/profile_summary.json",
    ("quest", 16384): "phase3_group_max_p080_exception16_llama31_16k/profile_summary.json",
    ("quest", 32768): "phase_long_group_max_p080_exception16_llama31_32k/profile_summary.json",
    ("quest", 65536): "phase_long_group_max_p080_exception16_llama31_64k/profile_summary.json",
    ("quest_twi", 4096): "phase2_group_max_p080_llama31_4k/profile_summary.json",
    ("quest_twi", 8192): "phase3_group_max_p080_llama31_8k/profile_summary.json",
    ("quest_twi", 16384): "phase3_group_max_p080_exception16_llama31_16k/profile_summary.json",
    ("quest_twi", 32768): "phase_long_group_max_p080_exception16_llama31_32k/profile_summary.json",
    ("quest_twi", 65536): "phase_long_group_max_p080_exception16_llama31_64k/profile_summary.json",
    ("quest_groupmax_ex16", 4096): "phase2_group_max_p080_llama31_4k/profile_summary.json",
    ("quest_groupmax_ex16", 8192): "phase3_group_max_p080_exception16_llama31_8k/profile_summary.json",
    ("quest_groupmax_ex16", 16384): "phase3_group_max_p080_exception16_llama31_16k/profile_summary.json",
    ("quest_groupmax_ex16", 32768): "phase_long_group_max_p080_exception16_llama31_32k/profile_summary.json",
    ("quest_groupmax_ex16", 65536): "phase_long_group_max_p080_exception16_llama31_64k/profile_summary.json",
    ("ds", 4096): "ds_ablation_llama31_group_max_p080_exception16_4k/profile_summary.json",
    ("ds", 8192): "ds_ablation_llama31_group_max_p080_exception16_8k/profile_summary.json",
    ("ds_twi", 4096): "ds_ablation_llama31_group_max_p080_exception16_4k/profile_summary.json",
    ("ds_twi", 8192): "ds_ablation_llama31_group_max_p080_exception16_8k/profile_summary.json",
    ("ds_groupmax_ex16", 4096): "ds_ablation_llama31_group_max_p080_exception16_4k/profile_summary.json",
    ("ds_groupmax_ex16", 8192): "ds_ablation_llama31_group_max_p080_exception16_8k/profile_summary.json",
}


METHOD_META = {
    "quest": ("quest", "selector_only", "per_head"),
    "quest_twi": ("quest", "twi", "per_head"),
    "quest_groupmax_ex16": ("quest", "group_max_p080_exception16", "group"),
    "ds": ("ds", "selector_only", "per_head"),
    "ds_twi": ("ds", "twi", "per_head"),
    "ds_groupmax_ex16": ("ds", "group_max_p080_exception16", "group"),
}


SELECTOR_ONLY_TOKEN_BUDGET = 8192


def get_module():
    return load(
        name="twilight_gqa_group_decode",
        sources=[str(ROOT / "csrc" / "src" / "gqa_group_decode.cu")],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
        ],
        verbose=False,
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(fn: Callable[[], torch.Tensor], device: torch.device, warmups: int, repeats: int) -> float:
    for _ in range(warmups):
        fn()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    synchronize(device)
    return (time.perf_counter() - start) / repeats * 1e6


def read_nested(data: dict, dotted_key: str) -> float:
    cur = data
    for part in dotted_key.split("."):
        cur = cur[part]
    return float(cur)


def summary_path(method: str, length: int) -> Path:
    try:
        rel = SUMMARY_PATHS[(method, length)]
    except KeyError as exc:
        raise ValueError(f"No summary path configured for method={method}, length={length}") from exc
    return ROOT / "benchmark" / "results_passkey_ours" / rel


def scenario_from_summary(method: str, length: int, num_heads: int, num_kv_heads: int) -> Scenario:
    source = summary_path(method, length)
    data = json.loads(source.read_text(encoding="utf-8"))
    selector, variant, kernel = METHOD_META[method]
    if variant == "selector_only":
        selected_value = min(float(length), float(SELECTOR_ONLY_TOKEN_BUDGET))
        loaded_kv_mean = selected_value * num_kv_heads
    elif kernel == "per_head":
        selected_value = read_nested(data, "gqa.pre_per_head_selected_mean.mean_us")
        loaded_kv_mean = read_nested(data, "gqa.pre_loaded_kv_tokens_mean.mean_us")
    else:
        selected_value = read_nested(data, "gqa.post_group_union_count_mean.mean_us")
        loaded_kv_mean = read_nested(data, "gqa.post_loaded_kv_tokens_mean.mean_us")
    selected = max(1, math.ceil(selected_value))
    seq_len_mean = read_nested(data, "seq_len.mean_us")
    if abs(seq_len_mean - length) > length * 0.2:
        raise ValueError(f"Unexpected seq_len in {source}: mean={seq_len_mean}, expected around {length}")
    return Scenario(
        method=method,
        selector=selector,
        variant=variant,
        length=length,
        kernel=kernel,
        source=source,
        selected=selected,
        loaded_kv_mean=loaded_kv_mean,
        selected_source_value=selected_value,
    )


def make_group_indices(batch_size: int, num_kv_heads: int, selected: int, seq_len: int, device: torch.device) -> torch.Tensor:
    base = torch.linspace(0, seq_len - 1, selected, device=device, dtype=torch.float32).round().to(torch.int32)
    return base.view(1, 1, selected).expand(batch_size, num_kv_heads, selected).contiguous()


def make_head_indices(batch_size: int, num_heads: int, selected: int, seq_len: int, device: torch.device) -> torch.Tensor:
    base = torch.linspace(0, seq_len - 1, selected, device=device, dtype=torch.float32).round().to(torch.int32)
    return base.view(1, 1, selected).expand(batch_size, num_heads, selected).contiguous()


def torch_group_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    outputs = []
    scale = 1.0 / math.sqrt(q.shape[-1])
    for batch_idx in range(q.shape[0]):
        batch_outputs = []
        for kv_idx in range(k.shape[1]):
            idx = indices[batch_idx, kv_idx].to(torch.long)
            k_sel = k[batch_idx, kv_idx].index_select(0, idx)
            v_sel = v[batch_idx, kv_idx].index_select(0, idx)
            q_group = q[batch_idx, kv_idx * group_size : (kv_idx + 1) * group_size]
            probs = torch.softmax(torch.matmul(q_group, k_sel.transpose(0, 1)) * scale, dim=-1)
            batch_outputs.append(torch.matmul(probs, v_sel))
        outputs.append(torch.cat(batch_outputs, dim=0))
    return torch.stack(outputs, dim=0)


def torch_per_head_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    outputs = []
    scale = 1.0 / math.sqrt(q.shape[-1])
    for batch_idx in range(q.shape[0]):
        head_outputs = []
        for head_idx in range(q.shape[1]):
            kv_idx = head_idx // group_size
            idx = indices[batch_idx, head_idx].to(torch.long)
            k_sel = k[batch_idx, kv_idx].index_select(0, idx)
            v_sel = v[batch_idx, kv_idx].index_select(0, idx)
            probs = torch.softmax(torch.matmul(q[batch_idx, head_idx], k_sel.transpose(0, 1)) * scale, dim=-1)
            head_outputs.append(torch.matmul(probs, v_sel))
        outputs.append(torch.stack(head_outputs, dim=0))
    return torch.stack(outputs, dim=0)


def run_scenario(
    module,
    scenario: Scenario,
    batch_size: int,
    device: torch.device,
    warmups: int,
    repeats: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    check: bool,
) -> dict:
    group_size = num_heads // num_kv_heads
    torch.manual_seed(0)
    q = torch.randn(batch_size, num_heads, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch_size, num_kv_heads, scenario.length, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch_size, num_kv_heads, scenario.length, head_dim, device=device, dtype=torch.float16)
    if scenario.kernel == "per_head":
        indices = make_head_indices(batch_size, num_heads, scenario.selected, scenario.length, device)
        fused = lambda: module.batched_per_head_sparse_decode(q, k, v, indices)
        ref = lambda: torch_per_head_decode(q, k, v, indices, group_size)
    else:
        indices = make_group_indices(batch_size, num_kv_heads, scenario.selected, scenario.length, device)
        fused = lambda: module.batched_gqa_group_shared_decode(q, k, v, indices)
        ref = lambda: torch_group_decode(q, k, v, indices, group_size)

    fused_out = fused()
    if check:
        torch.testing.assert_close(fused_out, ref(), rtol=2e-2, atol=2e-2)
    if not torch.isfinite(fused_out).all():
        raise RuntimeError(f"{scenario.method} batch={batch_size} produced NaN or Inf")

    fused_us = benchmark(fused, device, warmups, repeats)
    return {
        "method": scenario.method,
        "selector": scenario.selector,
        "variant": scenario.variant,
        "kernel": scenario.kernel,
        "length": scenario.length,
        "batch_size": batch_size,
        "selected": scenario.selected,
        "selected_source_value": scenario.selected_source_value,
        "loaded_kv_mean": scenario.loaded_kv_mean,
        "latency_us": fused_us,
        "tokens_per_second": batch_size * 1_000_000.0 / fused_us,
        "source": str(scenario.source.relative_to(PROJECT_ROOT)),
    }


def add_speedups(rows: List[dict]) -> None:
    selector_baseline: Dict[tuple, float] = {}
    twi_baseline: Dict[tuple, float] = {}
    for row in rows:
        key = (row["selector"], row["length"], row["batch_size"])
        if row["variant"] == "selector_only":
            selector_baseline[key] = row["latency_us"]
        if row["variant"] == "twi":
            twi_baseline[key] = row["latency_us"]
    for row in rows:
        key = (row["selector"], row["length"], row["batch_size"])
        selector_base = selector_baseline.get(key)
        twi_base = twi_baseline.get(key)
        row["speedup_vs_selector"] = (selector_base / row["latency_us"]) if selector_base else 1.0
        row["speedup_vs_own_twi"] = (twi_base / row["latency_us"]) if twi_base else 1.0


def render_markdown(rows: Iterable[dict], warmups: int, repeats: int) -> str:
    lines = [
        "# Batched Fused GQA Decode Efficiency Benchmark",
        "",
        "This benchmark uses random tensors and fused CUDA decode kernels to compare per-head Twi sparse decode with Method B group-shared decode.",
        "",
        "Settings:",
        "",
        f"- dtype: `torch.float16`",
        f"- warmups: `{warmups}`",
        f"- repeats: `{repeats}`",
        "- model shape: `num_heads=32`, `num_kv_heads=8`, `group_size=4`, `head_dim=128`",
        "- lengths: LLaMA profile-derived 4K / 8K selected sizes",
        "",
        "| method | length | batch | kernel | selected | loaded KV mean | latency us | tok/s | speedup vs selector | speedup vs Twi |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {length} | {batch_size} | {kernel} | {selected} | {loaded_kv_mean:.2f} | "
            "{latency_us:.2f} | {tokens_per_second:.2f} | {speedup_vs_selector:.2f}x | "
            "{speedup_vs_own_twi:.2f}x |".format(**row)
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
            "- `quest_twi` and `ds_twi` use the fused per-head sparse decode kernel.",
            "- `*_groupmax_ex16` means selector + Twi estimator/top-p + `group_max p=0.80` + `exception_budget=16`; it uses the fused GQA group-shared decode kernel.",
            "- The 4K Quest group-max row currently reuses the available `group_max p=0.80` profile source because no separate 4K Quest exception16 profile summary is present.",
            "- The selected counts are ceil-rounded from profile summary mean values.",
            "- This is a simulated kernel benchmark, not end-to-end model serving latency.",
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
        "latency_us",
        "tokens_per_second",
        "speedup_vs_selector",
        "speedup_vs_own_twi",
        "source",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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
    parser.add_argument("--check-only", action="store_true")
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
    scenarios = [
        scenario_from_summary(method, length, args.num_heads, args.num_kv_heads)
        for length in args.lengths
        for method in args.methods
    ]
    rows = [
        run_scenario(
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
        for scenario in scenarios
        for batch_size in args.batch_sizes
    ]
    add_speedups(rows)
    markdown = render_markdown(rows, args.warmups, args.repeats)
    print(markdown)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    if args.csv_output:
        write_csv(rows, args.csv_output)
    if args.check_only:
        print("check-only completed")


if __name__ == "__main__":
    main()
