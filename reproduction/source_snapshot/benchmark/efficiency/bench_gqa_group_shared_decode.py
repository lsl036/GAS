import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List

import torch


@dataclass(frozen=True)
class Scenario:
    name: str
    seq_len: int
    loaded_original: int
    loaded_group: int
    loaded_exception: int


DEFAULT_SCENARIOS = (
    Scenario("llama_p080_8k", 8192, 8500, 1766, 2080),
    Scenario("llama_p080_16k", 16384, 9076, 1806, 2077),
    Scenario("qwen3_p080_4k", 4096, 5387, 778, 1058),
    Scenario("qwen3_p080_8k", 8192, 10363, 1621, 1965),
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


def make_indices(count: int, seq_len: int, device: torch.device, groups: int) -> torch.Tensor:
    count = min(max(count, 1), seq_len)
    base = torch.linspace(0, seq_len - 1, count, device=device, dtype=torch.float32).round().to(torch.long)
    return base.unsqueeze(0).expand(groups, -1).contiguous()


def per_head_sparse_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    head_indices: torch.Tensor,
    num_kv_heads: int,
    group_size: int,
) -> torch.Tensor:
    # q: [num_heads, head_dim], k/v: [num_kv_heads, seq_len, head_dim]
    outputs = []
    scale = 1.0 / math.sqrt(q.shape[-1])
    for head_idx in range(q.shape[0]):
        kv_idx = head_idx // group_size
        indices = head_indices[head_idx]
        k_sel = k[kv_idx].index_select(0, indices)
        v_sel = v[kv_idx].index_select(0, indices)
        logits = torch.matmul(q[head_idx], k_sel.transpose(0, 1)) * scale
        probs = torch.softmax(logits, dim=-1)
        outputs.append(torch.matmul(probs, v_sel))
    return torch.stack(outputs, dim=0)


def group_shared_sparse_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    group_indices: torch.Tensor,
    num_kv_heads: int,
    group_size: int,
) -> torch.Tensor:
    outputs = []
    scale = 1.0 / math.sqrt(q.shape[-1])
    for kv_idx in range(num_kv_heads):
        indices = group_indices[kv_idx]
        k_sel = k[kv_idx].index_select(0, indices)
        v_sel = v[kv_idx].index_select(0, indices)
        q_group = q[kv_idx * group_size : (kv_idx + 1) * group_size]
        logits = torch.matmul(q_group, k_sel.transpose(0, 1)) * scale
        probs = torch.softmax(logits, dim=-1)
        outputs.append(torch.matmul(probs, v_sel))
    return torch.cat(outputs, dim=0)


def dense_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_kv_heads: int,
    group_size: int,
) -> torch.Tensor:
    outputs = []
    scale = 1.0 / math.sqrt(q.shape[-1])
    for kv_idx in range(num_kv_heads):
        q_group = q[kv_idx * group_size : (kv_idx + 1) * group_size]
        logits = torch.matmul(q_group, k[kv_idx].transpose(0, 1)) * scale
        probs = torch.softmax(logits, dim=-1)
        outputs.append(torch.matmul(probs, v[kv_idx]))
    return torch.cat(outputs, dim=0)


def bytes_read_kv(loaded_kv: int, head_dim: int, element_size: int) -> int:
    return loaded_kv * head_dim * 2 * element_size


def format_mb(num_bytes: int) -> str:
    return f"{num_bytes / 1024 / 1024:.2f}"


def run_scenario(
    scenario: Scenario,
    device: torch.device,
    dtype: torch.dtype,
    warmups: int,
    repeats: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> dict:
    group_size = num_heads // num_kv_heads
    torch.manual_seed(0)
    q = torch.randn(num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(num_kv_heads, scenario.seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(num_kv_heads, scenario.seq_len, head_dim, device=device, dtype=dtype)

    per_head_count = math.ceil(scenario.loaded_original / num_heads)
    group_count = math.ceil(scenario.loaded_group / num_kv_heads)
    exception_count = math.ceil(scenario.loaded_exception / num_kv_heads)

    head_indices = make_indices(per_head_count, scenario.seq_len, device, num_heads)
    group_indices = make_indices(group_count, scenario.seq_len, device, num_kv_heads)
    exception_indices = make_indices(exception_count, scenario.seq_len, device, num_kv_heads)

    dense_out = dense_decode(q, k, v, num_kv_heads, group_size)
    per_head_out = per_head_sparse_decode(q, k, v, head_indices, num_kv_heads, group_size)
    group_out = group_shared_sparse_decode(q, k, v, group_indices, num_kv_heads, group_size)
    exception_out = group_shared_sparse_decode(q, k, v, exception_indices, num_kv_heads, group_size)
    for name, out in (
        ("dense", dense_out),
        ("per_head", per_head_out),
        ("group", group_out),
        ("exception", exception_out),
    ):
        if out.shape != (num_heads, head_dim):
            raise RuntimeError(f"{name} output has invalid shape: {tuple(out.shape)}")
        if not torch.isfinite(out).all():
            raise RuntimeError(f"{name} output contains NaN or Inf")

    dense_us = benchmark(lambda: dense_decode(q, k, v, num_kv_heads, group_size), device, warmups, repeats)
    per_head_us = benchmark(
        lambda: per_head_sparse_decode(q, k, v, head_indices, num_kv_heads, group_size),
        device,
        warmups,
        repeats,
    )
    group_us = benchmark(
        lambda: group_shared_sparse_decode(q, k, v, group_indices, num_kv_heads, group_size),
        device,
        warmups,
        repeats,
    )
    exception_us = benchmark(
        lambda: group_shared_sparse_decode(q, k, v, exception_indices, num_kv_heads, group_size),
        device,
        warmups,
        repeats,
    )

    element_size = torch.empty((), dtype=dtype).element_size()
    return {
        "scenario": scenario.name,
        "seq_len": scenario.seq_len,
        "per_head_count": per_head_count,
        "group_count": group_count,
        "exception_count": exception_count,
        "loaded_original": scenario.loaded_original,
        "loaded_group": scenario.loaded_group,
        "loaded_exception": scenario.loaded_exception,
        "dense_us": dense_us,
        "per_head_us": per_head_us,
        "group_us": group_us,
        "exception_us": exception_us,
        "group_speedup": per_head_us / group_us,
        "exception_speedup": per_head_us / exception_us,
        "original_kv_mb": format_mb(bytes_read_kv(scenario.loaded_original, head_dim, element_size)),
        "group_kv_mb": format_mb(bytes_read_kv(scenario.loaded_group, head_dim, element_size)),
        "exception_kv_mb": format_mb(bytes_read_kv(scenario.loaded_exception, head_dim, element_size)),
    }


def render_markdown(rows: Iterable[dict], dtype: torch.dtype, warmups: int, repeats: int) -> str:
    lines: List[str] = [
        "# GQA Group-Shared Decode Microbenchmark",
        "",
        "This benchmark uses random tensors and PyTorch gather/matmul/softmax kernels.",
        "It is intended as a microbenchmark for memory-load trends, not as a production fused kernel result.",
        "",
        "Settings:",
        "",
        f"- dtype: `{dtype}`",
        f"- warmups: `{warmups}`",
        f"- repeats: `{repeats}`",
        "- batch size: `1`",
        "- num heads / KV heads / group size: `32 / 8 / 4`",
        "- head dim: `128`",
        "",
        "| scenario | seq len | original loaded KV | group loaded KV | exception loaded KV | original KV MB | group KV MB | exception KV MB | per-head us | group us | exception us | group speedup | exception speedup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {scenario} | {seq_len} | {loaded_original} | {loaded_group} | {loaded_exception} | "
            "{original_kv_mb} | {group_kv_mb} | {exception_kv_mb} | "
            "{per_head_us:.2f} | {group_us:.2f} | {exception_us:.2f} | "
            "{group_speedup:.2f}x | {exception_speedup:.2f}x |".format(**row)
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- `per-head us` simulates the original per-query-head sparse decode path.",
            "- `group us` simulates Method B group-shared KV reuse.",
            "- `exception us` simulates the robust variant with extra per-head tail tokens approximated as a larger group-level index set.",
            "- A fused CUDA/Triton kernel is still required for final end-to-end latency claims.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    dtype = getattr(torch, args.dtype)
    if args.num_heads % args.num_kv_heads != 0:
        raise ValueError("--num-heads must be divisible by --num-kv-heads")

    rows = [
        run_scenario(
            scenario,
            device,
            dtype,
            args.warmups,
            args.repeats,
            args.num_heads,
            args.num_kv_heads,
            args.head_dim,
        )
        for scenario in DEFAULT_SCENARIOS
    ]
    markdown = render_markdown(rows, dtype, args.warmups, args.repeats)
    print(markdown)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
