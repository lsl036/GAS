import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List

import torch
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Scenario:
    name: str
    seq_len: int
    selected: int


DEFAULT_SCENARIOS = (
    Scenario("llama_p080_8k_group", 8192, 221),
    Scenario("llama_p080_8k_exception16", 8192, 260),
    Scenario("llama_p080_16k_group", 16384, 226),
    Scenario("llama_p080_16k_exception16", 16384, 260),
    Scenario("qwen3_p080_4k_group", 4096, 98),
    Scenario("qwen3_p080_4k_exception16", 4096, 133),
    Scenario("qwen3_p080_8k_group", 8192, 203),
    Scenario("qwen3_p080_8k_exception16", 8192, 246),
)


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


def make_indices(count: int, seq_len: int, num_kv_heads: int, device: torch.device) -> torch.Tensor:
    base = torch.linspace(0, seq_len - 1, count, device=device, dtype=torch.float32).round().to(torch.int32)
    return base.unsqueeze(0).expand(num_kv_heads, -1).contiguous()


def torch_group_shared_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    outputs = []
    scale = 1.0 / math.sqrt(q.shape[-1])
    for kv_idx in range(k.shape[0]):
        idx = indices[kv_idx].to(torch.long)
        k_sel = k[kv_idx].index_select(0, idx)
        v_sel = v[kv_idx].index_select(0, idx)
        q_group = q[kv_idx * group_size : (kv_idx + 1) * group_size]
        probs = torch.softmax(torch.matmul(q_group, k_sel.transpose(0, 1)) * scale, dim=-1)
        outputs.append(torch.matmul(probs, v_sel))
    return torch.cat(outputs, dim=0)


def run_one(
    module,
    scenario: Scenario,
    device: torch.device,
    warmups: int,
    repeats: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> dict:
    group_size = num_heads // num_kv_heads
    torch.manual_seed(0)
    q = torch.randn(num_heads, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(num_kv_heads, scenario.seq_len, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(num_kv_heads, scenario.seq_len, head_dim, device=device, dtype=torch.float16)
    indices = make_indices(scenario.selected, scenario.seq_len, num_kv_heads, device)

    fused_out = module.gqa_group_shared_decode(q, k, v, indices)
    ref_out = torch_group_shared_decode(q, k, v, indices, group_size)
    torch.testing.assert_close(fused_out, ref_out, rtol=2e-2, atol=2e-2)

    fused_us = benchmark(lambda: module.gqa_group_shared_decode(q, k, v, indices), device, warmups, repeats)
    torch_us = benchmark(lambda: torch_group_shared_decode(q, k, v, indices, group_size), device, warmups, repeats)
    return {
        "scenario": scenario.name,
        "seq_len": scenario.seq_len,
        "selected": scenario.selected,
        "torch_us": torch_us,
        "fused_us": fused_us,
        "speedup": torch_us / fused_us,
    }


def render_markdown(rows: Iterable[dict], warmups: int, repeats: int) -> str:
    lines: List[str] = [
        "# Fused GQA Group-Shared Decode Microbenchmark",
        "",
        "This benchmark validates a prototype fused CUDA kernel for Method B decode attention.",
        "The kernel fuses shared KV load, QK, softmax, and PV for one GQA KV group per CUDA block.",
        "",
        "Settings:",
        "",
        f"- dtype: `torch.float16`",
        f"- warmups: `{warmups}`",
        f"- repeats: `{repeats}`",
        "- batch size: `1`",
        "- num heads / KV heads / group size: `32 / 8 / 4`",
        "- head dim: `128`",
        "",
        "| scenario | seq len | selected / KV group | PyTorch group us | fused us | speedup |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {scenario} | {seq_len} | {selected} | {torch_us:.2f} | {fused_us:.2f} | {speedup:.2f}x |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- This is a decode-only prototype for `q_len = 1` and contiguous `[num_kv_heads, seq_len, head_dim]` KV tensors.",
            "- It does not yet handle batch > 1, paged KV cache, prefill, RoPE, or direct integration into `attention.py`.",
            "- The result is suitable as a kernel-level feasibility point, not final production latency.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--selected", type=int)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.num_heads % args.num_kv_heads != 0:
        raise ValueError("--num-heads must be divisible by --num-kv-heads")

    module = get_module()
    if (args.seq_len is None) != (args.selected is None):
        raise ValueError("--seq-len and --selected must be provided together")
    scenarios = (
        (Scenario("custom", args.seq_len, args.selected),)
        if args.seq_len is not None
        else DEFAULT_SCENARIOS
    )
    rows = [
        run_one(
            module,
            scenario,
            device,
            args.warmups,
            args.repeats,
            args.num_heads,
            args.num_kv_heads,
            args.head_dim,
        )
        for scenario in scenarios
    ]
    markdown = render_markdown(rows, args.warmups, args.repeats)
    print(markdown)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
