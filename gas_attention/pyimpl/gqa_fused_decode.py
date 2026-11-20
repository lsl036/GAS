"""Lazy loader for Method B fused GQA decode kernels."""

from pathlib import Path

from torch.utils.cpp_extension import load


_module = None


def get_gqa_fused_decode_module():
    global _module
    if _module is None:
        root = Path(__file__).resolve().parents[2]
        _module = load(
            name="twilight_gqa_group_decode",
            sources=[str(root / "csrc" / "src" / "gqa_group_decode.cu")],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
            ],
            verbose=False,
        )
    return _module
