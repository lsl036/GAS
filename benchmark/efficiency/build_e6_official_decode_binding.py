#!/usr/bin/env python3
"""Link an upstream approx_attn.cu object to a minimal standalone binding."""

import argparse
from pathlib import Path
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[2]
QUEST = ROOT / "benchmark/efficiency/vendor/Quest"
LESS = ROOT / "benchmark/RULER/fig4a_baselines/vendor/LessIsMore"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("quest", "less"), required=True)
    parser.add_argument("--object", type=Path, required=True, help="Official CMake-built approx_attn.cu.o")
    args = parser.parse_args()
    root = QUEST if args.method == "quest" else LESS
    if not args.object.is_file():
        raise FileNotFoundError(args.object)
    flags = ["-O3", "--expt-extended-lambda", "--expt-relaxed-constexpr",
             "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
             "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__"]
    if args.method == "less":
        flags.append("-DE6_LESSISMORE")
    name = f"e6_{args.method}_official_decode"
    module = load(name=name, sources=[str(ROOT / "benchmark/efficiency/e6_official_decode_binding.cu")],
                  extra_include_paths=[str(root / "kernels/include"), str(root / "kernels/3rdparty/flashinfer/include")],
                  extra_cuda_cflags=flags, extra_ldflags=[str(args.object.resolve())], verbose=True)
    print(f"OFFICIAL_DECODE_BINDING={module.__file__}", flush=True)


if __name__ == "__main__":
    main()
