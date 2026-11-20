#!/usr/bin/env python3
"""Compile an author decode wrapper with only its launch stream adapted.

The vendor tree and underlying attention headers are unchanged. The generated
source and module are explicitly named stream adapters, not original builds.
"""
import argparse
import hashlib
import json
from pathlib import Path

from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "quest": ROOT / "benchmark/efficiency/vendor/Quest/quest/ops/csrc",
    "less": ROOT / "benchmark/RULER/fig4a_baselines/vendor/LessIsMore/tidal/ops/csrc",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(SOURCES), required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    args = parser.parse_args()
    source_dir = SOURCES[args.method]
    original = source_dir / "approx_attn.cu"
    code = original.read_text()
    marker = "/*stream=*/nullptr"
    if code.count(marker) != 1:
        raise ValueError("expected exactly one official default-stream launch")
    if code.count('#include "bsk_ops.h"') != 1:
        raise ValueError("expected official wrapper header")
    code = "#include <ATen/cuda/CUDAContext.h>\n" + code.replace(marker, "/*stream=*/at::cuda::getCurrentCUDAStream()")
    code = code.replace('#include "bsk_ops.h"', '#include "e6_official_decode_types.h"')
    args.generated_dir.mkdir(parents=True, exist_ok=True)
    generated = args.generated_dir / f"{args.method}_stream_approx_attn.cu"
    if generated.exists() and generated.read_text() != code:
        raise ValueError("existing generated source differs")
    generated.write_text(code)
    root = ROOT / ("benchmark/efficiency/vendor/Quest" if args.method == "quest" else
                   "benchmark/RULER/fig4a_baselines/vendor/LessIsMore")
    flags = ["-O3", "--expt-extended-lambda", "--expt-relaxed-constexpr", "-DBSK_TORCH_CHECK",
             "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
             "-U__CUDA_NO_BFLOAT16_CONVERSIONS__"]
    if args.method == "less":
        flags.append("-DE6_LESSISMORE")
    name = f"e6_{args.method}_stream_decode"
    module = load(name=name, sources=[str(ROOT / "benchmark/efficiency/e6_official_decode_binding.cu"), str(generated)],
                  extra_include_paths=[str(source_dir), str(ROOT / "benchmark/efficiency"), str(root / "kernels/include"),
                                       str(root / "kernels/3rdparty/flashinfer/include")],
                  extra_cuda_cflags=flags, verbose=True)
    info = {"method": args.method, "original": str(original), "original_sha256": sha(original),
            "generated": str(generated.resolve()), "generated_sha256": sha(generated),
            "module": str(Path(module.__file__).resolve()), "module_sha256": sha(Path(module.__file__)),
            "change": "replace wrapper header with matching standalone declaration, include CUDAContext and use current stream; attention headers unchanged"}
    (args.generated_dir / f"{args.method}_stream_build.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()
