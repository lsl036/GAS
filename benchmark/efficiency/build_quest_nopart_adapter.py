#!/usr/bin/env python3
"""Build an isolated Quest non-partition page correction."""
import hashlib
import json
from pathlib import Path

from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "benchmark/efficiency/vendor/Quest"
OVERLAY = ROOT / "benchmark/efficiency/quest_nopart_overlay"
WRAPPER = ROOT / "benchmark/results_efficiency/e6_stream_adapters_v2/quest_stream_approx_attn.cu"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sources = [ROOT / "benchmark/efficiency/e6_official_decode_binding.cu", WRAPPER]
    flags = ["-O3", "--expt-extended-lambda", "--expt-relaxed-constexpr", "-DBSK_TORCH_CHECK",
             "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
             "-U__CUDA_NO_BFLOAT16_CONVERSIONS__"]
    module = load(name="e6_quest_nopart_fixed", sources=[str(p) for p in sources],
                  extra_include_paths=[str(OVERLAY), str(VENDOR / "quest/ops/csrc"),
                                       str(ROOT / "benchmark/efficiency"), str(VENDOR / "kernels/include"),
                                       str(VENDOR / "kernels/3rdparty/flashinfer/include")],
                  extra_cuda_cflags=flags, verbose=True)
    files = sources + [OVERLAY / "decode/decode_page.cuh", OVERLAY / "decode/decode_attn.cuh"]
    info = {"module": str(Path(module.__file__).resolve()), "module_sha256": sha(Path(module.__file__)),
            "sources": {str(p): sha(p) for p in files},
            "change": "current-stream wrapper plus isolated non-partition history/last-page correction; not unmodified official kernel"}
    (OVERLAY / "build.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()
