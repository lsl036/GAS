"""
Copyright (c) 2025 by Bytedance.

Modified from:

Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Tuple, Union, Optional

import importlib

import torch

from .env import FTKA_CSRC_DIR, FTKA_INCLUDE_DIR

_module = None


def _load_with_flashinfer_legacy_jit():
    jit = importlib.import_module("flashinfer.jit")
    if not hasattr(jit, "load_cuda_ops"):
        return None

    if getattr(jit, "has_prebuilt_ops", False):
        from . import _kernels

        return _kernels

    return jit.load_cuda_ops(
        "ftka_ops_ours",
        sources=[
            FTKA_CSRC_DIR / "ftka_ops.cu",
        ],
        extra_include_paths=[
            jit.FLASHINFER_INCLUDE_DIR,
            jit.FLASHINFER_CSRC_DIR,
            FTKA_INCLUDE_DIR,
        ],
    )


def _load_with_torch_extension():
    from torch.utils.cpp_extension import load

    jit_env = importlib.import_module("flashinfer.jit.env")
    compilation_context = importlib.import_module("flashinfer.compilation_context")

    extra_include_paths = [
        str(jit_env.FLASHINFER_INCLUDE_DIR),
        str(jit_env.FLASHINFER_CSRC_DIR),
        str(jit_env.CUTLASS_INCLUDE_DIRS[0]),
        str(jit_env.CUTLASS_INCLUDE_DIRS[1]),
        str(jit_env.CCCL_INCLUDE_DIRS[0]),
        str(jit_env.CCCL_INCLUDE_DIRS[1]),
        str(jit_env.CCCL_INCLUDE_DIRS[2]),
        str(FTKA_INCLUDE_DIR),
    ]

    extra_cuda_cflags = [
        "--expt-relaxed-constexpr",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-DFLASHINFER_HAS_UINT8_VEC_T",
        *compilation_context.CompilationContext().get_nvcc_flags_list(),
    ]

    return load(
        name="ftka_ops_ours",
        sources=[str(FTKA_CSRC_DIR / "ftka_ops.cu")],
        extra_include_paths=extra_include_paths,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=False,
    )


def get_kernel_module():
    global _module
    if _module is None:
        module = _load_with_torch_extension()
        _module = module

    return _module
