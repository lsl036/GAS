"""
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

from types import SimpleNamespace
from typing import Tuple, Union, Optional

import torch

from flashinfer.utils import register_custom_op

try:
    from flashinfer.utils import get_cuda_stream
except ImportError:
    def get_cuda_stream(device):
        return torch.cuda.current_stream(device).cuda_stream

from .env import TWILIGHT_CSRC_DIR, TWILIGHT_INCLUDE_DIR

try:
    from flashinfer.jit import (
        has_prebuilt_ops,
        load_cuda_ops,
        FLASHINFER_INCLUDE_DIR,
        FLASHINFER_CSRC_DIR,
    )
except ImportError:
    has_prebuilt_ops = False
    load_cuda_ops = None
    from flashinfer.jit import env as flashinfer_jit_env

    FLASHINFER_INCLUDE_DIR = flashinfer_jit_env.FLASHINFER_INCLUDE_DIR
    FLASHINFER_CSRC_DIR = flashinfer_jit_env.FLASHINFER_CSRC_DIR


_sampling_module = None


def get_sampling_module():

    global _sampling_module
    if _sampling_module is None:
        if has_prebuilt_ops:
            from . import _kernels

            module = _kernels
        elif load_cuda_ops is not None:
            module = load_cuda_ops(
                "twilight_sampling",
                sources=[
                    TWILIGHT_CSRC_DIR / "sampling.cu",
                ],
                extra_include_paths=[
                    FLASHINFER_INCLUDE_DIR,
                    FLASHINFER_CSRC_DIR,
                    TWILIGHT_INCLUDE_DIR,
                ],
            )
        else:
            from torch.utils.cpp_extension import load

            module = load(
                name="twilight_sampling",
                sources=[str(TWILIGHT_CSRC_DIR / "sampling.cu")],
                extra_include_paths=[
                    str(FLASHINFER_INCLUDE_DIR),
                    str(FLASHINFER_CSRC_DIR),
                    str(TWILIGHT_INCLUDE_DIR),
                ],
                extra_cuda_cflags=[
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                ],
            )

        # Register ops
        @register_custom_op("flashinfer::top_p_fp16_return_mask", mutates_args=())
        def top_p_fp16_return_mask(
            probs: torch.Tensor,
            maybe_top_p_arr: Optional[torch.Tensor],
            top_p_val: float,
        ) -> torch.Tensor:
            with probs.device as device:
                maybe_top_p_arr = (
                    maybe_top_p_arr if maybe_top_p_arr is not None else None
                )
                mask = torch.zeros_like(probs, dtype=torch.bool)
                module.top_p_fp16_return_mask(
                    probs,
                    mask,
                    maybe_top_p_arr,
                    top_p_val,
                    get_cuda_stream(device),
                )
                return mask

        @register_custom_op("flashinfer::top_p_fp32_return_mask", mutates_args=())
        def top_p_fp32_return_mask(
            probs: torch.Tensor,
            maybe_top_p_arr: Optional[torch.Tensor],
            top_p_val: float,
        ) -> torch.Tensor:
            with probs.device as device:
                maybe_top_p_arr = (
                    maybe_top_p_arr if maybe_top_p_arr is not None else None
                )
                mask = torch.zeros_like(probs, dtype=torch.bool)
                module.top_p_fp32_return_mask(
                    probs,
                    mask,
                    maybe_top_p_arr,
                    top_p_val,
                    get_cuda_stream(device),
                )
                return mask

        # Register the module
        _sampling_module = SimpleNamespace(
            top_p_fp16_return_mask=top_p_fp16_return_mask,
            top_p_fp32_return_mask=top_p_fp32_return_mask,
        )

    return _sampling_module


def _to_tensor_scalar_tuple(x):
    if isinstance(x, torch.Tensor):
        return (x, 0)
    else:
        return (None, x)


def top_p_fp16_return_mask(
    probs: torch.Tensor,
    top_p: Union[torch.Tensor, float],
) -> torch.Tensor:
    return get_sampling_module().top_p_fp16_return_mask(
        probs, *_to_tensor_scalar_tuple(top_p)
    )


def top_p_fp32_return_mask(
    probs: torch.Tensor,
    top_p: Union[torch.Tensor, float],
) -> torch.Tensor:
    return get_sampling_module().top_p_fp32_return_mask(
        probs, *_to_tensor_scalar_tuple(top_p)
    )
