"""
Sparse GEMV Kernels of Flash-TopK Attention.

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

import torch

from .module import get_kernel_module


def single_sparse_gemv(
    q: torch.Tensor,
    o: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    """
    Perform single sparse GEMV operation.

    Args:
        q: [num_qo_heads, head_dim]
        o: [num_qo_heads, seq_len]
        k: [seq_len, num_kv_heads, head_dim] (NHD)
    """
    get_kernel_module().single_sparse_gemv(q, o, k)


        
def batched_sparse_gemv(
    q: torch.Tensor,
    o: torch.Tensor,
    k_cache: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
) -> torch.Tensor:
    """
    Perform batched sparse GEMV operation.

    Calculate q @ k_cache.T, indexed by kv_indices, kv_indptr and kv_last_page_len.
    
    Args:
        q: [batch_size, num_heads, head_dim]
        o: [batch_size, num_heads, 1, seq_len]
        k_cache: [NKVH]
        kv_indices: [NKV]
        kv_indptr: [NKV+1]
        kv_last_page_len: [NKV]
        budget: int
    
    Returns:
        out: [NHD1B]
    """

    get_kernel_module().batched_sparse_gemv(
        q,
        o,
        k_cache,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
    )
        

def quest_sparse_gemv(
    q: torch.Tensor,
    o: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
):
    get_kernel_module().quest_sparse_gemv(
        q,
        o,
        k_cache,
        v_cache,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
    )

        
def batched_sparse_gemv_int8_k(
    q: torch.Tensor,
    o: torch.Tensor,
    k_cache: torch.Tensor,
    quant_scales: torch.Tensor,
    quant_zeros: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
):
    get_kernel_module().batched_sparse_gemv_int8_k(
        q,
        o,
        k_cache,
        quant_scales,
        quant_zeros,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
    )


def batched_sparse_gemv_int4_k(
    q: torch.Tensor,
    o: torch.Tensor,
    k_cache: torch.Tensor,
    quant_scales: torch.Tensor,
    quant_zeros: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
):
    get_kernel_module().batched_sparse_gemv_int4_k(
        q,
        o,
        k_cache,
        quant_scales,
        quant_zeros,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
    )


def sparse_batched_gemv_int4_k(
    q: torch.Tensor,
    o: torch.Tensor,
    k_cache: torch.Tensor,
    quant_scales: torch.Tensor,
    quant_zeros: torch.Tensor,
    selected_indices: torch.Tensor,
    selected_valid: torch.Tensor,
):
    get_kernel_module().sparse_batched_gemv_int4_k(
        q,
        o,
        k_cache,
        quant_scales,
        quant_zeros,
        selected_indices,
        selected_valid,
    )
