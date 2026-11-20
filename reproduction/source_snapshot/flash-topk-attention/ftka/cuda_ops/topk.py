"""
TopK operators from Raft:h ttps://github.com/rapidsai/raft/

Copyright (c) 2025 by Bytedance.
"""

from typing import Tuple, Union, Optional

import torch

from .module import get_kernel_module


def raft_topk(
    input: torch.Tensor,
    input_idx: torch.Tensor,
    output: torch.Tensor,
    output_idx: torch.Tensor,
    buf: torch.Tensor,
    k: int,
):
    """
    Returns the k largest/smallest elements of the given input tensor along a given dimension.
    """

    get_kernel_module().raft_topk(
        input,
        input_idx,
        output,
        output_idx,
        buf,
        k,
    )
