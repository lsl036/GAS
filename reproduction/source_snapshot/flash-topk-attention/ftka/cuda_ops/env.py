"""
Environments of Flash-TopK Attention CUDA Kernels.

The compiled JIT ops are cached in `FLASHINFER_JIT_DIR/`.

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


import pathlib

FTKA_HOME = pathlib.Path(__file__).resolve().parents[2]
FTKA_CSRC_DIR = FTKA_HOME / "csrc" / "src"
FTKA_INCLUDE_DIR = FTKA_HOME / "csrc" / "include"


# def _get_workspace_dir_name() -> pathlib.Path:
#     try:
#         with warnings.catch_warnings():
#             # Ignore the warning for TORCH_CUDA_ARCH_LIST not set
#             warnings.filterwarnings(
#                 "ignore", r".*TORCH_CUDA_ARCH_LIST.*", module="torch"
#             )
#             flags = _get_cuda_arch_flags()
#         arch = "_".join(sorted(set(re.findall(r"compute_(\d+)", "".join(flags)))))
#     except Exception:
#         arch = "noarch"
#     # e.g.: $project_home/.cache/ftka/75_80_89_90/
#     return FTKA_HOME / ".cache" / "ftka" / arch


# use pathlib
# FTKA_WORKSPACE_DIR = _get_workspace_dir_name()
# FTKA_JIT_DIR = FTKA_WORKSPACE_DIR / "cached_ops"
# FTKA_GEN_SRC_DIR = FTKA_WORKSPACE_DIR / "generated"
