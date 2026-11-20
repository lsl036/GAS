# 本机验证环境

记录于 2026-11-19，仅描述本次 smoke test，**不是完整依赖锁文件**，也不能保证在其他机器上编译所有 CUDA 扩展。

| 项目 | 当前值 |
| --- | --- |
| GPU | NVIDIA A100-PCIE-40GB |
| NVIDIA driver | 550.163.01；`nvidia-smi` 显示 CUDA 12.4（驱动支持版本） |
| Python | 3.10.20，`/data/home/lsl/Software/miniconda3/envs/moba/bin/python` |
| PyTorch | 2.5.1+cu121 |
| transformers | 4.57.6 |
| flash-attn | 2.6.3 |
| flashinfer-python | 0.6.11.post1 |
| datasets | 4.8.5 |
| triton | 3.1.0 |
| numpy / scipy | 2.2.6 / 1.15.3 |
| huggingface-hub | 0.36.2 |

SciPy 此时从 Python user-site 加载；禁用 user-site 的 smoke test 因缺少 SciPy 失败。本次 GPU smoke 将 `TORCH_EXTENSIONS_DIR` 指向已有缓存 `/data/tmp/torch_ext_ours_fig4b_8192`；当前 shell 的 `nvcc` 不在 PATH 中，因此未验证从空扩展缓存编译。运行其他 kernel 前应单独配置与 PyTorch、驱动兼容的 CUDA Toolkit、C++ 编译器、第三方源码和 JIT 环境。

绘图验证所用的 top-level 包见 `requirements-figures-verified.txt`；`openpyxl` 被隔离安装在临时目录，临时目录可能被清理。Matplotlib 缺少 Times New Roman，字体回退导致重绘效果可能不同；Fig.4 已观察到部分标签重叠。
