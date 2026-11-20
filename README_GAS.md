# GAS 工作区与论文复现

本工作区由原 `Twilight_Ours` 重命名；旧路径仍为兼容符号链接。当前开发代码使用 Python 包名 `gas_attention`，安装分发包名为 `gas-attention-research`；它基于 Twilight 扩展，历史复现快照仍保留原 `twilight` 包名。

- 论文图表重绘、源码与数据快照、运行入口：[reproduction/README.md](reproduction/README.md)
- 算法实现、benchmark 和重跑流程：[reproduction/PIPELINES.md](reproduction/PIPELINES.md)
- 图表溯源与当前已知口径问题：[reproduction/PROVENANCE.md](reproduction/PROVENANCE.md)
- 原始模型质量及效率实验仍在 `benchmark/`；原始图表输入在 `Figures/`。

可重绘现有图并不等于重新验证图中所有实验。请优先阅读溯源说明，尤其注意质量点不一致、kernel 计时口径、TPOT 投影和 breakdown 数据缩放。

## Python 包

当前开发代码使用 `from gas_attention.pyimpl import enable_sparse_attention`。若已在合适的环境中安装本项目所需的 PyTorch、CUDA 和其他依赖，可以从仓库根目录安装开发版：

```bash
python -m pip install --no-deps --no-build-isolation -e .
```

这条命令不会自动安装依赖。仓库级 `csrc/` 和 `flash-topk-attention/` 仍是运行时所需资源；当前普通 wheel 不包含完整 CUDA 源码与第三方资源，不应当作为可独立运行的完整分发包。重命名前安装的 `twilight` 分发包或历史快照不由本次修改卸载；它们与新开发包是不同命名空间，使用时请确认 `gas_attention.__file__` 指向预期仓库。旧代码中的 `from twilight...` 需要迁移，新包不提供旧名称的兼容别名。
