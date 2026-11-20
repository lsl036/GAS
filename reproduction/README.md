# GAS 论文复现目录

本目录将论文当前图表输入、对应实验入口和历史结果索引集中保存。`../` 是 GAS 源码仓库；原来的 `/data/home/lsl/SparseAtt/Twilight_Ours` 保留为指向 `GAS` 的兼容链接。原始实验结果和原始 `Figures/` 均未覆盖。

> **先读 [PROVENANCE.md](PROVENANCE.md)。** 原样重绘仅说明可从当前 notebook 和数据重新生成图，不代表图中的数值已通过实验溯源。现有 Fig.4、Fig.6、Fig.7、Fig.8 输入有需要核查的问题，不能不加说明地视为统一口径的实测数据。

## 目录

```text
reproduction/
  figure_inputs/       原始绘图 notebook、输入数据及部分 PPTX/PDF 快照，manifest.json 含 SHA-256
  reference_results/   用于交叉审计的小体积 benchmark 汇总快照，manifest.json 含 SHA-256
  source_snapshot/     算法、CUDA、配置及 benchmark 源码快照，manifest.json 含 SHA-256
  source_snapshot.py   创建和验证源码快照，拒绝覆盖
  figures.py           原样重绘、快照完整性校验
  audit.py             对比 Fig.4/6/7/8 绘图输入与已保存实验汇总
  run_benchmark.py     Passkey、RULER Fig.4(a)、Qasper attention 的安全重跑入口
  PIPELINES.md         模型、数据、算法和 benchmark 说明及其他实验入口
  PROVENANCE.md        图表可追溯性、限制与已知差异
  ENVIRONMENT.md       本次实测环境、缓存和 CUDA 编译限制
  outputs/             本次生成的图及审计报告，后续重绘建议使用新目录
  runs/                本次 smoke test 与后续实验输出，非训练数据
```

`figure_inputs/` 是独立绘图输入快照；`reference_results/` 是小型结果汇总；`source_snapshot/` 保留迁移前的算法与 benchmark 源代码和配置，使用原 `twilight` 包名，不随当前开发代码重命名而改变。它们不包含模型权重、完整数据集、逐样本日志、大型第三方代码及历史实验工作区。要从模型重新执行实验，仍需文中指定的外部模型、数据、依赖和 CUDA 环境。历史完整结果仍在 `../benchmark/`。

## 1. 验证输入和原样绘图

需要 Python 3.10+，以及 `numpy`、`matplotlib`、`scipy`、`openpyxl`、`IPython`。本机 `moba` 环境的顶层版本记录在 `requirements-figures-verified.txt`；字体缺少 Times New Roman 时 matplotlib 会回退，布局与原 PDF 可能不同。此次 Fig.4 重绘能生成文件，但部分标注有重叠，不能不经检查直接用于排版。不要依赖 PDF 字节 hash 判断画面相同。

建议在独立虚拟环境安装 `requirements-figures-verified.txt`，不要在现有推理环境中升级 PyTorch。对于本机已验证的临时依赖，可用：

```bash
cd /data/home/lsl/SparseAtt/GAS
export PYTHONPATH=/data/tmp/gas_repro_plot_deps
export MPLBACKEND=Agg
PY=/data/home/lsl/Software/miniconda3/envs/moba/bin/python
"$PY" reproduction/figures.py verify
"$PY" reproduction/source_snapshot.py verify
"$PY" reproduction/figures.py render --figures all --output reproduction/outputs/my_render
"$PY" reproduction/audit.py --output reproduction/outputs/my_figure_audit.md
```

`/data/tmp/gas_repro_plot_deps` 只用于这次测试，可能被系统清理；其他机器应安装依赖而不是依赖该路径。输出目录必须不存在或为空，以防覆盖已有成果。只重绘单图时可将 `all` 替换为 `fig1`、`fig2`、`fig4`、`fig5`、`fig6`、`fig7`、`fig8` 或 `fig8_backup`。

Fig.3 以及 Fig.1/2 的组合图源是 PPTX，保留于 `figure_inputs/`，需要 PowerPoint/LibreOffice 编辑导出，不由 notebook 重建。Tab.2 是 Markdown 数据快照，未在现有目录发现生成该表的 notebook。输出的 PDF/PNG 是原始 notebook 的隔离重绘，不会自动修正 Fig.4/6/7/8 的数据问题。

## 2. 数据审计

`audit.py` 默认读取哈希固定的 `reference_results/`；`--live` 可对照当前 `../benchmark/` 的汇总。已运行的报告在 [figure_data_audit_frozen_20261119.md](outputs/figure_data_audit_frozen_20261119.md)。审计是局部交叉检查，不代表所有曲线点和 score 均已验证。

## 3. 模型/benchmark 重跑

完整协议及风险见 [PIPELINES.md](PIPELINES.md)。统一入口默认拒绝复用非空输出目录；先用 `--dry-run` 查看配置和命令。例如：

```bash
cd /data/home/lsl/SparseAtt/GAS
PY=/data/home/lsl/Software/miniconda3/envs/moba/bin/python
"$PY" reproduction/run_benchmark.py --dry-run passkey \
  --model /data2/lsl_data/models/Meta-Llama-3.1-8B-Instruct \
  --family llama31 --selector quest --fixed-length 8000 --iterations 5 \
  --output reproduction/runs/my_llama_quest_8k
```

确认命令、空闲 GPU、模型和数据后移除 `--dry-run` 执行。Passkey 输入当前由脚本随机生成，本入口没有补造固定随机种子，重跑不是逐样本 bitwise 重现。`moba` 当前 SciPy 位于 user-site；禁用 user-site 会导致 transformers 导入失败。运行当前开发代码时应检查依赖和 `gas_attention.__file__`；历史源码快照仍使用 `twilight`。不要盲目更改现有环境。

## 4. 验证与限制

```bash
cd /data/home/lsl/SparseAtt/GAS
/data/home/lsl/Software/miniconda3/envs/moba/bin/python -m unittest discover -s reproduction/tests -v
```

本地已验证快照、八个 notebook 的绘图执行、离线审计及一组 8K Passkey smoke test；这不等于重新跑完所有模型、任务、GPU shape 或论文分数。逐图来源、数据缺失和不一致必须保持显式标注。
