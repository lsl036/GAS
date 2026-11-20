# 图表数据来源与使用边界

本文描述 2026-11-19 采集的现有资料，不将原样重绘等同于实验复现。原始 `../Figures/` 不作修改；本目录 `figure_inputs/` 是保留 notebook、表格和演示文件的哈希快照，`reference_results/` 是部分小型实验结果快照。两者都不是原始推理数据的完整归档。完整历史数据保留在 `../benchmark/`；模型、数据集和部分第三方源码位于仓库之外。

## 逐图索引

| 图表 | 固定输入 | 可复现内容 | 当前限制 |
| --- | --- | --- | --- |
| Fig.1 | `fig1-PerpvsLoadedKV/` 的 workbook 和 notebook | 基于 workbook 的数值图 | 组合图是 PPTX；不能仅凭重绘确认 perplexity 实验及输入参数 |
| Fig.2 | `fig2-KV-Amplification/` notebook | notebook 内数据的数值图 | 部分数值写在 notebook；组合图是 PPTX，须另行核验原始实验 |
| Fig.3 | `fig3-GAS-Overview/` PPTX、PDF | 保存原图源及现成导出 | 无数值 notebook；不由 `figures.py` 自动重建 |
| Fig.4 | `fig4-KVvsScore/` notebook、Markdown | 当前图中的曲线和点 | 若干点与正式集汇总不一致，见下文；其他点尚未逐点验证 |
| Fig.5 | `fig5-Passkey-KV减少/` notebook、Markdown | 当前图中保留率/减少率 | 应区分相对 Twilight pre-union、相对 full 的不同分母；Passkey 样本数有限 |
| Fig.6 | `fig6-kernel speedup/SparseAttenKernel/` notebook及旁注 | 当前数组的图 | 某些 GAS-FI 点来自 head/group 不同路径；custom 为另一次测量，不能默认为同次配对 |
| Fig.7 | `fig7-TPOT result/NewEnv/` notebook及旁注 | 现有分量相加投影图 | 多个历史来源，见下文；不是完整端到端实测 TPOT |
| Fig.8 | `fig8-TimeBreakdown/` 主/备用 notebook 和 CSV | 原 notebook 的图和其重新导出的 CSV | 主 notebook 明确缩放部分数据；标签与说明不一致；备用数据也须溯源 |
| Tab.2 | `tab2-Accuracy/dataTest.md` | 保存当前制表输入 | 不自动证明各项任务、样本数及完整性一致 |

`python reproduction/figures.py verify` 检查快照是否被修改；`audit.py` 生成的 [交叉检查](outputs/figure_data_audit_frozen_20261119.md) 列出具体差异。检查只覆盖有限字段，不能证明其他值正确或不同实验可直接比较。PPTX、PDF 和 Markdown 是可供查阅的原件，并非可执行 benchmark。

## 已确认的来源差异

### Fig.4：正式 32K RULER

[正式汇总](reference_results/benchmark/RULER/fig4a_llama31/fig4a_summary.csv) 中 full score 是 60.58，而绘图 Markdown 是 64.58；DS calibrated loaded KV 为 1757.92，绘图表写 1458；Twilight calibrated 分别为 1767.00 和 1367。对应 score 在这些行可能一致，但不能把 score 归属于未验证的 KV 点。暂不自动修改用户图表；应先核对原预测 JSONL、配置和是否采用了别的结果版本。

GAS 的 99.5% 是 `1 - 1339.32465 / 259300.016 = 99.4835%`：两个方法各自先对单样本 decode 中第 2--31 层和 decode 记录求平均，再对每任务 50 个样本求平均，最后对 5 个 32K RULER 任务等权平均；不是逐样本减少率的均值。配置为 Quest 8192、Twilight `p_t=0.85`、GAS group-max `p_g=0.80` 和 per-head exception 16。指标是各 KV group 内 query-head mask 并集大小之和，属于逻辑 loaded-KV 计数，不证明 KV cache 显存或物理读取量同比减少。`niah_multikey_3`、`niah_multivalue` 存在重复 index 对应不同输入；进行配对时须检查输入内容，不能只用 index 去重。

### Fig.6：不同 attention 数据路径

快照中 GAS 的 FI 系列部分点匹配早期 head-level FI，另一些匹配 group-level FI，32K/B128 的值不匹配这两列。GAS-custom 是另外一次 paired replay 的结果；是否与基线按相同 trace、shape、计时方式重新采集须按每个来源单独核查。`reference_results/` 中的两个相关 CSV 和 [audit 报告](outputs/figure_data_audit_frozen_20261119.md) 可用来追查，但数值相同本身不等于证明完整来源。

Qasper 32K trace 通过重复文本构造；batch 通过交替使用两个请求形成，并非每个 batch 均有独立 KV pool。attention-only 排除 selection、estimator、top-p、refinement 和 FI plan/元数据构造；具体设置以每次实验输出 `config.json` 为准，不应称为端到端 latency。

### Fig.7：投影而非实测整步

当前 notebook 的 FA2、Quest、Twilight 和 Less-style 列可匹配早期 projection；FlashInfer 列匹配后来 tuned dense projection；GAS 列匹配后来的 custom projection。分量来自不同测量及后处理，不能当作同次配对或真实服务 TPOT。检查混合来源时，还应核实设备、精度、计时方法、层数、稀疏预算和是否包含 selection/refinement。请同时阅读冻结的 E7 `VALIDITY_NOTICE.md`。

### Fig.8：缩放和标签

主 notebook 包含 `BREAKDOWN_MS[method][:, [0, 1, 2, 4]] *= 0.9`；同时将 `common_ms` 标成 `Top-k Selector`，而相邻 README 将其解释为 transformer 公共计算。备用 notebook 的活动数据与 `REPORTED_TOTAL_US` 并非同一组。当前图可重绘，但在找到可信原始计时和一致标签以前，不应作为完整 measured pipeline breakdown。保留不一致是为了可审计，而非认可该计算。

## 交付时建议一并携带

1. `figure_inputs/manifest.json` 及完整 `figure_inputs/`，用于复现当前图的数值输入。
2. 本文件、`outputs/figure_data_audit_frozen_20261119.md` 和 `reference_results/`，用于交代口径与可追溯性。
3. `source_snapshot/` 与其 manifest 保留本次收集的算法、CUDA、配置和 benchmark 代码；不含 Quest vendor 构建目录、数据集、模型权重或 Qasper trace。若需重新运行实验，还须提供这些外部资源、特定 CUDA/Python 依赖及原始结果；见 [PIPELINES.md](PIPELINES.md)。
