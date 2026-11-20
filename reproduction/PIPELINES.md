# GAS 实现与实验流程

本目录是独立的绘图与实验入口，不复制整个 96 GB 工作树。仓库以 `GAS/` 为主路径，旧路径 `Twilight_Ours -> GAS` 是兼容链接；当前 Python 包名为 `gas_attention`。原始 `../Twilight` 和 `../Twilight_Upstream_20261113` 保留为独立参照。`source_snapshot/` 是迁移前的历史代码哈希快照，仍使用 `twilight` 包名；验证运行 `python reproduction/source_snapshot.py verify`。快照排除模型、结果、数据、RULER synthetic JSON 语料和 `benchmark/efficiency/vendor/` 等大体积外部依赖，使用这些功能时需另行提供。脚本中的旧绝对路径仍应逐项核对。

## 实现地图

| 内容 | 文件 | 作用 |
| --- | --- | --- |
| 模型 attention | `../gas_attention/pyimpl/attention.py` | LLaMA/Qwen attention patch；prefill 与 decode 分支，selector、estimator、top-p、refinement 和实际输出 |
| GAS mask 与逻辑统计 | `../gas_attention/pyimpl/gqa_refine.py` | group-max/group-mean 等 refinement；按 KV group 做 head union 统计 |
| 量化与 top-p | `../gas_attention/pyimpl/quantize.py`、`../gas_attention/pyimpl/top_p.py` | fake-quant/实验性真实 INT4 estimator 和 top-p 筛选 |
| Quest 与 DS | `../gas_attention/pyimpl/quest.py`、`../gas_attention/pyimpl/double_sparse.py` | 候选 token 选择；DS 需要模型对应 channel 配置 |
| 每层 profile | `../gas_attention/pyimpl/profiler.py`、`../benchmark/RULER/scripts/pred/model_wrappers.py` | decode 分项计时、逐样本 GQA 指标汇总 |
| 自定义 attention prototype | `../csrc/src/gas_cooperative_shared_pool.cu` | shared KV pool + shared/exception 索引 fused decode，服务于实验 replay |
| 索引生成 prototype | `../csrc/src/gas_compact_refine.cu`、`../benchmark/efficiency/gas_compact_refinement.py` | group refinement 和 compact metadata 实验实现 |
| 其他 fused decode | `../gas_attention/pyimpl/gqa_fused_decode.py`、`../csrc/src/gqa_group_decode.cu` | 实验性可选路径，不能默认等同于质量实验的原有执行路径 |

质量实验通常在模型 patch 中构建 mask 并做浮点注意力计算；其中 fake quant 不等于 packed INT4 算术。attention-only custom kernel replay 使用预先准备的索引和共享物理 KV pool，不能称为模型已端到端接入并加速。不同实现的计时范围应分别报告。

## 运行前检查

使用具备 CUDA、PyTorch、transformers、flash-attn、flashinfer 及项目依赖的环境，确保驱动和 PyTorch CUDA 兼容；运行前检查空闲 GPU 和足够显存。`moba` 当前 SciPy 来自 user-site，不能假设 `PYTHONNOUSERSITE=1` 可用。本环境还可能受外部 `tvm_ffi`、JIT 编译器和系统字体影响；依赖未被这次整理改写。

```bash
cd /data/home/lsl/SparseAtt/GAS
PY=/data/home/lsl/Software/miniconda3/envs/moba/bin/python
PYTHONPATH="$PWD:${PYTHONPATH:-}" "$PY" -c 'import sys, torch, scipy, gas_attention; print(sys.executable, torch.__version__, torch.version.cuda, scipy.__version__, gas_attention.__file__)'
nvidia-smi
```

以下是实验入口，不承诺可在任何 GPU 上复现相同时延。脚本需要相应模型、数据、扩展、GPU 和足够长的运行时间；先用 `--dry-run`，使用新输出目录，避免覆盖已有数据。随机样本、原始数据版本、模型和 seed 必须另行记录；只对齐超参数不意味着逐样本复现。

## Fig.1 与 Fig.2：PPL/union

当前输入为 `figure_inputs/fig1-PerpvsLoadedKV/TopP_ppl_KVamp.xlsx` 与 Fig.2 notebook 的数据。PG19 模型测试入口是 `../benchmark/PG19/eval_pg19_ppl.py`：

```bash
"$PY" benchmark/PG19/eval_pg19_ppl.py --help
```

从模型重新计算需要 PG19 测试文件、模型、方法配置和明确的 sample/sequence/eval 长度；不能把入口默认值假定成 workbook 的原始生成参数。union 计算定义见 `gqa_refine.py`。目前未建立 workbook 各点到每条原始推理记录的一一对应，重绘与重新测量应分开表述。

## Fig.4(a)：32K RULER quality / logical loaded KV

`run_benchmark.py fig4a` 调用 `../benchmark/RULER/fig4a/run_fig4a.py`，包含 calibration、选择预算、formal 和汇总，计算量很大。正式比较使用 5 个任务、每任务 50 条；校准与正式数据需明确隔离。以下 `CAL_DATA`、`FORMAL_DATA` 是各任务子目录所在的数据目录，不是预测输出目录：

```bash
CAL_DATA=/absolute/path/to/calibration/data
FORMAL_DATA=/absolute/path/to/formal/data
"$PY" reproduction/run_benchmark.py --dry-run fig4a \
  --model /data2/lsl_data/models/Meta-Llama-3.1-8B-Instruct \
  --channel-config benchmark/configs/ds_channel_llama31_q.json \
  --calibration-data-root "$CAL_DATA" --formal-data-root "$FORMAL_DATA" \
  --output reproduction/runs/my_fig4a
```

将 `--dry-run` 移除才会开始 GPU 实验。预算选择及正式集输出不能混同，表中旧点及 99.5% 统计的实际范围见 [PROVENANCE.md](PROVENANCE.md)。

## Fig.5：Passkey accuracy 与 GQA loaded-KV

Fig.5 使用 Quest/DS 加 Twilight 与 GAS 的 Passkey 数据；统一入口默认提供 GAS 配置，支持 LLaMA/Qwen3 与 8K/16K，`--selector ds` 会将 DS channel 路径解析为本仓库路径。README 中提供 `--dry-run` 示例，真实运行移除 `--dry-run`。重跑还需要相应模型和 DS channel 配置。对于 baseline，应显式运行原 Passkey profiler 并选用对应 JSON 配置，不能把 GAS 配置当作 baseline。

实际完成一次 LLaMA、Quest、8K、1 iteration 的 GPU smoke：`runs/smoke_llama_quest_8k_usersite_20261119/`，正确率 1/1，profile 3808 条。这只验证端到端入口可运行，不是原图完整复现。失败的无 user-site 尝试也保留在 `runs/`，不是成功结果。

开发包重命名后又运行了同样大小的一次测试：`runs/package_rename_llama_quest_8k_20261119/`，1/1 正确、profile 3808 条；该次测试使用已有 CUDA 扩展缓存，不能证明从零编译或所有 benchmark 已通过。

快照中的入口也可单独调用，例如把上面的 `reproduction/run_benchmark.py` 换成 `reproduction/source_snapshot/reproduction/run_benchmark.py`，并使用新的绝对输出路径；它会优先加载快照内的 `twilight` 和配置。已从该入口实测 LLaMA、Quest、8K、1 iteration：`runs/snapshot_llama_quest_8k_20261119/`，正确率 1/1，profile 3808 条。其他 benchmark 的数据、trace、第三方代码及相应环境仍需额外提供。

## Fig.6：Qasper attention replay

`run_benchmark.py qasper` 封装 `../benchmark/efficiency/bench_qasper_gas_paired.py` 的 GAS-FI、GAS-custom 同输入 replay。脚本依赖 `benchmark/results_efficiency/fig6_traces/` 两个固定 Qasper trace 及其所有 28 层，32K 是 repeated context；默认 `--timer author` 会导入独立的 `../Twilight_Upstream_20261113/benchmark/efficiency/bench_utils.py`。模型权重不在计时循环内，且是两个物理 KV pool 的共享重放。

```bash
"$PY" reproduction/run_benchmark.py --dry-run qasper \
  --context 16k --batch 32 --timer author \
  --output reproduction/runs/my_qasper_16k_b32
```

对 16K/32K 和 B32/64/96/128 分别使用新输出目录。该入口只覆盖 GAS 配对，不自动重测其他 baseline；Quest/Twilight/Less-style、dense 等入口可查 `bench_qasper_rate08_grouped.py`、`bench_qasper_rate08_dense_tuned.py` 及历史配置。先对齐输入、预算、计时与包含的开销，不要将不同实验拼成同次 paired 结果。图表数组和当前实验汇总存在已知差异，见 [PROVENANCE.md](PROVENANCE.md)。

从 `source_snapshot` 入口使用该 replay 时，原脚本仍按其项目根目录寻找 `benchmark/results_efficiency/fig6_traces/`，并按项目根目录的父目录寻找 `Twilight_Upstream_20261113`。两者不包含在源码快照中，需要在对应位置另行准备；仅验证了快照 Passkey，未从快照重新运行完整 Qasper 矩阵。

## Fig.7：分量投影

这是以实测分量构造的模拟 decoding-step 投影，而非实测端到端 TPOT。相应入口包括 `../benchmark/efficiency/summarize_qasper_rate08_tpot_tuned.py` 和 `summarize_qasper_gas_custom_tpot.py`。重跑汇总前须确认其输入 CSV、层数、分量范围和后端是否与目标图一致；脚本某些默认输出是历史结果，必须显式传新输出路径，不能直接覆盖。当前 notebook 混用来源，见 [PROVENANCE.md](PROVENANCE.md)。

## Fig.8：breakdown

当前绘图数据有显式缩放与标签问题；原样重绘不构成可发表的耗时 breakdown。实验性实现入口包含 `../benchmark/efficiency/run_upstream_breakdown_32k.py`、`bench_gas_compact_pipeline.py`。如果重新采集，应固定相同输入、预算和计时方式，分别测量 selector、estimator、top-p、refinement、attention，区分真实相连 pipeline 和离线分量求和，并保留原始数据及正确标签。

## Tab.2：LongBench 与 RULER

LongBench 选择器矩阵入口为 `../benchmark/LongBench/run_e5_selector_matrix.sh`，控制项包括 `E5_DATA_DIR`、`E5_MODEL_PATH`、`E5_OUTPUT_DIR`、`E5_TASKS` 及样本数；使用新目录并先审阅脚本/manifest。RULER 对应 `../benchmark/RULER/run_selector_matrix.sh`，任务、长度、样本量及输出由脚本与环境变量共同决定；重跑前必须核对它们，不应覆盖 `scripts/dataset/` 的历史结果。不同方法的 50 条、完整任务和中断产生的部分记录不可直接混为同样本比较。

每次正式结果至少归档模型和 tokenizer、数据与样本 ID、配置 JSON、代码版本、硬件和环境、完整性/失败记录、原始预测 JSONL 及汇总脚本版本。图表来源若不能从这些输入重算，应明确标注而不是补造数值。
