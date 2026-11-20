# Time Breakdown — 16K

仅保留16K上下文，比较batch 32/64/96/128下的Quest+Twilight、Quest adapter与GAS。依次运行下面两个代码单元格，即可绘图并导出PDF、PNG和CSV。

数据以显式数组放在第二个代码块顶部，维度为 `[batch, component]`，单位ms。已移除其他上下文数据、缺失点处理及多子图循环。

| 组件 | 数据含义 |
|---|---|
| Common layers | 32层公共模型计算；不是token selector |
| Independent frontend | 独立测量的选择、估分、refinement与metadata成本，累计30层 |
| Dense attention (2 layers) | 两层FlashInfer dense attention |
| Sparse attention (30 layers) | 30层对应方法的sparse attention |
| Group-Aware Sparsity | GAS 的 selection_ms 减去同 batch 下 quest_twi 的 selection_ms；用独立颜色显示 |

来源：`e7_v1_simplified_projection/E7_v1_frontend_inclusive.csv`。这是包含独立前端成本的组件估算，不是实测端到端serving时延。现有数据没有单独的token selector计时，不能用Common layers替代。

绘图略去 Output projection。GAS 的前端开销拆分为与 quest_twi 等长的 Top-p Pruning 和上述 Group-Aware Sparsity 差值，两者之和保持为原始 GAS selection_ms。此差值用于绘图分解，并非单独测量的组件计时。原始数组与 CSV 原始计时列保留；CSV 另附 top_p_pruning_ms、group_aware_sparsity_ms 和不含输出投影的 plotted_total_ms。

运行后更新同目录`time_breakdown.pdf`、`time_breakdown.png`、`time_breakdown_data.csv`。
