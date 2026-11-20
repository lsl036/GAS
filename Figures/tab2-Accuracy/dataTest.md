## Longbench

| Method | Complete protocol | First-50 macro | Full 3,150 macro |
|---|---:|---:|---:|
| Full attention | 13/13 | 48.84 | **50.33** |
| Full + Twilight | 7/13 | 47.73 | pilot only |
| Full + GAS| 5/13 | 48.74 | pilot only |
|-|-|-|-|
| Quest, 4% | 13/13 | 38.27 | 38.67 |
| Quest, 8% | 13/13 | 43.12 | 44.36 |
| Quest+Twilight | 13/13 | 49.59 | 49.84 |
| Quest+ GAS | 13/13 | 49.77 | 50.09 |
|-|-|-|-|
| DS, 4% | 13/13 | 46.26 | pilot only |
| DS, 8% | 13/13 | 48.47 | pilot only |
| DS+Twilight | 13/13 | 49.27 | pilot only |
| DS+GAS | 13/13 | 49.74 | pilot only |

## RULER
| 方法 | 16K | 32K | 64K | LongBench |
|---|---:|---:|---:|---:|
| Full attention | 80.97 | 78.98 | 78.33 | 50.33 |
| Full + Twilight | 81.23 | 78.66 | 76.69 | 47.73 |
| Full + GAS | 82.41 | 78.92 | 77.80 | 48.74 |
|-|-|-|-|-|
| Quest 4% | 69.00 | 71.71 | 72.21 | 38.67 |
| Quest 8% | 76.56 | 75.72 | 75.18 | 44.36 |
| Quest + Twilight | 80.30 | 77.58 | 73.94 | 49.84 |
| Quest + GAS | 81.95 | 77.93 | 75.60 | 50.09 |
|-|-|-|-|-|
| DS 4% | 81.53 | 79.27 | 75.49 | 46.26 |
| DS 8% | 81.56 | 79.13 | 77.23 | 48.47 |
| DS + Twilight | 82.41 | 78.74 | 76.44 | 49.27 |
| DS + GAS | 81.86 | 78.91 | 77.88 | 49.74 |

## LessIsMore

| Method     | Budget  | RULER 16K | RULER 32K | RULER 64K | LongBench Avg. |
| ---------- | ------- | --------- | --------- | --------- | -------------- |
| LessIsMore | K=128   | 56.54     | 57.76     | 53.42     | 41.60          |
| LessIsMore | K=512   | 65.82     | 60.62     | 61.62     | 47.93          |
|            |         |           |           |           |                |
| LessIsMore | 动态 4% | 65.09     | 65.22     | 68.47     | 47.94          |
| LessIsMore | 动态 8% | 66.08     | 68.78     | 71.18     | 49.94          |