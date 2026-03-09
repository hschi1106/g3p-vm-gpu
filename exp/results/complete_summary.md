# GPU Reproduction Prototype Current Summary

這份 summary 只保留現行無 `validate` 口徑。`exp/repro_proto_bench.cu` 現在量的是：

- CPU preprocess
- host packing
- H2D
- GPU evaluation
- GPU selection + variation
- D2H child copy
- CPU 對照版 selection + variation
- sequential / overlap wall time

它不再量：

- `validate_genome()`
- cheap validate
- fallback to parent

## 1. Current result files

- pop=1024: [current_pop1024.json](/home/hschi1106/g3p-vm-gpu/exp/results/current_pop1024.json)
- pop=4096: [current_pop4096.json](/home/hschi1106/g3p-vm-gpu/exp/results/current_pop4096.json)

## 2. Raw results

| population | gpu_eval_ms | cpu_preprocess_ms | pack_total_ms | h2d_total_ms | gpu_sel+var_ms | gpu_sel+var+d2h_ms | cpu_sel+var_ms | sequential_wall_ms | overlap_wall_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 21.347 | 5.640 | 0.525 | 0.246 | 0.041 | 0.357 | 3.423 | 27.905 | 21.385 |
| 4096 | 41.786 | 21.887 | 2.121 | 0.592 | 0.068 | 1.070 | 14.253 | 69.690 | 44.223 |

## 3. Speedup

### 3.1 Pure selection + variation

公式：

`cpu_sel+var_ms / gpu_sel+var_ms`

| population | speedup |
| --- | ---: |
| 1024 | 83.488x |
| 4096 | 209.603x |

### 3.2 Including D2H

公式：

`cpu_sel+var_ms / gpu_sel+var+d2h_ms`

| population | speedup |
| --- | ---: |
| 1024 | 9.588x |
| 4096 | 13.321x |

解讀：

- GPU kernel 本體仍然非常快。
- `DtoH` 會明顯吃掉一部分優勢，但在 current 口徑下仍保留兩位數 speedup。

## 4. Overlap

### 4.1 Evaluation hide preprocess

公式：

`gpu_eval_ms / cpu_preprocess_ms`

| population | ratio |
| --- | ---: |
| 1024 | 3.785x |
| 4096 | 1.909x |

### 4.2 Evaluation hide preprocess + pack + H2D

公式：

`gpu_eval_ms / (cpu_preprocess_ms + pack_total_ms + h2d_total_ms)`

| population | ratio |
| --- | ---: |
| 1024 | 3.330x |
| 4096 | 1.699x |

### 4.3 Sequential vs overlap wall time

公式：

`sequential_wall_ms / overlap_wall_ms`

| population | speedup |
| --- | ---: |
| 1024 | 1.305x |
| 4096 | 1.576x |

解讀：

- 在兩個 population 下，`CPU preprocess + packing + H2D` 都仍可被 `GPU evaluation` 掩蓋一部分。
- `population=4096` 的 margin 較小，但 current benchmark 下仍然成立。

## 5. Current conclusions

1. 現行無 validate 口徑下，GPU reproduction-style variation 有很強的速度優勢。
2. 單看 kernel，本體仍是高倍數 speedup。
3. 把 `DtoH` 算進去後，GPU reproduction path 仍然明顯快於 CPU。
4. `CPU preprocess + packing + H2D` 目前仍能與 `GPU evaluation` overlap。
5. current 敘事應該是：GPU reproduction path 的主要價值在於把 selection + variation 壓到極低成本，並且前處理仍可被 evaluation 掩蓋一部分。

## 6. Next work

1. 讓 `exp/repro_proto_bench.cu` 與正式 evolution loop 共用更多 preprocessing / packing 元件，減少 bench-only 邏輯。
2. 針對更大的 population 再量一次，確認 overlap margin 何時開始明顯收斂。
3. 若要往正式整合前進，優先量 child copyback 與 host-side compile 接入後的成本，而不是重新引入 validate path。
