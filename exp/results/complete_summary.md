# GPU Reproduction Prototype 完整結果總整理

## 1. 實驗版本與口徑

目前 `exp/` 底下的 benchmark 一共演進成五個版本：

1. `proto`
只量簡化版 raw prefix slice selection + variation。

2. `pairwise`
改成每個 block 處理一對 tree，`mutation` 視為與 donor 全樹做 crossover，並納入 `DtoH child copy`。

3. `final`
在 `pairwise` 基礎上納入 formal `name/const remap`，並做真正的 `evaluation` 與 `CPU preprocess + H2D` overlap 測試。

4. `validate`
在 `final` 基礎上再納入正式 `validate_genome()` 與 `fallback to parent`。

5. `cheap-validate`
在 `validate` 基礎上加入 GPU cheap validate prefilter，並補做同版 `cheap on/off` 可比對照。

因此不同版本對應不同問題：

- `proto`：GPU raw variation 本身值不值得做？
- `pairwise`：改成 tree-pair model 後，連同 `DtoH` 還快不快？
- `final`：把 formal remap 與 overlap 納入後還快不快？
- `validate`：把 validate/fallback 納入後，整體還剩多少優勢？
- `cheap-validate`：先在 GPU 擋掉明顯 invalid child，能不能把 validate path 再壓下來？


## 2. 原始結果總表

### population = 1024

| 版本 | gpu_eval_ms | cpu_preprocess_ms | pack+h2d_ms | gpu_sel+var_ms | gpu_sel+var+d2h_ms | cpu_sel+var_ms | gpu_validate_ms | cpu_validate_ms | sequential_wall_ms | overlap_wall_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `proto` | 20.669 | 2.194 | 0.311 | 0.025 | - | 1.529 | - | - | - | - |
| `pairwise` | 20.716 | 5.974 | 0.441 | 0.044 | 0.200 | 2.661 | - | - | - | - |
| `final` | 20.727 | 6.023 | 0.725 | 0.074 | 0.232 | 4.486 | - | - | 27.731 | 21.640 |
| `validate` | 20.730 | 6.080 | 0.721 | 0.085 | 0.382 | 3.804 | 52.276 | 58.843 | 27.783 | 21.644 |
| `cheap-validate` | 20.724 | 6.284 | 0.710 | 0.092 | 0.426 | 4.168 | 35.255 | 37.661 | 27.930 | 21.673 |

### population = 4096

| 版本 | gpu_eval_ms | cpu_preprocess_ms | pack+h2d_ms | gpu_sel+var_ms | gpu_sel+var+d2h_ms | cpu_sel+var_ms | gpu_validate_ms | cpu_validate_ms | sequential_wall_ms | overlap_wall_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `proto` | 36.886 | 8.179 | 0.942 | 0.031 | - | 6.403 | - | - | - | - |
| `pairwise` | 35.138 | 20.983 | 1.364 | 0.056 | 0.589 | 9.987 | - | - | - | - |
| `final` | 33.267 | 20.985 | 2.442 | 0.086 | 0.619 | 16.672 | - | - | 57.911 | 36.126 |
| `validate` | 35.363 | 21.161 | 2.413 | 0.109 | 1.083 | 15.524 | 189.148 | 217.408 | 59.577 | 37.034 |
| `cheap-validate` | 34.689 | 22.411 | 2.383 | 0.121 | 1.127 | 15.482 | 127.660 | 143.735 | 58.993 | 37.655 |


## 3. 各種加速比

### 3.1 純 selection + variation speedup

公式：

`cpu_sel+var_ms / gpu_sel+var_ms`

| 版本 | pop=1024 | pop=4096 |
| --- | ---: | ---: |
| `proto` | 61.2x | 206.5x |
| `pairwise` | 60.5x | 178.3x |
| `final` | 60.6x | 193.9x |

解讀：

- 即使從 raw slice edit 走到 pairwise tree model，再走到 formal remap，GPU kernel 本體都仍然極快。
- 這代表真正值得注意的不是 GPU crossover/mutation kernel 夠不夠快，而是其他外圍成本會吃掉多少優勢。


### 3.2 把 DtoH 算進去的 GPU variation speedup

公式：

`cpu_sel+var_ms / gpu_sel+var+d2h_ms`

| 版本 | pop=1024 | pop=4096 |
| --- | ---: | ---: |
| `pairwise` | 13.3x | 17.0x |
| `final` | 19.3x | 26.9x |
| `validate` | 10.0x | 14.3x |

解讀：

- `DtoH` 一加入，加速比會從數十倍明顯掉到十幾倍。
- 不過在所有版本中，算入 `DtoH` 後 GPU 仍然明顯快於 CPU。
- `final` 比 `pairwise` 的含 `DtoH` speedup 更高，主要是 formal remap 讓 CPU variation 成本更高，而 GPU 端增加幅度小很多。


### 3.3 把 validate/fallback 也算進去的整體加速比

公式：

`(cpu_sel+var_ms + cpu_validate_ms) / (gpu_sel+var+d2h_ms + gpu_validate_ms)`

| 版本 | pop=1024 | pop=4096 |
| --- | ---: | ---: |
| `validate` | 1.19x | 1.22x |
| `cheap-validate` | 1.17x | 1.24x |

解讀：

- 這是目前最接近正式流程的 speedup。
- 一旦把 `validate/fallback` 納入，原本十幾倍到上百倍的優勢，最後只剩下約 `1.2x`。
- 結論不是 GPU 沒用，而是新的主瓶頸已經不在 GPU variation，而在 validation path。


### 3.4 overlap 對整體 wall time 的改善

公式：

`sequential_wall_ms / overlap_wall_ms`

| 版本 | pop=1024 | pop=4096 |
| --- | ---: | ---: |
| `final` | 1.281x | 1.603x |
| `validate` | 1.284x | 1.609x |
| `cheap-validate` | 1.289x | 1.567x |

解讀：

- 這個比值在 `final` 與 `validate` 幾乎一致。
- 代表 `validate/fallback` 雖然很重，但沒有推翻另一個結論：
  `CPU preprocess + packing + H2D` 的確能被 `GPU evaluation` 掩蓋掉一部分。


## 4. evaluation 對前處理的掩蓋能力

### 4.1 只看 CPU preprocess

公式：

`gpu_eval_ms / cpu_preprocess_ms`

| 版本 | pop=1024 | pop=4096 |
| --- | ---: | ---: |
| `proto` | 9.42x | 4.51x |
| `pairwise` | 3.47x | 1.68x |
| `final` | 3.44x | 1.59x |
| `validate` | 3.41x | 1.67x |
| `cheap-validate` | 3.30x | 1.55x |

解讀：

- 換成正式 subtree 標記後，CPU preprocess 成本上升很多。
- 但即使在最接近正式版的 `validate`，evaluation 仍然大於 preprocess。
- `population=4096` 已經逼近邊界，代表之後若 preprocess 再重，overlap 空間會開始變緊。


### 4.2 看 CPU preprocess + packing + H2D

公式：

`gpu_eval_ms / (cpu_preprocess_ms + pack+h2d_ms)`

| 版本 | pop=1024 | pop=4096 |
| --- | ---: | ---: |
| `proto` | 8.25x | 4.04x |
| `pairwise` | 3.23x | 1.57x |
| `final` | 3.07x | 1.42x |
| `validate` | 3.05x | 1.50x |
| `cheap-validate` | 2.97x | 1.40x |

解讀：

- 就算把 packing 與 H2D 一起算進來，evaluation 目前仍可覆蓋整段前處理。
- `population=4096` 的 margin 只剩 `1.4x ~ 1.5x`，表示這條結論是成立的，但已經不是非常寬鬆。


## 5. validate/fallback 為什麼把 speedup 吃掉

### 5.1 validate/fallback 在 GPU 路徑中的占比

公式：

`gpu_validate_ms / (gpu_sel+var+d2h_ms + gpu_validate_ms)`

| population | validate share |
| --- | ---: |
| 1024 | 99.3% |
| 4096 | 99.4% |

### 5.2 validate/fallback 在 CPU 路徑中的占比

公式：

`cpu_validate_ms / (cpu_sel+var_ms + cpu_validate_ms)`

| population | validate share |
| --- | ---: |
| 1024 | 93.9% |
| 4096 | 93.3% |

### 5.3 fallback count

| population | cpu_fallback_count | gpu_fallback_count |
| --- | ---: | ---: |
| 1024 | 10225 | 10220 |
| 4096 | 40915 | 40896 |

解讀：

- 現在最大的問題不是 GPU 不夠快，而是產生了太多 invalid child。
- 在目前 benchmark 下，GPU 路徑裡幾乎全部時間都花在 `validate/fallback`。
- 這也說明下一步最有價值的工作，不是再壓 GPU kernel，而是降低 invalid child 比例或降低 validation 成本。


## 6. cheap validate 的可比效果

這裡只比較**同一版 benchmark、同一套 `used_len` 與 full validate 邏輯**下：

- `cheap off`
- `cheap on`

因此這組數字是目前最公平的 cheap-validate 對照。

### 6.1 GPU 路徑總成本

公式：

- `cheap off = gpu_sel+var+d2h + gpu_validate_fallback`
- `cheap on = gpu_sel+var+d2h + gpu_cheap_validate + gpu_validate_fallback`

| population | cheap off | cheap on | 改善 |
| --- | ---: | ---: | ---: |
| 1024 | `40.708 ms` | `35.726 ms` | `1.139x` |
| 4096 | `147.971 ms` | `128.833 ms` | `1.149x` |

### 6.2 整體 GPU 對 CPU speedup

| population | cheap off | cheap on |
| --- | ---: | ---: |
| 1024 | `1.022x` | `1.171x` |
| 4096 | `1.075x` | `1.236x` |

### 6.3 解讀

- `cheap validate` 本身非常便宜，但確實能把 GPU validate path 再壓低一截。
- 在真正可比的口徑下，cheap validate 不是噪音，而是實際上把整體 speedup 拉高了。
- 它不是最終解法，但已經證明「便宜 structural prefilter 上 GPU」這條路是有效的。


## 7. 橫向比較後的完整結論

### 6.1 已經被實驗支持的結論

1. `GPU selection + crossover/mutation kernel` 本體極快，這點在所有版本都成立。
2. `DtoH` 會明顯吃掉一部分 speedup，但不足以推翻 GPU 優勢。
3. formal `name/const remap` 確實有成本，而且主要拉高 CPU variation 成本。
4. `CPU preprocess + packing + H2D` 目前仍可與 `GPU evaluation` overlap，並帶來約 `1.28x ~ 1.61x` 的 wall-time 改善。
5. 一旦納入 `validate/fallback`，端到端 speedup 只剩約 `1.2x`，而新瓶頸轉移到 validation path。
6. 加入 GPU cheap validate 後，在可比條件下整體 speedup 可從 `1.022x -> 1.171x`（1024）與 `1.075x -> 1.236x`（4096）。

### 6.2 目前最準確的研究敘事

如果不含 `validate/fallback`：

- GPU reproduction-style variation 有非常強的速度潛力。

如果含 `validate/fallback`：

- GPU 路線仍然略快，
- 但主要瓶頸不再是 crossover/mutation 本身，
- 而是 invalid child 導致的 `validate/fallback` 成本。

因此目前最準確的總結是：

`GPU 做 reproduction 並不是不可行；真正限制整體加速比的，是 formal validity path，而不是 GPU subtree edit kernel。`


## 8. 下一步最值得做的事

1. 讓 candidate / donor / pairing 更接近正式 typed 規則，降低 invalid child 比例。
2. 研究 cheaper prefilter 或 cheap validation，避免每個 child 都走完整 `validate_genome()`。
3. 量化 invalid child 比例下降後，整體 speedup 能回升多少。
