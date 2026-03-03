# GPU Cheap Validate 可比對照結果

這份結果只比較同一版 benchmark、同一套 `used_len` 與 full validate 邏輯下：

- `--disable-gpu-cheap-validate`
- 預設啟用 `gpu cheap validate`

因此這份對照比前一版更公平，因為唯一差異就是是否啟用 GPU cheap validate。


## population = 1024

來源：

- 無 cheap: [comparable_no_cheap_pop1024.json](/home/hschi1106/g3p-vm-gpu/exp/results/comparable_no_cheap_pop1024.json)
- 有 cheap: [cheap_validate_pop1024_rerun.json](/home/hschi1106/g3p-vm-gpu/exp/results/cheap_validate_pop1024_rerun.json)

### GPU 路徑總成本

公式：

- 無 cheap: `gpu_sel+var+d2h + gpu_validate_fallback`
- 有 cheap: `gpu_sel+var+d2h + gpu_cheap_validate + gpu_validate_fallback`

| 模式 | 總成本 |
| --- | ---: |
| 無 cheap | `40.708 ms` |
| 有 cheap | `35.726 ms` |

改善幅度：

- `40.708 / 35.726 = 1.139x`

### 整體 GPU 對 CPU speedup

公式：

- `overall = (cpu_sel+var + cpu_validate_fallback) / gpu_total`

| 模式 | speedup |
| --- | ---: |
| 無 cheap | `1.022x` |
| 有 cheap | `1.171x` |

### 單看 GPU formal validate

| 模式 | `gpu_validate_fallback_ms` |
| --- | ---: |
| 無 cheap | `40.285 ms` |
| 有 cheap | `35.255 ms` |

改善幅度：

- `1.143x`


## population = 4096

來源：

- 無 cheap: [comparable_no_cheap_pop4096.json](/home/hschi1106/g3p-vm-gpu/exp/results/comparable_no_cheap_pop4096.json)
- 有 cheap: [cheap_validate_pop4096.json](/home/hschi1106/g3p-vm-gpu/exp/results/cheap_validate_pop4096.json)

### GPU 路徑總成本

| 模式 | 總成本 |
| --- | ---: |
| 無 cheap | `147.971 ms` |
| 有 cheap | `128.833 ms` |

改善幅度：

- `147.971 / 128.833 = 1.149x`

### 整體 GPU 對 CPU speedup

| 模式 | speedup |
| --- | ---: |
| 無 cheap | `1.075x` |
| 有 cheap | `1.236x` |

### 單看 GPU formal validate

| 模式 | `gpu_validate_fallback_ms` |
| --- | ---: |
| 無 cheap | `146.873 ms` |
| 有 cheap | `127.660 ms` |

改善幅度：

- `1.151x`


## 結論

如果只比較真正可比的版本，`gpu cheap validate` 的效果是明確正面的：

1. GPU validate 路徑本身約改善 `1.14x ~ 1.15x`
2. GPU 整體 reproduction 路徑約改善 `1.14x ~ 1.15x`
3. 整體 GPU 對 CPU speedup 也明顯上升
   - `1024`: `1.022x -> 1.171x`
   - `4096`: `1.075x -> 1.236x`

因此可比結論是：

`cheap validate 有效，而且效果不是只體現在局部 validation time；它確實讓整體 GPU reproduction 路徑更有優勢。`
