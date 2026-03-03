# GPU Reproduction Prototype 整併指令

這份文件是給之後實作用的。

目標不是一次把整個 reproduction 搬上 GPU，而是把目前 `exp/repro_proto_bench.cu` 已驗證過的路徑，逐步整進正式 evolution pipeline。

整體原則：

1. 保留 CPU 前處理
2. 把 `selection + crossover/mutation + cheap validate` 變成 GPU 路徑
3. 只讓 cheap-pass child 回 CPU 做 full validate / compile
4. 先做成可切換 backend，不要直接取代原本 CPU reproduction


## 一、要整進正式流程的最小版本

先實作一個新的 reproduction engine，暫名：

- `cpu`：原本流程
- `gpu_proto`：新的混合流程

`gpu_proto` 每一代流程固定如下：

1. GPU evaluation 跑當代 population
2. CPU 同步做下一代前處理
   - `build_subtree_end`
   - candidate subtree 收集
   - donor 生成
   - host packing
3. GPU 做 `tournament selection`
4. GPU 做 `pairwise crossover/mutation`
5. GPU 做 `cheap validate`
6. 只把 cheap-pass child 搬回 CPU
7. CPU 對 cheap-pass child 做正式 `validate_genome()`
8. valid child 做 compile
9. invalid child fallback 到 parent


## 二、第一階段實作指令

### 1. 建立新的正式元件，不要直接把 benchmark 檔案塞進既有邏輯

新增一個正式 C++ 模組，目標是把 `exp/repro_proto_bench.cu` 裡和 reproduction 有關的邏輯拆成可重用元件。

建議檔案：

- `cpp/include/g3pvm/repro_gpu_proto.hpp`
- `cpp/src/repro_gpu_proto.cu`

這個模組至少要提供：

- host 端前處理資料結構
- packed host/device buffers
- GPU selection kernel
- GPU variation kernel
- GPU cheap validate kernel
- host 端 DtoH + full validate + fallback glue code

注意：

- 不要把 benchmark 的 `main()`、測試資料生成、JSON 輸出帶進正式模組
- 只搬可重用的 runtime parts


### 2. 把 benchmark 裡這些邏輯抽成正式 API

需要抽出的核心功能：

- formal subtree preprocessing
  - `build_subtree_end`
  - `collect_formal_candidates`
- donor pool 準備
- host packing
- device buffer allocate / upload
- GPU tournament selection
- GPU variation
- GPU cheap validate
- child copyback
- CPU full validate + fallback

正式 API 目標不是 expose 全部細節，而是讓 evolution loop 可以直接呼叫例如：

- `prepare_generation_inputs(...)`
- `run_gpu_reproduction_proto(...)`
- `finalize_children_after_gpu(...)`


### 3. 在正式 evolution config 中新增 reproduction backend 選項

需要新增一個 config / enum，例如：

- `ReproductionBackend::Cpu`
- `ReproductionBackend::GpuProto`

CLI 或 config 先做成可切換，例如：

- `--repro-backend cpu`
- `--repro-backend gpu_proto`

預設先維持 `cpu`，不要改既有預設行為。


### 4. 在正式 evolution loop 中插入 `gpu_proto` 分支

找到目前每代做：

- selection
- crossover
- mutation
- elite copy
- compile

的主流程位置。

當 `repro_backend == gpu_proto` 時：

- elite copy 先維持 CPU 舊邏輯
- 其餘 child 用 GPU proto 路徑產生

先不要把 elitism 也搬上 GPU。


## 三、第二階段實作指令

### 5. 保留 CPU 前處理，並與 evaluation overlap

不要先搬：

- subtree preprocessing
- candidate 收集
- donor 生成
- host packing

這些工作。

理由：

- 實驗已顯示這段目前仍可被 `gpu_eval` 掩蓋
- 現在先整進正式流程，比較低風險

實作上要做的是：

- evaluation 開始後，CPU 立刻開始做 next-gen preprocess
- preprocess 結果準備好後，進行 H2D staging

如果現有 session / evaluation API 不方便 overlap，先保留順序版本也可以，但 code 結構要保留之後 overlap 的空間。


### 6. GPU cheap validate 後，只搬 cheap-pass child

這一步是整併的關鍵。

不要沿用 benchmark 現在「全部 child 都 DtoH」的做法當最終版。

正式版目標：

- `cheap fail` child：
  不搬回 CPU
  直接在 host 端標記為 fallback to parent

- `cheap pass` child：
  才做 DtoH
  然後 full validate / compile

這需要在 GPU 端多輸出：

- `cheap_valid` flag
- child 索引映射

並在 host 端做 compact / gather。


### 7. CPU full validate / fallback 先保留原正式語義

不要在這一步偷改正式邏輯。

對 cheap-pass child：

- rebuild `ProgramGenome`
- 跑正式 `validate_genome()`
- invalid -> fallback parent
- valid -> compile

目標是：

- 先保持語義接近正式流程
- 只把明顯不必要的 full validate 工作量砍掉


## 四、第三階段實作指令

### 8. 補正式統計欄位

整進正式流程後，至少要新增以下統計：

- `gpu_repro_selection_ms`
- `gpu_repro_variation_ms`
- `gpu_repro_cheap_validate_ms`
- `gpu_repro_d2h_ms`
- `gpu_repro_cheap_reject_count`
- `gpu_repro_full_validate_count`
- `gpu_repro_full_validate_fail_count`
- `gpu_repro_compile_ms`

另外保留：

- end-to-end generation wall time
- evaluation wall time
- reproduction wall time

這些欄位之後要拿來驗證：

- benchmark 的結果能不能在正式流程重現
- 新瓶頸是不是 compile / full validate


### 9. 加 correctness 對照測試

至少補這幾類測試：

- `gpu_proto` 與 `cpu` 在同 seed 下 child 數量一致
- fallback 行為不會產生越界 / crash
- cheap validate fail 的 child 會正確 fallback
- cheap validate pass 但 full validate fail 的 child 也會正確 fallback

注意：

`gpu_proto` 不需要保證 child 與 `cpu` 逐一完全相同，因為 operator 本身已簡化。

但至少要保證：

- child buffer 合法
- fallback 合法
- compile/eval 不會因為 GPU proto child 而炸掉


## 五、目前不要做的事

以下先不要碰：

1. 不要把 full `validate_genome()` 直接搬上 GPU
2. 不要把 compile 搬上 GPU
3. 不要一次整合正式版所有 crossover/mutation 分支
4. 不要先優化 kernel micro-detail
5. 不要先追求 bitwise identical child sequence

這些都不是目前最有價值的工作。


## 六、實作順序

之後若要我直接開始實作，請照這個順序做：

1. 建 `repro_gpu_proto` 正式模組
2. 搬 benchmark 內可重用的 host/device logic
3. 在 evolution config / CLI 加 `gpu_proto`
4. 在正式 evolution loop 接 `gpu_proto`
5. 先做順序版整合
6. 確認 correctness 與 basic timing
7. 再做 overlap
8. 再做 cheap-pass only DtoH compact


## 七、這份文件的用途

如果之後要我實作，直接引用這份文件即可。

預設目標是：

`把 exp 中驗證過的 GPU reproduction prototype，以最小風險整進正式流程，並保留 CPU full validate/compile 作為後段保險。`
