# GPU Reproduction Integration Notes

這份文件只描述現行無 `validate` 口徑的整合方向。目標是把 `exp/repro_proto_bench.cu` 已經量到的 current path，逐步搬進正式 evolution pipeline。

現行原則：

1. 保留 CPU 前處理
2. GPU 化 `selection + crossover/mutation`
3. 保留 `DtoH child copy` 與 compile 接回 CPU
4. correctness 由 typed generation、typed operators、tests 與 parity 檢查保證
5. 不重新引入 heavyweight `validate`、cheap validate 或 fallback stage

## 一、Current benchmark flow

`exp/repro_proto_bench.cu` 目前每次量測的流程是：

1. GPU evaluation 跑當代 population
2. CPU 做下一代前處理
   - `build_subtree_end`
   - candidate subtree 收集
   - donor 生成
3. host packing
4. H2D upload
5. GPU tournament selection
6. GPU pairwise crossover/mutation
7. DtoH child copy
8. CPU 對照版 selection + variation timing
9. sequential / overlap wall time 對照

## 二、正式整合時應優先抽出的元件

建議拆出可重用模組，例如：

- host 端前處理資料結構
- packed host/device buffers
- GPU selection kernel
- GPU variation kernel
- child copyback 與統計

這些元件應讓正式 evolution loop 能直接呼叫，例如：

- `prepare_generation_inputs(...)`
- `upload_generation_inputs(...)`
- `run_gpu_reproduction(...)`
- `copy_children_back(...)`

## 三、正式流程建議

### 1. reproduction backend

在 evolution config 中新增：

- `ReproductionBackend::Cpu`
- `ReproductionBackend::GpuProto`

CLI 例如：

- `--repro-backend cpu`
- `--repro-backend gpu_proto`

預設維持 `cpu`。

### 2. evolution loop 接法

當 `repro_backend == gpu_proto` 時：

- elite copy 仍先維持 CPU 舊邏輯
- 其餘 child 走 GPU selection + variation
- child 回 CPU 後接現有 compile / eval 路徑

### 3. overlap

先保留 CPU 前處理，不要先搬：

- subtree preprocessing
- candidate 收集
- donor 生成
- host packing

理由是 current benchmark 已經顯示這段仍可被 `gpu_eval` 掩蓋一部分。

## 四、統計欄位

正式整合後至少保留：

- `gpu_repro_selection_ms`
- `gpu_repro_variation_ms`
- `gpu_repro_d2h_ms`
- `gpu_repro_compile_ms`
- `generation_wall_ms`
- `evaluation_wall_ms`
- `reproduction_wall_ms`

這些欄位主要用來驗證：

- `exp/` bench 的 current 結果能否在正式流程重現
- 新瓶頸是否轉移到 child copyback 或 compile

## 五、測試重點

至少補這幾類測試：

- `gpu_proto` 與 `cpu` 在同 seed 下 child 數量一致
- child buffer 合法，不會越界或 crash
- compile/eval 不會因為 GPU child 而炸掉
- CPU/GPU reproduction path 的 timing 與統計欄位可正常輸出

## 六、目前不要做的事

1. 不要重新把 `validate_genome()` 搬回公開主流程
2. 不要為了 benchmark 再加 cheap validate 路徑
3. 不要先把 compile 搬上 GPU
4. 不要先追求 bitwise identical child sequence
5. 不要先優化 kernel micro-detail

## 七、實作順序

1. 把 `exp/repro_proto_bench.cu` 中可重用的 host/device logic 拆成正式模組
2. 在 evolution config / CLI 加 `gpu_proto`
3. 在正式 evolution loop 接 `gpu_proto`
4. 先做順序版整合
5. 確認 correctness、parity 與 basic timing
6. 再做 preprocess/packing 與 evaluation overlap
