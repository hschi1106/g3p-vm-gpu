# C++ 重構計劃

本文是針對 `cpp/` 目錄的完整重構計劃。目標不是只做局部整理，而是把目前偏混亂、歷史堆疊式成長的 C++ 結構，整理成一個符合 KISS 原則、容易理解、容易維護、容易擴充到未來 `/exp` 新架構的版本。

---

## 1. 重構目標

本次重構的核心目標如下：

1. 讓 `cpp/` 的目錄結構一眼可懂，不需要靠「知道歷史」才能理解。
2. 移除不必要的 legacy code、未使用 function、過時命名與過渡期 API。
3. 讓 CPU runtime、GPU runtime、evolution、CLI、bench、test 的責任邊界清楚。
4. 把目前主線架構整理成可支援未來 `/exp` 新架構接入的形式。
5. 在不破壞目前語義的前提下，維持：
   - CPU/GPU fitness parity
   - 目前已接受的 speedup 水準
   - 演化確實有進步
6. 降低未來 agent 或人類開發者理解這個專案的成本。
7. 將 selection 策略收斂成單一路徑：
   - 只保留 `tournament`
   - selection pressure 作為明確可調參數
8. 將 mutation 收斂成單一路徑：
   - 對外只保留一個 `mutation`
   - 內部 sub-operator 必須具名
   - 不保留隱藏 magic ratio
   - operator 比例改成顯式參數

---

## 2. 重構原則

### 2.1 KISS

- 每個資料夾只承擔一類責任。
- 每個檔案只承擔一個主要職責。
- 每個公開 API 名稱要讓人不看 implementation 也能猜到用途。
- 不為假想需求提前抽象，但要為已知的 `/exp` 新架構預留清楚接點。

### 2.2 主線優先

- 以「目前正式支援的主線」為中心重構。
- 已經退役、僅過渡期存在、或已失去公開入口的邏輯應盡量刪除。
- 不再保留「可能以後會用到」但實際無使用者、無測試、無文件契約的路徑。
- `validate` 不保留為內部 debug scaffolding，而是列為完整移除目標。
- `selection` 不保留多模式；主線只保留 `tournament`。
- `mutation` 對外不保留多 mode；只保留單一 `mutation` 入口。

### 2.3 命名必須反映責任

- `evo_ast.cpp` 這種混合「AST + 生成 + mutation + crossover」的檔案，後續應拆成更明確的責任名稱。
- `vm_gpu.cu` / `device_*` / `host_pack` 等命名應整理成一致層次，而不是歷史殘留式命名。
- CLI、bench、runtime library 的名稱應能直接反映用途，而不是靠閱讀 CMake 才知道。

### 2.4 可驗證重構

- 每一步重構都要能編譯、測試、跑 benchmark。
- 不能用一次大搬家再慢慢修的方式進行。

---

## 3. 當前問題摘要

從目前 `cpp/` 結構來看，主要問題有：

1. `src/` 同層混放太多不同責任：
   - runtime
   - evolution
   - payload
   - CLI
   - benchmark
   - GPU host orchestration
   - GPU device implementation

2. `evo_ast.cpp` / `evolve.cpp` / `builtins.cpp` / `vm_cpu.cpp` 這些檔案名稱過於粗糙，無法直接看出責任邊界。

3. `vm_cpu_cli/` 是子目錄，但 evolution CLI 相關內容卻還在 `src/` 根層，整體風格不一致。

4. `build/`、`build_release/` 放在 `cpp/` 內，讓 source tree 視覺噪音變大。

5. CMake target 與 source layout 的對應關係不夠清楚，很多 target 名稱只是結果，不是架構。

6. GPU 端檔案命名偏 implementation 細節導向，但缺乏更高層次分組。

7. 尚可能殘留：
   - 已不再對外暴露的 legacy path
   - 未被實際使用的 helper / wrapper function
   - 舊的 validate 路徑、旗標與 fallback branch
   - 非 `tournament` 的 selection mode 與其相關參數
   - hidden mutation ratio 與命名不清的 mutation 分支
   - 名稱不再符合現況的 API

8. 文件雖已改善，但目前 `cpp/` source layout 本身仍然不是自說自話的結構。

---

## 4. 重構完成後的理想狀態

重構後，`cpp/` 應該達到以下狀態：

1. 新人只看目錄名稱，就知道：
   - 哪裡是核心資料結構
   - 哪裡是 CPU runtime
   - 哪裡是 GPU runtime
   - 哪裡是 evolution
   - 哪裡是 CLI
   - 哪裡是 benchmark
   - 哪裡是實驗性 `/exp` 對接點

2. 任何一個功能變更，都能快速定位影響範圍。

3. 主線 API 名稱與文件敘述完全一致。

4. 沒有「看起來像正式支援，實際上只是歷史殘骸」的 code path。

5. 若未來 `/exp` 新架構要接入，只需要對接明確的 facade / adapter，而不是直接侵入現有主線。

6. `selection` 主線只有：
   - `tournament`
   - `selection_pressure`

7. `mutation` 主線只有：
   - `mutation`
   - `mutation_subtree_prob`

---

## 5. 建議的新目錄結構

以下是建議方向，不要求一次到位，但最終應收斂到這種責任切分：

```text
cpp/
  include/g3pvm/
    core/
    runtime/
    evolution/
    cli/
    gpu/

  src/
    core/
    runtime/
      cpu/
      gpu/
      payload/
    evolution/
    cli/
    bench/
    compat/        # 若仍需暫時保留過渡層，集中放這裡

  tests/
    runtime/
    evolution/
    gpu/
    parity/

  exp/             # 未來新架構接點，先保留空間
```

### 目錄設計說明

- `core/`
  - `Value`
  - `Bytecode`
  - `Errors`
  - 共用型別與語義工具

- `runtime/cpu/`
  - CPU VM 與 CPU builtin semantics

- `runtime/gpu/`
  - GPU host orchestration
  - kernel launch
  - device execution

- `runtime/payload/`
  - payload registry
  - payload snapshot
  - container transport 相關共用邏輯

- `evolution/`
  - genome model
  - typed generation
  - mutation
  - mutation sub-operators
  - crossover
  - tournament selection
  - selection pressure
  - fitness aggregation

- `cli/`
  - evolve CLI
  - vm CLI
  - codec/json/options

- `bench/`
  - multi bench
  - micro bench
  - profiling entry

- `compat/`
  - 如果某些 rename 或 API 搬移需要過渡層，統一收在這裡
  - 最終目標是清空

---

## 6. 分階段實作計劃

## Phase 0：盤點與凍結現況

### 目標

先把現況完整盤點清楚，避免在重構過程中誤刪主線必需路徑。

### 要做的事

1. 盤點所有 target 與其 source file 對應。
2. 盤點所有 public header 與實際被誰 include。
3. 盤點所有 CLI/public API。
4. 盤點所有未使用 function、未使用檔案、僅測試使用的 helper。
5. 建立「現況 map」文件：
   - source -> target
   - target -> 責任
   - API -> 呼叫者
6. 盤點目前所有 selection mode、參數與 callsite，作為收斂依據。
7. 盤點目前 mutation 子操作、hidden ratio 與 callsite，作為收斂依據。

### 驗收標準

1. 有一份明確清單列出：
   - 可刪除檔案
   - 可刪除 function
   - 必須保留的主線 API
   - selection 收斂後保留與淘汰的接口
   - mutation 收斂後保留與淘汰的接口
2. 不做任何行為改動，只做盤點。

---

## Phase 1：建立新目錄骨架，不改語義

### 目標

先把結構框架搭好，但暫時不改邏輯。

### 要做的事

1. 建立新的 `src/core`, `src/runtime/cpu`, `src/runtime/gpu`, `src/runtime/payload`, `src/evolution`, `src/cli`, `src/bench`。
2. 建立對應的 `include/g3pvm/...` 子命名空間結構。
3. 先以「搬檔不改行為」為主，把現有檔案移到合理位置。
4. 更新 CMake，讓 target 使用新路徑。

### 驗收標準

1. `cmake --build cpp/build -j` 可通過。
2. `ctest --test-dir cpp/build --output-on-failure` 全綠。
3. 檔案位置已大致對齊責任分類，但函式名稱可暫時不改。

---

## Phase 2：拆大檔，收斂責任邊界

### 目標

把目前責任過大的檔案拆解。

### 主要拆解對象

1. `evo_ast.cpp`
   - 拆成：
     - genome model
     - random typed generation
     - mutation facade
     - typed_subtree_mutation
     - constant_perturbation
     - crossover

2. `evolve.cpp`
   - 拆成：
     - config / run loop
     - fitness evaluation
     - tournament selection
     - selection pressure handling
     - profiling/timing aggregation

3. `vm_gpu.cu`
   - 拆成：
     - session / orchestration
     - buffer/upload
     - kernel launch
     - payload upload

4. `builtins.cpp`
   - 依 numeric/container 或 CPU exact/compact fallback 邏輯整理

### 驗收標準

1. 不再存在超級混合檔案承擔多種主責任。
2. 每個檔案名稱能直接反映用途。
3. 測試全綠。

---

## Phase 3：API 與型別命名重整

### 目標

讓 API 名稱直接反映語意，降低理解成本。

### 命名重整原則

1. 名稱應反映「責任」而非「歷史」。
2. 名稱應反映「主線語意」而非「過渡期實作」。
3. 公開 API 必須比內部 helper 更穩定、更易懂。
4. 對外 API 與內部 sub-operator 要明確分層。

### 範例方向

- `evo_ast` 這種名稱應拆成更明確的詞，例如：
  - `genome`
  - `typed_generator`
  - `mutation`
  - `crossover`

- `selection`
  - 不再保留 `roulette` / `rank` / `truncation` / `random` 這種多策略命名
  - 收斂成更明確的詞，例如：
    - `tournament_selector`
    - `selection_pressure`

- `mutation`
  - 對外只保留 `mutation`
  - 內部拆成具名 sub-operator，例如：
    - `typed_subtree_mutation`
    - `constant_perturbation`
  - 不再保留語意不明的 hidden branch

- `vm_cpu` / `vm_gpu`
  - 如果包含的不只是 VM 執行本身，應拆分成：
    - `cpu_runtime`
    - `gpu_runtime`
    - `gpu_executor`
    - `gpu_session`

- `host_pack`
  - 若職責是 bytecode/case payload packing，命名要直接寫出來

- `opcode_map`
  - 若本質是 host/device encoding bridge，命名可更準確

### 驗收標準

1. 對外名稱不再需要靠註解說明歷史背景。
2. README / docs / spec / CMake target 命名一致。
3. 不存在名稱與責任明顯不符的主線 API。

---

## Phase 4：刪除 legacy code 與未使用程式

### 目標

把真正沒必要的東西刪乾淨。

### 刪除對象

1. 已無公開入口的 legacy API
2. 僅為過渡期保留、但已不再被主線使用的 helper
3. 沒有任何 callsite 的 function
4. 與現行單一路徑策略衝突的命名或 wrapper
5. 已被新命名取代的 compatibility shim
6. 所有 validate 相關邏輯與配置：
   - `validate_genome`
   - `debug_validate`
   - validate-only fallback branch
   - 只為 validate 存在的 helper、測試支架、CLI/config 欄位
7. 非 `tournament` 的 selection mode 與其參數：
   - `roulette`
   - `rank`
   - `truncation`
   - `random`
   - `truncation_ratio`
   - 任何只為多 selection mode 存在的 parser / CLI / config 欄位
8. 隱藏 mutation ratio 與其相關舊實作：
   - 硬編碼 `0.8 / 0.2`
   - 沒有公開命名的 mutation 分支
   - 任何只為 hidden ratio 存在的 helper 或 fallback 路徑

### 刪除原則

- 沒有測試、沒有文件、沒有 callsite、沒有未來規劃依據 -> 優先刪
- 只要主線已單一路徑化，就不應保留會讓人誤以為多路徑仍受支援的 code
- 對於 validate，本計劃採取「完整移除」而不是「降級成內部 debug-only」。
- 對於 selection，本計劃採取「只留 tournament，pressure 單參數化」。
- 對於 mutation，本計劃採取「對外單一路徑，內部具名 sub-operator，ratio 顯式參數化」。

### 驗收標準

1. `rg` 搜不到已決定淘汰的 API 名稱。
2. 沒有死檔案留在 source tree。
3. 沒有只為歷史兼容而存在的對外入口。
4. `rg` 搜不到 `validate_genome`、`debug_validate`、validate-only branch 等舊 validate 路徑。
5. `rg` 搜不到 `roulette`、`rank`、`truncation`、`random` 等已淘汰 selection mode。
6. `rg` 搜不到硬編碼 mutation ratio，例如 `0.8` 對應 hidden sub-operator 選擇。

---

## Phase 5：建立 `/exp` 新架構接點

### 目標

不是現在就把 `/exp` 全做完，而是把主線整理成能夠接入 `/exp`。

### 設計要求

1. `exp/` 不能直接依賴一堆私有 implementation 細節。
2. 主線應提供清楚的 façade / adapter interface，例如：
   - fitness evaluator interface
   - genome provider interface
   - runtime execution interface
   - experiment runner interface

3. `exp/` 應能重用：
   - value/bytecode/errors
   - payload transport
   - CPU/GPU evaluator
   - benchmark/reporting utilities

### 驗收標準

1. 可以明確指出 `/exp` 要接哪一層，而不是直接 include 一堆 `src/...` 細節。
2. 有最小可行的 integration seam 設計文件或 stub interface。

---

## Phase 6：完整驗證

### 目標

確認重構沒有破壞主線能力。

### 必跑項目

1. C++ build
2. C++ tests
3. Python tests（確認跨語言契約沒壞）
4. `bouncing-balls-1024` speedup benchmark
5. `exp-1024` evolution progress benchmark
6. 至少一次 PSB2 all-task smoke run

### 驗收標準

1. `ctest` 全綠。
2. Python tests 全綠。
3. `bouncing-balls-1024` speedup 不可出現不可接受退化。
4. `exp-1024` 仍可觀察到演化進步。
5. PSB2 主線工具仍可正常跑。
6. selection 對外接口已收斂為 `tournament` + `selection_pressure`。
7. mutation 對外接口已收斂為單一 `mutation`，並以 `mutation_subtree_prob` 控制子操作比例。

---

## 7. 具體命名與模組化建議

以下是命名方向，不是強制一次全部完成，但建議在重構中逐步收斂：

### 7.1 Core

- `value.hpp/cpp`
- `bytecode.hpp/cpp`
- `errors.hpp`
- `value_semantics.hpp/cpp`

### 7.2 Runtime

- `cpu_runtime.*`
- `cpu_builtins.*`
- `gpu_runtime.*`
- `gpu_session.*`
- `gpu_executor.*`
- `gpu_kernels.*`
- `payload_registry.*`
- `payload_snapshot.*`

### 7.3 Evolution

- `genome.*`
- `typed_generator.*`
- `mutation.*`
- `typed_subtree_mutation.*`
- `constant_perturbation.*`
- `crossover.*`
- `tournament_selector.*`
- `selection_pressure.*`
- `fitness_eval.*`
- `evolution_run.*`

### 7.4 CLI / Bench

- `evolve_cli_main.cpp`
- `vm_cli_main.cpp`
- `cpu_multi_bench_main.cpp`
- `gpu_multi_bench_main.cpp`

主程式檔應明確是 `*_main.cpp`，把實際邏輯留在 library code。

---

## 8. 風險與注意事項

### 8.1 最大風險

1. 重構時誤傷 CPU/GPU parity
2. payload exact path 與 compact fallback 斷裂
3. benchmark speedup 顯著退化
4. public CLI 或工具輸出格式被不小心改掉
5. mutation operator 行為改變導致演化品質退化

### 8.2 需要特別保護的契約

1. `fitness-cases-v1`
2. binary fitness semantics
3. `typed_subtree`
4. `tournament` + `selection_pressure`
5. GPU wrapper usage
6. benchmark report JSON 格式
7. CPU/GPU parity tests
8. mutation 對外接口與顯式參數名稱

### 8.3 不應做的事

1. 為了「看起來抽象漂亮」引入過度抽象
2. 同時改語義與改結構，導致問題來源不可分辨
3. 一次搬太多檔案但沒有中間驗證點
4. 把 `/exp` 需求提前內嵌成過度通用框架

---

## 9. 建議執行順序

我建議依照下面順序做：

1. Phase 0：盤點現況
2. Phase 1：重整目錄骨架
3. Phase 2：拆大檔
4. Phase 3：rename API
5. Phase 4：刪 legacy / dead code
6. Phase 5：建立 `/exp` 接點
7. Phase 6：完整驗證

原因：

- 先盤點，才知道哪些能刪。
- 先整理目錄，再拆檔與 rename，才不會反覆搬家。
- 先讓主線清楚，再談 `/exp` 接點，避免把實驗架構建立在舊混亂層上。

---

## 10. 完成定義

當以下條件同時成立時，才算這次 `cpp/` 重構完成：

1. `cpp/` 目錄結構已按責任分層。
2. 主線 API 命名清楚一致。
3. 無已知 dead code / legacy public path 殘留。
4. 文件與 code layout 一致。
5. `/exp` 新架構已有清楚接點。
6. 測試全綠。
7. speedup 維持在可接受範圍。
8. `exp-1024` 仍有演化進步。
9. selection 主線已收斂為 `tournament` + `selection_pressure`。
10. mutation 主線已收斂為單一 `mutation` + 顯式 `mutation_subtree_prob`。

---

## 11. 建議產出物

在實作過程中，建議同步維護以下產出物：

1. `cpp/` 新架構樹狀圖
2. legacy 刪除清單
3. rename 對照表
4. `/exp` 接口說明
5. benchmark before/after 對照報告

這些產出物可以讓後續 agent 或開發者快速理解這次重構的結果與邊界。
