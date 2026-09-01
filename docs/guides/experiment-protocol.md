# G3P-VM-GPU 實驗規格

狀態：draft for pre-registration  
適用版本：執行正式實驗時鎖定單一 git commit，禁止混合不同 commit 的結果  
主要參考：`ref/ASGP.pdf`、`ref/RSGP.pdf`

## 1. 實驗範圍與論文主張

本實驗分開驗證三件事，結果不得合併成單一「GPU 比 CPU 好」的結論：

1. **語意正確性**：CPU 與 GPU fitness 是否對相同 program/cases 產生相同結果。
2. **執行效能**：GPU evaluation、GPU reproduction 與 overlap 各自帶來多少加速。
3. **演化效能**：在相同 program-evaluation budget 下，解題成功率、收斂速度與泛化能力是否維持或改善。

目前 CPU fitness backend 是單執行緒實作。因此現階段的 speedup 必須寫成：

> speedup relative to the repository's sequential CPU backend (CPU-1T)

不得寫成相對於整顆 16-core CPU 的 speedup。若論文要主張 GPU 優於「optimized CPU」，正式投稿前必須增加多核心 CPU baseline。

## 2. 與參考論文的對齊方式

ASGP 的主要設定是每題 100 runs、population 1000、最大 tree depth 7、tournament size 2、最多 500,000 NFE，並以 held-out test set 零錯誤判定成功。RSGP 同樣使用 100 runs、population 1000 與 500,000 NFE，並把 train、validation/test 分離。

本研究沿用以下可比較部分：

| 項目 | 本研究設定 | 理由 |
| --- | --- | --- |
| independent runs | 100 seeds/problem/config | 對齊 ASGP/RSGP success-count protocol |
| population | 1000 | 對齊兩篇參考論文 |
| evaluation budget | 500,000 program evaluations | 對齊兩篇參考論文 |
| selection pressure | tournament size 2 | 對齊 ASGP，亦為 repo canonical path |
| success | train solved 且獨立 test set solved | 避免把 overfitting 算成成功 |
| program limit | depth 7、total nodes 80 | depth 對齊 ASGP；nodes 是本 repo 的額外安全界線 |

以下項目不可宣稱是直接 replication：

- 本 repo 的 public crossover 固定為 `typed_subtree` 且每對 parent 都先嘗試 crossover，沒有 ASGP 的 `pc=0.7` 或 RSGP 的 `pc=0.1` 控制項。
- 本 repo 的 fitness 定義依 `spec/fitness.md`，不是 RSGP 的 MAE/PCC fitness。
- 本 repo 未執行 ASGP/RSGP 的 postprocessing simplifier；program size 僅能報 raw AST size。
- GPU reproduction 與 CPU reproduction 遵守相同 high-level operator contract，但 RNG stream 與 permutation 不要求逐步相同。

因此與 ASGP/RSGP 表格的比較只定位為「相同 benchmark family 與相近 budget 下的 contextual comparison」，不做直接優劣宣稱。

## 3. Research Questions

| ID | Research question | Primary outcome |
| --- | --- | --- |
| RQ0 | CPU/GPU 執行與 fitness 是否語意一致？ | parity mismatch count |
| RQ1 | GPU fitness evaluation 相對 CPU-1T 的 cold/warm speedup 為何？ | `S_eval`、`S_total` |
| RQ2 | GPU reproduction 與 overlap 對 end-to-end generation time 的貢獻為何？ | phase time、ablation speedup |
| RQ3 | speedup 如何隨 population、case count、program depth/node count 與 payload type scaling？ | scaling curves、geometric-mean speedup |
| RQ4 | 相同 500,000 NFE 下，不同 backend mode 是否維持相同解題能力？ | test success rate、ERT-NFE、fitness AUC |
| RQ5 | 更快的 backend 是否改善 wall-clock 解題效率？ | ERT-seconds、time-to-solution survival curve |
| RQ6 | flat、LinearRec、ASGP-DC、ASGP-DP search spaces 對解題能力與成本的影響為何？ | success、ERT、AST size、time/eval |

## 4. Backend Modes

所有效能實驗至少包含下列四種 mode：

| Mode | `--engine` | `--repro-backend` | `--repro-overlap` | 用途 |
| --- | --- | --- | --- | --- |
| `cpu` | cpu | cpu | off | sequential CPU baseline |
| `gpu_eval` | gpu | cpu | off | 隔離 GPU evaluation 效果 |
| `gpu_repro` | gpu | gpu | off | 評估 GPU reproduction |
| `gpu_repro_overlap` | gpu | gpu | on | 完整系統 |

公平性規則：

- fixed-population timing 必須重用同一個 `population-seeds` 檔案。
- `cpu` 與 `gpu_eval` 使用 CPU reproduction 時，paired seed 應產生相同 best-fitness trajectory；不一致視為 parity failure，不當作隨機波動。
- `gpu_repro` 與 `gpu_repro_overlap` 應比較分布。除非測試已證明 RNG 與 execution order 完全相同，不要求逐 seed trajectory 相同。
- 所有 mode 使用相同 cases、grammar hash、limits、fuel、penalty、population size 與 budget。

## 5. Dataset 規格

### 5.1 Performance datasets

| ID | Dataset | 用途 |
| --- | --- | --- |
| `SYN-EXP` | `data/fixtures/simple_exp_1024.json` | canonical nonlinear speed workload |
| `SYN-X1` | `data/fixtures/simple_x_plus_1_1024.json` | depth/node/payload controlled workload |
| `PSB-CORE-1024` | `data/fixtures/psb1/` 的 5 題 1024/1024 fixtures | 真實 scalar/list/string workload timing |

`PSB-CORE-1024` 包含：

- `compare-string-lengths`
- `count-odds`
- `last-index-of-zero`
- `median`
- `smallest`

選擇理由：涵蓋 Bool、Int、String、IntList；其中 `count-odds`、`last-index-of-zero`、`median`、`smallest` 也出現在 ASGP/RSGP 的比較範圍。

### 5.2 Evolution-effectiveness datasets

正式 effectiveness fixtures 必須另行 materialize，不直接使用 performance 用的 1024-case train set。

#### Core paper-aligned set

| Problem | Train | Test | Grammar/data feature |
| --- | ---: | ---: | --- |
| compare-string-lengths | 200 | 2000 | string + boolean |
| count-odds | 200 | 2000 | IntList reduction；對齊 RSGP case count |
| last-index-of-zero | 150 | 1000 | IntList/index；對齊 RSGP case count |
| median | 200 | 2000 | scalar conditional/arithmetic |
| smallest | 200 | 2000 | scalar min/conditional |

Dataset sampling seed 固定為 `20260721`。Train 優先包含 edge cases，其餘由 random split 補足；test 僅從 random split 取樣。

**硬性要求：train/test raw rows 必須 disjoint。** 目前 converter 對補入 train 的 random rows 與 test rows 是獨立抽樣，正式 materialization 前須改為 without-replacement across splits，並在 manifest 記錄 overlap count = 0。

Test fixtures 在 protocol、commit、grammar configs 與 analysis code freeze 後才能批次評估一次。Pilot 只檢查 train-side execution/schema，不查看 test fitness；若要依資料選 hyperparameter，必須另建與 train/test disjoint 的 validation split。

#### Extended PSB1 set

- 使用 `benchmarks/psb_release_exclusions.json` 所列 28 個 supported PSB1 problems。
- `replace-space-with-newline` 因 multi-output contract 尚未支援而排除。
- 統一採 train 200 / test 2000；資料不足時才採 train 100 / test 1000，並在結果表標記，不得靜默 fallback。
- 每題使用獨立 task-specific compact grammar config，以 fixture schema 限制不相關 value types。

#### Optional PSB2 extension

- 使用 exclusion manifest 所列 21 個 supported PSB2 problems。
- `coin-sums`、`cut-vector`、`find-pair`、`mastermind` 暫時排除。
- PSB2 只作 external-validity extension，不取代 PSB1 primary results。

### 5.3 Dataset manifest

每個 fixture root 必須保存：

- raw edge/random file SHA-256
- generated train/test file SHA-256
- suite、problem、schema hash、sampling seed
- train/test row count、edge/random composition
- train/test overlap count，必須為 0
- converter commit hash 與完整 command

## 6. Evolution Config

### 6.1 Primary fixed-NFE config

本 repo 每個 generation loop 評估一個 population，最後另做一次 final population evaluation。為精確得到 500,000 evaluations：

```text
population_size = 1000
generations = 499
skip_final_eval = off
NFE = population_size * (generations + 1) = 500,000
```

其餘 primary config：

```yaml
selection_pressure: 2
mutation_rate: 0.5
mutation_subtree_prob: 0.8
penalty: 1.0
fuel: 20000
max_expr_depth: 7
max_stmts_per_block: 6
max_total_nodes: 80
max_for_k: 16
max_call_args: 3
blocksize: 1024
retain_final_population: off
timing: all
seeds: 0..99
```

不得用 `8192 × 100` 的 throughput config 取代此 effectiveness config。前者是 827,392 次評估（包含 final eval），且 population dynamics 不同。

### 6.2 Grammar configs

Primary backend comparison 固定使用同一個 grammar，建議先用 task-specific `compact` profile，避免不相關 type 擴大 search space。

RQ6 另做 representation ablation；所有 config 由同一份 full base config 派生，只改 structured expression switches：

| Grammar ID | `linear_rec` | `asgp_dc` | `asgp_dp1d` | `asgp_dp2d` |
| --- | ---: | ---: | ---: | ---: |
| `G0-flat` | off | off | off | off |
| `G1-rsgp` | on | off | off | off |
| `G2-asgp-dc` | off | on | off | off |
| `G3-asgp-dp` | off | off | on | on |
| `G4-all-structured` | on | on | on | on |

隔離原則：values、ordinary expressions、builtins、limits 與 task schema 必須完全相同。每個 generated config 保存 canonical JSON 與 SHA-256。

`G1-rsgp` 與 RSGP 論文只表示「使用本 repo 的 LinearRec representation」，不是 RSGP algorithm replication。`G2/G3` 同理，不代表完整 ASGP phase-specific system replication。

## 7. 實驗矩陣

### E0：Correctness and parity gate（必做）

目的：在任何 timing/quality run 前證明結果可比較。

1. 執行全部 Python 與 native tests。
2. 執行 GPU smoke、fitness CPU/GPU parity、evolution CPU/GPU parity tests。
3. 對每個正式 fixed population 比較 CPU/GPU 每個 program 的 fitness。
4. 對 scalar、String、IntList、FloatList、StringList、錯誤、timeout 各保留至少一組 parity fixture。

通過條件：mismatch count = 0。任何 mismatch 都阻擋後續正式結果。

### E1：Fixed-population backend speedup（必做）

Dataset：`SYN-EXP` 與 `PSB-CORE-1024`。  
Population：每個 dataset 固定一份 P=4096 population。  
Modes：四種 backend mode。  
Repetitions：5 warm-up runs 不記錄，30 measured runs。  
Execution：每次以新 process 執行；mode 順序使用 seeded random permutation 交錯，避免熱度與時間漂移偏差。

主要欄位：

- `generation_total_ms[0]`
- `generation_eval_ms[0]`
- `generation_repro_ms[0]`
- `gpu_eval_init_ms`
- `generation_gpu_eval_call_ms[0]`
- `generation_gpu_eval_kernel_ms[0]`
- GPU reproduction 的 prepare/pack/upload/kernel/copyback/decode fields

報告 cold 與 warm 兩組結果：

```text
cold GPU total = generation_total_ms[0] + gpu_eval_init_ms
warm proxy      = cold total - gpu_eval_init_ms - repro_setup_ms (GPU repro only)
```

不得只報 kernel speedup；primary 是 end-to-end generation speedup。

### E2：Scaling study（必做）

使用 `SYN-EXP`，固定 grammar 與 limits。

| Dimension | Values |
| --- | --- |
| population P | 64, 256, 1024, 4096, 8192, 16384 |
| cases C | 32, 128, 512, 1024 |
| modes | cpu, gpu_eval, gpu_repro, gpu_repro_overlap |
| repeats | 30 measured/cell |

Case subsets 必須是同一個 1024-case fixture 的 deterministic prefixes，並各自保存 hash。每個 `(P,C)` cell 使用同一份 fixed population across modes。

輸出：

- latency vs P、latency vs C
- throughput = `P*C / eval_seconds`
- speedup heatmap over `(P,C)`
- GPU crossover point：95% CI 下 GPU end-to-end 確實快於 CPU 的最小 `(P,C)`

### E3：Program complexity and payload study（建議必做）

使用 current AST 重新建立 controlled populations；舊 `data/exp/*` 中的 `ast-prefix-old` artifacts 不直接當正式資料。

Modes：cpu、gpu_eval、gpu_repro、gpu_repro_overlap。

分開做三個 one-factor studies，不做難以解釋的全因子混合：

| Study | Values | Fixed controls |
| --- | --- | --- |
| depth | 5, 7, 9, 11, 13, 15 | P=4096, C=1024, matched payload mix |
| node count | 20, 30, 40, 50, 60, 70 | P=4096, C=1024, matched depth band |
| payload | none, string, list, mixed | P=4096, C=1024, matched depth/node distribution |

每個 bucket 需記錄實際 depth/node histogram，而不是只記 generator target。每 cell 3 independent populations × 10 timing repeats，以 population 作為 random effect。

### E4：Block-size sensitivity（延伸）

```text
blocksize = 128, 256, 512, 1024
P = 1024, 8192, 16384
C = 1024
modes = gpu_eval, gpu_repro, gpu_repro_overlap
```

每 cell 30 repeats。Primary config 仍固定 1024；只有 E4 可以選最佳 blocksize，不得事後把最佳值回填至其他主實驗。

### E5：Backend evolutionary effectiveness（必做）

Dataset：Core paper-aligned set。  
Config：Section 6.1。  
Grammar：固定 task-specific compact grammar。  
Modes：四種 backend mode。  
Runs：100 seeds/problem/mode，seed IDs 0..99 paired across modes。

Primary outcomes：

- test success rate
- train success rate
- generalization rate = test solved / train solved
- first-solved NFE
- first-solved wall time
- ERT-NFE、ERT-seconds
- normalized best-so-far fitness AUC

CPU 與 GPU evaluation + CPU reproduction 的 trajectory 必須相同；這一對是 semantic replication check。GPU reproduction modes 以統計分布比較。

### E6：Extended PSB1 effectiveness（必做，資源允許後執行）

先以 10 seeds 做 smoke，確認每題無 schema/runtime failure；再以 100 seeds 執行。

為控制成本，正式主比較只跑：

- `cpu` 與 `gpu_repro_overlap`：28 tasks × 100 seeds
- `gpu_eval` 與 `gpu_repro`：先跑 Core set；只有出現異常 task 才擴展

這個縮減必須在看正式結果前決定。Primary aggregate 是 macro-average per-problem success，不得用 run 數加總讓容易題支配結果。

### E7：Representation ablation（必做於演算法章節）

Modes 固定 `gpu_repro_overlap`，避免 hardware mode 與 grammar 同時變動。  
Grammars：G0-G4。  
Datasets：Core set，另加入支援時可加入 recursion-centric fixtures。  
Runs：100 seeds/problem/grammar。

報告：success、ERT、fitness AUC、raw AST nodes/depth、generation time、runtime/type error rate。

若某 grammar 無法表示某題，必須以事前 grammar reachability analysis 標記 `N/A`，不得以 0 success 混入 aggregate。

### E8：GPU reproduction ablation（延伸）

使用 `--cpu-repro-ablation none|gpu_selection|gpu_candidates|gpu_coupled_donor` 隔離 selection、candidate preprocessing、donor coupling 的效果。固定 GPU evaluation、Core set、30 seeds；只有觀察到明顯效果後再擴充 100 seeds。

## 8. Metric 定義

### 8.1 Speedup

所有 ratio 都以 paired workload repetition 計算：

```text
S_eval  = T_cpu_eval / T_gpu_eval
S_repro = T_cpu_repro / T_gpu_repro
S_total = T_cpu_total / T_gpu_total
```

報告 median speedup、IQR、95% paired bootstrap CI。跨 problem aggregate 使用 geometric mean，不使用 arithmetic mean。

`S_repro` 只用 `cpu` 對 `gpu_repro`（overlap off）估計。Overlap on 時 `generation_repro_ms` 是 evaluation 後仍可見的等待時間，不是完整 reproduction work；overlap 效果只以 `S_total` 與 phase breakdown 解釋。

### 8.2 Throughput

```text
program-case executions/s = population_size * case_count / eval_seconds
program evaluations/s     = population_size / eval_seconds
```

兩者都要報，避免不同 case count 下誤解吞吐量。

### 8.3 Solved

依 `spec/fitness.md`：

- pure numeric target fitness = 0
- pure exact-match target fitness = number of cases
- mixed target fitness = number of exact-match cases

正式 success 定義：在 budget 內曾出現 train-solved program，且該**同一個 archived AST**在 held-out test fixture 也 solved。只看最後一代不夠，test 也不得參與 selection 或 early stopping。

### 8.4 First-solved NFE and time

若 generation index 從 0 開始，generation `g` 的 population 已完成評估時：

```text
NFE_first = (g + 1) * population_size
```

final evaluation solved 時 NFE = 500,000。一般 generation 的 cold time-to-solution 定義為：

```text
T_first(g) = init_population_ms + gpu_eval_init_ms
           + sum(generation_total_ms[j], j < g)
           + generation_eval_ms[g]
```

CPU 的 `gpu_eval_init_ms` 為 0。若只有 final evaluation 才 solved，使用所有 generation total 加 `final_eval_ms`。另報排除 `gpu_eval_init_ms` 的 warm value。

### 8.5 ERT

對未成功 run 使用最大 budget 作為 censor/cost：

```text
ERT-NFE = sum(cost_i for all runs) / number_of_successful_runs
ERT-sec = sum(time_i for all runs) / number_of_successful_runs
```

成功 run 的 cost 是 first-solved cost；失敗 run 是完整 budget。若 0 success，ERT = infinity。

### 8.6 Fitness convergence

每代先轉為 best-so-far。令 target gap：

```text
gap_g = target_fitness - best_so_far_fitness_g
normalized_gap_g = gap_g / max(gap_0, epsilon)
fitness_AUC = 1 - mean(clamp(normalized_gap_g, 0, 1))
```

`fitness_AUC` 越高越好。每題先正規化再 aggregate，不直接平均 raw fitness。

### 8.7 Program quality

至少報：raw AST node count、max depth、compiled bytecode length。沒有 simplifier 前不得與 ASGP/RSGP 的 simplified size 直接比較。

## 9. 統計分析

顯著水準 `alpha=0.05`，同一 RQ 內多重比較用 Holm correction。

| Outcome | Method |
| --- | --- |
| paired timing | paired bootstrap CI + Wilcoxon signed-rank |
| paired binary success | exact McNemar test + success-rate difference CI |
| success-rate CI | Wilson 95% CI |
| censored time/NFE to solution | Kaplan-Meier curve + stratified log-rank |
| fitness AUC / AST size | paired Wilcoxon + rank-biserial effect size |
| scaling | log-log regression with P、C、mode interaction；problem/population random effects where applicable |

100 runs 的主目的不是只取得 p-value，而是估計 success probability。所有表格同時報 effect size 與 CI。

不得刪除失敗、timeout 或 fallback runs。它們要分成：runtime failure、OOM、timeout、invalid output、infrastructure failure。只有可證明與演算法無關的 infrastructure failure 才能用相同 seed 重跑，且保留原始紀錄。

## 10. Hardware and execution controls

本機 baseline inventory（正式執行時重新擷取）：

```text
CPU: AMD Ryzen 9 7950X, 16 cores / 32 threads
RAM: 61 GiB
GPU: 2 x NVIDIA GeForce RTX 4090, 24 GiB each, compute capability 8.9
Current CUDA compiler: 12.2
```

正式 timing protocol：

1. 使用 Release build：`cmake -DCMAKE_BUILD_TYPE=Release`。
2. 固定單一 GPU：`G3PVM_CUDA_DEVICE=0`；另一張 GPU 在 timing 期間保持 idle。
3. 每次只跑一個 measured process，不與其他 GPU/CPU-heavy job 共存。
4. 記錄 GPU UUID、driver、CUDA、power limit、temperature、clock、CPU governor、kernel、compiler versions。
5. GPU 開始 measured runs 前做 5 次 warm-up，但 cold metric 仍由每個新 process 的 `gpu_eval_init_ms` 計算。
6. CPU-1T timing 可 pin 至固定 physical core；若新增 CPU-multicore baseline，另記 thread count 與 affinity。
7. 不得混用 Debug/Release、不同 driver、不同 GPU 或不同 git commit。
8. GPU profiling 只使用 `nsys`；profiling runs 與 timing runs 分開，profiled timing 不列入主表。

## 11. 執行命令範本

### 11.1 Release build and gates

```bash
cmake -S cpp -B cpp/build_release -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build_release -j

PYTHONPATH=python python3 -m unittest discover -s python/tests -p 'test_*.py' -v
ctest --test-dir cpp/build_release --output-on-failure
```

### 11.2 Fixed population

```bash
python3 tools/make_population_seeds.py \
  --cases data/fixtures/simple_exp_1024.json \
  --count 4096 \
  --start-seed 0 \
  --grammar-config configs/grammar/all.json \
  --out logs/experiment/fixed/simple_exp_p4096.seeds.json
```

```bash
G3PVM_CUDA_DEVICE=0 cpp/build_release/g3pvm_evolve_cli \
  --cases data/fixtures/simple_exp_1024.json \
  --population-json logs/experiment/fixed/simple_exp_p4096.seeds.json \
  --grammar-config configs/grammar/all.json \
  --engine gpu \
  --repro-backend gpu \
  --repro-overlap on \
  --blocksize 1024 \
  --generations 1 \
  --skip-final-eval on \
  --timing all \
  --out-json logs/experiment/timing/run.json
```

### 11.3 Effectiveness sweep

逗號分隔 seeds 應由 driver 產生 `0..99`，以下以 shell 展示其值：

```bash
SEEDS=$(seq -s, 0 99)
G3PVM_CUDA_DEVICE=0 python3 tools/run_psb_regression.py \
  --suite psb1 \
  --profile compact \
  --cases-root data/fixtures/experiment/psb1-core-paper \
  --problems compare-string-lengths,count-odds,last-index-of-zero,median,smallest \
  --seeds "$SEEDS" \
  --binary cpp/build_release/g3pvm_evolve_cli \
  --base-grammar-config configs/grammar/all.json \
  --engine gpu \
  --repro-backend gpu \
  --repro-overlap on \
  --population-size 1000 \
  --generations 499 \
  --selection-pressure 2 \
  --mutation-rate 0.5 \
  --mutation-subtree-prob 0.8 \
  --fuel 20000 \
  --eval-test \
  --out-dir logs/experiment/effectiveness/gpu_repro_overlap
```

## 12. Artifact and naming contract

建議 run root：

```text
logs/experiment/<study_id>/<commit>/<hardware_id>/<mode>/<problem>/<seed_or_repeat>/
```

每次 experiment batch 必須有：

- `manifest.json`：研究 ID、commit、dirty status、hostname、hardware/toolchain、完整 config
- `dataset_manifest.json`：fixture hashes 與 split audit
- `environment.txt`：`nvidia-smi`、`lscpu`、compiler/CUDA/CMake versions
- `run.json`、stdout、stderr、完整 command
- `summary.json`：raw-to-summary script version 與統計結果
- `failures.json`：所有非成功 process 與處理決策

結果圖至少包含：

1. end-to-end speedup by mode/problem
2. `(P,C)` scaling heatmap
3. eval/reproduction phase breakdown stacked plot
4. test success rate with Wilson CI
5. Kaplan-Meier time-to-solution curve
6. best-so-far normalized fitness curve
7. grammar ablation success/ERT plot

## 13. 正式實驗前的阻擋項目

下列項目未完成前，資料只能算 pilot：

1. `run_psb_regression.py` 目前以 final population best 判定成功，未保存 best-ever AST、first-solved generation/NFE/time；需補齊後才能正確計算 success 與 ERT。
2. evolution loop 目前不 early-stop。可維持 fixed full budget，但需保存 first-solved event；若實作 early-stop，所有 mode 必須同時使用相同規則。
3. PSB converter 需保證 train/test random rows 跨 split 不重複，manifest 必須驗證 overlap=0。
4. regression runner 應明確轉送並記錄 `--penalty` 與所有 `--max-*` limits，不能只依賴 CLI defaults。
5. controlled depth/node populations 必須以 current AST/grammar 重新 materialize 並通過 replay check。
6. 若論文主張相對於一般 CPU 的 speedup，需新增可重現的 CPU-multicore baseline；否則全文固定標示 CPU-1T。
7. 正式 run 前凍結 commit、Release binary SHA-256、dataset/config hashes 與 analysis script。

## 14. Go/No-Go criteria

正式實驗開始條件：

- 全部 correctness/parity tests 通過
- dataset overlap = 0，schema/runtime compatibility 全數通過
- 10-seed pilot 無 unexplained failure
- timing CV 在穩定 cell（P=4096, C=1024）低於 5%；否則先處理系統噪音
- raw artifact 可由 analysis script 重建所有 summary fields

結果接受條件不預設「GPU 必須更快或更會解題」。主要結論依預先定義的 CI、effect size 與 failure analysis；negative result 仍保留並報告。

## 15. 建議執行順序與規模

| Phase | Experiments | Formal measured runs（不含 warm-up/pilot） |
| --- | --- | ---: |
| A | E0 + E1 | 720 = 6 datasets × 4 modes × 30 |
| B | E2 | 2880 = 6 P × 4 C × 4 modes × 30 |
| C | E3 | 1920 = 16 buckets × 3 populations × 4 modes × 10 |
| D | E5 | 2000 = 5 problems × 4 modes × 100 seeds |
| E | E6 primary pair | 5600 = 28 problems × 2 modes × 100 seeds |
| F | E7 | 2500 = 5 problems × 5 grammars × 100 seeds |

E4、E8 與 PSB2 為延伸實驗，不計入上表。每個 phase 完成後先做完整 artifact audit，再開始下一 phase；不得在看到 phase D/E 的 test outcomes 後修改 primary config。
