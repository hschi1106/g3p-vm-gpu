# GPU reproduction prototype 實驗構想

## 1. 目的

這份實驗的目標不是重做整個 evolution pipeline，也不是立刻把完整 reproduction 流程搬上 GPU。

目前只想驗證三件事：

1. CPU 端進行 subtree 切分與 donor 生成是否費時
2. 把 prefix AST 與相關 metadata 打包搬上 GPU 是否費時
3. GPU 上進行 selection + crossover / mutation 是否比 CPU 快

前兩點的目的，是確認這些 CPU 與搬運工作是否能趁 GPU evaluation 進行時偷偷完成，從而被 overlap 掉。

第三點的目的，是確認即使 reproduction 的流程與原始版本不完全一樣，GPU 是否仍可能在這類結構操作上帶來速度優勢。


## 2. 目前的假設

### 2.1 不追求完全重現現有 reproduction

這個 prototype 不要求 100% 重現目前 CPU 版的 reproduction 行為。

只要滿足以下方向即可：

- selection 邏輯合理
- mutation / crossover 是合法的 prefix AST 操作
- 行為與現有流程相近
- 可以公平地與 CPU 版對照時間

因此，這個 prototype 比較像：

`GPU selection + GPU variation feasibility study`

而不是：

`full GPU evolution implementation`


### 2.2 selection 與 crossover / mutation 可以採用簡化版

目前可以接受的簡化包括：

- selection 先使用容易 GPU 化的方法，例如 tournament selection
- mutation 比例固定
- 剩餘 child 走 crossover
- crossover 的 subtree 不必完全仿照目前 typed subtree crossover 的所有細節
- 只要仍是從 candidate subtree 中抽樣並做 prefix slice replacement 即可

我目前的假設是：

即使這個流程與原始 CPU reproduction 有差異，只要差異不大，仍足以拿來驗證速度潛力。


## 3. 要驗證的三個問題

## 3.1 CPU subtree preprocessing 成本

想量的事情包括：

- 為每個 program 建立 `subtree_end`
- 為每個 program 選出固定數量 `K` 個 candidate subtrees
- 為 mutation 預先生成若干 donor subtrees

想回答的問題：

- 這些 CPU 預處理本身要花多少時間？
- 與 GPU evaluation 的時間相比，是否有機會完全藏在 evaluation 期間？

如果答案是可以 overlap 掉，那之後就有理由採用：

- GPU evaluation 跑的同時
- CPU 偷偷做下一輪 reproduction 前處理


## 3.2 host-to-device staging 成本

想量的事情包括：

- prefix AST program buffer 打包時間
- candidate subtree table 打包時間
- mutation donor pool 打包時間
- host-to-device copy 時間

想回答的問題：

- 搬運 prefix AST 與 metadata 到 GPU 的成本高不高？
- 這些成本是否也能和 GPU evaluation overlap？

如果這段搬運時間本身不小，但又能被 evaluation 掩蓋，那麼它在 end-to-end pipeline 中就不一定構成真正瓶頸。


## 3.3 GPU selection + crossover / mutation 成本

想量的事情包括：

- GPU selection
- GPU mutation
- GPU crossover

想回答的問題：

- 在簡化流程下，GPU 做 selection + variation 是否比 CPU 快？
- 即使 GPU 版不是完整 reproduction，是否已能看出明顯速度潛力？


## 4. 我目前想採用的 prototype 流程

### 4.1 evaluation 期間的 CPU 工作

當 GPU 正在做 bytecode evaluation 時，CPU 預先做：

- 對每個 prefix AST program 計算 `subtree_end`
- 挑出固定數量 `K` 個 candidate subtrees
- 生成 mutation donors
- 打包 program 與 metadata

這一段的核心目標不是最佳化演算法品質，而是量測：

- 這些工作到底要花多久
- 是否能被 evaluation 時間掩蓋


### 4.2 evaluation 期間的搬運

如果資料結構允許，則在 evaluation 期間同步進行：

- prefix AST 搬上 GPU
- candidate subtree table 搬上 GPU
- donor pool 搬上 GPU

這一段同樣要量：

- 純打包時間
- 純 copy 時間
- 是否值得使用 stream overlap


### 4.3 evaluation 結束後的 GPU reproduction prototype

evaluation 結束後，在 GPU 上做：

1. selection
2. child 類型決定
   - 一定比例走 mutation
   - 剩下走 crossover
3. launch 一批 blocks
   - 每個 block 處理一個 child
4. block 根據已上傳的 candidate subtree / donor 資料做：
   - subtree replacement
   - subtree swap
   - 或較簡單的 mutation

這裡不追求完整模擬現行 CPU reproduction，而是先測：

`在 prefix AST 的限制下，GPU 做這類 variation 是否有速度優勢`


## 5. selection 的暫定方案

selection 不打算一開始就做太複雜的版本。

暫定優先考慮：

- tournament selection

原因：

- 容易 GPU 化
- 不需要 prefix sum
- 不需要全域 scan
- 與目前 repo 的 selection 設定接近

因此 prototype 目標不是研究 selection 演算法本身，而是：

`找一個足夠快且足夠合理的 GPU selection 作為前置步驟`


## 6. 與原始 reproduction 的差異

這個 prototype 與原始 CPU reproduction 可能存在差異：

- mutation / crossover 的選點規則較簡化
- typed subtree 的細節不一定完全一致
- current benchmark 不含 fallback / validation；這部分不再是這份實驗要回答的問題
- mutation 與 crossover 比例可能固定

但這些差異目前是可以接受的，因為這份實驗想回答的問題不是 search quality，而是：

- CPU 前處理能否被藏起來
- 搬運能否被藏起來
- GPU variation 單獨看是否值得


## 7. 實驗輸出應該記錄什麼

建議至少記錄以下時間：

- CPU `subtree_end` 計算時間
- CPU candidate subtree 選取時間
- CPU donor 生成時間
- 打包時間
- H2D copy 時間
- GPU selection 時間
- GPU mutation / crossover kernel 時間
- CPU 對照版 selection + mutation / crossover 時間

建議另外記錄：

- population size
- 平均 program node 數
- candidate subtree 數量 `K`
- donor pool 大小
- mutation 比例


## 8. 建議的對照方式

### 8.1 CPU 對照組

CPU 對照組不必強迫使用完整正式 pipeline。

只要使用與 GPU prototype 近似的規則即可，例如：

- 一樣使用固定數量 candidate subtree
- 一樣使用固定 mutation 比例
- 一樣使用預生成 donor

這樣比較的是：

`同一個簡化 reproduction 任務，CPU 與 GPU 哪個快`

而不是：

`GPU prototype 是否完全重現正式 evolution`


### 8.2 GPU 對照重點

真正要比較的不是單一 kernel 是否快，而是：

- CPU preprocessing 是否能被 evaluation 掩蓋
- H2D staging 是否能被 evaluation 掩蓋
- GPU selection + variation 本身是否快於 CPU

如果三者同時成立，那就代表：

`把 prefix AST reproduction 的部分步驟搬上 GPU 是有潛力的`


## 9. 風險

目前主要風險包括：

- CPU subtree preprocessing 本身可能已經太重
- 打包與搬運可能太花時間
- GPU 端 child construction 可能被 variable-length prefix slice 卡住
- 為了方便 GPU 而做的簡化，可能使結果與正式 reproduction 差太多
- 即使 GPU reproduction 快，compile 可能仍然變成新的主要瓶頸


## 10. 目前結論

這個實驗的重點不是立刻證明 full GPU reproduction 可行，而是先回答：

1. CPU 前處理能不能被藏在 evaluation 期間
2. 搬運能不能被藏在 evaluation 期間
3. GPU selection + variation 單獨拿出來看，有沒有速度優勢

如果答案是正面的，才值得繼續往更完整的 GPU reproduction pipeline 推進。

因此，這個 prototype 的定位應該是：

`對 prefix AST reproduction 進行 GPU 化的可行性與成本驗證`
