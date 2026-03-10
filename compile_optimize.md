1. 先把 compile 改成平行化。
    目前 compile_population() 是單執行緒逐個編譯 evolve.cpp:76。這段是天然 embarrassingly parallel，先收集 cache miss，
    再用 thread pool / OpenMP 平行 compile miss 的 genome，通常是最直接的 ROI。
2. 做 exact dedup，不只是 compile cache。
    現在 cache 只是在 hash hit 時重用 BytecodeProgram evolve.cpp:82，但重複 genome 仍然會被重複 pack、重複 eval。更好的
    做法是每代先 dedup exact genome，只 compile/eval unique set，再把 fitness scatter 回原 population。這會同時省
    compile、pack、kernel、copyback。
3. 把 eval-only bytecode 從 string 版改成 compact 版。
    現在 Instr.op 是 std::string bytecode.hpp:10，compiler emit 的也是字串 compiler.cpp:99，GPU pack 時又把字串轉回
    opcode host_pack.cu:31、opcode_map.cpp:7。這是雙重浪費。最值得的是拆一個 EvalBytecodeProgram：

- opcode 改 enum / uint8_t
- label 改 int，不要 string label
- temp local 不要 string name
- eval path 不要帶 var2idx

4. 強化 hash / cache key，再更依賴 reuse。
    現在 hash_key 是 sampled structural hash，不是完整 canonical hash genome_meta.cpp:43。目前拿來做輕量 cache 還行，但
    如果你要做更積極的 dedup / reuse，我會先改成：

- full structural hash，或
- fast hash + canonical verify on hit
不然你之後會很難安心把更多邏輯壓在這個 key 上。