# exp

這個目錄只保留目前的 GPU reproduction current benchmark。

## 一鍵執行

先確認 `cpp/build` 已經存在並且有：

- `libg3pvm_cpu.a`
- `libg3pvm_gpu.a`

如果還沒建過，先在 repo root 跑：

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Debug
cmake --build cpp/build -j
```

之後直接跑：

```bash
bash exp/run_current.sh
```

這會做三件事：

1. 編譯 `exp/repro_proto_bench.cu` 到 `exp/bin/g3pvm_repro_proto_bench`
2. 透過 GPU wrapper 重跑 `population=1024` 與 `4096`
3. 重新產生 `exp/results/complete_summary.md`

## 目錄說明

- `repro_proto_bench.cu`: current benchmark 本體
- `build_repro_proto_bench.sh`: bench 編譯腳本
- `run_current.sh`: 一鍵重跑入口
- `write_current_summary.py`: 從 JSON 生成 markdown summary
- `results/`: current benchmark JSON 與 summary
