# TODO

## GPU Speedup Recovery

### Current status

- [ ] Latest fixed-pop benchmark is not acceptable: `logs/fixture_speedup_20260313_041139/summary.md` shows `total_cpu_over_gpu = 0.976x`.
- [ ] Numeric fixtures are the main failure mode: `bouncing_balls_1024`, `simple_affine_2x_plus_3_1024`, `simple_exp_1024`, `simple_square_x2_1024`, and `simple_x_plus_1_1024` are only about `0.356x` to `0.379x` eval speedup.
- [ ] The problem is not PCIe traffic. Recent reports and `nsys` show `pack_upload_ms` and `copyback_ms` are tiny; almost all GPU eval time is in the single `evaluate_fitness` kernel.
- [ ] There is a correctness risk in addition to the speed regression: `logs/quick_depth_sweep_7/simple_x_plus_1_1024_pop2048/cpu_gpu_compare.report.md` has a GPU `mean_fitness` that does not match CPU.

### Confirmed findings

- [ ] Regressions are severe and recent. `quick_depth_sweep_5` still had strong GPU wins, `quick_depth_sweep_6` degraded, and `quick_depth_sweep_7` collapsed for numeric workloads.
- [ ] Earlier same-day results in `logs/fixture_speedup_20260313_040952/summary.md` were still healthy, so the current state looks like a real regression, not a long-standing baseline.
- [ ] CUDA build is targeting `sm_70` by default instead of this machine's `RTX 4090 (sm_89)`. See [cpp/CMakeLists.txt](/home/hschi1106/g3p-vm-gpu/cpp/CMakeLists.txt#L16).
- [ ] Kernel resource usage is unhealthy. `cuobjdump --dump-resource-usage cpp/build/libg3pvm_gpu.a` reports `evaluate_fitness` at `REG:56 STACK:5712`.
- [ ] Per-thread state is likely too large. See [execute_bytecode_device.cuh](/home/hschi1106/g3p-vm-gpu/cpp/src/runtime/gpu/device/execute_bytecode_device.cuh#L35) for `stack[64]` and `locals[64]`, plus [builtins_device.cuh](/home/hschi1106/g3p-vm-gpu/cpp/src/runtime/gpu/device/builtins_device.cuh#L17) for `DThreadPayloadState`.
- [ ] Current kernel shape is probably amplifying divergence: [kernels.cuh](/home/hschi1106/g3p-vm-gpu/cpp/src/runtime/gpu/device/kernels.cuh#L19) maps `1 program = 1 block`, then splits cases across threads in the block.
- [ ] Numeric execution goes heavily through `double`, which is a bad fit for 4090-class FP64 throughput. See [value_semantics.hpp](/home/hschi1106/g3p-vm-gpu/cpp/include/g3pvm/core/value_semantics.hpp#L33).

### Priority actions

- [ ] Rebuild CUDA for `sm_89` and rerun the fixed-pop fixture sweep to re-establish a correct baseline.
- [ ] Investigate the parity drift in `simple_x_plus_1_1024_pop2048` before taking larger optimization steps.
- [ ] Split a numeric-only fast path from the payload/container path so numeric kernels do not carry `DThreadPayloadState` by default.
- [ ] Reduce per-thread local state and stack pressure enough to eliminate the current large device stack footprint.
- [ ] Revisit kernel mapping and specialization for deep programs; current `1 block = 1 program` mapping is likely too divergence-prone for depth-7 populations.
- [ ] After each change, rerun `fixture_speedup` and CPU/GPU parity checks, with `bouncing_balls_1024` and `simple_x_plus_1_1024` as the primary regression gates.
