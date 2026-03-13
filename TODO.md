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

### 2026-03-13 optimization pass result

- [x] Rebuilt default CUDA target for `sm_89` and verified the binary is no longer built for `sm_70`.
- [x] Restored exact CPU/GPU parity for the known payload-slice drift cases before landing performance work.
- [x] Split GPU evaluation into two internal paths:
  one kernel specialization for programs that do not need exact payload handling, and one for programs that do.
- [x] Raised the native GPU blocksize default from `256` to `1024` for the current `1024-case` benchmark shape.

### 2026-03-13 measured outcome

- [x] Baseline: `logs/opt_baseline_head_58b84c4/summary.md`
- [x] Final: `logs/opt_final_head_defaults1024/summary.md`
- [x] Average `eval_cpu_over_gpu` improved from `1.770x` to `4.715x` (`2.66x` relative improvement).
- [x] Average `total_cpu_over_gpu` improved from `1.665x` to `3.979x` (`2.39x` relative improvement).
- [x] Numeric fixtures recovered from GPU-losses to clear GPU wins:
  `simple_x_plus_1_1024 0.381x -> 1.440x`,
  `simple_affine_2x_plus_3_1024 0.383x -> 1.436x`,
  `simple_square_x2_1024 0.378x -> 1.439x`,
  `simple_exp_1024 0.395x -> 1.440x`,
  `bouncing_balls_1024 0.380x -> 1.378x`.

### 2026-03-13 design changes and tradeoffs

- [x] Semantics were kept unchanged; parity tests still pass after the optimization pass.
- [x] GPU evaluation is no longer a single monolithic kernel launch for all programs.
  It now partitions programs on the host and launches up to two subset kernels.
- [x] The no-payload specialization is materially lighter than the payload specialization.
  Current `cuobjdump` resource usage is `REG:48 STACK:2160` for no-payload and `REG:64 STACK:5808` for payload.
- [x] The new `1024` default is deliberately tuned for the current benchmark shape and this machine class.
  It may not be the best default for smaller GPUs or different case counts, so callers may still need to override `--blocksize`.
- [x] The subset split adds some host-side classification work and an extra kernel launch in mixed populations.
  This slightly increases pack/upload and launch overhead, but the kernel-time reduction is much larger on the current workloads.
