# G3P VM GPU: CPU/GPU Evolution Flow

## 1. Overview

- Same semantic
- Same representation
- Same fitness rule
- CPU / GPU difference: evaluation backend

high-level flow: 

`Genome / AST -> Bytecode -> CPU VM or GPU VM -> Fitness -> Selection/Crossover/Mutation -> Next Generation`

---

## 2. Overall structure

### 2.1 Intermidiate representation: Prefix AST

 `AstProgram` as the core representation of candidate programs.Not Python source code, but prefix AST token stream.

Benefits:

1. Easy mutation / crossover.
2. Easy to validate programs.
3. Easy to compile to bytecode

### 2.2 Single execute rules: Bytecode ISA

VM states contains:

- `ip`: instruction pointer
- `stack`: operand stack
- `locals`: local variables
- `consts`: constant pool
- `fuel`: instruction budget

Same stack machine for CPU / GPU.

### 2.3 Same evolution pipeline with different CPU/GPU evaluator

Selection、crossover、mutation stay's on host. Only evaluation phase on different devices.

- Evolution orchestration on C++ host
- Program execution semantics on bytecode VM
- CPU/GPU difference on batch evaluation

---

## 3. Experiment setup and summary

Three different tasks in `logs`: 

- `x + 1`
- `2x + 3`
- `x^2`

Common setting:

- population = `1024`
- generations = `40`
- seed = `0`
- selection = `tournament`
- crossover = `hybrid`
- 每題 case_count = `1024`

### 3.1 Experiment result

- Average end-to-end: `78.790x`
- Average inner_total:`80.868x`
- Average eval-only: `188.902x`
- CPU/GPU best fitness are both: `1024.0`

1. GPU accelerates the most heavy evaluation phase.
2. Same outcome between CPU / GPU.

### 3.2 Experiment data

| Case | CPU outer(ms) | GPU outer(ms) | End-to-end speedup | CPU inner(ms) | GPU inner(ms) | Inner speedup | CPU eval(ms) | GPU eval(ms) | Eval speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| x_plus_1 | 154976.870 | 2817.931 | 54.997x | 154923.397 | 2742.164 | 56.497x | 151470.613 | 1143.327 | 132.482x |
| affine_2x_plus_3 | 149018.483 | 2729.410 | 54.597x | 148966.757 | 2661.038 | 55.981x | 145307.086 | 1088.608 | 133.480x |
| square_x2 | 353008.721 | 2784.526 | 126.775x | 352947.196 | 2712.365 | 130.125x | 341206.209 | 1134.544 | 300.743x |

### 3.3 Why `x^2` is so fast

`x^2` is not fater on GPU, but much more slower on CPU.

- `x+1` CPU eval: `151470 ms`
- `2x+3` CPU eval: `145307 ms`
- `x^2` CPU eval: `341206 ms`

GPU eval around three problems: `1088 ms ~ 1143 ms`

Why: CPU cost more on heavy arithmetic tasks like `x^2`.

> GPU strength: not only fast, but the time growth with heavy arithmetic tasks is less than CPU.

---

## 4. End-to-End Flow

### 4.1 Step 1: Prepare fitness cases

### 4.2 Step 2: Initialize population

Randomly produce `1024` `ProgramGenome`。

Every genome contains

- AST structure
- metadata: hash、node_count、depth

### 4.4 Step 3: compile population

Before every evaluation, genome will be compile to `BytecodeProgram`。

Utilize compile cache：

- If genome hash doesn't change, reuse the compiled bytecode
- Avoid re-compile on same genome in different generaions.

### 4.5 Step 5: evaluate population

CPU version: 

- One program
- Execute bytecode case by case
- Accumulates exact match、abs error、runtime error
- Calculate fitness

GPU version: 

- One block for one program
- Threads in same block share different cases
- In-block reduction / atomicAdd
- Get fitness

### 4.6 Step 6: sorting + selection

Use `tournament selection` in this experiment

### 4.7 Step 7: crossover + mutation + elitism

### 4.8 Step 8: Repeat generations

---

## 5. CPU's evaluation mechanism

### 5.1 CPU evaluator's pipeline

```text
for each program:
    exact_match = 0
    error_count = 0
    abs_error_sum = 0
    for each case:
        run bytecode VM once
        compare result with expected
        accumulate fitness statistics
```

`1024 x 1024 = 1,048,576` times of program-case evaluation in one generation.

### 5.2 Execution model of CPU VM

Classic stack machine：

- `PUSH_CONST` push constant to stack
- `LOAD` load from locals
- `ADD/SUB/MUL/...` pop two values from stack pop, calcuate it and push back.
- `RETURN` take stack top as result

### 5.3 CPU timing structure

- outer total: `657004.074 ms`
- inner_total: `656837.350 ms`
- outer overhead: `166.724 ms`
- eval share in inner_total: `97.13%`

> CPU spends most off its time on evaluation, not I/O nor orchestration, but VM.

---

## 6. GPU's evaluation mechanism

### 6.1 Core idea: one block for one program

- `blockIdx.x = program index`
- `threadIdx.x = case worker`

### 6.2 Move bytecode to shared memory

kernel move program's bytecode from global memory to `shared_code[]`.

- In a single block, all threads repeatedly read same bytecode
- Lower access cost from shared memory than global memory

> block-shared code with case-specific threads

### 6.3 Every thread execute different cases

Threads execute different cases in stride approach.

```text
for local_case = tid; local_case < case_count; local_case += blockDim.x:
    execute this program on that case
```

### 6.4 thread accumulate local statistics

Every thread accumulates:

- `local_exact_match_count`
- `local_runtime_error_count`
- `local_non_numeric_mismatch_count`
- `local_abs_error_sum`

`atomicAdd` to block-level shared variables：

- `block_exact_match_count`
- `block_runtime_error_count`
- `block_non_numeric_mismatch_count`
- `block_abs_error_sum`

Thread with `tid == 0` pack all statistics to a single fitness value.

- Every thread calculate local case statistics
- Whole block combines the fitness of the program

### 6.5 GPU session reuse: avoid re-upload shared cases

`GPUFitnessSession` will be launch only at initialize session

- pick GPU device
- upload shared cases
- upload expected answers

For the upcoming generations, only upload:

- programs of this generation
- corresponding consts / code / metas

---

## 7. Fitness

```text
score =
    exact_match_count
    - round(mean_abs_error)
    - 10 * runtime_error_count
    - non_numeric_mismatch_count
```

### 7.1 Meanings

- `exact_match_count`
  - Number of right answers
- `mean_abs_error`
  - MAE of wrong answwer
- `runtime_error_count`
  - runtime error. eg: div by zero, type error, timeout
- `non_numeric_mismatch_count`
  - wrong non-numeric answer

---

## 8. Time bottleneck of GPU evaluation

- eval-only speedup: `132x ~ 300x`

> end-toend: `55x ~ 127x`?

### 8.1 Fix cost in GPU pipeline

- `gpu_session_init_total`: `220.600 ms`
- `gpu_program_compile_total`: `1711.377 ms`
- `gpu_pack_upload_total`: `750.145 ms`
- `gpu_kernel_total`: `289.474 ms`
- `gpu_copyback_total`: `0.889 ms`

Observation:
- `kernel` is quite small
- large overhead on compile, packing, upload

> kernel is fast enough, but GPU pipeline pays for setup and data movement costs.

### 9.2 reproduction is still in CPU

Looking at `x+1`：

- GPU `inner_eval_ms = 1143.327`
- GPU `inner_repro_ms = 1391.997`

Reproduction takes longer time than evaluation.

Valuable conclusion:

> After GPU's acceleration on evaluation, bottleneck transfers to host side selection / crossover / mutation.

For further improvement, kernel cost is not the major part, but:

1. Lower program compile cost
2. Lower pack/upload cost
3. Optimize host-side reproduction
