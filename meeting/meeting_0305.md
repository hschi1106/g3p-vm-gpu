# G3P VM GPU: CPU/GPU Evolution Flow

## 1. Overview

- Same semantics
- Same representation
- Same fitness rule
- CPU / GPU difference: evaluation backend

High-level flow:

`Genome / AST -> Bytecode -> CPU VM or GPU VM -> Fitness -> Selection/Crossover/Mutation -> Next Generation`

---

## 2. Overall structure

### 2.1 Intermediate representation: Prefix AST

`AstProgram` is the core representation of candidate programs. It is not Python source code, but a prefix AST token stream.

Benefits:

1. Easy mutation / crossover.
2. Easy to validate programs.
3. Easy to compile to bytecode.

### 2.2 Single execution contract: Bytecode ISA

VM state contains:

- `ip`: instruction pointer
- `stack`: operand stack
- `locals`: local variables
- `consts`: constant pool
- `fuel`: instruction budget

It is the same stack machine for CPU and GPU.

### 2.3 Same evolution pipeline with different CPU/GPU evaluator

Selection, crossover, and mutation stay on the host. Only the evaluation phase runs on different devices.

- Evolution orchestration on C++ host
- Program execution semantics on bytecode VM
- CPU/GPU difference on batch evaluation

### 2.4 Current grammar / language subset

The current language is a restricted Python-like subset. At the AST level, it can be summarized as:

```text
Expr ::=
    Const(v)
  | Var(x)
  | Unary(op, e)                  op ∈ {NEG, NOT}
  | Binary(op, e1, e2)            op ∈
      {ADD, SUB, MUL, DIV, MOD,
       LT, LE, GT, GE, EQ, NE,
       AND, OR}
  | IfExpr(cond, then_e, else_e)
  | Call(name, args)              name ∈ {abs, min, max, clip}

Stmt ::=
    Assign(x, e)
  | IfStmt(cond, then_blk, else_blk)
  | ForRange(x, K, body_blk)
  | Return(e)

Block ::= [s1; s2; ...; sn]
Program ::= Block
```

Important restrictions in the current grammar and semantics:

- Value domain is only `Int | Float | Bool | None`
- `Bool` is not numeric
- Conditions for `if`, `and`, `or`, and ternary must be `Bool`
- `ForRange(x, K, ...)` requires `K` to be a non-negative integer constant
- Only whitelisted builtins are allowed: `abs`, `min`, `max`, `clip`
- No containers, user-defined functions, recursion, exceptions, strings, or I/O

### 2.5 Validation before compile / evaluate

Candidate programs are not sent directly into compilation or VM execution. They first pass `validate_genome(...)`.

The current validation checks include:

- AST version must be `ast-prefix-v1`
- The prefix root must be `PROGRAM`
- Prefix structure must be well-formed, with no broken subtree boundaries
- There must be no trailing tokens after the program block
- Total node count must not exceed `max_total_nodes`
- Expression depth must not exceed `max_expr_depth`
- A block must end with `BLOCK_NIL`
- A block must not exceed `max_stmts_per_block`
- The top-level block must contain at least one `Return`
- `ForRange.k` must be non-negative
- `ForRange.k` must not exceed `max_for_k`
- `IfStmt` and `IfExpr` conditions must be `Bool`
- `AND/OR` expect `Bool`
- Arithmetic ops expect numeric operands
- Ordering comparisons expect numeric operands
- `EQ/NE` require compatible types
- Builtin arguments must be numeric
- Constant indices must be in range

Why this matters:

- Validation removes structurally broken genomes before they reach bytecode compilation
- It constrains the search space to programs that match the current subset
- It keeps CPU / GPU evaluation focused on semantic / runtime behavior rather than malformed ASTs

### 2.6 When validation happens, and what happens if it fails

Validation mostly happens during genome creation or modification, not as a full check over the whole population before every evaluation.

- `make_random_genome(...)`: validate immediately
- `crossover_* (...)`: validate the child after crossover
- `mutate(...)`: C++ only checks some structural limits, not full validation

If a program fails validation in C++:

- random generation: retry, then fall back to a simple safe program
- crossover: keep `parent_a`
- mutation: keep the original genome

So invalid programs are usually filtered out early, rather than being kept and given a low fitness later.

---

## 3. End-to-End Flow

### 3.1 Step 1: Prepare fitness cases

Prepare `fitness-cases-v1` inputs and expected outputs, then convert them into a shared case layout for batch evaluation.

### 3.2 Step 2: Initialize population

Randomly generate `1024` `ProgramGenome` instances.

Each genome contains:

- AST structure
- metadata: hash, `node_count`, `depth`

### 3.3 Step 3: Compile population

Before each evaluation, each genome is compiled into a `BytecodeProgram`.

The pipeline uses a compile cache:

- If genome hash doesn't change, reuse the compiled bytecode
- Avoid recompiling the same genome across generations

### 3.4 Step 4: Evaluate population

CPU version:

- One program
- Execute bytecode case by case
- Accumulate exact match, absolute error, and runtime error
- Calculate fitness

GPU version:

- One block for one program
- Threads in the same block process different cases
- In-block reduction / atomicAdd
- Get fitness

### 3.5 Step 5: Sorting + selection

This experiment uses `tournament selection`.

### 3.6 Step 6: Crossover + mutation + elitism

### 3.7 Step 7: Repeat generations

---

## 4. CPU evaluation mechanism

### 4.1 CPU evaluator pipeline

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

That means `1024 x 1024 = 1,048,576` program-case evaluations in one generation.

### 4.2 Execution model of CPU VM

Classic stack machine:

- `PUSH_CONST` push constant to stack
- `LOAD` load from locals
- `ADD/SUB/MUL/...` pop two values, calculate the result, and push it back
- `RETURN` take stack top as result

### 4.3 CPU timing structure

- outer total: `657004.074 ms`
- inner_total: `656837.350 ms`
- outer overhead: `166.724 ms`
- eval share in inner_total: `97.13%`

> CPU spends most of its time on evaluation, not on I/O or orchestration, but on repeatedly running the VM.

---

## 5. GPU evaluation mechanism

### 5.1 Core idea: one block for one program

- `blockIdx.x = program index`
- `threadIdx.x = case worker`

### 5.2 Move bytecode to shared memory

The kernel moves the program's bytecode from global memory to `shared_code[]`.

- In a single block, all threads repeatedly read same bytecode
- In a single block, all threads repeatedly read the same bytecode
- Accessing shared memory is cheaper than accessing global memory

> Block-shared code with case-specific threads.

### 5.3 Every thread executes different cases

Threads execute different cases in a strided pattern.

```text
for local_case = tid; local_case < case_count; local_case += blockDim.x:
    execute this program on that case
```

### 5.4 Each thread accumulates local statistics

Every thread accumulates:

- `local_exact_match_count`
- `local_runtime_error_count`
- `local_non_numeric_mismatch_count`
- `local_abs_error_sum`

`atomicAdd` to block-level shared variables:

- `block_exact_match_count`
- `block_runtime_error_count`
- `block_non_numeric_mismatch_count`
- `block_abs_error_sum`

The thread with `tid == 0` combines all statistics into a single fitness value.

- Every thread calculates local case statistics
- Whole block combines the fitness of the program

### 5.5 GPU session reuse: avoid re-uploading shared cases

`GPUFitnessSession` is initialized only once per session:

- pick GPU device
- upload shared cases
- upload expected answers

For later generations, only upload:

- programs of this generation
- corresponding consts / code / metas

---

## 6. Fitness

```text
score =
    exact_match_count
    - round(mean_abs_error)
    - 10 * runtime_error_count
    - non_numeric_mismatch_count
```

### 6.1 Meaning of each term

- `exact_match_count`
  - Number of exact matches
- `mean_abs_error`
  - Mean absolute error for numeric predictions
- `runtime_error_count`
  - Runtime errors, for example division by zero, type error, or timeout
- `non_numeric_mismatch_count`
  - Wrong non-numeric answers

---

## 7. Experiment setup and summary

Three different tasks in `logs`:

- `x + 1`
- `2x + 3`
- `x^2`

Common settings:

- population = `1024`
- generations = `40`
- seed = `0`
- selection = `tournament`
- crossover = `hybrid`
- case_count per task = `1024`

### 7.1 Experiment result

- Average end-to-end speedup: `78.790x`
- Average `inner_total` speedup: `80.868x`
- Average eval-only speedup: `188.902x`
- CPU/GPU best fitness are both `1024.0`

1. GPU accelerates the heaviest evaluation phase.
2. CPU and GPU reach the same final outcome on these tasks.

### 7.2 Experiment data

| Case | CPU outer(ms) | GPU outer(ms) | End-to-end speedup | CPU inner(ms) | GPU inner(ms) | Inner speedup | CPU eval(ms) | GPU eval(ms) | Eval speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| x_plus_1 | 154976.870 | 2817.931 | 54.997x | 154923.397 | 2742.164 | 56.497x | 151470.613 | 1143.327 | 132.482x |
| affine_2x_plus_3 | 149018.483 | 2729.410 | 54.597x | 148966.757 | 2661.038 | 55.981x | 145307.086 | 1088.608 | 133.480x |
| square_x2 | 353008.721 | 2784.526 | 126.775x | 352947.196 | 2712.365 | 130.125x | 341206.209 | 1134.544 | 300.743x |

### 7.3 Why `x^2` shows the largest speedup

`x^2` is not dramatically faster on GPU; instead, it is much slower on CPU.

- `x+1` CPU eval: `151470 ms`
- `2x+3` CPU eval: `145307 ms`
- `x^2` CPU eval: `341206 ms`

GPU eval across all three tasks stays around `1088 ms ~ 1143 ms`.

The reason is that CPU cost grows much more on heavier arithmetic tasks such as `x^2`.

> GPU strength is not only raw speed, but also slower time growth as arithmetic complexity increases.

---

## 8. Time bottleneck of GPU evaluation

- eval-only speedup: `132x ~ 300x`

> Why is end-to-end speedup only `55x ~ 127x`?

### 8.1 Fixed costs in the GPU pipeline

- `gpu_session_init_total`: `220.600 ms`
- `gpu_program_compile_total`: `1711.377 ms`
- `gpu_pack_upload_total`: `750.145 ms`
- `gpu_kernel_total`: `289.474 ms`
- `gpu_copyback_total`: `0.889 ms`

Observation:
- `kernel` is quite small
- large overhead on compile, packing, upload

> kernel is fast enough, but GPU pipeline pays for setup and data movement costs.

### 8.2 Reproduction is still on CPU

Looking at `x+1`:

- GPU `inner_eval_ms = 1143.327`
- GPU `inner_repro_ms = 1391.997`

Reproduction takes longer than evaluation.

Valuable conclusion:

> After GPU accelerates evaluation, the bottleneck shifts to host-side selection, crossover, and mutation.

For further improvement, the kernel is not the main target. The higher-priority targets are:

1. Lower program compile cost
2. Lower pack/upload cost
3. Optimize host-side reproduction
