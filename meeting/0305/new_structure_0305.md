# New GPU Reproduction Flow

## 1. Main idea

The goal is not to move the whole evolution pipeline to GPU at once.

The idea is simpler:

- keep the complicated correctness-sensitive steps on CPU
- move the repetitive, highly parallel reproduction work to GPU
- only send promising children back to CPU for final checking

So this is a **hybrid CPU/GPU pipeline**, not a full GPU rewrite.

---

## 2. What CPU does

CPU keeps the parts that are harder to parallelize and more sensitive to correctness:

- subtree preprocessing
- candidate collection
- donor preparation
- host-side packing
- full `validate_genome()`
- compile
- fallback decision

In short:

> CPU is the control and correctness side.

---

## 3. What GPU does

GPU handles the parts that are repeated many times across many children:

- tournament selection
- crossover / mutation
- cheap validate

In short:

> GPU is the high-throughput child-generation side.

---

## 4. Full generation flow

For one generation, the flow is:

1. GPU evaluates the current population.
2. CPU prepares the inputs for the next generation.
3. GPU runs selection.
4. GPU generates children with crossover / mutation.
5. GPU runs a cheap validation pass.
6. Only cheap-pass children are copied back to CPU.
7. CPU runs full `validate_genome()`.
8. Valid children are compiled.
9. Invalid children fall back to parent.

The important point is:

- GPU does the fast first pass
- CPU does the final exact check

---

## 5. Why this split makes sense

There are two main reasons.

### 5.1 Full validation is still better on CPU

Full validation is not a tiny check. It includes:

- AST structure checks
- block / expression checks
- type checks
- return-path checks
- range and index checks

That logic already exists on CPU and is easier to debug there.

### 5.2 Compile is also still better on CPU

Compile is tightly connected to the host-side AST representation.

Moving compile to GPU now would make the change much larger and riskier.

So the current design keeps:

- **GPU for fast filtering and child generation**
- **CPU for exact validation and compile**

---

## 6. Simple CPU/GPU timeline

```text
Time  ------------------------------------------------------------->

GPU:   [ evaluate current population ] [ selection + variation + cheap validate ]

CPU:   [ prepare next-gen inputs     ]                    [ full validate + compile ]

Flow:  current pop ----eval----> ranked pop ----GPU child generation----> cheap-pass children
                                                                  |
                                                                  v
                                                      CPU final check / compile / fallback
```

---

## 7. Why we think this can help: evidence from the earlier prototype work

The earlier prototype already tested the main pieces of this idea.

The important result is not that the whole pipeline is solved already.  
The important result is that the experiments show **where GPU helps** and **where the new bottleneck appears**.

Prototype labels used below:

- `proto`: raw GPU selection + variation only
- `pairwise`: tree-pair child generation with DtoH included
- `final`: `pairwise` plus formal remap and overlap with evaluation
- `validate`: `final` plus full validate / fallback
- `cheap-validate`: `validate` plus GPU cheap validate prefilter

### 7.1 Selection + variation is a strong GPU target

The table below measures only the selection + variation part:

`cpu_sel+var_ms / gpu_sel+var_ms`

| Version | pop=1024 | pop=4096 |
| --- | ---: | ---: |
| `proto` | 61.2x | 206.5x |
| `pairwise` | 60.5x | 178.3x |
| `final` | 60.6x | 193.9x |

This shows one clear point:

> the child-generation kernel itself is already very fast on GPU.

### 7.2 Copy-back reduces the gain, but GPU is still ahead

The next table adds DtoH child copy-back:

`cpu_sel+var_ms / gpu_sel+var+d2h_ms`

| Version | pop=1024 | pop=4096 |
| --- | ---: | ---: |
| `pairwise` | 13.3x | 17.0x |
| `final` | 19.3x | 26.9x |

This tells us:

- DtoH is a real cost
- but GPU reproduction is still clearly faster than CPU for this part

So the architecture direction is:

> only copy back cheap-pass children

### 7.3 Host preprocessing can be hidden behind GPU evaluation

The next table measures overlap benefit directly:

`sequential_wall_ms / overlap_wall_ms`

| Version | pop=1024 | pop=4096 |
| --- | ---: | ---: |
| `final` | 1.281x | 1.603x |

This supports a different design choice:

> keep preprocessing on CPU for now, and overlap it with GPU evaluation instead of moving it first.

### 7.4 The real bottleneck is now validation

Validation dominates the runtime:

- GPU validate share: `99.3%` at pop=1024, `99.4%` at pop=4096
- CPU validate share: `93.9%` at pop=1024, `93.3%` at pop=4096

This is the strongest argument for the new hybrid structure:

> GPU child generation is already fast enough. The next useful step is to reduce validation work, not to keep tuning the variation kernel.

### 7.5 Cheap validate already shows a measurable effect in the comparable on/off test

To judge cheap validate itself, the more reliable numbers are the dedicated **cheap on/off comparable** experiment.

GPU-side total cost:

| population | cheap off | cheap on | improvement |
| --- | ---: | ---: | ---: |
| 1024 | `40.708 ms` | `35.726 ms` | `1.139x` |
| 4096 | `147.971 ms` | `128.833 ms` | `1.149x` |

Overall GPU-vs-CPU speedup:

| population | cheap off | cheap on |
| --- | ---: | ---: |
| 1024 | `1.022x` | `1.171x` |
| 4096 | `1.075x` | `1.236x` |

This is important because it directly supports the proposed flow:

- GPU cheap validate is cheap
- filtering invalid children early helps
- sending only cheap-pass children back to CPU should reduce wasted work further

### 7.6 Practical conclusion

The experiments support three claims:

1. GPU is a strong target for selection + variation.
2. CPU preprocess can stay on host for now and still be useful with overlap.
3. The next meaningful gain comes from reducing full validation / fallback work.

So this new structure is not just a theoretical design.  
It is a direct response to the bottlenecks already observed in the earlier prototype work.
