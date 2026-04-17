# NewKernelBench

`NewKernelBench` is a benchmark and diagnosis workspace for LLM-driven kernel generation. It starts from the original `KernelBench` task set, adds a richer planning layer, and pushes toward a more realistic evaluation loop for real workloads.

## What is new compared with the original KernelBench

`NewKernelBench` adds several things that the original benchmark does not organize as first-class concepts:

- multi-axis task planning instead of level-only grouping
- dtype-aware variants, including `fp32`, `fp16`, `bf16`, and selected `int8` cases
- layout-aware variants, including contiguous, channels-last, transposed, and non-contiguous cases
- execution-mode variants: forward, backward, and forward-plus-backward
- tiered analysis planning so heavy profiling is only used when justified
- unified result bundles for correctness, gradient correctness, speedup, and failure taxonomy
- manual harvesting scripts for expanding the benchmark from mainstream libraries
- manual profiling scripts for `torch.profiler`, CUDA memory, and `ncu`
- a full runner that executes a seed task and a candidate implementation under a chosen variant configuration

## Current project layout

```text
NewKernelBench/
  docs/
    architecture.md
    encoding.md
    harvesting.md
    performance-tools.md
    verification-2026-04-17.md
    adr/
    plans/
  examples/
    candidates/
  scripts/
    bootstrap_from_kernelbench.py
    check_encoding.py
    harvest_library_candidates.py
    manual_profile_demo.py
    ncu_profile_task.py
    preview_benchmark.py
    run_candidate_benchmark.py
    smoke_runtime_validation.py
  src/newkernelbench/
    analysis_plan.py
    candidate_loader.py
    catalog.py
    evaluator.py
    kernelbench_adapter.py
    planner.py
    report.py
    schema.py
    task_loader.py
    taxonomy.py
    variants.py
  configs/
  results/
  profile_runs/
```

## Verified environment

The current validated execution path is:

- remote machine: `192.168.17.171`
- conda env: `aikg`
- GPU policy: `CUDA_VISIBLE_DEVICES=1`
- visible runtime device during validation: `NVIDIA H800`

## Encoding policy

All text files in this project should use UTF-8:

- markdown
- python
- json
- toml

The project includes:

- `.editorconfig` with `charset = utf-8`
- `docs/encoding.md`
- `scripts/check_encoding.py`

Check encoding with:

```bash
python scripts/check_encoding.py
```

## Quick start

### 1. Build and preview the starter manifest

```bash
cd /supercloud/llm-code/mkl/project/clang/KernelGen/NewKernelBench
CUDA_VISIBLE_DEVICES=1 conda run -n aikg python scripts/bootstrap_from_kernelbench.py   --kernelbench-root ../KernelBench/KernelBench   --output configs/starter_manifest.json

CUDA_VISIBLE_DEVICES=1 conda run -n aikg python scripts/preview_benchmark.py   --manifest configs/starter_manifest.json   --show-tasks 5
```

### 2. Harvest more workload candidates from real libraries

Example for operator-style harvesting:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n aikg python scripts/harvest_library_candidates.py   --module torch.nn   --limit 80   --output configs/harvest_torch_nn.json
```

Example for model- or block-style harvesting:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n aikg python scripts/harvest_library_candidates.py   --module transformers.models.bert.modeling_bert   --limit 40   --output configs/harvest_bert_modeling.json
```

These harvest outputs are intended for manual review. They help you decide which operators, blocks, or model components should become new seed tasks.

### 3. Run manual profiling tools

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n aikg python scripts/manual_profile_demo.py   --task ../KernelBench/KernelBench/level1/19_ReLU.py   --tool all   --output-dir profile_runs/relu_demo
```

This writes:

- `summary.json`
- `torch_profiler.txt`
- `cuda_memory.json`
- an `ncu` command you can run manually or trigger from the same script

### 4. Run the full benchmark runner on a candidate implementation

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n aikg python scripts/run_candidate_benchmark.py   --task ../KernelBench/KernelBench/level1/1_Square_matrix_multiplication_.py   --candidate examples/candidates/1_square_matmul_modelnew.py   --mode forward_backward   --with-compile-baseline   --output results/matmul_demo.json
```

A second example with parameters and channels-last layout:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n aikg python scripts/run_candidate_benchmark.py   --task ../KernelBench/KernelBench/level2/1_Conv2D_ReLU_BiasAdd.py   --candidate examples/candidates/1_conv2d_relu_biasadd_modelnew.py   --layout channels_last   --mode forward_backward   --output results/conv_demo.json
```

## What the full runner does

The full runner now covers the core loop that was missing before:

1. load a real seed task from the original `KernelBench`
2. apply variant settings for device, dtype, layout, and execution mode
3. load a candidate implementation from a Python file
4. attempt weight synchronization from the reference model when possible
5. evaluate correctness and input-gradient correctness
6. measure speedup versus eager and optionally versus `torch.compile`
7. save a UTF-8 result bundle plus profiler artifacts

## Validation completed so far

The current verified items include:

- UTF-8 repair for the paper summary markdown in both local Windows and remote Linux copies
- UTF-8 validation across `NewKernelBench`
- manifest generation for all 250 original seed tasks and 9494 planned variants
- smoke validation for forward and backward execution paths
- manual profiling with `torch.profiler` and CUDA memory stats
- `ncu` invocation path prepared and tested
- full runner executed successfully on:
  - matmul candidate demo
  - conv-plus-relu-plus-bias candidate demo

See `docs/verification-2026-04-17.md` for details.

## Current limitation

`ncu` is installed on the machine, but the current user does not have permission to access GPU performance counters. The prepared command path works, but the profiler returns `ERR_NVGPUCTRPERM` until the system-side counter permission is enabled.

## Recommended next step

The next useful step is to extend the full runner so it can ingest batches of candidate files, run multiple variants automatically, and emit aggregated summaries on top of the per-run JSON bundles.
