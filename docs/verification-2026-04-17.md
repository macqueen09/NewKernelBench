# 2026-04-17 Verification Notes

## Environment

- conda env: `aikg`
- runtime command pattern: `CUDA_VISIBLE_DEVICES=1 conda run -n aikg python ...`
- visible CUDA device count inside the process: `1`
- visible device name inside the process: `NVIDIA H800`
- torch version: `2.9.1+cu128`

## Encoding verification

- remote Linux locale is UTF-8
- local Windows shell was switched to code page `65001` with UTF-8 PowerShell output settings
- the repaired paper summary markdown is now detected as `charset=utf-8` on the remote host
- `scripts/check_encoding.py` passes across the current `NewKernelBench` tree

## Fixes applied before validation

- Rewrote corrupted Markdown files in ASCII to remove question-mark encoding artifacts.
- Rewrote the paper summary markdown as a real UTF-8 file and synced it between local Windows and remote Linux.
- Added `.editorconfig` and `docs/encoding.md`.
- Added `task_loader.py` so a real KernelBench seed task can be loaded into a runtime bundle.
- Added `smoke_runtime_validation.py` so the evaluator can be exercised on real seed tasks.
- Added `candidate_loader.py` and `run_candidate_benchmark.py` for full candidate execution.
- Added `harvest_library_candidates.py` for manual workload harvesting.
- Added `manual_profile_demo.py` and `ncu_profile_task.py` for manual performance-tool usage.

## Structural validation

Command:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n aikg python scripts/check_encoding.py
CUDA_VISIBLE_DEVICES=1 conda run -n aikg python -m compileall src scripts examples
```

Result:

- UTF-8 validation passed
- all modules compiled successfully

## Manifest validation

Command:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n aikg python scripts/bootstrap_from_kernelbench.py   --kernelbench-root ../KernelBench/KernelBench   --output configs/starter_manifest.json
```

Summary:

- source tasks: `250`
- variants: `9494`
- level distribution: `100 / 100 / 50`
- analysis tier distribution:
  - `light`: `527`
  - `standard`: `3981`
  - `deep`: `4986`

## Harvesting demos

### 1. Primitive operator harvesting from torch.nn

Command:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n aikg python scripts/harvest_library_candidates.py   --module torch.nn   --limit 80   --output configs/harvest_torch_nn.json
```

Observed result:

- record_count: `80`
- category counts included pooling, convolution, normalization, loss, and activation

### 2. Real workload module harvesting from transformers

Command:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n aikg python scripts/harvest_library_candidates.py   --module transformers.models.bert.modeling_bert   --limit 40   --output configs/harvest_bert_modeling.json
```

Observed result:

- record_count: `40`
- sample names included `BertAttention`, `BertEmbeddings`, `BertEncoder`, and several task heads

## Runtime smoke validations

### 1. Primitive matmul, fp32, forward+backward

- compiled: `true`
- correct: `true`
- grad_correct: `true`
- profiler artifacts: `torch_profiler`, `cuda_memory`

### 2. Fusion conv+relu+bias, channels_last, forward

- compiled: `true`
- correct: `true`
- profiler artifacts: `torch_profiler`, `cuda_memory`

### 3. Primitive matmul, fp16, non_contiguous, forward+backward

- compiled: `true`
- correct: `true`
- grad_correct: `true`
- profiler artifacts: `torch_profiler`, `cuda_memory`

### 4. Fusion conv+relu+bias, channels_last, forward+backward

- compiled: `true`
- correct: `true`
- grad_correct: `true`
- profiler artifacts: `torch_profiler`, `cuda_memory`

## Manual profiling demos

### 1. torch.profiler plus CUDA memory

Command:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n aikg python scripts/manual_profile_demo.py   --task ../KernelBench/KernelBench/level1/19_ReLU.py   --tool all   --output-dir profile_runs/relu_demo
```

Observed result:

- summary written
- torch profiler output written
- CUDA memory JSON written
- `ncu` command emitted for manual reuse

### 2. ncu invocation

Command:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n aikg python scripts/manual_profile_demo.py   --task ../KernelBench/KernelBench/level1/19_ReLU.py   --tool ncu   --run-ncu   --output-dir profile_runs/relu_ncu_demo
```

Observed result:

- `ncu` launched correctly
- result failed with `ERR_NVGPUCTRPERM`
- failure cause: missing permission for NVIDIA GPU performance counters on the current machine

## Full runner demos

### 1. Matmul candidate demo with compile baseline

Command:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n aikg python scripts/run_candidate_benchmark.py   --task ../KernelBench/KernelBench/level1/1_Square_matrix_multiplication_.py   --candidate examples/candidates/1_square_matmul_modelnew.py   --mode forward_backward   --with-compile-baseline   --warmup 1   --repeat 3   --output results/matmul_demo.json
```

Observed result:

- compiled: `true`
- correct: `true`
- grad_correct: `true`
- compile baseline enabled
- profiler artifact saved as `results/matmul_demo.torch_profiler.txt`

### 2. Conv+ReLU+Bias candidate demo with channels-last layout

Command:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n aikg python scripts/run_candidate_benchmark.py   --task ../KernelBench/KernelBench/level2/1_Conv2D_ReLU_BiasAdd.py   --candidate examples/candidates/1_conv2d_relu_biasadd_modelnew.py   --layout channels_last   --mode forward_backward   --warmup 1   --repeat 2   --output results/conv_demo.json
```

Observed result:

- compiled: `true`
- correct: `true`
- grad_correct: `true`
- profiler artifact saved as `results/conv_demo.torch_profiler.txt`

## Current limitations

- The current demo candidates are intentionally equivalent to the reference implementation. They validate the runner and result pipeline, not a novel optimized kernel.
- `ncu` is installed but blocked by GPU counter permissions on this machine.
- Batch execution over many candidate files and many variants is not implemented yet.

## Recommended next implementation step

Build a batch runner that:

1. reads a candidate directory,
2. maps candidate files to task ids,
3. runs multiple variants automatically,
4. aggregates all per-run JSON bundles into summary reports.
