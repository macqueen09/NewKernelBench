# Performance Tools

`NewKernelBench` currently uses or prepares the following profiling and analysis tools:

- eager wall-time timing
- compile-baseline timing through `torch.compile`
- `torch.profiler`
- CUDA peak memory stats from `torch.cuda`
- `ncu` for manual Nsight Compute runs
- `nsys` availability can also be checked from the system

## Verified availability on the current machine

Available in the remote environment:

- `ncu`
- `nsys`
- `nvidia-smi`
- `torch.profiler`
- CUDA memory statistics through PyTorch

## Manual torch profiler demo

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n aikg python scripts/manual_profile_demo.py   --task ../KernelBench/KernelBench/level1/19_ReLU.py   --tool all   --output-dir profile_runs/relu_demo
```

This produces:

- `profile_runs/relu_demo/summary.json`
- `profile_runs/relu_demo/torch_profiler.txt`
- `profile_runs/relu_demo/cuda_memory.json`

## Manual ncu demo

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n aikg python scripts/manual_profile_demo.py   --task ../KernelBench/KernelBench/level1/19_ReLU.py   --tool ncu   --run-ncu   --output-dir profile_runs/relu_ncu_demo
```

## Current ncu limitation on this machine

The command path is correct, but the current user does not have permission to access NVIDIA GPU performance counters. The observed error is:

```text
ERR_NVGPUCTRPERM
```

That means `ncu` is installed and callable, but the machine-side permission for GPU performance counters must be enabled before real Nsight Compute collection can succeed.

## Full runner demo

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n aikg python scripts/run_candidate_benchmark.py   --task ../KernelBench/KernelBench/level1/1_Square_matrix_multiplication_.py   --candidate examples/candidates/1_square_matmul_modelnew.py   --mode forward_backward   --with-compile-baseline   --output results/matmul_demo.json
```
