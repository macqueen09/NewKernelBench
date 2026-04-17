# 2026-04-16 NewKernelBench Design

## Scope

This first delivery covers:

- loading seed tasks from the original `KernelBench`
- reorganizing them into a multi-axis benchmark schema
- generating variants for each seed task
- assigning a tiered analysis plan to each variant
- generating a starter manifest and summary

## Recommended approach

The recommended path is "reuse old data, rebuild the planning layer" instead of modifying the original `KernelBench` code directly.

Reasons:

- lower risk
- old and new benchmarks can coexist
- later multi-backend work stays easier because execution is already separated from planning

## First delivery

The first delivery includes:

- the new project skeleton
- core schema, taxonomy, planner, and report modules
- automatic generation of `starter_manifest.json` from original `KernelBench` tasks
- CLI tools to bootstrap and preview the manifest

## Deferred work

- a full runner for generated candidate kernels
- broader library harvesting
- AST or FX-based operator analysis
- richer profiler result ingestion
