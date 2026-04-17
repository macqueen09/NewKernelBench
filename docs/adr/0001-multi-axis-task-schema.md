# ADR 0001: Use a multi-axis task schema

## Status
Accepted

## Context

The original `KernelBench` `Level 1/2/3` split is useful for coarse difficulty, but it does not express:

- functional category differences
- dtype differences
- layout differences
- forward versus backward execution differences

That compression makes failure diagnosis too weak for a richer benchmark.

## Decision

Each `NewKernelBench` benchmark entry is split into:

- `SourceTask`: the upstream seed task
- `TaskVariant`: a concrete shape, dtype, layout, and execution-mode combination
- `AnalysisPlan`: the recommended toolchain for that combination

## Consequences

Positive:

- It becomes easier to distinguish task-family difficulty from dtype or layout-specific failures.
- It supports combined analysis over correctness, gradients, speedup, and failure taxonomy.

Trade-offs:

- The manifest becomes much larger.
- The system needs budget controls to stop variant explosion.
