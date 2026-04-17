# NewKernelBench Architecture

## Requirements summary

Functional requirements:

- Reuse the original `KernelBench` task files.
- Expand each seed task into multi-axis benchmark variants.
- Select different profiling and analysis pipelines for different task difficulty levels.
- Produce reports that are useful for both leaderboard summaries and optimization feedback.

Non-functional requirements:

- Keep the design loosely coupled so multi-backend execution can be added later.
- Allow planning and manifest generation even when full GPU tooling is unavailable.
- Keep all planning artifacts JSON-serializable so they can feed later runners or dashboards.

## High-level architecture

```text
KernelBench seed tasks
        |
        v
kernelbench_adapter.py
        |
        v
SourceTask schema
        |
        v
variants.py ---------------------> analysis_plan.py
        |                               |
        v                               v
BenchmarkTaskPlan ----------------> tool recommendations
        |
        v
planner.py
        |
        +--> configs/starter_manifest.json
        +--> preview / future runner inputs
```

## Module responsibilities

- `taxonomy.py`
  - Defines enums for complexity, category, dtype, layout, execution mode, and analysis tier.
- `schema.py`
  - Defines the core dataclasses used during planning and reporting.
- `kernelbench_adapter.py`
  - Scans original seed tasks and infers category, complexity, differentiability, and library hints.
- `variants.py`
  - Expands each task into shape, dtype, layout, and execution-mode variants.
- `analysis_plan.py`
  - Chooses a light, standard, or deep toolchain for each variant.
- `planner.py`
  - Builds the manifest and computes global counts.
- `report.py`
  - Aggregates compile, correctness, gradient, speedup, and failure-taxonomy results.
- `evaluator.py`
  - Runs a callable pair through correctness, gradient, timing, and optional profiler checks.
- `task_loader.py`
  - Loads a real seed-task file and prepares model instances and tensor inputs for validation.
- `catalog.py`
  - Tracks next library sources for future task harvesting.

## Key decisions

1. Replace a single-axis level-only view with a multi-axis task schema.
2. Make `backward` and layout stress part of the standard benchmark plan.
3. Use a tiered analysis pipeline so deep tools are only used where justified.
4. Keep planning separate from execution so the schema stays stable while the runner evolves.

## Risks and mitigations

- Risk: category inference from filenames and source text is heuristic.
  - Mitigation: preserve task notes and keep room for future AST or FX-based refinement.
- Risk: the variant grid can grow too quickly.
  - Mitigation: use bounded shape buckets and restrict backward variants to the primary buckets.
- Risk: profiler availability differs by environment.
  - Mitigation: detect tool availability first and emit fallback notes in each analysis plan.
