# ADR 0002: Use a tiered analysis pipeline

## Status
Accepted

## Context

Running the heaviest profiler stack on every task causes three problems:

- high benchmark cost
- strong environment dependencies
- slow iteration on simple operators

At the same time, wall-time alone is not enough to explain why harder tasks are slow.

## Decision

The analysis pipeline is split into three tiers:

- `light`
  - correctness
  - eager timing
- `standard`
  - correctness
  - compile-baseline timing
  - torch profiler
  - stability sweep
- `deep`
  - everything in standard
  - CUDA memory stats
  - Nsight Compute or roofline-style analysis when available

## Consequences

Positive:

- Simple tasks stay cheap to run.
- Complex tasks get stronger diagnosis.
- Profiler outputs can later be fed back into optimization loops.

Trade-offs:

- The system needs explicit tier-selection logic.
- Tool availability must be checked per environment.
