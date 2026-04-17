# Harvesting New Workloads

This document describes the current manual extension path for adding more realistic workloads into `NewKernelBench`.

## Goal

The harvesting workflow is meant to help you find:

- primitive operators from foundational libraries
- fused blocks from model libraries
- architecture-level classes that can be decomposed into benchmark seed tasks

## Main script

Use:

```bash
python scripts/harvest_library_candidates.py --module torch.nn --limit 80
```

## Recommended library targets

Good starting points:

- `torch.nn`
- `torch.nn.functional`
- `transformers.models.bert.modeling_bert`
- `transformers.models.llama.modeling_llama`
- `timm.layers`
- `timm.models`
- `xformers.ops`
- quantization or optimizer libraries such as `torchao` or `bitsandbytes`

## Example commands

Primitive operator harvesting:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n aikg python scripts/harvest_library_candidates.py   --module torch.nn   --limit 80   --output configs/harvest_torch_nn.json
```

Transformer block harvesting:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n aikg python scripts/harvest_library_candidates.py   --module transformers.models.bert.modeling_bert   --limit 40   --output configs/harvest_bert_modeling.json
```

Submodule walk example:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n aikg python scripts/harvest_library_candidates.py   --module transformers.models   --walk-submodules   --max-submodules 10   --limit 120   --output configs/harvest_transformers_models.json
```

## Output fields

Each harvest record includes:

- `module`
- `name`
- `qualname`
- `member_type`
- `signature`
- `guessed_categories`

## Recommended manual review flow

1. Run the harvester on a library module.
2. Inspect the generated JSON.
3. Group the findings by category and real workload importance.
4. Select seed tasks to convert into benchmark tasks.
5. Add those tasks as new source workloads or adapter inputs.

## Current note

In the validated `aikg` environment:

- `torch.nn` harvesting worked
- `transformers.models.bert.modeling_bert` harvesting worked
- `torchvision` was not installed in that environment during this run
