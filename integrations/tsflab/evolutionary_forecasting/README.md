# EF (Evolutionary Forecasting)

Wraps any TSFLab catalog model. The base model is built with `pred_len = block_len` and rolled out block by block to the run's `task.pred_len`, following *To See Far, Look Close: Evolutionary Forecasting for Long-term Time Series* (NeurIPS 2026, [arXiv:2601.23114](https://arxiv.org/abs/2601.23114)).

## Parameters

| Name | Meaning |
|---|---|
| `base` | catalog name of the base model, e.g. `PatchTST` |
| `block_len` | block length `L`, the base model's own output horizon |
| `base_params` | parameters of the base model; set channel widths such as `enc_in` explicitly |

## Training

The training objective supervises only the first block, which is standard training at horizon `block_len`. Validation, early stopping and test metrics use the full rollout to `task.pred_len`. The paper's protocol instead trains the base model in its own run at `pred_len = block_len`; to follow it, evaluate that run's checkpoint through `evaluation.checkpoint_path`. The wrapper accepts the base model's checkpoint directly.

Source: https://github.com/PoorOtterBob/EvolutionaryForecasting
