"""Model-agnostic Evolutionary Forecasting (EF) rollout.

A model trained with output horizon ``L`` (the *block length*) reaches any
evaluation horizon ``H`` by ``K = ceil(H / L)`` sequential calls. Block ``k``
reads the length-``T`` window that ends right before it, built from the history
and the blocks already generated, and the concatenated blocks are truncated to
``H`` (Definitions 4.1 and 4.2 of the paper). With ``L >= H`` this reduces to a
single call, i.e. conventional Direct Forecasting.
"""

import math
from typing import Callable, Optional

import torch

BlockFn = Callable[[torch.Tensor, int], torch.Tensor]


def num_blocks(horizon: int, block_len: int) -> int:
    """Number of reasoning blocks ``K = ceil(H / L)``."""
    if horizon <= 0 or block_len <= 0:
        raise ValueError(f"horizon and block_len must be positive, got {horizon} and {block_len}")
    return math.ceil(horizon / block_len)


def rollout(block_fn: BlockFn, x: torch.Tensor, horizon: int, block_len: int) -> torch.Tensor:
    """Generic EF engine.

    Args:
        block_fn: ``block_fn(window, k)`` maps the input window of block ``k``
            (``[B, T, C]``, ``k`` starting at 0) to at least ``block_len`` future
            steps (``[B, >=L, C]``). Only the first ``block_len`` steps are used.
        x: history window ``[B, T, C]``.
        horizon: evaluation horizon ``H``.
        block_len: block length ``L`` the model was trained with.

    Returns:
        Forecast ``[B, H, C]``.
    """
    seq_len = x.shape[1]
    buffer = x
    blocks = []
    for k in range(num_blocks(horizon, block_len)):
        window = buffer[:, k * block_len:k * block_len + seq_len]
        block = block_fn(window, k)[:, :block_len]
        if block.shape[1] != block_len:
            raise ValueError(f"block_fn returned {block.shape[1]} steps, expected at least {block_len}")
        blocks.append(block)
        buffer = torch.cat([buffer, block.to(buffer.dtype)], dim=1)
    return torch.cat(blocks, dim=1)[:, :horizon]


def evolutionary_forecast(
    model: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    horizon: int,
    block_len: Optional[int] = None,
) -> torch.Tensor:
    """EF for any model mapping ``[B, T, C] -> [B, L, C]``.

    ``block_len`` defaults to ``model.pred_len`` when the model defines it.

    Example:
        >>> y_hat = evolutionary_forecast(model, x, horizon=720, block_len=96)
    """
    if block_len is None:
        block_len = getattr(model, "pred_len", None)
        if block_len is None:
            raise ValueError("block_len is required when the model has no `pred_len` attribute")
    return rollout(lambda window, _: model(window), x, horizon, block_len)
