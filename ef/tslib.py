"""Time-Series-Library (TSLib) integration.

TSLib models share the signature ``model(x_enc, x_mark_enc, x_dec, x_mark_dec)``
and return ``[B, >=pred_len, C]``. :class:`EvolutionaryForecaster` wraps such a
model trained with ``pred_len = L`` and exposes the same signature, so TSLib's
unmodified ``test()`` loop can evaluate it at any horizon ``H``: the horizon is
read from the decoder input, ``x_dec.shape[1] - label_len``.
"""

from typing import Optional

import torch
import torch.nn as nn

from .core import num_blocks, rollout


class EvolutionaryForecaster(nn.Module):
    """Wrap a TSLib forecaster trained with ``pred_len = block_len``.

    Args:
        model: TSLib model whose output horizon is ``block_len``.
        block_len: block length ``L``; defaults to ``model.pred_len``.
        label_len: TSLib ``label_len`` (start-token length of the decoder input).

    For every block, the decoder start tokens are taken from the end of the
    current input window, so only observed history and generated blocks are
    used. Time-feature marks are taken from ``x_mark_enc`` and ``x_mark_dec``;
    when the last block runs past ``H``, the final mark is repeated.
    """

    def __init__(self, model: nn.Module, block_len: Optional[int] = None, label_len: int = 0):
        super().__init__()
        self.model = model
        self.block_len = block_len if block_len is not None else model.pred_len
        self.label_len = label_len

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        horizon = x_dec.shape[1] - self.label_len
        seq_len, block_len, label_len = x_enc.shape[1], self.block_len, self.label_len
        marks = self._future_marks(x_mark_enc, x_mark_dec, num_blocks(horizon, block_len) * block_len)

        def block_fn(window, k):
            start = k * block_len
            dec_inp = torch.zeros(window.shape[0], label_len + block_len, window.shape[2],
                                  dtype=window.dtype, device=window.device)
            if label_len > 0:
                dec_inp[:, :label_len] = window[:, -label_len:]
            if marks is None:
                mark_enc = mark_dec = None
            else:
                mark_enc = marks[:, start:start + seq_len]
                mark_dec = marks[:, start + seq_len - label_len:start + seq_len + block_len]
            out = self.model(window, mark_enc, dec_inp, mark_dec)
            return out[:, -block_len:]

        return rollout(block_fn, x_enc, horizon, block_len)

    def _future_marks(self, x_mark_enc, x_mark_dec, rollout_len):
        """Marks for the history plus ``rollout_len`` future steps, or ``None``."""
        if x_mark_enc is None or x_mark_dec is None:
            return None
        marks = torch.cat([x_mark_enc, x_mark_dec[:, self.label_len:]], dim=1)
        missing = x_mark_enc.shape[1] + rollout_len - marks.shape[1]
        if missing > 0:
            marks = torch.cat([marks, marks[:, -1:].expand(-1, missing, -1)], dim=1)
        return marks
