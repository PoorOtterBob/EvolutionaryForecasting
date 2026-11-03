import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ef import EvolutionaryForecaster, evolutionary_forecast, num_blocks, rollout

B, T, C = 3, 24, 2


class LinearForecaster(nn.Module):
    """Channel-independent linear map [B, T, C] -> [B, L, C]."""

    def __init__(self, seq_len, pred_len):
        super().__init__()
        self.pred_len = pred_len
        self.proj = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        return self.proj(x.transpose(1, 2)).transpose(1, 2)


class TSLibStyle(nn.Module):
    """TSLib signature; uses the encoder marks like iTransformer does."""

    def __init__(self, seq_len, pred_len, mark_dim=4):
        super().__init__()
        self.pred_len = pred_len
        self.core = LinearForecaster(seq_len, pred_len)
        self.mark = nn.Linear(mark_dim, 1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        out = self.core(x_enc)
        if x_mark_enc is not None:
            out = out + self.mark(x_mark_enc).mean(dim=1, keepdim=True)
        return out[:, -self.pred_len:]


def reference_paper_rollout(model, batch_x, batch_x_mark, batch_y, batch_y_mark, label_len, pred_len):
    """`Exp_Long_Term_Forecast._model` from the original paper code (CoT branch and DF branch)."""
    cot_step = math.ceil((batch_y.shape[1] - label_len) / model.pred_len)
    if cot_step == 1:
        dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float()
        return model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[:, :pred_len]
    preds_list = []
    marks = torch.cat([batch_x_mark, batch_y_mark[:, label_len:]], dim=1)
    inputs = F.pad(batch_x, (0, 0, 0, cot_step * model.pred_len), "constant", 0)
    for step in range(cot_step):
        s = step * model.pred_len
        e = s + model.seq_len
        dec_inp = torch.zeros_like(batch_y[:, -model.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, s:s + label_len], dec_inp], dim=1).float()
        pred = model(inputs[:, s:e], marks[:, s:e], dec_inp, marks[:, e - label_len:e + model.pred_len])
        preds_list.append(pred)
        inputs[:, e:e + model.pred_len] = inputs[:, e:e + model.pred_len] + preds_list[-1]
    return torch.cat(preds_list, dim=1)[:, :batch_y.shape[1] - label_len]


def test_num_blocks():
    assert num_blocks(720, 96) == 8
    assert num_blocks(96, 96) == 1
    assert num_blocks(96, 720) == 1
    with pytest.raises(ValueError):
        num_blocks(0, 96)


def test_direct_forecasting_is_special_case():
    torch.manual_seed(0)
    model, x = LinearForecaster(T, 16), torch.randn(B, T, C)
    assert torch.allclose(evolutionary_forecast(model, x, horizon=16), model(x))
    assert torch.allclose(evolutionary_forecast(model, x, horizon=10), model(x)[:, :10])


@pytest.mark.parametrize("horizon,block_len", [(30, 8), (50, 24), (100, 7)])
def test_matches_definition(horizon, block_len):
    """Covers direct, semi-extrapolation and pure-extrapolation phases."""
    torch.manual_seed(0)
    model, x = LinearForecaster(T, block_len), torch.randn(B, T, C)
    y = evolutionary_forecast(model, x, horizon)
    assert y.shape == (B, horizon, C)
    preds = torch.zeros(B, 0, C)
    for k in range(num_blocks(horizon, block_len)):
        offset = k * block_len
        if offset == 0:
            window = x
        elif offset < T:
            window = torch.cat([x[:, offset:], preds[:, :offset]], dim=1)
        else:
            window = preds[:, offset - T:offset]
        preds = torch.cat([preds, model(window)], dim=1)
    assert torch.allclose(y, preds[:, :horizon], atol=1e-6)


def test_rollout_passes_block_index():
    seen = []
    rollout(lambda w, k: seen.append(k) or torch.zeros(w.shape[0], 5, w.shape[2]), torch.zeros(1, T, 1), 12, 5)
    assert seen == [0, 1, 2]


def test_rollout_rejects_short_blocks():
    with pytest.raises(ValueError):
        rollout(lambda w, k: torch.zeros(w.shape[0], 3, w.shape[2]), torch.zeros(1, T, 1), 12, 5)


@pytest.mark.parametrize("horizon,block_len,label_len", [(48, 12, 6), (40, 12, 0), (12, 12, 6), (30, 48, 6)])
def test_tslib_wrapper_matches_paper_code(horizon, block_len, label_len):
    torch.manual_seed(0)
    model = TSLibStyle(T, block_len)
    model.seq_len = T
    batch_x, batch_x_mark = torch.randn(B, T, C), torch.randn(B, T, 4)
    batch_y, batch_y_mark = torch.randn(B, label_len + horizon, C), torch.randn(B, label_len + horizon, 4)
    dec_inp = torch.cat([batch_y[:, :label_len], torch.zeros(B, horizon, C)], dim=1)

    ours = EvolutionaryForecaster(model, label_len=label_len)(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    ref = reference_paper_rollout(model, batch_x, batch_x_mark, batch_y, batch_y_mark, label_len, horizon)
    assert ours.shape == (B, horizon, C)
    assert torch.allclose(ours, ref, atol=1e-6)


def test_tslib_wrapper_without_marks_and_partial_last_block():
    torch.manual_seed(0)
    model = TSLibStyle(T, 10)
    wrapper = EvolutionaryForecaster(model, label_len=4)
    x = torch.randn(B, T, C)
    y = wrapper(x, None, torch.zeros(B, 4 + 25, C), None)
    assert y.shape == (B, 25, C)
    assert torch.allclose(y, evolutionary_forecast(model.core, x, 25), atol=1e-6)
    # marks shorter than the last full block are padded, not an error
    y = wrapper(x, torch.randn(B, T, 4), torch.zeros(B, 4 + 25, C), torch.randn(B, 4 + 25, 4))
    assert y.shape == (B, 25, C)
