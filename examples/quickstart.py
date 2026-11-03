"""Train a linear forecaster on a short block (L=24) and roll it out to H=192.

Run: python examples/quickstart.py
"""

import math

import torch
import torch.nn as nn

from ef import evolutionary_forecast

T, L, H = 96, 24, 192
torch.manual_seed(0)

# A toy series with daily and weekly periods.
t = torch.arange(6000, dtype=torch.float32)
series = (torch.sin(2 * math.pi * t / 24) + 0.5 * torch.sin(2 * math.pi * t / 168)
          + 0.1 * torch.randn_like(t)).unsqueeze(-1)


class Linear(nn.Module):
    def __init__(self, seq_len, pred_len):
        super().__init__()
        self.pred_len = pred_len
        self.proj = nn.Linear(seq_len, pred_len)

    def forward(self, x):  # [B, T, C] -> [B, L, C]
        return self.proj(x.transpose(1, 2)).transpose(1, 2)


def windows(data, start, end, out_len):
    idx = range(start, end - T - out_len)
    x = torch.stack([data[i:i + T] for i in idx])
    y = torch.stack([data[i + T:i + T + out_len] for i in idx])
    return x, y


def fit(out_len):
    model = Linear(T, out_len)
    x, y = windows(series, 0, 4000, out_len)
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(300):
        optim.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        optim.step()
    return model


x_test, y_test = windows(series, 4000, 6000, H)
df = fit(H)  # Direct Forecasting: one model per horizon, L = H
ef = fit(L)  # Evolutionary Forecasting: one short-block model for every horizon
with torch.no_grad():
    mse_df = nn.functional.mse_loss(df(x_test), y_test).item()
    mse_ef = nn.functional.mse_loss(evolutionary_forecast(ef, x_test, horizon=H), y_test).item()

print(f"DF  (L=H={H}):     test MSE {mse_df:.4f}")
print(f"EF  (L={L}, H={H}): test MSE {mse_ef:.4f}  ({math.ceil(H / L)} blocks)")
