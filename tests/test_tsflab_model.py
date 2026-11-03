import importlib.util
from pathlib import Path

import torch

from test_ef import TSLibStyle

MODEL_PY = Path(__file__).resolve().parents[1] / "integrations" / "tsflab" / "evolutionary_forecasting" / "model.py"
_spec = importlib.util.spec_from_file_location("tsflab_ef_model", MODEL_PY)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
Model = _module.Model


def test_loads_base_model_checkpoint():
    torch.manual_seed(0)
    base = TSLibStyle(24, 8)
    fresh = Model(TSLibStyle(24, 8), label_len=0)
    fresh.load_state_dict(base.state_dict())
    for name, value in base.state_dict().items():
        assert torch.equal(fresh.model.state_dict()[name], value)


def test_loads_own_checkpoint():
    torch.manual_seed(0)
    saved = Model(TSLibStyle(24, 8), label_len=0)
    fresh = Model(TSLibStyle(24, 8), label_len=0)
    fresh.load_state_dict(saved.state_dict())
    for name, value in saved.state_dict().items():
        assert torch.equal(fresh.state_dict()[name], value)
