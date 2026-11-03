"""Model specification for Evolutionary Forecasting (EF) on TSFLab.

``EF`` wraps any catalog model: the base model is built with
``pred_len = block_len`` and rolled out to the run's ``task.pred_len``.
Training optimizes only the first block (Appendix E of the paper).
"""

from __future__ import annotations

import copy

from pydantic import BaseModel, Field

from tsflab.catalog.registry.models import MODEL_CATALOG, ModelSpec
from tsflab.experiments.runner.model_io import call_forecaster, slice_prediction_target
from tsflab.models.evolutionary_forecasting.model import Model


class ModelParameterConfig(BaseModel):
    base: str = Field(min_length=1)
    block_len: int = Field(gt=0)
    base_params: dict = Field(default_factory=dict)


def build_model(cfg, params):
    """Build the base model at horizon ``block_len`` and wrap it."""
    base_cfg = copy.deepcopy(cfg)
    base_cfg.task.pred_len = params["block_len"]
    base = MODEL_CATALOG.get(params["base"]).build(base_cfg, params["base_params"])
    return Model(base, block_len=params["block_len"], label_len=cfg.task.label_len)


def first_block_objective(model, batch, criterion):
    """Supervise only the first reasoning block, i.e. standard training at horizon ``block_len``."""
    span = model.label_len + model.block_len
    y_mark = None if batch.y_mark is None else batch.y_mark[:, :span]
    outputs = call_forecaster(model, batch.x, batch.x_mark, batch.dec_inp[:, :span], y_mark)
    outputs, target = slice_prediction_target(outputs, batch.y[:, :span], model.block_len, batch.features)
    return None, criterion(outputs, target)


SPEC = ModelSpec(
    name="EF",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    contract_task={"seq_len": 96, "pred_len": 192, "label_len": 0},
    training_objective=first_block_objective,
)
