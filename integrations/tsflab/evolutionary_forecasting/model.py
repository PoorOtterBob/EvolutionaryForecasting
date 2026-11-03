"""TSFLab model: a catalog forecaster with output horizon ``block_len`` rolled out to ``task.pred_len``."""

from __future__ import annotations

from ef import EvolutionaryForecaster


class Model(EvolutionaryForecaster):
    """EF wrapper that also loads a checkpoint saved by the base model's own run."""

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        if set(state_dict) == set(self.model.state_dict()):
            state_dict = {f"model.{key}": value for key, value in state_dict.items()}
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
