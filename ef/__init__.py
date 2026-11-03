"""Evolutionary Forecasting (EF): train on a short block, roll it out to any horizon."""

from .core import evolutionary_forecast, num_blocks, rollout
from .tslib import EvolutionaryForecaster

__all__ = ["evolutionary_forecast", "rollout", "num_blocks", "EvolutionaryForecaster"]
__version__ = "1.0.0"
