"""
The model ladder. The *gaps between rungs* are the measurement:

  rung 0  NaiveForecaster    random walk (benchmark floor)
  rung 1  ARForecaster       linear autocorrelation structure
  rung 2  GARCHForecaster    predictable volatility (direction may still be random)
  rung 3  FoundationForecaster  nonlinear structure (Chronos / TimesFM)

Reading which rungs beat naive tells you the *type* of predictability, not just
its presence. See characterize.classify_structure.
"""

from __future__ import annotations

from typing import List

from .base import Forecaster, SkipModel
from .naive import NaiveForecaster
from .linear import ARForecaster
from .garch import GARCHForecaster
from .foundation import FoundationForecaster

__all__ = [
    "Forecaster", "SkipModel",
    "NaiveForecaster", "ARForecaster", "GARCHForecaster", "FoundationForecaster",
    "default_ladder",
]


def default_ladder(include_foundation: bool = True,
                   **foundation_kwargs) -> List[Forecaster]:
    """Construct the standard ladder. Foundation models are optional so the
    statistical core runs anywhere; when their deps are missing they self-skip.
    `foundation_kwargs` pass through to FoundationForecaster (backend,
    context_length, num_paths, kronos_model, ...)."""
    ladder: List[Forecaster] = [
        NaiveForecaster(),
        ARForecaster(max_lag=5),
        GARCHForecaster(),
    ]
    if include_foundation:
        foundation_kwargs.setdefault("backend", "auto")
        ladder.append(FoundationForecaster(**foundation_kwargs))
    return ladder
