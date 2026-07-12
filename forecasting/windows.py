"""
Walk-forward window construction with a purge/embargo.

Standard k-fold shuffles time and leaks the future -- forbidden here. We use an
expanding (or sliding) training window, forecast `horizon` steps immediately after
the origin, then advance the origin.

Two concepts that are easy to conflate (and were, in an earlier version):

  * FORECAST ALIGNMENT. The model conditions only on data up to and including the
    origin, and forecasts steps 1..horizon = indices origin+1 .. origin+horizon.
    The test window therefore sits IMMEDIATELY after the origin. Pushing it away
    would silently score a k-step forecast against a (k+gap)-step realization.

  * PURGE / EMBARGO. This protects the *training* set, not the test window. If the
    training targets are themselves built with look-ahead (e.g. supervised pairs
    (X_t, y_{t+h})), the last few training rows depend on data inside the test
    window. Purging removes the last `embargo` observations from the right edge of
    the training window so no training label overlaps the test. For models fit
    directly on the return series (AR/GARCH/foundation) no purge is needed, so the
    default is 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Literal

import numpy as np


@dataclass(frozen=True)
class Split:
    """Index ranges for one walk-forward step (all half-open [start, stop))."""

    train_start: int
    train_stop: int      # exclusive; right edge of training AFTER purge
    origin: int          # last observation the model may condition on
    test_start: int      # first forecast target index (== origin + 1)
    test_stop: int       # exclusive; == test_start + horizon

    @property
    def train_slice(self) -> slice:
        return slice(self.train_start, self.train_stop)

    @property
    def test_slice(self) -> slice:
        return slice(self.test_start, self.test_stop)


def walk_forward(
    n: int,
    *,
    horizon: int = 1,
    min_train: int = 252,
    step: int = 1,
    embargo: int = 0,
    mode: Literal["expanding", "sliding"] = "expanding",
    window: int | None = None,
) -> Iterator[Split]:
    """Yield walk-forward splits over a series of length `n`.

    Parameters
    ----------
    horizon : forecast steps produced at each origin (targets origin+1..origin+h).
    min_train : minimum training observations before the first origin.
    step : how many observations to advance the origin between folds. Use
        `horizon` for non-overlapping (independent) test windows, or 1 for the
        densest evaluation (more folds, but autocorrelated).
    embargo : PURGE. Observations removed from the right edge of the training
        window to prevent look-ahead training targets from overlapping the test
        window. Default 0 (safe for models fit directly on the series).
    mode : 'expanding' grows the train set; 'sliding' fixes it at `window`.
    window : train length for sliding mode.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if embargo < 0:
        raise ValueError("embargo must be >= 0")
    if mode == "sliding" and window is None:
        window = min_train

    # origin is the last usable index; first origin leaves min_train observations
    # (+ purge) available for training.
    origin = min_train - 1 + embargo
    while True:
        test_start = origin + 1
        test_stop = test_start + horizon
        if test_stop > n:
            break
        train_stop = origin + 1 - embargo  # purge the right edge
        train_start = 0 if mode == "expanding" else max(0, train_stop - int(window))
        if train_stop - train_start >= min_train:
            yield Split(
                train_start=train_start,
                train_stop=train_stop,
                origin=origin,
                test_start=test_start,
                test_stop=test_stop,
            )
        origin += step


def collect_splits(n: int, **kw) -> List[Split]:
    return list(walk_forward(n, **kw))
