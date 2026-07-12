#!/usr/bin/env python3
"""
generate_historical_features.py

Generates historical feature vectors day-by-day for the training period.

For each trading day with sufficient bar history:
  1. Compute technical indicators from bars history up to that day
  2. Run HMM regime detection
  3. Generate Tier 1 + Tier 2 signals
  4. Pull macro / yield data from DB (backfilled separately)
  5. Assemble the 26-feature vector
  6. Store the feature vector in the `feature_vectors` table

Usage:
    python generate_historical_features.py --end 2025-04-04
    python generate_historical_features.py --instrument EUR_USD --start 2020-01-01 --end 2025-04-04
    python generate_historical_features.py --instrument EUR_USD --min-bars 100
"""

import sys
import os
import argparse
import logging
from datetime import datetime, timedelta
from decimal import Decimal

import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so local imports resolve correctly
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features.technical import compute_all_features
from features.vector import assemble_feature_vector, store_feature_vector
from models.hmm_regime import RegimeDetector
from models.database import fetch_all, get_connection
from signals.tier1 import generate_all_tier1
from signals.tier2 import generate_all_tier2
from signals.composite import compute_composite, store_signals

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
DEFAULT_REGIME = {
    "state_id": 1,
    "state_label": "unknown",
    "confidence": 0.33,
    "days_in_regime": 0,
    "transition_prob": {
        "low_vol": 0.33,
        "high_vol_choppy": 0.34,
        "high_vol_crash": 0.33,
    },
    "model_version": "default",
}

HMM_MIN_BARS = 100          # minimum daily closes the HMM needs to fit
HMM_REFIT_INTERVAL = 30     # re-fit the HMM every N calendar days


def _decimals_to_floats(df: pd.DataFrame) -> pd.DataFrame:
    """Convert any Decimal columns in *df* to float."""
    for col in df.columns:
        if df[col].dtype == object:
            # Check first non-null value
            sample = df[col].dropna().head(1)
            if not sample.empty and isinstance(sample.iloc[0], Decimal):
                df[col] = df[col].apply(
                    lambda v: float(v) if isinstance(v, Decimal) else v
                )
    return df


def _fetch_distinct_dates(instrument: str) -> list:
    """Return sorted list of distinct dates with complete daily bars."""
    rows = fetch_all(
        """
        SELECT DISTINCT date(bar_time) AS bar_date
        FROM bars
        WHERE instrument = %s
          AND granularity = 'D'
          AND complete = TRUE
        ORDER BY bar_date
        """,
        (instrument,),
    )
    return [r['bar_date'] for r in rows]


def _load_daily_bars(instrument: str, up_to_date) -> pd.DataFrame:
    """Load all daily bars up to and including *up_to_date*."""
    rows = fetch_all(
        """
        SELECT bar_time, open, high, low, close, volume
        FROM bars
        WHERE instrument = %s
          AND granularity = 'D'
          AND complete = TRUE
          AND date(bar_time) <= %s
        ORDER BY bar_time
        """,
        (instrument, up_to_date),
    )
    df = pd.DataFrame(rows, columns=["bar_time", "open", "high", "low", "close", "volume"])
    return _decimals_to_floats(df)


def _load_h4_bars(instrument: str, run_date, lookback_days: int = 30) -> pd.DataFrame:
    """Load H4 bars for the last *lookback_days* days relative to *run_date*."""
    start_date = run_date - timedelta(days=lookback_days)
    rows = fetch_all(
        """
        SELECT bar_time, open, high, low, close, volume
        FROM bars
        WHERE instrument = %s
          AND granularity = 'H4'
          AND complete = TRUE
          AND date(bar_time) BETWEEN %s AND %s
        ORDER BY bar_time
        """,
        (instrument, start_date, run_date),
    )
    df = pd.DataFrame(rows, columns=["bar_time", "open", "high", "low", "close", "volume"])
    return _decimals_to_floats(df)


def _fit_hmm_on_df(detector, df: pd.DataFrame) -> None:
    """Fit HMM on the last lookback_days of *df* in a lookahead-free way."""
    import numpy as np
    from hmmlearn.hmm import GaussianHMM
    
    train_df = df.tail(detector.lookback_days).copy()
    train_df["log_return"] = np.log(train_df["close"] / train_df["close"].shift(1))
    train_df["vol_5d"] = train_df["log_return"].rolling(5).std()
    train_df = train_df.dropna()
    
    if len(train_df) < 60:
        raise ValueError(f"Need at least 60 valid bars for HMM training, got {len(train_df)}")
        
    X = train_df[["log_return", "vol_5d"]].values
    detector.model = GaussianHMM(
        n_components=detector.n_states,
        covariance_type="diag",
        n_iter=200,
        random_state=42,
        tol=1e-4,
    )
    detector.model.fit(X)
    
    means = detector.model.means_
    vol_means = means[:, 1]
    sorted_states = np.argsort(vol_means)
    detector.state_map = {
        int(sorted_states[0]): 0,
        int(sorted_states[1]): 1,
        int(sorted_states[2]): 2,
    }
    detector._model_version = "hmm_backtest"


def _predict_regime_on_df(detector, df: pd.DataFrame) -> dict:
    """Predict regime state for the latest bar in *df* in a lookahead-free way."""
    import numpy as np
    from models.hmm_regime import REGIME_LABELS
    
    if detector.model is None:
        return detector._default_regime()
        
    pred_df = df.copy()
    pred_df["log_return"] = np.log(pred_df["close"] / pred_df["close"].shift(1))
    pred_df["vol_5d"] = pred_df["log_return"].rolling(5).std()
    pred_df = pred_df.dropna()
    
    if len(pred_df) < 10:
        return detector._default_regime()
        
    X = pred_df[["log_return", "vol_5d"]].values
    raw_states = detector.model.predict(X)
    posteriors = detector.model.predict_proba(X)
    
    current_raw = int(raw_states[-1])
    current_semantic = detector.state_map.get(current_raw, 1)
    current_confidence = float(posteriors[-1][current_raw])
    
    days_in = 1
    for i in range(len(raw_states) - 2, -1, -1):
        if detector.state_map.get(int(raw_states[i]), -1) == current_semantic:
            days_in += 1
        else:
            break
            
    trans_row = detector.model.transmat_[current_raw].tolist()
    trans_mapped = {}
    for raw_s, sem_s in detector.state_map.items():
        trans_mapped[REGIME_LABELS[sem_s]] = round(trans_row[raw_s], 4)
        
    return {
        "state_id": current_semantic,
        "state_label": REGIME_LABELS[current_semantic],
        "confidence": round(current_confidence, 4),
        "days_in_regime": days_in,
        "transition_prob": trans_mapped,
        "model_version": detector._model_version,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run(instrument: str, start_date, end_date, min_bars: int) -> None:
    logger.info(
        "Starting historical feature generation for %s  "
        "(start=%s, end=%s, min_bars=%d)",
        instrument,
        start_date or "earliest",
        end_date or "latest",
        min_bars,
    )

    all_dates = _fetch_distinct_dates(instrument)
    if not all_dates:
        logger.warning("No daily bars found for %s — nothing to do.", instrument)
        return

    # Apply optional date filters
    if start_date:
        all_dates = [d for d in all_dates if d >= start_date]
    if end_date:
        all_dates = [d for d in all_dates if d <= end_date]

    if not all_dates:
        logger.warning("No dates remain after applying --start / --end filters.")
        return

    logger.info("Total candidate dates: %d", len(all_dates))

    # ---- HMM setup --------------------------------------------------------
    detector = RegimeDetector()
    hmm_fitted = False
    last_hmm_fit_date = None

    # ---- Counters ----------------------------------------------------------
    total = 0
    successes = 0
    failures = 0

    for idx, run_date in enumerate(all_dates):
        # We need at least `min_bars` of history *before* (and including) this day.
        # The index in the original (unfiltered) sorted date list tells us how
        # many bars are available up to this date.  But since we may have filtered
        # by --start, we count directly.
        bars_available = idx + 1  # dates are sorted; idx+1 = count up to this date
        # Actually, we need to count from the *full* unfiltered list.  Reload
        # the position from the full date list.
        # Instead, just count with a quick query-free approach: we already
        # filtered the dates list from the *full* sorted list, and min_bars
        # applies to the number of daily bars available in the DB up to this
        # date.  We'll load the daily bars below and check length.

        total += 1

        try:
            # (a) Load daily bars up to this date
            daily_df = _load_daily_bars(instrument, run_date)
            if len(daily_df) < min_bars:
                logger.debug(
                    "Skipping %s — only %d daily bars (need %d)",
                    run_date, len(daily_df), min_bars,
                )
                successes += 1  # not a failure, just skipped
                continue

            # (b) Load H4 bars (last 30 days)
            h4_df = _load_h4_bars(instrument, run_date, lookback_days=30)

            # (c) Compute technical indicators
            technical = compute_all_features(daily_df, h4_df)

            # (d) HMM regime detection
            need_fit = (
                not hmm_fitted
                or (
                    last_hmm_fit_date is not None
                    and (run_date - last_hmm_fit_date).days >= HMM_REFIT_INTERVAL
                )
            )
            if len(daily_df) >= HMM_MIN_BARS and need_fit:
                try:
                    _fit_hmm_on_df(detector, daily_df)
                    hmm_fitted = True
                    last_hmm_fit_date = run_date
                    logger.debug("HMM fitted on %s", run_date)
                except Exception as hmm_err:
                    logger.warning("HMM fit failed on %s: %s", run_date, hmm_err)

            if hmm_fitted:
                try:
                    regime = _predict_regime_on_df(detector, daily_df)
                except Exception as regime_err:
                    logger.warning(
                        "HMM predict failed on %s: %s — using default regime",
                        run_date, regime_err,
                    )
                    regime = DEFAULT_REGIME
            else:
                regime = DEFAULT_REGIME

            # Store regime to DB
            try:
                detector.store_regime(run_date, instrument, regime)
            except Exception as store_err:
                logger.warning("store_regime failed on %s: %s", run_date, store_err)

            # (e) Generate signals
            regime_state_id = regime.get("state_id", 1)
            tier1_signals = generate_all_tier1(
                run_date, instrument, regime_state_id, technical
            )
            # Derive proposed direction from tier1 for tier2
            composite_t1 = compute_composite(tier1_signals, {})
            proposed_direction = composite_t1.get("composite_direction", "flat")

            tier2_signals = generate_all_tier2(
                run_date, instrument, technical, proposed_direction
            )

            # Final composite
            composite = compute_composite(tier1_signals, tier2_signals)

            # Store signals
            store_signals(run_date, instrument, tier1_signals, tier2_signals)

            # Build signals summary for the feature vector
            signals_summary = {
                "tier1": tier1_signals,
                "tier2": tier2_signals,
                "composite": composite,
            }

            # (f) Assemble and store feature vector
            vector = assemble_feature_vector(
                run_date, instrument, technical, regime, signals_summary
            )
            store_feature_vector(run_date, instrument, vector)

            successes += 1

        except Exception as exc:
            failures += 1
            logger.error("Day %s FAILED: %s", run_date, exc, exc_info=True)

        # Progress logging every 50 days
        if total % 50 == 0:
            logger.info(
                "Progress: %d / %d days processed  (successes=%d, failures=%d)",
                total, len(all_dates), successes, failures,
            )

    # ---- Summary -----------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Historical feature generation complete for %s", instrument)
    logger.info("  Total days processed : %d", total)
    logger.info("  Successes            : %d", successes)
    logger.info("  Failures             : %d", failures)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_date(value: str):
    """Parse a YYYY-MM-DD string into a datetime.date."""
    return datetime.strptime(value, "%Y-%m-%d").date()


def main():
    parser = argparse.ArgumentParser(
        description="Generate historical feature vectors day-by-day."
    )
    parser.add_argument(
        "--instrument",
        type=str,
        default="EUR_USD",
        help="Instrument name (default: EUR_USD)",
    )
    parser.add_argument(
        "--start",
        type=_parse_date,
        default=None,
        help="Start date in YYYY-MM-DD (default: earliest available)",
    )
    parser.add_argument(
        "--end",
        type=_parse_date,
        default=None,
        help="End date in YYYY-MM-DD (default: latest available)",
    )
    parser.add_argument(
        "--min-bars",
        type=int,
        default=60,
        help="Minimum daily bars before we start generating features (default: 60)",
    )
    args = parser.parse_args()

    run(
        instrument=args.instrument,
        start_date=args.start,
        end_date=args.end,
        min_bars=args.min_bars,
    )


if __name__ == "__main__":
    main()
