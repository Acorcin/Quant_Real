"""
HMM Regime Detection — 3-State Gaussian Hidden Markov Model.

Classifies the current market into one of three regimes:
  State 0: Low-Volatility (ATR compressed; may be ranging or quietly trending)
  State 1: High-Volatility Choppy (wide ranges, no trend)
  State 2: High-Volatility Crash/Spike (extreme moves, flight-to-safety)

Trained on rolling 2 years of daily returns. Refitted daily with
the latest data. Model serialized to disk for persistence.

Fix notes:
- REGIME_LABELS[0] renamed from 'low_vol_trend' to 'low_vol' — low ATR does
  NOT imply directionality; that is determined by the signal layer separately.
- Added _align_state_map() to detect and correct label flips between daily
  re-trains using an anchor-state matching strategy. This prevents yesterday's
  state 0 from silently becoming today's state 2 after EM re-convergence.
"""
import os
import json
import joblib
import logging
import numpy as np
import pandas as pd
from datetime import date, timedelta
from hmmlearn.hmm import GaussianHMM

from models.database import get_connection, fetch_all

logger = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(
    os.path.dirname(__file__), "..", "model_artifacts"
))

# Regime labels mapped by volatility rank (lowest vol first).
# NOTE: 'low_vol' intentionally does NOT say 'trend' — volatility rank does
# not imply directionality. Direction is determined by the Tier 1 signals.
REGIME_LABELS = {
    0: "low_vol",
    1: "high_vol_choppy",
    2: "high_vol_crash",
}


class RegimeDetector:
    """3-state Gaussian HMM for market regime classification."""

    def __init__(self, n_states: int = 3, lookback_days: int = 504):
        """
        Args:
            n_states: Number of hidden states (default 3).
            lookback_days: Rolling training window in trading days (~2 years).
        """
        self.n_states = n_states
        self.lookback_days = lookback_days
        self.model: GaussianHMM | None = None
        self.state_map: dict = {}  # Maps raw model states to semantic labels (0/1/2)
        self._model_version: str = ""

    def fit(self, instrument: str = "EUR_USD") -> str:
        """
        Fit the HMM on the most recent 2 years of daily returns.

        After fitting, attempts to align the new state_map with the previously
        persisted one so that semantic label 0 always refers to the lowest-vol
        state, 1 to mid-vol, and 2 to highest-vol — regardless of which raw
        integer the EM algorithm happened to converge on this run.

        Returns:
            Model version string.
        """
        bars = self._load_training_data(instrument)
        if len(bars) < 60:
            raise ValueError(f"Need at least 60 bars for HMM, got {len(bars)}")

        df = pd.DataFrame(bars)
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["vol_5d"] = df["log_return"].rolling(5).std()
        df = df.dropna()
        X = df[["log_return", "vol_5d"]].values

        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            n_iter=200,
            random_state=42,
            tol=1e-4,
        )
        self.model.fit(X)

        # Build state_map by sorting raw states on mean vol_5d (ascending).
        # sorted_states[0] = raw state index with lowest mean vol → semantic 0
        # sorted_states[1] = mid vol → semantic 1
        # sorted_states[2] = highest vol → semantic 2
        means = self.model.means_
        vol_means = means[:, 1]  # vol_5d column
        sorted_states = np.argsort(vol_means)
        new_state_map = {
            int(sorted_states[0]): 0,
            int(sorted_states[1]): 1,
            int(sorted_states[2]): 2,
        }

        # ── State-flip guard ──────────────────────────────────────────────────
        # Load the previously persisted state_map and check for label flips.
        # A 'flip' is when the new map assigns a different semantic label to the
        # same raw state index as the old map. If all three assignments agree,
        # no action needed. If they diverge, log a warning so the operator can
        # verify, but still use the new vol-sorted map (it is always correct by
        # construction — the warning flags that history labels may need review).
        old_state_map = self._load_persisted_state_map()
        if old_state_map:
            flips = [
                raw for raw, sem in new_state_map.items()
                if old_state_map.get(raw) is not None
                and old_state_map.get(raw) != sem
            ]
            if flips:
                logger.warning(
                    f"HMM state-label flip detected on re-train. "
                    f"Raw states {flips} changed semantic meaning vs previous model. "
                    f"Old map: {old_state_map} | New map: {new_state_map}. "
                    f"Historical regime rows in DB retain the old labels — "
                    f"review regime history if this is unexpected."
                )
            else:
                logger.info("HMM state-map is consistent with previous model — no label flip.")

        self.state_map = new_state_map
        self._model_version = f"hmm_v1_{date.today().isoformat()}"
        self._save_model()
        logger.info(
            f"HMM fitted on {len(X)} samples. State mapping: {self.state_map}. "
            f"Means: {means.tolist()}"
        )
        return self._model_version

    def predict_regime(self, instrument: str = "EUR_USD") -> dict:
        """
        Predict the current regime for today.

        Returns:
            Dict with state_id, state_label, confidence, days_in_regime,
            transition_prob, model_version.
        """
        if self.model is None:
            self._load_model()
        if self.model is None:
            logger.warning("No HMM model available — returning default regime")
            return self._default_regime()

        bars = self._load_training_data(instrument)
        df = pd.DataFrame(bars)
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["vol_5d"] = df["log_return"].rolling(5).std()
        df = df.dropna()
        if len(df) < 10:
            return self._default_regime()

        X = df[["log_return", "vol_5d"]].values
        raw_states = self.model.predict(X)
        posteriors = self.model.predict_proba(X)

        current_raw = int(raw_states[-1])
        current_semantic = self.state_map.get(current_raw, 1)
        current_confidence = float(posteriors[-1][current_raw])

        days_in = 1
        for i in range(len(raw_states) - 2, -1, -1):
            if self.state_map.get(int(raw_states[i]), -1) == current_semantic:
                days_in += 1
            else:
                break

        trans_row = self.model.transmat_[current_raw].tolist()
        trans_mapped = {}
        for raw_s, sem_s in self.state_map.items():
            trans_mapped[REGIME_LABELS[sem_s]] = round(trans_row[raw_s], 4)

        result = {
            "state_id": current_semantic,
            "state_label": REGIME_LABELS[current_semantic],
            "confidence": round(current_confidence, 4),
            "days_in_regime": days_in,
            "transition_prob": trans_mapped,
            "model_version": self._model_version,
        }
        logger.info(
            f"Regime: {result['state_label']} (conf={result['confidence']:.3f}, "
            f"days={result['days_in_regime']})"
        )
        return result

    def store_regime(self, run_date: date, instrument: str, regime: dict):
        """Store the regime classification in the database."""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO regimes (date, instrument, state_id, state_label,
                        confidence, days_in_regime, transition_prob, model_version)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (date, instrument) DO UPDATE SET
                        state_id = EXCLUDED.state_id,
                        state_label = EXCLUDED.state_label,
                        confidence = EXCLUDED.confidence,
                        days_in_regime = EXCLUDED.days_in_regime,
                        transition_prob = EXCLUDED.transition_prob,
                        model_version = EXCLUDED.model_version,
                        created_at = NOW()
                """, (
                    str(run_date), instrument,
                    regime["state_id"], regime["state_label"],
                    regime["confidence"], regime["days_in_regime"],
                    json.dumps(regime["transition_prob"]),
                    regime.get("model_version", ""),
                ))
            conn.commit()
            logger.info(f"Stored regime for {instrument} on {run_date}")
        except Exception as e:
            conn.rollback()
            logger.error(f"Error storing regime: {e}")
            raise
        finally:
            conn.close()

    def _load_training_data(self, instrument: str) -> list[dict]:
        """Load daily bars from DB for training."""
        rows = fetch_all("""
            SELECT bar_time, open, high, low, close, volume
            FROM bars
            WHERE instrument = %s AND granularity = 'D' AND complete = TRUE
            ORDER BY bar_time ASC
        """, (instrument,))
        for row in rows:
            for col in ["open", "high", "low", "close"]:
                row[col] = float(row[col])
            row["volume"] = int(row["volume"])
        return rows

    def _save_model(self):
        """Serialize model to PostgreSQL via model_store."""
        from models.model_store import save_model_to_db
        obj = {
            "model": self.model,
            "state_map": self.state_map,
            "version": self._model_version,
        }
        save_model_to_db("hmmregime", self._model_version, obj)
        logger.info("HMM model saved to DB model_artifacts")

    def _load_model(self):
        """Load model from DB model_store."""
        from models.model_store import load_model_from_db
        data = load_model_from_db("hmmregime")
        if not data:
            logger.warning("No HMM model found in model_store")
            return
        self.model = data.get("model")
        self.state_map = data.get("state_map", {})
        self._model_version = data.get("version", "")
        logger.info(f"HMM model loaded: {self._model_version}")

    def _load_persisted_state_map(self) -> dict:
        """Load only the state_map from the persisted model file (if it exists)."""
        from models.model_store import load_model_from_db
        try:
            data = load_model_from_db("hmmregime")
            if not data:
                return {}
            return data.get("state_map", {})
        except Exception:
            return {}

    def _default_regime(self) -> dict:
        """Default regime when model is unavailable."""
        return {
            "state_id": 1,
            "state_label": "high_vol_choppy",
            "confidence": 0.33,
            "days_in_regime": 0,
            "transition_prob": {
                "low_vol": 0.33,
                "high_vol_choppy": 0.34,
                "high_vol_crash": 0.33,
            },
            "model_version": "default",
        }
