"""
Meta-Model: XGBoost Binary Classifier with CPCV Validation.

This is Model 2 in the Lopez de Prado meta-labeling framework.
The primary signals (Model 1) propose a direction (long/short).
This model predicts the PROBABILITY that the proposed trade
will actually be profitable tomorrow (T+1).

- Probability < 0.55 → FLAT (no trade)
- Probability 0.55–0.70 → half position
- Probability > 0.70 → full position

Validation: Purged Combinatorial Cross-Validation (CPCV)
  N=6 groups, k=2 → 15 backtest paths
  Purge: 5 days | Embargo: 2 days
"""
import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import date
from itertools import combinations

from scipy import stats
from sklearn.preprocessing import StandardScaler
try:
    from utils.trading_calendar import next_trading_day
except Exception as exc:  # pragma: no cover - defensive fallback
    logger = logging.getLogger(__name__)
    logger.warning("utils.trading_calendar import failed, using weekday fallback: %s", exc)

    def next_trading_day(run_date: date) -> date:
        """Fallback next-trading-day helper (weekends only)."""
        from datetime import timedelta

        candidate = run_date + timedelta(days=1)
        while candidate.weekday() >= 5:
            candidate += timedelta(days=1)
        return candidate

logger = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(
    os.path.dirname(__file__), "..", "model_artifacts"
))

# Feature columns in the order expected by the model.
# Intraday CME futures configuration: FX carry (swap pips) and sentiment
# features (OANDA retail, Perplexity AI) removed; Kronos foundation-model
# outputs and offline Databento L3 placeholders added.
FEATURE_COLS = [
    "regime_state", "days_in_regime",
    "yield_spread_bps", "yield_spread_change_5d", "yield_spread_change_20d",
    "kronos_p50_scaled", "kronos_uncertainty",
    "kronos_context_vol", "kronos_regime_encoded",
    "l3_order_book_imbalance", "l3_vpin",
    "atr_14", "rsi_14", "price_vs_ma50", "price_vs_ma200",
    "body_direction", "body_pct_of_range",
    "eod_event_reversal", "event_surprise_magnitude",
    "day_of_week", "is_friday",
    "primary_signal_direction", "primary_signal_count",
    "composite_strength", "tier2_confirmation_count",
]

# Kronos structural regime encoding (must match features.vector.KRONOS_REGIME_MAP)
KRONOS_REGIME_EFFICIENT = 0


def _get_xgb_device() -> str:
    """Return 'cuda' if a CUDA-capable GPU is visible, else 'cpu'.

    Uses a lightweight xgboost probe so the rest of the codebase stays
    dependency-free — no torch / cupy required.
    """
    try:
        import xgboost as xgb  # noqa: F401 — already a hard dep
        import numpy as np
        probe = xgb.XGBClassifier(tree_method="hist", device="cuda", n_estimators=1)
        probe.fit(np.array([[1]]), np.array([0]))
        logger.info("XGBoost GPU (CUDA) detected — training on GPU.")
        return "cuda"
    except Exception:
        logger.info("XGBoost GPU not available — training on CPU.")
        return "cpu"


class MetaModel:
    """XGBoost meta-labeling model with CPCV validation."""

    def __init__(self):
        self.model = None
        self.scaler: StandardScaler | None = None
        self.model_version: str = ""
        self.cpcv_results: dict = {}
        self.shap_importance: list = []

    def train_from_db(
        self,
        instrument: str = "EUR_USD",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict:
        """
        Load historical feature vectors and bars from Postgres, generate binary
        labels based on realized next-day returns, and train the XGBoost meta-model.
        """
        from models.database import fetch_all
        from datetime import date
        
        # 1. Fetch feature vectors
        query = "SELECT date, features FROM feature_vectors WHERE instrument = %s"
        params = [instrument]
        if start_date:
            query += " AND date >= %s"
            params.append(start_date)
        if end_date:
            query += " AND date <= %s"
            params.append(end_date)
        query += " ORDER BY date ASC"
        
        fv_rows = fetch_all(query, tuple(params))
        if not fv_rows:
            raise ValueError(f"No feature vectors found for {instrument} in date range")
            
        # 2. Fetch daily bars close prices to calculate next-day returns
        bars_rows = fetch_all(
            """SELECT bar_time::date as d, open, close 
               FROM bars 
               WHERE instrument = %s AND granularity = 'D' AND complete = TRUE
               ORDER BY bar_time ASC""",
            (instrument,)
        )
        if not bars_rows:
            raise ValueError(f"No daily bars found for {instrument} to generate labels")
            
        close_prices = {r["d"]: float(r["close"]) for r in bars_rows}
        open_prices = {r["d"]: float(r["open"]) for r in bars_rows}
        
        # Map next trading day prices to calculate realized close-to-close returns
        # We enter at the Open of T+1, and exit at the Close of T+1.
        # Realized return = Close[T+1]/Open[T+1] - 1.0
        # So for a features vector date T, the trade is on T+1.
        feature_vectors = []
        labels = []
        sample_dates = []
        
        for row in fv_rows:
            d = row["date"]
            features = row["features"]
            if not features:
                continue
                
            nd = next_trading_day(d)
            guard = 0
            while nd not in open_prices and guard < 30:
                nd = next_trading_day(nd)
                guard += 1
                
            if nd not in open_prices or nd not in close_prices:
                continue
                
            open_price = open_prices[nd]
            close_price = close_prices[nd]
            ret = (close_price / open_price) - 1.0
            
            # Get proposed direction from primary signal
            # 1 = long, -1 = short, 0 = flat
            direction = features.get("primary_signal_direction", 0)
            if direction == 0:
                continue # No trade proposed, so no label to train on
                
            # Label is 1 if trade was profitable, else 0
            if direction > 0:
                label = 1 if ret > 0 else 0
            else:
                label = 1 if ret < 0 else 0
                
            feature_vectors.append(features)
            labels.append(label)
            sample_dates.append(d)
            
        if len(feature_vectors) < 50:
            raise ValueError(f"Need at least 50 trade samples to train, got {len(feature_vectors)}")
            
        return self.train(
            feature_vectors=feature_vectors,
            labels=labels,
            sample_dates=sample_dates,
            instrument=instrument,
            allow_synthetic_return_proxy=False
        )

    def train(
        self,
        feature_vectors: list[dict],
        labels: list[int],
        sample_dates: list[date] | None = None,
        instrument: str | None = None,
        allow_synthetic_return_proxy: bool = False,
    ) -> dict:
        """
        Train the XGBoost model on historical feature vectors.

        Args:
            feature_vectors: List of feature dicts (len(FEATURE_COLS) features each).
            labels: Binary labels (1=profitable, 0=not).
            sample_dates: One calendar date per row (same order as feature_vectors).
                Required for CPCV to use realized next-day returns from `bars`.
            instrument: e.g. EUR_USD; used with sample_dates to load bar returns.
            allow_synthetic_return_proxy: Allow CPCV to fall back to fixed ±0.1% returns
                when realized bar returns are unavailable. Defaults to False.

        Returns:
            Training results dict with CPCV validation and clearly labeled in-sample fit.
        """
        import xgboost as xgb
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        df = pd.DataFrame(feature_vectors)

        # Ensure all expected columns exist
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0

        X_raw = df[FEATURE_COLS].fillna(0).values
        y = np.array(labels)

        if len(X_raw) < 50:
            raise ValueError(f"Need at least 50 samples for training, got {len(X_raw)}")

        if sample_dates is not None and len(sample_dates) != len(feature_vectors):
            raise ValueError("sample_dates must match feature_vectors length when provided")
        if (sample_dates is None or not instrument) and not allow_synthetic_return_proxy:
            raise ValueError(
                "CPCV requires sample_dates + instrument for realized returns. "
                "Pass allow_synthetic_return_proxy=True to use the simplified fallback."
            )

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X_raw)

        # CPCV validation
        cpcv = self._run_cpcv(
            X_raw,
            y,
            n_groups=6,
            k_test=2,
            purge=5,
            embargo=2,
            sample_dates=sample_dates,
            instrument=instrument,
            allow_synthetic_return_proxy=allow_synthetic_return_proxy,
        )

        # Detect GPU availability for XGBoost acceleration
        _xgb_device = _get_xgb_device()

        # Train final model on all data
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=5,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
            tree_method="hist",
            device=_xgb_device,
        )
        self.model.fit(X, y)

        # Full-dataset fit (optimistic — same data used to train and evaluate)
        y_pred = self.model.predict(X)
        in_sample_train_metrics = {
            "accuracy": round(accuracy_score(y, y_pred), 4),
            "precision": round(precision_score(y, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y, y_pred, zero_division=0), 4),
            "samples": len(X),
            "positive_rate": round(y.mean(), 4),
        }

        # SHAP importance
        self.shap_importance = self._compute_shap(X)

        self.model_version = f"xgb_v1_{date.today().isoformat()}"
        self.cpcv_results = cpcv
        self._save_model()

        logger.info(
            f"Meta-model trained (in-sample metrics for diagnostics only): {in_sample_train_metrics}. "
            f"CPCV PSR: {cpcv.get('probabilistic_sharpe_ratio', 'N/A')}"
        )

        from models.database import get_connection
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO model_runs (
                        run_date, model_type, model_version, training_samples,
                        cpcv_results, shap_importance, metrics, model_path
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    date.today(), "xgboost_meta", self.model_version, len(X),
                    json.dumps(cpcv), json.dumps(self.shap_importance),
                    json.dumps(in_sample_train_metrics), "db://model_artifacts/metamodel"
                ))
            conn.commit()
            logger.info("Successfully recorded training run in model_runs table")
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to record model_run to DB: {e}")
        finally:
            conn.close()

        return {
            "model_version": self.model_version,
            "in_sample_train_metrics": in_sample_train_metrics,
            "cpcv": cpcv,
            "top_features": self.shap_importance[:10],
        }

    def predict(self, feature_vector: dict, threshold_half: float = 0.55, threshold_full: float = 0.70) -> dict:
        """
        Predict probability of profitable trade for a single day.

        Args:
            feature_vector: Dict of features (see FEATURE_COLS).
            threshold_half: Overriding probability gate for half size (default 0.55)
            threshold_full: Overriding probability gate for full size (default 0.70)

        Returns:
            Dict with direction, probability, size_multiplier, top_shap.
        """
        # ─── Kronos structural veto gate ───
        # If the forecasting pipeline reads the market as an 'efficient' random
        # walk, no amount of XGBoost probability is trustworthy — the measured
        # structure says there is nothing to predict. Force flat, bypassing all
        # other logic. Fires ONLY on an explicit 'efficient' reading: missing /
        # unknown regimes encode as -1 (see features.vector.KRONOS_REGIME_MAP),
        # so legacy vectors without Kronos features are never vetoed.
        kronos_regime = feature_vector.get("kronos_regime_encoded", -1)
        try:
            kronos_regime = int(kronos_regime)
        except (TypeError, ValueError):
            kronos_regime = -1
        if kronos_regime == KRONOS_REGIME_EFFICIENT:
            logger.info(
                "Kronos structural veto: regime 'efficient' — forcing flat "
                "(XGBoost bypassed)"
            )
            return {
                "direction": "flat",
                "probability": 0.0,
                "size_multiplier": 0.0,
                "model_version": self.model_version or "structural_veto",
                "top_shap": [],
                "structural_veto": True,
            }

        if self.model is None:
            self._load_model()

        if self.model is None:
            logger.warning("No meta-model available — returning default prediction")
            return self._default_prediction(feature_vector)

        # Build feature array
        X = np.array([[feature_vector.get(col, 0.0) for col in FEATURE_COLS]], dtype=float)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        prob = float(self.model.predict_proba(X)[0][1])

        # Position sizing from probability
        if prob < threshold_half:
            size_mult = 0.0
            direction = "flat"
        elif prob < threshold_full:
            size_mult = 0.5
            direction = "long" if feature_vector.get("primary_signal_direction", 0) > 0 else "short"
        else:
            size_mult = 1.0
            direction = "long" if feature_vector.get("primary_signal_direction", 0) > 0 else "short"

        # If primary signal is flat, meta-model can't override
        if feature_vector.get("primary_signal_direction", 0) == 0:
            direction = "flat"
            size_mult = 0.0

        result = {
            "direction": direction,
            "probability": round(prob, 4),
            "size_multiplier": size_mult,
            "model_version": self.model_version,
            "top_shap": self.shap_importance[:5] if self.shap_importance else [],
        }

        logger.info(
            f"Meta prediction: {direction} (prob={prob:.3f}, size={size_mult}x)"
        )
        return result

    def _run_cpcv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_groups: int = 6,
        k_test: int = 2,
        purge: int = 5,
        embargo: int = 2,
        sample_dates: list[date] | None = None,
        instrument: str | None = None,
        allow_synthetic_return_proxy: bool = False,
    ) -> dict:
        """
        Purged Combinatorial Cross-Validation.

        N=6 groups, k=2 test → C(6,2) = 15 backtest paths.
        Each path has a purge window of 5 days and embargo of 2 days
        to prevent label leakage.
        """
        import xgboost as xgb

        n = len(X)
        returns_all = None
        if sample_dates is not None and instrument:
            returns_all = _next_trading_day_pct_returns(instrument, sample_dates)
            if returns_all is None or np.all(np.isnan(returns_all)):
                returns_all = None
        if returns_all is None and not allow_synthetic_return_proxy:
            return {
                "paths_tested": 0,
                "error": (
                    "Realized returns unavailable for CPCV. Provide sample_dates + instrument "
                    "or enable allow_synthetic_return_proxy."
                ),
            }
        if returns_all is None:
            logger.warning("CPCV using fixed 0.1%% synthetic return proxy")
        group_size = n // n_groups
        groups = []
        for i in range(n_groups):
            start = i * group_size
            end = start + group_size if i < n_groups - 1 else n
            groups.append(list(range(start, end)))

        test_combos = list(combinations(range(n_groups), k_test))
        path_returns = []
        all_path_daily_returns: list[float] = []

        for combo in test_combos:
            test_idx = set()
            for g in combo:
                test_idx.update(groups[g])

            # Purge + embargo: remove samples near test boundaries
            purged = set()
            for idx in test_idx:
                for offset in range(-purge, embargo + 1):
                    purged.add(idx + offset)

            train_idx = [i for i in range(n) if i not in test_idx and i not in purged]
            test_idx = sorted(test_idx)

            if len(train_idx) < 30 or len(test_idx) < 10:
                continue

            scaler = StandardScaler()
            X_train_raw, y_train = X[train_idx], y[train_idx]
            X_test_raw, y_test = X[test_idx], y[test_idx]
            X_train = scaler.fit_transform(X_train_raw)
            X_test = scaler.transform(X_test_raw)

            model = xgb.XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, use_label_encoder=False,
                eval_metric="logloss",
                tree_method="hist",
                device=_get_xgb_device(),
            )
            model.fit(X_train, y_train, verbose=False)

            probs = model.predict_proba(X_test)[:, 1]
            # Simulate returns: if prob > 0.55, take the trade
            signals = (probs > 0.55).astype(float)
            if returns_all is not None:
                r_next = returns_all[test_idx]
                r_next = np.nan_to_num(r_next, nan=0.0)
                # Same sign convention as the old proxy: label 1 → +|r|, 0 → −|r|
                daily_returns = signals * (2 * y_test - 1) * np.abs(r_next)
            else:
                daily_returns = signals * (2 * y_test - 1) * 0.001

            all_path_daily_returns.extend(daily_returns.tolist())

            if daily_returns.std() > 0:
                sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
            else:
                sharpe = 0.0
            path_returns.append(sharpe)

        if not path_returns:
            return {"paths_tested": 0, "error": "Insufficient data for CPCV"}

        sharpe_array = np.array(path_returns)
        sharpe_mean = float(sharpe_array.mean())
        sharpe_std = float(sharpe_array.std(ddof=1)) if len(sharpe_array) > 1 else 0.0

        n_paths = len(sharpe_array)
        if n_paths > 1 and sharpe_std > 1e-12:
            t_res = stats.ttest_1samp(sharpe_array, 0.0, alternative="greater")
            path_t_stat = float(t_res.statistic)
            path_p_value = float(t_res.pvalue)
            if not np.isfinite(path_p_value):
                path_p_value = 1.0
        else:
            path_t_stat = 0.0
            path_p_value = 1.0

        # Bailey & López de Prado — Probabilistic Sharpe Ratio (skew/kurtosis of returns)
        psr = None
        if all_path_daily_returns and len(all_path_daily_returns) > 2:
            r = np.array(all_path_daily_returns, dtype=float)
            r = r[np.isfinite(r)]
            if len(r) > 2 and np.std(r, ddof=1) > 0:
                psr = _probabilistic_sharpe_ratio_from_returns(r)

        return {
            "paths_tested": n_paths,
            "sharpe_mean": round(sharpe_mean, 4),
            "sharpe_std": round(sharpe_std, 4),
            "path_sharpe_t_statistic": round(path_t_stat, 4),
            "path_sharpe_p_value": round(path_p_value, 6),
            "probabilistic_sharpe_ratio": round(psr, 4) if psr is not None else None,
            "uses_synthetic_returns": returns_all is None,
            "statistically_significant": (path_p_value < 0.05) and (sharpe_mean > 0),
            "path_sharpes": [round(s, 4) for s in path_returns],
        }

    def _compute_shap(self, X: np.ndarray) -> list[dict]:
        """Compute SHAP feature importance."""
        try:
            import shap
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)
            mean_abs = np.abs(shap_values).mean(axis=0)

            importance = []
            for i, col in enumerate(FEATURE_COLS):
                importance.append({
                    "feature": col,
                    "mean_abs_shap": round(float(mean_abs[i]), 6),
                })
            importance.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
            return importance
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            # Fallback to XGBoost native importance
            if self.model:
                imp = self.model.feature_importances_
                importance = [
                    {"feature": col, "mean_abs_shap": round(float(imp[i]), 6)}
                    for i, col in enumerate(FEATURE_COLS)
                ]
                importance.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
                return importance
            return []

    def store_prediction(self, run_date: date, instrument: str,
                         prediction: dict, regime_state: int,
                         composite_strength: float):
        """Store the prediction in the database."""
        from models.database import get_connection
        import json

        prediction_for = next_trading_day(run_date)

        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO predictions (date, prediction_for, instrument,
                        direction, probability, size_multiplier, model_version,
                        top_shap, regime_state, composite_strength)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (date, instrument) DO UPDATE SET
                        prediction_for = EXCLUDED.prediction_for,
                        direction = EXCLUDED.direction,
                        probability = EXCLUDED.probability,
                        size_multiplier = EXCLUDED.size_multiplier,
                        model_version = EXCLUDED.model_version,
                        top_shap = EXCLUDED.top_shap,
                        regime_state = EXCLUDED.regime_state,
                        composite_strength = EXCLUDED.composite_strength,
                        created_at = NOW()
                """, (
                    str(run_date), str(prediction_for), instrument,
                    prediction["direction"],
                    prediction["probability"],
                    prediction["size_multiplier"],
                    prediction.get("model_version", ""),
                    json.dumps(prediction.get("top_shap", [])),
                    regime_state,
                    composite_strength,
                ))
            conn.commit()
            logger.info(f"Stored prediction for {instrument} on {run_date}")
        except Exception as e:
            conn.rollback()
            logger.error(f"Error storing prediction: {e}")
            raise
        finally:
            conn.close()

    def _save_model(self):
        """Serialize model to DB via model_store."""
        from models.model_store import save_model_to_db
        obj = {
            "model": self.model,
            "scaler": self.scaler,
            "version": self.model_version,
            "cpcv": self.cpcv_results,
            "shap": self.shap_importance,
            "feature_cols": FEATURE_COLS,
        }
        save_model_to_db("metamodel", self.model_version, obj)
        logger.info("Meta-model saved to DB model_store")

    def _load_model(self):
        """Load model from DB via model_store."""
        from models.model_store import load_model_from_db
        data = load_model_from_db("metamodel")
        if not data:
            logger.warning("No meta-model found in model_store")
            return
            
        self.model = data.get("model")
        self.scaler = data.get("scaler")
        self.model_version = data.get("version")
        self.cpcv_results = data.get("cpcv", {})
        self.shap_importance = data.get("shap", [])
        logger.info(f"Meta-model loaded: {self.model_version}")

    def _default_prediction(self, feature_vector: dict) -> dict:
        """Default when no model is available — conservative flat."""
        return {
            "direction": "flat",
            "probability": 0.50,
            "size_multiplier": 0.0,
            "model_version": "no_model",
            "top_shap": [],
        }


def _normalize_date(d):
    if isinstance(d, date):
        return d
    if isinstance(d, str):
        from datetime import datetime as dt

        return dt.fromisoformat(d).date()
    return d


def _next_trading_day_pct_returns(instrument: str, sample_dates: list) -> np.ndarray | None:
    """Close-to-close return from each bar date to the next trading session (OANDA daily)."""
    from models.database import fetch_all

    rows = fetch_all(
        """SELECT bar_time::date AS d, close FROM bars
           WHERE instrument = %s AND granularity = 'D' AND complete = TRUE
           ORDER BY bar_time ASC""",
        (instrument,),
    )
    if not rows:
        return None
    close_by_date = {r["d"]: float(r["close"]) for r in rows}
    out = np.zeros(len(sample_dates))
    for i, raw in enumerate(sample_dates):
        d = _normalize_date(raw)
        if d not in close_by_date:
            out[i] = np.nan
            continue
        nd = next_trading_day(d)
        guard = 0
        while nd not in close_by_date and guard < 30:
            nd = next_trading_day(nd)
            guard += 1
        if nd not in close_by_date:
            out[i] = np.nan
        else:
            out[i] = (close_by_date[nd] / close_by_date[d]) - 1.0
    return out


def _probabilistic_sharpe_ratio_from_returns(returns: np.ndarray) -> float:
    """
    Bailey & López de Prado Probabilistic Sharpe Ratio (approx.),
    using skewness and kurtosis of the return series (AFML Ch. 14).
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    T = len(r)
    if T < 3:
        return 0.5
    m = np.mean(r)
    s = np.std(r, ddof=1)
    if s <= 0:
        return 0.5
    sr = (m / s) * np.sqrt(252.0)
    skew = float(stats.skew(r, bias=False))
    kurt_excess = float(stats.kurtosis(r, fisher=True, bias=False))
    var_sr = (
        1.0
        + 0.5 * sr ** 2
        - skew * sr
        + (kurt_excess / 4.0) * sr ** 2
    ) / max(T - 1, 1)
    var_sr = max(var_sr, 1e-12)
    z = sr / np.sqrt(var_sr)
    return float(stats.norm.cdf(z))
