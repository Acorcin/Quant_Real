#!/usr/bin/env python3
"""
walkforward_test.py

Runs a tick-by-tick out-of-sample walkforward test over the specified period.
Loads the trained meta-model and simulates trade execution using raw CME MDP3 tick data.
"""

import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import logging
import csv
import io
import json
import glob
from datetime import datetime, date, timedelta, time, timezone
from zoneinfo import ZoneInfo
from collections import defaultdict
from decimal import Decimal
import numpy as np
import pandas as pd
import zstandard as zstd

from models.database import fetch_all, get_connection, fetch_one
from models.meta_model import MetaModel
from models.hmm_regime import RegimeDetector, REGIME_LABELS
from features.technical import compute_all_features
from features.vector import assemble_feature_vector
from signals.tier1 import generate_all_tier1
from signals.tier2 import generate_all_tier2
from signals.composite import compute_composite
from utils.trading_calendar import next_trading_day

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

NY_TZ = ZoneInfo("America/New_York")
PRICE_SCALE = 1e-9
HMM_MIN_BARS = 100
HMM_REFIT_INTERVAL = 30


def _decimals_to_floats(df: pd.DataFrame) -> pd.DataFrame:
    """Convert any Decimal columns in *df* to float."""
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(1)
            if not sample.empty and isinstance(sample.iloc[0], Decimal):
                df[col] = df[col].apply(
                    lambda v: float(v) if isinstance(v, Decimal) else v
                )
    return df


def _parse_timestamp(ts_str: str) -> datetime:
    """Parse Databento timestamp with 9 decimal places into microsecond UTC datetime."""
    ts_clean = ts_str.rstrip("Z")
    if "." in ts_clean:
        date_part, frac = ts_clean.split(".", 1)
        frac = frac[:6].ljust(6, "0")
        ts_clean = f"{date_part}.{frac}"
    return datetime.fromisoformat(ts_clean).replace(tzinfo=ZoneInfo("UTC"))


def _parse_price(price_str: str) -> float:
    """Parse price string into float (pretty price or raw scaled)."""
    if "." in price_str:
        return float(price_str)
    return int(price_str) * PRICE_SCALE


def _fetch_oos_dates(instrument: str, start_date: date, end_date: date) -> list:
    """Fetch OOS dates that have complete daily bars."""
    rows = fetch_all(
        """
        SELECT DISTINCT date(bar_time) AS bar_date
        FROM bars
        WHERE instrument = %s
          AND granularity = 'D'
          AND complete = TRUE
          AND date(bar_time) BETWEEN %s AND %s
        ORDER BY bar_date ASC
        """,
        (instrument, start_date, end_date),
    )
    return [r['bar_date'] for r in rows]


def _load_daily_bars(instrument: str, up_to_date) -> pd.DataFrame:
    """Load daily bars up to and including *up_to_date*."""
    rows = fetch_all(
        """
        SELECT bar_time, open, high, low, close, volume
        FROM bars
        WHERE instrument = %s
          AND granularity = 'D'
          AND complete = TRUE
          AND date(bar_time) <= %s
        ORDER BY bar_time ASC
        """,
        (instrument, up_to_date),
    )
    df = pd.DataFrame(rows, columns=["bar_time", "open", "high", "low", "close", "volume"])
    return _decimals_to_floats(df)


def _load_h4_bars(instrument: str, run_date, lookback_days: int = 30) -> pd.DataFrame:
    """Load H4 bars up to 5 PM NY on *run_date* for the last *lookback_days*."""
    start_date = run_date - timedelta(days=lookback_days)
    cutoff_dt = datetime.combine(run_date, time(17, 0), tzinfo=NY_TZ)
    rows = fetch_all(
        """
        SELECT bar_time, open, high, low, close, volume
        FROM bars
        WHERE instrument = %s
          AND granularity = 'H4'
          AND complete = TRUE
          AND bar_time BETWEEN %s AND %s
        ORDER BY bar_time DESC
        LIMIT 180
        """,
        (instrument, start_date, cutoff_dt),
    )
    # Reverse to ensure chronological order
    rows = list(reversed(rows))
    df = pd.DataFrame(rows, columns=["bar_time", "open", "high", "low", "close", "volume"])
    return _decimals_to_floats(df)


def _fit_hmm_on_df(detector, df: pd.DataFrame) -> None:
    """Fit HMM on the last lookback_days of *df* in a lookahead-free way."""
    train_df = df.tail(detector.lookback_days).copy()
    train_df["log_return"] = np.log(train_df["close"] / train_df["close"].shift(1))
    train_df["vol_5d"] = train_df["log_return"].rolling(5).std()
    train_df = train_df.dropna()
    
    if len(train_df) < 60:
        raise ValueError(f"Need at least 60 valid bars for HMM training, got {len(train_df)}")
        
    X = train_df[["log_return", "vol_5d"]].values
    from hmmlearn.hmm import GaussianHMM
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


def _get_tick_filepath(data_dir: str, target_date: date) -> str | None:
    """Return path to tick data file for target_date if it exists."""
    date_str = target_date.strftime("%Y%m%d")
    filename = f"glbx-mdp3-{date_str}.mbp-1.csv.zst"
    path = os.path.join(data_dir, filename)
    if os.path.exists(path):
        return path
    return None


def simulate_intraday_trade(
    filepath: str, direction: str, initial_equity: float, size_multiplier: float
) -> dict:
    """
    Load tick data from filepath, filter to front-month trades,
    simulate entry/exit, track intraday MAE/MFE, and calculate P&L.
    """
    trades = []
    symbol_counts = defaultdict(int)
    
    dctx = zstd.ZstdDecompressor()
    with open(filepath, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            csv_reader = csv.DictReader(text_stream)
            for row in csv_reader:
                if row.get("action") != "T":
                    continue
                symbol = row.get("symbol", "")
                if "-" in symbol:
                    continue
                
                ts_event = _parse_timestamp(row["ts_event"])
                price = _parse_price(row["price"])
                size = int(row["size"])
                
                bid_px = float(row.get("bid_px_00") or 0.0) if row.get("bid_px_00") else None
                ask_px = float(row.get("ask_px_00") or 0.0) if row.get("ask_px_00") else None
                
                symbol_counts[symbol] += 1
                trades.append({
                    "ts_event": ts_event,
                    "symbol": symbol,
                    "price": price,
                    "size": size,
                    "bid_px": bid_px,
                    "ask_px": ask_px,
                })
                
    if not trades:
        return {"error": "No trades found in file"}
        
    front_month = max(symbol_counts, key=symbol_counts.get)
    fm_trades = [t for t in trades if t["symbol"] == front_month]
    if not fm_trades:
        return {"error": f"No trades for front month {front_month}"}
        
    fm_trades.sort(key=lambda x: x["ts_event"])
    
    entry_tick = fm_trades[0]
    entry_price = entry_tick["price"]
    exit_tick = fm_trades[-1]
    exit_price = exit_tick["price"]
    
    bid = entry_tick["bid_px"]
    ask = entry_tick["ask_px"]
    if bid and ask and ask > bid:
        spread_price = ask - bid
    else:
        spread_price = 0.00015  # 1.5 pips fallback
        
    spread_pct = spread_price / entry_price
    
    # Calculate excursions
    max_pnl = -999.0
    min_pnl = 999.0
    for t in fm_trades:
        price = t["price"]
        if direction == "long":
            pnl = (price - entry_price) / entry_price
        else:
            pnl = (entry_price - price) / entry_price
        if pnl > max_pnl:
            max_pnl = pnl
        if pnl < min_pnl:
            min_pnl = pnl
            
    mfe = max(0.0, max_pnl)
    mae = min(0.0, min_pnl)
    
    if direction == "long":
        raw_return = (exit_price - entry_price) / entry_price
    else:
        raw_return = (entry_price - exit_price) / entry_price
        
    trade_return_pct = raw_return - spread_pct
    leverage = 10.0
    leveraged_return = trade_return_pct * leverage * size_multiplier
    pnl_usd = leveraged_return * initial_equity
    
    return {
        "symbol": front_month,
        "entry_time": entry_tick["ts_event"].isoformat(),
        "exit_time": exit_tick["ts_event"].isoformat(),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "spread_pips": spread_price * 10000.0,
        "raw_return_pct": raw_return * 100.0,
        "spread_pct": spread_pct * 100.0,
        "trade_return_pct": trade_return_pct * 100.0,
        "leveraged_return_pct": leveraged_return * 100.0,
        "pnl_usd": pnl_usd,
        "mfe_pct": mfe * 100.0,
        "mae_pct": mae * 100.0,
    }


def _max_drawdown(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    mdd = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        dd = (v / peak) - 1.0 if peak > 0 else 0.0
        mdd = min(mdd, dd)
    return float(mdd)


def _annualized_sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if std <= 1e-12:
        return 0.0
    return float(np.mean(returns) / std * np.sqrt(252.0))


def _annualized_sortino(returns: np.ndarray) -> float:
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) < 2:
        return 0.0
    downside_std = float(np.std(downside, ddof=1))
    if downside_std <= 1e-12:
        return 0.0
    return float(np.mean(returns) / downside_std * np.sqrt(252.0))


def auto_train_model(instrument: str, end_date: date) -> MetaModel | None:
    """Train the meta-model on all data up to *end_date*."""
    logger.info("No meta-model found in DB. Training model on data up to %s...", end_date)
    model = MetaModel()
    try:
        result = model.train_from_db(instrument=instrument, end_date=str(end_date))
        logger.info("Meta-model successfully trained! Version: %s", result["model_version"])
        return model
    except Exception as e:
        logger.error("Failed to train meta-model from DB: %s", e, exc_info=True)
        return None


def run_walkforward(
    instrument: str,
    start_date: date,
    end_date: date,
    initial_equity: float,
    data_dir: str,
    force_train: bool = False,
) -> dict:
    # 1. Fetch OOS dates
    logger.info("Fetching Out-of-Sample (OOS) dates in range %s to %s...", start_date, end_date)
    oos_dates = _fetch_oos_dates(instrument, start_date, end_date)
    if not oos_dates:
        return {"error": f"No daily bars found in range {start_date} to {end_date}"}
    
    logger.info("Found %d candidate trading days for OOS walkforward.", len(oos_dates))

    # Clean up previous backtest data from Postgres to prevent cross-contamination
    logger.info("Cleaning up previous backtest runs from database...")
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM live_trades WHERE ticket_id LIKE 'backtest_%'")
            cur.execute("""
                DELETE FROM opportunity_measurements 
                WHERE event_id IN (
                    SELECT id FROM events 
                    WHERE date BETWEEN %s AND %s AND instrument = %s
                )
            """, (start_date, end_date, instrument))
            cur.execute("DELETE FROM events WHERE date BETWEEN %s AND %s AND instrument = %s", (start_date, end_date, instrument))
            cur.execute("DELETE FROM learning_runs WHERE date BETWEEN %s AND %s AND instrument = %s", (start_date, end_date, instrument))
        conn.commit()
    except Exception as e:
        logger.error("Cleanup failed: %s", e)
    finally:
        conn.close()
    
    # 2. Load model
    model = MetaModel()
    model_loaded = False
    
    if not force_train:
        try:
            model._load_model()
            if model.model is not None:
                model_loaded = True
        except Exception as e:
            logger.warning("Could not load model: %s", e)
            
    if not model_loaded or force_train:
        # Train on data up to day before start_date
        training_cutoff = start_date - timedelta(days=1)
        trained_model = auto_train_model(instrument, training_cutoff)
        if trained_model is not None:
            model = trained_model
            model_loaded = True
        else:
            logger.warning("No meta-model available — walkforward will run flat defaults.")

    # 3. Initialize HMM regime detector
    detector = RegimeDetector()
    hmm_fitted = False
    last_hmm_fit_date = None

    # Closed-loop parameter tracking
    current_learning_params = {
        "kelly_fraction": 0.15,
        "probability_threshold_half": 0.55,
        "probability_threshold_full": 0.70,
        "max_spread_pips": 2.0,
        "velocity_entry_threshold": 1.5e-6,
        "velocity_exit_threshold": 0.0
    }
    
    # 4. Results tracking
    equity = initial_equity
    equity_curve = [equity]
    results = []
    trade_returns = []
    daily_returns = []
    
    total_trades = 0
    wins = 0
    losses = 0
    total_mfe = 0.0
    total_mae = 0.0
    
    # 5. OOS Loop
    for idx, run_date in enumerate(oos_dates):
        try:
            # (a) Load bars up to run_date
            daily_df = _load_daily_bars(instrument, run_date)
            h4_df = _load_h4_bars(instrument, run_date)
            
            if len(daily_df) < 60:
                logger.warning("Skipping %s: insufficient daily bars (%d)", run_date, len(daily_df))
                continue
                
            # (b) Compute technicals
            technical = compute_all_features(daily_df, h4_df)
            
            # (c) HMM Regime Detection
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
                    logger.warning("HMM predict failed on %s: %s", run_date, regime_err)
                    regime = detector._default_regime()
            else:
                regime = detector._default_regime()
                
            # (d) Generate signals on-the-fly
            regime_state_id = regime.get("state_id", 1)
            tier1_signals = generate_all_tier1(run_date, instrument, regime_state_id, technical)
            composite_t1 = compute_composite(tier1_signals, {})
            proposed_dir = composite_t1.get("composite_direction", "flat")
            
            tier2_signals = generate_all_tier2(run_date, instrument, technical, proposed_dir)
            composite = compute_composite(tier1_signals, tier2_signals)
            
            signals_summary = {
                "tier1": tier1_signals,
                "tier2": tier2_signals,
                "composite": composite,
            }
            
            # (e) Assemble features
            vector = assemble_feature_vector(run_date, instrument, technical, regime, signals_summary)
            
            # (f) Meta-model prediction
            prediction = {"direction": "flat", "probability": 0.5, "size_multiplier": 0.0}
            if model_loaded and model.model is not None:
                prediction = model.predict(
                    vector,
                    threshold_half=current_learning_params["probability_threshold_half"],
                    threshold_full=current_learning_params["probability_threshold_full"]
                )
                
            dir_pred = prediction["direction"]
            prob = prediction["probability"]
            size_mult = prediction["size_multiplier"]
            
            # Record day details
            day_result = {
                "date": run_date.isoformat(),
                "regime": regime["state_label"],
                "proposed_direction": proposed_dir,
                "model_direction": dir_pred,
                "probability": prob,
                "size_multiplier": size_mult,
                "traded": False,
                "trade": None,
                "daily_return": 0.0,
                "equity": equity,
            }
            
            # (g) Trade execution on T+1
            if dir_pred != "flat" and size_mult > 0.0:
                t1_date = next_trading_day(run_date)
                tick_file = _get_tick_filepath(data_dir, t1_date)
                
                if tick_file:
                    # Scale sizing dynamically using adjusted Kelly fraction relative to base 0.15
                    adjusted_size_mult = size_mult * (current_learning_params["kelly_fraction"] / 0.15)
                    logger.info("Executing trade on %s (proposed=%s, size=%.3fx, original=%.1fx)", t1_date, dir_pred, adjusted_size_mult, size_mult)
                    trade = simulate_intraday_trade(tick_file, dir_pred, equity, adjusted_size_mult)
                    
                    if "error" not in trade:
                        pnl = trade["pnl_usd"]
                        equity += pnl
                        day_result["traded"] = True
                        day_result["trade"] = trade
                        day_result["daily_return"] = trade["leveraged_return_pct"] / 100.0
                        day_result["equity"] = equity
                        
                        trade_returns.append(trade["trade_return_pct"] / 100.0)
                        daily_returns.append(trade["leveraged_return_pct"] / 100.0)
                        
                        total_trades += 1
                        total_mfe += trade["mfe_pct"]
                        total_mae += trade["mae_pct"]
                        
                        if pnl > 0:
                            wins += 1
                        else:
                            losses += 1
                            
                        # Log backtest trade to live_trades DB table to let ClosedLoopLearner analyze slippage
                        conn = get_connection()
                        try:
                            ticket_id = f"backtest_{instrument}_{t1_date.isoformat()}"
                            pnl_pips = float(trade["trade_return_pct"] * 100.0)
                            
                            with conn.cursor() as cur:
                                cur.execute("""
                                    INSERT INTO live_trades (ticket_id, instrument, direction, entry_time, entry_price, position_size, exit_time, exit_price, pnl_pips, pnl_amount, exit_reason)
                                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                    ON CONFLICT (ticket_id) DO UPDATE SET
                                        entry_price = EXCLUDED.entry_price,
                                        exit_price = EXCLUDED.exit_price,
                                        pnl_pips = EXCLUDED.pnl_pips,
                                        pnl_amount = EXCLUDED.pnl_amount,
                                        exit_time = EXCLUDED.exit_time
                                """, (
                                    ticket_id, instrument, dir_pred,
                                    datetime.combine(t1_date, time(9, 0), tzinfo=timezone.utc),
                                    float(trade["entry_price"]),
                                    int(equity * adjusted_size_mult),
                                    datetime.combine(t1_date, time(16, 0), tzinfo=timezone.utc),
                                    float(trade["exit_price"]),
                                    pnl_pips, float(pnl), "close"
                                ))
                            conn.commit()
                        except Exception as e:
                            logger.error("Failed to log backtest trade: %s", e)
                        finally:
                            conn.close()
                    else:
                        logger.warning("Trade failed on %s: %s", t1_date, trade["error"])
                else:
                    logger.warning("No tick data file found for trade date %s", t1_date)
            else:
                daily_returns.append(0.0)
                
            equity_curve.append(equity)
            results.append(day_result)
            
            # (h) Closed-loop Event Extraction, Opportunity Measurement, and Feedback
            try:
                # 1. Event Extraction (Layer 3)
                from events.extractor import EventExtractor
                extractor = EventExtractor()
                extractor.extract_and_store(run_date, instrument, composite, prediction)
                
                # 2. Opportunity Measurement (Layer 4) for yesterday's event
                if idx > 0:
                    prev_event_date = oos_dates[idx - 1]
                    prev_event_row = fetch_one(
                        "SELECT id, direction FROM events WHERE date = %s AND instrument = %s",
                        (prev_event_date, instrument)
                    )
                    if prev_event_row:
                        from opportunity.measurement import OpportunityMeasurer
                        measurer = OpportunityMeasurer()
                        measurer.measure_and_store(
                            event_id=prev_event_row["id"],
                            event_date=prev_event_date,
                            trade_date=run_date,
                            instrument=instrument,
                            direction=prev_event_row["direction"]
                        )
                
                # 3. System Learning Feedback (Layer 8)
                from learning.feedback import ClosedLoopLearner
                learner = ClosedLoopLearner(lookback_days=20)
                feedback = learner.run_feedback_cycle(run_date, instrument)
                
                # Update current trading parameters for the next iteration
                current_learning_params = feedback["adjusted_parameters"]
                
            except Exception as cl_err:
                logger.error("Closed-loop execution failed on %s: %s", run_date, cl_err)
            
        except Exception as e:
            logger.error("Error processing date %s: %s", run_date, e, exc_info=True)
            
        if (idx + 1) % 10 == 0:
            logger.info("Processed %d / %d OOS days. Current Equity: $%.2f", idx + 1, len(oos_dates), equity)
            
    # 6. Compute summary metrics
    logger.info("Computing walkforward performance summary...")
    total_return = (equity / initial_equity) - 1.0
    days = len(oos_dates)
    years = days / 252.0
    cagr = (equity / initial_equity) ** (1.0 / max(years, 0.001)) - 1.0 if equity > 0 else -1.0
    
    trade_ret_arr = np.array(trade_returns)
    daily_ret_arr = np.array(daily_returns)
    
    sharpe = _annualized_sharpe(daily_ret_arr)
    sortino = _annualized_sortino(daily_ret_arr)
    mdd = _max_drawdown(equity_curve)
    calmar = abs(cagr / mdd) if mdd < 0 else 0.0
    
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    avg_win = np.mean(trade_ret_arr[trade_ret_arr > 0]) * 100.0 if len(trade_ret_arr[trade_ret_arr > 0]) > 0 else 0.0
    avg_loss = np.mean(trade_ret_arr[trade_ret_arr < 0]) * 100.0 if len(trade_ret_arr[trade_ret_arr < 0]) > 0 else 0.0
    profit_factor = (
        abs(np.sum(trade_ret_arr[trade_ret_arr > 0]) / np.sum(trade_ret_arr[trade_ret_arr < 0]))
        if len(trade_ret_arr[trade_ret_arr < 0]) > 0 and np.sum(trade_ret_arr[trade_ret_arr < 0]) != 0
        else 0.0
    )
    
    exposure = total_trades / days if days > 0 else 0.0
    
    summary = {
        "instrument": instrument,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "initial_equity": initial_equity,
        "final_equity": round(equity, 2),
        "total_return": round(total_return, 4),
        "cagr": round(cagr, 4),
        "max_drawdown": round(mdd, 4),
        "calmar": round(calmar, 4),
        "annualized_sharpe": round(sharpe, 4),
        "annualized_sortino": round(sortino, 4),
        "total_trades": total_trades,
        "win_rate": round(win_rate, 4),
        "wins": wins,
        "losses": losses,
        "avg_win_pct": round(avg_win, 4),
        "avg_loss_pct": round(avg_loss, 4),
        "profit_factor": round(profit_factor, 4),
        "avg_mfe_pct": round(total_mfe / total_trades, 4) if total_trades > 0 else 0.0,
        "avg_mae_pct": round(total_mae / total_trades, 4) if total_trades > 0 else 0.0,
        "exposure_pct": round(exposure, 4),
    }
    
    return {
        "summary": summary,
        "daily_results": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Tick-by-tick out-of-sample walkforward backtest."
    )
    parser.add_argument(
        "--instrument",
        type=str,
        default="EUR_USD",
        help="Instrument name (default: EUR_USD)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2025-04-05",
        help="Start date in YYYY-MM-DD (default: 2025-04-05)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2026-04-03",
        help="End date in YYYY-MM-DD (default: 2026-04-03)",
    )
    parser.add_argument(
        "--equity",
        type=float,
        default=10000.0,
        help="Initial equity (default: 10000.0)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=r"C:\Users\angel\OneDrive\Apps\Quant EOD\GLBX-20260405-TX6GF64XBR",
        help="Directory containing Databento .zst files",
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Force training of the meta-model up to start date",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save JSON results",
    )
    args = parser.parse_args()
    
    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
    
    results = run_walkforward(
        instrument=args.instrument,
        start_date=start_date,
        end_date=end_date,
        initial_equity=args.equity,
        data_dir=args.data_dir,
        force_train=args.force_train,
    )
    
    if "error" in results:
        logger.error(results["error"])
        sys.exit(1)
        
    summary = results["summary"]
    logger.info("=" * 60)
    logger.info("WALKFORWARD TEST PERFORMANCE SUMMARY:")
    logger.info(f"  Instrument      : {summary['instrument']}")
    logger.info(f"  Period          : {summary['start_date']} -> {summary['end_date']}")
    logger.info(f"  Trades          : {summary['total_trades']}")
    logger.info(f"  Win Rate        : {summary['win_rate']*100:.2f}% (Wins: {summary['wins']}, Losses: {summary['losses']})")
    logger.info(f"  Total Return    : {summary['total_return']*100:.2f}% (Final Equity: ${summary['final_equity']})")
    logger.info(f"  CAGR            : {summary['cagr']*100:.2f}%")
    logger.info(f"  Max Drawdown    : {summary['max_drawdown']*100:.2f}%")
    logger.info(f"  Sharpe / Sortino: {summary['annualized_sharpe']:.2f} / {summary['annualized_sortino']:.2f}")
    logger.info(f"  Avg Win / Loss  : {summary['avg_win_pct']:.2f}% / {summary['avg_loss_pct']:.2f}%")
    logger.info(f"  Profit Factor   : {summary['profit_factor']:.2f}")
    logger.info(f"  Avg MFE / MAE   : {summary['avg_mfe_pct']:.2f}% / {summary['avg_mae_pct']:.2f}%")
    logger.info(f"  Market Exposure : {summary['exposure_pct']*100:.2f}%")
    logger.info("=" * 60)
    
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Detailed walkforward results saved to {args.output}")


if __name__ == "__main__":
    main()
