"""
Feature Engineering: Technical Indicators.

Computes ATR, RSI, Moving Averages, candle body analysis, and other
technical features from OHLCV bar data. All indicators are computed
in pure pandas — no TA-Lib dependency.
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range (ATR).

    Args:
        df: DataFrame with 'high', 'low', 'close' columns.
        period: Lookback period (default 14).

    Returns:
        Series of ATR values.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return true_range.ewm(span=period, adjust=False).mean()


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI).

    Uses exponential moving average (Wilder's smoothing).
    """
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_ma(df: pd.DataFrame, period: int) -> pd.Series:
    """Simple Moving Average of close prices."""
    return df["close"].rolling(window=period, min_periods=period).mean()


def compute_ema(df: pd.DataFrame, period: int) -> pd.Series:
    """Exponential Moving Average of close prices."""
    return df["close"].ewm(span=period, adjust=False).mean()


def compute_body_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Candle body analysis — direction, body-to-range ratio, wick ratios.

    Returns DataFrame with columns:
        body_direction: 1 (bullish), -1 (bearish), 0 (doji)
        body_pct_of_range: ratio of body size to total range (0-1)
        upper_wick_pct: upper wick as % of range
        lower_wick_pct: lower wick as % of range
    """
    body = df["close"] - df["open"]
    total_range = df["high"] - df["low"]

    # Avoid division by zero on doji candles
    safe_range = total_range.replace(0, np.nan)

    direction = np.sign(body).astype(int)
    body_pct = body.abs() / safe_range
    upper_wick = (df["high"] - df[["open", "close"]].max(axis=1)) / safe_range
    lower_wick = (df[["open", "close"]].min(axis=1) - df["low"]) / safe_range

    result = pd.DataFrame({
        "body_direction": direction,
        "body_pct_of_range": body_pct.fillna(0),
        "upper_wick_pct": upper_wick.fillna(0),
        "lower_wick_pct": lower_wick.fillna(0),
    }, index=df.index)

    return result


def detect_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect key daily candle patterns.

    Returns DataFrame with boolean columns:
        is_engulfing_bull, is_engulfing_bear,
        is_pin_bar_bull, is_pin_bar_bear,
        is_inside_bar, is_doji
    """
    body = (df["close"] - df["open"]).abs()
    prev_body = body.shift(1)
    total_range = df["high"] - df["low"]
    safe_range = total_range.replace(0, np.nan)

    direction = np.sign(df["close"] - df["open"])
    prev_direction = direction.shift(1)

    # Engulfing: current body fully covers previous body, opposite direction
    engulfing_bull = (direction == 1) & (prev_direction == -1) & (body > prev_body)
    engulfing_bear = (direction == -1) & (prev_direction == 1) & (body > prev_body)

    # Pin bar: long wick (>60% of range) on one side, small body
    body_ratio = body / safe_range
    lower_wick = (df[["open", "close"]].min(axis=1) - df["low"]) / safe_range
    upper_wick = (df["high"] - df[["open", "close"]].max(axis=1)) / safe_range

    pin_bar_bull = (lower_wick > 0.6) & (body_ratio < 0.25)
    pin_bar_bear = (upper_wick > 0.6) & (body_ratio < 0.25)

    # Inside bar: today's range entirely within yesterday's range
    inside_bar = (df["high"] <= df["high"].shift(1)) & (df["low"] >= df["low"].shift(1))

    # Doji: body < 10% of range
    doji = body_ratio < 0.10

    return pd.DataFrame({
        "is_engulfing_bull": engulfing_bull.fillna(False),
        "is_engulfing_bear": engulfing_bear.fillna(False),
        "is_pin_bar_bull": pin_bar_bull.fillna(False),
        "is_pin_bar_bear": pin_bar_bear.fillna(False),
        "is_inside_bar": inside_bar.fillna(False),
        "is_doji": doji.fillna(False),
    }, index=df.index)


def compute_all_features(daily_bars: pd.DataFrame, h4_bars: pd.DataFrame | None = None) -> dict:
    """
    Compute the full technical feature set from daily OHLCV bars.

    Args:
        daily_bars: DataFrame with columns [open, high, low, close, volume]
            sorted by bar_time ascending. At least 60 rows expected.
        h4_bars: Optional H4 bars DataFrame used for intraday context.

    Returns:
        Dict of feature name → value for the LATEST bar.
    """
    if len(daily_bars) < 20:
        logger.warning(f"Only {len(daily_bars)} bars — need at least 20 for indicators")
        return {}

    df = daily_bars.copy().sort_values("bar_time").reset_index(drop=True)

    # Technical indicators
    df["atr_14"] = compute_atr(df, 14)
    df["rsi_14"] = compute_rsi(df, 14)
    df["ma_50"] = compute_ma(df, 50)
    df["ma_200"] = compute_ma(df, 200)
    df["ema_20"] = compute_ema(df, 20)

    # Body analysis
    body = compute_body_analysis(df)
    df = pd.concat([df, body], axis=1)

    # Candle patterns
    patterns = detect_candle_patterns(df)
    df = pd.concat([df, patterns], axis=1)

    # Latest row
    latest = df.iloc[-1]
    close = latest["close"]

    features = {
        # Core price
        "close": float(close),
        "daily_return_pct": float((close / df.iloc[-2]["close"] - 1) * 100) if len(df) > 1 else 0.0,

        # Volatility
        "atr_14": float(latest["atr_14"]) if pd.notna(latest["atr_14"]) else None,

        # Momentum
        "rsi_14": float(latest["rsi_14"]) if pd.notna(latest["rsi_14"]) else None,

        # Trend — price vs MAs (% distance)
        "price_vs_ma50": float((close / latest["ma_50"] - 1) * 100) if pd.notna(latest["ma_50"]) else None,
        "price_vs_ma200": float((close / latest["ma_200"] - 1) * 100) if pd.notna(latest["ma_200"]) else None,
        "ma_50_value": float(latest["ma_50"]) if pd.notna(latest["ma_50"]) else None,
        "ma_200_value": float(latest["ma_200"]) if pd.notna(latest["ma_200"]) else None,

        # Body analysis
        "body_direction": int(latest["body_direction"]),
        "body_pct_of_range": float(latest["body_pct_of_range"]),
        "upper_wick_pct": float(latest["upper_wick_pct"]),
        "lower_wick_pct": float(latest["lower_wick_pct"]),

        # Candle patterns
        "is_engulfing_bull": bool(latest["is_engulfing_bull"]),
        "is_engulfing_bear": bool(latest["is_engulfing_bear"]),
        "is_pin_bar_bull": bool(latest["is_pin_bar_bull"]),
        "is_pin_bar_bear": bool(latest["is_pin_bar_bear"]),
        "is_inside_bar": bool(latest["is_inside_bar"]),
        "is_doji": bool(latest["is_doji"]),

        # Rolling volatility (5d and 20d returns std dev)
        "volatility_5d": float(df["close"].pct_change().tail(5).std()) if len(df) >= 6 else None,
        "volatility_20d": float(df["close"].pct_change().tail(20).std()) if len(df) >= 21 else None,
    }

    if h4_bars is not None and len(h4_bars) >= 2:
        h4 = h4_bars.copy().sort_values("bar_time").reset_index(drop=True)
        h4_ret = h4["close"].pct_change()
        last_close = float(h4.iloc[-1]["close"])
        prev_close = float(h4.iloc[-2]["close"])
        h4_body = float(h4.iloc[-1]["close"] - h4.iloc[-1]["open"])
        h4_target = h4_ret.tail(min(len(h4_ret), 6))
        features.update({
            "h4_return_last_pct": float((last_close / prev_close - 1.0) * 100.0) if prev_close else 0.0,
            "h4_body_direction": 1 if h4_body > 0 else (-1 if h4_body < 0 else 0),
            "h4_trend_6bars_pct": float((last_close / float(h4.iloc[-min(len(h4), 6)]["close"]) - 1.0) * 100.0)
            if len(h4) >= 6 else None,
            "h4_volatility_6bars": float(h4_target.std()) if len(h4_target) >= 2 else None,
        })

    return features
