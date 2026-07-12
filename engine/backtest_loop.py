#!/usr/bin/env python3
"""
Historical backtest loop for the live daily prediction logic.

Runs saved feature vectors through the current meta-model and evaluates
next-trading-day close-to-close PnL.
"""
import argparse
import json
import logging
from datetime import date, datetime
import numpy as np

from config.settings import PRIMARY_INSTRUMENT
from models.database import fetch_all
from models.meta_model import MetaModel
try:
    from utils.trading_calendar import next_trading_day
except Exception as exc:  # pragma: no cover - defensive fallback
    logger = logging.getLogger("backtest_loop")
    logger.warning("utils.trading_calendar import failed, using weekday fallback: %s", exc)

    def next_trading_day(run_date: date) -> date:
        from datetime import timedelta

        candidate = run_date + timedelta(days=1)
        while candidate.weekday() >= 5:
            candidate += timedelta(days=1)
        return candidate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backtest_loop")


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return datetime.fromisoformat(value).date()


def _load_price_map(instrument: str) -> dict[date, dict]:
    rows = fetch_all(
        """SELECT bar_time::date AS d, open, close
           FROM bars
           WHERE instrument = %s AND granularity = 'D' AND complete = TRUE
           ORDER BY bar_time ASC""",
        (instrument,),
    )
    return {r["d"]: {"open": float(r["open"]), "close": float(r["close"])} for r in rows}


def _resolve_next_price(price_map: dict[date, dict], d: date) -> tuple[date | None, dict | None]:
    nd = next_trading_day(d)
    guard = 0
    while nd not in price_map and guard < 30:
        nd = next_trading_day(nd)
        guard += 1
    if nd not in price_map:
        return None, None
    return nd, price_map[nd]


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


def _annualized_sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if std <= 1e-12:
        return 0.0
    return float(np.mean(returns) / std * np.sqrt(periods_per_year))


def _annualized_sortino(returns: np.ndarray, periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) < 2:
        return 0.0
    downside_std = float(np.std(downside, ddof=1))
    if downside_std <= 1e-12:
        return 0.0
    return float(np.mean(returns) / downside_std * np.sqrt(periods_per_year))


def run_backtest(
    instrument: str,
    start: date | None,
    end: date | None,
    initial_equity: float,
    contract_multiplier: float = 50.0,
    tick_size: float = 0.25,
    ticks_cost_per_side: float = 1.0,
    contracts: int = 1,
) -> dict:
    """
    Historical backtest with CME futures friction math.

    PnL is computed in ABSOLUTE dollars from points and contract multipliers
    (no FX leverage/spread-in-price abstractions):

        points  = (exit - entry) * direction_sign
        cost    = 2 * ticks_cost_per_side * tick_size          [round trip, points]
        pnl_usd = size_mult * contracts * contract_multiplier * (points - cost)

    Args:
        contract_multiplier: $ per 1.0 price point (ES=$50, NQ=$20, M6E=$12,500
            per 1.0 EUR/USD point — i.e., $1.25 per 0.0001 tick).
        tick_size: minimum price increment in points (ES=0.25, M6E=0.0001).
        ticks_cost_per_side: friction per side in ticks (spread + slippage +
            commission-equivalent). 1.0 = cross the spread once each way.
        contracts: base number of contracts at full size.
    """
    where = ["instrument = %s"]
    params: list = [instrument]
    if start:
        where.append("date >= %s")
        params.append(str(start))
    if end:
        where.append("date <= %s")
        params.append(str(end))

    fv_rows = fetch_all(
        f"""SELECT date, features
            FROM feature_vectors
            WHERE {' AND '.join(where)}
            ORDER BY date ASC""",
        tuple(params),
    )
    if not fv_rows:
        return {"error": "No feature_vectors rows found for selection"}

    price_map = _load_price_map(instrument)
    model = MetaModel()

    equity = initial_equity
    curve = []
    wins = 0
    losses = 0
    traded = 0
    prev_size = 0.0
    turnover = 0.0
    position_sizes: list[float] = []
    pnl_series: list[float] = []
    
    # CME futures friction: absolute points × contract multiplier, with a
    # round-trip cost in ticks. No leverage abstraction — a futures position's
    # exposure IS the contract; equity moves by dollar PnL, additively.
    round_trip_cost_points = 2.0 * ticks_cost_per_side * tick_size

    for row in fv_rows:
        d = row["date"]
        features = row["features"] or {}
        pred = model.predict(features)

        nd, prices_nd = _resolve_next_price(price_map, d)
        if nd is None or prices_nd is None:
            continue

        # Real Execution logic:
        # Feature vector is computed after day T closes.
        # We enter at the Open of T+1, exit at the Close of T+1.
        entry_price = prices_nd["open"]
        exit_price = prices_nd["close"]

        move_points = exit_price - entry_price

        direction = pred["direction"]
        size = float(pred.get("size_multiplier", 0.0) or 0.0)

        # Directional points captured, minus round-trip friction in points,
        # converted to dollars via the contract multiplier.
        if direction == "long":
            net_points = move_points - round_trip_cost_points
        elif direction == "short":
            net_points = -move_points - round_trip_cost_points
        else:
            net_points = 0.0

        pnl_usd = (size * contracts * contract_multiplier * net_points
                   if direction in ("long", "short") else 0.0)

        if direction in ("long", "short") and size > 0:
            traded += 1
            if pnl_usd > 0:
                wins += 1
            elif pnl_usd < 0:
                losses += 1

        turnover += abs(size - prev_size)
        prev_size = size
        position_sizes.append(size)
        equity_before = equity
        equity += pnl_usd
        # return-on-equity series (for Sharpe/Sortino/MDD on the equity curve)
        pnl_series.append(pnl_usd / equity_before if equity_before > 0 else 0.0)
        curve.append(
            {
                "date": str(d),
                "prediction_for": str(nd),
                "direction": direction,
                "probability": pred.get("probability"),
                "size": size,
                "move_points": round(move_points, 6),
                "net_points": round(net_points, 6),
                "pnl_usd": round(pnl_usd, 2),
                "equity": round(equity, 2),
            }
        )

    total_return = (equity / initial_equity - 1.0) if initial_equity > 0 else 0.0
    win_rate = (wins / traded) if traded > 0 else 0.0
    periods = len(pnl_series)
    years = periods / 252.0 if periods > 0 else 0.0
    cagr = ((equity / initial_equity) ** (1 / years) - 1.0) if years > 0 and initial_equity > 0 else 0.0
    pnl_arr = np.array(pnl_series, dtype=float) if pnl_series else np.array([], dtype=float)
    equity_vals = [float(x["equity"]) for x in curve]
    mdd = _max_drawdown(equity_vals)
    sharpe = _annualized_sharpe(pnl_arr)
    sortino = _annualized_sortino(pnl_arr)
    calmar = (cagr / abs(mdd)) if mdd < 0 else 0.0
    exposure = float(np.mean(np.array(position_sizes) > 0)) if position_sizes else 0.0
    avg_position_size = float(np.mean(position_sizes)) if position_sizes else 0.0
    turnover_per_day = (turnover / periods) if periods > 0 else 0.0

    performance_report = {
        "periods": periods,
        "years": round(years, 4),
        "cagr": round(cagr, 4),
        "annualized_sharpe": round(sharpe, 4),
        "annualized_sortino": round(sortino, 4),
        "max_drawdown": round(mdd, 4),
        "calmar": round(calmar, 4),
        "exposure": round(exposure, 4),
        "avg_position_size": round(avg_position_size, 4),
        "turnover_total": round(turnover, 4),
        "turnover_per_day": round(turnover_per_day, 4),
        "avg_daily_pnl": round(float(np.mean(pnl_arr)) if periods > 0 else 0.0, 6),
        "vol_daily_pnl": round(float(np.std(pnl_arr, ddof=1)) if periods > 1 else 0.0, 6),
    }

    return {
        "instrument": instrument,
        "start_date": str(fv_rows[0]["date"]),
        "end_date": str(fv_rows[-1]["date"]),
        "rows_processed": len(fv_rows),
        "trades": traded,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4),
        "initial_equity": initial_equity,
        "final_equity": round(equity, 2),
        "total_return": round(total_return, 4),
        "performance_report": performance_report,
        "equity_curve": curve,
    }


def main():
    parser = argparse.ArgumentParser(description="Historical backtest loop using stored feature vectors")
    parser.add_argument("--instrument", default=PRIMARY_INSTRUMENT, help="Instrument, e.g. ES, M6E")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--equity", type=float, default=10000.0, help="Initial equity ($)")
    parser.add_argument("--contract-multiplier", type=float, default=50.0,
                        help="$ per 1.0 price point (ES=50, NQ=20, M6E=12500)")
    parser.add_argument("--tick-size", type=float, default=0.25,
                        help="Minimum price increment in points (ES=0.25, M6E=0.0001)")
    parser.add_argument("--ticks-cost-per-side", type=float, default=1.0,
                        help="Friction per side in ticks (spread+slippage+commission)")
    parser.add_argument("--contracts", type=int, default=1,
                        help="Contracts at full size")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    result = run_backtest(
        instrument=args.instrument,
        start=_parse_date(args.start),
        end=_parse_date(args.end),
        initial_equity=args.equity,
        contract_multiplier=args.contract_multiplier,
        tick_size=args.tick_size,
        ticks_cost_per_side=args.ticks_cost_per_side,
        contracts=args.contracts,
    )

    if "error" in result:
        logger.error(result["error"])
        raise SystemExit(1)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.info("Backtest written to %s", args.output)
    else:
        logger.info(
            "Backtest %s %s->%s | trades=%s win_rate=%.2f%% total_return=%.2f%%",
            result["instrument"],
            result["start_date"],
            result["end_date"],
            result["trades"],
            result["win_rate"] * 100,
            result["total_return"] * 100,
        )
        report = result.get("performance_report", {})
        logger.info(
            "Perf | CAGR=%.2f%% Sharpe=%.2f Sortino=%.2f MaxDD=%.2f%% Calmar=%.2f Exposure=%.2f%% Turnover/day=%.3f",
            report.get("cagr", 0.0) * 100,
            report.get("annualized_sharpe", 0.0),
            report.get("annualized_sortino", 0.0),
            report.get("max_drawdown", 0.0) * 100,
            report.get("calmar", 0.0),
            report.get("exposure", 0.0) * 100,
            report.get("turnover_per_day", 0.0),
        )


if __name__ == "__main__":
    main()
