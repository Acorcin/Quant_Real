#!/usr/bin/env python3
"""
Retrain the XGBoost meta-model on the 25-column Kronos feature set, sourcing
training rows from the forecasting pipeline's md schema.

Meta-labeling construction (Lopez de Prado):
  * Model 1 (primary): kronos_directional_signal — sign of the sigma-scaled
    P50, with the same noise floor the live signal uses. Origins where the
    primary is flat produce no trade, hence no label, hence no training row.
  * Model 2 (this): P(primary's proposed trade is profitable next bar),
    labeled against the realized next-bar scaled return already stored per
    origin (md.kronos_features.y_true_scaled).

CPCV + PSR run unchanged; realized returns are passed directly
(y_true_scaled * sigma_256 = raw next-bar log return), so no synthetic proxy.

Honesty note: only kronos_* + primary-signal columns carry information here —
technical/macro/time features aren't computed per volume-bar origin yet, so
they train as constant zeros (zero split gain, harmless). On a series the
instrument measured `efficient`, expect CPCV/PSR to report NO significant edge;
that is the machinery telling the truth, not a defect. The point of this
retrain is a model artifact whose column contract matches the live path.

    DB_PORT=5433 python train_meta_from_md.py --instrument M6EH6
"""
import argparse
import logging
import sys

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_meta_from_md")

_MD_ROWS_SQL = """
    WITH src AS (
        SELECT r.run_id
        FROM md.runs r
        WHERE r.instrument = %s
          AND EXISTS (SELECT 1 FROM md.kronos_features k
                      WHERE k.run_id = r.run_id
                        AND k.kronos_p50_scaled IS NOT NULL)
        ORDER BY r.started_at DESC
        LIMIT 1
    )
    SELECT k.origin_seq, k.ts, k.sigma_256, k.y_true_scaled,
           k.kronos_p50_scaled, k.kronos_spread_scaled,
           k.structure::text AS structure, k.gate_state::text AS gate_state
    FROM md.kronos_features k
    JOIN src USING (run_id)
    WHERE k.kronos_p50_scaled IS NOT NULL
      AND k.y_true_scaled IS NOT NULL
    ORDER BY k.origin_seq
"""


def build_training_set(instrument: str):
    from models.database import fetch_all
    from features.vector import KRONOS_REGIME_MAP
    from signals.tier1 import KRONOS_MIN_EDGE_SCALED

    rows = fetch_all(_MD_ROWS_SQL, (instrument,))
    if not rows:
        raise SystemExit(f"no kronos-bearing md run found for {instrument}")
    logger.info("md rows for %s: %d origins", instrument, len(rows))

    vectors, labels, dates, rets = [], [], [], []
    skipped_flat = 0
    for r in rows:
        p50 = float(r["kronos_p50_scaled"])
        spread = float(r["kronos_spread_scaled"] or 0.0)
        sigma = float(r["sigma_256"] or 0.0)
        y_true = float(r["y_true_scaled"])

        # primary signal: same rule as the live tier-1 kronos_directional
        if spread <= 0 or abs(p50) < KRONOS_MIN_EDGE_SCALED:
            skipped_flat += 1
            continue
        direction = 1 if p50 > 0 else -1

        vec = {
            "kronos_p50_scaled": p50,
            "kronos_uncertainty": spread,
            "kronos_context_vol": sigma,
            "kronos_regime_encoded": KRONOS_REGIME_MAP.get(r["structure"], -1),
            "primary_signal_direction": direction,
            "primary_signal_count": 1,
            "composite_strength": min(abs(p50) / (abs(p50) + spread), 1.0),
            "day_of_week": r["ts"].weekday() if r["ts"] else 0,
            "is_friday": 1 if (r["ts"] and r["ts"].weekday() == 4) else 0,
        }  # remaining FEATURE_COLS default to 0.0 inside MetaModel.train

        vectors.append(vec)
        labels.append(1 if direction * y_true > 0 else 0)
        dates.append(r["ts"].date() if r["ts"] else None)
        rets.append(y_true * sigma)   # scaled * sigma = raw next-bar log return

    logger.info("training rows: %d (skipped %d flat-primary origins)",
                len(vectors), skipped_flat)
    return vectors, labels, dates, np.array(rets)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instrument", default=None,
                    help="md series (default: settings.KRONOS_INSTRUMENT)")
    args = ap.parse_args()

    from config.settings import KRONOS_INSTRUMENT
    instrument = args.instrument or KRONOS_INSTRUMENT

    vectors, labels, dates, rets = build_training_set(instrument)

    from models.meta_model import MetaModel
    meta = MetaModel()
    result = meta.train(
        feature_vectors=vectors,
        labels=labels,
        sample_dates=dates,
        instrument=instrument,
        sample_returns=rets,
    )

    cpcv = result["cpcv"]
    print("\n=== meta-model retrain (25-col Kronos set) ===")
    print(f"  instrument           : {instrument}")
    print(f"  model version        : {result['model_version']}")
    print(f"  training rows        : {result['in_sample_train_metrics']['samples']}")
    print(f"  positive-label rate  : {result['in_sample_train_metrics']['positive_rate']}")
    print(f"  CPCV paths           : {cpcv.get('paths_tested')}")
    print(f"  CPCV sharpe mean/std : {cpcv.get('sharpe_mean')} / {cpcv.get('sharpe_std')}")
    print(f"  path-sharpe p-value  : {cpcv.get('path_sharpe_p_value')}")
    print(f"  Bailey PSR           : {cpcv.get('probabilistic_sharpe_ratio')}")
    print(f"  significant edge     : {cpcv.get('statistically_significant')}")
    print("  top features         :")
    for f in result["top_features"][:6]:
        print(f"      {f['feature']:<28} {f['mean_abs_shap']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
