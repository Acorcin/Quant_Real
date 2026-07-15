# Phase 2 — ES execution-reality test: RESULTS (single pre-registered run)

**Run:** 2026-07-15 · ESZ5 trades Sep 15 → Dec 12 2025 (Databento GLBX.MDP3,
quoted $0.00 at execution) · 31.6M trades, 93.6M contracts.
One process crash mid-run (host DB went down) was resumed from data checkpoints
with the identical locked config before any results were observed; the model
run itself executed once, end to end. Full log preserved; features:
`data/ESZ5_features.parquet` (1,089 rows), backfilled to md as run
`0cd30c22-e48a-4965-9fd1-ce26c16ed9f5`.

## Locked parameters → what the rules chose

| rule (fixed a priori) | outcome |
|---|---|
| smallest 500-multiple with median RTH bar ≥ 45 s (first 5 sessions only) | **2,500 contracts** (median 50.2 s; table: 500→9.4s … 10,000→204.5s) |
| step for ~1,100 origins | 37,280 bars → **step 34 → 1,089 origins** |
| model | Kronos-small · ctx 256 · 24 paths · horizon 4 |

## Primary endpoints

**Structure verdict: `efficient`.** Hurst 0.493, permutation entropy 1.000,
predictable horizon 0 steps, 16 structural breaks, turbulent regime
unpredictable.

Step-1 ladder (MASE / R²oos / DM p vs naive):

| rung | MASE | R²oos | DMp | dir |
|---|---|---|---|---|
| naive-rw | 0.725 | +0.0000 | — | — |
| ar | 0.726 | +0.0009 | 0.717 | 0.485 |
| garch | 0.725 | −0.0002 | 0.468 | 0.496 |
| kronos | 0.779 | −0.1483 | **6.4e-05 (worse)** | 0.531 |

Step-4 DM vs naive (same archive, no second run): ar +1.61 (p=0.108),
garch +1.88 (p=0.061), kronos **+2.24 (p=0.025)** — positive statistic =
higher loss than naive, i.e. all rungs at-or-worse than the random walk at
every horizon.

Kronos R²oos by step: 1: −0.148 · 2: −0.045 · 3: −0.041 · 4: −0.032
(monotonically less bad with horizon, never positive).

Gate: 1,089 decisions → 99% VETO / 1% REDUCED / 0% GO; reference-strategy
Sharpe uplift −0.12 (nothing worth gating).

## Conclusion

At physically-executable volume bars (≈50 s formation vs our ~5 s inference),
ES close-to-close returns in Sep–Dec 2025 are **efficient with respect to this
model ladder**. The positive nonlinear case did not materialize on a primary
contract either. Chain to date: synthetic oracles ✓ recovered, M6E → efficient,
ES → efficient. The instrument keeps refusing to certify structure — which is
the honest product: a system that will not trade noise.

## Secondary, hypothesis-generating only (explicitly NOT a finding)

Kronos step-1 **sign** agreement: 584/1,089 = 53.6%, binomial p = 0.018 —
computed post-hoc from the run's saved features. Direction is slightly right
while magnitude is badly wrong; that shape is what meta-labeling exists for
(primary direction + meta sizing). One nominally significant secondary among
several diagnostics is multiple-testing bait, not an edge. If pursued, it gets
its own pre-registration on a NEW quarter: kronos sign as tier-1 primary,
meta-labeled, single run.
