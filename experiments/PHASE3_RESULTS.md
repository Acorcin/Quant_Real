# Phase 3 — Kronos-sign meta-labeling test: RESULTS (single pre-registered run)

**Run:** 2026-07-15 · ESM6 trades Mar 16 → Jun 12 2026 (new quarter, untouched
by all prior analysis; quoted $0.00) · 32.1M trades, 94.3M contracts.
Bar rule independently re-chose **2,500 contracts** (median RTH 47.2 s) — same
as Phase 2, confirming the physics rule is stable across quarters.
37,560 bars → step 34 → 1,098 origins, horizon 1. Persisted as md run
`6a9c4ce9`; features `data/ESM6_features.parquet`.

## Endpoint A — sign replication (decisive first gate)

| | Phase 2 (ESZ5, hypothesis source) | **Phase 3 (ESM6, new quarter)** |
|---|---|---|
| primaries | 1,089 origins | 983 of 1,098 origins |
| sign accuracy | 53.6% (p=0.018 two-sided) | **49.75% (489/983)** |
| one-sided binomial p | — | **0.576** |

**FAILS TO REPLICATE.** Per the decision rule fixed before the data was
touched: *Phase 2's sign signal was noise. Book closed.*

## Endpoint B — reported for completeness, moot by the decision rule

CPCV (15 paths, purge 5 / embargo 2): Sharpe 0.55 ± 0.69, path t-test
p = 0.0037 (nominally "tradable"); pooled PSR 1.0 (known inflation).
This is NOT a rescue: the gate structure was A-first by design, a 49.75%
primary underneath a "significant" meta result means the meta-model found
in-quarter conditional patterns whose out-of-quarter validity is exactly what
Endpoint A just falsified, and CPCV path Sharpes share training data (correlated
paths overstate the t-test). Chasing B after A failed would be the precise
p-hacking this program exists to prevent.

## Context (the quarter itself)

ESM6 measured **efficient** — Hurst 0.482, entropy 1.000, no rung beats naive
(kronos magnitude significantly worse, DMp 1e-10), gate 99% VETO. Fourth
consecutive series with the same verdict.

## Program conclusion

Two pre-registered out-of-sample tests, zero parameter fishing:

| test | claim tested | verdict |
|---|---|---|
| Phase 2 (ESZ5) | nonlinear structure at executable bars | efficient; no |
| Phase 3 (ESM6) | Kronos step-1 sign carries direction | **does not replicate** |

The measurement instrument has now done its job in both directions: it found
the structure planted in synthetic oracles, and it refused to certify structure
in real CME series where honest statistics say there is none. The system that
refuses to trade noise has, so far, correctly refused everything — including
its own most tempting artifact.
