"""
L3 puller correctness: MBO book maintenance and VPIN volume bucketing.
Pure in-memory — no network, no database.
"""

from __future__ import annotations

from datetime import datetime, timezone

from forecasting.l3_puller import Book, VpinState


TS = datetime(2026, 7, 13, 14, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# order book
# ---------------------------------------------------------------------------

def test_book_add_cancel_modify_fill():
    b = Book()
    b.apply("A", "B", 1, 100.00, 5)
    b.apply("A", "B", 2, 99.75, 3)
    b.apply("A", "A", 3, 100.25, 4)
    snap = b.snapshot()
    assert (snap["bid_px"], snap["ask_px"]) == (100.00, 100.25)
    assert (snap["bid_sz"], snap["ask_sz"]) == (5, 4)

    # modify moves order 1 down in price and size
    b.apply("M", "B", 1, 99.75, 2)
    snap = b.snapshot()
    assert snap["bid_px"] == 99.75 and snap["bid_sz"] == 5  # 3 + 2 same level

    # partial fill reduces, full fill removes
    b.apply("F", "A", 3, 100.25, 3)
    assert b.snapshot()["ask_sz"] == 1
    b.apply("F", "A", 3, 100.25, 1)
    assert b.snapshot() is None          # ask side empty -> one-sided book

    b.apply("A", "A", 4, 100.50, 2)
    b.apply("C", "B", 2, 0, 0)
    snap = b.snapshot()
    assert snap["bid_sz"] == 2 and snap["ask_px"] == 100.50


def test_book_imbalance_and_reset():
    b = Book()
    b.apply("A", "B", 1, 10.0, 9)
    b.apply("A", "A", 2, 10.5, 1)
    snap = b.snapshot()
    assert abs(snap["imbalance_l1"] - 0.8) < 1e-12       # (9-1)/(9+1)
    assert abs(snap["imbalance_d10"] - 0.8) < 1e-12
    b.apply("R", "N", 0, 0, 0)
    assert b.snapshot() is None and len(b.orders) == 0


def test_book_depth_imbalance_uses_top_10_levels():
    b = Book()
    for i in range(15):                                   # 15 bid levels
        b.apply("A", "B", 100 + i, 100.0 - i * 0.25, 10)
    b.apply("A", "A", 200, 100.25, 10)
    snap = b.snapshot()
    # only 10 best bid levels count: (100 - 10) / (100 + 10)
    assert abs(snap["imbalance_d10"] - (90 / 110)) < 1e-12


# ---------------------------------------------------------------------------
# VPIN
# ---------------------------------------------------------------------------

def test_vpin_bucket_close_and_split():
    v = VpinState(bucket_vol=10)
    assert v.add_trade("B", 6, TS) == []                  # bucket at 6/10
    closed = v.add_trade("A", 7, TS)                      # fills 10, spills 3
    assert len(closed) == 1
    b = closed[0]
    assert b["buy"] == 6 and b["sell"] == 4 and b["seq"] == 0
    assert v.fill == 3 and v.sell == 3                    # spillover carried

    # one giant trade closes multiple buckets
    closed = v.add_trade("B", 27, TS)
    assert [c["seq"] for c in closed] == [1, 2, 3]
    assert closed[0]["sell"] == 3 and closed[0]["buy"] == 7


def test_vpin_imbalance_math():
    v = VpinState(bucket_vol=100)
    closed = v.add_trade("B", 80, TS) + v.add_trade("A", 20, TS)
    b = closed[0]
    vpin_contrib = abs(b["buy"] - b["sell"]) / (b["buy"] + b["sell"])
    assert abs(vpin_contrib - 0.6) < 1e-12               # |80-20|/100
