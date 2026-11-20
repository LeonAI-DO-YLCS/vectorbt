import numpy as np
import pandas as pd
import pytest

from vectorbt.portfolio import fast_sim

PT = 0.01
T0 = pd.Timestamp("2024-01-01")


def _bars(rows):
    """rows: (open, high, low, close, spread) in price units."""
    idx = pd.date_range(T0, periods=len(rows), freq="15min")
    return pd.DataFrame(
        rows, columns=["open", "high", "low", "close", "spread"], index=idx
    )


def _req(side, stop, tp=np.nan, exit_bar=9, **kw):
    return pd.DataFrame(
        [
            {
                "entry_time": T0,
                "exit_time": T0 + pd.Timedelta(minutes=15 * exit_bar),
                "side": side,
                "entry_price": np.nan,
                "stop_points": stop,
                "target_points": tp,
                **kw,
            }
        ]
    )


FLAT = (100.0, 100.0, 100.0, 100.0, 10)


def _one(rows, **kw):
    return fast_sim.simulate(_bars(rows), _req(**kw), PT).iloc[0]


def test_buy_stop_hit_at_ask_entry():
    r = _one([FLAT, (100, 100.2, 99.5, 99.6, 10), FLAT], side=1, stop=30, tp=100)
    assert (r.exit_reason, r.points) == ("stop", -30)
    assert (r.entry_price, r.exit_price) == pytest.approx((100.1, 99.8))


def test_sell_target_hit():
    r = _one([FLAT, (100, 100.1, 99.4, 99.5, 10), FLAT], side=-1, stop=100, tp=50)
    assert (r.exit_reason, r.points) == ("target", 50)
    assert (r.entry_price, r.exit_price) == pytest.approx((100.0, 99.5))


def test_both_hit_in_one_bar_stop_first():
    r = _one([FLAT, (100, 101, 99, 100.5, 10), FLAT], side=-1, stop=50, tp=50)
    assert r.exit_reason == "target"  # rising bar, sell: low first, so target
    r = _one([FLAT, (100, 101, 99, 100.5, 10), FLAT], side=1, stop=50, tp=50)
    assert r.exit_reason == "stop"  # rising bar, buy: low first, so stop


def test_time_exit_at_open():
    r = _one(
        [FLAT, FLAT, (100.2, 100.2, 100.2, 100.2, 10)], side=1, stop=50, exit_bar=2
    )
    assert (r.exit_reason, r.points) == ("time", 10)
    assert r.exit_price == pytest.approx(100.2)


def test_breakeven_then_stop_at_entry():
    rows = [
        FLAT,
        (100.1, 100.3, 100.1, 100.3, 10),
        (100.3, 100.3, 100.0, 100.0, 10),
        FLAT,
    ]
    r = _one(rows, side=1, stop=50, breakeven_points=15)
    assert r.exit_reason == "breakeven"
    assert r.exit_price == pytest.approx(100.1)


def test_trail_stops_out():
    rows = [
        FLAT,
        (100.1, 100.5, 100.1, 100.5, 10),
        (100.5, 100.5, 100.2, 100.2, 10),
        FLAT,
    ]
    r = _one(rows, side=1, stop=100, trail_start_points=10, trail_points=20)
    assert r.exit_reason == "trail"
    assert r.exit_price == pytest.approx(100.3)


def test_end_of_data_exit():
    rows = [FLAT, (100, 100.2, 99.9, 100.1, 10), (100.1, 100.3, 100.0, 100.2, 10)]
    t = fast_sim.simulate(_bars(rows), _req(1, 500, exit_bar=50), PT)
    r = t.iloc[0]
    assert (r.exit_reason, r.exit_time) == ("end_of_data", _bars(rows).index[-1])
    assert r.exit_price == pytest.approx(100.1)
