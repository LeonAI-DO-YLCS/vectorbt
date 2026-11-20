# Licensed under Apache 2.0 with Commons Clause, like the rest of vectorbt (see LICENSE.md).
"""Fast trade simulator for market-entry requests (one numba loop, one row per request).

Fill rules:

* A buy fills at the ask (bar open + spread) and a sell at the bid (bar open) on the entry bar.
* Stop and target are checked from the entry bar on; the entry bar's own high/low can hit them.
* Stop before target when both hit in one bar, unless a gap decides (a gap fills at the bar open).
  Otherwise the bar direction decides (MT5 tick order: a rising bar goes low first).
* Breakeven and trail use the best price up to the previous bar, then move the stop.
* A time exit fills at the exit bar's open; an exit time past the data fills at the last bar
  (reason "end_of_data").

Input: `bars` (index = bar open time; columns open, high, low, close, spread in points) and
`requests` (entry_time, exit_time, side, stop_points, target_points, optional trail_start_points,
trail_points, breakeven_points). Output columns: entry_time, exit_time, side, entry_price,
exit_price, stop_points, exit_reason, points.
"""

import numpy as np
import pandas as pd
from numba import njit

_COLUMNS = [
    "entry_time", "exit_time", "side", "entry_price", "exit_price",
    "stop_points", "exit_reason", "points",
]  # fmt: skip
_REASONS = np.array(["stop", "target", "time", "end_of_data", "trail", "breakeven"])


@njit(cache=True)
def _q(x, pt):
    return np.rint(x / pt) * pt


@njit(cache=True)
def _trail(k, i, buy, h, lo, sp, pt, spx, trs, trl, be, best, ent, sk):
    """After bar i (no exit): update the best price, then tighten the stop for the next bar."""
    if buy:
        best[k] = max(best[k], h[i])
        gain = best[k] - ent[k]
    else:
        best[k] = min(best[k], _q(lo[i] + sp[i], pt))
        gain = ent[k] - best[k]
    sgn = 1.0 if buy else -1.0
    if (
        not np.isnan(be[k])
        and gain >= be[k] * pt - 1e-9
        and sgn * (ent[k] - spx[k]) > 0
    ):
        spx[k] = ent[k]
        sk[k] = 5
    if not np.isnan(trl[k]):
        start = 0.0 if np.isnan(trs[k]) else trs[k]
        new = _q(best[k] - sgn * trl[k] * pt, pt)
        if gain >= start * pt - 1e-9 and sgn * (new - spx[k]) > 0:
            spx[k] = new
            sk[k] = 4


@njit(cache=True)
def _run(o, h, lo, c, sp, pt, side, i0, xs, stop, tp, trs, trl, be):
    r = len(side)
    ex = np.zeros(r, np.int64)
    ep = np.zeros(r)
    xp = np.zeros(r)
    rs = np.zeros(r, np.int64)
    spx = np.zeros(1)
    best = np.zeros(1)
    ent = np.zeros(1)
    sk = np.zeros(1, np.int64)
    trs1 = np.zeros(1)
    trl1 = np.zeros(1)
    be1 = np.zeros(1)
    for k in range(r):
        buy = side[k] > 0
        i = i0[k]
        p = _q(o[i] + sp[i], pt) if buy else o[i]
        ep[k] = p
        stp = _q(p - stop[k] * pt if buy else p + stop[k] * pt, pt)
        tpx = np.nan
        if not np.isnan(tp[k]):
            tpx = _q(p + tp[k] * pt if buy else p - tp[k] * pt, pt)
        spx[0] = stp
        best[0] = ent[0] = p
        sk[0] = 0
        trs1[0], trl1[0], be1[0] = trs[k], trl[k], be[k]
        j = i  # the fill bar's own high/low count too (fx_sim does the same)
        while True:
            if j >= xs[k]:  # time exit (or last bar) at the bar's open
                ex[k] = xs[k]
                xp[k] = o[xs[k]] if buy else _q(o[xs[k]] + sp[xs[k]], pt)
                rs[k] = 2
                break
            if buy:
                sh, th = lo[j] <= spx[0], h[j] >= tpx
                sg, tg = o[j] <= spx[0], o[j] >= tpx
            else:
                sh = _q(h[j] + sp[j], pt) >= spx[0]
                th = _q(lo[j] + sp[j], pt) <= tpx
                sg = _q(o[j] + sp[j], pt) >= spx[0]
                tg = _q(o[j] + sp[j], pt) <= tpx
            if j == i:  # a gap at the fill bar's own open is not a gap
                sg = tg = False
            if sh and th:
                if sg:
                    th = False
                elif tg:
                    sh = False
                elif (c[j] > o[j]) == buy:
                    th = False
                else:
                    sh = False
            if th or sh:
                gap = o[j] if buy else _q(o[j] + sp[j], pt)
                ex[k] = j
                if th:
                    rs[k] = 1
                    xp[k] = gap if tg else tpx
                else:
                    rs[k] = sk[0]
                    xp[k] = gap if sg else spx[0]
                break
            if j > i:
                _trail(0, j, buy, h, lo, sp, pt, spx, trs1, trl1, be1, best, ent, sk)
            j += 1
    return ex, ep, xp, rs


def simulate(bars: pd.DataFrame, requests: pd.DataFrame, point: float) -> pd.DataFrame:
    """Trade list (fx_sim's columns) for market-entry requests; same drops and exit rules as fx_sim."""
    idx, n = bars.index, len(bars)
    o, h, lo, c = (
        np.rint(bars[k].to_numpy(float) / point) * point
        for k in ("open", "high", "low", "close")
    )
    sp = bars["spread"].to_numpy(float) * point
    req = requests.reset_index(drop=True)
    i0 = idx.searchsorted(req["entry_time"]).astype(np.int64)
    ok = (req["entry_time"] >= idx[0]).to_numpy() & (i0 < n - 1)
    req, i0 = req[ok], i0[ok]
    xs = np.minimum(np.maximum(idx.searchsorted(req["exit_time"]), i0 + 1), n - 1)
    nan = np.full(len(req), np.nan)
    opt = [req[k].to_numpy(float) if k in req else nan for k in
           ("trail_start_points", "trail_points", "breakeven_points")]  # fmt: skip
    side = req["side"].to_numpy(float)
    ex, ep, xp, rs = _run(
        o, h, lo, c, sp, float(point), side, i0, xs.astype(np.int64),
        req["stop_points"].to_numpy(float), req["target_points"].to_numpy(float), *opt,
    )  # fmt: skip
    rs = np.where((rs == 2) & (req["exit_time"] > idx[-1]).to_numpy(), 3, rs)
    return pd.DataFrame(
        {
            "entry_time": idx[i0],
            "exit_time": idx[ex],
            "side": np.where(side > 0, 1, -1),
            "entry_price": ep,
            "exit_price": xp,
            "stop_points": req["stop_points"].to_numpy(),
            "exit_reason": _REASONS[rs],
            "points": np.round(side * (xp - ep) / point, 6),
        },
        columns=_COLUMNS,
    )
