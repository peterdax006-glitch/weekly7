"""T11/T13 and canon C33 (Bible Phases 21-23, 32): the shared trading Session. Checks that decisions made at a close
fill at the NEXT session's open, the time fence refuses future prices, replays are deterministic, the future-scramble
gate holds (and can fail), costs bite, crypto is never bought, and the empty case does nothing."""
import numpy as np
import pandas as pd
import pytest

from engine import adaptive as A
from engine.universe import CRYPTO_TICKERS

CFG = {"k": 2, "exit_q": 0.8, "rebalance_weeks": 1, "brake": None, "max_per_sector": None, "w_model": 1.0,
       "pick": "top", "pool_q": 0.7, "liq_q": 0.0, "vol_filter": False, "stress_thr": None, "stress_k": 2,
       "trend_filter": None, "trend_gross": 0.0}
N_T = 12


def world(n=60, seed=0, tickers=None):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-07", periods=n)
    tk = tickers or [f"T{i:02d}" for i in range(N_T)]
    closes = pd.DataFrame(20 * np.exp(np.cumsum(rng.normal(0, 0.02, (n, len(tk))), axis=0)), index=idx, columns=tk)
    opens = closes.shift(1).fillna(closes.iloc[0]) * np.exp(rng.normal(0, 0.01, closes.shape))
    return closes, opens


def snap_for(day, tickers, mu):
    p = pd.DataFrame({"mu_raw": mu, "evidence": 0.5, "vol20": 0.02, "max20": 0.05, "log_dv": 18.0,
                      "ev_red_flag": 0.0, "ev_offering": 0.0, "r5": 0.0, "m_vix": 0.5, "m_vix_term": 0.9,
                      "m_spy_ma200": 1.05}, index=pd.Index(tickers, name="ticker"))
    return p


def snaps_from(closes, fn):
    """fn(day, closes_to_day) -> mu vector; a snapshot for every session (the Session asks only when it needs one)."""
    return {str(d.date()): snap_for(d, closes.columns, fn(d, closes.loc[:d])) for d in closes.index}


def honest(d, c):
    return c.iloc[-5:].pct_change().sum().values if len(c) > 5 else np.zeros(c.shape[1])   # past momentum only


def run(closes, opens, snaps, **kw):
    S = A.replay(CFG, snaps, closes, kw.pop("bps", 0.0), {}, opens=opens, **kw)
    return S


def test_decision_fills_at_next_open():
    closes, opens = world()
    S = run(closes, opens, snaps_from(closes, honest))
    days = list(closes.index.strftime("%Y-%m-%d"))
    decided = [d for d, _ in S.decisions]
    filled = sorted({o[0] for o in S.orders})
    assert filled, "no trades at all"
    for f in filled:
        prev = days[days.index(f) - 1]
        assert prev in decided, f"order on {f} without a decision at the previous close"
        assert f not in decided or days.index(f) > 0


def test_fill_uses_open_price_not_close():
    closes, opens = world()
    opens = closes * 1.10                                        # every open is 10% above the close
    S = run(closes, opens, snaps_from(closes, honest))
    d0, t0, dv, _ = S.orders[0]
    S2 = A.Session(CFG, {}, 0.0)
    qty_at_open = dv / opens.loc[d0, t0]
    qty_at_close = dv / closes.loc[d0, t0]
    # replay again stepwise to read the first position size
    for i, d in enumerate(closes.index):
        nxt = i + 1 >= len(closes) or closes.index[i + 1].isocalendar().week != d.isocalendar().week
        sn = snaps_from(closes.iloc[:i + 1], honest).get(str(d.date())) if S2.needs_snapshot(nxt) else None
        S2.on_day(d, closes.loc[d], closes.loc[:d], nxt, sn, opens.loc[d])
        if S2.pos:
            break
    assert S2.pos[t0] == pytest.approx(qty_at_open, rel=1e-6)
    assert abs(S2.pos[t0] - qty_at_close) > 1e-6


def test_time_fence_refuses_future_prices():
    closes, opens = world()
    S = A.Session(CFG, {}, 0.0)
    d = closes.index[10]
    with pytest.raises(A.TimeFence):
        S.on_day(d, closes.loc[d], closes.loc[:closes.index[11]], False, None)


def test_replay_is_deterministic():
    closes, opens = world(seed=3)
    sn = snaps_from(closes, honest)
    r1 = run(closes, opens, sn, adaptive=True).result()
    r2 = run(closes, opens, sn, adaptive=True).result()
    assert r1["year_return"] == r2["year_return"] and r1["adaptations"] == r2["adaptations"]


def test_future_scramble_leaves_earlier_decisions_unchanged():
    closes, opens = world(n=80, seed=4)
    sn = snaps_from(closes, honest)
    cut = closes.index[40]
    S1 = run(closes, opens, sn, adaptive=True)
    S2 = run(closes, opens, sn, adaptive=True, scramble_after=cut, seed=9)
    before = lambda S: [d for d in S.decisions if pd.Timestamp(d[0]) <= cut]
    assert before(S1) == before(S2)


def test_scramble_gate_can_fail_on_a_peeking_rule():
    """Control: a decision rule that reads tomorrow's price must be exposed by the scramble gate."""
    closes, opens = world(n=80, seed=5)
    # the cut must fall ON a decision day: a one-day peek from any earlier decision never crosses it
    dec = [pd.Timestamp(d) for d, _ in run(closes, opens, snaps_from(closes, honest)).decisions]
    cut = next(d for d in dec if d >= closes.index[40])
    rng = np.random.default_rng(9)
    scr = closes.copy()
    m = scr.index > cut
    scr.loc[m] = scr.loc[m].values * np.exp(rng.normal(0, 0.2, scr.loc[m].shape))

    def peek(src):
        return lambda d, c: (src.shift(-1).loc[d] / src.loc[d]).fillna(0).values   # tomorrow's return
    d1 = run(closes, opens, snaps_from(closes, peek(closes))).decisions
    d2 = run(scr, opens, snaps_from(scr, peek(scr))).decisions
    before = lambda D: [d for d in D if pd.Timestamp(d[0]) <= cut]
    assert before(d1) != before(d2)


def test_costs_reduce_the_result():
    closes, opens = world(seed=6)
    sn = snaps_from(closes, honest)
    free = run(closes, opens, sn, bps=0.0).result()["year_return"]
    paid = run(closes, opens, sn, bps=50.0).result()["year_return"]
    assert paid < free


def test_crypto_is_never_bought():
    cz = sorted(CRYPTO_TICKERS)[:2]
    tk = cz + [f"T{i:02d}" for i in range(6)]
    closes, opens = world(tickers=tk, seed=7)
    best = lambda d, c: np.array([10.0, 10.0] + [0.0] * 6)       # crypto names ranked best every day
    S = run(closes, opens, snaps_from(closes, best))
    assert not any(o[1] in cz for o in S.orders)
    assert S.orders, "the rest of the universe should still trade"


def test_no_snapshots_means_no_trades():
    closes, opens = world()
    S = run(closes, opens, {})
    assert S.orders == [] and S.result()["year_return"] == pytest.approx(0.0)


def test_weekly_returns_recorded_once_per_week():
    closes, opens = world(n=60)
    S = run(closes, opens, snaps_from(closes, honest))
    n_weeks = closes.index.to_series().dt.isocalendar().week.nunique()
    assert len(S.weeks) == n_weeks
