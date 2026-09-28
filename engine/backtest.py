"""Daily event-driven backtest of the full pipeline (Blueprint Part H1) plus the baselines.

Decision at the close of day t (live: 15:40 ET). Fills at the close with a cost; stops and
partial takes are checked against day t+1's high/low. Weeks run Monday-open to Friday-close."""
import numpy as np
import pandas as pd

from . import config as K
from .portfolio import control
from . import policy


def regime_of(row) -> str:
    vix, ma200, chg = row.get("m_vix", 20), row.get("m_spy_ma200", 0), row.get("m_vix_chg5", 0)
    if vix > 30 or (ma200 < 0 and chg > 0.2):
        return "stress"
    if ma200 > 0 and vix < 20:
        return "calm"
    return "choppy"


class Book:
    def __init__(self, cash):
        self.cash, self.pos = cash, {}   # pos: ticker -> dict(sh, entry, stop, day, took)

    def value(self, px):
        return self.cash + sum(p["sh"] * px.get(t, p["entry"]) for t, p in self.pos.items())

    def trade(self, t, sh_delta, price, cost_bps):
        if sh_delta == 0 or not np.isfinite(price):
            return 0.0
        amt = sh_delta * price
        cost = abs(amt) * cost_bps / 1e4
        self.cash -= amt + cost
        p = self.pos.setdefault(t, {"sh": 0.0, "entry": price, "stop": 0.0, "day": 0, "took": False})
        if sh_delta > 0 and p["sh"] >= 0:
            p["entry"] = (p["entry"] * p["sh"] + amt) / (p["sh"] + sh_delta) if p["sh"] > 0 else price
        p["sh"] += sh_delta
        if abs(p["sh"]) < 1e-9:
            del self.pos[t]
        return cost


def run(P, score, X, stocks, market, sic, start=None, n_scen=3000, name="weekly7", log=print, seed=1,
        objective="goal", vol_filter=False, mu_shrink=0.5, end=None, rebalance="weekly",
        min_dv=2e7, keep_pct=0.90, stop_atr=None):
    stop_atr = K.STOP_ATR if stop_atr is None else stop_atr
    rng = np.random.default_rng(seed)
    C, H, L = stocks["Close"], stocks["High"], stocks["Low"]
    rets = np.log(C / C.shift(1))
    spy_r = np.log(market["Close"]["SPY"] / market["Close"]["SPY"].shift(1)).reindex(C.index)
    beta = (rets.rolling(120, min_periods=60).cov(spy_r) / spy_r.rolling(120, min_periods=60).var()).clip(0, 3)
    sec_map = sic.set_index("ticker")["sic"].astype(str).str[:1].reindex(C.columns).fillna("9")
    sec_codes = sorted(sec_map.unique())
    # sector factor returns = equal-weight mean of members minus market
    sec_ret = pd.DataFrame({s: rets.loc[:, sec_map == s].mean(axis=1) - spy_r for s in sec_codes})
    fac = pd.concat([spy_r.rename("mkt"), sec_ret], axis=1)

    dates = sorted(set(score.index.get_level_values(0)))
    if start:
        dates = [d for d in dates if d >= pd.Timestamp(start)]
    if end:
        dates = [d for d in dates if d <= pd.Timestamp(end)]
    book = Book(K.START_CASH)
    eq, week_start_val, logs = [], K.START_CASH, []
    stats = {"turnover": 0.0, "cost": 0.0, "stop": 0, "partial": 0, "time": 0, "stop_pnl": 0.0}
    all_dates = C.index
    for i, d in enumerate(dates[:-1]):
        nxt = all_dates[all_dates.get_loc(d) + 1]
        px = C.loc[d].to_dict()
        val = book.value(px)
        # new week begins after the last session of a calendar week
        is_week_end = nxt.isocalendar().week != d.isocalendar().week or (nxt - d).days > 3
        week_ret = val / week_start_val - 1
        days_left = 0 if is_week_end else max(1, 4 - d.weekday())
        if is_week_end:
            need, horizon = K.WEEKLY_TARGET, 5
        else:
            need, horizon = (1 + K.WEEKLY_TARGET) / (1 + week_ret) - 1, days_left
        xr = X.xs(d, level=0)
        regime = regime_of(xr.iloc[0]) if len(xr) else "choppy"
        s = score.xs(d, level=0).dropna()
        pr = P.xs(d, level=0).reindex(s.index)
        params = {"objective": objective, "vol_filter": vol_filter, "min_dv": min_dv, "keep_pct": keep_pct,
                  "mu_shrink": mu_shrink}
        top = s[policy.eligible(xr.reindex(s.index), params)].nlargest(K.N_CANDIDATES).index
        gross, mode = control(0 if is_week_end else week_ret, days_left, bool((pr.loc[top, "mu_raw"] > 0).any()), regime)
        hold_week = rebalance == "weekly" and not is_week_end and len(book.pos) > 0 and mode not in ("brake",)
        cur = pd.Series({t: book.pos[t]["sh"] * px[t] / val for t in book.pos if np.isfinite(px.get(t, np.nan))}, dtype=float)
        if hold_week:
            # mid-week: keep the week's plan; only the risk rules below act (no churn)
            target_w, info = cur, {"p_target": np.nan}
        elif mode == "bank" and not is_week_end:
            target_w, info = cur * (gross / max(1e-9, cur.sum())), {"p_target": np.nan}
        else:
            target_w, info = policy.plan(xr, pr, s, list(book.pos), gross, need, horizon, beta.loc[d], sec_map,
                                         fac.loc[:d].tail(250), rng=rng, n_scen=n_scen, params=params)
        # rebalance with a no-trade band
        cost = 0.0
        cur_w = {t: book.pos[t]["sh"] * px[t] / val for t in book.pos}
        for t in set(cur_w) | set(target_w.index):
            dw = target_w.get(t, 0.0) - cur_w.get(t, 0.0)
            if abs(dw) < K.REBALANCE_BAND and t in target_w.index and t in cur_w:
                continue
            price = px.get(t)
            if not price or not np.isfinite(price):
                continue
            liq = xr.loc[t, "log_dv"] if t in xr.index else 0
            bps = K.COST_BPS_LIQUID if liq > np.log1p(5e7) else K.COST_BPS_ILLIQUID
            cost += book.trade(t, dw * val / price, price, bps)
            stats["turnover"] += abs(dw)
            if t in book.pos and dw > 0:
                atr = xr.loc[t, "atr_pct"] * price if t in xr.index else 0.05 * price
                book.pos[t]["stop"] = price - stop_atr * atr
                book.pos[t]["day"] = 0
        # next session: stops / partial takes against its range, then mark to its close
        for t in list(book.pos):
            p = book.pos[t]
            hi, lo, cl = H.at[nxt, t], L.at[nxt, t], C.at[nxt, t]
            if not np.isfinite(cl):
                continue
            p["day"] += 1
            if lo <= p["stop"]:
                stats["stop"] += 1; stats["stop_pnl"] += p["sh"] * (min(p["stop"], cl) - p["entry"])
                book.trade(t, -p["sh"], min(p["stop"], C.at[nxt, t] if cl < p["stop"] else p["stop"]), K.COST_BPS_ILLIQUID)
                continue
            if not p["took"] and hi >= p["entry"] * (1 + K.PARTIAL_TAKE):
                stats["partial"] += 1
                book.trade(t, -p["sh"] / 2, p["entry"] * (1 + K.PARTIAL_TAKE), K.COST_BPS_LIQUID)
                if t in book.pos:
                    book.pos[t]["took"] = True
                continue
            if rebalance != "weekly" and p["day"] >= K.TIME_STOP_DAYS and cl < p["entry"]:
                stats["time"] += 1
                book.trade(t, -p["sh"], cl, K.COST_BPS_LIQUID)
        stats["cost"] += cost
        v_next = book.value(C.loc[nxt].to_dict())
        eq.append((nxt, v_next))
        logs.append({"date": d, "regime": regime, "mode": mode, "need": need, "p_target": info.get("p_target"),
                     "n": len(target_w), "cost": cost, "week_ret": week_ret})
        if is_week_end:
            week_start_val = val
        if i % 100 == 0:
            log(f"  {name} {d.date()} equity {v_next:,.0f} regime {regime} holdings {len(book.pos)}")
    run.stats = stats
    return pd.Series(dict(eq)).rename(name), pd.DataFrame(logs)


def baseline_score(score, C, k=8, start=None, name="equal_weight"):
    return _weekly_pick(score, C, k, start, name)


def weekly(equity: pd.Series) -> pd.Series:
    return equity.resample("W-FRI").last().pct_change().dropna()


def baseline_trend(X, C, k=8, start=None):
    """Trend-Chaser: every Friday buy the top-k by 20-day return x volume surge, equal weight."""
    s = (X["r20"].groupby(level=0).rank(pct=True) + X["vol_surge5"].groupby(level=0).rank(pct=True))
    return _weekly_pick(s, C, k, start, "trend_chaser")


def baseline_random(X, C, k=8, start=None, seed=3):
    rng = np.random.default_rng(seed)
    s = pd.Series(rng.random(len(X)), index=X.index)
    s = s.where(X["vol20"].groupby(level=0).rank(pct=True) > 0.7)     # high-vol, matched to Weekly7's appetite
    return _weekly_pick(s, C, k, start, "random_vol")


def _weekly_pick(s, C, k, start, name):
    fr = C.resample("W-FRI").last()
    val, out = K.START_CASH, {}
    wk = [d for d in fr.index if (start is None or d >= pd.Timestamp(start))]
    days = C.index
    for a, b in zip(wk[:-1], wk[1:]):
        da = days[days <= a][-1]; db = days[days <= b][-1]
        try:
            picks = s.xs(da, level=0).dropna().nlargest(k).index
        except KeyError:
            continue
        r = (C.loc[db, picks] / C.loc[da, picks] - 1).fillna(0).mean() - 2 * K.COST_BPS_LIQUID / 1e4
        val *= 1 + r
        out[db] = val
    return pd.Series(out).rename(name)


def run_topk(score, X, stocks, start=None, end=None, k=4, exit_q=0.8, bank=None, brake=None,
             bank_exposure=0.4, brake_exposure=1 / 3, params=None, cost_liquid=K.COST_BPS_LIQUID,
             cost_illiquid=K.COST_BPS_ILLIQUID, name="topk", sectors=None, max_per_sector=None):
    """Daily simulation of the champion constructor. Rebalance at each week's last close;
    optional intra-week bank (+x%: cut to bank_exposure) and brake (-y%: cut to brake_exposure),
    checked at each close. Returns (equity Series, stats)."""
    C = stocks["Close"]
    days = C.index
    sdays = sorted(set(score.index.get_level_values(0)))
    if start: sdays = [d for d in sdays if d >= pd.Timestamp(start)]
    if end: sdays = [d for d in sdays if d <= pd.Timestamp(end)]
    sset = set(sdays)
    w = pd.Series(dtype=float)       # current weights (fraction of equity at last rebalance)
    sh = {}                          # shares per $1 of equity
    val, cash = K.START_CASH, K.START_CASH
    pos = {}                         # ticker -> shares
    eq, stats = {}, {"turnover": 0.0, "cost": 0.0, "banks": 0, "brakes": 0}
    week_start, capped = val, False
    i0 = days.get_loc(sdays[0])
    for i in range(i0, days.get_loc(sdays[-1]) + 1):
        d = days[i]
        px = C.loc[d]
        val = cash + sum(q * px.get(t, np.nan) if np.isfinite(px.get(t, np.nan)) else 0 for t, q in pos.items())
        nxt = days[i + 1] if i + 1 < len(days) else d + pd.Timedelta(days=3)
        week_end = nxt.isocalendar().week != d.isocalendar().week or (nxt - d).days > 3
        wr = val / week_start - 1
        target = None
        if week_end and d in sset:
            xr = X.xs(d, level=0)
            s = score.xs(d, level=0).dropna()
            ok = policy.eligible(xr.reindex(s.index), params).fillna(False)
            target = policy.topk_targets(s[ok], list(pos), k, exit_q, sectors, max_per_sector)
        elif not capped and bank is not None and wr >= bank:
            stats["banks"] += 1; capped = True
            target = pd.Series({t: q * px[t] / val for t, q in pos.items()}) * bank_exposure
        elif not capped and brake is not None and wr <= -brake:
            stats["brakes"] += 1; capped = True
            target = pd.Series({t: q * px[t] / val for t, q in pos.items()}) * brake_exposure
        if target is not None:
            xr_liq = X.xs(d, level=0)["log_dv"] if d in sset else pd.Series(dtype=float)
            for t in set(pos) | set(target.index):
                p = px.get(t, np.nan)
                if not np.isfinite(p):
                    continue
                cur = pos.get(t, 0.0) * p
                dv = target.get(t, 0.0) * val - cur
                if abs(dv) < 1e-6:
                    continue
                bps = cost_liquid if xr_liq.get(t, 0) > np.log1p(5e7) else cost_illiquid
                c = abs(dv) * bps / 1e4
                cash -= dv + c
                pos[t] = pos.get(t, 0.0) + dv / p
                if abs(pos[t]) * p < 0.01:
                    pos.pop(t)
                stats["turnover"] += abs(dv) / val; stats["cost"] += c
            val = cash + sum(q * px[t] for t, q in pos.items() if np.isfinite(px.get(t, np.nan)))
        eq[d] = val
        if week_end:
            week_start, capped = val, False
    return pd.Series(eq).rename(name), stats
