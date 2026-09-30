"""Daily event-driven backtest of the full pipeline (Blueprint Part H1) plus the baselines.

Decision at the close of day t. Canon C33 / Bible Phase 1.4 (INTEGRATION B01): every rebalance order is priced by
engine.pit's Executor at the STRICT next session's open (`fill="next_open"`, the default) and the run's own fill ledger is
proven afterwards by engine.fill_audit -> pit.audit_fills, the same gate the Test loop uses; a failed proof raises
pit.FailClosed. `fill="close"` is kept only as the legacy control that reproduces pre-C33 numbers; its audit FAILS by
design (same-close fills). Stops and partial takes are resting orders checked against day t+1's high/low.
Weeks run Monday-open to Friday-close.

B07 (INTEGRATION): the per-type TrustTable and the DirectionEngine can re-weight / filter the score through ScoreHooks,
behind explicit flags that are OFF by default (OFF is byte-identical to no hooks). Neither has a measured edge (direction
51.8%, Brier = base rate): ON is an experiment switch, not an improvement.
B12 (INTEGRATION): `run` (the full-pipeline backtest, a major run) writes a checkpoint bundle (checkpoint.write_checkpoint)
and a Phase 36 run report (run_report.build_report / write_report) through record_major_run."""
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as K
from .portfolio import control
from . import policy
from . import pit, fill_audit

FILL_MODES = ("next_open", "close")


# ==================================================================================================================
# B01: decide at the close, fill at the next open, prove it
# ==================================================================================================================
class Fills:
    """The backtest's order ledger in the Session.fills shape, so fill_audit.audit_session proves a backtest exactly as it
    proves a Test-loop Session (reuse, not a second audit). next_open orders are priced ONLY by a pit.Executor bound to the
    decision close; an order whose next session has no open is left unfilled (never priced at a substitute)."""

    def __init__(self, stocks, closes: pd.DataFrame, mode: str = "next_open"):
        if mode not in FILL_MODES:
            raise ValueError(f"fill must be one of {FILL_MODES}, not {mode!r}")
        self.mode = mode
        self.closes = closes
        O = stocks.get("Open") if hasattr(stocks, "get") else None
        if O is None and mode == "next_open":
            raise pit.FailClosed("next-open fills need stocks['Open']: without opening prices C33 cannot be honoured")
        self.opens = None if O is None else O.reindex(index=closes.index, columns=closes.columns)
        self.store = None
        if mode == "next_open":
            self.store = pit.PITStore(pit.Calendar(closes.index)).add_prices("prices", {"Open": self.opens})
        self.fills, self.unfilled = [], []
        self._n, self._ex = {}, {}

    def _executor(self, d):
        ex = self._ex.get(d)
        if ex is None:
            ex = self._ex[d] = self.store.executor(d)
        return ex

    def order(self, d, ticker, shares, close_px):
        """Place an order decided at the close of `d`. Returns the fill price, or None when it could not be filled."""
        d = pd.Timestamp(d)
        n = self._n[(d, ticker)] = self._n.get((d, ticker), 0) + 1
        oid = f"{d.date()}:{ticker}:{n}"
        if self.mode == "close":                              # legacy control: fails the audit by construction
            price, fdate = float(close_px), d
        else:
            try:
                rec = self._executor(d).fill_next_open(ticker, shares, oid)
            except pit.FillTimingError as e:
                self.unfilled.append({"order_id": oid, "ticker": ticker, "decision_date": str(d.date()), "why": str(e)})
                return None
            price, fdate = rec["fill_price"], rec["fill_date"]
        self.fills.append({"order_id": oid, "ticker": ticker, "decision_date": str(d.date()), "fill_date": str(fdate.date()),
                           "fill_price": float(price), "shares": float(shares)})
        return float(price)

    def audit(self) -> dict:
        """fill_audit's verdict over this run (PASS / FAIL / UNAUDITABLE) plus the unfilled count."""
        r = fill_audit.audit_session(self, self.opens, self.closes)
        r["mode"], r["n_unfilled"] = self.mode, len(self.unfilled)
        return r


def _span(C: pd.DataFrame, first, last) -> pd.DataFrame:
    """Closes from the first decision through the session after the last one (the last fill needs its open)."""
    idx = C.index
    lo = idx.searchsorted(pd.Timestamp(first))
    hi = min(len(idx), idx.searchsorted(pd.Timestamp(last), side="right") + 1)
    return C.iloc[lo:hi]


def _close_audit(fills: Fills, log=print) -> dict:
    """Run the proof; a next-open run whose own ledger fails it is a simulator bug and must not produce numbers."""
    r = fills.audit()
    if fills.mode == "next_open" and r["status"] != "PASS":
        raise pit.FailClosed(f"backtest fill audit {r['status']}: {r.get('errors', [])[:3]}")
    if fills.mode == "close":
        log(f"  LEGACY close fills: fill audit {r['status']} (same-close execution violates canon C33)")
    return r


# ==================================================================================================================
# B07: trust table and direction engine, OFF by default
# ==================================================================================================================
@dataclass
class ScoreHooks:
    """Explicit switches for the two B07 modules where the score is consumed. Both OFF = the score passes through untouched.
    trust ON:     each indicator's cross-sectional z-score is scaled by the stock type's posterior reliability
                  (TrustTable.neutralize: unreliable types contribute exactly 0) and the trusted composite is blended
                  into the score's rank with weight `trust_weight`.
    direction ON: only rows the DirectionEngine bets UP on stay tradable (a closed engine abstains on every row, so ON with
                  a closed engine trades nothing - that is the honest consequence of no measured direction edge).
    Each fitted object carries the `now` it was fitted at; using it on a decision date before that is look-ahead."""
    trust_on: bool = False
    trust: object = None
    trust_indicators: tuple = ()
    trust_info: pd.DataFrame | None = None
    trust_weight: float = 0.5
    direction_on: bool = False
    direction: object = None
    direction_inputs: pd.DataFrame | None = None
    direction_now: object = None
    log: list = field(default_factory=list)

    def validate(self):
        if self.trust_on:
            if self.trust is None or getattr(self.trust, "now", None) is None:
                raise ValueError("trust_on needs a fitted TrustTable (with .now)")
            if not self.trust_indicators:
                raise ValueError("trust_on needs trust_indicators")
            if not 0.0 < float(self.trust_weight) <= 1.0:
                raise ValueError("trust_weight must be in (0, 1]")
            if self.trust_info is None:
                raise ValueError("trust_on needs trust_info (the (date, ticker) type facets)")
        if self.direction_on:
            if self.direction is None or self.direction_inputs is None or self.direction_now is None:
                raise ValueError("direction_on needs a fitted DirectionEngine, its input panel and the now it was fitted at")
        return self

    @property
    def active(self) -> bool:
        return bool(self.trust_on or self.direction_on)

    def apply(self, d, s: pd.Series, xr: pd.DataFrame) -> pd.Series:
        if not self.active:
            return s
        d = pd.Timestamp(d)
        out, note = s, {"date": str(d.date()), "n_in": int(len(s))}
        if self.trust_on:
            if pd.Timestamp(self.trust.now) > d:
                raise pit.LookAheadError(f"trust table fitted at {pd.Timestamp(self.trust.now).date()} used on {d.date()}")
            ind = [c for c in self.trust_indicators if c in xr.columns]
            Z = xr.reindex(out.index)[ind].astype(float)
            Z = ((Z - Z.mean()) / Z.std(ddof=0).replace(0, np.nan)).fillna(0.0)
            info = _day(self.trust_info, d).reindex(out.index)
            trusted = self.trust.neutralize(Z, info).sum(axis=1)
            w = float(self.trust_weight)
            out = (1 - w) * out.rank(pct=True) + w * trusted.rank(pct=True)
            note["trust_nonzero"] = int((trusted != 0).sum())
        if self.direction_on:
            if pd.Timestamp(self.direction_now) > d:
                raise pit.LookAheadError(f"direction engine fitted at {pd.Timestamp(self.direction_now).date()} used on {d.date()}")
            F = _day(self.direction_inputs, d).reindex(out.index)
            dec = self.direction.decide(F)
            keep = (dec["bet"] & (dec["side"] > 0)).reindex(out.index).fillna(False).to_numpy(dtype=bool)
            out = out[keep]
            note["direction_bets"] = int(keep.sum())
        note["n_out"] = int(len(out))
        self.log.append(note)
        return out


def _day(panel: pd.DataFrame, d) -> pd.DataFrame:
    try:
        return panel.xs(d, level=0)
    except KeyError:
        return panel.iloc[0:0].droplevel(0)


# ==================================================================================================================
# B12: every major run leaves a checkpoint bundle and a Phase 36 report
# ==================================================================================================================
def _jsonable(o):
    import json
    from .checkpoint import _finite
    return _finite(json.loads(json.dumps(o, default=str)))


def record_major_run(kind, name, config, metrics, summary, seed, weekly=None, week_dates=None, windows=None, gates=None,
                     turnover=None, costs=None, unproven=(), logs=None, now=None, ckpt_root=None, report_dir=None):
    """checkpoint.write_checkpoint + run_report.build_report/write_report for one finished run. The run id carries the
    microsecond so two runs in the same second never collide (both writers refuse to overwrite). Returns
    {"run_id", "bundle", "report_json", "report_txt", "decision"}."""
    from . import checkpoint, provenance, run_report
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    run_id = f"{kind}_{re.sub(r'[^A-Za-z0-9_.-]', '_', str(name))}_{now:%Y%m%dT%H%M%S%f}"
    st = provenance.stamp(_jsonable(config), seed)
    seeds = {"seed": seed} if not isinstance(seed, dict) else seed
    bundle = checkpoint.write_checkpoint(ckpt_root or checkpoint.default_root(), run_id, _jsonable(config), _jsonable(metrics),
                                         _jsonable(seeds), _jsonable(summary), str(now), logs=logs, provenance=_jsonable(st))
    run_rec = {"run_id": run_id, "git_commit": st.get("git_commit"), "canon_hash": st.get("canon_sha"),
               "blueprint_version": st.get("blueprint_version"), "config_hash": st.get("config_hash"),
               "data_snapshot": st.get("data_snapshot"), "seed": seed if not isinstance(seed, dict) else str(seed),
               "windows": windows or {}, "window_ids": [kind], "weekly": weekly, "week_dates": week_dates,
               "turnover": turnover, "costs": costs, "gates": gates or {}, "unproven": list(unproven)}
    rep = run_report.build_report(run_rec, now)
    j, t = run_report.write_report(rep, out_dir=report_dir)
    return {"run_id": run_id, "bundle": str(bundle), "report_json": str(j), "report_txt": str(t), "decision": rep["decision"]}


def _record_backtest(name, config, equity, stats, fills_audit, seed, record, log):
    """B12 for a backtest: weekly returns from the equity curve, the fill proof as the time_fence gate."""
    opts = record if isinstance(record, dict) else {}
    wk = weekly(equity) if len(equity) > 1 else pd.Series(dtype=float)
    metrics = {"final": float(equity.iloc[-1]) if len(equity) else None, "n_days": int(len(equity)), "n_weeks": int(len(wk)),
               "mean_week": float(wk.mean()) if len(wk) else None, **{k: v for k, v in stats.items() if k != "fill_audit"}}
    summary = {"fill_audit": fills_audit, "start": str(equity.index[0].date()) if len(equity) else None,
               "end": str(equity.index[-1].date()) if len(equity) else None}
    windows = {"test": {"start": summary["start"], "end": summary["end"]}} if len(equity) else {}
    unproven = ["price panel is survivor-only: every backtest here is survivorship-biased"]
    if fills_audit.get("mode") == "close":
        unproven.append("LEGACY close fills: execution at the decision close violates canon C33")
    try:
        out = record_major_run("backtest", name, config, metrics, summary, seed, weekly=wk.values, week_dates=list(wk.index),
                               windows=windows, gates={"time_fence": fills_audit.get("status") == "PASS"},
                               turnover=stats.get("turnover"), costs=stats.get("cost"), unproven=unproven,
                               now=opts.get("now"), ckpt_root=opts.get("ckpt_root"), report_dir=opts.get("report_dir"))
        log(f"  {name}: checkpoint {out['bundle']}, report {out['report_json']} ({out['decision']})")
        return out
    except Exception as e:                           # noqa: BLE001 - a finished backtest's numbers are kept; the gap is loud
        log(f"  {name}: RUN RECORD FAILED ({type(e).__name__}: {e}) - no checkpoint/report for this run")
        return {"error": f"{type(e).__name__}: {e}"}


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
        min_dv=2e7, keep_pct=0.90, stop_atr=None, fill="next_open", hooks: ScoreHooks | None = None, record=True):
    """fill: 'next_open' (C33) or the legacy 'close' control. hooks: B07 ScoreHooks (default OFF).
    record: True (default dirs) / {"ckpt_root", "report_dir", "now"} / False - B12 checkpoint + run report.
    run.stats carries 'fill_audit' and 'unfilled'; run.record the written paths."""
    stop_atr = K.STOP_ATR if stop_atr is None else stop_atr
    hooks = (hooks or ScoreHooks()).validate()
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
    fills = Fills(stocks, _span(C, dates[0], dates[-1]) if dates else C.iloc[0:0], fill)
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
        s = hooks.apply(d, score.xs(d, level=0).dropna(), xr)
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
            shares = dw * val / price                     # sized on the close that was known; priced at the fill
            price = fills.order(d, t, shares, price)
            if price is None:
                continue
            cost += book.trade(t, shares, price, bps)
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
    audit = _close_audit(fills, log)
    stats["fill_audit"], stats["unfilled"] = audit["status"], len(fills.unfilled)
    run.stats, run.fills, run.hooks_log = stats, fills, hooks.log
    equity = pd.Series(dict(eq), dtype=float).rename(name)
    run.record = None
    if record:
        cfg = {"name": name, "start": start, "end": end, "n_scen": n_scen, "objective": objective, "vol_filter": vol_filter,
               "mu_shrink": mu_shrink, "rebalance": rebalance, "min_dv": min_dv, "keep_pct": keep_pct, "stop_atr": stop_atr,
               "fill": fill, "trust_on": hooks.trust_on, "direction_on": hooks.direction_on}
        run.record = _record_backtest(name, cfg, equity, stats, audit, seed, record, log)
    return equity, pd.DataFrame(logs)


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
             cost_illiquid=K.COST_BPS_ILLIQUID, name="topk", sectors=None, max_per_sector=None,
             pick="top", pool_q=0.95, peek=None, fill="next_open", hooks: ScoreHooks | None = None, record=False):
    """Daily simulation of the champion constructor. Rebalance at each week's last close; optional intra-week bank
    (+x%: cut to bank_exposure) and brake (-y%: cut to brake_exposure), checked at each close. Returns (equity, stats).
    fill='next_open' (C33): an order decided at close d is filled at the next session's open by pit's Executor and the
    ledger is proven by fill_audit afterwards (stats['fill_audit']); 'close' is the legacy control (audit FAILS).
    hooks: B07 ScoreHooks, OFF by default. record: B12 checkpoint + report (off by default: a grid calls this hundreds of
    times; the grid scripts gate launches through experiment_memory instead).
    peek: CHEATING CONTROL ONLY - a Series of future returns used to prove the harness can tell a clairvoyant picker from
    ours. Never set in real tests."""
    hooks = (hooks or ScoreHooks()).validate()
    C = stocks["Close"]
    days = C.index
    sdays = sorted(set(score.index.get_level_values(0)))
    if start: sdays = [d for d in sdays if d >= pd.Timestamp(start)]
    if end: sdays = [d for d in sdays if d <= pd.Timestamp(end)]
    if not sdays:
        return pd.Series(dtype=float).rename(name), {"turnover": 0.0, "cost": 0.0, "banks": 0, "brakes": 0,
                                                     "fill_audit": "EMPTY", "unfilled": 0}
    sset = set(sdays)
    val, cash = K.START_CASH, K.START_CASH
    pos = {}                         # ticker -> shares
    eq, stats = {}, {"turnover": 0.0, "cost": 0.0, "banks": 0, "brakes": 0}
    week_start, capped = val, False
    fills = Fills(stocks, _span(C, sdays[0], sdays[-1]), fill)
    pending = []                     # next_open: (decision date, ticker, shares, bps) filled at the next session's open

    def execute(d, t, q, bps, p):
        nonlocal cash
        price = fills.order(d, t, q, p)
        if price is None:
            return
        dv = q * price
        c = abs(dv) * bps / 1e4
        cash -= dv + c
        pos[t] = pos.get(t, 0.0) + q
        if abs(pos[t]) * price < 0.01:
            pos.pop(t)
        stats["turnover"] += abs(dv) / max(val, 1e-9); stats["cost"] += c

    i0 = days.get_loc(sdays[0])
    for i in range(i0, days.get_loc(sdays[-1]) + 1):
        d = days[i]
        px = C.loc[d]
        if pending:                  # morning: yesterday's close decisions fill at today's open
            val = cash + sum(q * px.get(t, np.nan) if np.isfinite(px.get(t, np.nan)) else 0 for t, q in pos.items())
            for dd, t, q, bps, p in pending:
                execute(dd, t, q, bps, p)
            pending = []
        val = cash + sum(q * px.get(t, np.nan) if np.isfinite(px.get(t, np.nan)) else 0 for t, q in pos.items())
        nxt = days[i + 1] if i + 1 < len(days) else d + pd.Timedelta(days=3)
        week_end = nxt.isocalendar().week != d.isocalendar().week or (nxt - d).days > 3
        wr = val / week_start - 1
        target = None
        if week_end and d in sset:
            xr = X.xs(d, level=0)
            s = score.xs(d, level=0).dropna()
            ok = policy.eligible(xr.reindex(s.index), params).fillna(False)
            sc = hooks.apply(d, s[ok], xr)
            if peek is not None:
                sc = peek.xs(d, level=0).reindex(sc.index).fillna(-1)
            target = policy.topk_targets(sc, list(pos), k, exit_q, sectors, max_per_sector,
                                         pick=pick, vol=xr["vol20"], pool_q=pool_q)
        elif not capped and bank is not None and wr >= bank:
            stats["banks"] += 1; capped = True
            target = pd.Series({t: q * px[t] / val for t, q in pos.items()}) * bank_exposure
        elif not capped and brake is not None and wr <= -brake:
            stats["brakes"] += 1; capped = True
            target = pd.Series({t: q * px[t] / val for t, q in pos.items()}) * brake_exposure
        if target is not None:
            xr_liq = X.xs(d, level=0)["log_dv"] if d in sset else pd.Series(dtype=float)
            for t in sorted(set(pos) | set(target.index)):
                p = px.get(t, np.nan)
                if not np.isfinite(p):
                    continue
                cur = pos.get(t, 0.0) * p
                dv = target.get(t, 0.0) * val - cur
                if abs(dv) < 1e-6:
                    continue
                bps = cost_liquid if xr_liq.get(t, 0) > np.log1p(5e7) else cost_illiquid
                if fill == "close":
                    execute(d, t, dv / p, bps, p)
                else:
                    pending.append((d, t, dv / p, bps, p))
            val = cash + sum(q * px[t] for t, q in pos.items() if np.isfinite(px.get(t, np.nan)))
        eq[d] = val
        if week_end:
            week_start, capped = val, False
    for dd, t, q, bps, p in pending:                  # decided on the last day: the next open is outside the run
        fills.unfilled.append({"order_id": None, "ticker": t, "decision_date": str(dd.date()), "why": "after the run"})
    audit = _close_audit(fills)
    stats["fill_audit"], stats["unfilled"] = audit["status"], len(fills.unfilled)
    run_topk.fills, run_topk.hooks_log, run_topk.record = fills, hooks.log, None
    equity = pd.Series(eq, dtype=float).rename(name)
    if record:
        cfg = {"name": name, "start": start, "end": end, "k": k, "exit_q": exit_q, "bank": bank, "brake": brake,
               "bank_exposure": bank_exposure, "brake_exposure": brake_exposure, "params": params, "pick": pick,
               "pool_q": pool_q, "max_per_sector": max_per_sector, "fill": fill, "trust_on": hooks.trust_on,
               "direction_on": hooks.direction_on}
        run_topk.record = _record_backtest(name, cfg, equity, stats, audit, 0, record, print)
    return equity, stats
