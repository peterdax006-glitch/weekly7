"""Replay a random past year as if it were live (canon C9), with time running very fast.

No cheating:
  - the model is trained ONLY on data that ended >= 10 sessions before the year starts;
  - every trading day, the system sees only features computed from data up to that day
    (the panel is point-in-time: filings count from the session after they became public);
  - trades happen only on real trading days, at that day's close, with costs;
  - the future is used for exactly one thing: marking positions to the next day's price.
Uses the live champion's own rules (engine.policy): eligibility filter, top-k with hysteresis,
sector cap, weekly rebalance on the week's last session, daily -8% weekly brake.

usage: replay_year.py [year]   (no year = pick one at random)"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from engine import config as K, data, features, model, policy
from engine.explain import explain

t0 = time.time()
rng = np.random.default_rng()
YEARS = list(range(2017, 2026))                       # full years with >= 4 years of prior training data
Y = int(sys.argv[1]) if len(sys.argv) > 1 else int(rng.choice(YEARS))
print(f"Replaying {Y} as if live (drawn at random from {YEARS[0]}-{YEARS[-1]})", flush=True)

stocks, market = data.load("stocks"), data.load("market")
C = stocks["Close"]
X = pd.read_parquet(K.CACHE / "panel.parquet")
sic = pd.read_parquet(K.CACHE / "sic.parquet")
divs = {t: policy.sic_division(c) for t, c in zip(sic["ticker"], sic["sic"])}

# ---------- train the model the way it would have existed on Jan 1 of Y ----------
dates = X.index.get_level_values(0)
udates = pd.DatetimeIndex(sorted(dates.unique()))
first_day = udates[udates.year == Y][0]
cut = udates[udates.searchsorted(first_day) - model.EMBARGO]
lo = first_day - pd.DateOffset(years=model.TRAIN_YEARS)
C1 = C.shift(1)
atr = np.maximum(stocks["High"] - stocks["Low"], np.maximum((stocks["High"] - C1).abs(), (stocks["Low"] - C1).abs())).rolling(14, min_periods=10).mean()
yb, fw = features.labels(stocks, atr)
train_days = udates[(udates >= lo) & (udates < cut)][::2]
Xtr = X[dates.isin(train_days)]
y = yb.stack(future_stack=True).reindex(Xtr.index)
f = fw.stack(future_stack=True).reindex(Xtr.index)
# a label for day t needs prices up to t+5: those must also end before the year starts
last_label_day = udates[udates.searchsorted(train_days.max()) + 5]
assert last_label_day < first_day, "leak: a training label reaches into the replay year"
ok = y.notna().values & f.notna().values
Rtr = model.normalise(Xtr)[ok]
m = model.fit_models(Rtr, y[ok], f[ok])
print(f"model trained on {len(Rtr):,} rows, {train_days.min().date()} .. {train_days.max().date()} "
      f"(labels end {last_label_day.date()}; replay starts {first_day.date()})  {time.time() - t0:.0f}s", flush=True)
del Xtr, Rtr

# ---------- the year, one trading day at a time ----------
days = udates[udates.year == Y]
spy = market["Close"]["SPY"]
cash, pos = K.START_CASH, {}                 # pos: ticker -> shares
entry, why = {}, {}
week_start_val, capped = K.START_CASH, False
log_days, trades, weeks = [], [], []
cols = m["cols"]

def is_week_end(i):
    return i + 1 >= len(days) or days[i + 1].isocalendar().week != days[i].isocalendar().week

week_open = days[0]
for i, d in enumerate(days):
    if i > 0 and days[i - 1].isocalendar().week != d.isocalendar().week:
        week_open = d
    xr = X.xs(d, level=0)                    # what the system can see today (point-in-time)
    px = C.loc[d]
    val = cash + sum(q * px[t] for t, q in pos.items() if np.isfinite(px.get(t, np.nan)))
    wr = val / week_start_val - 1
    action = None
    if is_week_end(i) or not pos:
        R = model.normalise(X.xs(d, level=0, drop_level=False)).xs(d, level=0)
        R = R.reindex(columns=cols)
        pred = pd.DataFrame({"mu_raw": m["reg"].predict(R)}, index=R.index)
        pred["evidence"] = model.evidence_score(R)
        s = policy.score(pred)
        okm = policy.eligible(xr.reindex(s.index)).fillna(False)
        target = policy.topk_targets(s[okm], list(pos), k=4, exit_q=0.8, sectors=divs, max_per_sector=2)
        contrib = pd.DataFrame(m["reg"].booster_.predict(R.loc[target.index], pred_contrib=True)[:, :-1],
                               index=target.index, columns=cols)
        for t in target.index:
            if t not in pos:
                why[t] = explain(contrib.loc[t])
        action = "rebalance" if pos else "initial build"
    elif not capped and wr <= -policy.TOPK["brake"]:
        target = pd.Series({t: q * px[t] / val for t, q in pos.items()}) * policy.TOPK["brake_exposure"]
        capped, action = True, f"weekly brake ({wr:.1%})"
    if action:
        target = target * 0.985
        for t in sorted(set(pos) | set(target.index), key=lambda t: target.get(t, 0)):
            p = px.get(t, np.nan)
            if not np.isfinite(p):
                continue
            dv = target.get(t, 0.0) * val - pos.get(t, 0.0) * p
            if abs(dv) < 1.0:
                continue
            bps = K.COST_BPS_LIQUID if xr["log_dv"].get(t, 0) > np.log1p(5e7) else K.COST_BPS_ILLIQUID
            cash -= dv + abs(dv) * bps / 1e4
            pos[t] = pos.get(t, 0.0) + dv / p
            trades.append({"date": str(d.date()), "ticker": t, "side": "BUY" if dv > 0 else "SELL",
                           "dollars": round(dv, 2), "price": round(float(p), 2), "reason": action,
                           "why": why.get(t, []) if dv > 0 else []})
            if dv > 0 and t not in entry:
                entry[t] = float(p)
            if abs(pos[t]) * p < 0.5:
                pos.pop(t); entry.pop(t, None)
        val = cash + sum(q * px[t] for t, q in pos.items() if np.isfinite(px.get(t, np.nan)))
    log_days.append({"date": str(d.date()), "equity": round(val, 2), "holdings": sorted(pos)})
    if is_week_end(i):
        prev_end = spy.loc[:week_open].iloc[-2] if len(spy.loc[:week_open]) > 1 else np.nan
        wk_spy = spy.loc[:d].iloc[-1] / prev_end - 1
        weeks.append({"week_end": str(d.date()), "ret": val / week_start_val - 1, "spy": float(wk_spy),
                      "equity": round(val, 2), "holdings": sorted(pos), "brake": capped})
        week_start_val, capped = val, False

eq = pd.Series({pd.Timestamp(r["date"]): r["equity"] for r in log_days})
w = pd.Series([x["ret"] for x in weeks])
sp_year = spy.loc[str(Y)]
summary = {"year": Y, "start": K.START_CASH, "end": round(float(eq.iloc[-1]), 2), "year_return": float(eq.iloc[-1] / K.START_CASH - 1),
           "spy_year_return": float(sp_year.iloc[-1] / spy.loc[:days[0]].iloc[-2] - 1),
           "weeks": len(w), "weeks_ge_7": int((w >= 0.07).sum()), "weeks_positive": int((w > 0).sum()),
           "weeks_le_m7": int((w <= -0.07).sum()), "best_week": float(w.max()), "worst_week": float(w.min()),
           "mean_week": float(w.mean()), "max_drawdown": float((eq / eq.cummax() - 1).min()),
           "trades": len(trades), "brakes": int(sum(x["brake"] for x in weeks)),
           "model_trained_through": str(train_days.max().date()), "labels_end": str(last_label_day.date()),
           "runtime_seconds": round(time.time() - t0)}
out = K.STATE / "replays"
out.mkdir(parents=True, exist_ok=True)
(out / f"replay_{Y}.json").write_text(json.dumps({"summary": summary, "weeks": weeks, "trades": trades, "days": log_days},
                                                  indent=1, default=float))
from engine.improve import log_experiment
log_experiment({"event": "replay_year", **summary})
print(json.dumps(summary, indent=1, default=float))
