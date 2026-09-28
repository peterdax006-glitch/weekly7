"""Pattern Explorer data (canon C13): which indicator patterns preceded the moves that get us toward +7% a week.

Uses only hidden years that have already been revealed (their adjustments are locked). Each year is rebuilt
through its own sealed feed, so the rows are exactly what the blind system saw on each decision day.
Writes docs/patterns.json for docs/patterns.html."""
import glob, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree
from engine import config as K, livesim, policy
from engine.live import plain

cycles = json.loads((livesim.DIR / "cycles.json").read_text())["cycles"]
revealed = [c for c in cycles if c.get("revealed_year")]
src = open("scripts/livesim_cycle.py", encoding="utf-8").read()
ns = {"__file__": "scripts/livesim_cycle.py"}
_argv = sys.argv; sys.argv = ["x"]
exec(compile(src[:src.index("def worker(")], "lc", "exec"), ns)
sys.argv = _argv


def holdings_path(cfg, snaps, closes, divs):
    """Which names the system held after each decision (same rules as the live trader)."""
    held, out, wk = [], {}, 0
    sessions = closes.index
    dec = {pd.Timestamp(k): v for k, v in snaps.items()}
    for i, d in enumerate(sessions):
        if d in dec and (wk % cfg.get("rebalance_weeks", 1) == 0 or not held):
            p = dec[d]
            s = policy.score(p, cfg["w_model"])
            ok = ~((p["ev_red_flag"] > 0) | ((p["ev_offering"] > 0) & (p["log_dv"].rank(pct=True) < 0.5)))
            if cfg["vol_filter"]:
                ok &= ~((p["vol20"].rank(pct=True) > 0.9) | (p["max20"].rank(pct=True) > 0.9))
            ok &= p["log_dv"].rank(pct=True) >= cfg["liq_q"]
            held = list(policy.topk_targets(s[ok], held, cfg["k"], cfg["exit_q"], divs if cfg["max_per_sector"] else None,
                                            cfg["max_per_sector"], pick=cfg["pick"], vol=p["vol20"], pool_q=cfg["pool_q"]).index)
            out[str(d.date())] = set(held)
        if i + 1 >= len(sessions) or sessions[i + 1].isocalendar().week != d.isocalendar().week:
            wk += 1
    return out


FLAGS = {"ev_offering", "ev_shelf", "ev_agreement", "ev_red_flag", "earn_in_week", "ins_opportunistic30", "news5"}
rows = []
for c in revealed:
    rid, year = c["run_id"], c["revealed_year"]
    feed = livesim.Feed(livesim.SealedYear(rid))
    feed.precompute_features()
    closes = feed._stocks["Close"].loc[feed.first_live:]
    a = livesim.DIR / rid
    snaps = {f.stem[5:]: pd.read_parquet(f) for f in sorted(a.glob("snap_*.parquet"))}
    sc = pd.read_parquet(a / "sic.parquet")
    divs = {t: policy.sic_division(x) for t, x in zip(sc["ticker"], sc["sic"])}
    held = holdings_path(c["config"], snaps, closes, divs)
    weeks_end = [d for i, d in enumerate(closes.index)
                 if i + 1 >= len(closes.index) or closes.index[i + 1].isocalendar().week != d.isocalendar().week]
    for sdate in snaps:
        d = pd.Timestamp(sdate)
        nxt = [w for w in weeks_end if w > d][:1]
        if not nxt or d not in feed._X.index.get_level_values(0):
            continue
        X = feed._X.xs(d, level=0)
        fwd = (closes.loc[nxt[0]] / closes.loc[d] - 1).reindex(X.index)
        X = X.drop(columns=[c_ for c_ in X.columns if c_.startswith("ev_activist")])   # disabled signal
        R = X.rank(pct=True)                                   # cross-sectional rank: comparable across eras
        for col in X.columns:
            if col.startswith("m_") or col in FLAGS:
                R[col] = X[col]                                # market levels and yes/no flags stay as they are
        R["fwd"] = fwd.values
        R["held"] = X.index.isin(held.get(sdate, set()))
        R["year"] = year
        R["era"] = f"{year // 10 * 10}s"
        rows.append(R.dropna(subset=["fwd"]))
    print(f"{rid} ({year}): {sum(len(r) for r in rows if r['year'].iloc[0] == year):,} stock-decisions", flush=True)

D = pd.concat(rows, ignore_index=True)
D["hit7"] = (D["fwd"] >= 0.07).astype(float)
feats = [c for c in D.columns if c not in ("fwd", "held", "year", "era", "hit7")]
base_hit, base_ret = float(D["hit7"].mean()), float(D["fwd"].mean())

# 1) every indicator in ten buckets
deciles = {}
for f in feats:
    x = D[f]
    if x.nunique() < 2:
        continue
    if f in FLAGS or x.nunique() < 3:
        q = (x > 0).astype(int)                                          # 0 = no, 1 = yes
    else:
        q = pd.qcut(x.rank(method="first"), 10, labels=False)
    g = D.groupby(q)
    yrs = D.groupby([q, "year"])["hit7"].mean().unstack()
    yr_base = D.groupby("year")["hit7"].mean()
    consistency = (yrs.gt(yr_base, axis=1)).mean(axis=1)             # share of years the bucket beat that year's base
    deciles[f] = {"name": plain(f), "market_wide": f.startswith("m_"), "flag": bool(f in FLAGS or x.nunique() < 3),
                  "buckets": [{"b": int(b), "hit7": float(g["hit7"].mean()[b]), "ret": float(g["fwd"].mean()[b]),
                               "n": int(g.size()[b]), "years_beating_base": float(consistency.get(b, np.nan))} for b in g.size().index],
                  "spread_hit7": float(g["hit7"].mean().iloc[-1] - g["hit7"].mean().iloc[0]),
                  "spread_ret": float(g["fwd"].mean().iloc[-1] - g["fwd"].mean().iloc[0])}

# 2) combined patterns: a shallow tree, every leaf is a readable rule
Xf = D[feats].fillna(0.5)
tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=max(200, len(D) // 400), random_state=0).fit(Xf, D["hit7"])
t = tree.tree_
leaf = tree.apply(Xf)
rules = []


def walk(node, conds):
    if t.feature[node] == _tree.TREE_UNDEFINED:
        m = leaf == node
        sub = D[m]
        by_year = sub.groupby("year")["hit7"].mean()
        yr_base = D.groupby("year")["hit7"].mean().reindex(by_year.index)
        rules.append({"rule": conds, "n": int(m.sum()), "hit7": float(sub["hit7"].mean()), "ret": float(sub["fwd"].mean()),
                      "lift": float(sub["hit7"].mean() / base_hit) if base_hit else None,
                      "years_seen": int(by_year.size), "years_beating_base": int((by_year > yr_base).sum())})
        return
    f, thr = feats[t.feature[node]], float(t.threshold[node])
    def txt(le):
        if f in FLAGS:
            return f"{plain(f)}: {'no' if le else 'yes'}"
        if f.startswith("m_"):
            return f"{plain(f)} {'≤' if le else '>'} {thr:.3g}"
        pct = int(round(thr * 100))
        return f"{plain(f)} in the {'bottom' if le else 'top'} {pct if le else 100 - pct}%"
    walk(t.children_left[node], conds + [txt(True)])
    walk(t.children_right[node], conds + [txt(False)])


walk(0, [])
rules.sort(key=lambda r: -r["hit7"])

# 3) the system's own positions: winners vs losers, indicator by indicator
H = D[D["held"]]
prof = []
if len(H) > 20:
    win, lose = H[H["fwd"] > 0], H[H["fwd"] <= 0]
    for f in feats:
        if f.startswith("m_"):
            continue
        sd = D[f].std() or 1
        prof.append({"feature": f, "name": plain(f), "winners": float(win[f].mean()), "losers": float(lose[f].mean()),
                     "gap_sd": float((win[f].mean() - lose[f].mean()) / sd)})
    prof.sort(key=lambda r: -abs(r["gap_sd"]))

top = sorted(deciles.items(), key=lambda kv: -abs(kv[1]["spread_hit7"]))
out = {"generated_from": {"hidden_years": sorted(int(y) for y in D["year"].unique()), "stock_decisions": int(len(D)),
                          "held_positions": int(D["held"].sum())},
       "baseline": {"hit7": base_hit, "ret": base_ret},
       "indicators_ranked": [{"feature": f, **v} for f, v in top],
       "rules": rules, "held_winners_vs_losers": prof,
       "note": "Ranks are within each decision day, so 'top 10%' means relative to the other stocks that day. "
               "A +7% week is a single stock rising 7% or more in the week after the decision. "
               "Patterns found here are hypotheses: the learning loop still has to confirm them on unseen hidden years."}
(K.SITE / "patterns.json").write_text(json.dumps(out, default=float))
print(f"patterns: {len(D):,} stock-decisions across {D['year'].nunique()} revealed years; base +7% rate {base_hit:.1%}")
for r in rules[:5]:
    print(f"  {r['hit7']:.1%} ({r['lift']:.1f}x) n={r['n']}: " + " AND ".join(r["rule"]))
