"""Why did this memory count? Diagnostics for the factor-weighted memory (Bible Phase 9, canon C34).

engine.memory answers "how good is arm X right now?". This module answers the questions an auditor asks next:

  explain / episode_table      which episodes carried the estimate, and how much of each weight was recency,
                               market similarity, source, shock, era or modernity
  factor_attribution           leave-one-factor-out: how far each factor moved the estimate (a factor that never moves
                               anything is dead weight; one that moves everything is a hidden assumption)
  concentration                effective sample size, top-share, Gini of the weights: is one week doing all the talking
  memory_health                size per arm, staleness, breaks, era mix, error-type mix, capacity use, evictions
  walk_forward_skill           does the memory PREDICT? each outcome is predicted from earlier ones only, then scored
  factor_ablation / sweep      skill with each factor removed / across a parameter grid: which factor earns its keep
  eviction_regret              what a bounded memory loses against an unbounded one on the same stream
  detect_lag                   how many weeks after a planted break the CUSUM noticed
Everything is read-only against the Memory it is handed (it works on copies of the weights), deterministic, and takes
no clock: the caller passes week_now and the market context."""
import numpy as np
import pandas as pd

from .memory import CTX, Memory, scan_lessons

FACTORS = ("recency", "similarity", "source", "shock", "era", "modernity")
# what to set so a factor becomes a no-op (used by ablation)
_OFF = {"recency": {"mem_half_life": 1e12}, "similarity": {"mem_bandwidth": 1e12}, "source": {"mem_prior_scale": 1.0},
        "shock": {"mem_shock_cut": 1.0}, "era": {"mem_era_other": 1.0}, "modernity": {"mem_modern_half_life": None},
        "shrinkage": {"mem_shrink": 0.0}}


def _shrunk(w, x, shrink, min_var=1e-8):
    """The estimate() arithmetic on plain arrays: (shrunk mean, se, n_eff)."""
    sw = w.sum()
    if not len(w) or sw <= 0:
        return 0.0, np.inf, 0.0
    n_eff = sw ** 2 / (w ** 2).sum()
    mean = float((w * x).sum() / (sw + shrink * w.mean()))
    var = float((w * (x - (w * x).sum() / sw) ** 2).sum() / sw)
    return mean, float(np.sqrt(max(var, min_var) / max(n_eff, 1.0))), float(n_eff)


def episode_table(mem, arm, week_now, ctx_now):
    """One row per stored episode of `arm`: its factors, final weight, share of the total weight and error type."""
    rows = []
    for i in mem._by_arm.get(arm, []):
        e, info = mem.ep[i], mem.info[i]
        f = mem.factors(e, week_now, ctx_now, info)
        w = f["recency"] * f["similarity"] * f["shock"] * f["era"] * f["modernity"] * (f["source"] if e[4] else 1.0)
        rows.append({"idx": i, "week": e[1] if not e[4] else np.nan, "origin": "long_term" if e[4] else "window",
                     "era_label": info["era"], "outcome": e[3], "error": info["err"], **{k: f[k] for k in FACTORS}, "weight": w})
    cols = ["idx", "week", "origin", "era_label", "outcome", "error", *FACTORS, "weight", "share", "dominant_cut"]
    if not rows:
        return pd.DataFrame(columns=cols)
    t = pd.DataFrame(rows)
    t["share"] = t["weight"] / t["weight"].sum() if t["weight"].sum() > 0 else 0.0
    # the factor that cut this episode the most (smallest multiplier); 'none' if every factor is ~1
    fac = t[list(FACTORS)]
    t["dominant_cut"] = np.where(fac.min(axis=1) > 0.999, "none", fac.idxmin(axis=1))
    return t[cols]


def concentration(w):
    """How concentrated is a weight vector? n_eff (Kish), top-1 and top-5 shares, Gini and normalised entropy."""
    w = np.asarray(w, dtype=float)
    w = w[np.isfinite(w) & (w > 0)]
    if not len(w):
        return {"n": 0, "n_eff": 0.0, "top1": 0.0, "top5": 0.0, "gini": 0.0, "entropy": 0.0}
    p = np.sort(w / w.sum())[::-1]
    n = len(p)
    asc = p[::-1]
    gini = float((2 * np.arange(1, n + 1) - n - 1) @ asc / (n * asc.sum())) if n > 1 else 0.0
    ent = float(-(p * np.log(p)).sum() / np.log(n)) if n > 1 else 0.0
    return {"n": n, "n_eff": float(1 / (p ** 2).sum()), "top1": float(p[0]), "top5": float(p[:5].sum()), "gini": gini, "entropy": ent}


def factor_attribution(mem, arm, week_now, ctx_now):
    """Estimate with each factor switched off in turn (and with shrinkage off). Returns a DataFrame:
    factor, mean_without, delta (full - without), n_eff_without. delta is what that factor contributed."""
    t = episode_table(mem, arm, week_now, ctx_now)
    if t.empty:
        return pd.DataFrame(columns=["factor", "mean_without", "delta", "n_eff_without"])
    x = t["outcome"].to_numpy()
    full, _, _ = _shrunk(t["weight"].to_numpy(), x, mem.p["mem_shrink"], mem.p["mem_min_var"])
    src = (t["origin"] == "long_term").to_numpy()
    rows = []
    for fct in FACTORS:
        cols = {k: t[k].to_numpy() for k in FACTORS}
        cols[fct] = np.ones(len(t))
        w = cols["recency"] * cols["similarity"] * cols["shock"] * cols["era"] * cols["modernity"] * np.where(src, cols["source"], 1.0)
        m, _, ne = _shrunk(w, x, mem.p["mem_shrink"], mem.p["mem_min_var"])
        rows.append({"factor": fct, "mean_without": m, "delta": full - m, "n_eff_without": ne})
    m, _, ne = _shrunk(t["weight"].to_numpy(), x, 0.0, mem.p["mem_min_var"])
    rows.append({"factor": "shrinkage", "mean_without": m, "delta": full - m, "n_eff_without": ne})
    return pd.DataFrame(rows)


def explain(mem, arm, week_now, ctx_now, top=5):
    """Why an arm's estimate is what it is. Returns {estimate, concentration, top_episodes, attribution, text}."""
    det = mem.estimate_detail(arm, week_now, ctx_now)
    t = episode_table(mem, arm, week_now, ctx_now)
    attr = factor_attribution(mem, arm, week_now, ctx_now)
    conc = concentration(t["weight"].to_numpy()) if len(t) else concentration([])
    tops = t.sort_values("weight", ascending=False).head(top) if len(t) else t
    lines = [f"arm {arm!r}: shrunk mean {det['mean']:+.4f} (raw {det['raw_mean']:+.4f}), se {det['se']:.4f}, "
             f"n_eff {det['n_eff']:.1f} of {det['n']} episodes, reliability {det['reliability']:.2f}"]
    if arm in mem.breaks:
        lines.append(f"  structural break detected at week {mem.breaks[arm]:.0f}: earlier evidence cut to {mem.p['mem_shock_cut']:.0%}")
    if len(attr):
        mv = attr.reindex(attr["delta"].abs().sort_values(ascending=False).index).head(3)
        lines.append("  biggest movers of the estimate: " + ", ".join(f"{r.factor} {r.delta:+.4f}" for r in mv.itertuples()))
    if len(tops):
        lines.append(f"  top {len(tops)} episodes carry {tops['share'].sum():.0%} of the weight")
    return {"estimate": det, "concentration": conc, "top_episodes": tops, "attribution": attr, "table": t, "text": "\n".join(lines)}


def memory_health(mem, week_now=None):
    """Whole-memory census. Returns {'summary': dict, 'arms': DataFrame}."""
    wk_own = [e[1] for e in mem.ep if not e[4]]
    now = max(wk_own) if week_now is None and wk_own else (0.0 if week_now is None else week_now)
    ctx0 = np.zeros(len(CTX))
    rows = []
    for arm in mem.arms():
        idx = mem._by_arm[arm]
        own = [mem.ep[i] for i in idx if not mem.ep[i][4]]
        d = mem.estimate_detail(arm, now, ctx0)
        rows.append({"arm": repr(arm), "own": len(own), "long_term": len(idx) - len(own),
                     "last_week": max((e[1] for e in own), default=np.nan),
                     "stale_weeks": now - max((e[1] for e in own), default=np.nan) if own else np.nan,
                     "mean_outcome": float(np.mean([mem.ep[i][3] for i in idx])), "n_eff": d["n_eff"],
                     "reliability": d["reliability"], "break_week": mem.breaks.get(arm, np.nan)})
    arms = pd.DataFrame(rows, columns=["arm", "own", "long_term", "last_week", "stale_weeks", "mean_outcome", "n_eff",
                                       "reliability", "break_week"])
    eras = pd.Series([i["era"] or "unknown" for i in mem.info], dtype=object).value_counts().to_dict()
    errs = pd.Series([i["err"] or "unscored" for i in mem.info], dtype=object).value_counts().to_dict()
    cap = mem.p["mem_capacity"]
    summ = {"episodes": len(mem.ep), "arms": len(mem._by_arm), "window": len(wk_own), "long_term": len(mem.ep) - len(wk_own),
            "breaks": len(mem.break_log), "capacity": cap, "utilisation": (len(mem.ep) / cap) if cap else None,
            "evicted": mem.evicted, "rejected": mem.rejected, "eras": eras, "errors": errs, "week_now": now}
    return {"summary": summ, "arms": arms}


def shock_report(mem):
    """Every detected break: arm, week, direction, plus how many episodes it has since discounted."""
    rows = []
    for arm, wk, direction in mem.break_log:
        n_cut = sum(1 for i in mem._by_arm.get(arm, []) if mem.ep[i][4] or mem.ep[i][1] < wk)
        rows.append({"arm": repr(arm), "week": wk, "direction": direction, "episodes_cut": n_cut})
    return pd.DataFrame(rows, columns=["arm", "week", "direction", "episodes_cut"])


def capacity_report(mem):
    """What the evictions removed: count by reason, by source, and the age spread of what was lost."""
    if not mem.eviction_log:
        return {"evicted": mem.evicted, "by_reason": {}, "by_source": {}, "median_age_weeks": None}
    df = pd.DataFrame(mem.eviction_log, columns=["arm", "week", "src", "why"])
    latest = max((e[1] for e in mem.ep if not e[4]), default=0.0)
    own = df[df["src"] == 0]
    return {"evicted": mem.evicted, "by_reason": df["why"].value_counts().to_dict(),
            "by_source": df["src"].map({0: "window", 1: "long_term"}).value_counts().to_dict(),
            "median_age_weeks": float(latest - own["week"].median()) if len(own) else None}


def leak_check(mem, tickers=(), window_ids=(), exact_outcomes=()):
    """Phase 9.2 in one call: export the sanitised bank and scan it. Returns the findings (empty = clean)."""
    return scan_lessons(mem.export_lessons(tickers=tickers), tickers, window_ids, exact_outcomes)


# ---------------------------------------------------------------------------------------------------------------
# does the memory predict? walk-forward skill
# ---------------------------------------------------------------------------------------------------------------
def walk_forward_skill(records, params=None, long_term=None, warmup=5):
    """records: [(arm, week, ctx, outcome), ...] in time order. Before each outcome is recorded the memory is asked
    for its estimate of that arm at that week and context; the pair (prediction, outcome) is kept. Nothing after the
    outcome is used. Returns (per-step DataFrame, summary dict). skill = 1 - MSE(memory) / MSE(predict zero): above 0
    means the memory beats knowing nothing; hit_rate is the share of non-zero predictions with the right sign."""
    M = Memory(params, long_term=long_term)
    rows = []
    seen = {}
    for arm, wk, ctx, y in records:
        n = seen.get(arm, 0)
        mean, se, n_eff = M.estimate(arm, wk, ctx)
        if n >= warmup:
            rows.append({"week": wk, "arm": repr(arm), "pred": mean, "se": se if np.isfinite(se) else np.nan, "outcome": y})
        M.record(arm, wk, ctx, y)
        seen[arm] = n + 1
    df = pd.DataFrame(rows, columns=["week", "arm", "pred", "se", "outcome"])
    if df.empty:
        return df, {"n": 0, "skill": 0.0, "mse": np.nan, "mse_zero": np.nan, "hit_rate": np.nan, "ic": 0.0}
    mse, mse0 = float(((df["pred"] - df["outcome"]) ** 2).mean()), float((df["outcome"] ** 2).mean())
    nz = df[df["pred"].abs() > 1e-12]
    ic = float(df["pred"].rank().corr(df["outcome"].rank())) if df["pred"].nunique() > 1 else 0.0
    return df, {"n": len(df), "mse": mse, "mse_zero": mse0, "skill": 1 - mse / mse0 if mse0 > 0 else 0.0,
                "hit_rate": float((np.sign(nz["pred"]) == np.sign(nz["outcome"])).mean()) if len(nz) else np.nan,
                "ic": ic if ic == ic else 0.0}


def factor_ablation(records, params=None, warmup=5):
    """Walk-forward skill with each factor (and shrinkage) switched off, one at a time. A factor helps if removing it
    lowers skill. Returns a DataFrame sorted by how much skill the factor contributes (full - without)."""
    base = walk_forward_skill(records, params, warmup=warmup)[1]
    rows = [{"removed": "none", "skill": base["skill"], "ic": base["ic"], "hit_rate": base["hit_rate"], "contribution": 0.0}]
    for name, off in _OFF.items():
        s = walk_forward_skill(records, {**(params or {}), **off}, warmup=warmup)[1]
        rows.append({"removed": name, "skill": s["skill"], "ic": s["ic"], "hit_rate": s["hit_rate"],
                     "contribution": base["skill"] - s["skill"]})
    return pd.DataFrame(rows).sort_values("contribution", ascending=False).reset_index(drop=True)


def sweep(records, param, values, params=None, warmup=5):
    """Walk-forward skill across a grid of one memory parameter (half-life, bandwidth, shrink ...)."""
    rows = []
    for v in values:
        s = walk_forward_skill(records, {**(params or {}), param: v}, warmup=warmup)[1]
        rows.append({param: v, "skill": s["skill"], "ic": s["ic"], "hit_rate": s["hit_rate"], "n": s["n"]})
    return pd.DataFrame(rows)


def eviction_regret(records, capacity, params=None, warmup=5):
    """Skill lost by capping the memory at `capacity` episodes versus keeping everything, on the same stream.
    Negative regret (capped is better) is possible and informative: forgetting can help after a regime change."""
    free = walk_forward_skill(records, {**(params or {}), "mem_capacity": None}, warmup=warmup)[1]
    capped = walk_forward_skill(records, {**(params or {}), "mem_capacity": capacity}, warmup=warmup)[1]
    return {"skill_unbounded": free["skill"], "skill_capped": capped["skill"], "regret": free["skill"] - capped["skill"],
            "capacity": capacity}


def detect_lag(records, arm, true_break_week, params=None):
    """Weeks between a planted break and the CUSUM flagging it (None if it never did, negative = false early alarm)."""
    M = Memory(params)
    for a, wk, ctx, y in records:
        M.record(a, wk, ctx, y)
        if arm in M.breaks and M.breaks[arm] >= true_break_week - 1e-9:
            return float(M.breaks[arm] - true_break_week)
    return None


def report(mem, week_now=None, ctx_now=None, arms=None, top=3):
    """Plain-text report: census, then an explanation for each requested arm (default: the three most-evidenced)."""
    h = memory_health(mem, week_now)
    s = h["summary"]
    now = s["week_now"]
    ctx = np.zeros(len(CTX)) if ctx_now is None else ctx_now
    lines = [f"memory: {s['episodes']} episodes ({s['window']} this window, {s['long_term']} long-term) over {s['arms']} arms; "
             f"{s['breaks']} breaks, {s['evicted']} evicted, {s['rejected']} rejected",
             f"errors: {s['errors']}", f"eras: {s['eras']}"]
    if arms is None:
        names = list(h["arms"].sort_values("n_eff", ascending=False)["arm"].head(top))
        arms = [next(x for x in mem.arms() if repr(x) == a) for a in names]
    for a in arms:
        lines.append(explain(mem, a, now, ctx)["text"])
    return "\n".join(lines)
