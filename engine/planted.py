"""Bible Phase 25 (mandatory): planted-pattern calibration. Build synthetic markets whose true patterns are KNOWN,
run the full discovery pipeline on them, and measure whether it can tell planted signal from noise.

Plant kinds (the Bible's list, plus the lifecycle cases the miner claims to handle):
  strong / weak / negative  single conditions with a fixed effect everywhere
  zero                       a condition that is named in the scenario but has no effect (must not be admitted)
  pair                       effect only where A AND B both hold (neither alone carries the full effect)
  unless                     A AND B, except when C also holds (the exception cancels it)
  regime                     effect only when a market-context value is in a range (must be scoped, not averaged)
  hallucinated               effect only in the first half of history - real in discovery, gone by confirmation
                             (a data-snooped pattern: the pipeline must not admit it)
  decaying                   effect for most of history, dead in the recent stretch (must be benched or rescoped,
                             never held as-is - canon C43)

The generator is deliberately harder than white noise: week-common shocks, sector shocks, heavy tails, persistent
(autocorrelated) features and correlated feature pairs, so that clustering and redundancy logic is exercised.
Quintiles are computed exactly as the miner computes them, so a planted "q4" is the miner's q4.
Deterministic given the seed."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

ACTIVE = ("active", "rescoped")


@dataclass(frozen=True)
class Plant:
    name: str
    kind: str                                   # strong|weak|negative|zero|pair|unless|regime|hallucinated|decaying
    conds: tuple                                # ((feature, quintile), ...)
    effect: float = 0.0
    unless: tuple | None = None                 # (feature, quintile) that cancels the effect
    regime: tuple | None = None                 # (context column, lo, hi) on the raw context value
    active: tuple = (0.0, 1.0)                  # fraction of the timeline where the effect exists

    @property
    def should_admit(self) -> bool:
        """Truth: should a correct pipeline hold this pattern for use NOW (at the end of the sample)?"""
        return self.kind in ("strong", "weak", "negative", "pair", "unless", "regime")

    @property
    def key(self):
        return canon_key(self.conds, self.unless)


@dataclass
class Scenario:
    name: str
    plants: list
    weeks: int = 260
    stocks: int = 220
    n_feat: int = 12
    tail_df: float = 4.0                        # Student-t degrees of freedom for idiosyncratic noise
    noise_sd: float = 0.05
    week_sd: float = 0.02
    sector_sd: float = 0.01
    n_sectors: int = 8
    persistence: float = 0.7                    # AR(1) of each stock's feature values week to week
    corr_pairs: tuple = ((10, 11),)             # feature pairs made strongly correlated (redundancy test)
    notes: str = ""

    def spec(self):
        d = asdict(self)
        d["plants"] = [asdict(p) for p in self.plants]
        return d


def canon_key(conds, unless=None):
    """Order-independent identity of a condition set: ('f1 q4', 'f0 q4') == ('f0 q4', 'f1 q4')."""
    c = tuple(sorted(f"{f} q{q}" for f, q in conds))
    return (c, None if unless is None else f"{unless[0]} q{unless[1]}")


def parse_named(name: str):
    """Miner key_named -> canon key. Formats: 'f0 q4', 'f0 q4 & f1 q2', 'f0 q4 & f1 q2 unless f3 q0'."""
    body, _, unl = name.partition(" unless ")
    conds = []
    for part in body.split(" & "):
        f, q = part.rsplit(" q", 1)
        conds.append((f.strip(), int(q)))
    u = None
    if unl:
        f, q = unl.rsplit(" q", 1)
        u = (f.strip(), int(q))
    return canon_key(conds, u)


def quintiles(frame: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional quintile per date, identical to PatternMiner._quintiles."""
    r = frame.groupby(level=0).rank(pct=True).fillna(0.5)
    return np.minimum((r * 5).astype(np.int8), 4)


# ---------------------------------------------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------------------------------------------
def generate(sc: Scenario, seed: int = 0):
    """-> (X, y, truth) where X is (date, ticker) x [f0..f{n-1}, m_vix, m_vix_term, m_breadth] and y is the forward
    excess return (demeaned per week). truth is a DataFrame of per-row planted contributions (one column per plant)."""
    rng = np.random.default_rng(seed)
    W, S, F = sc.weeks, sc.stocks, sc.n_feat
    dates = pd.bdate_range("2008-01-01", periods=W * 5)[::5]
    tick = [f"T{i:03d}" for i in range(S)]
    idx = pd.MultiIndex.from_product([dates, tick], names=["date", "ticker"])
    # persistent features: AR(1) per stock through time, unit variance
    Z = np.empty((W, S, F))
    Z[0] = rng.standard_normal((S, F))
    a = sc.persistence
    for t in range(1, W):
        Z[t] = a * Z[t - 1] + np.sqrt(1 - a * a) * rng.standard_normal((S, F))
    for i, j in sc.corr_pairs:                                  # near-duplicate feature (correlation ~0.95)
        if i < F and j < F:
            Z[:, :, j] = 0.95 * Z[:, :, i] + np.sqrt(1 - 0.95 ** 2) * Z[:, :, j]
    X = pd.DataFrame(Z.reshape(W * S, F), index=idx, columns=[f"f{i}" for i in range(F)])
    # market context: a persistent weekly regime series plus two distractors
    reg = np.empty(W); reg[0] = rng.standard_normal()
    for t in range(1, W):
        reg[t] = 0.9 * reg[t - 1] + np.sqrt(1 - 0.81) * rng.standard_normal()
    X["m_vix"] = np.repeat(reg, S)
    X["m_vix_term"] = np.repeat(rng.standard_normal(W), S)
    X["m_breadth"] = np.repeat(rng.standard_normal(W), S)
    # returns: heavy-tailed idiosyncratic + week shock + sector shock
    t_noise = rng.standard_t(sc.tail_df, W * S) * sc.noise_sd / np.sqrt(sc.tail_df / (sc.tail_df - 2))
    sector = rng.integers(0, sc.n_sectors, S)
    sec_shock = rng.normal(0, sc.sector_sd, (W, sc.n_sectors))[:, sector].reshape(-1)
    y = pd.Series(t_noise + np.repeat(rng.normal(0, sc.week_sd, W), S) + sec_shock, index=idx)
    Q = quintiles(X[[f"f{i}" for i in range(F)]])
    tpos = np.repeat(np.arange(W) / max(W - 1, 1), S)
    truth = pd.DataFrame(index=idx)
    for p in sc.plants:
        m = np.ones(len(idx), dtype=bool)
        for f, q in p.conds:
            m &= (Q[f].values == q)
        if p.unless is not None:
            m &= ~(Q[p.unless[0]].values == p.unless[1])
        if p.regime is not None:
            col, lo, hi = p.regime
            m &= (X[col].values > lo) & (X[col].values <= hi)
        m &= (tpos >= p.active[0]) & (tpos < p.active[1] + 1e-12)
        contrib = p.effect * m
        truth[p.name] = contrib
        y = y + contrib
    y = y - y.groupby(level=0).transform("mean")
    return X, y, truth


# ---------------------------------------------------------------------------------------------------------------
# standard scenarios
# ---------------------------------------------------------------------------------------------------------------
def standard_plants():
    return [
        Plant("strong", "strong", (("f0", 4),), 0.012),
        Plant("weak", "weak", (("f1", 4),), 0.004),
        Plant("negative", "negative", (("f2", 4),), -0.008),
        Plant("regime", "regime", (("f3", 4),), 0.010, regime=("m_vix", 0.43, 99.0)),
        Plant("pair", "pair", (("f4", 4), ("f5", 0)), 0.020),
        Plant("unless", "unless", (("f6", 4), ("f7", 4)), 0.020, unless=("f8", 4)),
        Plant("hallucinated", "hallucinated", (("f9", 0),), 0.012, active=(0.0, 0.5)),
        Plant("decaying", "decaying", (("f1", 0),), -0.010, active=(0.0, 0.75)),
        Plant("zero", "zero", (("f10", 4),), 0.0),
    ]


def scenarios():
    """The calibration suite: the full mix, pure noise, heavy tails, and a power curve for the effect size."""
    out = [Scenario("standard", standard_plants()),
           Scenario("noise_only", [Plant("zero", "zero", (("f10", 4),), 0.0)]),
           Scenario("heavy_tails", standard_plants(), tail_df=2.5, notes="fatter tails: weekly outliers")]
    for e in (0.002, 0.004, 0.006, 0.009, 0.012):
        out.append(Scenario(f"power_{e:.3f}", [Plant("strong", "strong", (("f0", 4),), e)], notes="power curve"))
    return out


# ---------------------------------------------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------------------------------------------
def _overlaps(key, plant_keys):
    conds = set(key[0])
    return any(conds & set(pk[0]) for pk in plant_keys)


TRUE_MIN = 0.001        # a pattern whose exact true recent effect is under 0.1%/week has no real effect


def true_recent_effects(keys, X, truth, recent_frac=0.2):
    """EXACT ground truth per pattern: the mean planted contribution over the pattern's rows in the most recent
    `recent_frac` of history (where an admitted pattern is used), after the same per-week demeaning as y. This counts
    the mechanical side effects of demeaning (if the top fifth of f0 gains +1.2%, the rest of f0 truly loses a
    little) as the real effects they are - the condition-overlap rule called them false discoveries."""
    feats = [c for c in X.columns if not c.startswith("m_")]
    Q = quintiles(X[feats])
    tr = truth.sum(axis=1) if truth.shape[1] else pd.Series(0.0, index=X.index)
    tr = tr - tr.groupby(level=0).transform("mean")
    d = X.index.get_level_values(0)
    ud = np.sort(d.unique())
    recent = np.asarray(d >= ud[int(len(ud) * (1 - recent_frac))])
    out = []
    for conds, unless in keys:
        m = recent.copy()
        for c in conds:
            f, q = c.rsplit(" q", 1)
            if f not in Q:
                m[:] = False; break
            m &= (Q[f].values == int(q))
        if unless is not None:
            f, q = unless.rsplit(" q", 1)
            if f in Q:
                m &= ~(Q[f].values == int(q))
        out.append(float(tr.values[m].mean()) if m.any() else 0.0)
    return np.array(out)


def score_run(P: pd.DataFrame, sc: Scenario, X=None, truth=None) -> dict:
    """P: PatternMiner.patterns after fit. Returns the Phase 25 measures for one run. With X and the generator's truth
    frame, false discoveries are also judged by EXACT ground truth (true_recent_effects); both are reported."""
    P = P.copy()
    P["ck"] = [parse_named(n) for n in P["key_named"]]
    act = P[P["status"].isin(ACTIVE)]
    real_keys = [p.key for p in sc.plants if p.should_admit]
    bad_keys = [p.key for p in sc.plants if not p.should_admit]
    per = {}
    for p in sc.plants:
        hit = P[P["ck"] == p.key]
        st = hit["status"].iloc[0] if len(hit) else "not_tested"
        est = float(hit["effect"].iloc[0]) if len(hit) else np.nan
        children = act[[set(p.key[0]) <= set(k[0]) and k != p.key for k in act["ck"]]]
        scope = hit["scope"].iloc[0] if len(hit) and "scope" in hit else None
        per[p.name] = {
            "kind": p.kind, "should_admit": p.should_admit, "status": st,
            "admitted": st in ACTIVE, "admitted_via_child": bool(len(children)) and st not in ACTIVE,
            "true_effect": p.effect, "est_effect": est,
            "effect_ratio": float(est / p.effect) if p.effect and np.isfinite(est) else np.nan,
            "sign_ok": bool(np.sign(est) == np.sign(p.effect)) if p.effect and np.isfinite(est) else None,
            "p_real": float(hit["p_real"].iloc[0]) if len(hit) else np.nan,
            "scope": None if scope is None or (isinstance(scope, float) and np.isnan(scope)) else str(scope),
        }
    # a false discovery: an admitted pattern sharing no condition with any pattern that should be admitted
    false_act = [n for n, k in zip(act["key_named"], act["ck"]) if not _overlaps(k, real_keys)]
    snooped = [n for n, k in zip(act["key_named"], act["ck"]) if _overlaps(k, bad_keys) and not _overlaps(k, real_keys)]
    truth_overlap = np.array([_overlaps(k, real_keys) for k in P["ck"]])
    out = {"scenario": sc.name, "per_plant": per, "n_tested": int(len(P)), "n_active": int(len(act)),
           "false_active": false_act, "n_false_active": len(false_act), "planted_bad_admitted": snooped,
           "calib": list(zip(P["p_real"].astype(float).round(4).tolist(), truth_overlap.tolist()))}
    if X is not None and truth is not None:
        te = true_recent_effects(list(P["ck"]), X, truth)
        real = (np.abs(te) >= TRUE_MIN) & (np.sign(te) == np.sign(P["effect"].astype(float).values))
        is_act = P["status"].isin(ACTIVE).values
        out["false_active_true"] = [n for n, r, a in zip(P["key_named"], real, is_act) if a and not r]
        out["n_false_active_true"] = len(out["false_active_true"])
        out["calib_true"] = list(zip(P["p_real"].astype(float).round(4).tolist(), real.tolist()))
    return out


def calibration_table(pairs, bins=(-0.01, 0.2, 0.5, 0.8, 0.9, 1.0)):
    """P(real) reliability: for each bin, share of candidates that truly involve a real planted condition.
    Also Brier score and expected calibration error (ECE)."""
    if not pairs:
        return {"bins": {}, "brier": None, "ece": None, "n": 0}
    C = pd.DataFrame(pairs, columns=["p", "truth"]).dropna()
    C["truth"] = C["truth"].astype(float)
    C["bin"] = pd.cut(C["p"], list(bins))
    g = C.groupby("bin", observed=True).agg(mean_p=("p", "mean"), share=("truth", "mean"), n=("truth", "size"))
    ece = float((g["n"] * (g["mean_p"] - g["share"]).abs()).sum() / max(g["n"].sum(), 1))
    return {"bins": {str(b): {"mean_p": float(r.mean_p), "share_truly_real": float(r.share), "n": int(r.n)}
                     for b, r in g.iterrows()},
            "brier": float(((C["p"] - C["truth"]) ** 2).mean()), "ece": ece, "n": int(len(C))}


def summarise(runs: list) -> dict:
    """Aggregate many scored runs (any mix of scenarios) into the Phase 25 measures."""
    out = {"n_runs": len(runs), "by_scenario": {}}
    for scn in sorted({r["scenario"] for r in runs}):
        R = [r for r in runs if r["scenario"] == scn]
        kinds = {}
        for r in R:
            for name, d in r["per_plant"].items():
                k = kinds.setdefault(name, {"kind": d["kind"], "should_admit": d["should_admit"], "admitted": [],
                                            "via_child": [], "ratio": [], "sign": [], "status": {}})
                k["admitted"].append(d["admitted"]); k["via_child"].append(d["admitted_via_child"])
                if np.isfinite(d["effect_ratio"]):
                    k["ratio"].append(d["effect_ratio"])
                if d["sign_ok"] is not None:
                    k["sign"].append(d["sign_ok"])
                k["status"][d["status"]] = k["status"].get(d["status"], 0) + 1
        plants = {}
        for name, k in kinds.items():
            rate = float(np.mean(k["admitted"]))
            plants[name] = {"kind": k["kind"], "should_admit": k["should_admit"],
                            ("detection_rate" if k["should_admit"] else "false_admission_rate"): rate,
                            "admitted_via_child_rate": float(np.mean(k["via_child"])),
                            "median_effect_ratio": float(np.median(k["ratio"])) if k["ratio"] else None,
                            "sign_accuracy": float(np.mean(k["sign"])) if k["sign"] else None,
                            "status_counts": k["status"]}
        n_act = sum(r["n_active"] for r in R)
        n_false = sum(r["n_false_active"] for r in R)
        out["by_scenario"][scn] = {
            "runs": len(R), "plants": plants,
            "active_per_run": n_act / len(R), "false_active_per_run": n_false / len(R),
            "false_discovery_rate": n_false / n_act if n_act else 0.0,
            "rejection_rate_of_bad_plants": (float(np.mean([not d["admitted"] for r in R for d in r["per_plant"].values()
                                                            if not d["should_admit"]]))
                                             if any(not d["should_admit"] for r in R for d in r["per_plant"].values())
                                             else None),
            "calibration": calibration_table([c for r in R for c in r["calib"]]),
        }
        if all("n_false_active_true" in r for r in R):
            n_ft = sum(r["n_false_active_true"] for r in R)
            out["by_scenario"][scn].update({
                "false_active_per_run_true": n_ft / len(R),
                "false_discovery_rate_true": n_ft / n_act if n_act else 0.0,
                "calibration_true": calibration_table([c for r in R for c in r["calib_true"]])})
    return out


CRITERIA = {
    "strong_detection": ("standard", "strong", "detection_rate", ">=", 0.9),
    "negative_detection": ("standard", "negative", "detection_rate", ">=", 0.8),
    "negative_sign": ("standard", "negative", "sign_accuracy", ">=", 0.95),
    "hallucinated_rejected": ("standard", "hallucinated", "false_admission_rate", "<=", 0.2),
    "decaying_not_held": ("standard", "decaying", "false_admission_rate", "<=", 0.2),
    "zero_rejected": ("standard", "zero", "false_admission_rate", "<=", 0.1),
}


def verdict(summary: dict) -> dict:
    """The Bible's rule: if the system cannot distinguish planted signal from noise, the pattern system is NOT
    validated. Every criterion is reported; missing evidence counts as a failure (fail closed)."""
    res = {}
    for cname, (scn, plant, metric, op, thr) in CRITERIA.items():
        v = summary["by_scenario"].get(scn, {}).get("plants", {}).get(plant, {}).get(metric)
        ok = v is not None and (v >= thr if op == ">=" else v <= thr)
        res[cname] = {"value": v, "rule": f"{metric} {op} {thr}", "pass": bool(ok)}
    noise = summary["by_scenario"].get("noise_only")
    v = noise["active_per_run"] if noise else None
    res["noise_only_quiet"] = {"value": v, "rule": "active_per_run <= 1.0", "pass": v is not None and v <= 1.0}
    std = summary["by_scenario"].get("standard")
    # exact ground truth (true recent effect) when the runs carry it; the condition-overlap figure is kept as
    # "fdr_by_condition" for comparability with earlier reports but no longer decides
    exact = bool(std) and "false_discovery_rate_true" in std
    v = (std["false_discovery_rate_true"] if exact else std["false_discovery_rate"]) if std else None
    res["fdr"] = {"value": v, "rule": "false_discovery_rate <= 0.10" + (" (exact truth)" if exact else " (condition overlap)"),
                  "pass": v is not None and v <= 0.10}
    if exact:
        res["fdr_by_condition"] = {"value": std["false_discovery_rate"], "rule": "reported only (legacy definition)",
                                   "pass": True}
    cal = (std or {}).get("calibration_true" if exact else "calibration", {})
    top = cal.get("bins", {}).get("(0.9, 1.0]", {}).get("share_truly_real")
    res["p_real_top_bin"] = {"value": top, "rule": "P(real) in (0.9,1] truly real >= 0.85" + (" (exact truth)" if exact else ""),
                             "pass": top is not None and top >= 0.85}
    return {"validated": all(r["pass"] for r in res.values()), "criteria": res}
