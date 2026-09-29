"""Bible Phase 10 - lesson memory / learning from mistakes.

Serves the "must NOT memorise historical answers" rule: a lesson is a rule over ABSTRACT feature conditions
("when m_vix is above X and f3 is below Y, this kind of decision loses"), never over a ticker or a date.

Pipeline
  post_mortem()   turns resolved decisions and untaken candidates into Episodes: situation, features, context,
                  decision, outcome, error category, confidence, counterfactual. Only outcomes resolved by `now`
                  are used (no look-ahead). Identity (ticker, week) survives ONLY as salted hashes used to count
                  DISTINCT episodes; nothing keyed on them is ever matched against.
  LessonBook.learn()  generalises episodes to lessons only with support across many distinct weeks AND tickers,
                  a Benjamini-Hochberg-controlled effect over every condition tried, and sign-agreement on a later
                  validation slice of the episodes. Redundant lessons are dropped.
  trust / expiry  each lesson carries a Beta trust updated from live feedback, decays, and expires after `ttl`
                  ticks of the book's own clock (a tick count, not a calendar date, so nothing is date-keyed).
  audit_identity() proves no ticker string or date string is stored in the serialised book.

Feature columns that are near-deterministic per ticker (price level, static codes) or track the calendar are
excluded before mining (identity_proxy_columns): they would let a "lesson" recall a name or a date."""
import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

CATEGORIES = ("losing_decision", "false_positive", "false_negative", "missed_winner", "pattern_failure",
              "regime_failure", "ok")
FORBIDDEN_KEYS = {"ticker", "symbol", "date", "asof", "as_of", "day", "permno", "cusip", "isin", "name", "id"}
DEFAULTS = {
    "min_support": 40, "min_weeks": 6, "min_tickers": 8, "q_levels": (0.2, 0.4, 0.6, 0.8), "fdr": 0.05,
    "min_abs_delta": 0.002, "max_cover": 0.5, "stab_blocks": 4, "stab_min": 0.75, "reval_min_n": 20, "reval_z": 1.0,
    "reval_retire_z": 1.64, "val_frac": 0.35, "val_min_n": 10, "val_t_floor": 0.5, "down": 0.5, "up": 1.5,
    "ttl": 400, "trust_decay": 0.98, "trust_floor": 0.35, "min_uses": 10, "max_lessons": 25,
    "overlap_max": 0.8, "pair_gain": 1.2, "top_single_for_pairs": 8, "prior_n": 10.0,
    "loss_thr": 0.01, "untaken_frac": 0.3, "cost": 0.0005, "regime_z": 2.0, "factor_clip": (0.25, 2.0),
    "proxy_within_ticker": 0.02, "proxy_date_corr": 0.95,
}
_N = NormalDist()


class IdentityLeak(ValueError):
    """Raised when an episode or lesson would carry a ticker/date as its predictive identity."""


def _hash(*parts, salt=""):
    return hashlib.sha256((salt + "|".join(map(str, parts))).encode()).hexdigest()[:12]


def _p_two_sided(t):
    return 2.0 * (1.0 - _N.cdf(abs(t))) if np.isfinite(t) else 1.0


# ------------------------------------------------------------------ identity guards
def identity_proxy_columns(X, within_ticker=0.02, date_corr=0.95):
    """Columns that could stand in for a name or a date: almost constant within a ticker (price level, static codes),
    or a monotone function of the calendar. `m_` context columns are date-determined by design, so only the second
    test (calendar trend) applies to them. Returns {column: reason}."""
    bad = {}
    if not isinstance(X.index, pd.MultiIndex):
        return bad
    dates = X.index.get_level_values(0)
    tk = X.index.get_level_values(1)
    for c in X.columns:
        x = pd.to_numeric(X[c], errors="coerce")
        ok = x.notna().to_numpy()
        if ok.sum() < 20 or x[ok].nunique() < 2:
            continue
        xs = x[ok]
        if not c.startswith("m_"):
            tot = float(xs.var())
            grp = pd.Series(tk[ok])
            within = float((xs.to_numpy() - xs.groupby(grp.to_numpy()).transform("mean").to_numpy()).var(ddof=1))
            if tot > 0 and (grp.value_counts() >= 3).mean() > 0.5 and within / tot < within_ticker:
                bad[c] = "ticker-constant"
                continue
        dm = xs.groupby(dates[ok]).mean()
        if len(dm) >= 10:
            r = pd.Series(dm.to_numpy()).rank().corr(pd.Series(np.arange(len(dm), dtype=float)))
            if np.isfinite(r) and abs(r) > date_corr:
                bad[c] = "calendar-trend"
    return bad


def usable_features(X, params=None):
    p = {**DEFAULTS, **(params or {})}
    bad = identity_proxy_columns(X, p["proxy_within_ticker"], p["proxy_date_corr"])
    cols = [c for c in X.columns if c not in bad and c.lower() not in FORBIDDEN_KEYS
            and pd.api.types.is_numeric_dtype(X[c])]
    return cols, bad


def _check_no_identity(d, where):
    for k, v in d.items():
        if str(k).lower() in FORBIDDEN_KEYS:
            raise IdentityLeak(f"{where}: key {k!r} is an identity field")
        if isinstance(v, (str, pd.Timestamp, np.datetime64)):
            raise IdentityLeak(f"{where}: value of {k!r} is not numeric ({type(v).__name__})")


# ------------------------------------------------------------------ episodes
@dataclass
class Episode:
    eid: str
    situation: str
    features: dict
    context: dict
    decision: dict
    outcome: dict
    category: str
    confidence: float
    counterfactual: dict
    origin: dict = field(default_factory=dict)       # hashed week/ticker, ONLY for counting distinct episodes
    seq: int = 0                                     # resolution order within the batch (not a date)

    def __post_init__(self):
        if self.category not in CATEGORIES:
            raise ValueError(f"unknown category {self.category!r}")
        for name in ("features", "context", "decision", "outcome"):
            _check_no_identity(getattr(self, name), name)

    @property
    def pnl(self):
        return self.outcome["pnl"]


def _counterfactual(taken, side, y, pnl, cost, mae=None, stop=None):
    """What each alternative would have earned; the best alternative and the regret of what was done."""
    if taken:
        cf = {"no_trade": 0.0, "flip": -side * y - cost}
        if mae is not None and stop is not None and np.isfinite(mae) and np.isfinite(stop):
            cf["tight_stop"] = (-stop - cost) if mae <= -stop else pnl
    else:
        cf = {"take": side * y - cost}
    best = max(cf, key=cf.get)
    actual = pnl if taken else 0.0
    return {**cf, "best": best, "regret": float(max(0.0, cf[best] - actual))}


def _situation(ctx, ref_mu, ref_sd, side, category):
    z = {k: (v - ref_mu.get(k, 0.0)) / ref_sd.get(k, 1.0) for k, v in ctx.items()}
    top = sorted(z, key=lambda k: -abs(z[k]))[:2]
    bits = [f"{k[2:]}:{'hi' if z[k] > 0 else 'lo'}" for k in top if abs(z[k]) > 0.5]
    return "|".join(bits + ["long" if side > 0 else "short", category])


def post_mortem(frame, X, now, params=None, seed=0, salt="lessons"):
    """Post-mortem generator.

    frame: DataFrame indexed (date, ticker) with columns score, y (forward return), resolved (Timestamp when y became
    known), taken (bool); optional side, pattern, mae, stop. X: feature panel on the same index; `m_` columns are the
    market context. Only rows with resolved <= now are used. Returns a list of Episodes.
    Categories: taken loss -> regime_failure (context far from its earlier reference), else pattern_failure (its pattern
    loses on average in this frame), else false_positive (high-ranked) / losing_decision; untaken winner ->
    missed_winner (was high-ranked) / false_negative (scored low); everything else ok."""
    p = {**DEFAULTS, **(params or {})}
    if frame.empty:
        return []
    f = frame[(frame["resolved"] <= pd.Timestamp(now)).to_numpy() & frame.index.isin(X.index)]
    if f.empty:
        return []
    cols, _ = usable_features(X, p)
    ctx_cols = [c for c in cols if c.startswith("m_")]
    feat_cols = [c for c in cols if not c.startswith("m_")]
    rng = np.random.default_rng(seed)
    keep = f["taken"].astype(bool).to_numpy() | (rng.random(len(f)) < p["untaken_frac"])   # uniform untaken sample
    f = f[keep]
    side = f["side"] if "side" in f else np.sign(f["score"]).replace(0, 1)
    pnl_if = side * f["y"] - p["cost"]
    taken = f["taken"].astype(bool)
    d0 = f.index.get_level_values(0).min()
    prior = X.loc[X.index.get_level_values(0) < d0, ctx_cols] if ctx_cols else pd.DataFrame()
    ref = prior if len(prior) > 30 else (X.loc[f.index, ctx_cols] if ctx_cols else pd.DataFrame())
    mu = ref.mean().to_dict() if ctx_cols else {}
    sd = {k: (v if v > 1e-9 else 1.0) for k, v in ref.std().fillna(1.0).to_dict().items()} if ctx_cols else {}
    rank_pct = f["score"].abs().groupby(level=0).rank(pct=True)
    pat_mean = {}
    if "pattern" in f:
        tk = taken & f["pattern"].notna()
        pat_mean = pnl_if[tk].groupby(f.loc[tk, "pattern"]).mean().to_dict()
    order = f["resolved"].rank(method="first").astype(int) - 1
    Xf = X.loc[f.index]
    out = []
    for i, idx in enumerate(f.index):
        dt, tkr = idx
        row = Xf.iloc[i]
        ctx = {c: float(row[c]) if np.isfinite(row[c]) else 0.0 for c in ctx_cols}
        feats = {c: float(row[c]) for c in feat_cols if np.isfinite(row[c])}
        pnl, s, tk_, rp = float(pnl_if.iloc[i]), float(side.iloc[i]), bool(taken.iloc[i]), float(rank_pct.iloc[i])
        if tk_:
            if pnl < -p["loss_thr"]:
                zmax = max((abs((v - mu.get(k, 0)) / sd.get(k, 1)) for k, v in ctx.items()), default=0.0)
                pat = f["pattern"].iloc[i] if "pattern" in f else None
                if zmax > p["regime_z"]:
                    cat = "regime_failure"
                elif pat is not None and pd.notna(pat) and pat_mean.get(pat, 0.0) < 0:
                    cat = "pattern_failure"
                else:
                    cat = "false_positive" if rp >= 0.5 else "losing_decision"
            else:
                cat = "ok"
        else:
            cat = ("missed_winner" if rp >= 0.8 else "false_negative") if pnl > p["loss_thr"] else "ok"
        mae = float(f["mae"].iloc[i]) if "mae" in f else None
        stop = float(f["stop"].iloc[i]) if "stop" in f else None
        sc = float(f["score"].iloc[i])
        out.append(Episode(
            eid=_hash(salt, dt, tkr, s, tk_), situation=_situation(ctx, mu, sd, s, cat), features=feats,
            context=ctx, decision={"taken": float(tk_), "side": s, "score": sc, "rank_pct": rp},
            outcome={"pnl": pnl, "y": float(f["y"].iloc[i])}, category=cat,
            confidence=float(min(1.0, abs(sc) / (float(f["score"].abs().max()) + 1e-12))),
            counterfactual=_counterfactual(tk_, s, float(f["y"].iloc[i]), pnl, p["cost"], mae, stop),
            origin={"week": _hash(salt, *pd.Timestamp(dt).isocalendar()[:2]), "tk": _hash(salt, tkr)},
            seq=int(order.iloc[i])))
    return out


# ------------------------------------------------------------------ lessons
@dataclass
class Lesson:
    lid: str
    conds: list                  # [(feature, ">"|"<=", threshold), ...]
    direction: int               # -1 downweight, +1 upweight
    factor: float
    category: str                # dominant error category among mistakes in the region (explanatory only)
    n: int
    n_weeks: int
    n_tickers: int
    delta: float
    t: float
    p: float
    val_delta: float
    a: float                     # Beta trust pseudo-counts
    b: float
    born_tick: int
    ttl: int
    uses: int = 0
    status: str = "active"
    stability: float = 1.0       # share of chronological blocks in which the effect had the same sign

    @property
    def trust(self):
        return self.a / (self.a + self.b)

    def weight(self):
        """0 at trust 0.5 (no evidence either way), 1 at trust 0.9 and above."""
        return float(np.clip((self.trust - 0.5) / 0.4, 0.0, 1.0))

    def mask(self, X):
        m = np.ones(len(X), bool)
        for f, op, thr in self.conds:
            if f not in X:
                return np.zeros(len(X), bool)
            v = X[f].to_numpy(float)
            m &= (v > thr) if op == ">" else (v <= thr)
        return m

    def describe(self):
        c = " and ".join(f"{f} {op} {thr:.4g}" for f, op, thr in self.conds)
        return f"{'down' if self.direction < 0 else 'up'}weight x{self.factor:g} when {c} [{self.category}]"


def _welch(n1, s1, ss1, n0, s0, ss0):
    with np.errstate(divide="ignore", invalid="ignore"):
        m1, m0 = s1 / n1, s0 / n0
        v1 = (ss1 - n1 * m1 ** 2) / np.maximum(n1 - 1, 1)
        v0 = (ss0 - n0 * m0 ** 2) / np.maximum(n0 - 1, 1)
        t = (m1 - m0) / np.sqrt(v1 / n1 + v0 / n0)
    return m1 - m0, np.where(np.isfinite(t), t, 0.0)


class LessonBook:
    def __init__(self, params=None, seed=0):
        self.p = {**DEFAULTS, **(params or {})}
        self.seed = seed
        self.episodes = {}
        self.lessons = {}
        self.tick_n = 0
        self.rejected_mining = {}          # reason -> count, for the report

    # ---- episodes
    def record(self, episodes):
        for e in episodes:
            self.episodes.setdefault(e.eid, e)
        return len(self.episodes)

    def category_counts(self):
        return pd.Series([e.category for e in self.episodes.values()], dtype=object).value_counts().to_dict()

    def _arrays(self):
        eps = sorted(self.episodes.values(), key=lambda e: e.seq)
        names = sorted({k for e in eps for k in {**e.features, **e.context}})
        F = np.array([[({**e.features, **e.context}).get(k, np.nan) for k in names] for e in eps], float)
        F = F.reshape(len(eps), len(names))
        pnl = np.array([e.pnl for e in eps], float)
        return eps, names, F, pnl, pd.factorize(np.array([e.origin["week"] for e in eps], dtype=object))[0], \
            pd.factorize(np.array([e.origin["tk"] for e in eps], dtype=object))[0]

    # ---- generalisation
    def learn(self):
        """Mine lessons from the stored episodes; returns the list of NEW lessons. Existing lessons keep their trust."""
        p = self.p
        self.rejected_mining = {}
        eps, names, F, pnl, wk, tk = self._arrays()
        n = len(eps)
        if n < 2 * p["min_support"]:
            self.rejected_mining["too_few_episodes"] = n
            return []
        split = int(n * (1 - p["val_frac"]))
        cands = []                                              # (feature idx, op, thr)
        for j in range(F.shape[1]):
            col = F[:, j]
            ok = np.isfinite(col)
            if ok.sum() < p["min_support"] * 2 or np.nanstd(col) < 1e-12:
                continue
            for q in sorted({float(np.quantile(col[ok], q)) for q in p["q_levels"]}):
                cands += [(j, ">", q), (j, "<=", q)]
        if not cands:
            self.rejected_mining["no_candidates"] = 1
            return []
        M = np.column_stack([(F[:, j] > t) if op == ">" else (F[:, j] <= t) for j, op, t in cands])
        found = self._score_masks(M, pnl, wk, tk, split, [(c,) for c in cands])
        # pairs of the strongest same-direction singles: a conjunction must beat both parents by pair_gain
        top = sorted(found, key=lambda r: -abs(r["t"]))[: p["top_single_for_pairs"]]
        pair_masks, pair_conds = [], []
        for a in range(len(top)):
            for b in range(a + 1, len(top)):
                ra, rb = top[a], top[b]
                if np.sign(ra["delta"]) != np.sign(rb["delta"]) or ra["conds"][0][0] == rb["conds"][0][0]:
                    continue
                pair_masks.append(ra["mask"] & rb["mask"])
                pair_conds.append(ra["conds"] + rb["conds"])
        n_tests = len(cands) + len(pair_masks)
        allr = list(found)
        if pair_masks:
            for r in self._score_masks(np.column_stack(pair_masks), pnl, wk, tk, split, pair_conds):
                par = [x for x in top if x["conds"][0] in r["conds"]]
                if abs(r["delta"]) >= p["pair_gain"] * max(abs(x["delta"]) for x in par):
                    allr.append(r)
                else:
                    self.rejected_mining["pair_no_gain"] = self.rejected_mining.get("pair_no_gain", 0) + 1
        # Benjamini-Hochberg over EVERY condition tried (not just the survivors)
        allr.sort(key=lambda r: r["p"])
        cut = 0
        for k, r in enumerate(allr, 1):
            if r["p"] <= p["fdr"] * k / n_tests:
                cut = k
        if len(allr) > cut:
            self.rejected_mining["fdr"] = len(allr) - cut
        chosen = self._greedy_residual(allr[:cut], pnl, wk, tk, split)
        if cut > len(chosen):
            self.rejected_mining["redundant"] = cut - len(chosen)
        cats = np.array([e.category for e in eps])
        new = []
        for r in chosen:
            mk = r["mask"]
            mist = pd.Series(cats[mk & (cats != "ok")])
            cat = mist.value_counts().index[0] if len(mist) else "ok"
            direction = int(np.sign(r["delta"]))
            trust0 = float(np.clip(_N.cdf(r["t_val"]), 0.05, 0.95))
            conds = [(names[j], op, float(t)) for j, op, t in r["conds"]]
            lid = _hash("lesson", conds, direction)
            if lid in self.lessons:
                continue
            L = Lesson(lid, conds, direction, p["up"] if direction > 0 else p["down"], cat, int(mk.sum()),
                       r["n_weeks"], r["n_tickers"], float(r["delta"]), float(r["t"]), float(r["p"]),
                       float(r["val_delta"]), trust0 * p["prior_n"], (1 - trust0) * p["prior_n"], self.tick_n, p["ttl"],
                       stability=float(r["stability"]))
            self.lessons[lid] = L
            new.append(L)
        return new

    def _greedy_residual(self, pool, pnl, wk, tk, split):
        """Forward selection among the FDR survivors. After each pick, its region is removed from the data and every
        remaining candidate is re-scored on what is left, so a condition that only looks good because it is the
        complement of an already-chosen bad region ("f2 low is better" beside "f2 high loses") has no residual effect
        and is dropped. Returned stats are the residual ones, and each pick must keep its sign and residual p <= fdr."""
        p = self.p
        chosen = []
        active = np.ones(len(pnl), bool)
        saved = dict(self.rejected_mining)
        left = list(pool)
        while left and len(chosen) < p["max_lessons"]:
            M = np.column_stack([r["mask"] for r in left])[active]
            sp = int(np.searchsorted(np.flatnonzero(active), split))
            res = self._score_masks(M, pnl[active], wk[active], tk[active], sp, [r["conds"] for r in left])
            ok = []
            for r in res:
                orig = next(o for o in left if o["conds"] == r["conds"])
                if r["p"] <= p["fdr"] and np.sign(r["delta"]) == np.sign(orig["delta"]):
                    full = np.zeros(len(pnl), bool)
                    full[np.flatnonzero(active)[r["mask"]]] = True
                    ok.append({**r, "mask": full})
            if not ok:
                break
            best = min(ok, key=lambda r: r["p"])
            chosen.append(best)
            active &= ~best["mask"]
            left = [o for o in left if o["conds"] != best["conds"]]
        self.rejected_mining = saved
        return chosen

    def _score_masks(self, M, pnl, wk, tk, split, conds):
        """Welch effect of each condition (inside vs outside) on all data and on the later validation slice; the
        gates (support, distinctness, size, validation) each count what they reject."""
        p = self.p
        n = len(pnl)
        Mf = M.astype(float)
        n1, s1, ss1 = Mf.sum(0), Mf.T @ pnl, Mf.T @ pnl ** 2
        d, t = _welch(n1, s1, ss1, n - n1, pnl.sum() - s1, (pnl ** 2).sum() - ss1)
        va = np.arange(n) >= split
        Mv, pv = Mf[va], pnl[va]
        nv1, sv1, ssv1 = Mv.sum(0), Mv.T @ pv, Mv.T @ pv ** 2
        dv, tv = _welch(nv1, sv1, ssv1, va.sum() - nv1, pv.sum() - sv1, (pv ** 2).sum() - ssv1)
        nb = p["stab_blocks"]
        edges = np.linspace(0, n, nb + 1).astype(int)
        agree = np.zeros(M.shape[1])
        for lo, hi in zip(edges[:-1], edges[1:]):
            mb, pb = Mf[lo:hi], pnl[lo:hi]
            b1, bs1, bss1 = mb.sum(0), mb.T @ pb, mb.T @ pb ** 2
            db, _ = _welch(b1, bs1, bss1, len(pb) - b1, pb.sum() - bs1, (pb ** 2).sum() - bss1)
            agree += (np.sign(db) == np.sign(d)) & (b1 >= 5)
        stab = agree / nb
        out = []
        for c in range(M.shape[1]):
            mk = M[:, c]
            nw, nt = len(np.unique(wk[mk])), len(np.unique(tk[mk]))
            if n1[c] < p["min_support"] or n - n1[c] < p["min_support"]:
                reason = "support"
            elif nw < p["min_weeks"] or nt < p["min_tickers"]:
                reason = "not_distinct"            # one week or one name is an episode, not a lesson
            elif abs(d[c]) < p["min_abs_delta"]:
                reason = "tiny_effect"
            elif n1[c] > p["max_cover"] * n:
                reason = "too_broad"               # a lesson is a specific situation; the complement of one is not
            elif stab[c] < p["stab_min"]:
                reason = "unstable"                # the effect must hold in most chronological blocks, not one stretch
            elif np.sign(dv[c]) != np.sign(d[c]) or nv1[c] < p["val_min_n"] or abs(tv[c]) < p["val_t_floor"]:
                reason = "not_validated"
            else:
                out.append({"conds": list(conds[c]), "mask": mk, "delta": float(d[c]), "t": float(t[c]),
                            "p": _p_two_sided(t[c]), "n_weeks": nw, "n_tickers": nt, "val_delta": float(dv[c]),
                            "stability": float(stab[c]), "t_val": float(tv[c] * np.sign(d[c]))})
                continue
            self.rejected_mining[reason] = self.rejected_mining.get(reason, 0) + 1
        return out

    # ---- use
    def active(self):
        return [L for L in self.lessons.values() if L.status == "active"]

    def factor(self, X):
        """Multiplier per row from every active lesson that applies; each is pulled toward 1 by (1 - trust weight)."""
        f = np.ones(len(X))
        for L in self.active():
            w = L.weight()
            if w > 0:
                f[L.mask(X)] *= 1.0 + w * (L.factor - 1.0)
        lo, hi = self.p["factor_clip"]
        return pd.Series(np.clip(f, lo, hi), index=X.index)

    def adjust(self, score, X):
        return score * self.factor(X).reindex(score.index).to_numpy()

    def tick(self, n=1):
        """Advance the book's own clock; lessons older than their ttl expire."""
        self.tick_n += n
        for L in self.lessons.values():
            if L.status == "active" and self.tick_n - L.born_tick > L.ttl:
                L.status = "expired"

    def feedback(self, X, pnl):
        """Update trust from realised outcomes (pnl: Series aligned to X, e.g. side*y - cost for every candidate).
        A lesson helped on a day when the rows it applies to did better (up) / worse (down) than that day's average."""
        ex = (pnl - pnl.mean()).to_numpy()
        for L in self.active():
            m = L.mask(X)
            if m.sum() < 3:
                continue
            helped = float(ex[m].mean() * L.direction > 0)
            d = self.p["trust_decay"]
            L.a, L.b = L.a * d + helped, L.b * d + (1 - helped)
            L.uses += 1
            if L.uses >= self.p["min_uses"] and L.trust < self.p["trust_floor"]:
                L.status = "retired"

    # ---- ageing: re-test old lessons on episodes they have never seen, then mine the new ones
    def revalidate(self, episodes):
        """Score every active lesson on NEW episodes only (out of sample for it): inside-vs-outside effect of the
        lesson's region. A confirming effect adds to trust, a contradicting one subtracts, and a significant reversal
        retires it. Returns one record per lesson tested."""
        p = self.p
        eps = list(episodes)
        if not eps:
            return []
        names = sorted({k for e in eps for k in {**e.features, **e.context}})
        F = pd.DataFrame([{**e.features, **e.context} for e in eps], columns=names)
        pnl = np.array([e.pnl for e in eps], float)
        report = []
        for L in self.active():
            m = L.mask(F)
            n_in, n_out = int(m.sum()), int((~m).sum())
            if n_in < p["reval_min_n"] or n_out < p["reval_min_n"]:
                report.append({"lid": L.lid, "n": n_in, "action": "untested"})
                continue
            d, t = _welch(np.array([n_in]), np.array([pnl[m].sum()]), np.array([(pnl[m] ** 2).sum()]),
                          np.array([n_out]), np.array([pnl[~m].sum()]), np.array([(pnl[~m] ** 2).sum()]))
            z = float(t[0]) * L.direction
            if z > p["reval_z"]:
                L.a += 1.0; action = "confirmed"
            elif z < -p["reval_z"]:
                L.b += 1.0; action = "contradicted"
            else:
                action = "inconclusive"
            if z < -p["reval_retire_z"]:
                L.status = "retired"; action = "retired"
            L.uses += 1
            report.append({"lid": L.lid, "n": n_in, "delta": float(d[0]), "z": z, "action": action})
        return report

    def refresh(self, episodes):
        """One learning cycle on a newly closed batch: re-test old lessons on it first (before it can influence them),
        then record it and mine anything new. Returns (revalidation records, new lessons)."""
        eps = list(episodes)
        rev = self.revalidate(eps)
        self.record(eps)
        self.tick()
        return rev, self.learn()

    # ---- inspection
    def explain(self, X, score):
        """Per row: base score, multiplier, adjusted score and which lessons fired - for audit and the dashboard."""
        applied = [[] for _ in range(len(X))]
        for L in self.active():
            if L.weight() > 0:
                for i in np.flatnonzero(L.mask(X)):
                    applied[i].append(L.lid)
        f = self.factor(X)
        return pd.DataFrame({"score": score.to_numpy(), "factor": f.to_numpy(), "adjusted": score.to_numpy() * f.to_numpy(),
                             "lessons": [",".join(a) for a in applied]}, index=X.index)

    def diagnostics(self, X):
        """Coverage of each active lesson, pairs that pull in opposite directions on the same rows, and the spread of
        the combined multiplier. Overlapping opposite lessons cancel and are worth a look."""
        act = [L for L in self.active() if L.weight() > 0]
        masks = {L.lid: L.mask(X) for L in act}
        conflicts = []
        for i, a in enumerate(act):
            for b in act[i + 1:]:
                both = int((masks[a.lid] & masks[b.lid]).sum())
                if both and a.direction != b.direction:
                    conflicts.append({"a": a.lid, "b": b.lid, "rows": both})
        f = self.factor(X).to_numpy()
        return {"coverage": {L.lid: float(masks[L.lid].mean()) for L in act}, "conflicts": conflicts,
                "factor_quantiles": {q: float(np.quantile(f, q)) for q in (0.01, 0.1, 0.5, 0.9, 0.99)} if len(f) else {},
                "rows_touched": float((f != 1.0).mean()) if len(f) else 0.0}

    # ---- experiment interface (shared with engine.antimemo)
    def items(self):
        return [L.lid for L in self.active()]

    def subset(self, ids):
        b = LessonBook(self.p, self.seed)
        b.tick_n = self.tick_n
        b.lessons = {i: Lesson(**asdict(self.lessons[i])) for i in ids if i in self.lessons}
        return b

    # ---- persistence and audit
    def to_json(self):
        return json.dumps({"tick": self.tick_n, "lessons": [asdict(L) for L in self.lessons.values()],
                           "episodes": len(self.episodes), "categories": self.category_counts()}, default=float)

    @classmethod
    def from_json(cls, s, params=None):
        d = json.loads(s)
        b = cls(params)
        b.tick_n = d["tick"]
        for x in d["lessons"]:
            x["conds"] = [tuple(c) for c in x["conds"]]
            b.lessons[x["lid"]] = Lesson(**x)
        return b

    def save(self, path):
        Path(path).write_text(self.to_json(), encoding="utf-8")

    def summary(self):
        return [{"lid": L.lid, "rule": L.describe(), "n": L.n, "weeks": L.n_weeks, "tickers": L.n_tickers,
                 "delta": round(L.delta, 5), "t": round(L.t, 2), "trust": round(L.trust, 3), "status": L.status}
                for L in self.lessons.values()]


def audit_identity(book, tickers=(), dates=()):
    """Ticker strings, date strings or identity-named features found in the serialised lessons (empty = clean)."""
    text = book.to_json()
    hits = [str(t) for t in tickers if len(str(t)) >= 2 and f'"{t}"' in text]
    for d in dates:
        ts = pd.Timestamp(d)
        if ts.strftime("%Y-%m-%d") in text or ts.strftime("%Y%m%d") in text:
            hits.append(ts.strftime("%Y-%m-%d"))
    hits += [f for L in book.lessons.values() for f, _, _ in L.conds if f.lower() in FORBIDDEN_KEYS]
    return hits
