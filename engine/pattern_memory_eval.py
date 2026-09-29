"""Bible phases 4, 5, 9 - walk-forward evaluation helpers for the timeline pattern memory (canon C57-C60).

Pure functions over a lifecycle-style `Panel` (engine.pattern_lifecycle): per-quarter observations of a pattern, block
schedules, a memory builder that ingests ONLY post-registration quarters that had matured by each block end, a scorer that
compares what the memory view uses against what it disregards against random controls, and a walk-forward threshold fit
(parameters chosen on early blocks, judged on later ones). scripts/pattern_memory_real.py drives these on the real panel;
tests/test_pattern_memory_eval.py drives them on synthetic panels.

Why post-registration only: a pattern is registered (its candidate try counted) at the block end T where a miner found it.
The window that selected it is contaminated by that selection, so its quarters inside/before T never become evidence;
observations begin with the first whole quarter after T. Everything the memory holds about a pattern is therefore
out-of-sample relative to the selection, and the scorer's next-window value is out-of-sample relative to the memory.
Pattern membership uses per-date cross-sectional quintiles, so a quarter's numbers do not depend on later dates."""
import itertools
import tempfile

import numpy as np
import pandas as pd

from . import pattern_memory as pm

HORIZON_SESSIONS = 7                     # label = next-open to close five sessions on, plus slack (matches run_pattern_bank)
MIN_QUARTER_DATES = 6
FIT_GRID = {"n_universal": [3, 4, 6, 8], "t_fail": [1.0, 1.5, 2.5], "tau_time_years": [1.5, 3.0, 6.0],
            "intra_k": [2, 3, 4], "use_fdr": [True, False]}


# ---------------------------------------------------------------- schedules and per-quarter evidence
def block_schedule(first, last, months=6):
    """Block-end dates T_0 < T_1 < ... from `first` while <= `last`, `months` apart."""
    out, t = [], pd.Timestamp(first)
    while t <= pd.Timestamp(last):
        out.append(t)
        t = t + pd.DateOffset(months=months)
    return out


class Quarters:
    """Calendar-quarter bucketing of a Panel's dates, computed once. `last_date[q]` is the quarter's last panel date (the
    observation date) and `first_date[q]` its first."""

    def __init__(self, panel):
        self.dates = panel.dates
        per = self.dates.to_period("Q")
        self.codes, uniq = pd.factorize(per, sort=True)
        self.nq = len(uniq)
        s = pd.Series(self.dates)
        g = s.groupby(self.codes)
        self.first_date = pd.DatetimeIndex(g.min().values)
        self.last_date = pd.DatetimeIndex(g.max().values)
        ctx = panel.ctx
        self.ctx = ctx.groupby(self.codes).mean() if len(ctx.columns) else pd.DataFrame(index=range(self.nq))
        self.ctx_cols = list(self.ctx.columns)


def quarter_stats(means, qs, min_dates=MIN_QUARTER_DATES):
    """Vectorised per-quarter (effect, t, n) of one pattern's per-date mean series. NaN rows where < min_dates dates."""
    fin = np.isfinite(means)
    v = np.where(fin, means, 0.0)
    n = np.bincount(qs.codes, weights=fin, minlength=qs.nq)
    sm = np.bincount(qs.codes, weights=v, minlength=qs.nq)
    sq = np.bincount(qs.codes, weights=v * v, minlength=qs.nq)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = sm / n
        var = (sq - n * mu * mu) / (n - 1)
        t = mu / np.sqrt(var / n)
    ok = (n >= min_dates) & np.isfinite(t) & (var > 1e-16)
    return np.where(ok, mu, np.nan), np.where(ok, np.clip(t, -99, 99), np.nan), n.astype(int)


def window_excess(panel, means, t0, t1, min_dates=4):
    """Mean of a pattern's per-date series over dates in (t0, t1], NaN if fewer than `min_dates` valid dates."""
    sel = (panel.dates > t0) & (panel.dates <= t1)
    v = means[sel]
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) >= min_dates else np.nan


def ctx_at(panel, T):
    """Market context (m_ columns) at the last panel date <= T, as a plain dict of floats (no dates)."""
    ix = panel.dates.searchsorted(pd.Timestamp(T), side="right") - 1
    if ix < 0 or not len(panel.ctx.columns):
        return {}
    row = panel.ctx.iloc[ix]
    return {c: float(v) for c, v in row.items() if np.isfinite(v)}


# ---------------------------------------------------------------- building the memory from block outputs
class Registry:
    """Patterns known to the memory: key -> {names, sign, T_reg, ingested quarter set}. Registered at the block end that
    found them; never removed."""

    def __init__(self):
        self.items = {}

    def register(self, key, names, sign, T):
        if key not in self.items:
            self.items[key] = {"names": names, "sign": 1 if sign >= 0 else -1, "T_reg": pd.Timestamp(T), "done": set(),
                               "first_block": pd.Timestamp(T)}
            return True
        return False

    def __len__(self):
        return len(self.items)


def ingest_block(mem, reg, panel, qs, means_cache, T, run_id, n_tried, horizon=HORIZON_SESSIONS):
    """Add to the memory every not-yet-ingested quarter of every registered pattern that starts after the pattern's
    registration and whose labels matured by T. Records the block's try count. Returns number of observations."""
    obs = []
    for key, it in reg.items.items():
        eff, tt, nn = quarter_stats(means_cache[key], qs)
        for q in range(qs.nq):
            if q in it["done"] or not np.isfinite(tt[q]) or qs.first_date[q] <= it["T_reg"]:
                continue
            if pm.add_sessions(qs.last_date[q], horizon) > pd.Timestamp(T):
                continue
            ctx = {c: float(qs.ctx.iloc[q][c]) for c in qs.ctx_cols if np.isfinite(qs.ctx.iloc[q][c])}
            obs.append({"key": key, "obs_date": qs.last_date[q], "effect": float(eff[q]), "n": int(nn[q]), "t": float(tt[q]),
                        "ctx": ctx, "horizon": horizon})
            it["done"].add(q)
    n = mem.add_observations(run_id, T, obs, n_tried=n_tried)
    return n


# ---------------------------------------------------------------- random controls
def random_patterns(panel, n, rng, kinds=("s", "p", "u"), q_levels=(0, 1, 3, 4)):
    """`n` random identity-free patterns over the panel's features, as name tuples in Panel.mask format. Extreme quintiles
    are sampled more (the miner mostly finds tails); no data is looked at."""
    f = panel.feats
    out = []
    while len(out) < n:
        kind = kinds[int(rng.integers(len(kinds)))]
        k = {"s": 1, "p": 2, "u": 3}[kind]
        if len(f) < k:
            continue
        pick = rng.choice(len(f), size=k, replace=False)
        qv = [int(rng.choice(q_levels)) for _ in range(k)]
        names = (kind,) + tuple(x for i, j in enumerate(pick) for x in (f[j], qv[i]))
        out.append(names)
    return out


def names_to_key(names):
    """Inverse of engine.pattern_lifecycle.parse_key_named for tuples: ('p','r5',4,'vol20',1) -> 'r5 q4 & vol20 q1'."""
    lits = [f"{names[i]} q{names[i + 1]}" for i in range(1, len(names), 2)]
    if names[0] == "u":
        return f"{lits[0]} & {lits[1]} unless {lits[2]}"
    return " & ".join(lits)


# ---------------------------------------------------------------- scoring a memory over blocks
def _tstat(x):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if len(x) < 3 or x.std(ddof=1) < 1e-15:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


class Scorer:
    """Everything needed to score memory views block by block without recomputing patterns: the next-window signed excess
    of each registered pattern, the market context at each block end, and random-pattern controls per block."""

    def __init__(self, panel, reg, means_cache, blocks, seed=0, n_random=100, subset_draws=30):
        self.blocks, self.reg = list(blocks), reg
        self.seed, self.subset_draws = seed, subset_draws
        self.ctx = [ctx_at(panel, T) for T in self.blocks]
        self.next_ex = {}                                  # key -> array over blocks: raw next-window mean excess
        for key in reg.items:
            m = means_cache[key]
            self.next_ex[key] = np.array([window_excess(panel, m, self.blocks[k], self.blocks[k + 1])
                                          if k + 1 < len(self.blocks) else np.nan for k in range(len(self.blocks))])
        rng = np.random.default_rng(seed)
        self.random_val = []
        for k in range(len(self.blocks)):
            if k + 1 >= len(self.blocks):
                self.random_val.append(np.nan)
                continue
            vals = []
            for names in random_patterns(panel, n_random, rng):
                m = panel.means(names)
                if m is None:
                    continue
                past = (panel.dates <= self.blocks[k] - pd.Timedelta(days=10))
                pv = m[past & np.isfinite(m)]
                if len(pv) < 20:
                    continue
                d = 1.0 if pv.mean() >= 0 else -1.0             # direction the way a naive trailing-sign miner would set it
                e = window_excess(panel, m, self.blocks[k], self.blocks[k + 1])
                if np.isfinite(e):
                    vals.append(d * e)
            self.random_val.append(float(np.mean(vals)) if vals else np.nan)

    def score(self, mem, cfg=None, block_ids=None):
        """Per-block table for one parameter set. Columns: n_use, n_off (disregarded/gated), mean and weight-weighted signed
        next-window excess of used patterns, mean of disregarded, all-registry baseline (miner sign), random-subset control
        (same count drawn from the patterns in the view), random-pattern control, and mode counts."""
        old = dict(mem.p)
        mem.p = {**old, **(cfg or {})}
        rows = []
        try:
            for k in (range(len(self.blocks) - 1) if block_ids is None else block_ids):
                T = self.blocks[k]
                v = mem.view(T + pd.Timedelta(days=1), self.ctx[k])
                use, off, wts, allv = [], [], [], []
                inview = []
                for key, w in v.weights.items():
                    e = self.next_ex.get(key, np.full(len(self.blocks), np.nan))[k]
                    if not np.isfinite(e):
                        continue
                    val = w.direction * e
                    inview.append(val)
                    (use if w.weight > 0 else off).append(val)
                    if w.weight > 0:
                        wts.append(w.weight)
                for key, it in self.reg.items.items():
                    if it["T_reg"] <= T:
                        e = self.next_ex[key][k]
                        if np.isfinite(e):
                            allv.append(it["sign"] * e)
                rng = np.random.default_rng(self.seed + 1000 + k)
                sub = np.nan
                if use and len(inview) >= len(use):
                    sub = float(np.mean([np.mean(rng.choice(inview, size=len(use), replace=False))
                                         for _ in range(self.subset_draws)]))
                modes = pd.Series([w.mode for w in v.weights.values()]).value_counts().to_dict()
                rows.append({"block": k, "T": T, "n_use": len(use), "n_off": len(off), "n_view": len(v.weights),
                             "use": float(np.mean(use)) if use else np.nan,
                             "use_w": float(np.average(use, weights=wts)) if use else np.nan,
                             "off": float(np.mean(off)) if off else np.nan,
                             "baseline": float(np.mean(allv)) if allv else np.nan, "subset": sub,
                             "random": self.random_val[k], **{f"n_{m}": modes.get(m, 0) for m in
                                                              ("universal", "local", "disregarded", "gated")}})
        finally:
            mem.p = old
        return pd.DataFrame(rows)


def summarise(table, min_use=5):
    """Across blocks: mean of each arm, paired differences with t-stats (blocks are the independent unit)."""
    if table.empty:
        return {"blocks": 0}
    t = table[table["n_use"] >= min_use]
    out = {"blocks": len(table), "blocks_with_use": len(t), "mean_n_use": float(table["n_use"].mean()),
           "mean_n_off": float(table["n_off"].mean())}
    for c in ("use", "use_w", "off", "baseline", "subset", "random"):
        out[c] = float(t[c].mean()) if len(t) and t[c].notna().any() else float("nan")
    for a, b in (("use", "off"), ("use", "subset"), ("use", "random"), ("use", "baseline")):
        d = (t[a] - t[b]).values if len(t) else np.array([])
        out[f"{a}_minus_{b}"] = float(np.nanmean(d)) if len(d) and np.isfinite(d).any() else float("nan")
        out[f"{a}_minus_{b}_t"] = _tstat(d)
    return out


# ---------------------------------------------------------------- walk-forward threshold fit
def objective(table, min_use=5):
    """Value of the view over a random selection of the same size, averaged over blocks; blocks where the view uses too
    few patterns contribute 0 (a config that switches everything off has no edge, not a good one)."""
    if table.empty:
        return 0.0
    v = []
    for r in table.itertuples():
        ok = r.n_use >= min_use and np.isfinite(r.use_w) and np.isfinite(r.subset)
        v.append(r.use_w - r.subset if ok else 0.0)
    return float(np.mean(v))


def sample_configs(grid, n, seed):
    """Seeded random subset of the full grid (full grid if n >= its size); deterministic order."""
    keys = sorted(grid)
    full = [dict(zip(keys, c)) for c in itertools.product(*[grid[k] for k in keys])]
    if n >= len(full):
        return full
    rng = np.random.default_rng(seed)
    return [full[i] for i in sorted(rng.choice(len(full), n, replace=False))]


def fit_thresholds(mem, scorer, fit_blocks, judge_blocks, defaults=None, grid=None, n_configs=60, seed=0):
    """Choose parameters by `objective` on `fit_blocks` ONLY, then report chosen vs defaults on `judge_blocks` (later).
    Raises if the two sets overlap or judge blocks are not strictly later, so the split cannot be blurred."""
    if max(fit_blocks) >= min(judge_blocks):
        raise ValueError("judge blocks must all come after the fit blocks")
    defaults = {k: pm.PARAMS[k] for k in (grid or FIT_GRID)} if defaults is None else defaults
    cfgs = sample_configs(grid or FIT_GRID, n_configs, seed)
    if defaults not in cfgs:
        cfgs.append(dict(defaults))
    fitted = []
    for c in cfgs:
        tab = scorer.score(mem, c, fit_blocks)
        fitted.append({**c, "objective": objective(tab)})
    fitted.sort(key=lambda r: -r["objective"])
    best = {k: fitted[0][k] for k in defaults}
    res = {"chosen": best, "defaults": dict(defaults), "fit_ranking": fitted[:10],
           "fit_objective_chosen": fitted[0]["objective"],
           "fit_objective_defaults": next(r["objective"] for r in fitted if all(r[k] == defaults[k] for k in defaults))}
    jt_best, jt_def = scorer.score(mem, best, judge_blocks), scorer.score(mem, defaults, judge_blocks)
    res["judge_chosen"], res["judge_defaults"] = summarise(jt_best), summarise(jt_def)
    res["judge_objective_chosen"], res["judge_objective_defaults"] = objective(jt_best), objective(jt_def)
    res["judge_tables"] = {"chosen": jt_best, "defaults": jt_def}
    return res


# ---------------------------------------------------------------- one-shot builder used by the real script and tests
def build_memory(root, panel, blocks, found_by_block, seed=0, horizon=HORIZON_SESSIONS):
    """found_by_block[k] = (DataFrame with key_named / effect, n_tried) for block end blocks[k]. Registers new patterns,
    ingests matured post-registration quarters, and returns (mem, registry, means_cache, qs)."""
    from .pattern_lifecycle import parse_key_named
    mem = pm.PatternMemory(root)
    reg, cache, qs = Registry(), {}, Quarters(panel)
    for k, T in enumerate(blocks):
        frame, n_tried = found_by_block.get(k, (pd.DataFrame(), 0))
        for _, r in (frame if frame is not None else pd.DataFrame()).iterrows():
            key = str(r["key_named"])
            try:
                names = parse_key_named(key)
            except ValueError:
                continue
            m = panel.means(names)
            if m is None:
                continue
            if reg.register(key, names, float(r["effect"]), T):
                cache[key] = m
        ingest_block(mem, reg, panel, qs, cache, T, f"block{k:03d}", n_tried, horizon)
    return mem, reg, cache, qs


def temp_root():
    return tempfile.mkdtemp(prefix="pm_eval_")
