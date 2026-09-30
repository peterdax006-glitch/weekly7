"""Future-scramble probes for the research-side components (INTEGRATION B01): the PatternMiner, Memory and the adaptive
Session, plus the model via engine.train. Kept apart from engine.train because train is on the trader's import closure
(improve -> retrain_guarded) and the blind path must never reach the pattern bank (leak channel 8d, F06 finding 29 Sep)."""
import numpy as np
import pandas as pd

from . import pit
from .train import LABEL_HORIZON, _model_outputs, future_scramble_gate as _gate

SCRAMBLE_COMPONENTS = ("model", "miner", "memory", "adaptive")


def _miner_outputs(fr, as_of, horizon, params):
    from .patterns import PatternMiner
    X, L = fr["X"], fr["L"].reindex(fr["X"].index)
    ud = pd.DatetimeIndex(sorted(X.index.get_level_values(0).unique()))
    Xp, yp = pit.purged_training_set(X, L["f"], as_of, horizon, pit.Calendar(ud))
    pm = PatternMiner(params).fit(Xp, yp.dropna(), pd.Timestamp(as_of))
    cols = [c for c in ("key_named", "effect", "status") if c in pm.patterns.columns]
    pats = pm.patterns[cols].reset_index(drop=True) if len(pm.patterns) else pd.DataFrame(columns=cols)
    return {"patterns": pats, "score": pm.score(X.xs(pd.Timestamp(as_of), level=0)).round(12)}


def _memory_outputs(fr, as_of):
    """Weekly lessons per momentum-quintile arm, recorded only for weeks whose outcome closed by as_of."""
    from .memory import Memory, CTX
    C = fr["closes"].loc[:pd.Timestamp(as_of)]
    wk_end = C.groupby(C.index.to_period("W-FRI")).apply(lambda g: g.index[-1])
    mem, ctx = Memory(), np.zeros(len(CTX))
    for w, (a, b) in enumerate(zip(wk_end.values[:-1], wk_end.values[1:])):
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        hist = C.loc[:a]
        if len(hist) < 6:
            continue
        mom = hist.iloc[-1] / hist.iloc[-6] - 1
        q = np.minimum((mom.rank(pct=True) * 5).astype(int), 4)
        ret = C.loc[b] / C.loc[a] - 1
        for k in range(5):
            names = q.index[q.values == k]
            if len(names):
                mem.record(("mom_q", int(k)), float(w), ctx, float(ret[names].mean()), date=b)
    ranks = mem.rank_arms(mem.arms(), float(len(wk_end)), ctx) if len(mem) else pd.DataFrame()
    return {"fingerprint": mem.fingerprint(), "ranks": ranks.drop(columns=["arm"], errors="ignore").round(12)}


def _adaptive_outputs(fr, as_of, cfg):
    """The adaptive Session replayed over the window; what it decided and held up to as_of."""
    from . import adaptive
    closes, opens, as_of = fr["closes"], fr.get("opens"), pd.Timestamp(as_of)
    snaps = {}
    for d in closes.index:
        c = closes.loc[:d]
        mu = (c.iloc[-1] / c.iloc[-6] - 1).fillna(0.0).values if len(c) > 5 else np.zeros(c.shape[1])
        snaps[str(d.date())] = pd.DataFrame({"mu_raw": mu, "evidence": 0.5, "vol20": 0.02, "max20": 0.05, "log_dv": 18.0,
                                             "ev_red_flag": 0.0, "ev_offering": 0.0, "r5": 0.0, "m_vix": 0.5, "m_vix_term": 0.9,
                                             "m_spy_ma200": 1.05}, index=pd.Index(closes.columns, name="ticker"))
    S = adaptive.replay(cfg, snaps, closes, 5.0, {}, adaptive=True, opens=opens)
    return {"decisions": [x for x in S.decisions if pd.Timestamp(x[0]) < as_of],
            "days": [(d, round(v, 9)) for d, v in S.days if pd.Timestamp(d) <= as_of],
            "fills": [(f["order_id"], round(f["fill_price"], 9)) for f in S.fills if pd.Timestamp(f["fill_date"]) <= as_of]}


ADAPTIVE_CFG = {"k": 2, "exit_q": 0.8, "rebalance_weeks": 1, "brake": None, "max_per_sector": None, "w_model": 1.0,
                "pick": "top", "pool_q": 0.7, "liq_q": 0.0, "vol_filter": False, "stress_thr": None, "stress_k": 2,
                "trend_filter": None, "trend_gross": 0.0}
MINER_FAST = {"min_n": 60, "max_pairs": 200, "max_unless": 50, "null_reps": 1, "top_singles": 20, "unless_top_pairs": 10,
              "null_max_patterns": 200, "shrink_k": 100}


def scramble_pipeline(components=SCRAMBLE_COMPONENTS, horizon=LABEL_HORIZON, miner_params=None, adaptive_cfg=None):
    """pipeline(frames, as_of) -> {component: outputs at as_of}, for pit.future_scramble. frames: X ((date, ticker)
    features), L (labels y, f on the same index), closes / opens (wide). Unknown components are refused."""
    bad = set(components) - set(SCRAMBLE_COMPONENTS)
    if bad or not components:
        raise ValueError(f"unknown or empty scramble components {sorted(bad)}; choose from {SCRAMBLE_COMPONENTS}")
    mp = {**MINER_FAST, **(miner_params or {})}
    cfg = adaptive_cfg or ADAPTIVE_CFG

    def pipeline(fr, as_of):
        out = {}
        if "model" in components:
            out["model"] = _model_outputs(fr, as_of, horizon)
        if "miner" in components:
            out["miner"] = _miner_outputs(fr, as_of, horizon, mp)
        if "memory" in components:
            out["memory"] = _memory_outputs(fr, as_of)
        if "adaptive" in components:
            out["adaptive"] = _adaptive_outputs(fr, as_of, cfg)
        return out
    return pipeline


def future_scramble_gate(frames, as_of, components=SCRAMBLE_COMPONENTS, seed=0, horizon=LABEL_HORIZON, require=True, **kw):
    """engine.train.future_scramble_gate over any of the four real components."""
    return _gate(frames, as_of, components, seed=seed, horizon=horizon, require=require,
                 pipeline=scramble_pipeline(components, horizon, **kw))
