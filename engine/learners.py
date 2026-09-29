"""Generalising learners (Bible Phases 11, 21-23, 45; canon C54, C55, C56).

The owner's product is the LEARNING system (C54), and the harness (engine.learning_delta) has measured that the system's own
learners memorise: same-year gains, zero transfer. This module builds learners meant to TRANSFER, and is honest about the
only test that counts: the harness's transfer delta on unseen LATER years with a 95% CI above zero.

What a learner here does. It looks at many past windows at once (never a window that ended after the target year began, C56),
replays a small library of candidate SETTINGS on each of them cheaply (a ledger of weekly returns), and adopts a candidate only
when its advantage over the current setting
  * exists in enough distinct years (n years, share of years positive, t of the per-year deltas),
  * has the same sign in the earliest two thirds of those years AND in the latest third (a temporal hold-out), and
  * costs nothing on the guard metrics (mean weekly return, worst-5% week),
after shrinking the estimate toward zero (`prior_n` phantom zero-years). Anything less leaves the state untouched, so a world
with nothing to learn produces a learner that does nothing rather than one that chases noise.

Setting = cfg overrides (k, pool_q, pick, liq_q, w_move, ...) + RULES (keep-windows on a cross-sectional rank: veto names outside
them) + a learned MOVE SCORE that replaces p_move + an optional REGIME map (rules per market-regime bucket). Rules and the move
score act by patching the snapshot the player is handed (`patch_snapshot`): vetoed names get ev_red_flag = 1 (the system's own
eligibility veto), the move score replaces p_move. A learned state therefore contains only numbers about features - no window
id, no date, no ticker - and passes the blindness audit unchanged.

Learners (each a candidate generator over the shared gate):
  BandCfgLearner       (a) cfg knobs that move the weekly move toward the 5-10% band, chosen across years
  BandPoolLearner      (a) rank windows on volatility/max20/p_move that keep the pool where the portfolio lands in band
  RegimeMapLearner     (b) regime bucket -> pool rule, learned across years with heavy shrinkage, one bucket at a time
  LessonLearner        (c) veto lessons ("names in the top decile of X lose") kept only if they held in >= N distinct years
  MoverUseLearner      (d) which movers to hold: a movement score fitted on realised |move| across years, sign-consistent only
Everything is deterministic and past-only. Adopt nothing from here without the harness verdict."""
from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import learning_delta as L
from . import policy

PANEL_COLS = ("mu_raw", "p_move", "evidence", "vol20", "max20", "log_dv", "ev_red_flag", "ev_offering", "r5")
CTX_COLS = ("m_vix", "m_vix_term", "m_spy_ma200", "m_spy_ma50", "m_breadth", "m_dispersion", "m_spy_r5", "m_vix_chg5")
RULE_FEATS = ("mu_raw", "p_move", "evidence", "vol20", "max20", "log_dv", "r5", "r5abs", "muabs")
MOVE_FEATS = ("p_move", "vol20", "max20", "r5abs", "muabs", "log_dv")
BAND = (0.05, 0.10)
HORIZON = 5                                # sessions a weekly pick is held
GRIDS = {"k": [1, 2, 3, 4, 6, 8, 12], "pool_q": [0.5, 0.7, 0.8, 0.9, 0.95, 0.98], "pick": ["top", "hivol"],
         "liq_q": [0.0, 0.2, 0.3, 0.5, 0.7], "vol_filter": [False, True], "w_model": [0.5, 0.7, 0.85, 1.0],
         "w_move": [0.0, 0.3, 0.5, 0.7]}     # every value sits on engine.adaptive.STEPS, so the adapter can still step from it
CFG_DEFAULTS = {"k": 4, "exit_q": 0.8, "max_per_sector": None, "pick": "top", "pool_q": 0.95, "w_model": 0.5, "liq_q": 0.0,
                "vol_filter": False, "stress_thr": None, "stress_k": 3, "trend_filter": None, "trend_gross": 0.5, "w_move": 0.0}
PATCH_KEYS = ("rules", "move_w", "regime")   # the only extra-state keys this module writes and its player reads


# ---------------------------------------------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------------------------------------------
def full_cfg(cfg):
    """A complete selection cfg: the defaults under whatever the state carries (non-scalars such as the evidence weights are
    dropped: the ledger scores with the snapshot's own `evidence`)."""
    out = dict(CFG_DEFAULTS)
    for k, v in (cfg or {}).items():
        if k == "ew" or isinstance(v, (dict, list, tuple, set)):
            continue
        out[k] = v
    return out


@dataclass(frozen=True)
class Setting:
    """One complete way of selecting names. Hashable, so ledgers can be memoised on it."""
    cfg: tuple = ()          # sorted (key, value) pairs of the FULL selection cfg
    rules: tuple = ()        # ((feature, lo_rank, hi_rank), ...): a name survives only if every feature's rank is inside
    move_w: tuple = ()       # ((feature, weight), ...): movement score that replaces p_move
    regime: tuple = ()       # (feature, (edge, ...), (rules_of_bucket_0, rules_of_bucket_1, ...)) or ()

    @staticmethod
    def make(cfg=None, rules=(), move_w=(), regime=()):
        return Setting(tuple(sorted(full_cfg(cfg).items())), tuple(tuple(r) for r in rules), tuple(tuple(m) for m in move_w), regime)

    @staticmethod
    def from_state(state):
        ex = state.extra or {}
        rules = tuple((str(f), float(a), float(b)) for f, a, b in ex.get("rules", []))
        mw = tuple(sorted((str(f), float(w)) for f, w in (ex.get("move_w") or {}).items()))
        rg = ex.get("regime") or ()
        if rg:
            rg = (str(rg["feat"]), tuple(float(e) for e in rg["edges"]),
                  tuple(tuple((str(f), float(a), float(b)) for f, a, b in bucket) for bucket in rg["rules"]))
        return Setting.make(state.cfg, rules, mw, rg)

    @property
    def cfgd(self):
        return dict(self.cfg)

    def with_cfg(self, **kw):
        d = self.cfgd
        d.update(kw)
        return replace(self, cfg=tuple(sorted(d.items())))

    def with_rule(self, rule):
        """Add a keep-window; a second rule on the same feature REPLACES the first (windows never stack on one feature)."""
        kept = tuple(r for r in self.rules if r[0] != rule[0])
        return replace(self, rules=kept + (tuple(rule),))

    def key(self):
        return hashlib.sha1(json.dumps([self.cfg, self.rules, self.move_w, self.regime], default=str).encode()).hexdigest()[:16]

    def describe(self, base=None):
        d = self.cfgd
        diff = {k: v for k, v in d.items() if base is None or base.cfgd.get(k) != v}
        parts = [f"{k}={v}" for k, v in sorted(diff.items())] if base is not None else []
        parts += [f"keep {f} rank in [{a:.2f},{b:.2f}]" for f, a, b in self.rules]
        if self.move_w:
            parts.append("move score " + "+".join(f"{w:.2f}*{f}" for f, w in self.move_w))
        if self.regime:
            parts.append(f"regime map on {self.regime[0]} ({len(self.regime[1]) + 1} buckets)")
        return "; ".join(parts) if parts else "no change"

    def apply(self, state):
        """The LearnedState after adopting this setting: cfg overrides in cfg, rules / move score / regime map in extra."""
        cfg = dict(state.cfg)
        base = full_cfg(cfg)
        for k, v in self.cfg:
            if base.get(k) != v or k in cfg:
                cfg[k] = v
        extra = {k: v for k, v in state.extra.items() if k not in PATCH_KEYS}
        if self.rules:
            extra["rules"] = [[f, a, b] for f, a, b in self.rules]
        if self.move_w:
            extra["move_w"] = {f: w for f, w in self.move_w}
        if self.regime:
            extra["regime"] = {"feat": self.regime[0], "edges": list(self.regime[1]),
                               "rules": [[[f, a, b] for f, a, b in bucket] for bucket in self.regime[2]]}
        return cfg, extra


# ---------------------------------------------------------------------------------------------------------------
# cross-sectional features, rules, selection (shared verbatim by the ledger and by the player)
# ---------------------------------------------------------------------------------------------------------------
def _rank(x):
    return pd.Series(np.asarray(x, float)).rank(pct=True).to_numpy()


def rank_features(cols):
    """Percentile ranks (0..1, average ties, NaN stays NaN) of every rule feature within one cross-section. `r5abs` and
    `muabs` are magnitudes: how far a name moved lately / how strongly the model leans either way."""
    out = {}
    for f in RULE_FEATS:
        if f == "r5abs":
            out[f] = _rank(np.abs(cols["r5"]))
        elif f == "muabs":
            out[f] = _rank(np.abs(cols["mu_raw"]))
        else:
            out[f] = _rank(cols[f])
    return out


def regime_bucket(regime, ctx):
    """Bucket index of a week's market reading, or -1 (no map / reading missing)."""
    if not regime or ctx is None:
        return -1
    v = ctx.get(regime[0], float("nan"))
    if v is None or not np.isfinite(v):
        return -1
    return int(np.searchsorted(np.asarray(regime[1], float), v, side="right"))


def rules_for_week(rules, regime, ctx):
    """Global rules plus the rules of this week's regime bucket (a bucket rule on a feature replaces the global one)."""
    b = regime_bucket(regime, ctx)
    if b < 0 or b >= len(regime[2]):
        return tuple(rules)
    have = {r[0] for r in regime[2][b]}
    return tuple(r for r in rules if r[0] not in have) + tuple(regime[2][b])


def rule_mask(cols, rules, ranks=None):
    """True for names that survive every rule. Rank NaN never survives a rule (an unknown feature earns no benefit of doubt)."""
    n = len(cols["mu_raw"])
    keep = np.ones(n, bool)
    if not rules:
        return keep
    ranks = ranks or rank_features(cols)
    for f, lo, hi in rules:
        r = ranks[f]
        keep &= np.isfinite(r) & (r >= lo - 1e-12) & (r <= hi + 1e-12)
    return keep


def move_score(cols, move_w, ranks=None):
    """Weighted mean of feature ranks: higher = more movement expected. Values in [0, 1] like p_move's rank."""
    ranks = ranks or rank_features(cols)
    tot = sum(w for _, w in move_w)
    if tot <= 0:
        return np.asarray(cols["p_move"], float)
    s = np.zeros(len(cols["mu_raw"]))
    for f, w in move_w:
        s += w * np.nan_to_num(ranks[f], nan=0.5)
    return s / tot


class WeekView:
    """One cross-section prepared once and scored under many Settings: the ranks every setting needs are computed a single time
    (the ledger evaluates dozens of settings on every week of every window, so this is where the time goes)."""

    def __init__(self, cols, ctx):
        self.cols, self.ctx = cols, dict(ctx or {})
        self.n = len(cols["mu_raw"])
        self._ranks = None
        self.mr, self.lr = _rank(cols["mu_raw"]), _rank(cols["log_dv"])
        self.vr, self.xr, self.pm = _rank(cols["vol20"]), _rank(cols["max20"]), _rank(cols["p_move"])
        self.ok0 = ~((cols["ev_red_flag"] > 0) | ((cols["ev_offering"] > 0) & (self.lr < 0.5)))
        self.vol = pd.Series(cols["vol20"])

    @property
    def ranks(self):
        if self._ranks is None:
            self._ranks = rank_features(self.cols)
        return self._ranks

    def select(self, setting):
        """Positions of the names a Setting holds this week. The system's own selection (policy.score blend -> eligibility ->
        policy.regime_targets), less the crypto lookup (inert on code names), with the rules applied as a veto. Held names are
        ignored (no hysteresis): every week is a fresh pick, identical across settings, so comparisons are fair."""
        cfg, c = setting.cfgd, self.cols
        pm = self.pm
        if setting.move_w:
            pm = _rank(move_score(c, setting.move_w, self.ranks))
        s = cfg["w_model"] * self.mr + (1 - cfg["w_model"]) * c["evidence"]
        wm = cfg.get("w_move", 0.0)
        if wm > 0:
            s = (1 - wm) * s + wm * pm
        ok = self.ok0.copy()
        if cfg["vol_filter"]:
            ok &= ~((self.vr > 0.9) | (self.xr > 0.9))
        ok &= self.lr >= cfg["liq_q"]
        rules = rules_for_week(setting.rules, setting.regime, self.ctx)
        if rules:
            ok &= rule_mask(c, rules, self.ranks)
        if not ok.any():
            return np.array([], int)
        ss = pd.Series(s)
        t = policy.regime_targets(ss[ok], [], cfg, self.vol, {}, self.ctx)
        return np.asarray(t.index, int)


def select_names(cols, ctx, setting):
    return WeekView(cols, ctx).select(setting)


def patch_snapshot(snap, extra):
    """What a learned state does to a snapshot a player is about to trade: rules veto names (ev_red_flag = 1, the system's own
    eligibility veto), a move score replaces p_move. Same rule_mask / move_score as the ledger, so learning and playing agree.
    No extra state, or none of this module's keys: the snapshot is returned untouched."""
    rules, mw, rg = extra.get("rules"), extra.get("move_w"), extra.get("regime")
    if not (rules or mw or rg) or not len(snap):
        return snap
    cols = {c: snap[c].to_numpy(float) if c in snap else np.zeros(len(snap)) for c in PANEL_COLS}
    ctx = {c: float(snap[c].iloc[0]) for c in CTX_COLS if c in snap}
    st = Setting.from_state(L.LearnedState({}, {}, None, {k: extra[k] for k in PATCH_KEYS if k in extra}))
    ranks = rank_features(cols)
    keep = rule_mask(cols, rules_for_week(st.rules, st.regime, ctx), ranks)
    out = snap.copy()
    if not keep.all():
        out["ev_red_flag"] = np.where(keep, snap["ev_red_flag"].to_numpy(float) if "ev_red_flag" in snap else 0.0, 1.0)
    if st.move_w:
        out["p_move"] = move_score(cols, st.move_w, ranks)
    return out


class PolicyReplayPlayer(L.ReplayPlayer):
    """The real adaptive layer (engine.adaptive.replay) playing snapshots patched by the learned rules / move score."""

    def patch(self, snaps, visible):
        return {k: patch_snapshot(s, visible.extra) for k, s in snaps.items()}


# ---------------------------------------------------------------------------------------------------------------
# panels and ledgers
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class Panel:
    """A window reduced to what selection needs: per decision week a float32 matrix of PANEL_COLS, the realised
    HORIZON-session return of every name, and the week's market context. No names, no dates."""
    wid: str
    real_start: pd.Timestamp
    real_end: pd.Timestamp
    X: list                     # per week (n, len(PANEL_COLS)) float32
    fwd: list                   # per week (n,) float32
    ctx: np.ndarray             # (weeks, len(CTX_COLS)) float64, NaN when absent
    bps: float = 10.0
    n_untradeable: int = 0

    @property
    def n_weeks(self):
        return len(self.X)

    def cols(self, w):
        X = self.X[w].astype(np.float64)
        return {c: X[:, j] for j, c in enumerate(PANEL_COLS)}

    def ctxd(self, w):
        return {c: float(self.ctx[w, j]) for j, c in enumerate(CTX_COLS) if np.isfinite(self.ctx[w, j])}


def build_panel(snaps, closes, wid, real_start, real_end, bps=10.0, horizon=HORIZON):
    """Panel from a window's snapshots and closes. A week is kept only if `horizon` sessions of prices follow it. A name with no
    valid price at the decision or at the end of the holding period is untradeable: flagged ev_red_flag = 1, return 0."""
    idx = closes.index
    X, fwd, ctx, bad_n = [], [], [], 0
    for k in sorted(snaps, key=pd.Timestamp):
        d = pd.Timestamp(k)
        if d not in idx:
            continue
        i = idx.get_loc(d)
        if i + horizon >= len(idx):
            continue
        s = snaps[k]
        p0 = closes.iloc[i].reindex(s.index).to_numpy(float)
        p1 = closes.iloc[i + horizon].reindex(s.index).to_numpy(float)
        with np.errstate(all="ignore"):
            r = p1 / p0 - 1
        bad = ~np.isfinite(r) | ~(p0 > 0)
        bad_n += int(bad.sum())
        cols = np.column_stack([s[c].to_numpy(float) if c in s else np.zeros(len(s)) for c in PANEL_COLS])
        cols[bad, PANEL_COLS.index("ev_red_flag")] = 1.0
        X.append(cols.astype(np.float32))
        fwd.append(np.where(bad, 0.0, r).astype(np.float32))
        ctx.append([float(s[c].iloc[0]) if c in s and len(s) else np.nan for c in CTX_COLS])
    return Panel(wid, pd.Timestamp(real_start), pd.Timestamp(real_end), X, fwd,
                 np.array(ctx, float).reshape(len(X), len(CTX_COLS)), float(bps), bad_n)


def save_panel(panel, path):
    off = np.cumsum([0] + [len(x) for x in panel.X])
    np.savez_compressed(path, X=np.concatenate(panel.X) if panel.X else np.zeros((0, len(PANEL_COLS)), np.float32),
                        fwd=np.concatenate(panel.fwd) if panel.fwd else np.zeros(0, np.float32), off=off, ctx=panel.ctx,
                        meta=np.array([panel.wid, str(panel.real_start.date()), str(panel.real_end.date()), panel.bps,
                                       panel.n_untradeable], dtype=object))


def load_panel(path):
    z = np.load(path, allow_pickle=True)
    off, m = z["off"], z["meta"]
    return Panel(str(m[0]), pd.Timestamp(m[1]), pd.Timestamp(m[2]), [z["X"][a:b] for a, b in zip(off[:-1], off[1:])],
                 [z["fwd"][a:b] for a, b in zip(off[:-1], off[1:])], z["ctx"], float(m[3]), int(m[4]))


def week_returns_many(panel, settings):
    """Net weekly return series of several Settings on a Panel: equal weight in the names each holds, minus a round trip of
    costs. A week in which nothing is eligible is cash (0). Returns a list of arrays, one per setting."""
    cost = 2 * panel.bps / 1e4
    out = np.zeros((len(settings), panel.n_weeks))
    for w in range(panel.n_weeks):
        view = WeekView(panel.cols(w), panel.ctxd(w))
        for j, st in enumerate(settings):
            pos = view.select(st)
            out[j, w] = float(panel.fwd[w][pos].mean()) - cost if len(pos) else 0.0
    return list(out)


def week_returns(panel, setting):
    return week_returns_many(panel, [setting])[0]


def _evaluator_hash():
    src = "".join(inspect.getsource(f) for f in (WeekView, week_returns_many, rule_mask, move_score, rank_features, rules_for_week,
                                                   regime_bucket, full_cfg))
    return hashlib.sha1(src.encode()).hexdigest()[:10]


class Book:
    """The library of past windows a learner may consult. It knows each window's real dates (harness side) so it can refuse a
    window that has not ended before the target year begins (C56); a learner only ever sees ids that `before()` returns.
    Weekly-return ledgers are memoised per (window, setting) in memory and, when `cache_dir` is given, on disk under a name
    that carries a hash of the evaluator source (a code edit invalidates it instead of mixing old and new numbers)."""

    def __init__(self, cache_dir=None, bps=10.0):
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.bps = bps
        self.entries = {}        # wid -> {"start","end","loader"|"panel"}
        self.mem = {}            # (wid, setting key) -> ndarray
        self.dirty = set()
        self.evaluator = _evaluator_hash()
        self.n_evals = 0

    # ---- registry
    def register(self, wid, start, end, loader=None, panel=None):
        self.entries[wid] = {"start": pd.Timestamp(start), "end": pd.Timestamp(end), "loader": loader, "panel": panel}

    def register_window(self, w):
        """Register a learning_delta.Window (tests / planted worlds)."""
        self.register(w.id, w.real_start, w.real_end,
                      panel=build_panel(w.snaps, w.closes, w.id, w.real_start, w.real_end, w.bps))

    def before(self, real_start):
        """Ids of windows whose real end precedes `real_start`, oldest first."""
        t = pd.Timestamp(real_start)
        return [k for k, _ in sorted(((k, e["end"]) for k, e in self.entries.items() if e["end"] < t), key=lambda x: (x[1], x[0]))]

    def end_of(self, wid):
        return self.entries[wid]["end"]

    def panel(self, wid):
        e = self.entries[wid]
        if e["panel"] is None:
            e["panel"] = e["loader"]()
        return e["panel"]

    def drop_panels(self, keep=()):
        for k, e in self.entries.items():
            if e["loader"] is not None and k not in keep:
                e["panel"] = None

    # ---- ledgers
    def _file(self, wid):
        return None if self.cache_dir is None else self.cache_dir / f"ledger_{wid}_{self.evaluator}.json"

    def _load_disk(self, wid):
        f = self._file(wid)
        if f is not None and f.exists() and (wid, "_loaded") not in self.mem:
            for k, v in json.loads(f.read_text()).items():
                self.mem[(wid, k)] = np.array(v, float)
            self.mem[(wid, "_loaded")] = np.zeros(0)

    def flush(self):
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for wid in sorted(self.dirty):
            f = self._file(wid)
            data = {k[1]: v.tolist() for k, v in self.mem.items() if k[0] == wid and k[1] != "_loaded"}
            f.write_text(json.dumps(data))
        self.dirty.clear()

    def ensure(self, wid, settings):
        """Fill the ledger of `wid` for every setting not yet known, in one pass over the window's weeks."""
        self._load_disk(wid)
        seen, miss = set(), []
        for st in settings:
            k = st.key()
            if (wid, k) not in self.mem and k not in seen:
                seen.add(k)
                miss.append(st)
        if miss:
            for st, r in zip(miss, week_returns_many(self.panel(wid), miss)):
                self.mem[(wid, st.key())] = r
            self.dirty.add(wid)
            self.n_evals += len(miss)

    def returns(self, wid, setting):
        self.ensure(wid, [setting])
        return self.mem[(wid, setting.key())]

    def ctx(self, wid):
        return self.panel(wid).ctx

    # ---- the current window: added from the run the harness handed the learner
    def add_current(self, ctx, run, bps=None):
        """Register the window being learned from (its Run-1 snapshots and closes). If the same id was registered from disk, that
        panel is kept when it has the same number of weeks; otherwise it is rebuilt from the run."""
        cur = self.entries.get(ctx.window_id)
        if cur is not None and cur["end"] == pd.Timestamp(ctx.real_end):
            try:
                if self.panel(ctx.window_id).n_weeks:
                    return
            except Exception:
                pass
        p = build_panel(run.snaps, run.closes, ctx.window_id, ctx.real_start, ctx.real_end, self.bps if bps is None else bps)
        self.register(ctx.window_id, ctx.real_start, ctx.real_end, panel=p)
        for k in [k for k in self.mem if k[0] == ctx.window_id]:
            del self.mem[k]


# ---------------------------------------------------------------------------------------------------------------
# statistics: the gate every adoption passes
# ---------------------------------------------------------------------------------------------------------------
def series_metric(w, metric):
    """One window's value of a metric from its weekly returns."""
    w = np.asarray(w, float)
    if not len(w):
        return float("nan")
    if metric == "in_band":
        a = np.abs(w)
        return float(((a >= BAND[0]) & (a <= BAND[1])).mean())
    if metric == "mean_week":
        return float(w.mean())
    if metric == "worst5":
        return float(np.quantile(w, 0.05)) if len(w) > 5 else float(w.min())
    if metric == "pos_share":
        return float((w > 0).mean())
    if metric == "abs_mean":
        return float(np.abs(w).mean())
    raise ValueError(f"unknown metric {metric}")


@dataclass(frozen=True)
class GateParams:
    min_years: int = 8          # distinct past windows behind the estimate
    prior_n: float = 6.0        # phantom zero-delta years the estimate is shrunk toward
    min_share: float = 0.65     # share of years with a positive delta
    min_pos_years: int = 5      # ... and at least this many of them
    t_min: float = 2.5          # t-statistic (a best-of-many pick needs more than 2) of the per-year deltas
    oos_frac: float = 1 / 3     # latest share of years held out to confirm
    min_oos: int = 3
    guard_mean_tol: float = -0.0003    # cost allowed on mean weekly return (per week)
    guard_worst5_tol: float = -0.01    # ... and on the worst-5% week


@dataclass
class Gate:
    n: int
    mean: float
    shrunk: float
    t: float
    share_pos: float
    n_pos: int
    train_mean: float
    oos_mean: float
    oos_n: int
    passed: bool
    why: str

    def row(self):
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def gate(deltas, gp=GateParams()):
    """Adopt-or-not for a sequence of per-year deltas (oldest year first). All of: enough years, positive share, t, positive in
    the training part AND in the latest hold-out part. `shrunk` (sum / (n + prior_n)) is the ranking key."""
    d = np.asarray([x for x in deltas if np.isfinite(x)], float)
    n = len(d)
    if n == 0:
        return Gate(0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0, False, "no years")
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else float("inf")
    t = mean / (sd / np.sqrt(n)) if n > 1 and sd > 0 else (float("inf") if mean > 0 and n > 1 else 0.0)
    shrunk = float(d.sum() / (n + gp.prior_n))
    n_pos = int((d > 0).sum())
    k = max(gp.min_oos, int(round(n * gp.oos_frac)))
    tr, oo = (d[:-k], d[-k:]) if n > k else (d, d[:0])
    tm, om = (float(tr.mean()) if len(tr) else 0.0), (float(oo.mean()) if len(oo) else 0.0)
    why = []
    if n < gp.min_years:
        why.append(f"only {n} years (< {gp.min_years})")
    if n_pos / n < gp.min_share or n_pos < gp.min_pos_years:
        why.append(f"positive in {n_pos}/{n} years")
    if not t >= gp.t_min:
        why.append(f"t {t:.2f} < {gp.t_min}")
    if not (tm > 0):
        why.append("not positive in the earlier years")
    if len(oo) < gp.min_oos or not (om > 0):
        why.append("not positive in the latest held-out years")
    return Gate(n, mean, shrunk, float(t), n_pos / n, n_pos, tm, om, len(oo), not why, "; ".join(why) or "passes")


@dataclass
class Scored:
    setting: Setting
    gate: Gate
    guard_mean: float
    guard_worst5: float
    guard_ok: bool

    @property
    def adoptable(self):
        return self.gate.passed and self.guard_ok and self.gate.shrunk > 0


def score_candidates(book, ids, base, cands, metric, gp=GateParams(), sub=None):
    """Per-year delta of every candidate over `base` on `metric` (windows in `ids`, oldest first), gated. `sub(wid) -> bool mask
    over weeks` restricts the comparison to some weeks of each window (regime buckets); windows with < 4 such weeks are skipped."""
    for w in ids:
        book.ensure(w, [base] + list(cands))
    base_r = {w: book.returns(w, base) for w in ids}
    masks = {w: (np.ones(len(base_r[w]), bool) if sub is None else sub(w)) for w in ids}
    use = [w for w in ids if masks[w].sum() >= (1 if sub is None else 4)]
    b_m = np.array([series_metric(base_r[w][masks[w]], metric) for w in use])
    b_mean = np.array([series_metric(base_r[w][masks[w]], "mean_week") for w in use])
    b_w5 = np.array([series_metric(base_r[w][masks[w]], "worst5") for w in use])
    out = []
    for c in cands:
        r = {w: book.returns(w, c) for w in use}
        dm = np.array([series_metric(r[w][masks[w]], metric) for w in use]) - b_m
        gm = float(np.mean([series_metric(r[w][masks[w]], "mean_week") for w in use] - b_mean))
        g5 = float(np.mean([series_metric(r[w][masks[w]], "worst5") for w in use] - b_w5))
        g = gate(dm, gp)
        out.append(Scored(c, g, gm, g5, gm >= gp.guard_mean_tol and g5 >= gp.guard_worst5_tol))
    out.sort(key=lambda s: -s.gate.shrunk)
    return out


# ---------------------------------------------------------------------------------------------------------------
# candidate libraries
# ---------------------------------------------------------------------------------------------------------------
def cfg_candidates(base):
    """Every single-knob change on the grids. `pool_q` only matters when the pick is 'hivol'."""
    out = []
    cur = base.cfgd
    for knob, vals in GRIDS.items():
        if knob == "pool_q" and cur.get("pick") != "hivol":
            continue
        for v in vals:
            if cur.get(knob) != v:
                out.append(base.with_cfg(**{knob: v}))
    return out


def pool_candidates(base, feats=("vol20",), los=(0.0, 0.3, 0.5, 0.7, 0.85), his=(0.5, 0.7, 0.85, 0.95, 1.0)):
    """Keep-windows on the rank of a feature: the pool the picks are drawn from. Empty windows (hi - lo < 0.14) are skipped."""
    out = []
    for f in feats:
        for lo in los:
            for hi in his:
                if hi - lo >= 0.14 and not (lo == 0.0 and hi == 1.0):
                    out.append(base.with_rule((f, lo, hi)))
    return out


def lesson_candidates(base, feats=RULE_FEATS, tails=(0.8, 0.9, 0.95)):
    """Vetoes of a tail of one feature: drop the top or the bottom (1 - q) share of names by rank."""
    out = []
    have = {r[0] for r in base.rules}
    for f in feats:
        if f in have:
            continue
        for q in tails:
            out.append(base.with_rule((f, 0.0, q)))
            out.append(base.with_rule((f, 1.0 - q, 1.0)))
    return out


def fit_move_weights(book, ids, feats=MOVE_FEATS, min_share=0.7, max_rows=60000, seed=0):
    """A movement score fitted across windows: per window, least squares of the within-week rank of realised |return| on the
    within-week ranks of `feats`; a feature keeps a weight only if its coefficient is positive in at least `min_share` of the
    windows (the sign held across years), weight = its mean coefficient. Weights are normalised to sum to 1. Returns
    (weights tuple, per-feature diagnostics)."""
    rng = np.random.default_rng(seed)
    coefs = []
    for wid in ids:
        pan = book.panel(wid)
        A, y = [], []
        for w in range(pan.n_weeks):
            rk = rank_features(pan.cols(w))
            ok = np.isfinite(_rank(np.abs(pan.fwd[w]))) & (pan.X[w][:, PANEL_COLS.index("ev_red_flag")] == 0)
            A.append(np.column_stack([np.nan_to_num(rk[f], nan=0.5) for f in feats])[ok])
            y.append(_rank(np.abs(pan.fwd[w]))[ok])
        if not A:
            continue
        A, y = np.concatenate(A), np.concatenate(y)
        if len(y) > max_rows:
            pick = rng.choice(len(y), max_rows, replace=False)
            A, y = A[pick], y[pick]
        Ac = np.column_stack([A - A.mean(0), np.ones(len(A)) * 0.0])[:, :-1]
        yc = y - y.mean()
        coefs.append(np.linalg.solve(Ac.T @ Ac + 1e-3 * len(A) * np.eye(len(feats)), Ac.T @ yc))
    if not coefs:
        return (), {}
    C = np.array(coefs)
    diag = {f: {"mean": float(C[:, j].mean()), "share_pos": float((C[:, j] > 0).mean())} for j, f in enumerate(feats)}
    w = np.array([C[:, j].mean() if (C[:, j] > 0).mean() >= min_share and C[:, j].mean() > 0 else 0.0 for j in range(len(feats))])
    if w.sum() <= 0:
        return (), diag
    w = w / w.sum()
    return tuple((f, round(float(x), 4)) for f, x in zip(feats, w) if x > 0), diag


# ---------------------------------------------------------------------------------------------------------------
# learners
# ---------------------------------------------------------------------------------------------------------------
class CrossYearLearner:
    """Base: consult the past windows in the Book (all that ended before the target year began, plus the window just played),
    generate candidates, adopt through the gate. Subclasses supply `propose`. The learner never reads a window's own run outcome
    beyond its snapshots and closes, and never anything dated after `ctx.real_start` other than the window it is learning from."""
    name = "cross_year"
    metric = "in_band"

    def __init__(self, book, metric=None, max_rounds=1, gp=GateParams(), log_fn=None):
        self.book, self.max_rounds, self.gp, self.log_fn = book, max_rounds, gp, log_fn
        self.metric = metric or self.metric
        self.last_ids = []
        self.last_scored = []
        self.last_notes = []

    def propose(self, cur, ids):
        raise NotImplementedError

    def history(self, ctx, run):
        """Ids usable now, oldest first: past windows (ended before ctx.real_start) then the current one. The invariant is asserted."""
        self.book.add_current(ctx, run)
        ids = self.book.before(ctx.real_start)
        ids = [i for i in ids if i != ctx.window_id]
        bad = [i for i in ids if self.book.end_of(i) >= pd.Timestamp(ctx.real_start)]
        if bad:
            raise L.BlindnessError(f"history contains windows that did not end before {ctx.real_start.date()}: {bad}")
        ids = ids + [ctx.window_id]
        self.last_ids = ids
        return ids

    def learn(self, state, run, ctx):
        ids = self.history(ctx, run)
        cur = base = Setting.from_state(state)
        notes = [f"{self.name}: {len(ids) - 1} past windows + current"]
        self.last_scored = []
        for rnd in range(self.max_rounds):
            cands = [c for c in self.propose(cur, ids) if c != cur]
            if not cands:
                break
            sc = score_candidates(self.book, ids, cur, cands, self.metric, replace(self.gp, t_min=self.gp.t_min + 0.5 * rnd))
            self.last_scored.append(sc[:5])
            best = next((s for s in sc if s.adoptable), None)
            if best is None:
                top = sc[0]
                notes.append(f"round {rnd + 1}: nothing passed the gate (best {top.setting.describe(cur)}: {top.gate.why})")
                break
            g = best.gate
            notes.append(f"round {rnd + 1}: adopt [{best.setting.describe(cur)}] delta {g.mean:+.4f} over {g.n} years "
                         f"({g.n_pos} positive, t {g.t:.1f}, held-out {g.oos_mean:+.4f})")
            cur = best.setting
        self.book.flush()
        self.last_notes = notes
        cfg, extra = cur.apply(state)
        return L.LearnedState(cfg, dict(state.meta), state.ltm, extra, state.lineage + notes)


class BandCfgLearner(CrossYearLearner):
    """(a) The cfg knobs (k, pool, pick, liquidity, movement weight) that put more weeks in the 5-10% band, chosen on the
    ledger of all past years; mean return and the worst week may not pay for it (guards)."""
    name = "band_cfg"
    metric = "in_band"

    def __init__(self, book, **kw):
        kw.setdefault("max_rounds", 2)
        super().__init__(book, **kw)

    def propose(self, cur, ids):
        return cfg_candidates(cur)


class BandPoolLearner(CrossYearLearner):
    """(a) Which slice of the cross-section to draw picks from so the portfolio's weekly move lands in band: keep-windows on the
    rank of volatility / recent max / movement probability, chosen across years."""
    name = "band_pool"
    metric = "in_band"

    def __init__(self, book, feats=("vol20", "max20", "p_move"), **kw):
        super().__init__(book, **kw)
        self.feats = feats

    def propose(self, cur, ids):
        return pool_candidates(cur, self.feats)


class LessonLearner(CrossYearLearner):
    """(c) Learning from mistakes aggregated ACROSS years: a lesson is a veto ('names in the top decile of X do worse'). It is kept
    only if the veto helped in at least `min_years` distinct past years (gate: n, positive share, t, and the latest years
    confirm it out of sample). Up to `max_rounds` lessons stack."""
    name = "lessons"
    metric = "mean_week"

    def __init__(self, book, min_years=10, feats=RULE_FEATS, **kw):
        kw.setdefault("max_rounds", 3)
        kw.setdefault("gp", GateParams(min_years=min_years, min_pos_years=max(5, int(min_years * 0.65)), t_min=2.5,
                                       guard_mean_tol=-1.0))
        super().__init__(book, **kw)
        self.feats = feats

    def propose(self, cur, ids):
        return lesson_candidates(cur, self.feats)


class MoverUseLearner(CrossYearLearner):
    """(d) Which movers to hold. Fit a movement score on the realised |move| across past windows (a feature counts only if its
    effect had the same sign in >= 70% of the years), then test blending it in as the movement weight of the pick. The gate
    decides; if no feature is sign-stable the learner does nothing."""
    name = "mover_use"
    metric = "in_band"

    def __init__(self, book, weights=(0.3, 0.5, 0.7), **kw):
        super().__init__(book, **kw)
        self.weights = weights
        self.diag = {}

    def propose(self, cur, ids):
        mw, self.diag = fit_move_weights(self.book, ids)
        if not mw:
            return []
        return [replace(cur, move_w=mw).with_cfg(w_move=v) for v in self.weights]


class RegimeMapLearner(CrossYearLearner):
    """(b) Regime -> pool mapping learned across many years with heavy shrinkage. Weeks are bucketed by a market reading (edges =
    quantiles of the pooled past weeks). For each bucket, each candidate pool rule is judged only on that bucket's weeks, year by
    year, against the current setting, with `prior_n` large (12). A bucket keeps the global setting unless its own rule passes the
    gate; the assembled map must then pass the gate as a whole on all weeks. Not week-to-week knob chasing: one map per state."""
    name = "regime_map"
    metric = "in_band"

    def __init__(self, book, feat="m_vix", n_buckets=3, feats=("vol20",), **kw):
        kw.setdefault("gp", GateParams(prior_n=12.0))
        super().__init__(book, **kw)
        self.feat, self.n_buckets, self.feats = feat, n_buckets, feats
        self.edges = ()

    def _edges(self, ids):
        j = CTX_COLS.index(self.feat)
        v = np.concatenate([self.book.ctx(w)[:, j] for w in ids])
        v = v[np.isfinite(v)]
        if len(v) < 40:
            return ()
        qs = np.linspace(0, 1, self.n_buckets + 1)[1:-1]
        e = np.unique(np.round(np.quantile(v, qs), 4))
        return tuple(float(x) for x in e)

    def learn(self, state, run, ctx):
        ids = self.history(ctx, run)
        base = Setting.from_state(state)
        edges = self._edges(ids)
        self.edges = edges
        notes = [f"{self.name}: {len(ids) - 1} past windows + current, buckets on {self.feat}"]
        if not edges:
            notes.append("no market reading in the history: no map")
            self.last_notes = notes
            return L.LearnedState(dict(state.cfg), dict(state.meta), state.ltm, dict(state.extra), state.lineage + notes)
        j = CTX_COLS.index(self.feat)
        bucket_of = {w: np.searchsorted(np.asarray(edges), self.book.ctx(w)[:, j], side="right") for w in ids}
        cands = [c for c in pool_candidates(base, self.feats) if c != base]
        rules_by_bucket = []
        for b in range(len(edges) + 1):
            sc = score_candidates(self.book, ids, base, cands, self.metric, self.gp, sub=lambda w, b=b: bucket_of[w] == b)
            best = next((s for s in sc if s.adoptable), None)
            self.last_scored.append(sc[:3])
            if best is None:
                rules_by_bucket.append(tuple(base.rules))
                notes.append(f"bucket {b}: kept global ({sc[0].gate.why if sc else 'no candidates'})")
            else:
                rules_by_bucket.append(tuple(best.setting.rules))
                notes.append(f"bucket {b}: {best.setting.describe(base)} delta {best.gate.mean:+.4f} over {best.gate.n} years")
        cur = base
        if any(r != tuple(base.rules) for r in rules_by_bucket):
            comp = replace(base, regime=(self.feat, edges, tuple(rules_by_bucket)))
            final = score_candidates(self.book, ids, base, [comp], self.metric, self.gp)[0]
            if final.adoptable:
                cur = comp
                notes.append(f"map adopted: overall delta {final.gate.mean:+.4f}, {final.gate.n_pos}/{final.gate.n} years positive")
            else:
                notes.append(f"map rejected as a whole: {final.gate.why if not final.gate.passed else 'guards'}")
        self.book.flush()
        self.last_notes = notes
        cfg, extra = cur.apply(state)
        return L.LearnedState(cfg, dict(state.meta), state.ltm, extra, state.lineage + notes)


LEARNERS = {"band_cfg": BandCfgLearner, "band_pool": BandPoolLearner, "regime_map": RegimeMapLearner, "lessons": LessonLearner,
            "mover_use": MoverUseLearner}


def make_learner(name, book, **kw):
    if name == "chain":
        return L.ChainLearner([make_learner(n, book, **kw) for n in ("band_pool", "lessons", "mover_use")])
    return LEARNERS[name](book, **kw)


# ---------------------------------------------------------------------------------------------------------------
# players for planted worlds (a light close-to-close weekly player that obeys the same patch as the real one)
# ---------------------------------------------------------------------------------------------------------------
class PolicyLedgerPlayer:
    """Weekly top-k from `select_names` under the state's cfg, after the learned patch; hold HORIZON sessions; equal weight.
    The real system's replay is far slower; this player exists so planted worlds can prove the learner without it."""

    def __init__(self, base_cfg=None):
        self.base_cfg = base_cfg or {}

    def play(self, pres, visible):
        cfg = {**self.base_cfg, **visible.cfg}
        st = Setting.make(cfg)
        idx = pres.closes.index
        weekly, equity, decisions, e = [], [1.0], [], 1.0
        for d, s in sorted(pres.snaps.items()):
            t = pd.Timestamp(d)
            if t not in idx:
                continue
            i = idx.get_loc(t)
            if i + HORIZON >= len(idx):
                continue
            sp = patch_snapshot(s, visible.extra)
            cols = {c: sp[c].to_numpy(float) if c in sp else np.zeros(len(sp)) for c in PANEL_COLS}
            ctx = {c: float(sp[c].iloc[0]) for c in CTX_COLS if c in sp}
            pos = select_names(cols, ctx, st)
            names = sorted(sp.index[pos])
            if names:
                r = pres.closes.iloc[i + HORIZON][names] / pres.closes.iloc[i][names] - 1
                w = float(np.nan_to_num(r.to_numpy(float)).mean()) - 2 * pres.bps / 1e4
            else:
                w = 0.0
            weekly.append(w)
            e *= 1 + w
            equity.append(e)
            decisions.append((str(t.date()), names))
        return L.Run(np.array(weekly), np.array(equity), decisions, None, {}, snaps=pres.snaps, closes=pres.closes)


def planted_window(seed, wid, year, world="band", n_stocks=120, n_weeks=30, first="2150-01-05"):
    """A synthetic window with a known law, dated `year` in the real era.
    band   weekly move scales with vol20, in EVERY year: a vol pool rule that lands the portfolio in the 5-10% band exists and generalises.
    lesson names in the top decile of max20 lose (-6% a week) in EVERY year: a veto lesson exists and generalises.
    flip   the max20 effect flips sign with the year's parity: nothing generalises (a learner must not adopt it).
    null   returns are independent of every feature: there is nothing to learn (a learner must adopt nothing).
    year   the vol20 effect exists only in years divisible by 4 (a year-specific fingerprint).
    regime weekly move scales with vol20 only in weeks when the market reading m_vix is above 17 (a regime -> pool law that holds every year)."""
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(first, periods=5 * n_weeks + 6)
    codes = sorted(f"S{k:04d}" for k in rng.permutation(n_stocks))
    fwd = np.zeros((n_weeks, n_stocks))
    vix_w = 17 + 4 * np.sin(np.arange(n_weeks) * 1.3 + rng.uniform(0, 6.3)) + rng.normal(0, 0.5, n_weeks)
    snapcols = []
    for w in range(n_weeks):
        vol = rng.uniform(0.008, 0.05, n_stocks)
        mx = rng.uniform(0.01, 0.2, n_stocks)
        mu = rng.normal(0, 0.005, n_stocks)
        z = rng.normal(size=n_stocks)
        if world == "band" or (world == "year" and year % 4 == 0) or (world == "regime" and vix_w[w] > 17):
            scale = vol * np.sqrt(5)
        elif world in ("null", "flip", "lesson", "year", "regime"):
            scale = np.full(n_stocks, 0.03)
        f = scale * z
        top = _rank(mx) > 0.9
        if world == "lesson":
            f = f - 0.06 * top
        elif world == "flip":
            f = f + (0.06 if year % 2 else -0.06) * top
        fwd[w] = np.clip(f, -0.6, 3.0)
        snapcols.append((vol, mx, mu))
    logp = np.zeros((len(sessions), n_stocks))
    for w in range(n_weeks):
        i0 = 5 * w + 4
        step = np.log1p(fwd[w]) / 5.0
        for s in range(1, 6):
            logp[i0 + s] = logp[i0 + s - 1] + step
    logp[5 * n_weeks + 5:] = logp[5 * n_weeks + 4]
    closes = pd.DataFrame(100 * np.exp(logp), index=sessions, columns=codes)
    opens = closes.shift(1).fillna(closes.iloc[0])
    vix = vix_w
    snaps = {}
    for w in range(n_weeks):
        vol, mx, mu = snapcols[w]
        df = pd.DataFrame({"mu_raw": mu, "p_move": rng.uniform(size=n_stocks), "evidence": rng.uniform(size=n_stocks),
                           "vol20": vol, "max20": mx, "log_dv": rng.uniform(15, 20, n_stocks), "ev_red_flag": 0.0,
                           "ev_offering": 0.0, "r5": rng.normal(0, 0.03, n_stocks), "e_dist_52wh": rng.uniform(size=n_stocks)},
                          index=pd.Index(codes, name="ticker"))
        for c, val in (("m_vix", vix[w]), ("m_vix_term", 0.95), ("m_spy_ma200", 0.02), ("m_spy_ma50", 0.01), ("m_breadth", 0.5),
                       ("m_dispersion", 0.03), ("m_spy_r5", 0.0), ("m_vix_chg5", 0.0)):
            df[c] = val
        snaps[str(sessions[5 * w + 4].date())] = df
    divs = {c: "D%d" % (k % 6) for k, c in enumerate(codes)}
    return L.Window(wid, snaps, closes, opens, 10.0, divs, pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31"),
                    real_tickers=frozenset({f"REAL{k}" for k in range(3)}))


PLANTED_BASE = {"k": 4, "pick": "top", "w_model": 1.0, "exit_q": 0.8, "pool_q": 0.95, "liq_q": 0.0, "vol_filter": False}


def planted_pairs(world, learner_name, n_hist=10, n_pairs=6, seed=0, n_stocks=80, n_weeks=24, **learner_kw):
    """Run a learner on a planted world through the real harness (learning_delta.run_pair): the learning windows are years
    1960 + 3i; each transfer window is a LATER year (2010 + 3i; three-year steps so the parity of the year alternates). Returns (records, learner, book)."""
    hist = [planted_window(L.derive_seed(seed, world, "h", i), f"h{i:02d}", 1960 + 3 * i, world, n_stocks, n_weeks) for i in range(n_hist)]
    learn_w = [planted_window(L.derive_seed(seed, world, "w", i), f"w{i:02d}", 1960 + 3 * (n_hist + i), world, n_stocks, n_weeks) for i in range(n_pairs)]
    trans = [planted_window(L.derive_seed(seed, world, "t", i), f"t{i:02d}", 2010 + 3 * i, world, n_stocks, n_weeks) for i in range(n_pairs)]
    book = Book()
    for w in hist:
        book.register_window(w)
    learner = make_learner(learner_name, book, **learner_kw)
    s0 = L.LearnedState(dict(PLANTED_BASE), {})
    player = PolicyLedgerPlayer()
    recs = [L.run_pair(w, t, player, learner, s0, L.derive_seed(seed, world, "pair"), log_fn=None) for w, t in zip(learn_w, trans)]
    return recs, learner, book
