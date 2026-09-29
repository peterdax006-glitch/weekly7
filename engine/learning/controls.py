"""The five frozen controls of the same-year rerun experiment (contract C62 section 25; checklist E03-E07, L01-L05; canon C57, C58).
STATUS: IMPLEMENTED — NOT VALIDATED (planted-world unit tests only; no real data was touched).

One interface -- begin_run / fit / observe / decide / end_run -- for five different learners, so the harness (same_year.py) can
run them side by side on identical disguised presentations:

  A  NoLearning          decisions never depend on any experience (a seeded lottery); the floor every learner must beat.
  B  LegitimateLearner   pluggable; the default wraps engine.pattern_memory.PatternMemory (the C58 date filter and a reliability
                         veto) around an identity-free (feature, quintile) evidence ledger read through the same date filter.
  C  IdentityMemoriser   remembers the realised outcome of every row it has seen, keyed by ticker+date (or by the exact numbers
                         of the row). It is the control that shows what fake learning looks like.
  D  RandomLearner       runs B's own update path but replaces every update by a random vector of exactly the same magnitude:
                         it 'learns' as much as B in the sense of moving, and must show no curve.
  E  LeakyLearner        is handed the future outcome through a channel a legitimate learner does not have. It must be caught by
                         the future firewall (same_year.LeakGuard) and must be distinguishable from B by its first-run skill.

FROZEN (L01-L05). Every control is registered as a FrozenControl: a code fingerprint (bytecode of its own methods, so a patch at
run time changes it, not only an edit on disk) plus a canonical config hash. Building a control whose code or config differs from
the record raises ControlChanged; the only way forward is ControlRegistry.supersede(), which appends a new version and keeps the
old one (history is immutable, section 49). A control is never tuned after seeing a result.

TIME. Anything that must know the real market date (the memory's C58 filter) receives it through a Moment and a MemoryGate; each
read is logged, so an audit can show which control looked at real dates. A learner works with the disguised date otherwise."""
from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from engine import pattern_memory as PM
from engine.learning.core import FirewallBreach, as_date, stable_hash
from engine.learning.planted_world import CANARY_PREFIX

LETTERS = ("A", "B", "C", "D", "E")
HORIZON_DAYS = 7                                   # a weekly forward return matures one week after its features
VETO_REASONS = ("failed_recently", "broke_within_year", "failed_in_similar_context", "lapsed", "broke_and_not_restored")


class ControlChanged(RuntimeError):
    """A frozen control's code or config no longer matches its record. Nothing may run (L01-L05)."""


class ControlContractError(ValueError):
    """A control object does not satisfy the shared fit/observe/decide interface."""


# ---------------------------------------------------------------------------------------------------------------
# time: what a control is told about 'now'
# ---------------------------------------------------------------------------------------------------------------
class MemoryGate:
    """Capability to read the real market date. Only the memory's date filter needs it (C58); every read is logged on the Moment."""

    def __init__(self, owner: str):
        if not owner:
            raise ValueError("a MemoryGate needs an owner name for the audit log")
        self.owner = owner


@dataclass(frozen=True)
class Moment:
    """A decision time. `disguised` is what a player sees; the real date is reachable only through a MemoryGate."""
    disguised: pd.Timestamp
    week: int
    real_ts: pd.Timestamp = field(repr=False, compare=False)
    reads: list = field(default_factory=list, repr=False, compare=False)

    def real(self, gate: MemoryGate) -> pd.Timestamp:
        if not isinstance(gate, MemoryGate):
            raise FirewallBreach("the real date was requested without a MemoryGate")
        self.reads.append(gate.owner)
        return self.real_ts


# ---------------------------------------------------------------------------------------------------------------
# identity-free evidence: per-week sums for every (feature, quintile) key
# ---------------------------------------------------------------------------------------------------------------
def feature_columns(X: pd.DataFrame) -> list[str]:
    """Plain features f*, market context m_* excluded. A canary column is a breach, never a feature."""
    bad = [c for c in X.columns if str(c).startswith(CANARY_PREFIX)]
    if bad:
        raise FirewallBreach(f"future-leak canary column(s) reached a control: {bad}")
    return [c for c in X.columns if str(c).startswith("f")]


def quintile_codes(X: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """(rows, features) int8 cross-sectional quintiles of one week's cross-section; ties get the average rank (as the miner does)."""
    if len(X) == 0 or not cols:
        return np.zeros((len(X), len(cols)), dtype=np.int8)
    r = X[cols].rank(pct=True).fillna(0.5).to_numpy()
    return np.minimum((r * 5).astype(np.int8), 4)


def key_names(cols: list[str]) -> list[str]:
    return [f"{c} q{q}" for c in cols for q in range(5)]


def week_key_stats(X: pd.DataFrame, y: pd.Series, cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """(n_k, sum excess, sum excess^2, rows) for every key of one matured cross-section. y is the raw forward return; the week's
    mean is removed here so a market-wide move cannot look like a pattern."""
    ye = (y.reindex(X.index).astype(float) - float(y.mean())).to_numpy()
    ye = np.nan_to_num(ye)
    codes = quintile_codes(X, cols)
    ind = (codes[:, :, None] == np.arange(5)[None, None, :]).reshape(len(X), -1).astype(np.float64)
    return ind.sum(0), ind.T @ ye, ind.T @ (ye * ye), len(X)


def cell_z(rec: dict, min_n: float = 5.0) -> np.ndarray:
    """z-score of every key's mean excess return in one week's record (NaN where the key had too few rows)."""
    n, s, ss = rec["n"], rec["s"], rec["ss"]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(n > 0, s / n, 0.0)
        var = np.maximum(np.where(n > 0, ss / n - mean ** 2, 0.0), 1e-12)
        z = mean / np.sqrt(var / np.maximum(n, 1))
    return np.where(n >= min_n, z, np.nan)


def estimate_rerun_weight(prev: list, cur: list, fallback: float, min_cells: int = 200, w_min: float = 0.02) -> tuple:
    """Effective extra information of a rerun, from the between-run correlation of the evidence (pseudo-replication).
    For the same real week, the per-key z-scores of two runs are correlated because they share the year's returns. Two estimates whose
    noise correlates by rho carry (1+rho)/2 of the variance of one, so a rerun adds (1-rho)/(1+rho) of a fresh sample. rho is measured on
    the keys that show no signal in either run (|pooled t| < 2 over the run), so the year's true patterns, which are SHARED signal and not
    shared noise, do not inflate it. Fewer than `min_cells` matched cells -> (fallback, nan): not enough to estimate, the stated constant."""
    a = {pd.Timestamp(r["obs"]): r for r in prev}
    pairs = [(a[pd.Timestamp(r["obs"])], r) for r in cur if pd.Timestamp(r["obs"]) in a]
    if len(pairs) < 4:
        return fallback, float("nan")
    za = np.array([cell_z(p) for p, _ in pairs])
    zb = np.array([cell_z(c) for _, c in pairs])
    with np.errstate(invalid="ignore"):
        ta = np.nansum(za, 0) / np.sqrt(np.maximum(np.sum(~np.isnan(za), 0), 1))
        tb = np.nansum(zb, 0) / np.sqrt(np.maximum(np.sum(~np.isnan(zb), 0), 1))
    quiet = (np.abs(ta) < 2.0) & (np.abs(tb) < 2.0)
    u, v = za[:, quiet].ravel(), zb[:, quiet].ravel()
    ok = ~(np.isnan(u) | np.isnan(v))
    if ok.sum() < min_cells or np.std(u[ok]) < 1e-12 or np.std(v[ok]) < 1e-12:
        return fallback, float("nan")
    rho = float(np.clip(np.corrcoef(u[ok], v[ok])[0, 1], 0.0, 0.999))
    return float(np.clip((1.0 - rho) / (1.0 + rho), w_min, 1.0)), rho


class KeyEvidence:
    """Ledger of week-level key sums with the C58 date filter built in.

    Records of the CURRENT run are usable as soon as they are added (an outcome that has just matured). Records of PRIOR runs are
    usable only once their own mature date is <= the real 'now' being decided, so run k+1 at week t never sees run k's week t or
    later. `advance` refuses to move real time backwards inside a run."""

    def __init__(self, n_keys: int, prior_weight: float | str = 1.0, fallback_weight: float = 0.35):
        auto = prior_weight == "auto"
        if not auto and not 0.0 < float(prior_weight) <= 1.0:
            raise ValueError("prior_weight must be in (0, 1] or 'auto'")
        if not 0.0 < fallback_weight <= 1.0:
            raise ValueError("fallback_weight must be in (0, 1]")
        self.K = int(n_keys)
        self.auto = auto
        self.fallback_weight = float(fallback_weight)
        # a rerun of a week is not an independent sample of it: prior-run evidence counts for less. 'auto' estimates how much less from the
        # between-run correlation of the evidence (see estimate_rerun_weight); the fallback applies until two finished runs exist.
        self.prior_weight = self.fallback_weight if auto else float(prior_weight)
        self.rho_history: list = []
        self._prev_cur: list = []
        self._prior = {"mature": np.zeros(0, "datetime64[ns]"), "obs": np.zeros(0, "datetime64[ns]"), "n": np.zeros((0, self.K)),
                       "s": np.zeros((0, self.K)), "ss": np.zeros((0, self.K))}
        self._cur: list = []
        self._ptr = 0
        self.N, self.S, self.SS = (np.zeros(self.K) for _ in range(3))
        self.now: pd.Timestamp | None = None
        self.newest_obs: pd.Timestamp | None = None
        self.used_prior = 0

    def __len__(self):
        return len(self._prior["mature"]) + len(self._cur)

    def prior_table(self) -> pd.DataFrame:
        """The prior-run evidence as a dated table (obs_real / mature_real), the shape the audited TimeGate in learning_delta serves."""
        return pd.DataFrame({"obs_real": pd.to_datetime(self._prior["obs"]), "mature_real": pd.to_datetime(self._prior["mature"]),
                             "k": np.arange(len(self._prior["obs"]))})

    def prior_count(self) -> int:
        """How many prior-run records have been released to the estimates so far in this run."""
        return int(self._ptr)

    def begin_run(self):
        """Fold the finished run into the prior set (sorted by mature date) and reset the running sums."""
        if self.auto and self._cur and self._prev_cur:
            w, rho = estimate_rerun_weight(self._prev_cur, self._cur, self.fallback_weight)
            self.prior_weight = w
            self.rho_history.append(rho)
        if self._cur:
            self._prev_cur = list(self._cur)
            add = {k: np.array([r[k] for r in self._cur]) for k in ("mature", "obs", "n", "s", "ss")}
            add["mature"] = add["mature"].astype("datetime64[ns]")
            add["obs"] = add["obs"].astype("datetime64[ns]")
            merged = {k: np.concatenate([self._prior[k], add[k]]) for k in self._prior}
            order = np.argsort(merged["mature"], kind="stable")
            self._prior = {k: v[order] for k, v in merged.items()}
        self._cur = []
        self._ptr = 0
        self.N, self.S, self.SS = (np.zeros(self.K) for _ in range(3))
        self.now, self.newest_obs, self.used_prior = None, None, 0

    def advance(self, now_real: pd.Timestamp):
        now = pd.Timestamp(now_real)
        if self.now is not None and now < self.now:
            raise FirewallBreach(f"real time moved backwards inside a run: {now.date()} < {self.now.date()}")
        self.now = now
        m = self._prior["mature"]
        end = int(np.searchsorted(m, np.datetime64(now), side="right"))     # mature <= now
        if end > self._ptr:
            sl = slice(self._ptr, end)
            self.N += self.prior_weight * self._prior["n"][sl].sum(0)
            self.S += self.prior_weight * self._prior["s"][sl].sum(0)
            self.SS += self.prior_weight * self._prior["ss"][sl].sum(0)
            top = pd.Timestamp(self._prior["obs"][sl].max())
            self.newest_obs = top if self.newest_obs is None else max(self.newest_obs, top)
            self.used_prior += end - self._ptr
            self._ptr = end

    def add(self, obs_real: pd.Timestamp, n: np.ndarray, s: np.ndarray, ss: np.ndarray):
        obs = pd.Timestamp(obs_real)
        mature = obs + pd.Timedelta(days=HORIZON_DAYS)
        if self.now is not None and mature > self.now:
            raise FirewallBreach(f"evidence maturing {mature.date()} added before it is known (now={self.now.date()})")
        self._cur.append({"mature": mature.to_datetime64(), "obs": obs.to_datetime64(), "n": n, "s": s, "ss": ss})
        self.N += n
        self.S += s
        self.SS += ss
        self.newest_obs = obs if self.newest_obs is None else max(self.newest_obs, obs)

    def estimates(self) -> tuple[np.ndarray, np.ndarray]:
        """(mean excess return, t) per key from the records visible now."""
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = np.where(self.N > 0, self.S / self.N, 0.0)
            var = np.maximum(np.where(self.N > 0, self.SS / self.N - mean ** 2, 0.0), 1e-12)
            t = np.where(self.N > 1, mean / np.sqrt(var / self.N), 0.0)
        return mean, t

    def current_records(self) -> list:
        return list(self._cur)

    def evidence(self):
        return self


class PatternMemoryGate:
    """Wraps engine.pattern_memory.PatternMemory as a reliability veto. It records one observation per (key, month) at the end of
    every run and, at decision time, `vetoed(real_now)` names the keys its C58/C59/C60 view has DISREGARDED for a reason that means
    'used to work and stopped' (recent failure, break inside the year, lapse). It can only switch a key off; it never invents one."""

    def __init__(self, keys: list[str], params: Mapping | None = None):
        self._tmp = tempfile.TemporaryDirectory(prefix="w7_samey_")
        self.mem = PM.PatternMemory(self._tmp.name, params={"use_fdr": False, **(params or {})})
        self.keys = list(keys)
        self._cache: tuple | None = None

    def record_run(self, run_id: str, records: list, min_n: int = 30):
        by_month: dict = {}
        for r in records:
            d = pd.Timestamp(r["obs"])
            m = by_month.setdefault((d.year, d.month), {"d": d, "n": np.zeros(len(self.keys)), "s": np.zeros(len(self.keys)),
                                                       "ss": np.zeros(len(self.keys))})
            m["d"] = max(m["d"], d)
            m["n"] += r["n"]; m["s"] += r["s"]; m["ss"] += r["ss"]
        obs = []
        for _, m in sorted(by_month.items()):
            for i, k in enumerate(self.keys):
                n = m["n"][i]
                if n < min_n:
                    continue
                mean = m["s"][i] / n
                var = max(m["ss"][i] / n - mean * mean, 1e-12)
                obs.append({"key": k, "obs_date": m["d"], "effect": float(mean), "n": int(n), "t": float(mean / np.sqrt(var / n))})
        if not obs:
            return 0
        last = max(o["obs_date"] for o in obs)
        return self.mem.add_observations(run_id, last + pd.Timedelta(days=14), obs)

    def vetoed(self, real_now: pd.Timestamp) -> set:
        self.mem.refresh()
        n_mat = len(self.mem._matured(real_now))
        if self._cache is not None and self._cache[0] == n_mat:
            return self._cache[1]
        if n_mat == 0:
            veto = set()
        else:
            view = self.mem.view(real_now)
            veto = {k for k, w in view.weights.items() if w.mode == "disregarded" and w.reason in VETO_REASONS}
        self._cache = (n_mat, veto)
        return veto

    def audit(self) -> dict:
        return dict(self.mem.last_audit)


# ---------------------------------------------------------------------------------------------------------------
# the shared interface
# ---------------------------------------------------------------------------------------------------------------
class Control:
    """fit / observe / decide with the run bookkeeping the harness needs. Subclasses override only what they use."""
    letter = "?"
    name = "control"
    expects_breach = False          # E is designed to be caught; everything else raising FirewallBreach is a defect
    keeps_identity = False          # True: the control stores things keyed by ticker/date/exact numbers

    def __init__(self, config: Mapping | None = None, seed: int = 0):
        self.config = dict(config or {})
        self.seed = int(seed)
        self.run_id = -1
        self.gate = MemoryGate(self.letter)

    # -- lifecycle
    def begin_run(self, run_id: int) -> None:
        self.run_id = int(run_id)

    def end_run(self, run_id: int) -> None:
        return None

    def fit(self, X: pd.DataFrame, y: pd.Series, moment: Moment) -> None:
        """Warm start from a matured panel (date, ticker). Default: replay it week by week through observe()."""
        dates = X.index.get_level_values(0).unique()
        for i, d in enumerate(dates):
            Xd, yd = X.xs(d, level=0), y.xs(d, level=0) if len(y) else y
            self.observe(Xd, yd, Moment(pd.Timestamp(d), -1 - i, pd.Timestamp(d)), moment)

    def observe(self, X: pd.DataFrame, y: pd.Series, moment: Moment, now: Moment) -> None:
        """One matured cross-section (features of `moment`, their realised forward return) is now known at `now`."""
        return None

    def decide(self, X: pd.DataFrame, moment: Moment) -> pd.Series:
        raise NotImplementedError

    def attach_oracle(self, oracle: Callable | None) -> None:
        return None

    # -- audit surface
    def seen_through(self) -> pd.Timestamp | None:
        """Real observation date of the newest outcome this control has used for its CURRENT decision (None = no experience)."""
        return None

    def state_size(self) -> int:
        return 0

    def evidence(self):
        """The KeyEvidence ledger behind this control's date filter, or None if it keeps none (used by the time-gate referee)."""
        return getattr(self, "ev", None)

    def describe(self) -> dict:
        return {"letter": self.letter, "name": self.name, "state_size": self.state_size(), "config": dict(self.config)}


def check_interface(obj) -> None:
    missing = [m for m in ("begin_run", "end_run", "fit", "observe", "decide", "seen_through", "state_size") if not callable(getattr(obj, m, None))]
    if missing:
        raise ControlContractError(f"{type(obj).__name__} lacks {missing}")


def _series(X: pd.DataFrame, values) -> pd.Series:
    return pd.Series(np.asarray(values, float), index=X.index)


class NoLearning(Control):
    """A. Decisions never depend on experience: a lottery seeded by (seed, week), independent of names and of anything observed."""
    letter, name = "A", "no_learning"

    def decide(self, X, moment):
        g = np.random.default_rng([self.seed, abs(int(moment.week)), 0xA])
        return _series(X, g.random(len(X)))


class EvidenceLearner(Control):
    """The default legitimate learner: shrunk (feature, quintile) effect estimates from evidence that passed the date filter, keys
    below a multiple-testing t bar ignored, keys the PatternMemory has disregarded switched off. Carries memory across runs."""
    letter, name = "B", "evidence_learner"

    DEFAULTS = {"t_thr": 2.5, "min_n": 30, "use_memory_veto": True, "shrink": 1.0, "prior_weight": "auto", "prior_weight_fallback": 0.35}

    def __init__(self, config: Mapping | None = None, seed: int = 0):
        super().__init__({**self.DEFAULTS, **(config or {})}, seed)
        self.cols: list[str] | None = None
        self.names: list[str] = []
        self.ev: KeyEvidence | None = None
        self.pmg: PatternMemoryGate | None = None
        self._used_obs: pd.Timestamp | None = None

    def _init(self, X):
        if self.cols is None:
            self.cols = feature_columns(X)
            self.names = key_names(self.cols)
            self.ev = KeyEvidence(len(self.names), self.config["prior_weight"], self.config["prior_weight_fallback"])
            self.pmg = PatternMemoryGate(self.names) if self.config["use_memory_veto"] else None

    def begin_run(self, run_id):
        super().begin_run(run_id)
        if self.ev is not None:
            self.ev.begin_run()

    def end_run(self, run_id):
        if self.pmg is not None and self.ev is not None:
            self.pmg.record_run(f"run{run_id}", self.ev.current_records(), int(self.config["min_n"]))

    def state_size(self):
        return 0 if self.ev is None else len(self.ev)

    def update_vector(self, X, y) -> np.ndarray:
        """B's raw per-key update from one matured week (the quantity D matches in magnitude)."""
        n, s, _, _ = week_key_stats(X, y, self.cols)
        return np.where(n > 0, s / np.maximum(n, 1), 0.0)

    def observe(self, X, y, moment, now):
        self._init(X)
        if len(X) == 0:
            return
        obs = moment.real(self.gate)
        now_real = now.real(self.gate)
        self.ev.advance(now_real)
        n, s, ss, _ = week_key_stats(X, y, self.cols)
        self.ev.add(obs, n, s, ss)

    def _weights(self, real_now) -> np.ndarray:
        mean, t = self.ev.estimates()
        ok = (np.abs(t) >= self.config["t_thr"]) & (self.ev.N >= self.config["min_n"])
        w = np.where(ok, mean * self.config["shrink"], 0.0)
        if self.pmg is not None and ok.any():
            veto = self.pmg.vetoed(real_now)
            if veto:
                w = np.where([k in veto for k in self.names], 0.0, w)
        return w

    def decide(self, X, moment):
        self._init(X)
        real_now = moment.real(self.gate)
        self.ev.advance(real_now)
        self._used_obs = self.ev.newest_obs
        if len(self.ev) == 0 or len(X) == 0:
            return _series(X, np.zeros(len(X)))
        w = self._weights(real_now).reshape(len(self.cols), 5)
        codes = quintile_codes(X, self.cols)
        return _series(X, w[np.arange(len(self.cols))[None, :], codes].sum(1))

    def seen_through(self):
        return self._used_obs

    def active_keys(self, real_now) -> list[str]:
        """Keys with non-zero weight at `real_now` (owner-side diagnostic used by the recovery tests)."""
        if self.ev is None:
            return []
        self.ev.advance(real_now)
        return [k for k, w in zip(self.names, self._weights(real_now)) if w != 0.0]


class LegitimateLearner(Control):
    """B, pluggable. `factory(config, seed) -> Control-like` builds the learner; the default is EvidenceLearner."""
    letter, name = "B", "legitimate_learner"
    FROZEN_WITH = (EvidenceLearner,)                 # the default inner learner is part of what is frozen

    def __init__(self, config: Mapping | None = None, seed: int = 0, factory: Callable | None = None):
        super().__init__(config, seed)
        self.inner = (factory or EvidenceLearner)(self.config, seed)
        check_interface(self.inner)

    def __getattr__(self, item):
        if item in ("inner", "config", "seed"):
            raise AttributeError(item)
        return getattr(self.inner, item)

    def begin_run(self, run_id):
        super().begin_run(run_id); self.inner.begin_run(run_id)

    def end_run(self, run_id):
        self.inner.end_run(run_id)

    def fit(self, X, y, moment):
        self.inner.fit(X, y, moment)

    def observe(self, X, y, moment, now):
        self.inner.observe(X, y, moment, now)

    def decide(self, X, moment):
        return self.inner.decide(X, moment)

    def seen_through(self):
        return self.inner.seen_through()

    def state_size(self):
        return self.inner.state_size()

    def evidence(self):
        return self.inner.evidence()


class IdentityMemoriser(Control):
    """C. Stores the realised excess return of every row it has seen and replays it. mode 'ticker_date' keys on (code, disguised
    date); 'row_hash' keys on the row's exact numbers (recognises a year that keeps its numbers even under new codes). It ignores
    the C58 filter on purpose: it may recall a week that lies after 'now' in a previous run -- that is the point of the control."""
    letter, name = "C", "identity_memoriser"
    keeps_identity = True

    def __init__(self, config: Mapping | None = None, seed: int = 0):
        super().__init__({"mode": "ticker_date", "decimals": 5, **(config or {})}, seed)
        if self.config["mode"] not in ("ticker_date", "row_hash"):
            raise ControlContractError(f"mode {self.config['mode']!r} not ticker_date|row_hash")
        self.store: dict = {}
        self.hits = 0
        self.lookups = 0

    def _keys(self, X, moment):
        if self.config["mode"] == "ticker_date":
            d = pd.Timestamp(moment.disguised).value
            return [(str(t), d) for t in X.index]
        arr = np.round(X.to_numpy(dtype=np.float64), int(self.config["decimals"]))
        return [row.tobytes() for row in arr]

    def state_size(self):
        return len(self.store)

    def observe(self, X, y, moment, now):
        if len(X) == 0:
            return
        ye = y.reindex(X.index).astype(float) - float(y.mean())
        for k, v in zip(self._keys(X, moment), np.nan_to_num(ye.to_numpy())):
            self.store[k] = float(v)

    def decide(self, X, moment):
        keys = self._keys(X, moment)
        got = np.array([self.store.get(k, np.nan) for k in keys])
        self.lookups += len(keys)
        self.hits += int(np.isfinite(got).sum())
        g = np.random.default_rng([self.seed, abs(int(moment.week)), 0xC])
        return _series(X, np.where(np.isfinite(got), got, 0.0) + g.random(len(X)) * 1e-9)

    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0


class RandomLearner(Control):
    """D. Same update path and the same update MAGNITUDE as B's raw per-key update, in a random direction. If B's curve came from
    'the learner changed by a lot', D would show it too; it must show none."""
    letter, name = "D", "random_learner"

    def __init__(self, config: Mapping | None = None, seed: int = 0):
        super().__init__({"lr": 1.0, **(config or {})}, seed)
        self.cols: list[str] | None = None
        self.w: np.ndarray | None = None
        self._rng = np.random.default_rng([self.seed, 0xD])
        self.updates = 0
        self.mass = 0.0

    def state_size(self):
        return self.updates

    def observe(self, X, y, moment, now):
        if len(X) == 0:
            return
        if self.cols is None:
            self.cols = feature_columns(X)
            self.w = np.zeros(len(self.cols) * 5)
        n, s, _, _ = week_key_stats(X, y, self.cols)
        g = np.where(n > 0, s / np.maximum(n, 1), 0.0)
        r = self._rng.standard_normal(g.size)
        norm = float(np.linalg.norm(g))
        self.w += self.config["lr"] * r * (norm / max(float(np.linalg.norm(r)), 1e-12))
        self.updates += 1
        self.mass += norm

    def decide(self, X, moment):
        if self.w is None or len(X) == 0:
            g = np.random.default_rng([self.seed, abs(int(moment.week)), 0xD])
            return _series(X, g.random(len(X)) * 1e-9)
        codes = quintile_codes(X, self.cols)
        return _series(X, self.w.reshape(len(self.cols), 5)[np.arange(len(self.cols))[None, :], codes].sum(1))


class LeakyLearner(Control):
    """E. Ranks by the realised outcome of the very week it is deciding, obtained from an oracle the harness attaches. It needs no
    learning to look brilliant, which is exactly what makes it distinguishable from B (skill at run 1, no curve) and what the
    future firewall must catch (its newest outcome is dated at 'now', not before it)."""
    letter, name = "E", "leaky_learner"
    expects_breach = True

    def __init__(self, config: Mapping | None = None, seed: int = 0):
        super().__init__({"noise": 0.5, **(config or {})}, seed)
        self._oracle: Callable | None = None
        self._peek: pd.Timestamp | None = None

    def attach_oracle(self, oracle):
        self._oracle = oracle

    def decide(self, X, moment):
        if self._oracle is None:
            raise ControlContractError("the leaky control was built without a leak channel")
        fut, obs_real = self._oracle(moment)
        self._peek = pd.Timestamp(obs_real)
        f = fut.reindex(X.index).astype(float).fillna(0.0).to_numpy()
        g = np.random.default_rng([self.seed, abs(int(moment.week)), 0xE])
        sd = float(np.std(f)) or 1.0
        return _series(X, f + g.standard_normal(len(X)) * sd * self.config["noise"])

    def seen_through(self):
        return self._peek


CONTROL_CLASSES: dict = {"A": NoLearning, "B": LegitimateLearner, "C": IdentityMemoriser, "D": RandomLearner, "E": LeakyLearner}


# ---------------------------------------------------------------------------------------------------------------
# freezing (L01-L05)
# ---------------------------------------------------------------------------------------------------------------
def _code_bits(code: types.CodeType) -> list:
    consts = [_code_bits(c) if isinstance(c, types.CodeType) else repr(c) for c in code.co_consts]
    return [code.co_code.hex(), consts, list(code.co_names), list(code.co_varnames)]


def _class_bits(base, controls_only: bool) -> list:
    parts = []
    for b in base.__mro__:
        if b is object or (controls_only and not issubclass(b, Control)):
            continue
        for name, val in sorted(vars(b).items()):
            fn = val.__func__ if isinstance(val, (classmethod, staticmethod)) else val
            if isinstance(fn, types.FunctionType):
                parts.append([b.__name__, name, _code_bits(fn.__code__), repr(fn.__defaults__)])
            elif name in ("letter", "name", "expects_breach", "keeps_identity", "DEFAULTS"):
                parts.append([b.__name__, name, repr(val)])
    return parts


def foreign_fingerprint(obj) -> str:
    """Fingerprint of a plug-in (a class or function from any module), so a pluggable learner is frozen with its control."""
    if obj is None:
        return ""
    if isinstance(obj, types.FunctionType):
        return stable_hash(_code_bits(obj.__code__), 24)
    return stable_hash(_class_bits(obj, False), 24)


def code_fingerprint(cls, factory: Callable | None = None) -> str:
    """Hash of the bytecode of every function the class defines or inherits from this module (plus the module helpers it calls and
    any plug-in), so patching a method at run time changes it, not only editing the file."""
    parts = _class_bits(cls, True)
    for extra in getattr(cls, "FROZEN_WITH", ()):
        parts.append(_class_bits(extra, True))
    for helper in (week_key_stats, quintile_codes, feature_columns, key_names, cell_z, estimate_rerun_weight, KeyEvidence.advance, KeyEvidence.add,
                   KeyEvidence.begin_run, KeyEvidence.estimates, PatternMemoryGate.vetoed, PatternMemoryGate.record_run):
        parts.append([helper.__name__, _code_bits(helper.__code__)])
    parts.append(foreign_fingerprint(factory))
    return stable_hash(parts, 24)


def config_fingerprint(config: Mapping) -> str:
    return stable_hash(dict(config), 24)


@dataclass(frozen=True)
class FrozenRecord:
    letter: str
    name: str
    version: int
    code_hash: str
    config_hash: str
    config_json: str
    note: str = ""

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


class FrozenControl:
    """A control class + config as frozen at a moment. `build(seed)` re-verifies both hashes every time."""

    def __init__(self, cls, config: Mapping | None = None, version: int = 1, note: str = "", factory: Callable | None = None):
        if cls.letter not in LETTERS:
            raise ControlContractError(f"{cls.__name__} has no control letter")
        cfg = json.loads(json.dumps(dict(config or {}), sort_keys=True))          # detached copy: later mutation cannot reach it
        self.cls, self.factory = cls, factory
        self.record = FrozenRecord(cls.letter, cls.name, int(version), code_fingerprint(cls, factory), config_fingerprint(cfg),
                                   json.dumps(cfg, sort_keys=True), note)

    @property
    def config(self) -> dict:
        return json.loads(self.record.config_json)

    def verify(self) -> None:
        rec = self.record
        if code_fingerprint(self.cls, self.factory) != rec.code_hash:
            raise ControlChanged(f"control {rec.letter} ({rec.name}): code changed since it was frozen")
        if config_fingerprint(self.config) != rec.config_hash:
            raise ControlChanged(f"control {rec.letter} ({rec.name}): config no longer matches its hash")

    def build(self, seed: int) -> Control:
        self.verify()
        obj = self.cls(self.config, seed, factory=self.factory) if self.factory else self.cls(self.config, seed)
        check_interface(obj)
        return obj


class ControlRegistry:
    """Persistent record of the frozen versions. `verify` refuses (ControlChanged) when a control's code or config differs from the
    latest recorded version; a deliberate change goes through `supersede`, which appends and never overwrites."""

    def __init__(self, path: str | os.PathLike | None = None):
        self.path = None if path is None else os.fspath(path)
        self.history: list[dict] = []
        if self.path and os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as fh:
                self.history = json.load(fh)

    def latest(self, letter: str) -> dict | None:
        rows = [r for r in self.history if r["letter"] == letter]
        return max(rows, key=lambda r: r["version"]) if rows else None

    def _save(self):
        if self.path:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(self.history, fh, indent=1, sort_keys=True)

    def register(self, fc: FrozenControl) -> bool:
        """First use freezes it; a later use must match. True if newly frozen."""
        cur = self.latest(fc.record.letter)
        if cur is None:
            self.history.append(fc.record.as_dict())
            self._save()
            return True
        self.verify(fc)
        return False

    def verify(self, fc: FrozenControl) -> None:
        fc.verify()
        cur = self.latest(fc.record.letter)
        if cur is None:
            raise ControlChanged(f"control {fc.record.letter} was never frozen")
        for f in ("code_hash", "config_hash"):
            if cur[f] != getattr(fc.record, f):
                raise ControlChanged(f"control {fc.record.letter}: {f} differs from frozen version {cur['version']}")

    def supersede(self, fc: FrozenControl, reason: str) -> FrozenControl:
        if not reason.strip():
            raise ControlContractError("superseding a frozen control needs a written reason")
        cur = self.latest(fc.record.letter)
        nxt = FrozenControl(fc.cls, fc.config, (cur["version"] + 1) if cur else 1, reason, fc.factory)
        self.history.append(nxt.record.as_dict())
        self._save()
        return nxt


def standard_controls(seed_free: bool = True, overrides: Mapping | None = None, legit_factory: Callable | None = None) -> dict:
    """The five controls exactly as frozen for the same-year experiment."""
    ov = overrides or {}
    cfg = {"A": {}, "B": {"t_thr": 2.5, "min_n": 30, "use_memory_veto": True}, "C": {"mode": "ticker_date"}, "D": {"lr": 1.0},
           "E": {"noise": 0.5}}
    return {L: FrozenControl(CONTROL_CLASSES[L], {**cfg[L], **ov.get(L, {})}, factory=legit_factory if L == "B" else None)
            for L in LETTERS}
