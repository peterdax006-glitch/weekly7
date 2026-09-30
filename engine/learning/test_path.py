"""The Test path of canon C64: a blind trader, a knowing curator (contract C62 sections 3, 4, 28-30, 55, 87).
IMPLEMENTED - NOT VALIDATED (C63: wiring and unit tests only; no result is claimed).

C64: while a window runs, the system does not know the year and receives information day by day, as it became available; what it
learns is filed under the real year it operated in; which memory matters now is decided by a separate system that DOES know the
year, out of the trader's sight.  Two halves, one file, one rule between them:

  TRUSTED (knows the real date)                          TRADER (blind)
  PathRunner.on_tick   once per session ------------->   PathTrader.act(TraderDay, DayData)  -> DayResult
    Curator.day / release   (curator.py)                   learner.decide_batch / resolve_and_learn  (learner.py)
    OutcomeBook   next-open returns, matured strictly      MemoryAdvisor  reads ONLY the release the curator handed over
                  before today, from the hardened feed
    MemoryFiler   files what the learner learned under the real year, after the outcome matured

The trader is handed exactly two things per session: the curator's `TraderDay` (a run-relative step, today's market state as
z-scores of its own past, and a weighted, date-free release of memory) and `DayData` (the hardened feed's rows for today, the outcomes
that matured strictly before today, the disguised clock).  It never receives a store, a curator, a real date or a reason.  A static
check (`trader_side_violations`) walks the AST of the trader classes and refuses any reference to curator symbols;
trader_view.assert_trader_path_clean covers the import closure of livesim / learner / adaptive, which must never reach this file.

The learner runs as a SHADOW decider in this wave: its section-3 decisions are recorded and it learns, while the adaptive.Session
keeps trading (so the re-tester, future-scramble and fill-audit gates of scripts/livesim_loop2.py still replay the traded path
exactly).  Letting its picks trade is a Stage-3 decision, taken only after the learning curve says so (canon C54/C55)."""
from __future__ import annotations

import ast
import inspect
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .core import FirewallBreach, TemporalClass, as_date, stable_hash
from .curator import Curator, RelevanceConfig
from .learner import EpisodeSummary, LearnerConfig, LegitimateLearner
from . import research_policy as RP
from . import wiring as W
from .loop_hooks import RunGuard, append_curve_record, curve_report, run_same_year_harness, safe_token
from .trader_view import (CURATOR_SYMBOLS, STORE_MARKERS, TraderDay, TraderRelease, assert_trader_safe, find_violations, opaque_token,
                          release_year_hits)

LABEL = "IMPLEMENTED - NOT VALIDATED"
M_PREFIX = "m_"
RESERVED_KEYS = ("level", "strength_band")               # feature keys of a released pattern that are not the pattern's feature
ADVICE_MODES = ("off", "record", "veto")
DEFAULT_FEATURES = ("r5", "r20", "r60", "mom_12_1", "dist_ma50", "dist_ma200", "dist_52wh", "vol20", "vol_ratio", "atr_pct",
                    "range_compress", "log_dv", "vol_surge1", "vol_surge5", "max20", "min20", "skew60", "frog", "gap_today",
                    "ind_mom20", "ind_mom60", "rel_ind20", "rel_ind60", "days_since_earn", "news5")


@dataclass(frozen=True)
class PathConfig:
    """Every knob of the test path in one frozen record; a changed config is a different run."""
    horizon_sessions: int = 5            # outcome: open of the session after the decision -> open this many sessions later
    release_k: int = 8                   # memories the curator may hand over per decision day
    release_daily: bool = False          # False: the clock advances every session, a release is served on decision days only
    advice: str = "record"               # off | record (vote is logged, picks unchanged) | veto (strongly opposed picks are dropped)
    veto_vote: float = -0.5              # a pick whose memory vote is below this is dropped in veto mode
    sample_names: int | None = 400       # fixed hash-ordered subset of code names shown to the learner (cost), None = all
    min_rows: int = 40                   # a decision day with fewer rows is skipped and counted
    file_min_abs_t: float = 1.0          # a weekly pattern estimate is filed only if |edge / se| reaches this
    file_max_per_week: int = 10          # ... and only the strongest few per decision week
    preroll_days: int = 120              # warm-up sessions replayed through the curator's clock so today's state has a past
    seed: int = 0
    store_root: str | None = None        # curator store (one hash-chained lane per real year); None keeps it in memory
    features: tuple[str, ...] = DEFAULT_FEATURES
    min_feature_coverage: float = 0.9    # a candidate feature must be observed on this share of the warm-up rows to be shown at all
    check_path: bool = True              # run trader_view.assert_trader_path_clean when a runner is built
    learn_root: str | None = None        # where the learner persists what it learns (None: <store_root>/../loop, or a temp dir)
    checkpoint_every: int = 20           # sessions between progress checkpoints of the run (checkpoints.CheckpointStore)
    harness_runs: int = 0                # >0: after the window, run the same-year harness this many runs as a REAL worker process

    def validate(self) -> list[str]:
        errs = []
        if self.horizon_sessions < 1:
            errs.append("horizon_sessions < 1")
        if self.release_k < 1:
            errs.append("release_k < 1")
        if self.advice not in ADVICE_MODES:
            errs.append(f"advice must be one of {ADVICE_MODES}")
        if not -1.0 <= self.veto_vote <= 0.0:
            errs.append("veto_vote must lie in [-1, 0]")
        if self.sample_names is not None and self.sample_names < 2:
            errs.append("sample_names < 2 cannot rank a cross-section")
        if self.min_rows < 2 or self.file_max_per_week < 1 or self.preroll_days < 0 or self.file_min_abs_t < 0:
            errs.append("min_rows >= 2, file_max_per_week >= 1, preroll_days >= 0, file_min_abs_t >= 0")
        if not self.features:
            errs.append("no candidate features")
        if not 0.0 < self.min_feature_coverage <= 1.0:
            errs.append("min_feature_coverage outside (0, 1]")
        if self.checkpoint_every < 1 or self.harness_runs < 0:
            errs.append("checkpoint_every >= 1 and harness_runs >= 0")
        return errs

    def digest(self) -> str:
        return stable_hash(self)


def choose_features(history: pd.DataFrame, cfg: PathConfig, tail_rows: int = 5000) -> tuple[str, ...]:
    """The configured candidate features that the warm-up rows actually observe (>= min_feature_coverage) and that vary. A column
    the feed cannot fill would trip the learner's DATA firewall on the first day, so it is left out up front and reported."""
    cols = [c for c in cfg.features if c in history.columns]
    tail = history[cols].tail(tail_rows)
    keep = []
    for c in cols:
        x = tail[c]
        if len(x) and x.notna().mean() >= cfg.min_feature_coverage and x.nunique() > 1:
            keep.append(c)
    return tuple(keep)


def make_learner(cfg: PathConfig, workdir: str | Path | None = None, code_hash_fn: Callable[[], str] | None = None,
                 features: Sequence[str] | None = None, **learner_kw: Any) -> LegitimateLearner:
    """The learner a window uses: the section-4 conductor over this path's candidate features, deterministic given cfg.seed."""
    lc = LearnerConfig(seed=cfg.seed, horizon_days=7, candidate_features=tuple(features if features is not None else cfg.features),
                       top_n=5, min_cs_n=20, min_coverage=0.05, **learner_kw)
    return LegitimateLearner(lc, workdir=workdir, code_hash_fn=code_hash_fn)


# ------------------------------------------------------------------------------------------------ what crosses to the trader
@dataclass(frozen=True)
class DayData:
    """The hardened feed's information for one session, in the feed's own (disguised) clock. `panel` is today's cross-section on a
    decision day, else None; `outcomes` holds only labels that matured strictly before `now`."""
    now: pd.Timestamp
    panel: pd.DataFrame | None = None
    outcomes: pd.DataFrame | None = None

    def check(self) -> None:
        if self.panel is not None and len(self.panel):
            d = pd.DatetimeIndex(self.panel.index.get_level_values(0))
            if d.min() != d.max() or d.max() != pd.Timestamp(self.now):
                raise FirewallBreach(f"panel is not the cross-section of {self.now.date()} alone")
        if self.outcomes is not None and len(self.outcomes):
            if not {"ret", "matured"} <= set(self.outcomes.columns):
                raise FirewallBreach("outcomes need columns ret and matured")
            late = self.outcomes["matured"] >= pd.Timestamp(self.now)
            if bool(late.any()):
                raise FirewallBreach(f"{int(late.sum())} outcomes have not matured strictly before {self.now.date()}")


@dataclass(frozen=True)
class DayResult:
    step: int
    decided: bool
    n_rows: int = 0
    picks: tuple = ()                    # (private row key, size) of the shadow LONG picks that survived the advisor
    n_long: int = 0                      # learner LONG rows before the advisor
    vetoed: int = 0
    learned: tuple = ()                  # EpisodeSummary of every episode resolved today
    unresolved: tuple = ()               # decision dates (feed clock) of episodes dropped for lack of outcomes
    n_memory: int = 0                    # items in today's release
    mean_vote: float | None = None
    skipped: str = ""                    # why no decision was made on a decision day


# ------------------------------------------------------------------------------------------------ trader side: memory advice
class MemoryAdvisor:
    """Turns the curator's release into one number per row. A released pattern says `feature is in quantile level L -> lean`; the
    row's vote is the sum over items of weight x strength_band x lean over the items whose (feature, level) the row matches.
    Weights sum to 1 and bands lie in [0,1], so a vote lies in [-1, 1]; a lone weak memory cannot read as strong because its
    band is small (the release weight alone would be 1.0)."""

    def __init__(self, n_quantiles: int = 5):
        if n_quantiles < 2:
            raise ValueError("n_quantiles < 2")
        self.n_q = n_quantiles

    @staticmethod
    def pattern_of(item) -> tuple[str, int, float] | None:
        """(feature, level, band) of a released pattern item, or None if it does not describe a (feature, level) cell."""
        f = item.features
        names = [k for k in f if k not in RESERVED_KEYS]
        if item.kind != "pattern" or len(names) != 1 or "level" not in f:
            return None
        return names[0], int(round(float(f["level"]))), float(f.get("strength_band", 1.0))

    def levels(self, panel: pd.DataFrame, feature: str) -> pd.Series:
        """Cross-sectional quantile level of `feature` in today's panel (-1 where unobserved): the learner's own definition."""
        x = panel[feature].astype(float)
        n = int(x.count())
        if n < 2:
            return pd.Series(-1, index=panel.index)
        r = x.rank(method="first")
        lv = (((r - 1) * self.n_q) // n).clip(0, self.n_q - 1)
        return lv.where(x.notna(), -1).astype(int)

    def votes(self, release: TraderRelease, panel: pd.DataFrame) -> pd.Series:
        out = pd.Series(0.0, index=panel.index)
        for item in release.items:
            pat = self.pattern_of(item)
            if pat is None or pat[0] not in panel.columns:
                continue
            feat, level, band = pat
            out += (self.levels(panel, feat) == level).astype(float) * (item.weight * band * item.lean)
        return out.clip(-1.0, 1.0)


class PathTrader:
    """The blind decider. `act` is its only entry: a TraderDay and a DayData in, a DayResult out."""

    def __init__(self, learner: LegitimateLearner, cfg: PathConfig):
        self.learner, self.cfg = learner, cfg
        self.advisor = MemoryAdvisor(learner.cfg.n_quantiles)
        self.days = 0

    def _resolve_ready(self, now: pd.Timestamp, outcomes: pd.DataFrame | None) -> tuple[list[EpisodeSummary], list[str]]:
        learned: list[EpisodeSummary] = []
        dropped: list[str] = []
        L = self.learner
        if outcomes is None or not len(outcomes):
            return learned, dropped
        for eid in sorted(L.pending):
            ep = L.pending[eid]
            if not L.ready(ep, now):
                continue
            have = sum(r.key in outcomes.index for r in ep.rows)
            if have < max(2, len(ep.rows) // 2):          # learn from a biased remainder? no: drop the episode, say so
                del L.pending[eid]
                dropped.append(str(as_date(ep.now)))
                continue
            learned.append(L.resolve_and_learn(ep, now, outcomes))
        return learned, dropped

    def act(self, day: TraderDay, data: DayData) -> DayResult:
        assert_trader_safe(day.to_dict(), "TraderDay")
        data.check()
        self.days += 1
        step, release = day.situation.step, day.release
        if data.panel is None:
            return DayResult(step, False, n_memory=len(release))
        panel = data.panel
        if len(panel) < self.cfg.min_rows:
            return DayResult(step, True, len(panel), n_memory=len(release), skipped=f"only {len(panel)} rows")
        learned, dropped = self._resolve_ready(data.now, data.outcomes)
        ep = self.learner.decide_batch(data.now, panel)
        longs = self.learner.picks(ep)
        votes = self.advisor.votes(release, panel) if self.cfg.advice != "off" else pd.Series(0.0, index=panel.index)
        keep, vetoed = [], 0
        for key, dec in longs:
            v = float(votes.get(key, 0.0)) if key in votes.index else 0.0
            if self.cfg.advice == "veto" and v < self.cfg.veto_vote:
                vetoed += 1
                continue
            keep.append((key, dec.size))
        mean_vote = float(np.mean([votes.get(k, 0.0) for k, _ in longs])) if longs and self.cfg.advice != "off" else None
        return DayResult(step, True, len(panel), tuple(keep), len(longs), vetoed, tuple(learned), tuple(dropped), len(release), mean_vote)


TRADER_CLASSES = ("DayData", "DayResult", "MemoryAdvisor", "PathTrader")


def trader_side_violations(source: str | None = None) -> list[str]:
    """AST check of the trader half of THIS file: no reference to a curator symbol, a curator store marker, or the trusted classes.
    `source` defaults to this module's own text (tests pass a doctored copy to prove a leak is caught)."""
    src = inspect.getsource(sys.modules[__name__]) if source is None else source
    tree = ast.parse(src)
    banned = set(CURATOR_SYMBOLS) | {"PathRunner", "MemoryFiler", "OutcomeBook", "RealClock"}
    found = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in TRADER_CLASSES:
            for n in ast.walk(node):
                if isinstance(n, ast.Name) and n.id in banned:
                    found.append(f"{node.name}: uses {n.id} (line {n.lineno})")
                elif isinstance(n, ast.Attribute) and n.attr in banned:
                    found.append(f"{node.name}: uses .{n.attr} (line {n.lineno})")
                elif isinstance(n, ast.Constant) and isinstance(n.value, str) and any(m in n.value.lower() for m in STORE_MARKERS):
                    found.append(f"{node.name}: names the curator's store (line {n.lineno})")
    return found


# ------------------------------------------------------------------------------------------------ trusted side
@dataclass(frozen=True)
class RealClock:
    """Disguised feed date <-> real date. Only the trusted runner holds one; the shift is the feed's secret."""
    shift: pd.Timedelta

    def real(self, disguised) -> pd.Timestamp:
        return pd.Timestamp(disguised) - self.shift

    def disguised(self, real) -> pd.Timestamp:
        return pd.Timestamp(real) + self.shift


class OutcomeBook:
    """Labels for the decisions the trader has made, computed from the feed's own served prices. A decision on session i fills at
    the open of session i+1 and is scored at the open of session i+1+h; the label is served only once session i+1+h is strictly
    before today, and each decision's labels are served until the trader reports the episode resolved."""

    def __init__(self, horizon: int):
        self.h = int(horizon)
        self._open: dict[pd.Timestamp, pd.MultiIndex] = {}

    def record(self, day: pd.Timestamp, keys: pd.MultiIndex) -> None:
        self._open[pd.Timestamp(day)] = keys

    def consume(self, day) -> None:
        self._open.pop(pd.Timestamp(day), None)

    def __len__(self) -> int:
        return len(self._open)

    def outcomes(self, sessions: pd.DatetimeIndex, opens: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame | None:
        """`opens` are the Open prices the feed has shown up to `now` (positions align with `sessions`)."""
        pos_now = sessions.get_loc(now)
        frames = []
        for day, keys in sorted(self._open.items()):
            i0 = sessions.get_loc(day)
            e, x = i0 + 1, i0 + 1 + self.h
            if x >= pos_now:
                continue                                    # the exit open is today or later: not yet knowable as of today's close
            codes = [k[1] for k in keys]
            px_e = opens.iloc[e].reindex(codes).to_numpy(dtype=float)
            px_x = opens.iloc[x].reindex(codes).to_numpy(dtype=float)
            with np.errstate(all="ignore"):
                ret = px_x / px_e - 1.0
            ok = np.isfinite(ret)
            if ok.any():
                frames.append(pd.DataFrame({"ret": ret[ok], "matured": sessions[x]}, index=keys[ok]))
        return pd.concat(frames) if frames else None


class MemoryFiler:
    """Files what the learner learned into the curator under the REAL year it operated in, after the outcome matured. One memory
    per (pattern, decision week) whose estimate is strong enough, capped per week; the item is the opaque, date-free thing a
    later trader may be shown, the context is the m_* market state on the decision day (trusted side only)."""

    def __init__(self, curator: Curator, clock: RealClock, cfg: PathConfig):
        self.curator, self.clock, self.cfg = curator, clock, cfg
        self._cursor: dict[str, int] = {}
        self.filed = self.refused = self.no_state = self.orphans = self.weak = 0
        self.by_year: dict[int, int] = {}

    def _new_entries(self, learner: LegitimateLearner) -> dict[str, list]:
        new = {}
        for pid, wk in learner._weekly.items():
            start = self._cursor.get(pid, 0)
            if len(wk) > start:
                new[pid] = wk[start:]
            self._cursor[pid] = len(wk)
        return new

    def file(self, learner: LegitimateLearner, summaries: Sequence[EpisodeSummary], states: Mapping[str, Mapping[str, float]]) -> int:
        new = self._new_entries(learner)
        total = 0
        used = 0
        for s in summaries:
            if not s.learned or s.matured_on is None:
                continue
            cand = []
            for pid, entries in new.items():
                for mat, est, se, n in entries:
                    if str(mat) == str(s.matured_on):
                        used += 1
                        t = est / se if se > 0 else 0.0
                        if abs(t) >= self.cfg.file_min_abs_t:
                            cand.append((abs(t), pid, est, t, n))
                        else:
                            self.weak += 1
            state = states.get(str(s.decided_on))
            if not state:
                self.no_state += len(cand)
                continue
            real_d, real_m = self.clock.real(s.decided_on), self.clock.real(s.matured_on)
            for _, pid, est, t, n in sorted(cand, reverse=True)[: self.cfg.file_max_per_week]:
                total += self._file_one(pid, t, real_d, real_m, state)
        self.orphans += sum(len(v) for v in new.values()) - used
        return total

    def _file_one(self, pid: str, t: float, real_d: pd.Timestamp, real_m: pd.Timestamp, state: Mapping[str, float]) -> int:
        feature, _, level = pid.rpartition(":q")
        feats = {feature: 1.0, "level": float(int(level))}
        item_id = opaque_token({"pattern": pid})
        if find_violations({"item_id": item_id, "features": feats}):
            self.refused += 1                              # a feature name that reads as a date/rank/reason cannot cross the wall
            return 0
        z = abs(t) / math.sqrt(2.0)
        reliability = float(min(0.95, max(0.05, math.erf(z))))
        self.curator.file({"item_id": item_id, "kind": "pattern", "weight": 1.0, "features": feats,
                           "lean": float(math.tanh(t / 2.0)), "horizon": self.cfg.horizon_sessions},
                          real_d, real_d.year, state, matured_at=real_m, reliability=reliability, calibration_error=0.0,
                          temporal=TemporalClass.UNKNOWN)
        self.filed += 1
        self.by_year[real_d.year] = self.by_year.get(real_d.year, 0) + 1
        return 1


class PathRunner:
    """Drives the test path over a livesim.Feed (normally the hardened one). Build it when the warm-up ends and the clock is about
    to start; call `on_tick` once per session (livesim.run does, through its hook); read `report()` and `findings()` at the end."""

    def __init__(self, feed, cfg: PathConfig | None = None, curator: Curator | None = None, learner: LegitimateLearner | None = None,
                 learner_workdir: str | Path | None = None, code_hash_fn: Callable[[], str] | None = None):
        self.cfg = cfg or PathConfig()
        errs = self.cfg.validate()
        if errs:
            raise ValueError("invalid PathConfig: " + "; ".join(errs))
        if self.cfg.check_path:
            from .trader_view import assert_trader_path_clean
            assert_trader_path_clean()
        self.feed = feed
        self.clock = RealClock(pd.Timedelta(feed._shift))
        self.curator = curator or Curator(store_root=self.cfg.store_root, config=RelevanceConfig(default_k=self.cfg.release_k),
                                          code_hash=code_hash_fn() if code_hash_fn else None)
        hist, _ = feed.features_until_now()
        self.features = tuple(learner.cfg.candidate_features) if learner is not None and learner.cfg.candidate_features             else choose_features(hist, self.cfg)
        if not self.features:
            raise ValueError("no candidate feature is observed on the warm-up rows: the learner would see nothing")
        self.dropped_features = tuple(c for c in self.cfg.features if c not in self.features)
        self.run_key = safe_token(stable_hash([self.cfg.digest(), str(getattr(getattr(feed, "_sealed", None), "run_id", "")),
                                               str(pd.Timestamp(feed.now).date())], 12))
        self.learn_root = self._learn_root(learner_workdir)
        self.learner = learner or make_learner(self.cfg, self.learn_root, code_hash_fn, features=self.features)
        # section 58 around the run: resume only same-code progress, checkpoint on a cadence, an interruption record on failure,
        # and the run described as a compute Job so a scheduler can relaunch it in its own process (trusted side only)
        self.guard = None
        if self.learn_root is not None:
            self.guard = RunGuard(Path(self.learn_root) / "run", self.run_key, self.learner.code_hash, self.cfg.checkpoint_every)
            self.guard.job("test_path", {"config": self.cfg.digest(), "run": self.run_key}, self.cfg.seed, feed.now)
            self.resume = self.guard.start(feed.now)
        self._finalised = False
        self.hub_periods = 0
        self.curve: dict = {}
        if W.HUB.enabled:                                  # production hub (livesim_loop2 --learner legit): audit on the learner's board
            W.configure(board=self.learner.board)
        self.trader = PathTrader(self.learner, self.cfg)
        self.book = OutcomeBook(self.cfg.horizon_sessions)
        self.filer = MemoryFiler(self.curator, self.clock, self.cfg)
        self._states: dict[str, dict[str, float]] = {}
        self.results: list[DayResult] = []
        self.counts = {"days": 0, "decision_days": 0, "skipped_days": 0, "learned_episodes": 0, "unresolved_episodes": 0,
                       "releases": 0, "empty_releases": 0, "picks": 0, "vetoed": 0}
        self.release_hits: dict[str, int] = {}
        self._names: pd.Index | None = None
        self.preroll_days = self._preroll()

    def _learn_root(self, workdir) -> str | None:
        """The learner's persistent home: explicit workdir > cfg.learn_root > a sibling of the curator store. Named by an opaque
        run key (digits mapped to letters) so no folder on the trader's side can name a year."""
        if workdir is not None:
            return str(workdir)
        if self.cfg.learn_root is not None:
            return str(Path(self.cfg.learn_root) / self.run_key)
        if self.cfg.store_root is not None:
            return str(Path(self.cfg.store_root).parent / "loop" / self.run_key)
        return None

    def same_year(self, world, n_runs: int = 6, seed: int | None = None) -> dict:
        """The same_year / controls harness (C54/C55) as an entry the runner can call on a planted world or a window source."""
        return run_same_year_harness(world, n_runs=n_runs, seed=self.cfg.seed if seed is None else seed)

    # ---- inputs
    def _m_state(self, X: pd.DataFrame) -> dict[str, float]:
        cols = [c for c in X.columns if str(c).startswith(M_PREFIX)]
        if not cols or not len(X):
            return {}
        row = X[cols].iloc[0]
        return {c: float(v) for c, v in row.items() if np.isfinite(v)}

    def _preroll(self) -> int:
        """Replay the last warm-up sessions through the curator's clock (state only, no release) so today's market state has a past."""
        if self.cfg.preroll_days == 0:
            return 0
        X, _ = self.feed.features_until_now()
        cols = [c for c in X.columns if str(c).startswith(M_PREFIX)]
        if not cols:
            return 0
        daily = X[cols].groupby(level=0).first().tail(self.cfg.preroll_days)
        n = 0
        for d, row in daily.iterrows():
            state = {c: float(v) for c, v in row.items() if np.isfinite(v)}
            self.curator.day(self.clock.real(d), state)
            n += 1
        return n

    def _panel(self) -> pd.DataFrame | None:
        X = self.feed.features_today()
        if not len(X):
            return None
        X = X[[c for c in X.columns if c in self.features or str(c).startswith(M_PREFIX)]]
        if self.cfg.sample_names is not None:
            if self._names is None:
                codes = sorted(set(X.index.get_level_values(1)), key=lambda c: (stable_hash([c, self.cfg.seed], 12), c))
                self._names = pd.Index(codes[: self.cfg.sample_names])
            X = X[X.index.get_level_values(1).isin(self._names)]
        return X

    # ---- the daily clock
    def on_tick(self) -> DayResult:
        try:
            res = self._on_tick()
        except BaseException as e:                     # write what section 58 demands, then fail as before (never swallowed)
            if self.guard is not None:
                self.guard.interrupted(self.feed.now, f"{type(e).__name__}: {e}")
            raise
        if self.guard is not None:
            self.guard.tick(self.feed.now, self.counts)
        return res

    def _on_tick(self) -> DayResult:
        feed = self.feed
        now = pd.Timestamp(feed.now)
        Xtoday = feed.features_today()
        state = self._m_state(Xtoday)
        real = self.clock.real(now)
        decide = bool(feed.next_session_is_new_week()) and not feed.done()
        if decide or self.cfg.release_daily:
            sit, rel = self.curator.run_day(real, state, self.cfg.release_k)
            self.counts["releases"] += 1
            self.counts["empty_releases"] += int(len(rel) == 0)
            for k, v in release_year_hits(rel).items():
                self.release_hits[k] = self.release_hits.get(k, 0) + v
        else:
            sit = self.curator.day(real, state)
            rel = TraderRelease(max(self.curator.step, 0), ())
        day = TraderDay(sit, rel)
        panel = self._panel() if decide else None
        outcomes = None
        if decide:
            outcomes = self.book.outcomes(feed.sessions, feed.history()[0]["Open"], now)
        self._states[str(now.date())] = state
        res = self.trader.act(day, DayData(now, panel, outcomes))
        self._after(res, now, panel)
        return res

    def _after(self, res: DayResult, now: pd.Timestamp, panel: pd.DataFrame | None) -> None:
        c = self.counts
        c["days"] += 1
        c["decision_days"] += int(res.decided)
        c["skipped_days"] += int(res.decided and bool(res.skipped))
        c["picks"] += len(res.picks)
        c["vetoed"] += res.vetoed
        for s in res.learned:
            self.book.consume(s.decided_on)
        for d in res.unresolved:
            self.book.consume(d)
        c["learned_episodes"] += len(res.learned)
        c["unresolved_episodes"] += len(res.unresolved)
        if res.decided and not res.skipped and panel is not None:
            self.book.record(now, panel.index)
        if res.learned:
            self.filer.file(self.learner, res.learned, self._states)
            self._hub_period(now)
        self.results.append(res)

    def _hub_period(self, now: pd.Timestamp) -> None:
        """The hub's S20 period loop, once per learned period: its contradiction monitor runs over the hub graph, its open
        contradictions go through research_step into the learner's own research engine (one queue, one ledger), the plan is closed
        by the loop's proposer, and the dashboard inputs are written beside the learner's store. Inert while the hub is off."""
        rep = W.on_period(now)
        if rep is None:
            return
        L = self.learner
        assert L.experiments is not None
        step = W.research_step(rep, L.research, L.experiments, RP.ComputeBudget(cpu_minutes=L.cfg.cpu_minutes, ram_gb_free=8.0), now,
                               self.cfg.seed)
        L.hooks.close_research(step, L.last_learned_on or str(now.date()), now)
        if self.learn_root is not None:
            W.write_dashboard_inputs(rep, Path(self.learn_root) / "hub_dashboard")
        self.hub_periods += 1

    # ---- audit and report
    def findings(self) -> list:
        """blind_gates.Finding list: any 'fail' excludes the window. Trader path clean, no date-like hit in any release, the curator's
        year-lane chains intact, the release never named the store."""
        from .. import blind_gates as BG
        out = []
        try:
            from .trader_view import assert_trader_path_clean
            assert_trader_path_clean()
        except FirewallBreach as e:
            out.append(BG.Finding("legit-trader-path", "fail", str(e)))
        bad = trader_side_violations()
        if bad:
            out.append(BG.Finding("legit-trader-class", "fail", "; ".join(bad[:3])))
        if self.release_hits:
            out.append(BG.Finding("legit-release-dates", "fail", f"date-like content in releases: {self.release_hits}"))
        v = self.curator.store.verify()
        if not v["ok"]:
            out.append(BG.Finding("legit-store-chain", "fail", "curator store chain does not verify"))
        a = self.curator.audit_verify()
        if not a.get("ok", True):
            out.append(BG.Finding("legit-audit-chain", "fail", "curator audit chain does not verify"))
        if self.results and not self.counts["decision_days"]:
            out.append(BG.Finding("legit-no-decisions", "warn", "the learner never had a decision day"))
        return out

    def finalise(self) -> dict:
        """Persist everything the learner learned and write a closing checkpoint; idempotent, called by report()."""
        if self._finalised:
            return {}
        self._finalised = True
        out = {}
        if self.learner.last_learned_on is not None:
            out = self.learner.hooks.maybe_persist(pd.Timestamp(self.feed.now), force=True)
        if self.guard is not None:
            self.guard.store.save(str(pd.Timestamp(self.feed.now).date()), "test_path", "done: read the report", self.learner.code_hash,
                                  notes={"day": str(self.guard.n), "final": "1"})
        if self.learn_root is not None:                    # one curve record per run, appended to the lineage's file
            L = self.learner
            longs = [s.mean_edge_long for s in L.summaries if s.learned and s.mean_edge_long is not None]
            recs = append_curve_record(Path(self.learn_root).parent / "curve_records.jsonl",
                                       {"tag": "legit", "run": self.run_key, "config": self.cfg.digest(), "mean_week": float(np.mean(longs)) if longs else 0.0,
                                        "weeks": len(longs), "knowledge": len(L._pid_of), "production": len(L.production_ids())})
            self.curve = curve_report(recs)
            if self.cfg.harness_runs and self.guard is not None:
                self.curve["harness"] = self.guard.run_harness_process({"n_runs": self.cfg.harness_runs}, self.cfg.seed, self.feed.now)
        return out

    def report(self) -> dict:
        self.finalise()
        L = self.learner
        rep = L.report()
        picks_by_day = [len(r.picks) for r in self.results if r.decided and not r.skipped]
        votes = [r.mean_vote for r in self.results if r.mean_vote is not None]
        return {"label": LABEL, "mode": "legit", "config_hash": self.cfg.digest(), "advice": self.cfg.advice,
                "preroll_days": self.preroll_days, "counts": dict(self.counts), "features": list(self.features),
                "dropped_features": list(self.dropped_features),
                "curator_items_visible": self.curator.visible_count() if self.curator.now is not None else 0,
                "curator_years_filed": {str(y): n for y, n in self.filer.by_year.items()},
                "filed": self.filer.filed, "refused_at_the_wall": self.filer.refused, "weak_not_filed": self.filer.weak,
                "no_state": self.filer.no_state, "orphans": self.filer.orphans,
                "mean_shadow_picks_per_decision": float(np.mean(picks_by_day)) if picks_by_day else 0.0,
                "mean_memory_vote_on_picks": float(np.mean(votes)) if votes else None,
                "release_date_hits": dict(self.release_hits),
                "learner": {"episodes": rep["episodes"], "learned": rep["learned"], "knowledge": rep["knowledge"],
                            "skill": {k: v for k, v in dict(rep["skill"]).items() if isinstance(v, (int, float, str, bool, type(None)))},
                            "refusals": len(rep["refusals"]), "influence_log_ok": rep["influence_log_ok"],
                            "hooks": {k: v for k, v in rep["hooks"].items() if k in ("fired", "rows", "open_experiments", "persisted")}},
                "run": None if self.guard is None else {"key": self.run_key, "resume": self.resume, "job": self.guard.spec.key if self.guard.spec else None,
                                                        "checkpoints": len(self.guard.store.sequences())},
                "hub_periods": self.hub_periods, "curve": dict(self.curve),
                "findings": [f"{f.severity}:{f.gate}:{f.message}" for f in self.findings()]}


def hook_factory(cfg: PathConfig | None = None, **kw) -> Callable:
    """The callable livesim.run(hook_factory=...) expects: (feed, trader) -> a runner with on_tick(). Trusted scripts build it;
    engine/livesim.py never imports this module (the trader import closure must not reach the curator)."""
    def make(feed, trader):
        return PathRunner(feed, cfg, **kw)
    return make
