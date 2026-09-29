"""The final two-stage prediction architecture (contract C66 section 35; also sections 0, 10, 11, 29-31, 34; canon C64, C66).
IMPLEMENTED - NOT VALIDATED (C63: code and unit tests on planted worlds only).

    ALL ELIGIBLE STOCKS -> POINT-IN-TIME FEATURE STATE -> P(volatility) -> VOLATILITY RANK -> PREDICTED-MOVER UNIVERSE
    -> P(up | predicted mover), P(down | predicted mover) -> CALIBRATION -> FAILURE / REGIME / PATTERN HEALTH -> RISK FILTER
    -> POSITION / ABSTENTION

This module is the ORCHESTRATOR of that chain; it builds no fifth mover model and no new direction learner (RESEARCH_MAPPING design
risk 2): P(volatility) is engine.research.volatility_lab.VolatilityModel (the lab's champion form, fitted on matured rows only,
isotonic-calibrated on walk-forward scores), the direction gate is engine.research.direction_lab.DirectionGate over
direction_lab.assess_volatility_evidence (direction only after volatility has out-of-sample evidence), and the direction learner is
engine.direction_features.DirModel. What this module adds is the section-35 funnel as one auditable object:

  * the point-in-time state is built explicitly (`point_in_time_state`): rows of the decision date only, outcome columns removed and
    recorded; `decide` REFUSES a frame that still carries outcome columns or rows dated after `now` (fail closed, never trimmed);
  * direction is trained and evaluated ONLY on predicted movers - the movers a walk-forward volatility model would have picked, never
    the realised movers - and `evaluate` computes every direction metric on predicted movers alone;
  * knowledge reaches the pipeline only through the research/trader firewall (engine.research.firewall -> engine.learning.trader_view
    TraderRelease) or a matured record's `gate(now)`: `KnowledgeView` refuses anything else. Released 'pattern' items add volatility
    or direction features, 'lesson' items with a negative lean become risk rules, 'context' items with a negative lean are regime
    health vetoes;
  * every stage records what it removed and why (`Funnel`), so an abstention is always explained.

Public entry: `run_day(pipe, history, today, now, release=None)`; evaluation: `evaluate(decisions, matured, now)`."""
from __future__ import annotations

import dataclasses
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from engine.direction import brier, ece, wilson
from engine.learning import trader_view as TV
from engine.pattern_movers import auc as rank_auc
from engine.research import vol_hypotheses as VH
from engine.research.core import FirewallBreach, MaturedRecord, Namespace, _StrEnum, as_date, require_past, stable_hash

LABEL = "IMPLEMENTED - NOT VALIDATED"
OUTCOME_COLUMNS = tuple(VH.OUTCOME_COLUMNS)
FORBIDDEN_PREFIXES = ("fwd_", "y_", "future_", "label_")
# derived features that need the market's own past (expanding percentiles): a one-day point-in-time frame cannot compute them, and
# computing them on the research panel would read dates the decision day cannot see, so the decision chain does not use them
HISTORY_DEPENDENT = frozenset({"mkt_stress", "lv20_x_stress"})
LONG, SHORT, FLAT = 1, -1, 0


class Reason(_StrEnum):                    # why a name did not become a position
    INELIGIBLE = "INELIGIBLE"
    NO_VOL_MODEL = "NO_VOL_MODEL"
    NOT_PREDICTED_MOVER = "NOT_PREDICTED_MOVER"
    DIRECTION_GATE_CLOSED = "DIRECTION_GATE_CLOSED"
    NO_DIRECTION_MODEL = "NO_DIRECTION_MODEL"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    REGIME_UNHEALTHY = "REGIME_UNHEALTHY"
    RISK_TAIL = "RISK_TAIL"
    RISK_RULE = "RISK_RULE"
    SECTOR_CAP = "SECTOR_CAP"
    POSITION_CAP = "POSITION_CAP"
    SHORT_DISABLED = "SHORT_DISABLED"
    POSITION = "POSITION"


@dataclasses.dataclass(frozen=True)
class TwoStageConfig:
    base_features: tuple = ("lv20", "latr", "vol_ratio_short", "shock1", "xs_vol_rank", "dv_level", "compress", "mkt_vol")
    direction_features: tuple = ("rel_r5", "rel_r20", "near_hi", "near_lo", "gap_ratio", "shock1")
    min_log_dv: float | None = None          # eligibility: log dollar volume floor (None = no floor)
    min_logp: float | None = None            # eligibility: log price floor
    top_frac: float = 0.10                   # predicted-mover universe: this share of eligible names per day ...
    min_movers: int = 5                      # ... but at least this many ...
    max_movers: int = 50                     # ... and at most this many
    p_move_floor: float = 0.0
    p_long: float = 0.55                     # calibrated P(up | mover) needed for a long
    p_short: float = 0.55                    # calibrated P(down | mover) needed for a short
    allow_short: bool = False
    max_positions: int = 10
    max_per_sector: int = 3
    max_tail: float = 0.60                   # mag_q90 above this is a tail the risk filter will not hold
    wf_split: float = 0.5                    # share of matured dates used to fit the walk-forward volatility model
    calib_frac: float = 0.3                  # last share of predicted-mover training rows used only for calibration
    min_direction_rows: int = 120
    min_calib_rows: int = 40
    gate_min_weeks: int = 12
    gate_min_auc_lo: float = 0.52
    gate_min_lift_lo: float = 1.15
    dir_kind: str = "linear"
    seed: int = 0

    def validate(self) -> list[str]:
        errs = []
        bad = [f for f in (*self.base_features, *self.direction_features) if f not in VH.DERIVED]
        if bad:
            errs.append(f"unknown derived features {bad}")
        if not 0 < self.top_frac < 1 or self.min_movers < 1 or self.max_movers < self.min_movers:
            errs.append("top_frac in (0,1) and 1 <= min_movers <= max_movers required")
        if not (0.5 <= self.p_long < 1 and 0.5 <= self.p_short < 1):
            errs.append("p_long/p_short in [0.5,1) required")
        if not 0.2 <= self.wf_split <= 0.8 or not 0.1 <= self.calib_frac <= 0.5:
            errs.append("wf_split in [0.2,0.8] and calib_frac in [0.1,0.5] required")
        if self.max_positions < 1 or self.max_per_sector < 1:
            errs.append("position caps must be >= 1")
        return errs


# ------------------------------------------------------------------------------------------------ point-in-time state
@dataclasses.dataclass(frozen=True)
class PITState:
    decided_at: str
    frame: pd.DataFrame
    dropped_columns: tuple
    n_rows: int


def _forbidden(cols: Iterable[str]) -> list[str]:
    return [c for c in cols if c in OUTCOME_COLUMNS or str(c).startswith(FORBIDDEN_PREFIXES)]


def point_in_time_state(frame: pd.DataFrame, now) -> PITState:
    """The trader's view of the decision day: the newest date <= now, outcome columns removed (and named). A frame with no row on or
    before `now` yields an empty state."""
    if len(frame) == 0:
        return PITState(as_date(now).isoformat(), frame.iloc[0:0], tuple(_forbidden(frame.columns)), 0)
    dts = pd.to_datetime(frame.index.get_level_values(0))
    ok = dts <= pd.Timestamp(as_date(now))
    if not ok.any():
        return PITState(as_date(now).isoformat(), frame.iloc[0:0], tuple(_forbidden(frame.columns)), 0)
    day = dts[ok].max()
    sub = frame[np.asarray(dts == day)]
    drop = tuple(_forbidden(sub.columns))
    return PITState(day.date().isoformat(), sub.drop(columns=list(drop)), drop, len(sub))


def assert_point_in_time(frame: pd.DataFrame, now) -> None:
    """Fail closed: a decision frame may not carry outcome columns or rows dated after `now`."""
    bad = _forbidden(frame.columns)
    if bad:
        raise FirewallBreach(f"decision frame carries outcome/future columns {bad[:6]}")
    if len(frame):
        mx = pd.to_datetime(frame.index.get_level_values(0)).max()
        if mx > pd.Timestamp(as_date(now)):
            raise FirewallBreach(f"decision frame has rows dated {mx.date()} after now={as_date(now)}")


# ------------------------------------------------------------------------------------------------ knowledge through the firewall
@dataclasses.dataclass(frozen=True)
class RiskRule:
    """A released lesson: rows whose features all sit at or beyond the thresholds (in cross-sectional z units) are not held."""
    features: Mapping[str, float]
    weight: float

    def mask(self, D: pd.DataFrame) -> np.ndarray:
        m = np.ones(len(D), bool)
        for f, thr in self.features.items():
            if f not in D:
                return np.zeros(len(D), bool)
            v = D[f].to_numpy(float)
            z = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-12)
            m &= (z >= thr) if thr >= 0 else (z <= thr)
        return m


@dataclasses.dataclass(frozen=True)
class KnowledgeView:
    """What the decision chain may use from research, already through the firewall. Built only by `from_release` (a TraderRelease) or
    `from_records` (matured records gated at `now`)."""
    vol_features: tuple = ()
    direction_features: tuple = ()
    risk_rules: tuple = ()
    unhealthy_contexts: tuple = ()           # tuple of feature->threshold mappings on market columns
    n_items: int = 0
    digest: str = ""

    @staticmethod
    def empty() -> "KnowledgeView":
        return KnowledgeView(digest="none")

    @staticmethod
    def from_release(release: Any) -> "KnowledgeView":
        if release is None:
            return KnowledgeView.empty()
        if not isinstance(release, TV.TraderRelease):
            raise FirewallBreach(f"knowledge must arrive as a firewall TraderRelease, got {type(release).__name__}")
        return KnowledgeView._from_items([i.to_dict() for i in release.items])

    @staticmethod
    def from_records(records: Sequence[MaturedRecord], now) -> "KnowledgeView":
        """Matured research records, each passed through its own point-in-time gate and the trader-view scan (fail closed)."""
        items = []
        for r in records:
            if not isinstance(r, MaturedRecord):
                raise FirewallBreach(f"{type(r).__name__} is not a MaturedRecord")
            p = dict(r.gate(now))
            d = {"kind": p.get("trader_kind", "pattern"), "features": dict(p.get("features") or {}), "lean": float(p.get("lean", 0.0)),
                 "weight": 1.0, "horizon": int(p.get("horizon", 5))}
            TV.assert_trader_safe(d, f"record {r.record_id}")
            items.append(d)
        return KnowledgeView._from_items(items)

    @staticmethod
    def _from_items(items: Sequence[Mapping[str, Any]]) -> "KnowledgeView":
        vol, dirf, rules, ctx = [], [], [], []
        for it in items:
            feats = {str(k): float(v) for k, v in dict(it.get("features") or {}).items()}
            known = [f for f in feats if f in VH.DERIVED]
            lean, kind, w = float(it.get("lean", 0.0)), str(it.get("kind", "pattern")), float(it.get("weight", 1.0))
            if kind == "pattern":
                (dirf if abs(lean) > 0 else vol).extend(known)
            elif kind == "lesson" and lean < 0 and known:
                rules.append(RiskRule({f: feats[f] for f in known}, w))
            elif kind == "context" and lean < 0 and feats:
                ctx.append(dict(feats))
        dg = stable_hash({"v": sorted(set(vol)), "d": sorted(set(dirf)), "r": [dict(r.features) for r in rules], "c": ctx}, 12)
        return KnowledgeView(tuple(sorted(set(vol))), tuple(sorted(set(dirf))), tuple(rules), tuple(ctx), len(items), dg)


# ------------------------------------------------------------------------------------------------ fitting
@dataclasses.dataclass
class FitReport:
    now: str
    trained_through: str
    n_rows: int
    vol_features: tuple
    vol_ok: bool
    wf_rows: int
    gate_open: bool
    gate_reasons: tuple
    vol_evidence: dict
    dir_features: tuple
    dir_ok: bool
    dir_rows: int
    calib_rows: int
    ece_raw: float
    ece_cal: float
    notes: list = dataclasses.field(default_factory=list)


def _ok_features(features: Sequence[str], columns) -> tuple:
    return tuple(f for f in dict.fromkeys(features) if f in VH.DERIVED and f not in HISTORY_DEPENDENT
                 and not VH.missing_columns((f,), columns))


def predicted_movers(scores: pd.Series, cfg: TwoStageConfig) -> pd.Series:
    """Per date: the top max(min_movers, top_frac * n) names by score (capped at max_movers) whose score clears the floor."""
    s = scores.dropna()
    if s.empty:
        return pd.Series(False, index=scores.index)
    out = pd.Series(False, index=scores.index)
    for d, g in s.groupby(level=0, sort=True):
        k = int(min(cfg.max_movers, max(cfg.min_movers, math.ceil(cfg.top_frac * len(g)))))
        top = g[g >= cfg.p_move_floor].sort_values(ascending=False, kind="mergesort").index[:k]
        out.loc[top] = True
    return out


def _platt(raw: np.ndarray, y: np.ndarray):
    """Calibrator fitted on the calibration block only: logistic regression of the label on logit(raw)."""
    from sklearn.linear_model import LogisticRegression
    x = np.log(np.clip(raw, 1e-4, 1 - 1e-4) / (1 - np.clip(raw, 1e-4, 1 - 1e-4)))
    if len(np.unique(y)) < 2 or np.std(x) == 0:
        base = float(np.mean(y)) if len(y) else 0.5
        return lambda r: np.full(len(r), base)
    mu, sd = float(np.mean(x)), float(np.std(x))
    m = LogisticRegression(C=1e3, max_iter=300).fit(((x - mu) / sd)[:, None], y)

    def cal(r):
        z = np.log(np.clip(r, 1e-4, 1 - 1e-4) / (1 - np.clip(r, 1e-4, 1 - 1e-4)))
        return m.predict_proba(((z - mu) / sd)[:, None])[:, 1]
    return cal


class TwoStage:
    """The section-35 pipeline. `fit` learns from matured rows only; `decide` turns one point-in-time day into positions/abstentions."""

    def __init__(self, cfg: TwoStageConfig | None = None, lab_cfg=None, fit_cfg=None):
        from engine.research import volatility_lab as VL
        self.cfg = cfg or TwoStageConfig()
        errs = self.cfg.validate()
        if errs:
            raise ValueError("; ".join(errs))
        self.lab_cfg = lab_cfg or VL.LabConfig()
        self.fit_cfg = fit_cfg or VH.FitConfig(min_rows=200, min_events=20, gbm_trees=40)
        self.vol_model = None
        self.dir_model = None
        self.calibrator = None
        self.gate = None
        self.knowledge = KnowledgeView.empty()
        self.report: FitReport | None = None
        self.dir_cols: tuple = ()

    # -------------------------------------------------------------------------------------- fit
    def fit(self, history: pd.DataFrame, now, knowledge: KnowledgeView | None = None) -> FitReport:
        """Fit the chain on rows whose outcome ended strictly before `now`. The volatility model used for today's decisions is fitted
        on all matured rows; the DIRECTION model is fitted on the predicted movers of a walk-forward volatility model (first
        `wf_split` of dates -> scores on the rest), so direction never sees realised movers as its universe."""
        from engine.direction_features import DirModel
        from engine.research import direction_lab as DL
        from engine.research import volatility_lab as VL
        cfg = self.cfg
        self.knowledge = knowledge or KnowledgeView.empty()
        F, _ = VL.mature_only(history, now) if len(history) else (history, 0)
        F = F[F["touch"].notna()].sort_index() if len(F) else F
        vol_feats = _ok_features((*cfg.base_features, *self.knowledge.vol_features), F.columns)
        dir_feats = _ok_features((*cfg.direction_features, *self.knowledge.direction_features), F.columns)
        rep = FitReport(as_date(now).isoformat(), "", len(F), vol_feats, False, 0, False, (), {}, dir_feats, False, 0, 0,
                        float("nan"), float("nan"))
        self.vol_model = self.dir_model = self.calibrator = None
        self.gate, self.dir_cols = None, dir_feats
        if len(F) == 0 or not vol_feats:
            rep.notes.append("no matured rows or no derivable volatility feature: the chain abstains")
            self.report = rep
            return rep
        hyp = VH.Hypothesis("TS", "two-stage volatility", "the section-35 volatility stage: base features plus released knowledge",
                            vol_feats, origin="two_stage")
        dates = np.array(sorted(pd.unique(F.index.get_level_values(0))))
        cut = dates[int(len(dates) * cfg.wf_split)] if len(dates) >= 4 else None
        oos = None
        if cut is not None:
            ends = pd.to_datetime(F["end"])
            early = F[np.asarray(ends < pd.Timestamp(cut))]
            late = F[np.asarray(pd.to_datetime(F.index.get_level_values(0)) >= pd.Timestamp(cut))]
            if len(early) and len(late):
                mA = VL.VolatilityModel.fit(early, pd.Timestamp(cut), hyp, self.lab_cfg, self.fit_cfg)
                if mA.fitted.ok:
                    pA = mA.forecast(late.drop(columns=_forbidden(late.columns)), late.index.get_level_values(0).max())["p_move"]
                    oos = late.assign(**{"p_TS": pA.to_numpy()})
                    rep.wf_rows = len(oos)
        m = VL.VolatilityModel.fit(F, now, hyp, self.lab_cfg, self.fit_cfg,
                                   calib_oos=oos[["p_TS", "touch", "end"]] if oos is not None else None)
        rep.vol_ok = bool(m.fitted.ok)
        rep.trained_through = m.trained_through
        self.vol_model = m if m.fitted.ok else None
        if not rep.vol_ok:
            rep.notes.append(f"volatility model not fitted: {m.fitted.reason}")
        if oos is None or not len(oos):
            rep.notes.append("no walk-forward block: the direction gate cannot open")
            self.report = rep
            return rep
        ev = DL.assess_volatility_evidence(oos["p_TS"], oos["touch"], n_pick=max(cfg.min_movers, 1), n_boot=200, seed=cfg.seed)
        self.gate = DL.DirectionGate(cfg.gate_min_weeks, cfg.gate_min_auc_lo, cfg.gate_min_lift_lo).assess(ev)
        rep.vol_evidence, rep.gate_open, rep.gate_reasons = ev.to_dict(), self.gate.is_open, tuple(self.gate.reasons)
        mv = predicted_movers(oos["p_TS"], cfg)
        pool = oos[mv.to_numpy()]
        if not dir_feats or len(pool) < cfg.min_direction_rows:
            rep.notes.append(f"{len(pool)} predicted-mover rows (< {cfg.min_direction_rows}) or no direction features: no direction model")
            self.report = rep
            return rep
        X = VH.derive(pool, dir_feats).to_numpy(float)
        y = pool["up"].to_numpy(float)
        pd_dates = pool.index.get_level_values(0)
        cd = np.array(sorted(pd.unique(pd_dates)))
        c_cut = cd[int(len(cd) * (1 - cfg.calib_frac))] if len(cd) >= 3 else cd[-1]
        tr, ca = np.asarray(pd_dates < c_cut), np.asarray(pd_dates >= c_cut)
        if tr.sum() < cfg.min_direction_rows * (1 - cfg.calib_frac) or ca.sum() < cfg.min_calib_rows or len(np.unique(y[tr])) < 2:
            rep.notes.append(f"direction training {int(tr.sum())} / calibration {int(ca.sum())} rows: too few for a calibrated model")
            self.report = rep
            return rep
        dm = DirModel(cfg.dir_kind, cfg.seed).fit(X[tr], y[tr])
        raw = dm.prob(X[ca])
        self.calibrator = _platt(raw, y[ca])
        self.dir_model = dm
        rep.dir_ok, rep.dir_rows, rep.calib_rows = True, int(tr.sum()), int(ca.sum())
        rep.ece_raw, rep.ece_cal = ece(raw, y[ca]), ece(self.calibrator(raw), y[ca])
        self.report = rep
        return rep

    # -------------------------------------------------------------------------------------- decide
    def decide(self, today: pd.DataFrame, now) -> "DayDecision":
        """Run one point-in-time day through the funnel. `today` must already be the PIT state (see point_in_time_state)."""
        assert_point_in_time(today, now)
        cfg = self.cfg
        fun = Funnel()
        n = len(today)
        out = pd.DataFrame(index=today.index)
        out["eligible"], out["p_move"], out["mag_q90"], out["mover"] = False, np.nan, np.nan, False
        out["p_up"], out["p_down"], out["calibrated"], out["side"], out["weight"] = np.nan, np.nan, False, FLAT, 0.0
        out["reason"] = str(Reason.INELIGIBLE)
        fun.add("universe", n, n, "")
        if n == 0:
            return DayDecision(as_date(now).isoformat(), out, fun, self._gate_state(), self.knowledge.digest)
        elig = np.ones(n, bool)
        if cfg.min_log_dv is not None and "log_dv" in today:
            elig &= today["log_dv"].to_numpy(float) >= cfg.min_log_dv
        if cfg.min_logp is not None and "logp" in today:
            elig &= today["logp"].to_numpy(float) >= cfg.min_logp
        feats = self.report.vol_features if self.report else ()
        if feats:
            D = VH.derive(today, feats)
            elig &= np.isfinite(D.to_numpy(float)).all(1)
        out["eligible"] = elig
        fun.add("eligible", n, int(elig.sum()), "liquidity/price floors and derivable features")
        if self.vol_model is None or not elig.any():
            out.loc[out["eligible"].to_numpy(), "reason"] = str(Reason.NO_VOL_MODEL)
            fun.add("p_volatility", int(elig.sum()), 0, "no fitted volatility model")
            return DayDecision(as_date(now).isoformat(), out, fun, self._gate_state(), self.knowledge.digest)
        E = today[elig]
        fc = self.vol_model.forecast(E, now)
        out.loc[E.index, "p_move"] = fc["p_move"].to_numpy()
        out.loc[E.index, "mag_q90"] = fc["mag_q90"].to_numpy() if "mag_q90" in fc else np.nan
        fun.add("p_volatility", int(elig.sum()), int(np.isfinite(fc["p_move"].to_numpy(float)).sum()), "")
        mv = predicted_movers(out.loc[E.index, "p_move"], cfg)
        out.loc[mv.index, "mover"] = mv.to_numpy()
        out.loc[E.index[~mv.to_numpy()], "reason"] = str(Reason.NOT_PREDICTED_MOVER)
        movers = out.index[out["mover"].to_numpy(bool)]
        fun.add("predicted_movers", int(elig.sum()), len(movers), f"top {cfg.top_frac:.0%} by P(volatility)")
        if not len(movers):
            return DayDecision(as_date(now).isoformat(), out, fun, self._gate_state(), self.knowledge.digest)
        if self.gate is None or not self.gate.is_open:
            out.loc[movers, "reason"] = str(Reason.DIRECTION_GATE_CLOSED)
            fun.add("direction", len(movers), 0, "direction gate closed: " + "; ".join(self._gate_state()["reasons"][:2]))
            return DayDecision(as_date(now).isoformat(), out, fun, self._gate_state(), self.knowledge.digest)
        if self.dir_model is None:
            out.loc[movers, "reason"] = str(Reason.NO_DIRECTION_MODEL)
            fun.add("direction", len(movers), 0, "no direction model (too few predicted-mover rows)")
            return DayDecision(as_date(now).isoformat(), out, fun, self._gate_state(), self.knowledge.digest)
        M = today.loc[movers]
        Xd = VH.derive(M, self.dir_cols).to_numpy(float)
        p_up = self.calibrator(self.dir_model.prob(Xd))
        out.loc[movers, "p_up"], out.loc[movers, "p_down"], out.loc[movers, "calibrated"] = p_up, 1.0 - p_up, True
        fun.add("direction", len(movers), len(movers), "P(up | predicted mover) on predicted movers only")
        fun.add("calibration", len(movers), len(movers), f"Platt on the last {cfg.calib_frac:.0%} of predicted-mover training rows")
        alive = pd.Series(True, index=movers)
        if self.knowledge.unhealthy_contexts and self._regime_unhealthy(M):
            alive[:] = False
            out.loc[movers, "reason"] = str(Reason.REGIME_UNHEALTHY)
        fun.add("health", len(movers), int(alive.sum()), "released context vetoes")
        side = np.where(p_up >= cfg.p_long, LONG, np.where(1 - p_up >= cfg.p_short, SHORT, FLAT))
        conf = np.abs(p_up - 0.5)
        for i, ix in enumerate(movers):
            if not alive[ix]:
                continue
            if side[i] == FLAT:
                out.loc[ix, "reason"], alive[ix] = str(Reason.LOW_CONFIDENCE), False
            elif side[i] == SHORT and not cfg.allow_short:
                out.loc[ix, "reason"], alive[ix] = str(Reason.SHORT_DISABLED), False
        tail = out.loc[movers, "mag_q90"].to_numpy(float)
        for i, ix in enumerate(movers):
            if alive[ix] and np.isfinite(tail[i]) and tail[i] > cfg.max_tail:
                out.loc[ix, "reason"], alive[ix] = str(Reason.RISK_TAIL), False
        if self.knowledge.risk_rules:
            Dr = VH.derive(M, sorted({f for r in self.knowledge.risk_rules for f in r.features if f in VH.DERIVED}))
            hit = np.zeros(len(M), bool)
            for r in self.knowledge.risk_rules:
                hit |= r.mask(Dr)
            for i, ix in enumerate(movers):
                if alive[ix] and hit[i]:
                    out.loc[ix, "reason"], alive[ix] = str(Reason.RISK_RULE), False
        order = sorted((i for i, ix in enumerate(movers) if alive[ix]), key=lambda i: (-conf[i], str(movers[i])))
        per_sector: dict = {}
        held = []
        for i in order:
            ix = movers[i]
            sec = M["sector"].iloc[i] if "sector" in M else "all"
            if per_sector.get(sec, 0) >= cfg.max_per_sector:
                out.loc[ix, "reason"] = str(Reason.SECTOR_CAP)
                continue
            if len(held) >= cfg.max_positions:
                out.loc[ix, "reason"] = str(Reason.POSITION_CAP)
                continue
            per_sector[sec] = per_sector.get(sec, 0) + 1
            held.append(i)
        fun.add("risk_filter", int(alive.sum()), len(held), "tail cap, released lessons, sector and position caps")
        for i in held:
            ix = movers[i]
            out.loc[ix, "side"], out.loc[ix, "weight"], out.loc[ix, "reason"] = int(side[i]), 1.0 / len(held), str(Reason.POSITION)
        fun.add("positions", len(movers), len(held), "abstain on the rest (every abstention carries its reason)")
        return DayDecision(as_date(now).isoformat(), out, fun, self._gate_state(), self.knowledge.digest)

    def _regime_unhealthy(self, M: pd.DataFrame) -> bool:
        for ctx in self.knowledge.unhealthy_contexts:
            hit = True
            for col, thr in ctx.items():
                if col not in M:
                    hit = False
                    break
                v = float(np.nanmean(M[col].to_numpy(float)))
                hit &= (v >= thr) if thr >= 0 else (v <= thr)
            if hit:
                return True
        return False

    def _gate_state(self) -> dict:
        if self.gate is None:
            return {"open": False, "verdict": "NO_EVIDENCE", "reasons": ["no walk-forward volatility evidence yet"]}
        return {"open": bool(self.gate.is_open), "verdict": str(self.gate.verdict), "reasons": list(self.gate.reasons)}


@dataclasses.dataclass
class Funnel:
    rows: list = dataclasses.field(default_factory=list)       # (stage, n_in, n_out, note)

    def add(self, stage: str, n_in: int, n_out: int, note: str) -> None:
        self.rows.append((stage, int(n_in), int(n_out), note))

    def as_dict(self) -> dict:
        return {s: {"in": a, "out": b, "note": t} for s, a, b, t in self.rows}


@dataclasses.dataclass
class DayDecision:
    decided_at: str
    table: pd.DataFrame
    funnel: Funnel
    gate: dict
    knowledge_digest: str

    @property
    def positions(self) -> pd.DataFrame:
        return self.table[self.table["side"] != FLAT]

    def reasons(self) -> dict:
        return self.table["reason"].value_counts().to_dict()

    def summary(self) -> dict:
        return {"decided_at": self.decided_at, "n": len(self.table), "movers": int(self.table["mover"].sum()),
                "positions": int((self.table["side"] != FLAT).sum()), "gate_open": self.gate.get("open", False),
                "funnel": self.funnel.as_dict(), "reasons": self.reasons(), "knowledge": self.knowledge_digest}


# ------------------------------------------------------------------------------------------------ public entry
def run_day(pipe: TwoStage, history: pd.DataFrame, today: pd.DataFrame, now, release: Any = None, refit: bool = True) -> DayDecision:
    """PUBLIC ENTRY. `history`: matured research frame (outcomes allowed; only rows whose outcome ended before `now` are used);
    `today`: the point-in-time decision day; `release`: the firewall's TraderRelease (or None)."""
    kv = KnowledgeView.from_release(release)
    if refit or pipe.report is None or kv.digest != pipe.knowledge.digest:
        pipe.fit(history, now, kv)
    return pipe.decide(today, now)


# ------------------------------------------------------------------------------------------------ evaluation
@dataclasses.dataclass(frozen=True)
class Evaluation:
    now: str
    n_days: int
    n_rows: int
    vol_auc: float
    vol_auc_se: float
    mover_precision: float
    base_rate: float
    n_movers: int
    direction_acc: float
    direction_se: float
    n_direction: int
    acc_by_coverage: Mapping[str, Mapping[str, float]]
    best_acc_at_coverage: float
    best_acc_se: float
    best_acc_n: int
    ece_movers: float
    brier_movers: float
    n_positions: int
    mean_pnl: float
    catastrophic_rate: float
    worst_pnl: float
    abstentions: Mapping[str, int]
    matured_through: str

    def to_dict(self) -> dict:
        return json.loads(json.dumps(dataclasses.asdict(self), default=float))


def _nan() -> float:
    return float("nan")


def evaluate(decisions: Sequence[DayDecision], matured: pd.DataFrame, now, coverages: Sequence[float] = (0.05, 0.10, 0.25, 0.5, 1.0),
             catastrophic: float = -0.20, min_coverage: float = 0.05) -> Evaluation:
    """Score decisions against outcomes that matured strictly before `now`. Direction metrics use PREDICTED movers only (non-movers'
    outcomes cannot move them); volatility metrics use every eligible scored name."""
    from engine.research import volatility_lab as VL
    frames = [d.table.assign(_day=d.decided_at) for d in decisions if len(d.table)]
    M, _ = VL.mature_only(matured, now) if len(matured) else (matured, 0)
    if not frames or len(M) == 0:
        return Evaluation(as_date(now).isoformat(), len(decisions), 0, _nan(), _nan(), _nan(), _nan(), 0, _nan(), _nan(), 0, {}, _nan(), _nan(), 0,
                          _nan(), _nan(), 0, _nan(), _nan(), _nan(), {}, "")
    T = pd.concat(frames)
    J = T.join(M[["touch", "up", "close", "end"]], how="inner")
    J = J[J["touch"].notna()]
    scored = J[np.isfinite(J["p_move"].to_numpy(float))]
    aucs = [rank_auc(g["p_move"].to_numpy(float), g["touch"].to_numpy(bool)) for _, g in scored.groupby(level=0)]
    aucs = np.array([a for a in aucs if np.isfinite(a)])
    mv = J[J["mover"].to_numpy(bool)]
    dm = mv[np.isfinite(mv["p_up"].to_numpy(float))]
    y = dm["up"].to_numpy(float)
    p = dm["p_up"].to_numpy(float)
    correct = ((p >= 0.5) == (y >= 0.5)).astype(float) if len(dm) else np.array([])
    acc = float(correct.mean()) if len(correct) else _nan()
    se = float(math.sqrt(acc * (1 - acc) / len(correct))) if len(correct) else _nan()
    bands: dict = {}
    best, best_se, best_n = _nan(), _nan(), 0
    if len(dm):
        order = np.argsort(-np.abs(p - 0.5), kind="mergesort")
        for c in coverages:
            k = max(1, int(round(c * len(order))))
            sel = correct[order[:k]]
            lo, hi = wilson(sel.sum(), len(sel))
            a = float(sel.mean())
            bands[f"{c:.2f}"] = {"n": int(len(sel)), "acc": a, "lo": lo, "hi": hi}
            if c >= min_coverage and (not np.isfinite(best) or a > best):
                best, best_se, best_n = a, float(math.sqrt(max(a * (1 - a), 1e-9) / len(sel))), int(len(sel))
    pos = J[J["side"].to_numpy(int) != FLAT]
    pnl = (pos["side"].to_numpy(float) * pos["close"].to_numpy(float)) if len(pos) else np.array([])
    ab = J[J["side"].to_numpy(int) == FLAT]["reason"].value_counts().to_dict()
    return Evaluation(as_date(now).isoformat(), len(decisions), len(J),
                      float(aucs.mean()) if len(aucs) else _nan(), float(aucs.std(ddof=1) / math.sqrt(len(aucs))) if len(aucs) > 1 else _nan(),
                      float(mv["touch"].mean()) if len(mv) else _nan(), float(scored["touch"].mean()) if len(scored) else _nan(), len(mv),
                      acc, se, int(len(correct)), bands, best, best_se, best_n,
                      ece(p, y) if len(dm) else _nan(), brier(p, y) if len(dm) else _nan(), int(len(pos)),
                      float(pnl.mean()) if len(pnl) else _nan(), float((pnl <= catastrophic).mean()) if len(pnl) else _nan(),
                      float(pnl.min()) if len(pnl) else _nan(), {str(k): int(v) for k, v in ab.items()},
                      str(pd.to_datetime(J["end"]).max().date()) if len(J) else "")


def capability_readings(ev: Evaluation, as_of: str):
    """The controller's P1/P3/P4/P6 readings from an evaluation (only the metrics that were actually measured)."""
    from engine.research import controller as CT
    R, P = CT.CapabilityReading, CT.Phase
    out = []
    if np.isfinite(ev.vol_auc):
        out.append(R(P.VOLATILITY, "vol_auc", ev.vol_auc, as_of, None if not np.isfinite(ev.vol_auc_se) else ev.vol_auc_se,
                     ev.n_rows, "two_stage.evaluate"))
    if np.isfinite(ev.direction_acc) and ev.n_direction:
        out.append(R(P.DIRECTION, "direction_acc", ev.direction_acc, as_of, ev.direction_se, ev.n_direction, "two_stage.evaluate"))
    if np.isfinite(ev.best_acc_at_coverage) and ev.best_acc_n:
        out.append(R(P.DIRECTION_AT_COVERAGE, "acc_at_coverage", ev.best_acc_at_coverage, as_of, ev.best_acc_se, ev.best_acc_n,
                     "two_stage.evaluate"))
    if np.isfinite(ev.catastrophic_rate) and ev.n_positions:
        r = ev.catastrophic_rate
        out.append(R(P.CATASTROPHIC_LOSS, "catastrophic_rate", r, as_of, math.sqrt(max(r * (1 - r), 1e-4) / ev.n_positions),
                     ev.n_positions, "two_stage.evaluate"))
    return out


def decisions_to_frame(decisions: Sequence[DayDecision]) -> pd.DataFrame:
    frames = [d.table for d in decisions if len(d.table)]
    return pd.concat(frames) if frames else pd.DataFrame()
