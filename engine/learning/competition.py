"""Knowledge competition: several explanations of one relation stay alive until evidence separates them
(contract C62 section 39).  IMPLEMENTED - NOT VALIDATED.

    Pattern A causes movement          y ~ a
    Pattern A proxies volatility       y ~ vol            (A itself adds nothing)
    Pattern A only works in liquid     y ~ a  when liq > x
    Pattern A only works in trends     y ~ a  when trend > x

Each explanation is a HypothesisSpec that makes a predictive distribution for every new observation.  The arena scores them
PREQUENTIALLY: each hypothesis predicts a batch BEFORE seeing its outcomes, is scored (Student-t log score), and only then refits
on the batch.  Posterior weights = prior x Occam prior x exp(discounted cumulative log score).

Nothing is decided early.  A hypothesis can only be ELIMINATED when (a) the leader beats it by a Bayes factor above the
separation bar counted over DISCRIMINATING observations only (rows where the two actually predict differently: evidence from
rows both explain equally well cannot separate them) and (b) at least `min_disc` such rows exist and (c) its weight is tiny.
Elimination is reversible: eliminated hypotheses keep being scored and are revived if the evidence turns.  Explanations whose
predictions are practically identical are labelled EQUIVALENT instead of being waited on forever.
While the field is unresolved the arena predicts with the posterior-weighted mixture, and `next_experiment()` names the slice
of situations where the leading rivals disagree most - the data that would separate them (an input for research policy).

Builds on engine.learning.complexity (Occam prior).  Time (C56): a batch must be dated strictly before `now` and strictly after
every earlier batch; predictions for a batch never depend on that batch's outcomes (proved by test)."""
from __future__ import annotations

import dataclasses as dc
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.special import logsumexp

from engine.learning.complexity import RuleSpec, occam_log_prior, spec_from_hypothesis
from engine.learning.core import Edge, FirewallBreach, Unknown, _StrEnum, as_date, stable_hash


class CompetitionError(ValueError):
    pass


class HypFamily(_StrEnum):
    NULL = "NULL"
    CAUSAL = "CAUSAL"
    PROXY = "PROXY"
    CONDITIONAL = "CONDITIONAL"
    INTERACTION = "INTERACTION"


class HypStatus(_StrEnum):
    ALIVE = "ALIVE"
    LEADING = "LEADING"
    EQUIVALENT = "EQUIVALENT"            # cannot be told apart from the leader by any data seen
    ELIMINATED = "ELIMINATED"            # separated and negligible - still scored, revivable


_OPS = {">": np.greater, ">=": np.greater_equal, "<": np.less, "<=": np.less_equal}


@dc.dataclass(frozen=True)
class HypothesisSpec:
    hyp_id: str
    family: HypFamily
    description: str
    features: tuple[str, ...] = ()
    gate: tuple[tuple[str, str, float], ...] = ()      # ((column, op, value), ...) - the model applies only where ALL hold
    intercept: bool = True                              # False = mean fixed at zero (the 'no effect' story)
    prior_weight: float = 1.0
    ridge: float = 1e-6

    def __post_init__(self):
        object.__setattr__(self, "family", HypFamily.parse(self.family))

    def columns(self) -> tuple[str, ...]:
        """Raw data columns needed; an interaction feature 'a*liq' needs both a and liq."""
        parts = [p for f in self.features for p in f.split("*")]
        return tuple(dict.fromkeys(parts + [g[0] for g in self.gate]))

    def validate(self) -> list[str]:
        errs = []
        if not self.hyp_id:
            errs.append("empty hyp_id")
        if not self.prior_weight > 0:
            errs.append("prior_weight must be > 0")
        for g in self.gate:
            if len(g) != 3 or g[1] not in _OPS:
                errs.append(f"bad gate clause {g!r}")
        if self.family == HypFamily.NULL and (self.features or self.gate):
            errs.append("NULL hypothesis may not use features or a gate")
        return errs

    def rule_spec(self) -> RuleSpec:
        return spec_from_hypothesis(self.hyp_id, self.features, self.gate)


def null_spec(hyp_id: str = "H_null") -> HypothesisSpec:
    return HypothesisSpec(hyp_id, HypFamily.NULL, "no relationship: the outcome is unrelated to the pattern", (), (), False)


def causal_spec(signal: str, hyp_id: str = "H_causal") -> HypothesisSpec:
    return HypothesisSpec(hyp_id, HypFamily.CAUSAL, f"{signal} moves the outcome directly", (signal,), (), True)


def proxy_spec(proxy: str, hyp_id: str | None = None) -> HypothesisSpec:
    return HypothesisSpec(hyp_id or f"H_proxy_{proxy}", HypFamily.PROXY, f"the pattern only stands in for {proxy}", (proxy,), (), True)


def conditional_spec(signal: str, gate_col: str, op: str, value: float, hyp_id: str | None = None) -> HypothesisSpec:
    return HypothesisSpec(hyp_id or f"H_cond_{gate_col}{op}{value:g}", HypFamily.CONDITIONAL,
                          f"{signal} works only where {gate_col} {op} {value:g}", (signal,), ((gate_col, op, float(value)),), True)


def standard_field(signal: str, proxies: Sequence[str] = (), gates: Sequence[tuple[str, str, float]] = ()) -> list[HypothesisSpec]:
    """The canonical competing set of section 39: null, causal, one proxy story per proxy, one conditional story per gate."""
    out = [null_spec(), causal_spec(signal)]
    out += [proxy_spec(p) for p in proxies]
    out += [conditional_spec(signal, c, op, v) for c, op, v in gates]
    return out


def interaction_spec(signal: str, modifier: str, hyp_id: str | None = None) -> HypothesisSpec:
    """'The signal's effect scales with <modifier>': y ~ signal + signal*modifier + modifier."""
    return HypothesisSpec(hyp_id or f"H_int_{signal}x{modifier}", HypFamily.INTERACTION,
                          f"{signal}'s effect grows or shrinks with {modifier}", (signal, f"{signal}*{modifier}", modifier), (), True)


def _feature_values(d: pd.DataFrame, f: str) -> np.ndarray:
    v = np.ones(len(d))
    for part in f.split("*"):
        v = v * d[part].astype(float).values
    return v


def _gate_mask(df: pd.DataFrame, gate) -> np.ndarray:
    m = np.ones(len(df), bool)
    for col, op, val in gate:
        v = df[col].astype(float).values
        m &= _OPS[op](v, val)
    return m


# ------------------------------------------------------------------------------------------------ one hypothesis' model
class _Model:
    """Weighted ridge regression inside the gate, mean outside it.  Predictive distribution = Student-t location/scale."""

    def __init__(self, spec: HypothesisSpec, min_fit: int):
        self.spec = spec
        self.min_fit = min_fit
        self.fitted = False
        self.beta = np.zeros(1 + len(spec.features))
        self.sd_in = 1.0
        self.sd_out = 1.0
        self.out_mean = 0.0
        self.n_in = 0.0
        self.fallback_sd = 1.0

    def _design(self, d: pd.DataFrame) -> np.ndarray:
        cols = [np.ones(len(d))] if self.spec.intercept else []
        cols += [_feature_values(d, f) for f in self.spec.features]
        return np.column_stack(cols) if cols else np.zeros((len(d), 0))

    def fit(self, df: pd.DataFrame, ycol: str, forget: float) -> None:
        sp = self.spec
        cols = list(sp.columns()) + [ycol]
        age = np.arange(len(df))[::-1].astype(float)
        d = df[cols]
        ok = np.isfinite(d.astype(float).values).all(axis=1)
        d, age = d[ok], age[ok]
        y_all = d[ycol].astype(float).values
        self.fallback_sd = float(np.std(y_all, ddof=1)) if len(y_all) > 2 else 1.0
        if self.fallback_sd <= 0:
            self.fallback_sd = 1.0
        w = forget ** age
        m_in = _gate_mask(d, sp.gate)
        if m_in.sum() < self.min_fit:
            self.fitted = False
            return
        X, y, wi = self._design(d[m_in]), y_all[m_in], w[m_in]
        if X.shape[1] == 0:
            beta = np.zeros(0)
            res = y
        else:
            A = X.T @ (X * wi[:, None])
            pen = np.eye(X.shape[1]) * sp.ridge * wi.sum()
            if sp.intercept:
                pen[0, 0] = 0.0
            beta = np.linalg.solve(A + pen + 1e-12 * np.eye(X.shape[1]), X.T @ (wi * y))
            res = y - X @ beta
        self.beta = beta
        self.n_in = float(wi.sum() ** 2 / (wi ** 2).sum())
        self.sd_in = max(math.sqrt(float(np.sum(wi * res ** 2) / wi.sum())), 1e-9)
        m_out = ~m_in
        if m_out.sum() >= 5 and sp.gate:
            yo, wo = y_all[m_out], w[m_out]
            self.out_mean = float(np.sum(wo * yo) / wo.sum()) if sp.intercept else 0.0
            self.sd_out = max(math.sqrt(float(np.sum(wo * (yo - self.out_mean) ** 2) / wo.sum())), 1e-9)
        else:
            self.out_mean = 0.0
            self.sd_out = self.sd_in
        self.fitted = True

    def predict(self, d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(mu, sd) per row; NaN where a required column is missing.  Uses only what fit() saw."""
        sp = self.spec
        n = len(d)
        cols = list(sp.columns())
        good = np.isfinite(d[cols].astype(float).values).all(axis=1) if cols else np.ones(n, bool)
        mu = np.full(n, np.nan)
        sd = np.full(n, np.nan)
        if not self.fitted:
            mu[good] = 0.0
            sd[good] = self.fallback_sd
            return mu, sd
        inside = np.zeros(n, bool)
        inside[good] = _gate_mask(d[good], sp.gate)
        X = self._design(d)
        mu_in = X @ self.beta if X.shape[1] else np.zeros(n)
        infl = math.sqrt(1.0 + len(self.beta) / max(self.n_in, 1.0))
        mu = np.where(inside, mu_in, self.out_mean)
        sd = np.where(inside, self.sd_in, self.sd_out) * infl
        mu[~good], sd[~good] = np.nan, np.nan
        return mu, sd


# ------------------------------------------------------------------------------------------------ arena
@dc.dataclass(frozen=True)
class ArenaConfig:
    min_fit: int = 12
    forget: float = 0.998                # per-row discount of old evidence (regimes change; explanations can be revived)
    t_df: float = 5.0                    # Student-t scoring: one wild outcome must not decide a competition
    sep_factor: float = 20.0             # Bayes factor (over discriminating rows) the leader needs over a rival
    min_disc: int = 15
    disc_delta: float = 0.25             # rows count as discriminating when predictions differ by this many sd
    elim_weight: float = 0.02
    revive_weight: float = 0.10
    equiv_corr: float = 0.995
    equiv_min_rows: int = 40
    complexity_lambda: float = 0.5

    def validate(self) -> list[str]:
        errs = []
        if not 0 < self.forget <= 1:
            errs.append("forget in (0,1]")
        if self.sep_factor <= 1:
            errs.append("sep_factor must exceed 1")
        if self.revive_weight <= self.elim_weight:
            errs.append("revive_weight must exceed elim_weight (hysteresis)")
        return errs


@dc.dataclass(frozen=True)
class StepReport:
    date: str
    n_rows: int
    n_scored: int
    n_skipped: int
    weights: tuple[tuple[str, float], ...]
    leader: str
    separated: bool
    events: tuple[str, ...]


@dc.dataclass(frozen=True)
class DiscriminatingSlice:
    """Where the leading rivals disagree most: the situations to collect data in to separate them."""
    leader: str
    rival: str
    fraction_of_rows: float
    expected_llr_per_row: float
    columns: tuple[tuple[str, float, float, float, float], ...]   # (col, slice_q25, slice_q75, all_q25, all_q75)
    description: str


class Arena:
    """A field of competing hypotheses about one relation.  Feed it batches in time order with step()."""

    def __init__(self, relation_id: str, specs: Sequence[HypothesisSpec], y_col: str = "y", cfg: ArenaConfig = ArenaConfig()):
        errs = cfg.validate()
        for s in specs:
            errs += [f"{s.hyp_id}: {e}" for e in s.validate()]
        ids = [s.hyp_id for s in specs]
        if len(set(ids)) != len(ids):
            errs.append("duplicate hyp_id")
        if len(specs) < 2:
            errs.append("a competition needs at least two hypotheses")
        if errs:
            raise CompetitionError("; ".join(errs))
        self.relation_id, self.y_col, self.cfg = relation_id, y_col, cfg
        self.specs = list(specs)
        self.ids = ids
        k = len(specs)
        self._models = [_Model(s, cfg.min_fit) for s in specs]
        self._cols = sorted({c for s in specs for c in s.columns()})
        self._data = pd.DataFrame(columns=self._cols + [y_col], dtype=float)
        self._cum = np.zeros(k)
        self._llr = np.zeros((k, k))                      # discounted sum over discriminating rows of ls_i - ls_j
        self._nd = np.zeros((k, k))                       # discounted discriminating row counts
        self._sm = np.zeros(k)                            # prediction moments for equivalence
        self._sm2 = np.zeros(k)
        self._sx = np.zeros((k, k))
        self._nm = 0.0
        self._status = {i: HypStatus.ALIVE for i in self.ids}
        self._last: pd.Timestamp | None = None
        self.events: list[dict] = []
        self.history: list[StepReport] = []
        self._log_prior = np.array([math.log(s.prior_weight) + occam_log_prior(s.rule_spec().units(), cfg.complexity_lambda)
                                    for s in specs])
        self.n_seen = 0
        self._rec: list[dict] = []                       # per-step prediction records (enables late-entrant replay)
        self._pit: list[float] = []                      # prequential PIT of the mixture forecast

    # ---- scoring
    def _logscore(self, y, mu, sd) -> np.ndarray:
        z = (y - mu) / sd
        return sps.t.logpdf(z, self.cfg.t_df) - np.log(sd)

    def step(self, batch: pd.DataFrame, now) -> StepReport:
        cfg = self.cfg
        if self.y_col not in batch.columns:
            raise CompetitionError(f"outcome column {self.y_col!r} missing")
        miss = [c for c in self._cols if c not in batch.columns]
        if miss:
            raise CompetitionError(f"columns missing from batch: {miss}")
        b = batch[self._cols + [self.y_col]].copy()
        b.index = pd.to_datetime(b.index)
        if len(b) == 0:
            return StepReport("", 0, 0, 0, self._weight_pairs(), self.leader(), False, ())
        if b.index.max() >= pd.Timestamp(as_date(now)):
            raise FirewallBreach(f"competition[{self.relation_id}]: row dated {b.index.max().date()} is not before now={as_date(now)}")
        b = b.sort_index()
        if self._last is not None and b.index.min() <= self._last:
            raise CompetitionError(f"batch starting {b.index.min().date()} overlaps history through {self._last.date()}: no back-filling")
        b = b[np.isfinite(b[self.y_col].astype(float).values)]
        if len(b) == 0:
            return StepReport("", 0, 0, 0, self._weight_pairs(), self.leader(), False, ())
        k = len(self.ids)
        # 1. PREDICT before any learning from this batch
        mus = np.column_stack([m.predict(b)[0] for m in self._models])
        sds = np.column_stack([m.predict(b)[1] for m in self._models])
        y = b[self.y_col].astype(float).values
        ok = np.isfinite(mus).all(axis=1) & np.isfinite(sds).all(axis=1)
        n_ok = int(ok.sum())
        events: list[str] = []
        if n_ok:
            mu, sd, yy = mus[ok], sds[ok], y[ok]
            w0 = np.exp(self.log_weights())
            cdf = np.zeros(n_ok)
            for i in range(k):
                cdf += w0[i] * sps.t.cdf((yy - mu[:, i]) / sd[:, i], cfg.t_df)
            self._pit.extend(float(v) for v in cdf)
            ls = np.column_stack([self._logscore(yy, mu[:, i], sd[:, i]) for i in range(k)])
            g = cfg.forget ** n_ok
            self._cum = self._cum * g + ls.sum(axis=0)
            self._llr *= g
            self._nd *= g
            for i in range(k):
                for j in range(i + 1, k):
                    disc = np.abs(mu[:, i] - mu[:, j]) > cfg.disc_delta * np.maximum(sd[:, i], sd[:, j])
                    if disc.any():
                        d = float((ls[disc, i] - ls[disc, j]).sum())
                        self._llr[i, j] += d
                        self._llr[j, i] -= d
                        self._nd[i, j] += disc.sum()
                        self._nd[j, i] += disc.sum()
            self._sm += mu.sum(axis=0)
            self._sm2 += (mu ** 2).sum(axis=0)
            self._sx += mu.T @ mu
            self._nm += n_ok
        if n_ok:
            self._rec.append({"start": len(self._data), "end": len(self._data) + len(b), "ok": ok.copy(), "mu": mu, "sd": sd,
                              "ls": ls, "n_ok": n_ok})
        # 2. LEARN from the batch
        self._data = pd.concat([self._data, b]) if len(self._data) else b.copy()
        self._last = b.index.max()
        self.n_seen += len(b)
        for m in self._models:
            m.fit(self._data, self.y_col, cfg.forget)
        # 3. STATUS
        events += self._assess(str(self._last.date()))
        rep = StepReport(str(self._last.date()), len(b), n_ok, len(b) - n_ok, self._weight_pairs(), self.leader(),
                         self.separated(), tuple(events))
        self.history.append(rep)
        return rep

    # ---- posterior
    def log_weights(self) -> np.ndarray:
        lw = self._log_prior + self._cum
        return lw - logsumexp(lw)

    def weights(self) -> dict[str, float]:
        return dict(zip(self.ids, np.exp(self.log_weights())))

    def _weight_pairs(self) -> tuple[tuple[str, float], ...]:
        return tuple((i, round(float(w), 10)) for i, w in self.weights().items())

    def leader(self) -> str:
        return self.ids[int(np.argmax(self.log_weights()))]

    def _equivalent(self, i: int, j: int) -> bool:
        if self._nm < self.cfg.equiv_min_rows:
            return False
        n = self._nm
        vi = self._sm2[i] / n - (self._sm[i] / n) ** 2
        vj = self._sm2[j] / n - (self._sm[j] / n) ** 2
        if vi <= 1e-14 and vj <= 1e-14:                    # both constant predictors: equal iff same mean
            return abs(self._sm[i] - self._sm[j]) / n < 1e-9
        if vi <= 1e-14 or vj <= 1e-14:
            return False
        cov = self._sx[i, j] / n - (self._sm[i] / n) * (self._sm[j] / n)
        corr = cov / math.sqrt(vi * vj)
        scale = math.sqrt(max(vi, vj))
        return corr >= self.cfg.equiv_corr and abs(self._sm[i] - self._sm[j]) / n < 0.25 * scale

    def _sep(self, lead: int, j: int) -> tuple[bool, float, float]:
        llr, nd = float(self._llr[lead, j]), float(self._nd[lead, j])
        return (nd >= self.cfg.min_disc and llr >= math.log(self.cfg.sep_factor)), llr, nd

    def _assess(self, date: str) -> list[str]:
        cfg = self.cfg
        w = np.exp(self.log_weights())
        lead = int(np.argmax(w))
        ev = []
        prev_leader = next((i for i, s in self._status.items() if s == HypStatus.LEADING), None)
        for j, hid in enumerate(self.ids):
            if j == lead:
                self._status[hid] = HypStatus.LEADING
                continue
            was = self._status[hid]
            if self._equivalent(lead, j):
                self._status[hid] = HypStatus.EQUIVALENT
                continue
            sep, llr, nd = self._sep(lead, j)
            if sep and w[j] < cfg.elim_weight:
                if was != HypStatus.ELIMINATED:
                    ev.append(f"{date}: ELIMINATED {hid} (BF vs {self.ids[lead]} = e^{llr:.1f} over {nd:.0f} discriminating rows, weight {w[j]:.4f})")
                self._status[hid] = HypStatus.ELIMINATED
            elif was == HypStatus.ELIMINATED and (w[j] >= cfg.revive_weight or not sep):
                if w[j] >= cfg.revive_weight or llr < math.log(cfg.sep_factor) * 0.5:
                    ev.append(f"{date}: REVIVED {hid} (weight {w[j]:.3f}, BF e^{llr:.1f})")
                    self._status[hid] = HypStatus.ALIVE
            elif was != HypStatus.ELIMINATED:
                self._status[hid] = HypStatus.ALIVE
        if prev_leader is not None and prev_leader != self.ids[lead]:
            ev.append(f"{date}: LEADER CHANGE {prev_leader} -> {self.ids[lead]}")
        for e in ev:
            self.events.append({"event": e})
        return ev

    # ---- reading the state
    def status(self) -> dict[str, HypStatus]:
        return dict(self._status)

    def alive(self) -> list[str]:
        """Hypotheses still in contention (everything not eliminated; equivalents count as one live group)."""
        return [h for h, s in self._status.items() if s != HypStatus.ELIMINATED]

    def describe(self) -> str:
        w = self.weights()
        top = sorted(w.items(), key=lambda kv: -kv[1])[:3]
        tail = "separated" if self.separated() else "unresolved: " + self.undecided_reason()
        return f"{self.relation_id}: {len(self.alive())}/{len(self.ids)} alive after {self.n_seen} rows; leading {', '.join(f'{k} ({v:.2f})' for k, v in top)}; {tail}"

    def separated(self) -> bool:
        """True only when the leader beats every non-equivalent rival by the separation bar on discriminating rows."""
        lead = self.ids.index(self.leader())
        rivals = [j for j in range(len(self.ids)) if j != lead and not self._equivalent(lead, j)]
        return bool(rivals) and all(self._sep(lead, j)[0] for j in rivals) if rivals else self._nm >= self.cfg.equiv_min_rows

    def winner(self) -> str | None:
        return self.leader() if self.separated() else None

    def indistinguishable_from(self, hid: str) -> list[str]:
        i = self.ids.index(hid)
        return [self.ids[j] for j in range(len(self.ids)) if j != i and self._equivalent(i, j)]

    def undecided_reason(self) -> str:
        if self.separated():
            return ""
        lead = self.ids.index(self.leader())
        parts = []
        for j, hid in enumerate(self.ids):
            if j == lead or self._equivalent(lead, j):
                continue
            sep, llr, nd = self._sep(lead, j)
            if sep:
                continue
            if nd < self.cfg.min_disc:
                parts.append(f"{hid}: only {nd:.0f}/{self.cfg.min_disc} discriminating rows so far")
            else:
                parts.append(f"{hid}: BF e^{llr:.1f} below the bar e^{math.log(self.cfg.sep_factor):.1f}")
        return "; ".join(parts) or "no rival to compare"

    def report(self) -> pd.DataFrame:
        w = self.weights()
        lead = self.ids.index(self.leader())
        rows = []
        for j, (s, hid) in enumerate(zip(self.specs, self.ids)):
            sep, llr, nd = (True, 0.0, 0.0) if j == lead else self._sep(lead, j)
            rows.append({"hyp": hid, "family": str(s.family), "weight": w[hid], "log_score": float(self._cum[j]),
                         "units": s.rule_spec().units(), "status": str(self._status[hid]),
                         "llr_vs_leader": float(-llr) if j != lead else 0.0, "disc_rows": float(nd), "separated_from_leader": bool(sep)})
        return pd.DataFrame(rows).sort_values("weight", ascending=False).reset_index(drop=True)

    # ---- prediction while unresolved
    def predict(self, X: pd.DataFrame, mode: str = "mixture") -> tuple[np.ndarray, np.ndarray]:
        """Posterior-weighted mixture (default) or leader-only prediction.  Mixture variance = within + between (law of total
        variance), so unresolved disagreement shows up as uncertainty instead of a confident pick."""
        preds = [m.predict(X) for m in self._models]
        if mode == "leader":
            return preds[self.ids.index(self.leader())]
        if mode != "mixture":
            raise CompetitionError(f"unknown mode {mode!r}")
        w = np.exp(self.log_weights())
        mu = np.column_stack([p[0] for p in preds])
        sd = np.column_stack([p[1] for p in preds])
        m = np.nansum(mu * w, axis=1)
        var = np.nansum(w * (sd ** 2 + (mu - m[:, None]) ** 2), axis=1)
        bad = ~np.isfinite(mu).any(axis=1)
        m[bad], var[bad] = np.nan, np.nan
        return m, np.sqrt(var)

    # ---- what data would separate them
    def next_experiment(self, top_frac: float = 0.15) -> DiscriminatingSlice | None:
        """The slice of stored situations where the leader and its closest unseparated rival disagree most."""
        if len(self._data) < 2 * self.cfg.min_fit:
            return None
        lead = self.ids.index(self.leader())
        cands = [j for j in range(len(self.ids)) if j != lead and not self._equivalent(lead, j) and not self._sep(lead, j)[0]]
        if not cands:
            return None
        j = min(cands, key=lambda c: abs(self._llr[lead, c]))
        mu_l, sd_l = self._models[lead].predict(self._data)
        mu_j, sd_j = self._models[j].predict(self._data)
        d = np.abs(mu_l - mu_j) / np.maximum(sd_l, sd_j)
        ok = np.isfinite(d)
        if ok.sum() < 20:
            return None
        cut = np.quantile(d[ok], 1 - top_frac)
        sl = ok & (d >= cut)
        cols = []
        for c in self._cols:
            v = self._data[c].astype(float).values
            q_all = np.nanpercentile(v[ok], [25, 75])
            q_sl = np.nanpercentile(v[sl], [25, 75])
            cols.append((c, float(q_sl[0]), float(q_sl[1]), float(q_all[0]), float(q_all[1])))
        cols.sort(key=lambda t: -abs((t[1] + t[2]) / 2 - (t[3] + t[4]) / 2) / max(t[4] - t[3], 1e-9))
        top = cols[0]
        desc = (f"{self.ids[lead]} and {self.ids[j]} disagree most where {top[0]} is in [{top[1]:.3g}, {top[2]:.3g}] "
                f"(all data: [{top[3]:.3g}, {top[4]:.3g}])")
        return DiscriminatingSlice(self.ids[lead], self.ids[j], float(sl.sum() / ok.sum()),
                                   float(np.mean(d[sl] ** 2) / 2.0), tuple(cols), desc)

    # ---- graph and audit
    def edges(self) -> list[tuple[str, str, Edge]]:
        """Relations between the competitors as knowledge-graph edges: separated rivals CONTRADICT, indistinguishable ones are
        REDUNDANT_WITH, a gated version SPECIALIZES the ungated one on the same signal."""
        out = []
        n = len(self.ids)
        for i in range(n):
            for j in range(i + 1, n):
                si, sj = self.specs[i], self.specs[j]
                if self._equivalent(i, j):
                    out.append((self.ids[i], self.ids[j], Edge.REDUNDANT_WITH))
                    continue
                a, b = (i, j) if self._llr[i, j] >= 0 else (j, i)
                if self._sep(a, b)[0]:
                    out.append((self.ids[a], self.ids[b], Edge.CONTRADICTS))
                for x, y_ in ((si, sj), (sj, si)):
                    if x.gate and not y_.gate and set(x.features) == set(y_.features) and x.features:
                        out.append((x.hyp_id, y_.hyp_id, Edge.SPECIALIZES))
        return out

    def state(self) -> dict:
        return {"relation": self.relation_id, "n_seen": self.n_seen, "leader": self.leader(), "separated": self.separated(),
                "weights": {k: round(v, 10) for k, v in self.weights().items()},
                "status": {k: str(v) for k, v in self._status.items()}, "events": [e["event"] for e in self.events]}

    def state_hash(self) -> str:
        return stable_hash(self.state())

    def trajectory(self) -> pd.DataFrame:
        """Weight of every hypothesis after each step - the competition's own timeline."""
        rows = [{"date": h.date, **dict(h.weights)} for h in self.history]
        return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


# ------------------------------------------------------------------------------------------------ synthetic worlds (test rigs)
def synthetic_world(truth: str, n: int = 400, seed: int = 0, effect: float = 0.03, noise: float = 0.02,
                    start: str = "2015-01-05") -> pd.DataFrame:
    """Planted-truth observations for competition tests.  Columns: a (pattern signal, correlated with vol), vol, liq, trend, y.
    truth: 'null' | 'causal' | 'proxy' (y follows vol, a only correlated with it) | 'liq' (a works only where liq > 0)
    | 'trend' (a works only where trend > 0)."""
    rng = np.random.default_rng(seed)
    vol = rng.normal(size=n)
    a = 0.7 * vol + math.sqrt(1 - 0.49) * rng.normal(size=n)
    liq = rng.normal(size=n)
    trend = rng.normal(size=n)
    eps = rng.normal(scale=noise, size=n)
    if truth == "null":
        y = eps
    elif truth == "causal":
        y = effect * a + eps
    elif truth == "proxy":
        y = effect * vol + eps
    elif truth == "liq":
        y = np.where(liq > 0, effect * a, 0.0) + eps
    elif truth == "trend":
        y = np.where(trend > 0, effect * a, 0.0) + eps
    else:
        raise CompetitionError(f"unknown truth {truth!r}")
    idx = pd.date_range(start, periods=n, freq="W-MON")
    return pd.DataFrame({"a": a, "vol": vol, "liq": liq, "trend": trend, "y": y}, index=idx)


def run_arena(arena: Arena, df: pd.DataFrame, batch: int, now) -> list[StepReport]:
    """Feed a frame to an arena in consecutive batches (the only supported way to replay history)."""
    out = []
    for s in range(0, len(df), batch):
        out.append(arena.step(df.iloc[s:s + batch], now))
    return out


# ------------------------------------------------------------------------------------------------ analysis of a field
def pairwise_table(arena: Arena) -> pd.DataFrame:
    """Every ordered pair: discounted log Bayes factor over discriminating rows, how many such rows there were, and whether
    the pair is separated.  Rows are 'a beats b'."""
    rows = []
    n = len(arena.ids)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            sep, llr, nd = arena._sep(i, j)
            rows.append({"a": arena.ids[i], "b": arena.ids[j], "llr": llr, "disc_rows": nd, "separated": bool(sep),
                         "equivalent": bool(arena._equivalent(i, j))})
    return pd.DataFrame(rows)


def expected_rows_to_separate(arena: Arena, leader: str | None = None, rival: str | None = None) -> float:
    """Rough number of further observations (of the kind seen so far) needed to separate the leader from a rival:
    log(bar) / (KL per discriminating row * share of rows that discriminate).  inf when the two predict alike."""
    lead = arena.ids.index(leader or arena.leader())
    if rival is None:
        cands = [j for j in range(len(arena.ids)) if j != lead and not arena._equivalent(lead, j)]
        if not cands:
            return 0.0
        j = min(cands, key=lambda c: abs(arena._llr[lead, c]))
    else:
        j = arena.ids.index(rival)
    d = arena._data
    if len(d) < arena.cfg.min_fit:
        return float("inf")
    mu_l, sd_l = arena._models[lead].predict(d)
    mu_j, sd_j = arena._models[j].predict(d)
    ok = np.isfinite(mu_l) & np.isfinite(mu_j)
    if ok.sum() == 0:
        return float("inf")
    z = np.abs(mu_l[ok] - mu_j[ok]) / np.maximum(sd_l[ok], sd_j[ok])
    disc = z > arena.cfg.disc_delta
    if not disc.any():
        return float("inf")
    kl = float(np.mean(z[disc] ** 2) / 2.0)
    share = float(disc.mean())
    need_disc = max(math.log(arena.cfg.sep_factor) - float(arena._llr[lead, j]), 0.0) / max(kl, 1e-12)
    need_disc = max(need_disc, arena.cfg.min_disc - float(arena._nd[lead, j]), 0.0)
    return float(need_disc / share)


def conditional_from_boundary(signal: str, boundary: Any, hyp_id: str | None = None) -> HypothesisSpec:
    """Turn a learned boundary (engine.learning.boundary.Boundary, or anything with feature/threshold/works_when) into the
    competing story 'the signal works only inside this region'.  Boundary learning proposes; the arena decides."""
    op = "<=" if boundary.works_when == "<=" else ">"
    return conditional_spec(signal, boundary.feature, op, float(boundary.threshold),
                            hyp_id or f"H_bnd_{boundary.feature}{op}{float(boundary.threshold):.4g}")


def replay(specs: Sequence[HypothesisSpec], frames: Sequence[pd.DataFrame], now, relation_id: str = "replay", y_col: str = "y",
           cfg: ArenaConfig = ArenaConfig()) -> Arena:
    """Rebuild an arena from scratch by feeding the same batches in order.  Determinism check: the result's state_hash must equal
    the original arena's (used by the reproducibility tests)."""
    a = Arena(relation_id, specs, y_col, cfg)
    for f in frames:
        a.step(f, now)
    return a


def leakage_probe(arena: Arena, batch: pd.DataFrame) -> float:
    """Largest absolute change in any hypothesis' prediction for `batch` when the batch's OUTCOMES are scrambled.  Must be 0:
    predictions are made before the outcomes are learned.  A non-zero value means the arena peeked."""
    mu0 = np.column_stack([m.predict(batch)[0] for m in arena._models])
    b2 = batch.copy()
    rng = np.random.default_rng(0)
    b2[arena.y_col] = rng.permutation(b2[arena.y_col].values)
    mu1 = np.column_stack([m.predict(b2)[0] for m in arena._models])
    return float(np.nanmax(np.abs(mu0 - mu1))) if mu0.size else 0.0


# ------------------------------------------------------------------------------------------------ many relations at once
class CompetitionBook:
    """One arena per relation, advanced together.  Answers the research questions 'which competitions are unresolved, and which
    would resolve fastest?'.  Each relation keeps its own field of hypotheses."""

    def __init__(self, cfg: ArenaConfig = ArenaConfig()):
        self.cfg = cfg
        self.arenas: dict[str, Arena] = {}

    def open(self, relation_id: str, specs: Sequence[HypothesisSpec], y_col: str = "y") -> Arena:
        if relation_id in self.arenas:
            raise CompetitionError(f"competition for {relation_id!r} already open")
        self.arenas[relation_id] = Arena(relation_id, specs, y_col, self.cfg)
        return self.arenas[relation_id]

    def step(self, batches: Mapping[str, pd.DataFrame], now) -> dict[str, StepReport]:
        unknown = set(batches) - set(self.arenas)
        if unknown:
            raise CompetitionError(f"no open competition for {sorted(unknown)}")
        return {rid: self.arenas[rid].step(b, now) for rid, b in batches.items()}

    def settled(self) -> dict[str, str]:
        """Relations whose competition has a separated winner -> the winning hypothesis id."""
        return {rid: w for rid, a in self.arenas.items() if (w := a.winner()) is not None}

    def unresolved(self) -> list[str]:
        return sorted(rid for rid, a in self.arenas.items() if a.winner() is None)

    def priorities(self) -> pd.DataFrame:
        """Unresolved competitions ranked by how quickly more data would settle them (fewest expected rows first)."""
        rows = []
        for rid in self.unresolved():
            a = self.arenas[rid]
            rows.append({"relation": rid, "leader": a.leader(), "leader_weight": a.weights()[a.leader()], "n_seen": a.n_seen,
                         "rows_to_separate": expected_rows_to_separate(a), "why": a.undecided_reason()})
        if not rows:
            return pd.DataFrame(columns=["relation", "leader", "leader_weight", "n_seen", "rows_to_separate", "why"])
        return pd.DataFrame(rows).sort_values("rows_to_separate").reset_index(drop=True)

    def unknown_state(self, relation_id: str) -> Unknown | None:
        """CONFLICTED while rivals remain unseparated with enough data to judge; UNTESTED before any fitting; else None."""
        a = self.arenas[relation_id]
        if a.n_seen < a.cfg.min_fit:
            return Unknown.UNTESTED
        return None if a.separated() else Unknown.CONFLICTED

    def summary(self) -> pd.DataFrame:
        rows = []
        for rid, a in sorted(self.arenas.items()):
            w = a.weights()
            rows.append({"relation": rid, "leader": a.leader(), "weight": w[a.leader()], "separated": a.separated(),
                         "n_alive": sum(1 for s in a.status().values() if s != HypStatus.ELIMINATED),
                         "n_eliminated": sum(1 for s in a.status().values() if s == HypStatus.ELIMINATED), "n_seen": a.n_seen})
        return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ late entrants
def add_hypothesis(arena: Arena, spec: HypothesisSpec) -> None:
    """Admit a new competing explanation after the contest has started (e.g. a boundary the learner has just proposed).

    Fairness: the newcomer is scored PREQUENTIALLY over the whole history - at every past batch it is refit on only the data
    before that batch and predicts it blind - so its score, its pairwise Bayes factors and its equivalence statistics are exactly
    what they would have been had it been in the field from the start.  Requires its columns to exist and be finite on every row
    already scored (otherwise it cannot be judged on the same evidence and is refused rather than approximated)."""
    errs = spec.validate()
    if spec.hyp_id in arena.ids:
        errs.append("duplicate hyp_id")
    missing = [c for c in spec.columns() if c not in arena._cols]
    if missing:
        errs.append(f"columns {missing} were not stored by this arena")
    if errs:
        raise CompetitionError("; ".join(errs))
    cfg, k, y = arena.cfg, len(arena.ids), arena.y_col
    m = _Model(spec, cfg.min_fit)
    cum, sm, sm2 = 0.0, 0.0, 0.0
    llr_n, nd_n, sx_n = np.zeros(k), np.zeros(k), np.zeros(k)
    new_recs = []
    for rec in arena._rec:
        s0, e0 = rec["start"], rec["end"]
        if s0 > 0:
            m.fit(arena._data.iloc[:s0], y, cfg.forget)
        batch = arena._data.iloc[s0:e0]
        mu_n, sd_n = m.predict(batch)
        okm = rec["ok"]
        mu_n, sd_n = mu_n[okm], sd_n[okm]
        if not (np.isfinite(mu_n).all() and np.isfinite(sd_n).all()):
            raise CompetitionError(f"{spec.hyp_id}: columns not finite on rows already scored; cannot enter late")
        yy = batch[y].astype(float).values[okm]
        ls_n = arena._logscore(yy, mu_n, sd_n)
        g = cfg.forget ** rec["n_ok"]
        cum = cum * g + float(ls_n.sum())
        llr_n *= g
        nd_n *= g
        for j in range(k):
            disc = np.abs(mu_n - rec["mu"][:, j]) > cfg.disc_delta * np.maximum(sd_n, rec["sd"][:, j])
            if disc.any():
                llr_n[j] += float((ls_n[disc] - rec["ls"][disc, j]).sum())
                nd_n[j] += float(disc.sum())
        sm += float(mu_n.sum())
        sm2 += float((mu_n ** 2).sum())
        sx_n += mu_n @ rec["mu"]
        new_recs.append((mu_n, sd_n, ls_n))
    for rec, (mu_n, sd_n, ls_n) in zip(arena._rec, new_recs):
        rec["mu"] = np.column_stack([rec["mu"], mu_n])
        rec["sd"] = np.column_stack([rec["sd"], sd_n])
        rec["ls"] = np.column_stack([rec["ls"], ls_n])
    K = k + 1
    L, N, X = np.zeros((K, K)), np.zeros((K, K)), np.zeros((K, K))
    L[:k, :k], N[:k, :k], X[:k, :k] = arena._llr, arena._nd, arena._sx
    L[k, :k], L[:k, k] = llr_n, -llr_n
    N[k, :k], N[:k, k] = nd_n, nd_n
    X[k, :k], X[:k, k], X[k, k] = sx_n, sx_n, sm2
    arena._llr, arena._nd, arena._sx = L, N, X
    arena._cum = np.append(arena._cum, cum)
    arena._sm, arena._sm2 = np.append(arena._sm, sm), np.append(arena._sm2, sm2)
    arena._log_prior = np.append(arena._log_prior, math.log(spec.prior_weight) + occam_log_prior(spec.rule_spec().units(), cfg.complexity_lambda))
    m.fit(arena._data, y, cfg.forget)
    arena.specs.append(spec)
    arena.ids.append(spec.hyp_id)
    arena._models.append(m)
    arena._status[spec.hyp_id] = HypStatus.ALIVE
    arena.events.append({"event": f"{arena._last.date() if arena._last is not None else ''}: ENTERED {spec.hyp_id} (replayed {len(arena._rec)} batches)"})
    if arena._last is not None:
        arena._assess(str(arena._last.date()))


# ------------------------------------------------------------------------------------------------ honesty of the mixture
def mixture_calibration(arena: Arena) -> dict:
    """Is the field's own forecast honest?  Every scored row's outcome was forecast by the posterior-weighted mixture BEFORE it
    was learned; the probability-integral transforms of those forecasts must be Uniform(0,1).  A pile-up at 0/1 means the field
    is overconfident; a hump in the middle means it is underconfident."""
    p = np.asarray(arena._pit, float)
    if len(p) < 30:
        return {"n": int(len(p)), "verdict": str(Unknown.INSUFFICIENT_DATA)}
    ks = sps.kstest(p, "uniform")
    tails = float(np.mean((p < 0.05) | (p > 0.95)))
    middle = float(np.mean((p > 0.25) & (p < 0.75)))
    verdict = "CALIBRATED" if ks.pvalue >= 0.05 else ("OVERCONFIDENT" if tails > 0.10 else "UNDERCONFIDENT" if middle > 0.55 else "MISCALIBRATED")
    return {"n": int(len(p)), "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue), "tail_share": tails,
            "middle_share": middle, "verdict": verdict}


# ------------------------------------------------------------------------------------------------ calibration tools
def boundary_field(signal: str, df: pd.DataFrame, bsets: Iterable[Any], now, proxies: Sequence[str] = (), y_col: str = "y",
                   batch: int = 20, cfg: ArenaConfig = ArenaConfig()) -> Arena:
    """Run a competition whose field is the standard set PLUS one conditional story per accepted learned boundary.  This is how
    boundary learning feeds knowledge competition: boundaries propose, the arena decides whether 'only inside this region' beats
    'everywhere' and 'not at all'."""
    specs = [null_spec(), causal_spec(signal)] + [proxy_spec(p) for p in proxies]
    seen = set()
    for bs in bsets:
        for b in getattr(bs, "boundaries", ()):
            sp = conditional_from_boundary(signal, b)
            if sp.hyp_id not in seen and b.feature in df.columns:
                seen.add(sp.hyp_id)
                specs.append(sp)
    arena = Arena(f"field:{signal}", specs, y_col, cfg)
    run_arena(arena, df, batch, now)
    return arena


def rows_to_separate_curve(truth: str, seeds: Sequence[int] = tuple(range(8)), n: int = 600, batch: int = 20, effect: float = 0.03,
                           cfg: ArenaConfig = ArenaConfig(), gates=(("liq", ">", 0.0), ("trend", ">", 0.0))) -> dict:
    """How many observations does it take before the planted explanation is declared the separated winner (and is it the RIGHT
    one)?  Runs one arena per seed on synthetic_world(truth) and records the first step at which winner() is set.  The wrong-winner
    count is the competition's false-declaration rate on this world."""
    want = {"causal": "H_causal", "proxy": "H_proxy_vol", "liq": "H_cond_liq>0", "trend": "H_cond_trend>0", "null": "H_null"}[truth]
    rows, wrong, never = [], 0, 0
    for sd in seeds:
        df = synthetic_world(truth, n=n, seed=sd, effect=effect)
        a = Arena("cal", standard_field("a", ["vol"], list(gates)), cfg=cfg)
        first = None
        for s in range(0, n, batch):
            a.step(df.iloc[s:s + batch], "2100-01-01")
            if first is None and a.winner() is not None:
                first = min(s + batch, n)
                wrong += int(a.winner() != want)
                break
        if first is None:
            never += 1
        else:
            rows.append(first)
    return {"truth": truth, "n_seeds": len(seeds), "rows_median": float(np.median(rows)) if rows else float("nan"),
            "rows_max": float(max(rows)) if rows else float("nan"), "wrong_winner": wrong, "never_separated": never}


def hypothesis_card(arena: Arena, hyp_id: str) -> dict:
    """Everything the arena knows about one explanation: its structure, standing, fitted numbers and record against the leader."""
    if hyp_id not in arena.ids:
        raise CompetitionError(f"unknown hypothesis {hyp_id!r}")
    j = arena.ids.index(hyp_id)
    lead = arena.ids.index(arena.leader())
    m = arena._models[j]
    sep, llr, nd = (True, 0.0, 0.0) if j == lead else arena._sep(lead, j)
    return {"hyp_id": hyp_id, "family": str(arena.specs[j].family), "description": arena.specs[j].description,
            "status": str(arena._status[hyp_id]), "weight": arena.weights()[hyp_id], "units": arena.specs[j].rule_spec().units(),
            "fitted": m.fitted, "coefficients": [float(b) for b in m.beta], "sd_inside": m.sd_in, "sd_outside": m.sd_out,
            "outside_mean": m.out_mean, "log_score": float(arena._cum[j]), "log_bf_leader_over_it": float(llr),
            "discriminating_rows": float(nd), "separated_from_leader": bool(sep),
            "equivalent_to": arena.indistinguishable_from(hyp_id)}
