"""Eight independent learning firewalls behind one fail-closed gate (contract C62 section 55; serves sections 28-30, 56, 85
priority 1; Bible learning-validity canon C56/C58/C63).

Layers: DATA, TIME, MEMORY, IDENTITY, PROVENANCE, EXPERIMENT, EVALUATION, CODE-VERSION. Each layer is a `FirewallLayer`
that inspects a `GateContext` and returns a `LayerVerdict` made of typed `Finding`s. `LearningFirewallGate.evaluate`
runs every layer (one layer failing never hides another) and returns a `GateVerdict`; a learning result may be promoted
only when every RELEVANT layer passed (`GateVerdict.require`). Fail-closed rules:
  * a relevant layer whose inputs are missing FAILS (cannot demonstrate = not demonstrated); an empty-but-present
    input is a definite answer and passes;
  * an exception inside a layer is an ERROR status and counts as a failure;
  * no relevant layer at all is a failure (nothing checked is not clean).

Built on: engine.pit (label closes, training-row verification, implausible IC, calendar), engine.blind_gates
(forbidden information fields), engine.provenance (stale-code detection). The memory/identity/future specialists live in
memory_firewall.py, identity_firewall.py and future_firewall.py; this module only adapts their reports.

IMPLEMENTED - NOT VALIDATED."""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .core import (FirewallBreach, ValidationLabel, _StrEnum, as_date, canonical_json, stable_hash)


class Severity(_StrEnum):
    INFO = "INFO"
    WARN = "WARN"
    FAIL = "FAIL"


class LayerName(_StrEnum):
    DATA = "DATA"
    TIME = "TIME"
    MEMORY = "MEMORY"
    IDENTITY = "IDENTITY"
    PROVENANCE = "PROVENANCE"
    EXPERIMENT = "EXPERIMENT"
    EVALUATION = "EVALUATION"
    CODE_VERSION = "CODE_VERSION"


class LayerStatus(_StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_RELEVANT = "NOT_RELEVANT"
    ERROR = "ERROR"


LAYER_ORDER = tuple(LayerName)
# names that mark a column as a label, a future value or an outcome; a feature carrying one is rejected outright
FORBIDDEN_NAME = re.compile(r"(^y$|label|target|forward|fwd|future|next_|_next|outcome|realized|realised|answer|truth|"
                            r"ret_fwd|lead_)", re.I)


# ---------------------------------------------------------------- findings and verdicts
@dataclasses.dataclass(frozen=True)
class Finding:
    """One thing a firewall saw. `evidence` carries the numbers behind the message so a reviewer can re-derive it."""
    layer: LayerName
    check: str
    severity: Severity
    subject: str
    message: str
    evidence: Mapping[str, Any] = dataclasses.field(default_factory=dict, compare=False, hash=False)

    def validate(self) -> list[str]:
        errs = []
        if not self.check:
            errs.append("finding without a check name")
        if not self.message:
            errs.append("finding without a message")
        if not isinstance(self.layer, LayerName) or not isinstance(self.severity, Severity):
            errs.append("layer/severity must be enum members")
        return errs

    @property
    def is_fail(self) -> bool:
        return self.severity == Severity.FAIL

    def __str__(self):
        return f"[{self.severity}] {self.layer}.{self.check} ({self.subject}): {self.message}"

    def to_dict(self) -> dict:
        return {"layer": self.layer.value, "check": self.check, "severity": self.severity.value, "subject": self.subject,
                "message": self.message, "evidence": json.loads(canonical_json(dict(self.evidence)))}


def fail(layer, check, subject, message, **evidence) -> Finding:
    return Finding(layer, check, Severity.FAIL, str(subject), message, evidence)


def warn(layer, check, subject, message, **evidence) -> Finding:
    return Finding(layer, check, Severity.WARN, str(subject), message, evidence)


def info(layer, check, subject, message, **evidence) -> Finding:
    return Finding(layer, check, Severity.INFO, str(subject), message, evidence)


@dataclasses.dataclass(frozen=True)
class LayerVerdict:
    layer: LayerName
    status: LayerStatus
    findings: tuple[Finding, ...] = ()
    n_checked: int = 0

    @property
    def passed(self) -> bool:
        return self.status in (LayerStatus.PASS, LayerStatus.NOT_RELEVANT)

    @property
    def failures(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.is_fail)

    def digest(self) -> str:
        return stable_hash([self.layer, self.status, [f.to_dict() for f in self.findings], self.n_checked])

    def to_dict(self) -> dict:
        return {"layer": self.layer.value, "status": self.status.value, "n_checked": self.n_checked,
                "findings": [f.to_dict() for f in self.findings]}


@dataclasses.dataclass(frozen=True)
class GateVerdict:
    """The per-layer verdict of one gate run. `passed` is true only if at least one layer was relevant and every relevant
    layer passed. The gate never turns a pass into a validation claim: `label` stays IMPLEMENTED - NOT VALIDATED."""
    now: str
    subject: str
    verdicts: Mapping[LayerName, LayerVerdict]
    context_fingerprint: str = ""

    @property
    def relevant(self) -> tuple[LayerName, ...]:
        return tuple(k for k, v in self.verdicts.items() if v.status != LayerStatus.NOT_RELEVANT)

    @property
    def passed(self) -> bool:
        return bool(self.relevant) and all(v.passed for v in self.verdicts.values())

    @property
    def failed_layers(self) -> tuple[LayerName, ...]:
        return tuple(k for k, v in self.verdicts.items() if not v.passed)

    @property
    def label(self) -> ValidationLabel:
        return ValidationLabel.NOT_VALIDATED

    def findings(self, severity: Severity | None = None) -> list[Finding]:
        return [f for v in self.verdicts.values() for f in v.findings if severity is None or f.severity == severity]

    def require(self) -> "GateVerdict":
        """Raise FirewallBreach unless the result may be promoted. Callers must not catch-and-continue."""
        if not self.relevant:
            raise FirewallBreach(f"firewall gate ({self.subject}): no relevant layer was checked")
        if not self.passed:
            why = "; ".join(f"{k}: {v.failures[0].message if v.failures else v.status}"
                            for k, v in self.verdicts.items() if not v.passed)
            raise FirewallBreach(f"firewall gate ({self.subject}) rejected at now={self.now}: {why}")
        return self

    def digest(self) -> str:
        return stable_hash([self.now, self.subject, {k.value: v.digest() for k, v in self.verdicts.items()}])

    def to_dict(self) -> dict:
        return {"now": self.now, "subject": self.subject, "passed": self.passed, "label": self.label.value,
                "failed_layers": [k.value for k in self.failed_layers], "context_fingerprint": self.context_fingerprint,
                "layers": {k.value: v.to_dict() for k, v in self.verdicts.items()}, "digest": self.digest()}

    def markdown(self) -> str:
        lines = [f"# Firewall gate: {self.subject}", f"now = {self.now}   overall = {'PASS' if self.passed else 'REJECT'}"
                 f"   ({self.label})", "", "| layer | status | checked | failures |", "|---|---|---|---|"]
        for k in LAYER_ORDER:
            v = self.verdicts.get(k)
            if v is not None:
                lines.append(f"| {k} | {v.status} | {v.n_checked} | {len(v.failures)} |")
        for f in self.findings():
            if f.severity != Severity.INFO:
                lines.append(f"- {f}")
        return "\n".join(lines)


# ---------------------------------------------------------------- windows
def parse_window(w) -> tuple[pd.Timestamp, pd.Timestamp]:
    """A window is (start, end), 'start..end' or a dict with start/end. Returns normalised inclusive timestamps."""
    if isinstance(w, str):
        parts = re.split(r"\.\.|,|/| to ", w)
        if len(parts) != 2:
            raise ValueError(f"cannot parse window {w!r}")
        a, b = parts
    elif isinstance(w, Mapping):
        a, b = w["start"], w["end"]
    else:
        a, b = w
    s, e = pd.Timestamp(a).normalize(), pd.Timestamp(b).normalize()
    if e < s:
        raise ValueError(f"window ends before it starts: {w!r}")
    return s, e


def overlap_days(a, b) -> int:
    (s1, e1), (s2, e2) = parse_window(a), parse_window(b)
    lo, hi = max(s1, s2), min(e1, e2)
    return max(0, (hi - lo).days + 1)


# ---------------------------------------------------------------- records the experiment/evaluation layers inspect
@dataclasses.dataclass(frozen=True)
class ExperimentRecord:
    """What was declared before an experiment ran, and how its result was picked (section 1.2 'no fake learning')."""
    experiment_id: str
    registered_real: str = ""                  # wall-clock ISO time the plan was frozen
    first_result_real: str = ""                # wall-clock ISO time any result was first looked at
    config_hash: str = ""
    seed: int | None = None
    hypothesis: str = ""
    n_variants_tried: int = 1                  # configurations / runs examined before this one was reported
    selected_from: int = 1                     # of how many candidate runs this reported one was picked
    multiplicity_correction: str = "none"      # none | bonferroni | holm | fdr | holdout
    tuned_windows: tuple = ()
    training_windows: tuple = ()
    thresholds_changed_after_results: bool = False
    parameters_locked_real: str = ""
    baseline_declared: bool = False

    def validate(self) -> list[str]:
        errs = []
        if not self.experiment_id:
            errs.append("experiment_id missing")
        if self.n_variants_tried < 1 or self.selected_from < 1:
            errs.append("variant counts must be >= 1")
        if self.selected_from > self.n_variants_tried:
            errs.append("selected_from exceeds n_variants_tried")
        return errs


@dataclasses.dataclass(frozen=True)
class EvaluationRecord:
    """How a result was measured: which windows, sealed when, and whether the learner could touch them."""
    evaluation_windows: tuple = ()
    training_windows: tuple = ()
    sealed_real: str = ""                      # wall-clock ISO time the evaluation windows were sealed
    training_started_real: str = ""
    state_hash_before: str = ""                # learner state hash immediately before the evaluation ran
    state_hash_after: str = ""                 # ... and after; evaluation must not alter training state
    n_decisions: int = 0
    disguised: bool = False
    learner_told_rerun: bool = False           # the learner could tell this was a rerun
    labels_visible_before_decision: bool = False
    paired_baseline: bool = False
    best_of_runs_reported: bool = False        # the reported run is the best historical run

    def validate(self) -> list[str]:
        errs = []
        if self.n_decisions < 0:
            errs.append("n_decisions negative")
        for w in self.evaluation_windows + self.training_windows:
            try:
                parse_window(w)
            except (ValueError, KeyError, TypeError) as e:
                errs.append(f"bad window {w!r}: {e}")
        return errs


@dataclasses.dataclass(frozen=True)
class CodeState:
    """The code a result claims to come from, versus the code on disk now."""
    recorded: Mapping[str, Any] | str | None = None     # engine.provenance.stamp() dict (or a legacy bare hash string)
    current_hash: str | None = None                      # override for tests / remote checks
    accepted_hashes: frozenset = frozenset()             # code hashes deliberately allowed for old knowledge
    known_hashes: Mapping[str, str] = dataclasses.field(default_factory=dict)   # code_hash -> first-seen real date


@dataclasses.dataclass
class GateContext:
    """Everything the gate may look at, all read-only. Unset (None) means 'not supplied'; a relevant layer then fails."""
    now: Any
    subject: str = "learning-result"
    items: Sequence | None = None              # KnowledgeLike objects in play
    all_items: Sequence | None = None          # full store for lineage lookups (defaults to items)
    X: pd.DataFrame | None = None              # (date, ticker) x features
    y: pd.Series | None = None
    label_close: pd.Series | pd.DatetimeIndex | None = None
    horizon: int | None = None
    calendar: Any = None
    decisions: pd.DataFrame | None = None      # columns decision_date, fill_date
    frames: Mapping[str, Any] | None = None
    identity_report: Any = None
    experiment: ExperimentRecord | None = None
    evaluation: EvaluationRecord | None = None
    code: CodeState | None = None
    sealed_windows: tuple = ()
    relevant: frozenset | None = None          # None = all eight layers
    inputs: Sequence | None = None             # future_firewall.LearningInput records (optional extra screening)
    test_start: Any = None                     # first date of the evaluated period (enables the purge/embargo check)
    embargo_days: int = 0
    learned_states: Sequence | None = None     # memory_firewall.LearnedState records (leak channel 4)
    played: Sequence = ()                      # [{id, real_start, version}] windows and the state version each used
    registry_records: Sequence | None = None   # experiment-registry records (provenance audit)
    feature_specs: Sequence | None = None      # future_firewall.FeatureSpec declarations
    listings: Any = None                       # ticker / list_date / delist_date table for the survivorship check
    events: Sequence | None = None             # future_firewall.NetworkEvent records of the run
    cache: Sequence | None = None              # future_firewall.CacheEntry records of the run
    blind: bool = True
    sources: Iterable | None = None            # registered data source names

    def is_relevant(self, layer: LayerName) -> bool:
        return self.relevant is None or layer in self.relevant

    def store(self) -> Sequence:
        return self.all_items if self.all_items is not None else (self.items or ())

    def fingerprint(self) -> str:
        return stable_hash({"now": str(as_date(self.now)), "subject": self.subject,
                            "items": sorted(str(field_of(i, "knowledge_id", i)) for i in (self.items or ())),
                            "X": list(self.X.shape) if self.X is not None else None,
                            "experiment": self.experiment.experiment_id if self.experiment else None})


# ---------------------------------------------------------------- layer base
class FirewallLayer:
    name: LayerName

    def inspect(self, ctx: GateContext) -> tuple[list[Finding], int]:
        """Return (findings, number of things checked). Must not mutate ctx."""
        raise NotImplementedError

    def run(self, ctx: GateContext) -> LayerVerdict:
        if not ctx.is_relevant(self.name):
            return LayerVerdict(self.name, LayerStatus.NOT_RELEVANT)
        try:
            findings, n = self.inspect(ctx)
        except FirewallBreach as e:
            return LayerVerdict(self.name, LayerStatus.FAIL, (fail(self.name, "breach", ctx.subject, str(e)),), 0)
        except Exception as e:                                   # noqa: BLE001 - fail closed on any bug in a check
            return LayerVerdict(self.name, LayerStatus.ERROR,
                                (fail(self.name, "layer-error", ctx.subject, f"{type(e).__name__}: {e}"),), 0)
        bad = [f for f in findings if not isinstance(f, Finding) or f.validate()]
        if bad:
            return LayerVerdict(self.name, LayerStatus.ERROR, (fail(self.name, "malformed-finding", ctx.subject,
                                f"{len(bad)} malformed findings"),), n)
        status = LayerStatus.FAIL if any(f.is_fail for f in findings) else LayerStatus.PASS
        return LayerVerdict(self.name, status, tuple(findings), n)

    def missing(self, ctx: GateContext, what: str) -> tuple[list[Finding], int]:
        return [fail(self.name, "input-missing", ctx.subject,
                     f"{what} not supplied: this layer cannot demonstrate the result is clean")], 0


def field_of(obj, name, default=None):
    """Attribute-or-key access so records may be dataclasses, plain objects or dicts."""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _panel_dates(obj) -> pd.DatetimeIndex:
    idx = obj.index
    return pd.DatetimeIndex(idx.get_level_values(0) if isinstance(idx, pd.MultiIndex) else idx)



def panel_quality_findings(X: pd.DataFrame, layer: LayerName, subject: str, stale_run: int = 20, outlier_z: float = 50.0,
                           thin_frac: float = 0.4, max_cols: int = 400) -> list[Finding]:
    """Structural quality of a (date, ticker) panel that makes a learned pattern an artefact: values frozen for weeks
    (a forward-filled feed), absurd outliers, duplicated columns (two names for one signal), a cross-section that
    collapses on some dates (survivor-only tail), and dates that go backwards within a ticker."""
    out: list[Finding] = []
    if not isinstance(X.index, pd.MultiIndex) or not len(X):
        return out
    num = X.select_dtypes(include=[np.number]).iloc[:, :max_cols]
    per_date = pd.Series(1, index=X.index).groupby(level=0).sum()
    if len(per_date) >= 5:
        med = float(per_date.median())
        thin = per_date[per_date < thin_frac * med]
        if len(thin):
            out.append(warn(layer, "thin-cross-section", subject,
                            f"{len(thin)} dates hold under {thin_frac:.0%} of the median {med:.0f} names (first {thin.index.min():%Y-%m-%d})",
                            n_dates=int(len(thin))))
    for tk, grp in X.groupby(level=1, sort=False):
        d = pd.DatetimeIndex(grp.index.get_level_values(0))
        if len(d) > 1 and not d.is_monotonic_increasing:
            out.append(warn(layer, "dates-not-sorted", tk, "dates are not increasing within this ticker"))
            break
    seen: dict[int, str] = {}
    for c in num.columns:
        v = num[c].to_numpy(dtype=float)
        finite = v[np.isfinite(v)]
        if len(finite) < 10:
            continue
        sd = finite.std()
        if sd > 0:
            z = np.abs(finite - finite.mean()) / sd
            if z.max() > outlier_z:
                out.append(warn(layer, "extreme-outlier", c, f"|z| reaches {z.max():.0f} (bad tick or unit change)", z=float(z.max())))
        key = hash(np.round(finite[:200], 9).tobytes())
        if key in seen and np.array_equal(num[seen[key]].to_numpy(dtype=float)[np.isfinite(v)][:200], finite[:200]):
            out.append(warn(layer, "duplicate-column", c, f"identical to {seen[key]!r}: one signal counted twice"))
        seen.setdefault(key, c)
    for c in num.columns[:50]:
        s = num[c]
        if len(s) < stale_run * 3:
            continue
        for tk, grp in list(s.groupby(level=1, sort=False))[:20]:
            v = grp.to_numpy(dtype=float)
            if len(v) < stale_run:
                continue
            same = np.concatenate([[False], np.diff(v) == 0]).astype(int)
            run = int(max((len(x) for x in np.split(same, np.where(same == 0)[0]) if len(x) and x.sum()), default=0))
            if run >= stale_run:
                out.append(warn(layer, "frozen-values", f"{c}/{tk}", f"value unchanged for {run} consecutive rows (stale feed?)", run=run))
                break
    return out


def label_overlap_findings(dates: pd.DatetimeIndex, label_close: pd.DatetimeIndex, test_start, layer: LayerName,
                           subject: str, embargo_days: int = 0) -> list[Finding]:
    """Purge / embargo check: a training row whose label window (decision date .. label close) reaches into the test
    period leaks the test outcome into training even though its decision date is earlier."""
    out = []
    if len(dates) != len(label_close):
        return [fail(layer, "purge-shape", subject, "dates and label closes differ in length")]
    t0 = pd.Timestamp(test_start) - pd.Timedelta(days=embargo_days)
    train = dates < pd.Timestamp(test_start)
    reach = np.asarray(label_close >= t0) & np.asarray(train)
    if reach.any():
        out.append(fail(layer, "unpurged-overlap", subject,
                        f"{int(reach.sum())} training rows have labels reaching into the test period starting {pd.Timestamp(test_start).date()}"
                        + (f" (embargo {embargo_days}d)" if embargo_days else ""), n=int(reach.sum())))
    return out


# ---------------------------------------------------------------- 1 DATA
class DataFirewall(FirewallLayer):
    """The learning input itself: right shape, no label hiding in the features, no feature that already knows the answer."""
    name = LayerName.DATA

    def _inputs_only(self, ctx):
        from .future_firewall import check_network_cache, check_revision, check_survivorship
        out, n = [], 0
        for res in (check_revision(ctx.inputs, ctx.now), check_survivorship(ctx.inputs, ctx.now, ctx.listings)):
            out.extend(f for f in res.findings if f.layer == self.name)
            n += res.n_checked
        if ctx.events is not None or ctx.cache is not None:
            res = check_network_cache(ctx.events, ctx.cache, ctx.now, ctx.blind)
            out.extend(res.findings)
            n += res.n_checked
        return out, n

    def __init__(self, max_nan_frac: float = 0.5, ic_cap: float = 0.15, copy_corr: float = 0.999, min_dates: int = 2):
        self.max_nan_frac, self.ic_cap, self.copy_corr, self.min_dates = max_nan_frac, ic_cap, copy_corr, min_dates

    def inspect(self, ctx):
        X, y, L = ctx.X, ctx.y, self.name
        if X is None and not ctx.inputs:
            return self.missing(ctx, "feature panel X")
        if X is None:
            return self._inputs_only(ctx)
        out: list[Finding] = []
        if not len(X):
            return [fail(L, "empty-panel", ctx.subject, "feature panel has no rows")], 0
        if not isinstance(X.index, pd.MultiIndex) or X.index.nlevels != 2:
            out.append(fail(L, "index-shape", ctx.subject, "X must be indexed by MultiIndex (date, ticker)"))
            return out, len(X)
        if X.index.has_duplicates:
            out.append(fail(L, "duplicate-rows", ctx.subject, f"{int(X.index.duplicated().sum())} duplicated (date, ticker) rows",
                            n=int(X.index.duplicated().sum())))
        dates = _panel_dates(X)
        if dates.isna().any():
            out.append(fail(L, "null-dates", ctx.subject, f"{int(dates.isna().sum())} rows have no date"))
        if dates.nunique() < self.min_dates:
            out.append(warn(L, "few-dates", ctx.subject, f"only {dates.nunique()} distinct dates"))
        num = X.select_dtypes(include=[np.number])
        if num.shape[1] != X.shape[1]:
            out.append(warn(L, "non-numeric", ctx.subject, "non-numeric feature columns present: " +
                            ", ".join(map(str, X.columns.difference(num.columns)))))
        if num.shape[1]:
            arr = num.to_numpy(dtype=float, na_value=np.nan)
            if np.isinf(arr).any():
                cols = [str(c) for c, b in zip(num.columns, np.isinf(arr).any(axis=0)) if b]
                out.append(fail(L, "infinite-values", ctx.subject, f"infinite values in {cols[:5]}", columns=cols))
            nan_frac = np.isnan(arr).mean(axis=0)
            for c, fr in zip(num.columns, nan_frac):
                if fr > self.max_nan_frac:
                    out.append(fail(L, "mostly-missing", c, f"{fr:.0%} missing (limit {self.max_nan_frac:.0%})", frac=float(fr)))
            sd = np.nanstd(arr, axis=0)
            for c, s in zip(num.columns, sd):
                if not np.isnan(s) and s == 0:
                    out.append(warn(L, "constant-column", c, "zero variance"))
        for c in X.columns:
            if FORBIDDEN_NAME.search(str(c)) and not str(c).startswith("m_"):
                out.append(fail(L, "forbidden-name", c, f"feature name {c!r} marks a label / future value"))
        if y is not None:
            yy = y.reindex(X.index)
            if y.index.has_duplicates:
                out.append(fail(L, "duplicate-labels", ctx.subject, "label index has duplicates"))
            if yy.notna().sum() == 0:
                out.append(fail(L, "labels-unaligned", ctx.subject, "no label aligns with the feature rows"))
            else:
                for c in num.columns:
                    both = pd.concat([num[c], yy], axis=1, keys=["a", "b"]).dropna()
                    if len(both) >= 10 and both["a"].std() > 0 and both["b"].std() > 0 and \
                            abs(both["a"].corr(both["b"])) >= self.copy_corr:
                        out.append(fail(L, "label-copy", c, f"feature correlates {both['a'].corr(both['b']):.4f} with the label"))
                try:
                    from engine.pit import implausible_ic
                    tab = implausible_ic(X.select_dtypes(include=[np.number]), y, cap=self.ic_cap)
                    for name, r in tab.iterrows():
                        if bool(r["flag"]):
                            out.append(fail(L, "implausible-ic", name, f"mean rank IC {r['mean_ic']:.3f} exceeds {self.ic_cap}",
                                            mean_ic=float(r["mean_ic"])))
                except ImportError:
                    out.append(warn(L, "implausible-ic", ctx.subject, "engine.pit unavailable; IC screen skipped"))
        try:
            from engine.blind_gates import scan_information
            recs = [{"now": pd.Timestamp(as_date(ctx.now)), "fields": {str(c): None for c in X.columns}}]
            for f in scan_information(recs):
                if f.severity == "fail" and not any(x.check == "forbidden-name" and str(x.subject) in f.message for x in out):
                    out.append(fail(L, "forbidden-field", ctx.subject, f.message))
        except ImportError:
            pass
        out.extend(panel_quality_findings(X, self.name, ctx.subject))
        n = len(X)
        if ctx.inputs:
            from .future_firewall import check_revision, check_survivorship
            for res in (check_revision(ctx.inputs, ctx.now), check_survivorship(ctx.inputs, ctx.now, ctx.listings)):
                out.extend(f for f in res.findings if f.layer == self.name)
                n += res.n_checked
        if ctx.events is not None or ctx.cache is not None:
            from .future_firewall import check_network_cache
            res = check_network_cache(ctx.events, ctx.cache, ctx.now, ctx.blind)
            out.extend(res.findings)
            n += res.n_checked
        return out, n


# ---------------------------------------------------------------- 2 TIME
class TimeFirewall(FirewallLayer):
    """Nothing dated after `now`; labels closed strictly before it; decisions fill at a later session's open."""
    name = LayerName.TIME

    def inspect(self, ctx):
        L, out, n = self.name, [], 0
        now = pd.Timestamp(as_date(ctx.now))
        if ctx.X is None and ctx.frames is None and ctx.decisions is None and not ctx.inputs:
            return self.missing(ctx, "any dated data (X, frames or decisions)")
        if ctx.X is not None and len(ctx.X):
            d = _panel_dates(ctx.X)
            n += len(d)
            if d.isna().any():
                out.append(fail(L, "null-timestamp", ctx.subject, f"{int(d.isna().sum())} rows without a date"))
            late = d > now
            if late.any():
                out.append(fail(L, "future-rows", ctx.subject,
                                f"{int(late.sum())} feature rows dated after {now.date()} (latest {d.max().date()})", n=int(late.sum())))
            if getattr(d, "tz", None) is not None:
                out.append(warn(L, "tz-aware", ctx.subject, "timezone-aware index; dates compared as wall-clock"))
        for name, fr in (ctx.frames or {}).items():
            if fr is None or not len(fr):
                continue
            d = _panel_dates(fr)
            n += len(d)
            if pd.DatetimeIndex(d).max() > now:
                out.append(fail(L, "future-frame", name, f"frame {name!r} reaches {pd.DatetimeIndex(d).max().date()} after now"))
        lc = ctx.label_close
        if lc is None and ctx.X is not None and len(ctx.X) and ctx.horizon:
            from engine.pit import Calendar, label_close_dates
            lc = label_close_dates(_panel_dates(ctx.X), int(ctx.horizon), ctx.calendar or Calendar())
        if lc is not None and len(lc):
            lcd = pd.DatetimeIndex(lc.values if isinstance(lc, pd.Series) else lc)
            n += len(lcd)
            if lcd.isna().any():
                out.append(fail(L, "label-no-close", ctx.subject, f"{int(lcd.isna().sum())} labels have no close date"))
            ok = lcd.dropna()
            if len(ok) and (ok >= now).any():
                out.append(fail(L, "label-not-closed", ctx.subject,
                                f"{int((ok >= now).sum())} training labels close on/after {now.date()} (an outcome that matures on "
                                "now is not yet known)", n=int((ok >= now).sum())))
            if ctx.X is not None and len(lcd) == len(ctx.X):
                if (lcd < _panel_dates(ctx.X)).any():
                    out.append(fail(L, "label-before-row", ctx.subject, "a label closes before its own decision date"))
                if ctx.test_start is not None:
                    out.extend(label_overlap_findings(_panel_dates(ctx.X), lcd, ctx.test_start, L, ctx.subject, ctx.embargo_days))
        elif ctx.y is not None and ctx.X is not None:
            out.append(fail(L, "label-timing-unknown", ctx.subject, "labels supplied but neither label_close nor horizon given"))
        dec = ctx.decisions
        if dec is not None and len(dec):
            need = {"decision_date", "fill_date"}
            if not need <= set(dec.columns):
                out.append(fail(L, "decision-columns", ctx.subject, f"decisions need columns {sorted(need)}"))
            else:
                from engine.pit import Calendar
                cal = ctx.calendar or Calendar()
                dd, fd = pd.DatetimeIndex(pd.to_datetime(dec["decision_date"])), pd.DatetimeIndex(pd.to_datetime(dec["fill_date"]))
                n += len(dec)
                if (dd > now).any():
                    out.append(fail(L, "future-decision", ctx.subject, f"{int((dd > now).sum())} decisions dated after now"))
                earliest = cal.strict_next(dd)
                early = fd < earliest
                if early.any():
                    out.append(fail(L, "fill-not-next-session", ctx.subject,
                                    f"{int(early.sum())} fills at or before the decision's own close / on a non-session", n=int(early.sum())))
                if (~cal.is_session(fd)).any():
                    out.append(fail(L, "fill-off-session", ctx.subject, f"{int((~cal.is_session(fd)).sum())} fills on weekends/holidays"))
        if ctx.inputs:
            from .future_firewall import check_availability, check_timestamp
            for res in (check_timestamp(ctx.inputs, ctx.now), check_availability(ctx.inputs, ctx.now)):
                out.extend(res.findings)
                n += res.n_checked
        return out, n


# ---------------------------------------------------------------- 3 MEMORY
class MemoryFirewall(FirewallLayer):
    """Every learned item must be able to answer 'could this have existed at the decision timestamp?' (memory_firewall)."""
    name = LayerName.MEMORY

    def __init__(self, policy=None):
        self.policy = policy

    def inspect(self, ctx):
        if ctx.items is None and ctx.learned_states is None:
            return self.missing(ctx, "learned items or learned states")
        from .memory_firewall import audit_state_lineage, audit_store
        out, n = [], 0
        if ctx.items is not None:
            rep = audit_store(ctx.items, ctx.now, store=ctx.store(), sealed_windows=ctx.sealed_windows, policy=self.policy)
            out.extend(rep.findings)
            n += len(ctx.items)
            if not len(ctx.items):
                out.append(info(self.name, "empty-memory", ctx.subject, "no learned items in play (nothing to contaminate)"))
        if ctx.learned_states is not None:
            out.extend(audit_state_lineage(ctx.learned_states, ctx.played))
            n += len(ctx.learned_states)
        return out, n


# ---------------------------------------------------------------- 4 IDENTITY
class IdentityFirewall(FirewallLayer):
    """Performance must survive identity-preserving transformations (identity_firewall.IdentityReport)."""
    name = LayerName.IDENTITY

    def __init__(self, strict_inconclusive: bool = True):
        self.strict_inconclusive = strict_inconclusive

    def inspect(self, ctx):
        rep = ctx.identity_report
        if rep is None:
            return self.missing(ctx, "identity attack report")
        L, out = self.name, []
        verdicts = getattr(rep, "verdicts", None)
        if verdicts is None:
            return [fail(L, "report-shape", ctx.subject, "identity report has no verdicts")], 0
        if not len(verdicts):
            return [fail(L, "no-attacks", ctx.subject, "identity report contains no transformation runs")], 0
        for v in verdicts:
            status = str(getattr(v, "status", ""))
            kind = str(getattr(v, "kind", "?"))
            if status == "COLLAPSE":
                out.append(fail(L, "collapse", kind, f"performance collapsed under {kind}: retention "
                                f"{getattr(v, 'retention', float('nan')):.2f}", retention=float(getattr(v, "retention", float("nan")))))
            elif status == "NONDETERMINISTIC":
                out.append(fail(L, "nondeterministic", kind, "learner gave different answers on identical input"))
            elif status in ("INSUFFICIENT", "NO_SKILL"):
                sev = fail if self.strict_inconclusive else warn
                out.append(sev(L, "inconclusive", kind, f"{kind}: {status} - identity robustness not demonstrated"))
            elif status != "OK":
                out.append(fail(L, "unknown-status", kind, f"unrecognised verdict status {status!r}"))
        return out, len(verdicts)


# ---------------------------------------------------------------- 5 PROVENANCE
class ProvenanceFirewall(FirewallLayer):
    """Completeness and internal consistency of provenance: an item with holes in its history cannot be audited."""
    name = LayerName.PROVENANCE

    def __init__(self, require_hashes: Sequence[str] = ("code_hash", "data_hash", "config_hash", "experiment_id")):
        self.require_hashes = tuple(require_hashes)

    def inspect(self, ctx):
        if ctx.items is None and ctx.registry_records is None and ctx.feature_specs is None:
            return self.missing(ctx, "learned items, registry records or feature specs")
        L, out = self.name, []
        known = {str(field_of(i, "knowledge_id", i)) for i in ctx.store()}
        seen: dict[tuple, str] = {}
        extra = 0
        if ctx.registry_records is not None:
            out.extend(registry_findings(ctx.registry_records))
            extra += len(ctx.registry_records)
        if ctx.feature_specs is not None:
            from .future_firewall import check_feature_provenance
            res = check_feature_provenance(ctx.feature_specs, ctx.now, X=ctx.X, y=ctx.y, registered_sources=ctx.sources)
            out.extend(res.findings)
            extra += res.n_checked
        for it in ctx.items or ():
            kid = str(field_of(it, "knowledge_id", "?"))
            prov = field_of(it, "provenance", None)
            if prov is None:
                out.append(fail(L, "no-provenance", kid, "item carries no provenance"))
                continue
            for e in prov.check():
                out.append(fail(L, "provenance-invalid", kid, e))
            for h in self.require_hashes:
                if not getattr(prov, h, ""):
                    out.append(fail(L, "hash-missing", kid, f"provenance.{h} is empty"))
            if prov.seed is None:
                out.append(warn(L, "seed-missing", kid, "no random seed recorded; result cannot be re-derived"))
            for p in prov.parents:
                if p not in known:
                    out.append(fail(L, "parent-unresolved", kid, f"parent {p!r} is not in the store"))
                if p == kid:
                    out.append(fail(L, "self-parent", kid, "item lists itself as a parent"))
            key = (kid, field_of(it, "version", None))
            fp = stable_hash([prov.learned_at, prov.code_hash, prov.data_hash, prov.experiment_id])
            if key in seen and seen[key] != fp:
                out.append(fail(L, "version-conflict", kid, f"two different provenances claim version {key[1]}"))
            seen[key] = fp
            for s in prov.sealed_windows:
                try:
                    parse_window(s)
                except (ValueError, KeyError, TypeError):
                    out.append(fail(L, "sealed-window-unparseable", kid, f"sealed window {s!r} cannot be parsed"))
        return out, len(ctx.items or ()) + extra


# ---------------------------------------------------------------- 6 EXPERIMENT
class ExperimentFirewall(FirewallLayer):
    """Search hygiene: pre-registered, multiplicity accounted for, no tuning on the evaluation windows."""
    name = LayerName.EXPERIMENT

    def __init__(self, max_uncorrected_variants: int = 1):
        self.max_uncorrected = max_uncorrected_variants

    def inspect(self, ctx):
        ex = ctx.experiment
        if ex is None:
            return self.missing(ctx, "experiment record")
        L, out = self.name, []
        for e in ex.validate():
            out.append(fail(L, "record-invalid", ex.experiment_id or "?", e))
        if not ex.registered_real:
            out.append(fail(L, "not-preregistered", ex.experiment_id, "no registration time: the plan was not frozen before running"))
        elif ex.first_result_real and ex.first_result_real < ex.registered_real:
            out.append(fail(L, "registered-after-results", ex.experiment_id,
                            f"results were first seen {ex.first_result_real}, before registration {ex.registered_real}"))
        if ex.parameters_locked_real and ex.first_result_real and ex.parameters_locked_real > ex.first_result_real:
            out.append(fail(L, "params-locked-after-results", ex.experiment_id, "parameters were locked after results were seen"))
        if ex.thresholds_changed_after_results:
            out.append(fail(L, "post-hoc-thresholds", ex.experiment_id, "thresholds were changed after seeing results"))
        if ex.selected_from > 1 and ex.multiplicity_correction in ("", "none"):
            out.append(fail(L, "best-of-n", ex.experiment_id,
                            f"reported run was picked from {ex.selected_from} candidates with no correction", selected_from=ex.selected_from))
        elif ex.n_variants_tried > self.max_uncorrected and ex.multiplicity_correction in ("", "none"):
            out.append(warn(L, "uncorrected-search", ex.experiment_id, f"{ex.n_variants_tried} variants examined, none corrected"))
        if not ex.config_hash:
            out.append(fail(L, "config-unhashed", ex.experiment_id, "no configuration hash recorded"))
        if ex.seed is None:
            out.append(fail(L, "seed-missing", ex.experiment_id, "no seed recorded"))
        if not ex.baseline_declared:
            out.append(warn(L, "no-baseline", ex.experiment_id, "no baseline declared; improvement has nothing to be measured against"))
        ev = ctx.evaluation
        sealed = list(ev.evaluation_windows) if ev else []
        sealed += list(ctx.sealed_windows)
        n = 0
        for tw in tuple(ex.tuned_windows) + tuple(ex.training_windows):
            for sw in sealed:
                n += 1
                d = overlap_days(tw, sw)
                if d > 0:
                    out.append(fail(L, "tuned-on-evaluation", ex.experiment_id,
                                    f"window {tw} used for fitting/tuning overlaps sealed evaluation window {sw} by {d} days", days=d))
        return out, max(n, 1)


# ---------------------------------------------------------------- 7 EVALUATION
class EvaluationFirewall(FirewallLayer):
    """The measurement: sealed first, disjoint from training, state untouched, labels hidden, not a disguised-rerun giveaway."""
    name = LayerName.EVALUATION

    def __init__(self, min_decisions: int = 30, max_overlap_days: int = 0):
        self.min_decisions, self.max_overlap = min_decisions, max_overlap_days

    def inspect(self, ctx):
        ev = ctx.evaluation
        if ev is None:
            return self.missing(ctx, "evaluation record")
        L, out = self.name, []
        for e in ev.validate():
            out.append(fail(L, "record-invalid", ctx.subject, e))
        if not ev.evaluation_windows:
            out.append(fail(L, "no-evaluation-window", ctx.subject, "no evaluation window declared"))
        if not ev.sealed_real:
            out.append(fail(L, "not-sealed", ctx.subject, "evaluation windows were never sealed"))
        elif ev.training_started_real and ev.sealed_real > ev.training_started_real:
            out.append(fail(L, "sealed-after-training", ctx.subject,
                            f"sealed {ev.sealed_real} after training began {ev.training_started_real}"))
        n = 0
        for e in ev.evaluation_windows:
            for t in tuple(ev.training_windows) + tuple(ctx.experiment.training_windows if ctx.experiment else ()):
                n += 1
                d = overlap_days(e, t)
                if d > self.max_overlap:
                    out.append(fail(L, "train-eval-overlap", ctx.subject, f"training window {t} overlaps evaluation {e} by {d} days", days=d))
        if ev.state_hash_before and ev.state_hash_after:
            if ev.state_hash_before != ev.state_hash_after:
                out.append(fail(L, "state-changed-by-evaluation", ctx.subject, "learner state differs before vs after the evaluation "
                                "(test results altered training state)"))
        else:
            out.append(fail(L, "state-hash-missing", ctx.subject, "state hashes before/after evaluation not recorded"))
        if ev.labels_visible_before_decision:
            out.append(fail(L, "labels-visible", ctx.subject, "labels were visible to the learner before it decided"))
        if ev.learner_told_rerun:
            out.append(fail(L, "rerun-identified", ctx.subject, "learner could tell the evaluation was a rerun"))
        if ev.best_of_runs_reported:
            out.append(fail(L, "best-run-reported", ctx.subject, "the reported run was selected as the best historical run"))
        if ev.n_decisions < self.min_decisions:
            out.append(fail(L, "too-few-decisions", ctx.subject, f"{ev.n_decisions} decisions < {self.min_decisions}: the "
                            "measurement cannot support a claim", n=ev.n_decisions))
        if not ev.disguised:
            out.append(warn(L, "not-disguised", ctx.subject, "evaluation was not run on a disguised copy"))
        if not ev.paired_baseline:
            out.append(warn(L, "no-paired-baseline", ctx.subject, "no paired baseline; delta cannot be attributed to learning"))
        return out, max(n, 1)


# ---------------------------------------------------------------- 8 CODE VERSION
class CodeVersionFirewall(FirewallLayer):
    """A result must come from the code that is on disk now (a stale result from an edited module is rejected)."""
    name = LayerName.CODE_VERSION

    def inspect(self, ctx):
        cs = ctx.code
        if cs is None:
            return self.missing(ctx, "code state")
        L, out = self.name, []
        rec = cs.recorded
        if rec is None:
            return [fail(L, "no-code-record", ctx.subject, "result carries no code stamp")], 0
        if isinstance(rec, str):
            out.append(fail(L, "legacy-code-hash", ctx.subject, "bare code hash covers unrelated files; result is treated as stale"))
        elif "code_files" not in rec:
            out.append(fail(L, "code-files-missing", ctx.subject, "code stamp lists no files"))
        else:
            if rec.get("code_mixed"):
                out.append(fail(L, "code-edited-mid-run", ctx.subject, f"files edited after the process started: {list(rec['code_mixed'])[:5]}"))
            if cs.current_hash is not None:
                if cs.current_hash != rec.get("code_hash"):
                    out.append(fail(L, "code-changed", ctx.subject, f"recorded {rec.get('code_hash')} != current {cs.current_hash}"))
            else:
                from engine.provenance import stale
                if stale(rec):
                    out.append(fail(L, "stale-result", ctx.subject, "code on disk no longer matches the recorded code hash"))
        n = 1
        for it in ctx.items or ():
            prov = field_of(it, "provenance", None)
            if prov is None:
                continue
            n += 1
            h, kid = prov.code_hash, str(field_of(it, "knowledge_id", "?"))
            if cs.known_hashes and h not in cs.known_hashes:
                out.append(fail(L, "unregistered-code", kid, f"item was built by code {h!r} that was never registered"))
            elif h in cs.known_hashes and prov.created_real and str(cs.known_hashes[h]) > str(prov.created_real):
                out.append(fail(L, "code-postdates-record", kid,
                                f"code {h} first existed {cs.known_hashes[h]}, after the record was written ({prov.created_real})"))
            cur = cs.current_hash or (rec.get("code_hash") if isinstance(rec, Mapping) else None)
            if cur and h != cur and h not in cs.accepted_hashes:
                out.append(warn(L, "older-code", kid, f"built by code {h}; current is {cur} (not in the accepted list)"))
        return out, n


# ---------------------------------------------------------------- the gate
def default_layers() -> list[FirewallLayer]:
    return [DataFirewall(), TimeFirewall(), MemoryFirewall(), IdentityFirewall(), ProvenanceFirewall(), ExperimentFirewall(),
            EvaluationFirewall(), CodeVersionFirewall()]


class LearningFirewallGate:
    """Runs every layer and returns the per-layer verdict. Layers are independent: a bug or failure in one never skips
    another, and adding a layer is adding a `FirewallLayer` - the gate has no per-layer knowledge."""

    def __init__(self, layers: Iterable[FirewallLayer] | None = None):
        self.layers = list(layers) if layers is not None else default_layers()
        names = [layer.name for layer in self.layers]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate firewall layers: {names}")

    def evaluate(self, ctx: GateContext) -> GateVerdict:
        if ctx.now is None:
            raise FirewallBreach("gate needs an explicit `now` (no implicit clock)")
        verdicts = {layer.name: layer.run(ctx) for layer in self.layers}
        return GateVerdict(str(as_date(ctx.now)), ctx.subject, verdicts, ctx.fingerprint())

    def admit(self, ctx: GateContext) -> GateVerdict:
        """Evaluate and raise FirewallBreach on any failure: the only sanctioned road into promotion."""
        return self.evaluate(ctx).require()

    def filter_items(self, ctx: GateContext) -> tuple[list, list]:
        """Split ctx.items into (admitted, rejected) by the memory/provenance layers, one item at a time. Rejected items
        are returned, never deleted (section 13: retire is not delete)."""
        good, bad = [], []
        for it in ctx.items or ():
            sub = dataclasses.replace(ctx, items=[it], relevant=frozenset({LayerName.MEMORY, LayerName.PROVENANCE}))
            (good if self.evaluate(sub).passed else bad).append(it)
        return good, bad


# ---------------------------------------------------------------- ledger
class GateLedger:
    """Append-only, hash-chained JSON-lines record of gate verdicts (history is immutable, section 49)."""

    def __init__(self, path):
        self.path = Path(path)

    def _rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(ln) for ln in self.path.read_bytes().decode("utf-8").split("\n") if ln.strip()]

    def append(self, verdict: GateVerdict) -> dict:
        rows = self._rows()
        prev = rows[-1]["chain"] if rows else "GENESIS"
        body = verdict.to_dict()
        row = {"prev": prev, "verdict": body, "chain": stable_hash([prev, body["digest"]], 32)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "ab") as fh:                       # binary: text mode would add CRLF on Windows
            fh.write((json.dumps(row, sort_keys=True) + "\n").encode("utf-8"))
        return row

    def verify(self) -> list[str]:
        errs, prev = [], "GENESIS"
        for i, r in enumerate(self._rows()):
            if r["prev"] != prev:
                errs.append(f"row {i}: broken chain link")
            if r["chain"] != stable_hash([r["prev"], r["verdict"]["digest"]], 32):
                errs.append(f"row {i}: chain hash mismatch (row edited)")
            try:
                verdict_from_dict(r["verdict"])                       # recomputes the verdict digest from the stored content
            except (FirewallBreach, KeyError, ValueError) as e:
                errs.append(f"row {i}: verdict content does not match its digest ({e})")
            prev = r["chain"]
        return errs

    def history(self, subject: str | None = None) -> list[dict]:
        return [r["verdict"] for r in self._rows() if subject is None or r["verdict"]["subject"] == subject]


def summarize_layers(verdicts: Iterable[GateVerdict]) -> pd.DataFrame:
    """Rejection rate per layer over many gate runs: which firewall does the work, which one never fires."""
    rows = []
    for v in verdicts:
        for k, lv in v.verdicts.items():
            rows.append({"layer": k.value, "status": lv.status.value, "failures": len(lv.failures)})
    if not rows:
        return pd.DataFrame(columns=["layer", "runs", "relevant", "rejected", "reject_rate"])
    df = pd.DataFrame(rows)
    g = df.groupby("layer")
    out = pd.DataFrame({"runs": g.size(),
                        "relevant": g["status"].apply(lambda s: int((s != "NOT_RELEVANT").sum())),
                        "rejected": g["status"].apply(lambda s: int(s.isin(["FAIL", "ERROR"]).sum()))})
    out["reject_rate"] = out["rejected"] / out["relevant"].where(out["relevant"] > 0)
    return out.reindex([k.value for k in LAYER_ORDER]).dropna(how="all")


def planted_leak_probe(gate: LearningFirewallGate, clean: GateContext, mutations: Mapping[str, Callable[[GateContext], GateContext]]) -> dict:
    """Prove the gate can fail: the clean context must pass, and every planted mutation must be rejected by at least the
    layer it targets. Returns {mutation: {'rejected': bool, 'layers': [...]}} plus 'clean_passed'."""
    out = {"clean_passed": gate.evaluate(clean).passed}
    for name, mut in mutations.items():
        v = gate.evaluate(mut(clean))
        out[name] = {"rejected": not v.passed, "layers": [k.value for k in v.failed_layers]}
    return out


# ---------------------------------------------------------------- burned evaluation windows
class WindowUseLedger:
    """How many times each evaluation window has been used to judge something. A window judged many times has been
    selected on (a de-facto training set): the count is evidence, and past `max_uses` the window is burned."""

    def __init__(self, max_uses: int = 3):
        self.max_uses = max_uses
        self._uses: dict[tuple[str, str], list[str]] = {}

    def record(self, window, experiment_id: str) -> int:
        s, e = parse_window(window)
        key = (str(s.date()), str(e.date()))
        self._uses.setdefault(key, [])
        if experiment_id not in self._uses[key]:
            self._uses[key].append(experiment_id)
        return len(self._uses[key])

    def uses(self, window) -> int:
        """Uses of any recorded window that overlaps `window` (overlapping windows leak into each other)."""
        return len({x for k, v in self._uses.items() if overlap_days(k, window) > 0 for x in v})

    def burned(self, window) -> bool:
        return self.uses(window) >= self.max_uses

    def findings(self, windows: Iterable, subject: str = "evaluation") -> list[Finding]:
        out = []
        for w in windows:
            n = self.uses(w)
            if n >= self.max_uses:
                out.append(fail(LayerName.EVALUATION, "window-burned", subject,
                                f"window {w} has already been used by {n} experiments (limit {self.max_uses})", uses=n))
            elif n > 0:
                out.append(warn(LayerName.EVALUATION, "window-reused", subject, f"window {w} already used by {n} experiment(s)", uses=n))
        return out


# ---------------------------------------------------------------- what each kind of result must pass
RELEVANCE: dict[str, frozenset] = {
    "pattern": frozenset({LayerName.DATA, LayerName.TIME, LayerName.MEMORY, LayerName.PROVENANCE, LayerName.EXPERIMENT,
                          LayerName.CODE_VERSION}),
    "policy": frozenset(LayerName),
    "knowledge": frozenset({LayerName.TIME, LayerName.MEMORY, LayerName.IDENTITY, LayerName.PROVENANCE, LayerName.CODE_VERSION}),
    "learning_delta": frozenset(LayerName),
    "research_note": frozenset({LayerName.TIME, LayerName.PROVENANCE}),
}


def relevant_layers_for(kind: str) -> frozenset:
    """Layers a result of this kind must pass. An unknown kind gets ALL layers (fail closed, never a lighter check)."""
    return RELEVANCE.get(kind, frozenset(LayerName))


@dataclasses.dataclass(frozen=True)
class PromotionDecision:
    kind: str
    admitted: bool
    verdict_digest: str
    reasons: tuple[str, ...] = ()
    label: ValidationLabel = ValidationLabel.NOT_VALIDATED


def decide_promotion(gate: LearningFirewallGate, ctx: GateContext, kind: str) -> PromotionDecision:
    """Firewall half of a promotion decision (section 45 owns the rest): run the layers this kind requires and report why."""
    ctx = dataclasses.replace(ctx, relevant=relevant_layers_for(kind))
    v = gate.evaluate(ctx)
    reasons = tuple(f"{f.layer}.{f.check}: {f.message}" for f in v.findings(Severity.FAIL))
    return PromotionDecision(kind, v.passed, v.digest(), reasons)


# ---------------------------------------------------------------- comparing and explaining verdicts
def diff_verdicts(before: GateVerdict, after: GateVerdict) -> dict:
    """Which checks newly fail / were fixed between two gate runs: a firewall that quietly stopped firing shows here."""
    def keys(v):
        return {(f.layer.value, f.check, f.subject) for f in v.findings(Severity.FAIL)}
    a, b = keys(before), keys(after)
    return {"newly_failing": sorted(b - a), "fixed": sorted(a - b), "persisting": sorted(a & b),
            "status_changes": {k.value: (before.verdicts[k].status.value, after.verdicts[k].status.value)
                               for k in after.verdicts if k in before.verdicts and before.verdicts[k].status != after.verdicts[k].status}}


def explain(verdict: GateVerdict, limit: int = 3) -> str:
    """Plain-English account of a verdict for a reviewer: one paragraph per failing layer."""
    if verdict.passed:
        return (f"All {len(verdict.relevant)} relevant firewall layers passed at now={verdict.now}. This is not a validation claim: "
                f"the result stays '{verdict.label}'.")
    parts = []
    for k in verdict.failed_layers:
        v = verdict.verdicts[k]
        head = f"{k} ({v.status}):"
        body = " ".join(f"{f.check} on {f.subject} - {f.message}." for f in v.failures[:limit]) or "layer errored without a finding."
        extra = f" (+{len(v.failures) - limit} more)" if len(v.failures) > limit else ""
        parts.append(f"{head} {body}{extra}")
    if not verdict.relevant:
        parts.append("No layer was marked relevant, so nothing was checked; that is a rejection.")
    return "\n".join(parts)


def coverage_matrix(verdicts: Iterable[GateVerdict]) -> pd.DataFrame:
    """Layer x check table of how often each check fired across runs: checks that never fire in a long history are
    suspects for being unable to fail (section 62 'a check that cannot fail is worthless')."""
    rows = []
    for v in verdicts:
        for f in v.findings():
            rows.append({"layer": f.layer.value, "check": f.check, "severity": f.severity.value})
    if not rows:
        return pd.DataFrame(columns=["layer", "check", "INFO", "WARN", "FAIL"])
    t = pd.DataFrame(rows).pivot_table(index=["layer", "check"], columns="severity", aggfunc="size", fill_value=0)
    for c in ("INFO", "WARN", "FAIL"):
        if c not in t:
            t[c] = 0
    return t[["INFO", "WARN", "FAIL"]].reset_index()


def evaluate_many(gate: LearningFirewallGate, contexts: Sequence[GateContext]) -> dict:
    """Run the gate over many results; returns verdicts plus admission rate and per-layer rejection table."""
    vs = [gate.evaluate(c) for c in contexts]
    return {"verdicts": vs, "n": len(vs), "admitted": sum(v.passed for v in vs),
            "admission_rate": (sum(v.passed for v in vs) / len(vs)) if vs else float("nan"),
            "by_layer": summarize_layers(vs)}


# ---------------------------------------------------------------- adapters from existing engine artefacts
def experiment_record_from_stamp(stamp: Mapping, experiment_id: str, **kw) -> ExperimentRecord:
    """Build an ExperimentRecord from an engine.provenance.stamp() dict (config_hash, seed) plus explicit declarations."""
    return ExperimentRecord(experiment_id=experiment_id, config_hash=str(stamp.get("config_hash") or ""), seed=stamp.get("seed"), **kw)


def code_state_from_stamp(stamp: Mapping | str | None, current_hash: str | None = None, **kw) -> CodeState:
    return CodeState(recorded=stamp, current_hash=current_hash, **kw)


def context_from_panel(now, X: pd.DataFrame, y: pd.Series | None = None, horizon: int | None = None, **kw) -> GateContext:
    """Convenience constructor for the common case: a training panel judged at `now`."""
    return GateContext(now=now, X=X, y=y, horizon=horizon, **kw)


def assert_gate_can_fail(gate: LearningFirewallGate, clean: GateContext, planted: GateContext) -> None:
    """Self-check used at startup: the gate must pass a known-clean context and reject a known-planted one."""
    if not gate.evaluate(clean).passed:
        raise FirewallBreach("self-check: the gate rejected a known-clean context (over-strict or broken)")
    if gate.evaluate(planted).passed:
        raise FirewallBreach("self-check: the gate accepted a known-planted leak (it cannot fail)")


# ---------------------------------------------------------------- provenance audit of experiment records
def registry_findings(records: Sequence[Mapping], required: Sequence[str] | None = None, min_complete_share: float = 1.0) -> list[Finding]:
    """Provenance audit (state/research/registry_audit.json, checklist A05 / G01 / K14 / L14): every experiment record must
    carry the registry's REQUIRED fields (experiment id, git commit, canon hash, config hash, data snapshot, seed, ...).
    The audit that found 2 of 53 records complete is exactly what this rejects: a learning result whose record has holes
    cannot be reproduced and so cannot be promoted. Built on engine.registry.REQUIRED / _get / _empty."""
    from engine.registry import REQUIRED, _empty, _get
    req = tuple(required) if required is not None else tuple(REQUIRED)
    L = LayerName.PROVENANCE
    if records is None:
        return [fail(L, "registry-missing", "registry", "no experiment records supplied")]
    if not len(records):
        return [fail(L, "registry-empty", "registry", "the experiment registry holds no records: nothing shows any experiment was recorded")]
    out: list[Finding] = []
    missing_by_field: dict[str, int] = {f: 0 for f in req}
    complete = 0
    ids: dict[str, int] = {}
    for i, r in enumerate(records):
        miss = [f for f in req if _empty(_get(r, f))]
        for f in miss:
            missing_by_field[f] += 1
        complete += not miss
        eid = _get(r, "experiment_id")
        if eid:
            ids[eid] = ids.get(eid, 0) + 1
        if miss:
            out.append(fail(L, "record-incomplete", str(eid or f"record {i}"), f"missing {miss[:6]}{'...' if len(miss) > 6 else ''}", missing=miss))
    dups = sorted(k for k, v in ids.items() if v > 1)
    if dups:
        out.append(fail(L, "duplicate-experiment-id", dups[0], f"{len(dups)} experiment id(s) used by more than one record", n=len(dups)))
    share = complete / len(records)
    if share < min_complete_share:
        worst = sorted(missing_by_field.items(), key=lambda kv: -kv[1])[:4]
        out.append(fail(L, "registry-incomplete", "registry", f"only {complete} of {len(records)} records ({share:.0%}) carry every required field; "
                        f"most often missing: {[f'{k} ({v})' for k, v in worst if v]}", complete=complete, total=len(records)))
    return out


# ---------------------------------------------------------------- catalogue of checks (what each layer can and cannot catch)
@dataclasses.dataclass(frozen=True)
class CheckSpec:
    layer: LayerName
    check: str
    contract: str                              # contract section / checklist item
    catches: str


def _cs(layer, check, contract, catches):
    return CheckSpec(layer, check, contract, catches)


CATALOG: tuple[CheckSpec, ...] = (
    _cs(LayerName.DATA, "duplicate-rows", "H01", "the same (date, ticker) twice"),
    _cs(LayerName.DATA, "forbidden-name", "H05", "a feature named like a label or a future value"),
    _cs(LayerName.DATA, "label-copy", "H05", "a feature that is (nearly) the label"),
    _cs(LayerName.DATA, "implausible-ic", "H05", "a single feature with an unbelievable rank IC"),
    _cs(LayerName.DATA, "infinite-values", "H01", "infinities in the panel"),
    _cs(LayerName.DATA, "mostly-missing", "H01", "columns that are mostly NaN"),
    _cs(LayerName.DATA, "back-adjusted-levels", "H01/leak channel 2", "price levels adjusted for later splits"),
    _cs(LayerName.DATA, "future-listing-columns", "H08/leak channel 8c", "columns for names that list later"),
    _cs(LayerName.DATA, "column-order-reveals-identity", "H08/leak channel 8c", "real alphabetical column order"),
    _cs(LayerName.DATA, "absolute-market-level", "H01/leak channel 8c", "an absolute market level that dates the era"),
    _cs(LayerName.DATA, "survivor-only-panel", "H08/leak channel 1", "a panel with no dead names"),
    _cs(LayerName.DATA, "network-in-blind-run", "H11/leak channel 7", "network access during a blind run"),
    _cs(LayerName.DATA, "cache-past-now", "H11", "a cache holding data after the decision date"),
    _cs(LayerName.TIME, "future-rows", "H01", "feature rows dated after now"),
    _cs(LayerName.TIME, "label-not-closed", "H01", "training labels that close on/after now"),
    _cs(LayerName.TIME, "unpurged-overlap", "H01", "training labels reaching into the test period"),
    _cs(LayerName.TIME, "fill-not-next-session", "H01", "a fill at or before the decision's own close"),
    _cs(LayerName.TIME, "not-yet-published", "H06", "data used before its publication lag elapsed"),
    _cs(LayerName.TIME, "impossibly-fast", "H06", "an availability claim faster than physically possible"),
    _cs(LayerName.MEMORY, "learned-after-now", "H03", "an item learned at/after the decision time"),
    _cs(LayerName.MEMORY, "saw-future-outcomes", "H03", "an item that saw outcomes at/after now"),
    _cs(LayerName.MEMORY, "tainted-by-parent", "H03", "an item derived from a future parent"),
    _cs(LayerName.MEMORY, "saw-sealed-window", "H10", "an item that saw a sealed evaluation window"),
    _cs(LayerName.MEMORY, "outcome-labels-stored", "H03", "outcome labels stored in a payload"),
    _cs(LayerName.MEMORY, "answer-lookup-table", "H03", "a table keyed by ticker/date instead of a situation"),
    _cs(LayerName.MEMORY, "state-trained-on-future-window", "H03/leak channel 4", "learned state fitted on a later window"),
    _cs(LayerName.MEMORY, "state-trained-on-same-window", "H03/leak channel 4", "learned state fitted on the window being played"),
    _cs(LayerName.IDENTITY, "collapse", "H11", "skill that collapses when identities change"),
    _cs(LayerName.IDENTITY, "nondeterministic", "H11", "a learner that disagrees with itself"),
    _cs(LayerName.PROVENANCE, "hash-missing", "A05", "an item without code/data/config/experiment hashes"),
    _cs(LayerName.PROVENANCE, "parent-unresolved", "A13", "a parent that is not in the store"),
    _cs(LayerName.PROVENANCE, "record-incomplete", "G01/K14", "an experiment record missing required fields"),
    _cs(LayerName.PROVENANCE, "registry-incomplete", "G01/K14", "a registry where most records lack provenance"),
    _cs(LayerName.EXPERIMENT, "best-of-n", "H10", "the best of many runs reported without correction"),
    _cs(LayerName.EXPERIMENT, "tuned-on-evaluation", "H10", "fitting or tuning on a sealed evaluation window"),
    _cs(LayerName.EXPERIMENT, "post-hoc-thresholds", "H10", "thresholds changed after seeing results"),
    _cs(LayerName.EVALUATION, "not-sealed", "H10", "an evaluation window that was never sealed"),
    _cs(LayerName.EVALUATION, "train-eval-overlap", "H10", "training and evaluation windows that overlap"),
    _cs(LayerName.EVALUATION, "state-changed-by-evaluation", "H10", "test results that altered training state"),
    _cs(LayerName.EVALUATION, "rerun-identified", "H10", "a learner able to tell a disguised rerun"),
    _cs(LayerName.EVALUATION, "window-burned", "H10", "an evaluation window judged too many times"),
    _cs(LayerName.CODE_VERSION, "stale-result", "H12/section 62 test 12", "a result from code that has since changed"),
    _cs(LayerName.CODE_VERSION, "code-edited-mid-run", "H12/section 62 test 12", "source edited while a worker ran"),
    _cs(LayerName.CODE_VERSION, "unregistered-code", "A13", "a record made by code that was never registered"),
)


def catalogued_checks() -> dict[tuple[str, str], CheckSpec]:
    return {(c.layer.value, c.check): c for c in CATALOG}


def unexercised_checks(verdicts: Iterable[GateVerdict]) -> list[CheckSpec]:
    """Catalogued checks that NEVER produced a finding across the supplied verdicts. In a corpus that plants each defect,
    a check that never fires cannot fail - the situation section 62 calls worthless."""
    seen = {(f.layer.value, f.check) for v in verdicts for f in v.findings()}
    return [c for c in CATALOG if (c.layer.value, c.check) not in seen]


def uncatalogued_findings(verdicts: Iterable[GateVerdict]) -> list[tuple[str, str]]:
    """Checks that fire but appear in no catalogue entry: undocumented behaviour of the gate."""
    known = catalogued_checks()
    return sorted({(f.layer.value, f.check) for v in verdicts for f in v.findings() if (f.layer.value, f.check) not in known})


def verdict_from_dict(d: Mapping) -> GateVerdict:
    """Inverse of GateVerdict.to_dict (for reading the ledger back). Findings keep their evidence; the digest is recomputed
    and must match the stored one, or the row was edited."""
    verdicts: dict[LayerName, LayerVerdict] = {}
    for k, lv in d["layers"].items():
        fs = tuple(Finding(LayerName(f["layer"]), f["check"], Severity(f["severity"]), f["subject"], f["message"], f.get("evidence") or {})
                   for f in lv["findings"])
        verdicts[LayerName(k)] = LayerVerdict(LayerName(k), LayerStatus(lv["status"]), fs, int(lv["n_checked"]))
    gv = GateVerdict(d["now"], d["subject"], verdicts, d.get("context_fingerprint", ""))
    if d.get("digest") and gv.digest() != d["digest"]:
        raise FirewallBreach(f"verdict {d['subject']}@{d['now']} does not match its stored digest (edited or lossy)")
    return gv


def gate_from_names(names: Iterable[str]) -> LearningFirewallGate:
    """A gate with only the named layers (for tests of one layer in isolation). Unknown names are an error, not ignored."""
    want = [LayerName(n) for n in names]
    layers = [layer for layer in default_layers() if layer.name in want]
    if len(layers) != len(want):
        raise ValueError(f"unknown layer in {list(names)}")
    return LearningFirewallGate(layers)


def worst_findings(verdict: GateVerdict, n: int = 5) -> list[Finding]:
    """The most informative failures first: those with numeric evidence, then by layer order."""
    fs = verdict.findings(Severity.FAIL)
    return sorted(fs, key=lambda f: (-len(f.evidence), LAYER_ORDER.index(f.layer), f.check))[:n]
