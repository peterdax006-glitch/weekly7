"""Future-information firewall (contract C62 section 30; checklist H05-H09, I20; section 62 tests 7, 11, 12).

Every learning input must pass eight checks, and the firewall FAILS CLOSED: an input it cannot classify, cannot date, or
cannot show to be available at the decision time is rejected. A suspicious input is better rejected than silently accepted.

    1 timestamp          every observation is dated, and dated before (or, for same-close bars, at) `now`; labels are
                         only usable once they have MATURED strictly before `now`
    2 availability       when was it public? effective date + publication lag (filings, fundamentals, insider forms,
                         macro releases) must not exceed `now`; a claimed availability faster than physically possible
                         is itself a leak
    3 revision           the value used must be the vintage known at `now`; revisable series without a vintage, and
                         prices adjusted for splits that happen later, carry the future
    4 survivorship       the universe as of `now`: nobody listed later, nobody kept alive by surviving to the end
    5 feature provenance every feature declares its source, window and transform; centred windows, full-sample
                         normalisation, label use and implausibly good single features are rejected
    6 memory provenance  learned items must be able to exist at `now` (memory_firewall)
    7 code version       the result comes from the code on disk (firewalls.CodeVersionFirewall)
    8 network / cache    no network during a blind run; caches must not hold data past `now`; sealed paths stay sealed

Built on engine.pit (Calendar, verify_feature_availability, implausible_ic), engine.blind_gates (lookahead_probe,
scan_information, check_public_release), engine.leak_audit (NetworkGuard, macro_revision_risk), engine.isolation
(is_sealed_path). IMPLEMENTED - NOT VALIDATED."""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Iterable, Mapping, Sequence, cast

import numpy as np
import pandas as pd

from .core import FirewallBreach, _StrEnum, as_date, stable_hash
from .firewalls import (FORBIDDEN_NAME, CodeState, Finding, GateContext, LayerName, Severity, fail, info, warn)

CHECKS = ("timestamp", "availability", "revision", "survivorship", "feature_provenance", "memory_provenance",
          "code_version", "network_cache")
CHECK_LAYER = {"timestamp": LayerName.TIME, "availability": LayerName.TIME, "revision": LayerName.DATA,
               "survivorship": LayerName.DATA, "feature_provenance": LayerName.PROVENANCE, "memory_provenance": LayerName.MEMORY,
               "code_version": LayerName.CODE_VERSION, "network_cache": LayerName.DATA}


class InputKind(_StrEnum):
    PRICE = "PRICE"
    FUNDAMENTAL = "FUNDAMENTAL"
    MACRO = "MACRO"
    FILING = "FILING"
    EVENT = "EVENT"
    INSIDER = "INSIDER"
    NEWS = "NEWS"
    FEATURE = "FEATURE"
    LABEL = "LABEL"
    MEMORY = "MEMORY"
    CONFIG = "CONFIG"


# kinds whose values are later restated: without a vintage the point-in-time value cannot be shown
REVISABLE = frozenset({InputKind.FUNDAMENTAL, InputKind.MACRO})
# kinds observable at the close of their own date
SAME_DAY = frozenset({InputKind.PRICE, InputKind.EVENT, InputKind.NEWS, InputKind.FEATURE, InputKind.CONFIG})


@dataclasses.dataclass(frozen=True)
class AvailabilityRule:
    """Publication lag for a kind of input. `min_lag_days` is the fastest physically possible release; anything claiming
    to be available sooner is rejected as impossible. `typical_lag_days` is used when no availability date is given."""
    kind: InputKind
    min_lag_days: int
    typical_lag_days: int
    basis: str = ""


DEFAULT_RULES: dict[InputKind, AvailabilityRule] = {
    InputKind.PRICE: AvailabilityRule(InputKind.PRICE, 0, 0, "daily bar known at its own close"),
    InputKind.EVENT: AvailabilityRule(InputKind.EVENT, 0, 0, "event calendar entry stamped at its date"),
    InputKind.NEWS: AvailabilityRule(InputKind.NEWS, 0, 0, "published timestamp"),
    InputKind.FEATURE: AvailabilityRule(InputKind.FEATURE, 0, 0, "derived at its own date"),
    InputKind.CONFIG: AvailabilityRule(InputKind.CONFIG, 0, 0, "configuration frozen before the run"),
    InputKind.FILING: AvailabilityRule(InputKind.FILING, 1, 2, "SEC acceptance is after the period; visible next session"),
    InputKind.INSIDER: AvailabilityRule(InputKind.INSIDER, 1, 3, "Form 4 due within two business days of the trade"),
    InputKind.FUNDAMENTAL: AvailabilityRule(InputKind.FUNDAMENTAL, 20, 45, "quarter ends, filing follows 20-45 days later"),
    InputKind.MACRO: AvailabilityRule(InputKind.MACRO, 1, 30, "monthly releases lag the reference period"),
    InputKind.LABEL: AvailabilityRule(InputKind.LABEL, 1, 1, "an outcome is known only after its horizon closes"),
    InputKind.MEMORY: AvailabilityRule(InputKind.MEMORY, 1, 1, "a lesson exists after the outcome that taught it"),
}


@dataclasses.dataclass(frozen=True)
class FeatureSpec:
    """Declared provenance of one feature column."""
    name: str
    source: str
    source_kind: InputKind
    lookback: int = 1                        # sessions of history the value uses
    window_end_offset: int = 0               # sessions BEFORE the decision date the newest input sits (0 = same close)
    transform: str = "rolling"               # rolling | ewm | lag | rank | none | global_norm | centered
    uses_label: bool = False
    max_source_ts: Any = None                # newest input timestamp actually used, if recorded
    version: str = ""

    def validate(self) -> list[str]:
        errs = []
        if not self.name or not self.source:
            errs.append("feature spec needs a name and a source")
        if self.lookback < 1:
            errs.append(f"{self.name}: lookback must be >= 1")
        return errs


@dataclasses.dataclass(frozen=True)
class LearningInput:
    """One dated thing the learner is about to consume. Anything that cannot fill `kind` and `timestamp` is rejected."""
    name: str
    kind: InputKind | str
    timestamp: Any = None                    # the date the observation refers to
    available_at: Any = None                 # when it became public (None = derive from the rule)
    vintage: Any = None                      # date of the version of the value used
    latest_vintage_used: Any = None          # newest revision date reflected in the value
    adjusted_for_future_events: bool = False  # e.g. prices restated for splits after `now`
    source: str = ""
    frame: Any = None                        # DataFrame/Series with the actual rows (dates in the index or a `date` column)
    universe: tuple = ()                     # tickers covered
    cached_at: Any = None
    from_network: bool = False
    key: str = ""
    market: Any = None                       # wide market-price frame shown alongside (SPY, ...), for the absolute-level exposure
    real_names: Any = None                   # referee-only map code -> real ticker, to audit column order
    splits: Any = None                       # Series of split dates -> ratio known to the referee

    def parsed_kind(self) -> InputKind | None:
        try:
            return InputKind.parse(self.kind)
        except ValueError:
            return None


@dataclasses.dataclass(frozen=True)
class CheckResult:
    check: str
    ran: bool
    findings: tuple[Finding, ...] = ()
    n_checked: int = 0

    @property
    def passed(self) -> bool:
        return self.ran and not any(f.is_fail for f in self.findings)


@dataclasses.dataclass(frozen=True)
class FutureVerdict:
    now: str
    results: Mapping[str, CheckResult]

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results.values()) and set(self.results) == set(CHECKS)

    @property
    def failed_checks(self) -> tuple[str, ...]:
        return tuple(k for k, r in self.results.items() if not r.passed)

    def findings(self) -> list[Finding]:
        return [f for r in self.results.values() for f in r.findings]

    def digest(self) -> str:
        return stable_hash({k: [r.ran, [f.to_dict() for f in r.findings]] for k, r in self.results.items()})

    def require(self) -> "FutureVerdict":
        if not self.passed:
            why = "; ".join(f"{k}: {(self.results[k].findings[0].message if self.results[k].findings else 'not run')}"
                            for k in self.failed_checks)
            raise FirewallBreach(f"future-information firewall rejected at now={self.now}: {why}")
        return self


# ---------------------------------------------------------------- helpers
def _ts(x) -> pd.Timestamp | None:
    if x is None or (isinstance(x, float) and np.isnan(x)) or x is pd.NaT:
        return None
    try:
        t = pd.Timestamp(x)
    except (ValueError, TypeError):
        return None
    if t is pd.NaT:
        return None
    if t.tzinfo is not None:
        t = t.tz_convert("America/New_York").tz_localize(None)
    return t.normalize()


def frame_dates(frame) -> pd.DatetimeIndex:
    """Dates of a frame/series: the first index level if it is dated, else a `date`/`timestamp` column."""
    if frame is None:
        return pd.DatetimeIndex([])
    idx = frame.index
    lvl = idx.get_level_values(0) if isinstance(idx, pd.MultiIndex) else idx
    if isinstance(lvl, pd.DatetimeIndex):
        return lvl
    for c in ("date", "timestamp", "as_of", "effective"):
        if isinstance(frame, pd.DataFrame) and c in frame:
            return pd.DatetimeIndex(pd.to_datetime(frame[c], errors="coerce"))
    try:
        return pd.DatetimeIndex(pd.to_datetime(lvl, errors="coerce"))
    except (ValueError, TypeError):
        return pd.DatetimeIndex([])


# ---------------------------------------------------------------- 1 timestamp
def check_timestamp(inputs: Sequence[LearningInput], now) -> CheckResult:
    L, out, now_t = CHECK_LAYER["timestamp"], [], cast(Any, _ts(as_date(now)))
    for inp in inputs:
        kind = inp.parsed_kind()
        if kind is None:
            out.append(fail(L, "unknown-kind", inp.name, f"input kind {inp.kind!r} is not recognised: rejected (fail closed)"))
            continue
        t = _ts(inp.timestamp)
        dates = frame_dates(inp.frame)
        if t is None and not len(dates):
            out.append(fail(L, "undated-input", inp.name, "input carries no timestamp and no dated rows"))
            continue
        latest = max([d for d in (t, dates.max() if len(dates) else None) if d is not None and d is not pd.NaT])
        if len(dates) and dates.isna().any():
            out.append(fail(L, "null-row-dates", inp.name, f"{int(dates.isna().sum())} rows without a date"))
        if kind == InputKind.LABEL:
            if latest >= now_t:
                out.append(fail(L, "label-not-matured", inp.name, f"label dated {latest.date()} has not matured before now {now_t.date()}"))
        elif latest > now_t:
            out.append(fail(L, "future-timestamp", inp.name, f"newest observation {latest.date()} is after now {now_t.date()}",
                            n_future=int((dates > now_t).sum()) if len(dates) else 1))
        if len(dates) and (dates > now_t).any() and latest <= now_t:
            out.append(fail(L, "future-rows", inp.name, "rows dated after now hidden behind an earlier input timestamp"))
    return CheckResult("timestamp", True, tuple(out), len(inputs))


# ---------------------------------------------------------------- 2 availability
def check_availability(inputs: Sequence[LearningInput], now, rules: Mapping[InputKind, AvailabilityRule] | None = None) -> CheckResult:
    L, out, now_t = CHECK_LAYER["availability"], [], cast(Any, _ts(as_date(now)))
    rules = rules or DEFAULT_RULES
    for inp in inputs:
        kind = inp.parsed_kind()
        if kind is None:
            continue                                                   # reported by the timestamp check
        rule = rules.get(kind)
        if rule is None:
            out.append(fail(L, "no-availability-rule", inp.name, f"no publication-lag rule for {kind}: cannot show it was public"))
            continue
        t = _ts(inp.timestamp) or (frame_dates(inp.frame).max() if len(frame_dates(inp.frame)) else None)
        if t is None or t is pd.NaT:
            continue
        av = _ts(inp.available_at)
        if av is None:
            av = t + pd.Timedelta(days=rule.typical_lag_days)
            if av > now_t:
                out.append(fail(L, "not-yet-published", inp.name,
                                f"{kind} dated {t.date()} is typically public only by {av.date()} (lag {rule.typical_lag_days}d), after now {now_t.date()}",
                                lag_days=rule.typical_lag_days))
        else:
            if av < t:
                out.append(fail(L, "known-before-it-happened", inp.name, f"available {av.date()} before its own date {t.date()}"))
            if (av - t).days < rule.min_lag_days:
                out.append(fail(L, "impossibly-fast", inp.name,
                                f"claims availability {(av - t).days}d after the date; {kind} cannot be public in under {rule.min_lag_days}d ({rule.basis})"))
            if av > now_t:
                out.append(fail(L, "available-after-now", inp.name, f"available {av.date()}, after now {now_t.date()}"))
        if isinstance(inp.frame, pd.DataFrame) and "available_at" in inp.frame:
            fa = pd.to_datetime(inp.frame["available_at"], errors="coerce")
            if fa.isna().any():
                out.append(fail(L, "row-availability-missing", inp.name, f"{int(fa.isna().sum())} rows without an availability stamp"))
            late = fa > now_t
            if late.any():
                out.append(fail(L, "rows-available-after-now", inp.name, f"{int(late.sum())} rows became public after now", n=int(late.sum())))
            d = frame_dates(inp.frame)
            if len(d) == len(fa) and (fa.to_numpy() < d.to_numpy()).any():
                out.append(fail(L, "rows-known-early", inp.name, "some rows are stamped available before their own date"))
    return CheckResult("availability", True, tuple(out), len(inputs))


def lag_findings(effective, available, kind: InputKind, name: str = "series", rules=None) -> list[Finding]:
    """Row-level lag audit for aligned effective/available date arrays: any lag below the physical minimum is impossible."""
    rule = (rules or DEFAULT_RULES)[kind]
    e, a = pd.DatetimeIndex(pd.to_datetime(effective)), pd.DatetimeIndex(pd.to_datetime(available))
    if len(e) != len(a):
        return [fail(CHECK_LAYER["availability"], "lag-shape", name, "effective/available lengths differ")]
    lag = (a - e).days
    bad = np.asarray(lag < rule.min_lag_days)
    if bad.any():
        return [fail(CHECK_LAYER["availability"], "impossibly-fast-rows", name,
                     f"{int(bad.sum())} rows public in under {rule.min_lag_days}d (fastest {int(lag.min())}d)", n=int(bad.sum()))]
    return []


# ---------------------------------------------------------------- 3 revision
def check_revision(inputs: Sequence[LearningInput], now, unrevised_ok: Iterable[str] = ()) -> CheckResult:
    L, out, now_t = CHECK_LAYER["revision"], [], cast(Any, _ts(as_date(now)))
    ok = set(unrevised_ok)
    for inp in inputs:
        kind = inp.parsed_kind()
        if kind is None:
            continue
        v, lv = _ts(inp.vintage), _ts(inp.latest_vintage_used)
        if v is not None and v > now_t:
            out.append(fail(L, "vintage-after-now", inp.name, f"vintage {v.date()} is after now {now_t.date()}"))
        if lv is not None and lv > now_t:
            out.append(fail(L, "revision-after-now", inp.name, f"value reflects a revision dated {lv.date()}, after now {now_t.date()}"))
        if kind in REVISABLE and v is None and inp.name not in ok:
            out.append(fail(L, "no-vintage", inp.name, f"{kind} series is restated over time and carries no vintage: point-in-time value not shown"))
        if inp.adjusted_for_future_events:
            out.append(fail(L, "adjusted-for-future", inp.name, "values are adjusted for events (splits/dividends) that happen after now"))
        if kind == InputKind.PRICE and isinstance(inp.frame, pd.DataFrame) and not isinstance(inp.frame.index, pd.MultiIndex):
            out.extend(back_adjustment_findings(inp.frame, inp.name, now, splits=inp.splits))
            if inp.market is not None:
                out.extend(f for f in feed_shape_findings(inp.frame, inp.name, now, inp.real_names, inp.market) if f.check == "absolute-market-level")
        if kind == InputKind.MACRO and isinstance(inp.frame, pd.DataFrame):
            try:
                from engine.leak_audit import macro_revision_risk
                risk = macro_revision_risk(list(map(str, inp.frame.columns)))
                for _, r in risk.iterrows():
                    if bool(r["revised"]) and v is None and inp.name not in ok:
                        out.append(fail(L, "revised-macro-column", f"{inp.name}.{r['series']}", f"{r['series']}: {r['reason']}"))
            except ImportError:
                pass
    return CheckResult("revision", True, tuple(out), len(inputs))


# ---------------------------------------------------------------- 4 survivorship
def check_survivorship(inputs: Sequence[LearningInput], now, listings: pd.DataFrame | None = None,
                       expected_annual_attrition: float = 0.03, min_names: int = 30, min_years: float = 2.0) -> CheckResult:
    """Universe checks. `listings` has ticker, list_date, delist_date (NaT while alive). Without listings the check falls
    back to a panel heuristic on any PRICE input: a wide close panel in which nobody ever disappears is survivor-only."""
    L, out, now_t = CHECK_LAYER["survivorship"], [], cast(Any, _ts(as_date(now)))
    n = 0
    if listings is not None:
        need = {"ticker", "list_date", "delist_date"}
        if not need <= set(listings.columns):
            out.append(fail(L, "listings-shape", "listings", f"listings need columns {sorted(need)}"))
            listings = None
        else:
            ld = pd.to_datetime(listings["list_date"], errors="coerce")
            dd = pd.to_datetime(listings["delist_date"], errors="coerce")
            alive = ld.le(now_t) & (dd.isna() | dd.gt(now_t))
            members = set(listings.loc[alive, "ticker"])
            listed = dict(zip(listings["ticker"], ld))
            delisted = dict(zip(listings["ticker"], dd))
            for inp in inputs:
                for tk in inp.universe:
                    n += 1
                    if tk not in listed:
                        out.append(fail(L, "unlisted-ticker", f"{inp.name}/{tk}", "ticker is not in the listing table"))
                    elif pd.notna(listed[tk]) and listed[tk] > now_t:
                        out.append(fail(L, "listed-after-now", f"{inp.name}/{tk}", f"listed {listed[tk].date()}, after now"))
                    elif pd.notna(delisted[tk]) and delisted[tk] <= now_t and tk in inp.universe:
                        pass                                            # a dead name inside the universe is honest, not a leak
                if inp.universe:
                    out.extend(universe_findings(inp.universe, listings, now, inp.name, tolerance=0.2))
                    missing = sorted(members - set(inp.universe))
                    if len(members) >= min_names and len(missing) > 0.2 * len(members):
                        out.append(fail(L, "universe-missing-members", inp.name,
                                        f"{len(missing)} of {len(members)} names alive at now are absent from the universe", n_missing=len(missing)))
                    dead_in = [tk for tk in inp.universe if tk in delisted and pd.notna(delisted[tk]) and delisted[tk] <= now_t]
                    if len(members) >= min_names and not dead_in and (pd.notna(dd) & (dd <= now_t)).sum() >= 5:
                        out.append(fail(L, "no-dead-names", inp.name, "many names delisted before now but none appear in the universe: "
                                        "survivorship-biased selection"))
    for inp in inputs:
        if inp.parsed_kind() != InputKind.PRICE or not isinstance(inp.frame, pd.DataFrame) or isinstance(inp.frame.index, pd.MultiIndex):
            continue
        px = inp.frame[inp.frame.index <= now_t]
        out.extend(f for f in feed_shape_findings(inp.frame, inp.name, now, inp.real_names) if f.layer == L and f.check != "absolute-market-level")
        out.extend(attrition_findings(inp.frame, inp.name, now, min_names=min_names, min_years=min_years))
        if len(px) < 2 or px.shape[1] < min_names:
            n += px.shape[1]
            continue
        n += px.shape[1]
        first = px.apply(lambda c: c.first_valid_index())
        last = px.apply(lambda c: c.last_valid_index())
        years = (px.index[-1] - px.index[0]).days / 365.25
        if years < min_years:
            continue
        started = first <= px.index[0] + pd.Timedelta(days=30)
        ended = last >= px.index[-1] - pd.Timedelta(days=7)
        n0 = int(started.sum())
        if n0 >= min_names:
            surv = float((started & ended).sum()) / n0
            annual = 1.0 - surv ** (1.0 / years)
            if annual < expected_annual_attrition * 0.2:
                out.append(fail(L, "survivor-only-panel", inp.name,
                                f"annual attrition {annual:.2%} over {years:.1f}y vs about {expected_annual_attrition:.1%} expected: dead names dropped",
                                annual_attrition=float(annual)))
    if not n:
        out.append(warn(L, "nothing-to-check", "universe", "no universe, listings or price panel supplied for the survivorship check"))
    return CheckResult("survivorship", True, tuple(out), n)


# ---------------------------------------------------------------- 5 feature provenance
def check_feature_provenance(specs: Sequence[FeatureSpec] | None, now, columns: Sequence[str] | None = None,
                             registered_sources: Iterable[str] | None = None, X: pd.DataFrame | None = None,
                             y: pd.Series | None = None, ic_cap: float = 0.15) -> CheckResult:
    L, out, now_t = CHECK_LAYER["feature_provenance"], [], cast(Any, _ts(as_date(now)))
    if specs is None:
        return CheckResult("feature_provenance", True, (fail(L, "specs-missing", "features", "no feature specs supplied: provenance unknown"),), 0)
    by_name = {s.name: s for s in specs}
    if len(by_name) != len(specs):
        out.append(fail(L, "duplicate-spec", "features", "two specs share a feature name"))
    reg = set(registered_sources) if registered_sources is not None else None
    for s in specs:
        for e in s.validate():
            out.append(fail(L, "spec-invalid", s.name, e))
        if s.uses_label:
            out.append(fail(L, "uses-label", s.name, "feature is computed from the label"))
        if FORBIDDEN_NAME.search(s.name) and not s.name.startswith("m_"):
            out.append(fail(L, "forbidden-name", s.name, "feature name marks a label / future value"))
        if s.window_end_offset < 0:
            out.append(fail(L, "window-reaches-forward", s.name, f"window ends {-s.window_end_offset} sessions AFTER the decision date"))
        if s.transform in ("centered", "global_norm"):
            out.append(fail(L, "future-transform", s.name, f"transform {s.transform!r} uses statistics from after the decision date"))
        ts = _ts(s.max_source_ts)
        if ts is not None and ts > now_t:
            out.append(fail(L, "source-after-now", s.name, f"newest source timestamp {ts.date()} is after now {now_t.date()}"))
        if reg is not None and s.source not in reg:
            out.append(fail(L, "unregistered-source", s.name, f"source {s.source!r} is not a registered data source"))
        if s.source_kind in (InputKind.LABEL, InputKind.MEMORY) and s.source_kind == InputKind.LABEL:
            out.append(fail(L, "label-source", s.name, "a feature's source may not be labels"))
    cols = list(columns) if columns is not None else (list(X.columns) if X is not None else [])
    for c in cols:
        if c not in by_name:
            out.append(fail(L, "undeclared-feature", c, "column has no provenance spec: it could carry anything"))
    for s in specs:
        if cols and s.name not in cols:
            out.append(warn(L, "spec-without-column", s.name, "spec declared but no such column in the data"))
    if X is not None and y is not None and len(X):
        try:
            from engine.pit import implausible_ic
            tab = implausible_ic(X.select_dtypes(include=[np.number]), y, cap=ic_cap)
            for name, r in tab.iterrows():
                if bool(r["flag"]):
                    out.append(fail(L, "implausible-ic", name, f"mean rank IC {r['mean_ic']:.3f} > {ic_cap}: the feature already knows the answer"))
        except ImportError:
            out.append(warn(L, "ic-screen-skipped", "features", "engine.pit not importable"))
    return CheckResult("feature_provenance", True, tuple(out), len(specs))


def feature_future_probe(fn: Callable, data, now, seed: int = 0, trials: int = 3) -> list[Finding]:
    """Prove a feature function cannot see the future: scramble everything after `now` and demand a bit-identical result
    (engine.blind_gates.lookahead_probe). Findings are re-labelled into this firewall's vocabulary."""
    from engine.blind_gates import lookahead_probe
    return [fail(CHECK_LAYER["feature_provenance"], "feature-sees-future", "probe", f.message)
            for f in lookahead_probe(fn, data, pd.Timestamp(as_date(now)), seed=seed, trials=trials) if f.severity == "fail"]


# ---------------------------------------------------------------- 6 memory provenance
def check_memory_provenance(items: Sequence | None, now, store: Sequence | None = None, sealed_windows: Sequence = (),
                            policy=None, env=None) -> CheckResult:
    if items is None:
        return CheckResult("memory_provenance", True, (fail(CHECK_LAYER["memory_provenance"], "items-missing", "memory",
                                                            "no memory supplied: cannot show its provenance"),), 0)
    from .memory_firewall import audit_store
    rep = audit_store(items, now, store=store, sealed_windows=sealed_windows, policy=policy, env=env)
    return CheckResult("memory_provenance", True, rep.findings, len(rep.existences))


# ---------------------------------------------------------------- 7 code version
def check_code_version(state: CodeState | None, items: Sequence | None = None) -> CheckResult:
    from .firewalls import CodeVersionFirewall
    if state is None:
        return CheckResult("code_version", True, (fail(CHECK_LAYER["code_version"], "code-state-missing", "code", "no code state supplied"),), 0)
    findings, n = CodeVersionFirewall().inspect(GateContext(now="1970-01-02", code=state, items=items))
    return CheckResult("code_version", True, tuple(findings), n)


# ---------------------------------------------------------------- 8 network / cache
@dataclasses.dataclass(frozen=True)
class NetworkEvent:
    kind: str                                # network | dns | file_read | cache_read
    target: str
    at: str = ""
    run_id: str = ""


@dataclasses.dataclass(frozen=True)
class CacheEntry:
    path: str
    content_max_date: Any = None            # newest data date the cached object contains
    written_real: str = ""
    key_includes_as_of: bool = False        # was the cache key derived from the decision date?
    key: str = ""


def check_network_cache(events: Sequence[NetworkEvent] | None, cache: Sequence[CacheEntry] | None, now,
                        blind: bool = True, sealed_dir=None) -> CheckResult:
    L, out, now_t = CHECK_LAYER["network_cache"], [], cast(Any, _ts(as_date(now)))
    if events is None or cache is None:
        return CheckResult("network_cache", True, (fail(L, "logs-missing", "io", "network and cache logs not supplied: cannot show the run was closed "
                                                        "to outside information"),), 0)
    for e in events:
        if e.kind in ("network", "dns") and blind:
            out.append(fail(L, "network-in-blind-run", e.target, f"{e.kind} access to {e.target} during a blind run (run {e.run_id or '?'})"))
        if e.kind == "file_read":
            try:
                from engine.isolation import is_sealed_path
                if is_sealed_path(e.target, sealed_dir):
                    out.append(fail(L, "sealed-path-read", e.target, "a sealed evaluation path was read"))
            except ImportError:
                if "sealed" in e.target.lower():
                    out.append(fail(L, "sealed-path-read", e.target, "a sealed-looking path was read"))
    for c in cache:
        mx = _ts(c.content_max_date)
        if mx is None:
            out.append(fail(L, "cache-undated", c.path, "cache content has no recorded newest date: contents unknown"))
        elif mx > now_t:
            out.append(fail(L, "cache-past-now", c.path, f"cache holds data through {mx.date()}, after now {now_t.date()}"))
        elif not c.key_includes_as_of and mx >= now_t:
            out.append(fail(L, "cache-key-blind-to-asof", c.path, "cache key does not include the decision date and the cache reaches now: "
                            "a later run may have written it"))
        if not c.key_includes_as_of and mx is not None and mx <= now_t and c.written_real == "":
            out.append(warn(L, "cache-write-time-unknown", c.path, "no write time recorded for a cache whose key ignores the decision date"))
        try:
            from engine.isolation import is_sealed_path
            if is_sealed_path(c.path, sealed_dir):
                out.append(fail(L, "cache-in-sealed-path", c.path, "cache lives inside the sealed area"))
        except ImportError:
            pass
    if not events and not cache:
        out.append(info(L, "no-io", "io", "no network or cache activity recorded"))
    return CheckResult("network_cache", True, tuple(out), len(events) + len(cache))


def run_under_network_guard(fn: Callable, *args, **kw) -> tuple[Any, list[NetworkEvent]]:
    """Run fn with every socket connection and DNS lookup blocked (engine.leak_audit.NetworkGuard). Returns the result and
    the attempted accesses as NetworkEvents; a blocked attempt is re-raised only if fn lets it escape."""
    from engine.leak_audit import NetworkBlocked, NetworkGuard
    g = NetworkGuard()
    res = None
    with g:
        try:
            res = fn(*args, **kw)
        except NetworkBlocked:
            pass
    return res, [NetworkEvent("network", target) for _, target in g.blocked]


# ---------------------------------------------------------------- the firewall
class FutureFirewall:
    """Runs all eight checks. A check whose inputs were not supplied is reported as NOT RUN and fails the verdict."""

    def __init__(self, rules=None, expected_annual_attrition: float = 0.03, ic_cap: float = 0.15):
        self.rules, self.attr, self.ic_cap = rules, expected_annual_attrition, ic_cap

    def screen(self, now, inputs: Sequence[LearningInput] | None = None, specs: Sequence[FeatureSpec] | None = None,
               items: Sequence | None = None, store: Sequence | None = None, code: CodeState | None = None,
               events: Sequence[NetworkEvent] | None = None, cache: Sequence[CacheEntry] | None = None,
               listings: pd.DataFrame | None = None, X: pd.DataFrame | None = None, y: pd.Series | None = None,
               registered_sources: Iterable[str] | None = None, sealed_windows: Sequence = (), blind: bool = True,
               skip: Iterable[str] = ()) -> FutureVerdict:
        if now is None:
            raise FirewallBreach("future firewall needs an explicit `now`")
        skip = set(skip)
        unknown = skip - set(CHECKS)
        if unknown:
            raise ValueError(f"cannot skip unknown checks {sorted(unknown)}")
        inputs = list(inputs) if inputs is not None else None

        def need_inputs(check: str, fn: Callable[[], CheckResult]) -> CheckResult:
            if inputs is None:
                return CheckResult(check, False, (fail(CHECK_LAYER[check], "not-run", check, f"{check}: no inputs supplied, so it did not run"),))
            try:
                return fn()
            except FirewallBreach as e:
                return CheckResult(check, True, (fail(CHECK_LAYER[check], "breach", check, str(e)),))
            except Exception as e:                                   # noqa: BLE001 - a crashing check must reject, not pass
                return CheckResult(check, False, (fail(CHECK_LAYER[check], "check-error", check, f"{type(e).__name__}: {e}"),))

        results = {
            "timestamp": need_inputs("timestamp", lambda: check_timestamp(cast(Sequence[LearningInput], inputs), now)),
            "availability": need_inputs("availability", lambda: check_availability(cast(Sequence[LearningInput], inputs), now, self.rules)),
            "revision": need_inputs("revision", lambda: check_revision(cast(Sequence[LearningInput], inputs), now)),
            "survivorship": need_inputs("survivorship", lambda: check_survivorship(cast(Sequence[LearningInput], inputs), now, listings, self.attr)),
            "feature_provenance": check_feature_provenance(specs, now, registered_sources=registered_sources, X=X, y=y, ic_cap=self.ic_cap),
            "memory_provenance": check_memory_provenance(items, now, store, sealed_windows),
            "code_version": check_code_version(code, items),
            "network_cache": check_network_cache(events, cache, now, blind),
        }
        for k in skip:                                               # an explicit, recorded skip is not a silent pass
            results[k] = CheckResult(k, True, (warn(CHECK_LAYER[k], "skipped", k, f"{k} explicitly skipped by the caller"),), 0)
        return FutureVerdict(str(as_date(now)), results)

    def admit(self, now, **kw) -> FutureVerdict:
        return self.screen(now, **kw).require()


# ---------------------------------------------------------------- planting leaks (used by tests and the startup self-check)
def plant_future_rows(frame: pd.DataFrame, now, n: int = 3, seed: int = 0) -> pd.DataFrame:
    """Append `n` rows dated after `now` (copies of random existing rows) so a timestamp check has something to catch."""
    rng = np.random.default_rng(seed)
    idx = frame.index
    pick = rng.integers(0, len(frame), n)
    extra = frame.iloc[pick].copy()
    base = pd.Timestamp(as_date(now)) + pd.Timedelta(days=1)
    dates = [base + pd.Timedelta(days=int(i)) for i in range(n)]
    if isinstance(idx, pd.MultiIndex):
        extra.index = pd.MultiIndex.from_arrays([dates, extra.index.get_level_values(1)], names=idx.names)
    else:
        extra.index = pd.DatetimeIndex(dates, name=idx.name)
    return pd.concat([frame, extra])


def plant_label_leak(X: pd.DataFrame, y: pd.Series, name: str = "leaky", noise: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Add a feature that is the label plus a little noise: the canonical planted future leak."""
    rng = np.random.default_rng(seed)
    out = X.copy()
    yy = y.reindex(X.index)
    out[name] = yy.to_numpy(dtype=float) + rng.normal(0, noise * float(np.nanstd(yy)), len(yy))
    return out


def plant_next_period_feature(X: pd.DataFrame, y: pd.Series, name: str = "peek", lead: int = 1) -> pd.DataFrame:
    """Add a feature equal to the label of the SAME ticker `lead` rows later (a shifted label): a leak that hides behind
    an innocent timestamp because every individual value looks like an ordinary return. The rank-IC screen cannot see it when
    returns are not autocorrelated; `lint_feature_source` (negative shift) and `feature_future_probe` (scramble the future) can."""
    out = X.copy()
    yy = y.reindex(X.index)
    out[name] = yy.groupby(level=1).shift(-lead)
    return out


def input_digest(inputs: Sequence[LearningInput]) -> str:
    """Order-independent digest of what the learner was given (name, kind, dates, row count): logged with every run."""
    rows = []
    for i in inputs:
        d = frame_dates(i.frame)
        rows.append([i.name, str(i.kind), str(_ts(i.timestamp)), str(_ts(i.available_at)), str(_ts(i.vintage)),
                     int(len(d)), str(d.max()) if len(d) else ""])
    return stable_hash(sorted(rows))


def firewall_selfcheck(fw: FutureFirewall, clean_kwargs: Mapping[str, Any], mutations: Mapping[str, Callable[[dict], dict]], now) -> dict:
    """The firewall must pass a clean bundle and reject every planted mutation of it; returns which checks fired."""
    res: dict[str, Any] = {"clean": fw.screen(now, **clean_kwargs).passed}
    for name, mut in mutations.items():
        v = fw.screen(now, **mut(dict(clean_kwargs)))
        res[name] = {"rejected": not v.passed, "checks": list(v.failed_checks)}
    return res


# ---------------------------------------------------------------- leak channel 2: back-adjusted price levels
def back_adjustment_findings(close: pd.DataFrame, name: str = "prices", now=None, min_years: float = 8.0, max_ratio: float = 0.25,
                             edge_years: int = 3, splits: pd.Series | None = None) -> list[Finding]:
    """Leak channel 2 (state/research/leak_audit): prices back-adjusted for LATER splits/dividends encode the future in
    every absolute level. As-traded price levels stay roughly flat across decades; back-adjusted levels sit far lower in
    the old years (engine.leak_audit.price_level_drift). A wide date x ticker close panel whose early-year median is under
    `max_ratio` of its late-year median is flagged, as is any supplied split dated after `now` (the adjustment factor of
    every earlier bar contains it)."""
    L = CHECK_LAYER["revision"]
    out: list[Finding] = []
    if close is None or not len(close) or isinstance(close.index, pd.MultiIndex):
        return out
    px = close if now is None else close[close.index <= _ts(as_date(now))]
    if splits is not None and now is not None:
        later = pd.Series(splits).dropna()
        later = later[pd.DatetimeIndex(later.index) > _ts(as_date(now))]
        if len(later):
            out.append(fail(L, "future-split-in-levels", name, f"{len(later)} split(s) dated after now (first {pd.DatetimeIndex(later.index).min().date()}) "
                            "are folded into every earlier price level", n_splits=int(len(later))))
    if len(px) < 2:
        return out
    years = (px.index[-1] - px.index[0]).days / 365.25
    if years < min_years:
        return out
    from engine.leak_audit import price_level_drift
    drift = price_level_drift(px)["median_close"].dropna()
    if len(drift) < 2 * edge_years:
        return out
    early, late = float(drift.iloc[:edge_years].median()), float(drift.iloc[-edge_years:].median())
    if late > 0 and early / late < max_ratio:
        out.append(fail(L, "back-adjusted-levels", name,
                        f"median close of the first {edge_years} years is {early / late:.1%} of the last {edge_years}: prices are adjusted for later "
                        "corporate actions, so price levels (and any level-based tradability rule) carry the future",
                        early_median=early, late_median=late, ratio=early / late))
    return out


# ---------------------------------------------------------------- leak channel 8c: what the feed shows
def feed_shape_findings(close: pd.DataFrame, name: str = "feed", now=None, real_names: Mapping[str, str] | None = None,
                        market: pd.DataFrame | None = None, market_cols: Sequence[str] = ("SPY",), rebase_level: float = 100.0,
                        order_rho_cut: float = 0.5) -> list[Finding]:
    """Leak channel 8c: the exposed feed itself carries information. Three exposures (engine.leak_audit.feed_exposure):
      (a) column order equal to the REAL alphabetical order of the tickers (identity survives a rename),
      (b) columns for names that have not listed yet at `now` (future IPOs),
      (c) an absolute market level (SPY starts at 6 in 1975, 400 in 1995) that identifies the year.
    (a) needs `real_names` (code -> real ticker) held by the referee; without it the check is reported as unauditable."""
    L = CHECK_LAYER["survivorship"]
    out: list[Finding] = []
    if close is None or not len(close):
        return out
    vis = close if now is None else close[close.index <= _ts(as_date(now))]
    if len(vis):
        empty = [c for c in vis.columns if not vis[c].notna().any()]
        if empty:
            out.append(fail(L, "future-listing-columns", name, f"{len(empty)} column(s) hold no price yet at now (names that list later): "
                            f"their presence reveals the future universe", n=len(empty), first=str(empty[0])))
    if real_names is not None:
        real = pd.Series([real_names.get(c) for c in close.columns], dtype=object)
        if real.isna().any():
            out.append(fail(L, "real-names-incomplete", name, f"{int(real.isna().sum())} columns have no real-name entry"))
        else:
            rho = pd.Series(np.arange(len(real)), dtype=float).corr(real.rank(method="first"), method="spearman")
            if abs(rho) >= order_rho_cut:
                out.append(fail(L, "column-order-reveals-identity", name,
                                f"column order correlates {rho:.2f} with the real alphabetical ticker order", rho=float(rho)))
    elif len(close.columns) > 2:
        out.append(warn(L, "column-order-unauditable", name, "no real_names supplied: alphabetical-order exposure cannot be audited"))
    if market is not None and len(market):
        mv = market if now is None else market[market.index <= _ts(as_date(now))]
        for c in market_cols:
            if c in mv:
                s = mv[c].dropna()
                if len(s) and abs(float(s.iloc[0]) - rebase_level) > 1e-6:
                    out.append(fail(CHECK_LAYER["revision"], "absolute-market-level", f"{name}.{c}",
                                    f"{c} starts at {float(s.iloc[0]):.2f} instead of the rebased {rebase_level:g}: the absolute level identifies the era",
                                    first_level=float(s.iloc[0])))
    return out


# ---------------------------------------------------------------- vintages
class RevisionLedger:
    """Every release of every observation of a revisable series: (series, observation date, release date, value). Answers
    'what did this number say at `now`?' from release dates alone - never from today's (latest) value."""

    def __init__(self):
        self.rows: list[tuple[str, pd.Timestamp, pd.Timestamp, float]] = []

    def add(self, series: str, obs_date, release_date, value: float) -> None:
        o, r = _ts(obs_date), _ts(release_date)
        if o is None or r is None:
            raise ValueError("observation and release dates are required")
        if r < o:
            raise ValueError(f"{series}: released {r.date()} before it was observed {o.date()}")
        self.rows.append((series, o, r, float(value)))

    def _for(self, series: str, obs) -> list[tuple[pd.Timestamp, float]]:
        o = _ts(obs)
        return sorted((r, v) for s, ob, r, v in self.rows if s == series and ob == o)

    def as_of(self, series: str, obs_date, now) -> float | None:
        """Newest release of the observation that was public by `now`; None if it was not yet released."""
        n: Any = _ts(as_date(now))
        known = [(r, v) for r, v in self._for(series, obs_date) if r <= n]
        return known[-1][1] if known else None

    def first_release(self, series: str, obs_date) -> float | None:
        rs = self._for(series, obs_date)
        return rs[0][1] if rs else None

    def panel_as_of(self, series: str, now) -> pd.Series:
        obs = sorted({ob for s, ob, _, _ in self.rows if s == series})
        vals = {ob: self.as_of(series, ob, now) for ob in obs}
        return pd.Series({k: v for k, v in vals.items() if v is not None}, dtype=float)

    def revision_size(self, series: str) -> pd.DataFrame:
        """First vs latest release per observation: how much a leak of the latest vintage could matter."""
        rows = []
        for ob in sorted({o for s, o, _, _ in self.rows if s == series}):
            rs = self._for(series, ob)
            rows.append({"obs": ob, "first": rs[0][1], "latest": rs[-1][1], "n_releases": len(rs), "abs_revision": abs(rs[-1][1] - rs[0][1])})
        return pd.DataFrame(rows, columns=["obs", "first", "latest", "n_releases", "abs_revision"])

    def audit_used(self, series: str, used: pd.Series, now, tol: float = 1e-9) -> list[Finding]:
        """`used` = the values a pipeline consumed (index = observation dates). Any value that differs from what was public
        at `now` and equals a LATER release is a proven future-vintage leak."""
        L = CHECK_LAYER["revision"]
        out: list[Finding] = []
        n: Any = _ts(as_date(now))
        for ob, v in used.items():
            if _ts(ob) is None or cast(Any, _ts(ob)) > n:
                out.append(fail(L, "observation-after-now", series, f"observation {ob} is after now"))
                continue
            known = self.as_of(series, ob, now)
            if known is None:
                out.append(fail(L, "used-unreleased-value", series, f"{cast(Any, _ts(ob)).date()}: value used before any release was public"))
            elif abs(float(v) - known) > tol:
                later = [x for r, x in self._for(series, ob) if r > n and abs(x - float(v)) <= tol]
                out.append(fail(L, "used-future-vintage" if later else "value-not-in-any-vintage", series,
                                f"{cast(Any, _ts(ob)).date()}: used {float(v):.6g} but the value public at {n.date()} was {known:.6g}"))
        return out


# ---------------------------------------------------------------- label maturity
def label_maturity_mask(row_dates, horizon: int, now, calendar=None, entry_lag: int = 1) -> np.ndarray:
    """True for rows whose forward label has fully closed strictly before `now` (usable as a training label)."""
    from engine.pit import Calendar, label_close_dates
    lc = label_close_dates(pd.DatetimeIndex(row_dates), int(horizon), calendar or Calendar(), entry_lag)
    return np.asarray(lc < _ts(as_date(now)))


def label_maturity_findings(row_dates, horizon: int, now, calendar=None, entry_lag: int = 1, name: str = "labels") -> list[Finding]:
    m = label_maturity_mask(row_dates, horizon, now, calendar, entry_lag)
    if m.all():
        return []
    return [fail(CHECK_LAYER["timestamp"], "labels-not-matured", name,
                 f"{int((~m).sum())} of {len(m)} rows have labels that close on/after {as_date(now)} (horizon {horizon}, entry lag {entry_lag})",
                 n=int((~m).sum()))]


def publication_lag_report(effective, available, calendar=None, by=None, long_lag: int = 2) -> pd.DataFrame:
    """Lag distribution in sessions between when something happened and when it became public (engine.pit.lag_profile),
    with impossible (negative) and long lags counted: the evidence behind the availability rules."""
    from engine.pit import Calendar, lag_profile
    return lag_profile(pd.DatetimeIndex(pd.to_datetime(effective)), pd.DatetimeIndex(pd.to_datetime(available)), calendar or Calendar(),
                       by=by, long_lag=long_lag)


# ---------------------------------------------------------------- static look-ahead linter
@dataclasses.dataclass(frozen=True)
class LintHit:
    rule: str
    line: int
    snippet: str


_LABEL_IDENTS = frozenset({"y", "label", "labels", "target", "targets", "fwd", "fwd_ret", "forward_return", "future_return", "y_true", "outcome"})
_BACKFILL_CALLS = frozenset({"bfill", "backfill"})
_FULL_SAMPLE = frozenset({"mean", "std", "max", "min", "median", "quantile", "var", "sum"})
_WINDOWED = frozenset({"rolling", "expanding", "ewm", "groupby"})


def _neg_const(node) -> bool:
    import ast
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        return isinstance(node.operand.value, (int, float)) and node.operand.value > 0
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and node.value < 0


def _chain_names(node) -> list[str]:
    import ast
    names = []
    while isinstance(node, (ast.Call, ast.Attribute)):
        if isinstance(node, ast.Call):
            node = node.func
        else:
            names.append(node.attr)
            node = node.value
    return names


def lint_feature_source(src: str) -> list[LintHit]:
    """Static look-ahead lint of a feature function's source. Catches the patterns that let the future in through the
    arithmetic rather than the data: negative shifts / diffs / pct_change, centred windows, backfill and interpolation,
    np.roll with a negative shift, full-sample statistics used to normalise, and references to label-named variables.
    A lint hit is a rejection reason, never proof of safety when absent (feature_future_probe is the dynamic test)."""
    import ast
    tree = ast.parse(src)
    lines = src.splitlines()
    hits: list[LintHit] = []

    def add(rule, node):
        ln = getattr(node, "lineno", 0)
        hits.append(LintHit(rule, ln, lines[ln - 1].strip() if 0 < ln <= len(lines) else ""))

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            fn = node.func.attr
            kw = {k.arg: k.value for k in node.keywords if k.arg}
            if fn in ("shift", "diff", "pct_change"):
                arg = node.args[0] if node.args else kw.get("periods")
                if arg is not None and _neg_const(arg):
                    add(f"negative-{fn}", node)
            if fn == "rolling" and isinstance(kw.get("center"), ast.Constant) and cast(ast.Constant, kw["center"]).value is True:
                add("centered-window", node)
            if fn in _BACKFILL_CALLS or fn == "interpolate":
                add("backfill" if fn != "interpolate" else "interpolate", node)
            if fn == "fillna" and isinstance(kw.get("method"), ast.Constant) and str(cast(ast.Constant, kw["method"]).value) in _BACKFILL_CALLS:
                add("backfill", node)
            if fn == "roll" and isinstance(node.func.value, ast.Name) and node.func.value.id in ("np", "numpy"):
                sh = node.args[1] if len(node.args) > 1 else kw.get("shift")
                if sh is not None and _neg_const(sh):
                    add("negative-roll", node)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Sub, ast.Div)):
            for side in (node.left, node.right):
                for sub in ast.walk(side):
                    if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr in _FULL_SAMPLE:
                        chain = _chain_names(sub)
                        if not (set(chain) & _WINDOWED) and isinstance(sub.func.value, ast.Name):
                            add("full-sample-statistic", sub)
        if isinstance(node, ast.Name) and node.id in _LABEL_IDENTS and isinstance(node.ctx, ast.Load):
            add("label-reference", node)
    seen, uniq = set(), []
    for h in hits:
        if (h.rule, h.line) not in seen:
            seen.add((h.rule, h.line))
            uniq.append(h)
    return sorted(uniq, key=lambda h: (h.line, h.rule))


def lint_callable(fn: Callable) -> list[LintHit]:
    import inspect
    import textwrap
    try:
        src = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError) as e:
        return [LintHit("source-unavailable", 0, f"cannot read source: {e}")]
    return lint_feature_source(src)


_RULE_TO_TRANSFORM = {"centered-window": "centered", "full-sample-statistic": "global_norm"}


def spec_from_lint(name: str, source: str, kind: InputKind, hits: Sequence[LintHit], lookback: int = 1) -> FeatureSpec:
    """Fill a FeatureSpec from lint results so that provenance is derived from the code, not declared by hand: the
    dangerous properties can only be set by what the linter found."""
    rules = {h.rule for h in hits}
    transform = next((t for r, t in _RULE_TO_TRANSFORM.items() if r in rules), "rolling")
    neg = any(r.startswith("negative-") or r in ("backfill", "interpolate") for r in rules)
    return FeatureSpec(name, source, kind, lookback=lookback, window_end_offset=-1 if neg else 0, transform=transform,
                       uses_label="label-reference" in rules)


def lint_findings(hits: Sequence[LintHit], feature: str) -> list[Finding]:
    return [fail(CHECK_LAYER["feature_provenance"], f"lint-{h.rule}", feature, f"line {h.line}: {h.snippet}") for h in hits
            if h.rule != "source-unavailable"] + [warn(CHECK_LAYER["feature_provenance"], "lint-unavailable", feature, h.snippet)
                                                   for h in hits if h.rule == "source-unavailable"]


# ---------------------------------------------------------------- caches on disk
def cache_entries_from_files(paths: Iterable, content_max_date: Callable[[Any], Any], key_includes_as_of: bool = False) -> list[CacheEntry]:
    """Describe real cache files: the write time comes from the file system, the newest data date from the caller's reader
    (it must open the file and look; a name is not evidence)."""
    from pathlib import Path
    out = []
    for p in paths:
        pp = Path(p)
        written = pd.Timestamp(pp.stat().st_mtime, unit="s").isoformat() if pp.exists() else ""
        try:
            mx = content_max_date(pp)
        except Exception:                                            # noqa: BLE001 - unreadable cache = unknown contents
            mx = None
        out.append(CacheEntry(str(pp), mx, written, key_includes_as_of, pp.name))
    return out


# ---------------------------------------------------------------- reporting
def verdict_findings(v: FutureVerdict) -> list[Finding]:
    """All findings, failures first, with a NOT-RUN finding for every check that never ran."""
    out = v.findings()
    for c in CHECKS:
        if c not in v.results:
            out.append(fail(CHECK_LAYER[c], "not-run", c, f"{c} produced no result"))
    return sorted(out, key=lambda f: (not f.is_fail, f.layer.value, f.check))


def verdict_markdown(v: FutureVerdict, limit: int = 30) -> str:
    lines = [f"# Future-information firewall at now={v.now}", f"overall: {'PASS' if v.passed else 'REJECT'}   (IMPLEMENTED - NOT VALIDATED)", "",
             "| check | ran | passed | findings | failures |", "|---|---|---|---|---|"]
    for c in CHECKS:
        r = v.results.get(c)
        lines.append(f"| {c} | {r.ran if r else False} | {r.passed if r else False} | {len(r.findings) if r else 0} | "
                     f"{sum(f.is_fail for f in r.findings) if r else 0} |")
    for f in verdict_findings(v)[:limit]:
        if f.severity != Severity.INFO:
            lines.append(f"- {f}")
    return "\n".join(lines)


def inputs_from_panel(X: pd.DataFrame, now, y: pd.Series | None = None, source: str = "panel", kind: InputKind = InputKind.FEATURE,
                      label_matures=None) -> list[LearningInput]:
    """Wrap a training panel as LearningInputs (features, and the labels as a LABEL input) so it can be screened."""
    ins = [LearningInput("features", kind, timestamp=frame_dates(X).max(), source=source, frame=X,
                         universe=tuple(sorted(set(X.index.get_level_values(1)))) if isinstance(X.index, pd.MultiIndex) else ())]
    if y is not None:
        ins.append(LearningInput("labels", InputKind.LABEL, timestamp=label_matures if label_matures is not None else (frame_dates(y.dropna()).max() if y.notna().any() else None),
                              source=source, frame=y))
    return ins


# ---------------------------------------------------------------- survivorship, deeper
def attrition_findings(close: pd.DataFrame, name: str = "prices", now=None, min_names: int = 30, min_years: float = 5.0,
                       min_exit_rate: float = 0.002) -> list[Finding]:
    """Per-year exits from a wide close panel (engine.leak_audit.attrition_profile). A real universe loses names every year;
    a panel with almost no terminal exits in a long history was built from today's listings (survivor-only)."""
    L = CHECK_LAYER["survivorship"]
    if close is None or isinstance(close.index, pd.MultiIndex) or close.shape[1] < min_names:
        return []
    px = close if now is None else close[close.index <= _ts(as_date(now))]
    if len(px) < 2 or (px.index[-1] - px.index[0]).days / 365.25 < min_years:
        return []
    from engine.leak_audit import attrition_profile
    prof = attrition_profile(px)
    prof = prof[prof["n_alive"] >= min_names]
    if len(prof) < 3:
        return []
    rate = float(prof["n_exit"].sum() / prof["n_alive"].sum())
    if rate < min_exit_rate:
        return [fail(L, "no-exits-in-history", name, f"only {int(prof['n_exit'].sum())} terminal exits over {len(prof)} years of {prof['n_alive'].mean():.0f} "
                     f"names (rate {rate:.3%}/yr): dead names are absent from the panel", exit_rate=rate)]
    return []


def universe_as_of(listings: pd.DataFrame, now) -> list[str]:
    """Members alive at `now` from a listing table (ticker, list_date, delist_date): listed on/before now, not yet delisted.
    A delisting dated after `now` is not known at `now` (engine.pit.Listings semantics)."""
    from engine.pit import Listings
    return Listings(listings).members(_ts(as_date(now)))


def universe_findings(universe: Iterable[str], listings: pd.DataFrame, now, name: str = "universe", tolerance: float = 0.0) -> list[Finding]:
    """The chosen universe must equal the membership at `now`: names that list later (future IPOs) and names that were
    already gone (ghosts) are both errors; so is a universe missing live members beyond `tolerance` of their number."""
    L = CHECK_LAYER["survivorship"]
    uni, members = set(universe), set(universe_as_of(listings, now))
    out: list[Finding] = []
    ld = dict(zip(listings["ticker"], pd.to_datetime(listings["list_date"])))
    dd = dict(zip(listings["ticker"], pd.to_datetime(listings["delist_date"])))
    n = _ts(as_date(now))
    future = sorted(t for t in uni if t in ld and pd.notna(ld[t]) and ld[t] > n)
    ghosts = sorted(t for t in uni if t in dd and pd.notna(dd[t]) and dd[t] <= n)
    unknown = sorted(uni - set(ld))
    missing = sorted(members - uni)
    if future:
        out.append(fail(L, "future-listings-in-universe", name, f"{len(future)} names list after now (e.g. {future[:3]})", n=len(future)))
    if ghosts:
        out.append(fail(L, "delisted-names-in-universe", name, f"{len(ghosts)} names were already delisted at now (e.g. {ghosts[:3]})", n=len(ghosts)))
    if unknown:
        out.append(fail(L, "unlisted-names-in-universe", name, f"{len(unknown)} names are in no listing table (e.g. {unknown[:3]})", n=len(unknown)))
    if members and len(missing) > tolerance * len(members):
        out.append(fail(L, "members-missing", name, f"{len(missing)} of {len(members)} live members are absent: the universe was cut with hindsight",
                        n=len(missing)))
    return out


# ---------------------------------------------------------------- source registry
@dataclasses.dataclass(frozen=True)
class SourceSpec:
    name: str
    kind: InputKind
    revisable: bool = False
    vintage_available: bool = False
    lag_days: int | None = None            # overrides the kind's typical lag
    notes: str = ""


class SourceRegistry:
    """The data sources a learner may consume, with what is known about each one's timing. A feature may only cite a
    registered source; a source declared revisable without vintages is unusable and says so."""

    def __init__(self, specs: Iterable[SourceSpec] = ()):
        self._s: dict[str, SourceSpec] = {}
        for sp in specs:
            self.register(sp)

    def register(self, spec: SourceSpec) -> None:
        if spec.name in self._s:
            raise ValueError(f"source {spec.name!r} already registered")
        if spec.revisable and not spec.vintage_available and spec.kind not in REVISABLE:
            raise ValueError(f"{spec.name}: kind {spec.kind} is not one of the revisable kinds {sorted(map(str, REVISABLE))}")
        self._s[spec.name] = spec

    def names(self) -> set[str]:
        return set(self._s)

    def get(self, name: str) -> SourceSpec | None:
        return self._s.get(name)

    def rule_for(self, name: str) -> AvailabilityRule | None:
        sp = self._s.get(name)
        if sp is None:
            return None
        base = DEFAULT_RULES[sp.kind]
        return base if sp.lag_days is None else dataclasses.replace(base, typical_lag_days=max(sp.lag_days, base.min_lag_days))

    def unusable(self) -> list[str]:
        """Revisable sources with no vintages: their point-in-time value cannot be produced."""
        return sorted(n for n, s in self._s.items() if (s.revisable or s.kind in REVISABLE) and not s.vintage_available)

    def findings(self, specs: Iterable[FeatureSpec]) -> list[Finding]:
        L = CHECK_LAYER["feature_provenance"]
        out = []
        for s in specs:
            sp = self._s.get(s.source)
            if sp is None:
                out.append(fail(L, "unregistered-source", s.name, f"source {s.source!r} is not in the registry"))
            elif sp.kind != s.source_kind:
                out.append(fail(L, "source-kind-mismatch", s.name, f"declared {s.source_kind} but {s.source!r} is registered as {sp.kind}"))
            elif (sp.revisable or sp.kind in REVISABLE) and not sp.vintage_available:
                out.append(fail(L, "revisable-source-no-vintage", s.name, f"{s.source!r} is restated over time and has no vintages"))
        return out


# ---------------------------------------------------------------- run-time IO (channels 7 and 8d)
def record_run_io(fn: Callable, *args, blind: bool = True, root=None, **kw) -> tuple[Any, list[NetworkEvent]]:
    """Execute fn under the network guard AND a file-access recorder; return its result and every network attempt plus every
    project data/state file it opened, as NetworkEvents ready for `check_network_cache`. Opened files under data/cache are
    reported as cache_read, files under state/ (sealed windows live there) as file_read."""
    from engine.leak_audit import FileAccessRecorder, NetworkBlocked, NetworkGuard
    g = NetworkGuard()
    rec = FileAccessRecorder()
    res = None
    with g, rec:
        try:
            res = fn(*args, **kw)
        except NetworkBlocked:
            pass
    ev = [NetworkEvent("network", target) for _, target in g.blocked]
    files = rec.project_files(root)
    ev += [NetworkEvent("cache_read", p) for p in files["data_cache"]]
    ev += [NetworkEvent("file_read", p) for p in files["state"]]
    return res, ev


def cache_events_findings(events: Sequence[NetworkEvent], allowed_prefixes: Sequence[str] = ()) -> list[Finding]:
    """Cache reads outside the allowed prefixes: the run reached data it was not given (channel 8d)."""
    L = CHECK_LAYER["network_cache"]
    return [fail(L, "unlisted-cache-read", e.target, "run opened a cache file that is not on the allowed list")
            for e in events if e.kind == "cache_read" and not any(e.target.startswith(p) for p in allowed_prefixes)]


# ---------------------------------------------------------------- leak-channel report
CHANNEL_OF_CHECK: dict[str, str] = {
    "survivor-only-panel": "1", "no-exits-in-history": "1", "no-dead-names": "1", "future-listings-in-universe": "1",
    "delisted-names-in-universe": "1", "members-missing": "1",
    "back-adjusted-levels": "2", "future-split-in-levels": "2", "adjusted-for-future": "2",
    "revised-macro-column": "5", "no-vintage": "5", "used-future-vintage": "5", "revision-after-now": "5",
    "network-in-blind-run": "7",
    "feature-sees-future": "8b", "labels-not-matured": "8b", "label-not-matured": "8b", "window-reaches-forward": "8b",
    "future-transform": "8b", "uses-label": "8b", "implausible-ic": "8b",
    "future-listing-columns": "8c", "column-order-reveals-identity": "8c", "absolute-market-level": "8c",
    "sealed-path-read": "8d", "unlisted-cache-read": "8d", "cache-past-now": "8d",
    "saw-future-outcomes": "4", "tainted-by-parent": "4", "state-trained-on-future-window": "4", "state-trained-on-same-window": "4",
    "defaults-tuned-on-window": "4",
}


def channel_status(findings: Iterable[Finding]) -> dict[str, str]:
    """Map failures onto the leak-audit channels (state/research/leak_audit): LEAK when a check tied to the channel failed,
    CLEAN otherwise for channels this firewall can see, UNOBSERVED for those it has no check for (never CLEAN by default)."""
    seen = {c: "CLEAN" for c in sorted(set(CHANNEL_OF_CHECK.values()))}
    for f in findings:
        base = f.check.split(":")[0]
        ch = CHANNEL_OF_CHECK.get(base) or (CHANNEL_OF_CHECK.get(base.removeprefix("lint-")) if base.startswith("lint-") else None)
        if base.startswith("lint-negative") or base in ("lint-centered-window", "lint-backfill", "lint-interpolate"):
            ch = "8b"
        if ch and f.is_fail:
            seen[ch] = "LEAK"
    for ch in ("3", "6", "8a", "8e"):
        seen.setdefault(ch, "UNOBSERVED")
    return dict(sorted(seen.items()))


def pipeline_future_invariance(build: Callable[[Any], Any], data, now, seed: int = 0, trials: int = 3) -> list[Finding]:
    """A whole learning pipeline must give a bit-identical result when everything after `now` is scrambled versus deleted
    (memory_firewall.memory_future_invariance): the strongest dynamic test that the pipeline never touched the future."""
    from .memory_firewall import memory_future_invariance
    return [dataclasses.replace(f, layer=CHECK_LAYER["feature_provenance"],
                                check="pipeline-sees-future" if f.check == "memory-depends-on-future" else f.check)
            for f in memory_future_invariance(build, data, now, seed, trials) if f.is_fail]


# ---------------------------------------------------------------- feature availability, calendar and text leaks
def feature_availability_findings(row_dates, avail: pd.DataFrame | pd.Series, name: str = "features") -> list[Finding]:
    """Per-feature availability stamps against the decision date of each row (engine.pit.verify_feature_availability): a
    feature built from an input that became public after the row's date, or carrying no stamp, is a leak or unprovable."""
    from engine.pit import LookAheadError, verify_feature_availability
    try:
        verify_feature_availability(row_dates, avail)
    except LookAheadError as e:
        return [fail(CHECK_LAYER["availability"], "feature-input-not-yet-public", name, str(e))]
    except ValueError as e:
        return [fail(CHECK_LAYER["availability"], "availability-shape", name, str(e))]
    return []


def session_date_findings(dates, name: str = "dates", calendar=None) -> list[Finding]:
    """Observations dated on weekends or exchange holidays (with the session calendar in hand) mean the timeline was rebuilt
    or shifted by a fractional week: a disguise that breaks the calendar leaks that it is a disguise."""
    from engine.pit import Calendar
    d = pd.DatetimeIndex(pd.to_datetime(dates)).dropna()
    if not len(d):
        return []
    bad = ~(calendar or Calendar()).is_session(d)
    if bad.any():
        return [fail(CHECK_LAYER["timestamp"], "non-session-dates", name, f"{int(bad.sum())} of {len(d)} dates are not trading sessions "
                     f"(first {d[bad].min().date()})", n=int(bad.sum()))]
    return []


def text_leak_findings(texts: Iterable[str], real_tickers: Iterable[str] = (), real_years: Iterable[int] = (), name: str = "text") -> list[Finding]:
    """Free text served to a learner (names, notes, headlines) must not contain real tickers or real years
    (engine.blind_gates.scan_text_for_leaks). Returns one finding per offending text."""
    from engine.blind_gates import scan_text_for_leaks
    out = []
    for i, t in enumerate(texts):
        hits = [f for f in scan_text_for_leaks(str(t), tuple(real_tickers), tuple(real_years)) if f.severity == "fail"]
        for h in hits:
            out.append(fail(CHECK_LAYER["feature_provenance"], "text-leak", f"{name}[{i}]", h.message))
    return out


def screen_panel(fw: FutureFirewall, now, X: pd.DataFrame, y: pd.Series | None, specs: Sequence[FeatureSpec], horizon: int, **kw) -> FutureVerdict:
    """Screen a training panel end to end: wrap it as inputs, mark label maturity from the horizon (a label is available only once
    its horizon has closed), and run all eight checks. Extra keyword arguments pass to `FutureFirewall.screen`."""
    from engine.pit import Calendar, label_close_dates
    matures = None
    if y is not None and y.notna().any():
        rows = X.index[y.reindex(X.index).notna().to_numpy()]
        matures = label_close_dates(pd.DatetimeIndex(rows.get_level_values(0)), int(horizon), kw.pop("calendar", None) or Calendar()).max()
    ins = inputs_from_panel(X, now, y, label_matures=matures) + list(kw.pop("extra_inputs", ()))
    return fw.screen(now, inputs=ins, specs=specs, X=X, y=y, **kw)


# ---------------------------------------------------------------- which inputs sit closest to the boundary
def boundary_margins(inputs: Sequence[LearningInput], now, rules: Mapping[InputKind, AvailabilityRule] | None = None) -> pd.DataFrame:
    """Per input: newest observation date, the date it becomes public under its rule, and the margin in days before `now`.
    Small or negative margins are where a small timing error becomes a leak; reviewers read this table first."""
    rules = rules or DEFAULT_RULES
    n = _ts(as_date(now))
    rows = []
    for inp in inputs:
        kind = inp.parsed_kind()
        d = frame_dates(inp.frame)
        newest = _ts(inp.timestamp) or (d.max() if len(d) else None)
        if kind is None or newest is None or newest is pd.NaT:
            rows.append({"input": inp.name, "kind": str(inp.kind), "newest": None, "public_at": None, "margin_days": float("nan")})
            continue
        av = _ts(inp.available_at) or (newest + pd.Timedelta(days=rules[kind].typical_lag_days))
        rows.append({"input": inp.name, "kind": str(kind), "newest": newest, "public_at": av, "margin_days": float((n - av).days)})
    return pd.DataFrame(rows, columns=["input", "kind", "newest", "public_at", "margin_days"]).sort_values("margin_days").reset_index(drop=True)


def rules_from_registry(reg: SourceRegistry, base: Mapping[InputKind, AvailabilityRule] | None = None) -> dict[str, AvailabilityRule]:
    """Per-source rules (source name -> AvailabilityRule) from a registry, falling back to the kind defaults."""
    return {n: cast(AvailabilityRule, reg.rule_for(n)) for n in sorted(reg.names())}


def input_manifest(inputs: Sequence[LearningInput], now) -> dict:
    """Plain-dict description of what a learner was given, for run logs: per input kind, counts, newest dates and the digest."""
    kinds: dict[str, dict] = {}
    for i in inputs:
        d = frame_dates(i.frame)
        k = kinds.setdefault(str(i.kind), {"n": 0, "rows": 0, "newest": None})
        k["n"] += 1
        k["rows"] += int(len(d))
        nw = max([x for x in (_ts(i.timestamp), d.max() if len(d) else None) if x is not None and x is not pd.NaT], default=None)
        if nw is not None and (k["newest"] is None or nw > k["newest"]):
            k["newest"] = nw
    return {"now": str(as_date(now)), "digest": input_digest(inputs), "kinds": {k: {**v, "newest": str(v["newest"].date()) if v["newest"] is not None else None}
                                                                                  for k, v in sorted(kinds.items())}}


# ---------------------------------------------------------------- can the firewall fail? (startup self-check)
def _selfcheck_bundle(now: pd.Timestamp, seed: int = 0) -> dict[str, Any]:
    """A small bundle that passes all eight checks: flat as-traded prices with exits, a vintage-stamped macro series, a mature
    label, declared features, one clean memory item, matching code, and empty IO logs."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(now - pd.Timedelta(days=365 * 7), now)
    px = pd.DataFrame(40.0 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(dates), 35)), axis=0)), index=dates, columns=[f"P{i:02d}" for i in range(35)])
    for i in range(6):
        px.iloc[int(len(px) * (0.25 + 0.1 * i)):, i] = np.nan
    X = pd.DataFrame({"mom": rng.normal(size=50)}, index=pd.MultiIndex.from_product([dates[-12:-2:2], list("ABCDEFGHIJ")], names=["date", "ticker"]))
    from .core import Provenance
    learned = str((now - pd.Timedelta(days=60)).date())
    prov = Provenance(created_real="2026-09-29T00:00:00", learned_at=learned, code_hash="c", data_hash="d", experiment_id="e",
                      outcomes_seen_through=learned)
    return dict(inputs=[LearningInput("px", InputKind.PRICE, timestamp=now, frame=px, universe=tuple(px.columns)),
                        LearningInput("un", InputKind.MACRO, timestamp=now - pd.Timedelta(days=60), available_at=now - pd.Timedelta(days=30),
                                      vintage=now - pd.Timedelta(days=30)),
                        LearningInput("y", InputKind.LABEL, timestamp=now - pd.Timedelta(days=10))],
                specs=[FeatureSpec("mom", "px", InputKind.PRICE, lookback=20)], items=[{"knowledge_id": "k", "version": 1, "provenance": prov,
                                                                                      "contexts": {"vol": "high"}}],
                code=CodeState(recorded={"code_hash": "c", "code_files": ["a.py"], "code_mixed": []}, current_hash="c"), events=[], cache=[],
                X=X, registered_sources=["px"])


def future_selfcheck(now="2019-12-31", seed: int = 0) -> dict:
    """Pass a clean bundle, then reject one planted defect per check. Returns {'clean_passed', 'missed': [...]}; anything in
    `missed` is a check that has stopped being able to fail."""
    n = pd.Timestamp(now)
    fw = FutureFirewall()
    base = _selfcheck_bundle(n, seed)
    plants: dict[str, Any] = {
        "timestamp": {"inputs": [LearningInput("x", InputKind.PRICE, timestamp=n + pd.Timedelta(days=5))]},
        "availability": {"inputs": [LearningInput("f", InputKind.FUNDAMENTAL, timestamp=n - pd.Timedelta(days=3))]},
        "revision": {"inputs": [LearningInput("m", InputKind.MACRO, timestamp=n - pd.Timedelta(days=60), available_at=n - pd.Timedelta(days=30))]},
        "survivorship": {"inputs": [LearningInput("px", InputKind.PRICE, timestamp=n, frame=base["inputs"][0].frame.ffill().bfill())]},
        "feature_provenance": {"specs": []},
        "memory_provenance": {"items": [dict(base["items"][0], provenance=dataclasses.replace(
            base["items"][0]["provenance"], learned_at=str((n + pd.Timedelta(days=9)).date()),
            outcomes_seen_through=str((n + pd.Timedelta(days=9)).date())))]},
        "code_version": {"code": CodeState(recorded={"code_hash": "c", "code_files": ["a.py"], "code_mixed": []}, current_hash="other")},
        "network_cache": {"events": [NetworkEvent("network", "example.com")]},
    }
    out: dict[str, Any] = {"clean_passed": fw.screen(n, **base).passed, "missed": []}
    for check, over in plants.items():
        v = fw.screen(n, **{**base, **over})
        if check not in v.failed_checks:
            out["missed"].append(check)
    return out


def lint_files(paths: Iterable, root=None) -> dict[str, list[LintHit]]:
    """Run the static look-ahead lint over source files (feature builders on disk). Files that cannot be parsed are reported
    with a single `syntax-error` hit rather than skipped, because an unreadable feature module is not a clean one."""
    from pathlib import Path
    out: dict[str, list[LintHit]] = {}
    for p in paths:
        pp = Path(root) / p if root is not None else Path(p)
        try:
            out[str(p)] = lint_feature_source(pp.read_text(encoding="utf-8"))
        except SyntaxError as e:
            out[str(p)] = [LintHit("syntax-error", int(e.lineno or 0), str(e.msg))]
        except OSError as e:
            out[str(p)] = [LintHit("source-unavailable", 0, str(e))]
    return out
