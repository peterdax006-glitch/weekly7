"""Bible Phase 3.2 - pattern identity (canons C35, C37, C43).

Every pattern the miner can ever produce gets one permanent identity:
  * a CANONICAL EXPRESSION, independent of the order the terms were written in ('A q1 & B q2' == 'B q2 & A q1');
  * a SHA-256 hash over (expression, transform, target) and the short id derived from it - statistics never enter the
    hash, so the same idea keeps the same id across windows, eras and re-tests;
  * the feature set, quantile transform, target, discovery and validation date windows, context scope, statistics
    and lifecycle state.
Records serialise to plain JSON, verify their own hash on load (a hand-edited expression is refused), and convert
from/to the tuple keys and 'key_named' strings engine.patterns.PatternMiner uses today, so the swap is mechanical.
Nothing here reads market data: an Expression can be evaluated on a quantile matrix the caller built point-in-time.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field, replace
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

SCHEMA = 1
ID_LEN = 16
N_LEVELS = 5
TRANSFORMS = ("xs_quintile", "ts_quintile")            # cross-sectional per date / expanding time-series (market context)
FAMILIES = ("single", "pair", "unless", "bank", "context")
# Same vocabulary as engine.pattern_lifecycle (kept local so this module imports without pulling the bank stack).
STATES = ("candidate", "rejected", "duplicate", "no_gain", "active", "watch", "failed", "cause_search", "rescoped",
          "discarded")
TRANSITIONS = {
    None: {"candidate"},
    "candidate": {"rejected", "duplicate", "no_gain", "active"},
    "rejected": {"candidate"}, "duplicate": {"candidate"}, "no_gain": {"candidate"},
    "active": {"failed", "watch"},
    "watch": {"active", "failed"},
    "failed": {"cause_search"},
    "cause_search": {"rescoped", "discarded"},
    "rescoped": {"active", "failed", "watch"},
    "discarded": {"candidate", "cause_search"},
}
# Miner statuses that are not lifecycle states map onto the nearest one (patterns.py writes 'failed' transiently).
STATUS_ALIASES = {"confirmed": "active", "fail": "failed"}
USABLE = ("active", "rescoped")
SCOPE_LABELS = ("low", "mid", "high")

_TERM_RE = re.compile(r"^\s*(?P<f>[A-Za-z_][\w.]*)\s+q(?P<q>\d+)\s*$")


class IdentityError(ValueError):
    """An expression, record or serialised payload that cannot be a valid pattern identity."""


# ---------------------------------------------------------------- terms and expressions
@dataclass(frozen=True, order=True)
class Term:
    """One condition: `feature` sits in quantile `level` (0 = lowest)."""
    feature: str
    level: int

    def __post_init__(self):
        if not isinstance(self.feature, str) or not self.feature:
            raise IdentityError(f"term feature must be a non-empty string, got {self.feature!r}")
        if isinstance(self.level, bool) or not isinstance(self.level, (int, np.integer)):
            raise IdentityError(f"term level must be an integer, got {self.level!r}")
        if not 0 <= int(self.level) < N_LEVELS:
            raise IdentityError(f"term level {self.level} outside 0..{N_LEVELS - 1} for {self.feature}")
        object.__setattr__(self, "level", int(self.level))

    def __str__(self):
        return f"{self.feature} q{self.level}"

    @staticmethod
    def parse(text: str) -> "Term":
        m = _TERM_RE.match(text)
        if not m:
            raise IdentityError(f"cannot parse term {text!r} (expected '<feature> q<level>')")
        return Term(m["f"], int(m["q"]))


@dataclass(frozen=True)
class Expression:
    """A conjunction of terms with optional exceptions: base terms must ALL hold, exception terms must NOT hold.

    Always built through `Expression.make` / `Expression.parse`, which sort and de-duplicate so equal ideas compare
    (and hash) equal regardless of how they were written."""
    base: tuple
    unless: tuple = ()

    # -- construction
    @staticmethod
    def make(base: Iterable[Term], unless: Iterable[Term] = ()) -> "Expression":
        base_t = sorted(set(base))
        exc_t = sorted(set(unless))
        if not base_t:
            raise IdentityError("an expression needs at least one base term")
        by_feat = {}
        for t in base_t:
            if t.feature in by_feat:
                raise IdentityError(f"contradiction: {by_feat[t.feature]} and {t} can never both hold (empty set)")
            by_feat[t.feature] = t
        keep = []
        for e in exc_t:
            b = by_feat.get(e.feature)
            if b is not None and b.level == e.level:
                raise IdentityError(f"exception {e} removes the whole base term {b} (empty set)")
            if b is not None:
                continue              # base already pins this feature to a different level: the exception is redundant
            keep.append(e)
        # two exceptions on one feature ("unless q0 unless q4") are legal and distinct, so they are kept
        return Expression(tuple(base_t), tuple(keep))

    @staticmethod
    def single(feature: str, level: int) -> "Expression":
        return Expression.make([Term(feature, level)])

    @staticmethod
    def parse(text: str) -> "Expression":
        """Parse 'A q1 & B q2 unless C q0 [unless D q4]' in any term order."""
        if not isinstance(text, str) or not text.strip():
            raise IdentityError("empty expression text")
        head, *exc = re.split(r"\s+unless\s+", text.strip())
        base = [Term.parse(p) for p in head.split("&")]
        unl = []
        for e in exc:
            unl.extend(Term.parse(p) for p in e.split("&"))
        return Expression.make(base, unl)

    @staticmethod
    def from_key(key: Sequence, feats: Sequence[str]) -> "Expression":
        """Convert a PatternMiner tuple key ('s', j, q) / ('p', j1, q1, j2, q2) / ('u', j1, q1, j2, q2, j3, q3)."""
        try:
            kind = key[0]
            if kind == "s":
                return Expression.make([Term(feats[key[1]], key[2])])
            if kind == "p":
                return Expression.make([Term(feats[key[1]], key[2]), Term(feats[key[3]], key[4])])
            if kind == "u":
                return Expression.make([Term(feats[key[1]], key[2]), Term(feats[key[3]], key[4])],
                                       [Term(feats[key[5]], key[6])])
        except (IndexError, TypeError) as ex:
            raise IdentityError(f"key {key!r} does not fit a feature list of {len(feats)}: {ex}") from ex
        raise IdentityError(f"unknown key kind {key[0]!r}")

    # -- views
    @property
    def text(self) -> str:
        out = " & ".join(str(t) for t in self.base)
        for e in self.unless:
            out += f" unless {e}"
        return out

    def __str__(self):
        return self.text

    @property
    def features(self) -> tuple:
        return tuple(sorted({t.feature for t in self.base} | {t.feature for t in self.unless}))

    @property
    def family(self) -> str:
        if self.unless:
            return "unless"
        return "single" if len(self.base) == 1 else "pair" if len(self.base) == 2 else "multi"

    @property
    def order(self) -> int:
        """Complexity: number of terms of any kind. The miner prefers the simpler of two near-duplicates."""
        return len(self.base) + len(self.unless)

    def to_key(self, feats: Sequence[str]) -> tuple:
        """Back to the PatternMiner tuple key. The miner's keys hold at most two base terms and one exception, in the
        order the miner generated them; the canonical order here is alphabetical, which is a different but equivalent
        key (the mask is symmetric in the base terms)."""
        ix = {c: i for i, c in enumerate(feats)}
        try:
            b = [(ix[t.feature], t.level) for t in self.base]
            u = [(ix[t.feature], t.level) for t in self.unless]
        except KeyError as ex:
            raise IdentityError(f"feature {ex.args[0]!r} is not in the feature list") from ex
        if len(b) == 1 and not u:
            return ("s", b[0][0], b[0][1])
        if len(b) == 2 and not u:
            return ("p", b[0][0], b[0][1], b[1][0], b[1][1])
        if len(b) == 2 and len(u) == 1:
            return ("u", b[0][0], b[0][1], b[1][0], b[1][1], u[0][0], u[0][1])
        raise IdentityError(f"{self.text!r} has no PatternMiner tuple form (needs 1-2 base terms, at most 1 exception)")

    def mask(self, Q: np.ndarray, feats: Sequence[str]) -> np.ndarray:
        """Boolean row mask on a quantile matrix Q (rows x features, int levels), columns named by `feats`."""
        ix = {c: i for i, c in enumerate(feats)}
        for f in self.features:
            if f not in ix:
                raise IdentityError(f"feature {f!r} is not in the quantile matrix")
        m = np.ones(Q.shape[0], bool)
        for t in self.base:
            m &= Q[:, ix[t.feature]] == t.level
        for e in self.unless:
            col = Q[:, ix[e.feature]]
            m &= (col != e.level) & (col >= 0)     # -1 = level unknown (time-series warm-up): an unknown is no exception
        return m

    def parents(self) -> list:
        """Every simpler expression obtained by dropping one term (a child's effect is judged against these)."""
        out = []
        for i in range(len(self.base)):
            rest = self.base[:i] + self.base[i + 1:]
            if rest:
                out.append(Expression.make(rest, self.unless))
        for i in range(len(self.unless)):
            out.append(Expression.make(self.base, self.unless[:i] + self.unless[i + 1:]))
        return out


def canonical_expression(text_or_expr) -> str:
    """Canonical text of an expression given as text, an Expression or a miner 'key_named' string."""
    e = text_or_expr if isinstance(text_or_expr, Expression) else Expression.parse(str(text_or_expr))
    return e.text


def pattern_hash(expr: Expression, transform: str = "xs_quintile", target: str = "excess_5d") -> str:
    """Full SHA-256 hex over the identity fields only. Statistics, dates and state never enter it."""
    if transform not in TRANSFORMS:
        raise IdentityError(f"unknown transform {transform!r}; expected one of {TRANSFORMS}")
    if not isinstance(target, str) or not target:
        raise IdentityError("target must be a non-empty string")
    payload = json.dumps({"expr": expr.text, "transform": transform, "target": target, "levels": N_LEVELS},
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pattern_id(expr: Expression, transform: str = "xs_quintile", target: str = "excess_5d") -> str:
    return "P" + pattern_hash(expr, transform, target)[:ID_LEN]


# ---------------------------------------------------------------- scope, windows, stats
@dataclass(frozen=True)
class Scope:
    """Context scope of a rescoped pattern: it counts only while `context` sits inside a tercile of history.
    Mirrors PatternMiner's (ci, label, lo, hi) tuple with the column NAME instead of an index."""
    context: str
    label: str
    lo: float
    hi: float

    def __post_init__(self):
        if self.label not in SCOPE_LABELS:
            raise IdentityError(f"scope label must be one of {SCOPE_LABELS}, got {self.label!r}")
        if not (math.isfinite(self.lo) and math.isfinite(self.hi)) or self.lo > self.hi:
            raise IdentityError(f"scope bounds must be finite with lo <= hi, got ({self.lo}, {self.hi})")

    def contains(self, value: float) -> bool:
        """Same rule as PatternMiner._in_scope; an unknown context value never satisfies a scope."""
        if value is None or not np.isfinite(value):
            return False
        return value <= self.lo if self.label == "low" else value > self.hi if self.label == "high" \
            else self.lo < value <= self.hi

    @staticmethod
    def from_miner(scope, ctx_cols: Sequence[str]) -> Optional["Scope"]:
        if scope is None or (isinstance(scope, float) and np.isnan(scope)):
            return None
        ci, lab, lo, hi = scope
        if not 0 <= int(ci) < len(ctx_cols):
            raise IdentityError(f"scope context index {ci} outside {len(ctx_cols)} context columns")
        return Scope(ctx_cols[int(ci)], str(lab), float(lo), float(hi))

    def to_miner(self, ctx_cols: Sequence[str]) -> tuple:
        return (list(ctx_cols).index(self.context), self.label, self.lo, self.hi)

    def to_dict(self) -> dict:
        return {"context": self.context, "label": self.label, "lo": self.lo, "hi": self.hi}

    @staticmethod
    def from_dict(d) -> Optional["Scope"]:
        return None if d is None else Scope(str(d["context"]), str(d["label"]), float(d["lo"]), float(d["hi"]))


@dataclass(frozen=True)
class Window:
    """An inclusive date window, ISO strings on disk. Open ends are not allowed: a window that could grow into the
    future would be a look-ahead waiting to happen."""
    start: str
    end: str

    def __post_init__(self):
        try:
            s, e = pd.Timestamp(self.start), pd.Timestamp(self.end)
        except (ValueError, TypeError) as ex:
            raise IdentityError(f"window dates must be valid dates: {ex}") from ex
        if pd.isna(s) or pd.isna(e):
            raise IdentityError("window dates must be valid")
        if s > e:
            raise IdentityError(f"window starts after it ends: {self.start} > {self.end}")
        object.__setattr__(self, "start", s.strftime("%Y-%m-%d"))
        object.__setattr__(self, "end", e.strftime("%Y-%m-%d"))

    def __contains__(self, when) -> bool:
        return pd.Timestamp(self.start) <= pd.Timestamp(when) <= pd.Timestamp(self.end)

    def overlaps(self, other: "Window") -> bool:
        return not (pd.Timestamp(self.end) < pd.Timestamp(other.start) or pd.Timestamp(other.end) < pd.Timestamp(self.start))

    def to_list(self) -> list:
        return [self.start, self.end]

    @staticmethod
    def from_any(v) -> Optional["Window"]:
        if v is None:
            return None
        if isinstance(v, Window):
            return v
        a, b = v
        return Window(a, b)


STAT_FIELDS = ("m_disc", "t_disc", "m_conf", "t_conf", "m_all", "t_all", "n_eff", "m_recent", "t_recent", "n_recent",
               "p_coincidence", "p_hallucinated", "p_real", "effect", "fdr_pass", "q_value", "local_fdr", "n_rows")


def _clean_stat(k: str, v):
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return f if math.isfinite(f) else None       # JSON has no NaN; None means "not measured"
    raise IdentityError(f"statistic {k!r} must be numeric or bool, got {type(v).__name__}")


# ---------------------------------------------------------------- the record
@dataclass(frozen=True)
class PatternRecord:
    """Permanent identity + latest evidence for one pattern. Immutable: updates return a new record."""
    expression: Expression
    transform: str = "xs_quintile"
    target: str = "excess_5d"
    family: str = ""
    discovery: Optional[Window] = None
    validation: Optional[Window] = None
    scope: Optional[Scope] = None
    stats: dict = field(default_factory=dict)
    state: str = "candidate"
    duplicate_of: Optional[str] = None            # id of the stronger pattern this one overlaps
    history: tuple = ()                           # ((as_of, from_state, to_state, reason), ...) - the audit trail

    def __post_init__(self):
        if self.transform not in TRANSFORMS:
            raise IdentityError(f"unknown transform {self.transform!r}")
        if not self.target:
            raise IdentityError("target must be non-empty")
        if self.state not in STATES:
            raise IdentityError(f"unknown lifecycle state {self.state!r}; expected one of {STATES}")
        fam = self.family or self.expression.family
        if fam not in FAMILIES + ("multi",):
            raise IdentityError(f"unknown family {fam!r}")
        object.__setattr__(self, "family", fam)
        if self.discovery and self.validation and self.discovery.end >= self.validation.start:
            raise IdentityError(f"discovery window {self.discovery.to_list()} must end strictly before validation "
                                f"{self.validation.to_list()} (a validation that overlaps discovery validates nothing)")
        if self.state == "rescoped" and self.scope is None:
            raise IdentityError("a rescoped pattern must carry the scope it was rescoped to")
        object.__setattr__(self, "stats", {k: _clean_stat(k, v) for k, v in dict(self.stats).items()})

    # -- identity
    @property
    def hash(self) -> str:
        return pattern_hash(self.expression, self.transform, self.target)

    @property
    def id(self) -> str:
        return "P" + self.hash[:ID_LEN]

    @property
    def features(self) -> tuple:
        return self.expression.features

    @property
    def text(self) -> str:
        return self.expression.text

    @property
    def usable(self) -> bool:
        return self.state in USABLE

    # -- evolution
    def with_stats(self, **stats) -> "PatternRecord":
        return replace(self, stats={**self.stats, **stats})

    def transition(self, new_state: str, as_of, reason: str = "", scope: Optional[Scope] = None) -> "PatternRecord":
        """Move through the lifecycle. Illegal moves (e.g. candidate -> rescoped) raise: state history is evidence."""
        new_state = STATUS_ALIASES.get(new_state, new_state)
        if new_state not in STATES:
            raise IdentityError(f"unknown lifecycle state {new_state!r}")
        allowed = TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            raise IdentityError(f"{self.id}: {self.state} -> {new_state} is not allowed (allowed: {sorted(allowed)})")
        last = self.history[-1][0] if self.history else None
        when = pd.Timestamp(as_of).strftime("%Y-%m-%d")
        if last is not None and when < last:
            raise IdentityError(f"{self.id}: transition dated {when} is earlier than the previous one ({last})")
        sc = scope if new_state == "rescoped" else (None if new_state in ("discarded", "candidate") else self.scope)
        return replace(self, state=new_state, scope=sc,
                       history=self.history + ((when, self.state, new_state, reason),))

    # -- (de)serialisation
    def to_dict(self) -> dict:
        return {"schema": SCHEMA, "id": self.id, "hash": self.hash, "expression": self.expression.text,
                "features": list(self.features), "family": self.family, "transform": self.transform,
                "target": self.target, "discovery": self.discovery.to_list() if self.discovery else None,
                "validation": self.validation.to_list() if self.validation else None,
                "scope": self.scope.to_dict() if self.scope else None, "stats": dict(self.stats),
                "state": self.state, "duplicate_of": self.duplicate_of,
                "history": [list(h) for h in self.history]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @staticmethod
    def from_dict(d: dict) -> "PatternRecord":
        for k in ("expression", "transform", "target", "state", "id"):
            if k not in d:
                raise IdentityError(f"payload is missing {k!r}")
        if d.get("schema", SCHEMA) != SCHEMA:
            raise IdentityError(f"unsupported pattern schema {d.get('schema')!r} (this build reads {SCHEMA})")
        rec = PatternRecord(expression=Expression.parse(d["expression"]), transform=d["transform"], target=d["target"],
                            family=d.get("family", ""), discovery=Window.from_any(d.get("discovery")),
                            validation=Window.from_any(d.get("validation")), scope=Scope.from_dict(d.get("scope")),
                            stats=d.get("stats") or {}, state=d["state"], duplicate_of=d.get("duplicate_of"),
                            history=tuple(tuple(h) for h in d.get("history", ())))
        if rec.id != d["id"] or ("hash" in d and rec.hash != d["hash"]):
            raise IdentityError(f"identity check failed: payload says {d['id']} but its own fields hash to {rec.id} "
                                f"(the expression, transform or target was edited after the id was issued)")
        listed = d.get("features")
        if listed is not None and tuple(sorted(listed)) != rec.features:
            raise IdentityError(f"feature list {listed} disagrees with the expression's {list(rec.features)}")
        return rec

    @staticmethod
    def from_json(s: str) -> "PatternRecord":
        try:
            return PatternRecord.from_dict(json.loads(s))
        except json.JSONDecodeError as ex:
            raise IdentityError(f"not valid JSON: {ex}") from ex


def new_record(expression, transform="xs_quintile", target="excess_5d", discovery=None, validation=None,
               scope=None, stats=None, state="candidate", family="") -> PatternRecord:
    """Convenience constructor accepting an Expression or its text, and (start, end) pairs for the windows."""
    expr = expression if isinstance(expression, Expression) else Expression.parse(expression)
    return PatternRecord(expr, transform, target, family, Window.from_any(discovery), Window.from_any(validation),
                         scope, dict(stats or {}), state)


# ---------------------------------------------------------------- the miner bridge
def records_from_miner(patterns: pd.DataFrame, feats: Sequence[str], ctx_cols: Sequence[str], dates: pd.DatetimeIndex,
                       transform: str = "xs_quintile", target: str = "excess_5d", disc_frac: float = 0.7,
                       transform_of=None) -> list:
    """Turn a PatternMiner.patterns table into records, deriving the discovery/validation windows the same way fit()
    splits them (earlier 70% of unique dates vs the rest). Rows whose key cannot be expressed are skipped and counted
    by the caller through len(); a duplicate identity keeps its strongest row (highest |t_disc| + |t_conf|)."""
    if patterns is None or len(patterns) == 0:
        return []
    ud = np.sort(pd.DatetimeIndex(dates).unique())
    if len(ud) < 2:
        raise IdentityError("need at least two distinct dates to split discovery from validation")
    cut = int(len(ud) * disc_frac)
    cut = min(max(cut, 1), len(ud) - 1)
    disc = Window(ud[0], ud[cut - 1])
    val = Window(ud[cut], ud[-1])
    best = {}
    for row in patterns.to_dict("records"):
        expr = Expression.from_key(row["key"], feats)
        stats = {k: row[k] for k in STAT_FIELDS if k in row}
        state = STATUS_ALIASES.get(str(row.get("status", "candidate")), str(row.get("status", "candidate")))
        scope = Scope.from_miner(row.get("scope"), ctx_cols) if state == "rescoped" else None
        state = state if state in STATES else "candidate"
        rec = PatternRecord(expr, transform_of(expr) if transform_of else transform, target, "", disc, val, scope, stats, state,
                            duplicate_of=(row.get("duplicate_of") if isinstance(row.get("duplicate_of"), str) else None))
        strength = abs(stats.get("t_disc") or 0.0) + abs(stats.get("t_conf") or 0.0)
        if rec.id not in best or strength > best[rec.id][0]:
            best[rec.id] = (strength, rec)
    return [r for _, r in best.values()]


def records_to_prior(records: Iterable[PatternRecord]) -> pd.DataFrame:
    """The `prior` table PatternMiner.fit(prior=...) re-tests: only usable patterns with a miner-shaped key."""
    rows = []
    for r in records:
        if not r.usable:
            continue
        e = r.expression
        if len(e.base) > 2 or len(e.unless) > 1:
            continue
        kind = "s" if len(e.base) == 1 and not e.unless else "p" if not e.unless else "u"
        names = [kind]
        for t in e.base:
            names += [t.feature, t.level]
        for t in e.unless:
            names += [t.feature, t.level]
        rows.append({"names": names, "effect": r.stats.get("effect") or 0.0, "p_real": r.stats.get("p_real") or 0.0})
    return pd.DataFrame(rows, columns=["names", "effect", "p_real"])


# ---------------------------------------------------------------- a collection
class PatternBook:
    """Records keyed by id. Adding an identity that already exists merges (newer record wins, history is preserved
    and the two audit trails are unioned in date order) rather than silently duplicating it."""

    def __init__(self, records: Iterable[PatternRecord] = ()):
        self._r = {}
        for r in records:
            self.add(r)

    def __len__(self):
        return len(self._r)

    def __contains__(self, key):
        return (key.id if isinstance(key, PatternRecord) else key) in self._r

    def __iter__(self):
        return iter(self._r.values())

    def get(self, pid: str) -> Optional[PatternRecord]:
        return self._r.get(pid)

    @staticmethod
    def _recency(rec: PatternRecord) -> tuple:
        """Which of two records of one identity is the later view: last audited date, then trail length, then a
        canonical serialisation as a final tiebreak so that merging is independent of the order records arrive in."""
        return (rec.history[-1][0] if rec.history else "", len(rec.history), rec.to_json())

    def add(self, rec: PatternRecord) -> PatternRecord:
        old = self._r.get(rec.id)
        if old is not None:
            hist = tuple(sorted(set(old.history) | set(rec.history)))
            winner = old if self._recency(old) >= self._recency(rec) else rec
            rec = replace(winner, history=hist)
        self._r[rec.id] = rec
        return rec

    def by_state(self, *states) -> list:
        return [r for r in self._r.values() if r.state in states]

    def find_expression(self, text: str, transform="xs_quintile", target="excess_5d") -> Optional[PatternRecord]:
        return self._r.get(pattern_id(Expression.parse(text), transform, target))

    def counts(self) -> dict:
        out = {}
        for r in self._r.values():
            out[r.state] = out.get(r.state, 0) + 1
        return dict(sorted(out.items()))

    def to_json(self) -> str:
        return json.dumps({"schema": SCHEMA, "patterns": [self._r[k].to_dict() for k in sorted(self._r)]},
                          sort_keys=True, separators=(",", ":"))

    @staticmethod
    def from_json(s: str) -> "PatternBook":
        try:
            d = json.loads(s)
        except json.JSONDecodeError as ex:
            raise IdentityError(f"book is not valid JSON: {ex}") from ex
        if not isinstance(d, dict) or d.get("schema") != SCHEMA or "patterns" not in d:
            raise IdentityError("not a pattern book of this schema")
        book = PatternBook()
        seen = set()
        for p in d["patterns"]:
            rec = PatternRecord.from_dict(p)
            if rec.id in seen:
                raise IdentityError(f"duplicate id {rec.id} inside one book file")
            seen.add(rec.id)
            book.add(rec)
        return book

    def digest(self) -> str:
        """Order-independent fingerprint of the whole book (ids + states + stats)."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def diff(self, other: "PatternBook") -> dict:
        a, b = set(self._r), set(other._r)
        changed = sorted(k for k in a & b if self._r[k].to_dict() != other._r[k].to_dict())
        return {"only_here": sorted(a - b), "only_there": sorted(b - a), "changed": changed}
