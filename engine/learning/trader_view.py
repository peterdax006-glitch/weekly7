"""Trader-side records: what the BLIND side of the system is allowed to hold (canon C64, C56; contract sections 28-30, 55).
IMPLEMENTED - NOT VALIDATED.

C64 splits the system in two. The TRADER never knows the year: it gets information day by day, only as it became available,
and it is never handed a real date in any object, field, key, name, file path or string. The CURATOR (curator.py) is the
trusted, invisible side that files memory under the real year and decides which memories matter now. Everything the curator
hands across the wall is one of the frozen records below, and their constructors REFUSE anything that looks like a real date
or year, so a leak cannot be built by accident and an adversarial payload cannot be carried across.

What counts as date-like (each is a counted category, see `find_violations`): ISO / compact / separated dates, any 4-digit
year 1900-2100 not glued to other digits (lookarounds, not \\b: \\b treats '_' as a word character and misses AAPL_2008),
quarter labels, era words (covid, lehman, ...), datetime / date / Timestamp / datetime64 / Period objects, numbers in the
year range or shaped like compact dates, epoch seconds / ms / ns and date ordinals, and field NAMES that describe time or
ranking reasons (year, date, era, priority, why ...). `scrub` strips what it can and counts; `refuse=True` raises instead.

Static guard: `trader_path_violations` walks the import closure of the modules on the trader path (default livesim, the
learner's decide path, adaptive) and reports any edge that reaches curator internals or reads the curator's store."""
from __future__ import annotations

import ast
import dataclasses
import datetime as dt
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .core import FirewallBreach, canonical_json, stable_hash

# ------------------------------------------------------------------------------------------------ what looks like a date
_SEP = r"[-/._ ]?"
_ISO = re.compile(rf"(?<!\d)(?:19|20)\d{{2}}{_SEP}(?:0[1-9]|1[0-2]){_SEP}(?:0[1-9]|[12]\d|3[01])(?!\d)")
_DMY = re.compile(r"(?<!\d)(?:0?[1-9]|[12]\d|3[01])[-/.](?:0?[1-9]|1[0-2])[-/.](?:19|20)?\d{2}(?!\d)")
_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_MONTH_NAME = re.compile(r"(?i)(?<![a-z])(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?[ _-]*\d{1,2}(?:st|nd|rd|th)?"
                         r"(?:[ ,_-]+\d{2,4})?(?!\d)")
_QUARTER = re.compile(r"(?i)(?<![a-z0-9])(?:q[1-4]|[1-4]q)[ _'-]?(?:19|20)?\d{2}(?!\d)")
_FISCAL = re.compile(r"(?i)(?<![a-z0-9])fy[ _-]?\d{2,4}(?!\d)")
_ERA = re.compile(r"(?i)(?<![a-z])(?:dot-?com|covid(?:-?19)?|corona-?virus|pandemic|gfc|lehman|subprime|great[ _-]?recession|"
                  r"black[ _-]?monday|financial[ _-]?crisis|taper[ _-]?tantrum|flash[ _-]?crash|brexit|y2k|volmageddon|"
                  r"meme[ _-]?stock|september[ _-]?11|nine[ _-]?eleven|oil[ _-]?crash|euro[ _-]?crisis)(?![a-z])")
STRING_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("iso_date", _ISO), ("dmy_date", _DMY), ("month_name_date", _MONTH_NAME), ("year", _YEAR), ("quarter_label", _QUARTER),
    ("fiscal_label", _FISCAL), ("era_word", _ERA))

# time-describing or ranking-explaining FIELD NAMES the trader must never see (checked per underscore/camel token)
FORBIDDEN_TOKENS = frozenset({
    "year", "years", "yr", "date", "dates", "datetime", "day_of_year", "era", "eras", "epoch", "calendar", "month", "quarter",
    "timestamp", "ts", "asof", "filed", "matured", "occurred", "vintage", "regime_year",
    "priority", "priorities", "rank", "ranking", "why", "reason", "reasons", "explanation", "because", "rationale",
    "curator", "relevance", "consistency", "similarity", "components", "audit"})
FORBIDDEN_PHRASES = ("as_of", "real_date", "real_year", "day_of_year", "filed_year", "matured_at", "occurred_at", "learned_at")

EPOCH_RANGES = (("epoch_seconds", 6.0e8, 4.2e9), ("epoch_millis", 6.0e11, 4.2e12), ("epoch_micros", 6.0e14, 4.2e15),
                ("epoch_nanos", 6.0e17, 4.2e18), ("date_ordinal", 693_596.0, 767_000.0))
MAX_DEPTH = 8
MAX_NODES = 20_000
MAX_ABS_VALUE = 1.0e6                       # trader features are relative (z / percentile / return); huge numbers are ids in disguise


@dataclasses.dataclass(frozen=True)
class Violation:
    path: str
    category: str
    detail: str

    def __str__(self):
        return f"{self.path or '<root>'}: {self.category} ({self.detail})"


def _scalar(x):
    """numpy scalar -> python scalar; anything else unchanged."""
    if isinstance(x, np.generic):
        return x.item()
    return x


def number_reason(x) -> str | None:
    """Why a NUMBER looks like a date, or None. Year-range values are refused whether int or float (2008 and 2008.0 and
    2008.37 are all a year); compact yyyymmdd; epoch seconds / ms / us / ns; date ordinals. The price is that a feature which
    happens to equal 1950.3 is refused too - trader features are relative, never raw levels, so this is the intended trade."""
    x = _scalar(x)
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    if isinstance(x, float) and not math.isfinite(x):
        return "non_finite"
    a = abs(float(x))
    if 1900.0 <= a < 2101.0:
        return "year_number"
    if float(x).is_integer() and 19000101 <= a <= 21001231:
        s = str(int(a))
        if 1 <= int(s[4:6]) <= 12 and 1 <= int(s[6:8]) <= 31:
            return "date_number"
    if float(x).is_integer():
        for name, lo, hi in EPOCH_RANGES:
            if lo <= a <= hi:
                return name
    if a > MAX_ABS_VALUE:
        return "oversized_number"
    return None


def string_reasons(s: str) -> list[str]:
    """Every date-like category present in a string (an id, a key, a value, a path)."""
    return [name for name, pat in STRING_PATTERNS if pat.search(s)]


def _key_tokens(k: str) -> list[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", k)
    return [t for t in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if t]


def key_reason(k: str) -> str | None:
    low = k.lower()
    if any(p in low for p in FORBIDDEN_PHRASES):
        return "forbidden_key"
    toks = _key_tokens(k)
    if any(t in FORBIDDEN_TOKENS for t in toks):
        return "forbidden_key"
    return None


def _datetime_kind(x) -> str | None:
    if isinstance(x, (dt.datetime, dt.date, np.datetime64)):
        return type(x).__name__
    tname = type(x).__name__
    if tname in ("Timestamp", "Period", "DatetimeIndex", "PeriodIndex", "DatetimeArray", "NaTType"):
        return tname
    return None


def find_violations(obj: Any, path: str = "", _depth: int = 0, _budget: list | None = None) -> list[Violation]:
    """Everything in `obj` (recursively) that would tell the trader the year. Pure; never mutates. Unsupported types are a
    violation too (an arbitrary object could carry a date in an attribute nobody scans)."""
    budget = _budget if _budget is not None else [MAX_NODES]
    budget[0] -= 1
    out: list[Violation] = []
    if budget[0] < 0:
        return [Violation(path, "too_large", f"more than {MAX_NODES} nodes")]
    if _depth > MAX_DEPTH:
        return [Violation(path, "too_deep", f"nesting deeper than {MAX_DEPTH}")]
    obj = _scalar(obj)
    if obj is None or isinstance(obj, bool):
        return out
    kind = _datetime_kind(obj)
    if kind:
        return [Violation(path, "datetime_object", kind)]
    if isinstance(obj, (int, float)):
        r = number_reason(obj)
        return [Violation(path, r, repr(obj))] if r else out
    if isinstance(obj, bytes):
        try:
            obj = obj.decode("utf-8")
        except UnicodeDecodeError:
            return [Violation(path, "opaque_bytes", "undecodable bytes")]
    if isinstance(obj, str):
        return [Violation(path, r, obj[:40]) for r in string_reasons(obj)]
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            kp = f"{path}.{k}" if path else str(k)
            if not isinstance(k, str):
                out.append(Violation(kp, "non_string_key", type(k).__name__))
                continue
            out += [Violation(kp, "key_" + r, k[:40]) for r in string_reasons(k)]
            kr = key_reason(k)
            if kr:
                out.append(Violation(kp, kr, k))
            out += find_violations(v, kp, _depth + 1, budget)
        return out
    if isinstance(obj, (list, tuple, set, frozenset)):
        for i, v in enumerate(obj):
            out += find_violations(v, f"{path}[{i}]", _depth + 1, budget)
        return out
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            out += find_violations(getattr(obj, f.name), f"{path}.{f.name}" if path else f.name, _depth + 1, budget)
        return out
    return [Violation(path, "unsupported_type", type(obj).__name__)]


def assert_trader_safe(obj: Any, what: str = "object") -> None:
    """Fail closed: raise FirewallBreach naming the first few violations."""
    v = find_violations(obj)
    if v:
        raise FirewallBreach(f"{what} is not blind-safe: " + "; ".join(str(x) for x in v[:4]) + (f" (+{len(v) - 4} more)" if len(v) > 4 else ""))


# ------------------------------------------------------------------------------------------------ scrubbing
@dataclasses.dataclass(frozen=True)
class ScrubReport:
    """What `scrub` removed, by category. `nodes` is how many values were inspected; `removed` how many were dropped or edited."""
    counts: Mapping[str, int]
    nodes: int
    removed: int

    @property
    def clean(self) -> bool:
        return self.removed == 0

    def total(self, category: str | None = None) -> int:
        return int(sum(self.counts.values())) if category is None else int(self.counts.get(category, 0))


def _strip_string(s: str, counts: Counter) -> str | None:
    """Remove date-like substrings; None if nothing meaningful is left or a date-like residue survives."""
    cur = s
    for name, pat in STRING_PATTERNS:
        cur, n = pat.subn("", cur)
        if n:
            counts[name] += n
    cur = re.sub(r"[_\-\s.:/]{2,}", "_", cur).strip("_-. /:")
    if not cur or string_reasons(cur):
        return None
    return cur


def scrub(obj: Any, refuse: bool = False) -> tuple[Any, ScrubReport]:
    """A copy of `obj` with every date-like thing removed, plus a count by category. Dict entries with a forbidden or date-like
    key, datetime objects and year-like numbers are dropped; strings have the date part cut out (dropped if nothing is
    left). `refuse=True` raises FirewallBreach on the first offence instead. A scrubbed result always re-checks clean."""
    if refuse:
        assert_trader_safe(obj, "payload")
    counts: Counter = Counter()
    nodes = [0]

    def walk(x, depth):
        nodes[0] += 1
        x = _scalar(x)
        if depth > MAX_DEPTH:
            counts["too_deep"] += 1
            return _DROP
        if x is None or isinstance(x, bool):
            return x
        if _datetime_kind(x):
            counts["datetime_object"] += 1
            return _DROP
        if isinstance(x, (int, float)):
            r = number_reason(x)
            if r:
                counts[r] += 1
                return _DROP
            return x
        if isinstance(x, bytes):
            try:
                x = x.decode("utf-8")
            except UnicodeDecodeError:
                counts["opaque_bytes"] += 1
                return _DROP
        if isinstance(x, str):
            if not string_reasons(x):
                return x
            fixed = _strip_string(x, counts)
            return _DROP if fixed is None else fixed
        if isinstance(x, Mapping):
            out = {}
            for k, v in x.items():
                if not isinstance(k, str):
                    counts["non_string_key"] += 1
                    continue
                if key_reason(k):
                    counts["forbidden_key"] += 1
                    continue
                if string_reasons(k):
                    counts["key_with_date"] += 1
                    continue
                nv = walk(v, depth + 1)
                if nv is not _DROP:
                    out[k] = nv
            return out
        if isinstance(x, (list, tuple, set, frozenset)):
            items = sorted(x, key=repr) if isinstance(x, (set, frozenset)) else x
            kept = [nv for nv in (walk(v, depth + 1) for v in items) if nv is not _DROP]
            return kept
        counts["unsupported_type"] += 1
        return _DROP

    clean = walk(obj, 0)
    if clean is _DROP:
        clean = None
    rep = ScrubReport(dict(counts), nodes[0], int(sum(counts.values())))
    return clean, rep


_DROP = object()


class ScrubLedger:
    """Running totals of what scrubbing removed across many payloads - the trusted side reports this; the trader never sees it."""

    def __init__(self):
        self.counts: Counter = Counter()
        self.calls = 0
        self.dirty_calls = 0

    def scrub(self, obj: Any, refuse: bool = False) -> Any:
        clean, rep = scrub(obj, refuse=refuse)
        self.calls += 1
        self.dirty_calls += 0 if rep.clean else 1
        self.counts.update(rep.counts)
        return clean

    def summary(self) -> dict:
        return {"calls": self.calls, "dirty_calls": self.dirty_calls, "by_category": dict(sorted(self.counts.items()))}


# ------------------------------------------------------------------------------------------------ opaque ids
def opaque_token(obj: Any, n: int = 12, salt: str = "") -> str:
    """A letters-only token derived from content. No digits means it can never contain a year, compact date or epoch, whatever
    the hash. Same content -> same token everywhere, so identical items look identical to the trader."""
    h = stable_hash({"salt": salt, "obj": obj}, 64)
    letters = "".join(chr(97 + int(h[i:i + 2], 16) % 26) for i in range(0, 2 * n, 2))
    return letters


_ID = re.compile(r"[a-z]{6,40}")
KINDS = frozenset({"pattern", "lesson", "context"})
MAX_HORIZON = 756


def _finite_float(x, name: str, lo: float, hi: float) -> float:
    x = _scalar(x)
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        raise FirewallBreach(f"{name}={x!r} is not a finite number")
    if not lo <= float(x) <= hi:
        raise FirewallBreach(f"{name}={x!r} outside [{lo}, {hi}]")
    return float(x)


# ------------------------------------------------------------------------------------------------ frozen trader records
@dataclasses.dataclass(frozen=True)
class TraderMemoryItem:
    """One anonymised, weighted memory as the trader sees it. Nothing here says when it was learned, which year it came from,
    or why it was chosen: only what it says (`features`, `lean`), how far ahead it speaks (`horizon`, trading days) and how
    much to lean on it now (`weight`). The id is a content token (letters only)."""
    item_id: str
    kind: str
    weight: float
    features_json: str
    lean: float
    horizon: int

    def __post_init__(self):
        v = find_violations({"item_id": self.item_id, "kind": self.kind, "features": self._parse(self.features_json)})
        if v:
            raise FirewallBreach("TraderMemoryItem refused: " + "; ".join(str(x) for x in v[:3]))
        if not isinstance(self.item_id, str) or not _ID.fullmatch(self.item_id):
            raise FirewallBreach(f"TraderMemoryItem.item_id {self.item_id!r} is not an opaque letters-only token")
        if self.kind not in KINDS:
            raise FirewallBreach(f"TraderMemoryItem.kind {self.kind!r} not in {sorted(KINDS)}")
        object.__setattr__(self, "weight", _finite_float(self.weight, "weight", 0.0, 1.0))
        object.__setattr__(self, "lean", _finite_float(self.lean, "lean", -1.0, 1.0))
        h = _scalar(self.horizon)
        if isinstance(h, bool) or not isinstance(h, int) or not 1 <= h <= MAX_HORIZON:
            raise FirewallBreach(f"TraderMemoryItem.horizon {self.horizon!r} must be an int in 1..{MAX_HORIZON}")
        object.__setattr__(self, "horizon", int(h))
        feats = self._parse(self.features_json)
        for k, x in feats.items():
            _finite_float(x, f"features.{k}", -MAX_ABS_VALUE, MAX_ABS_VALUE)

    @staticmethod
    def _parse(text: str) -> dict:
        try:
            d = json.loads(text)
        except (TypeError, ValueError) as e:
            raise FirewallBreach(f"features_json is not JSON: {e}") from e
        if not isinstance(d, dict):
            raise FirewallBreach("features_json must encode an object")
        return d

    @classmethod
    def make(cls, item_id: str, kind: str, weight: float, features: Mapping[str, float], lean: float, horizon: int) -> "TraderMemoryItem":
        """Convenience constructor from a mapping; validates through __post_init__ like every other path."""
        if not isinstance(features, Mapping):
            raise FirewallBreach("features must be a mapping of name -> number")
        return cls(item_id, kind, weight, canonical_json(dict(features)), lean, horizon)

    @property
    def features(self) -> dict:
        return self._parse(self.features_json)

    def to_dict(self) -> dict:
        return {"item_id": self.item_id, "kind": self.kind, "weight": self.weight, "features": self.features,
                "lean": self.lean, "horizon": self.horizon}

    def with_weight(self, weight: float) -> "TraderMemoryItem":
        return dataclasses.replace(self, weight=weight)


@dataclasses.dataclass(frozen=True)
class TraderSituation:
    """Today as the trader is allowed to know it: how many sessions into the run it is (a run-relative counter, not a date) and
    the market state as positions relative to its own trailing history. `warm` is False until enough history exists."""
    step: int
    features_json: str
    warm: bool = True

    def __post_init__(self):
        s = _scalar(self.step)
        if isinstance(s, bool) or not isinstance(s, int) or s < 0:
            raise FirewallBreach(f"TraderSituation.step {self.step!r} must be a non-negative int")
        object.__setattr__(self, "step", int(s))
        feats = TraderMemoryItem._parse(self.features_json)
        v = find_violations(feats)
        if v:
            raise FirewallBreach("TraderSituation refused: " + "; ".join(str(x) for x in v[:3]))
        for k, x in feats.items():
            _finite_float(x, f"features.{k}", -MAX_ABS_VALUE, MAX_ABS_VALUE)

    @classmethod
    def make(cls, step: int, features: Mapping[str, float], warm: bool = True) -> "TraderSituation":
        return cls(step, canonical_json(dict(features)), bool(warm))

    @property
    def features(self) -> dict:
        return TraderMemoryItem._parse(self.features_json)

    def to_dict(self) -> dict:
        return {"step": self.step, "features": self.features, "warm": self.warm}


@dataclasses.dataclass(frozen=True)
class TraderRelease:
    """The batch of memories the curator hands over on one session. Items are ordered by their opaque id - never by priority -
    and the weights sum to 1 (or the batch is empty), so the trader learns how to split its trust but not why."""
    step: int
    items: tuple[TraderMemoryItem, ...] = ()

    def __post_init__(self):
        s = _scalar(self.step)
        if isinstance(s, bool) or not isinstance(s, int) or s < 0:
            raise FirewallBreach(f"TraderRelease.step {self.step!r} must be a non-negative int")
        object.__setattr__(self, "step", int(s))
        items = tuple(self.items)
        if any(not isinstance(i, TraderMemoryItem) for i in items):
            raise FirewallBreach("TraderRelease holds only TraderMemoryItem")
        ids = [i.item_id for i in items]
        if len(set(ids)) != len(ids):
            raise FirewallBreach("TraderRelease has duplicate item ids")
        if ids != sorted(ids):
            raise FirewallBreach("TraderRelease items must be ordered by item id (order must not encode priority)")
        if items and abs(sum(i.weight for i in items) - 1.0) > 1e-6:
            raise FirewallBreach(f"TraderRelease weights sum to {sum(i.weight for i in items):.6f}, not 1")
        object.__setattr__(self, "items", items)

    def __len__(self):
        return len(self.items)

    def to_dict(self) -> dict:
        return {"step": self.step, "items": [i.to_dict() for i in self.items]}

    def digest(self) -> str:
        return stable_hash(self.to_dict())


def blind_json(obj: Any) -> str:
    """Serialise a trader record (or dict) for the trader, after re-checking it. The single exit for trader-bound text."""
    d = obj.to_dict() if hasattr(obj, "to_dict") else obj
    assert_trader_safe(d, type(obj).__name__)
    return canonical_json(d)


def leak_scan_text(text: str) -> dict[str, int]:
    """Count date-like hits in any text (a serialised release, a log line, a file path)."""
    return {name: len(pat.findall(text)) for name, pat in STRING_PATTERNS if pat.search(text)}


def release_year_hits(release: TraderRelease) -> dict[str, int]:
    """Date-like hits in everything the trader would receive from a release, both as text and as parsed numbers."""
    hits = leak_scan_text(canonical_json(release.to_dict()))
    hits.update({v.category: hits.get(v.category, 0) + 1 for v in find_violations(release.to_dict())})
    return hits


def item_identity(item: TraderMemoryItem) -> str:
    """What makes two items look the same to the trader: everything except the weight."""
    return stable_hash({"id": item.item_id, "kind": item.kind, "f": item.features_json, "lean": item.lean, "h": item.horizon})


def indistinguishable(a: TraderRelease, b: TraderRelease, tol: float = 1e-9) -> bool:
    """True when the trader could not tell two releases apart: same item identities and the same weights."""
    if [item_identity(i) for i in a.items] != [item_identity(i) for i in b.items]:
        return False
    return all(abs(x.weight - y.weight) <= tol for x, y in zip(a.items, b.items))


# ------------------------------------------------------------------------------------------------ static trader-path guard
CURATOR_MODULES = frozenset({"engine.learning.curator"})
CURATOR_SYMBOLS = frozenset({"Curator", "CuratorStore", "FiledMemory", "PriorityRecord"})
STORE_MARKERS = ("curator_store", "curator_audit", "year_filed", "filed_by_year")
TRADER_ENTRIES = ("engine/livesim.py", "engine/learning/learner.py", "engine/adaptive.py")
_DYNAMIC = frozenset({"import_module", "__import__"})


@dataclasses.dataclass(frozen=True)
class PathViolation:
    file: str
    line: int
    kind: str          # import | dynamic_import | symbol | store_read
    target: str
    via: str           # the chain of modules from a trader entry to this file

    def __str__(self):
        return f"{self.file}:{self.line} {self.kind} {self.target} (via {self.via})"


def _root(root) -> Path:
    if root is not None:
        return Path(root)
    from engine import config as K
    return Path(K.ROOT)


def module_name(path: Path, root: Path) -> str:
    rel = path.resolve().relative_to(root.resolve()).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def module_file(dotted: str, root: Path) -> Path | None:
    base = root.joinpath(*dotted.split("."))
    for cand in (base.with_suffix(".py"), base / "__init__.py"):
        if cand.exists():
            return cand
    return None


def _import_targets(tree: ast.AST, modname: str, is_pkg: bool) -> list[tuple[int, str, str]]:
    """(line, dotted module, kind) for every import - top level and lazy in-function alike - with relative imports resolved."""
    out: list[tuple[int, str, str]] = []
    pkg = modname if is_pkg else modname.rpartition(".")[0]
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                out.append((n.lineno, a.name, "import"))
        elif isinstance(n, ast.ImportFrom):
            if n.level:
                parts = pkg.split(".") if pkg else []
                base = ".".join(parts[:len(parts) - (n.level - 1)]) if n.level - 1 <= len(parts) else ""
                mod = f"{base}.{n.module}" if n.module and base else (n.module or base)
            else:
                mod = n.module or ""
            if mod:
                out.append((n.lineno, mod, "import"))
                for a in n.names:
                    out.append((n.lineno, f"{mod}.{a.name}", "import"))
        elif isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None
            if name in _DYNAMIC and n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
                out.append((n.lineno, n.args[0].value, "dynamic_import"))
    return out


def trader_path_violations(entries: Sequence[str] | None = None, root=None,
                           curator_modules: Iterable[str] = CURATOR_MODULES) -> list[PathViolation]:
    """Walk the import closure (only modules that exist under `root`) from each trader-path entry and report every edge that
    reaches a curator module, every use of a curator symbol, and every string that names the curator's store. Entries that do
    not exist are skipped (a missing file cannot import anything) but a closure of zero files is reported by the caller's test."""
    root = _root(root)
    curators = frozenset(curator_modules)
    found: list[PathViolation] = []
    seen: dict[str, str] = {}
    todo: list[tuple[str, str]] = []
    for e in (TRADER_ENTRIES if entries is None else entries):
        p = root / e
        if p.exists():
            m = module_name(p, root)
            todo.append((m, m))
    while todo:
        mod, via = todo.pop()
        if mod in seen or mod in curators:
            continue
        seen[mod] = via
        path = module_file(mod, root)
        if path is None:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = path.relative_to(root).as_posix()
        for line, target, kind in _import_targets(tree, mod, path.name == "__init__.py"):
            if target in curators:
                found.append(PathViolation(rel, line, kind, target, via))
            elif module_file(target, root) is not None and target not in seen:
                todo.append((target, f"{via} -> {target}"))
        for n in ast.walk(tree):
            if isinstance(n, ast.Name) and n.id in CURATOR_SYMBOLS:
                found.append(PathViolation(rel, n.lineno, "symbol", n.id, via))
            elif isinstance(n, ast.Attribute) and n.attr in CURATOR_SYMBOLS:
                found.append(PathViolation(rel, n.lineno, "symbol", n.attr, via))
            elif isinstance(n, ast.Constant) and isinstance(n.value, str):
                low = n.value.lower()
                for mk in STORE_MARKERS:
                    if mk in low:
                        found.append(PathViolation(rel, n.lineno, "store_read", mk, via))
    return sorted(set(found), key=lambda v: (v.file, v.line, v.kind, v.target))


def trader_closure(entries: Sequence[str] | None = None, root=None) -> list[str]:
    """The modules a blind run can reach from the trader entries (for reports and for tests that the guard saw something)."""
    root = _root(root)
    seen: set[str] = set()
    todo = [module_name(root / e, root) for e in (TRADER_ENTRIES if entries is None else entries) if (root / e).exists()]
    while todo:
        mod = todo.pop()
        if mod in seen:
            continue
        seen.add(mod)
        path = module_file(mod, root)
        if path is None:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for _, target, _k in _import_targets(tree, mod, path.name == "__init__.py"):
            if module_file(target, root) is not None and target not in seen:
                todo.append(target)
    return sorted(seen)


def assert_trader_path_clean(entries: Sequence[str] | None = None, root=None) -> int:
    """Raise FirewallBreach if the trader path touches the curator; returns the number of modules inspected."""
    v = trader_path_violations(entries, root)
    if v:
        raise FirewallBreach("trader path reaches the curator: " + "; ".join(str(x) for x in v[:3]))
    closure = trader_closure(entries, root)
    if not closure:
        raise FirewallBreach("trader path guard inspected no modules (entries missing?): it cannot pass on nothing")
    return len(closure)


# ------------------------------------------------------------------------------------------------ more blind-side surfaces
def path_violations(path: str | Path) -> list[Violation]:
    """A file path handed to (or built by) trader-side code: every component is checked, because a cache called AAPL_2008.parquet
    or a folder called y2008 tells the year as surely as a field does."""
    out = []
    parts = [p for p in re.split(r"[\\/]+", str(path)) if p and p not in (".", "..") and not re.fullmatch(r"[A-Za-z]:", p)]
    for i, part in enumerate(parts):
        out += [Violation(f"path[{i}]", r, part[:40]) for r in string_reasons(part)]
    return out


def assert_blind_paths(paths: Iterable[str | Path], what: str = "trader path") -> None:
    bad = [v for p in paths for v in path_violations(p)]
    if bad:
        raise FirewallBreach(f"{what} names the year: " + "; ".join(str(v) for v in bad[:3]))


def object_leak_scan(obj: Any, max_depth: int = 4, max_nodes: int = 5000) -> list[Violation]:
    """Walk a LIVE object (attributes, __slots__, containers) the way a curious trader could, and report anything date-like it
    holds. Cycle-safe; skips functions, classes and modules. Used on feed consumers, sessions and the records above."""
    seen: set[int] = set()
    out: list[Violation] = []
    budget = [max_nodes]

    def walk(x, path, depth):
        budget[0] -= 1
        if budget[0] < 0 or depth > max_depth or id(x) in seen:
            return
        x = _scalar(x)
        if x is None or isinstance(x, (bool, type)) or callable(x) or type(x).__name__ == "module":
            return
        if isinstance(x, (str, bytes, int, float)) or _datetime_kind(x):
            out.extend(Violation(path, v.category, v.detail) for v in find_violations(x, path))
            return
        seen.add(id(x))
        if isinstance(x, Mapping):
            for k, v in list(x.items())[:500]:
                out.extend(Violation(f"{path}[{k!r}]", w.category, w.detail) for w in find_violations({str(k): 0}))
                walk(v, f"{path}[{k!r}]", depth + 1)
        elif isinstance(x, (list, tuple, set, frozenset)):
            for i, v in enumerate(list(x)[:500]):
                walk(v, f"{path}[{i}]", depth + 1)
        else:
            names = list(getattr(x, "__dict__", {}).keys())
            for cls in type(x).__mro__:
                names += [s for s in getattr(cls, "__slots__", ()) if isinstance(s, str)]
            for name in names:
                if name.startswith("__"):
                    continue
                kr = key_reason(name)
                if kr:
                    out.append(Violation(f"{path}.{name}", "attribute_" + kr, name))
                try:
                    walk(getattr(x, name), f"{path}.{name}", depth + 1)
                except AttributeError:
                    continue

    walk(obj, type(obj).__name__, 0)
    return out


@dataclasses.dataclass(frozen=True)
class TraderDay:
    """Everything the trader is given for one session: today's relative situation and the curator's weighted memory. The step in
    both must agree, so a stale release can never be paired with a newer situation."""
    situation: TraderSituation
    release: TraderRelease

    def __post_init__(self):
        if self.situation.step != self.release.step:
            raise FirewallBreach(f"situation step {self.situation.step} != release step {self.release.step}")

    def to_dict(self) -> dict:
        return {"situation": self.situation.to_dict(), "release": self.release.to_dict()}

    def json(self) -> str:
        return blind_json(self)


class BlindLog:
    """A logger for trader-side code: every message is checked before it is stored, and a message that names a date is refused
    (or scrubbed and counted). Trader code logs through this so a stack trace or a debug print cannot carry the year."""

    def __init__(self, refuse: bool = True):
        self.refuse = refuse
        self.lines: list[str] = []
        self.ledger = ScrubLedger()

    def log(self, message: str, **fields: Any) -> None:
        payload = {"message": str(message), **fields}
        clean = self.ledger.scrub(payload, refuse=self.refuse)
        self.lines.append(canonical_json(clean))


def weight_entropy(release: TraderRelease) -> float:
    """Shannon entropy (nats) of the release weights: 0 = all trust on one memory, log(n) = evenly spread."""
    w = np.asarray([i.weight for i in release.items], dtype="float64")
    w = w[w > 0]
    return float(-(w * np.log(w)).sum()) if len(w) else 0.0


def compare_releases(a: TraderRelease, b: TraderRelease) -> dict:
    """How two releases differ as the trader would see it: ids only in one, and the L1 distance between weight vectors."""
    wa, wb = {i.item_id: i.weight for i in a.items}, {i.item_id: i.weight for i in b.items}
    keys = sorted(set(wa) | set(wb))
    return {"only_a": sorted(set(wa) - set(wb)), "only_b": sorted(set(wb) - set(wa)), "shared": len(set(wa) & set(wb)),
            "weight_l1": float(sum(abs(wa.get(k, 0.0) - wb.get(k, 0.0)) for k in keys)),
            "jaccard": (len(set(wa) & set(wb)) / len(keys)) if keys else 1.0}
