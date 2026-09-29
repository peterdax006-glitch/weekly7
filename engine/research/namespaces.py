"""The two namespaces of the research brain (contract C66 sections 29-31; canon C56, C64, C66; checklist D, RT07-RT09).
IMPLEMENTED — NOT VALIDATED (C63: code + unit tests only; no real-data run was made).

Section 29 asks for LIVE_POINT_IN_TIME_STATE and MATURED_RESEARCH_STATE with no implicit shared state, no hidden shortcuts, no
convenience imports, no future-derived cache or feature, no timestamp ambiguity, and provenance on every information object.
This module is the storage half of that; engine.research.firewall is the only road between the two stores.

  InfoObject       one dated fact (price, filing, label, research result, pattern status, ...). Its FIRST USABLE DATE is computed
                   from its own dates, its payload's dates, its provenance and its kind's publication lag - never taken on trust.
                   A payload holding a date later than the object's declared availability is a lie and is refused.
  NamespaceStore   append-only, hash-chained, deep-frozen (payloads are copied and made read-only on the way in, so no container is
                   ever shared between the stores). LiveStore has a forward-only clock and refuses anything not yet knowable at it;
                   research-origin objects enter it only with a single-use CrossingTicket signed by the firewall.
                   ResearchStore may hold years and tickers (trusted side) and hands out core.MaturedRecord views.
  SharedStateAudit object-graph and module-level checks that the two worlds share nothing implicitly.

Built on: engine.research.core (Namespace, MaturedRecord), engine.learning.core (Provenance, FirewallBreach, stable_hash),
engine.learning.future_firewall (DEFAULT_RULES publication lags, frame_dates). Nothing here reads a clock or draws randomly."""
from __future__ import annotations

import ast
import dataclasses
import datetime as dt
import re
import types
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from engine.learning.core import FirewallBreach, Provenance, _StrEnum, as_date, canonical_json, stable_hash
from engine.learning.future_firewall import DEFAULT_RULES, InputKind, frame_dates
from engine.research.core import MaturedRecord, Namespace

LABEL = "IMPLEMENTED — NOT VALIDATED"


class InfoKind(_StrEnum):                  # section 29: every class of information the trader must never see early
    PRICE = "PRICE"
    VOLUME = "VOLUME"
    FILING = "FILING"
    EARNINGS = "EARNINGS"
    LABEL = "LABEL"
    METADATA = "METADATA"
    CORPORATE_ACTION = "CORPORATE_ACTION"
    MARKET_STATE = "MARKET_STATE"
    EVENT = "EVENT"
    FEATURE = "FEATURE"
    CONFIG = "CONFIG"
    LEARNED_STATE = "LEARNED_STATE"
    RESEARCH_RESULT = "RESEARCH_RESULT"
    PATTERN_STATUS = "PATTERN_STATUS"
    PATTERN_HEALTH = "PATTERN_HEALTH"
    EXPERIMENT_RESULT = "EXPERIMENT_RESULT"
    YEAR_IDENTITY = "YEAR_IDENTITY"


# outcomes: known only STRICTLY after every date they rest on (an outcome that matures on `now` is not yet known)
MATURING = frozenset({InfoKind.LABEL, InfoKind.LEARNED_STATE, InfoKind.RESEARCH_RESULT, InfoKind.PATTERN_STATUS,
                      InfoKind.PATTERN_HEALTH, InfoKind.EXPERIMENT_RESULT})
# only the research (audit) world may hold these at all
RESEARCH_ONLY = frozenset({InfoKind.YEAR_IDENTITY})
# publication-lag rule reused from the future firewall (one table of lags in the repo, not two)
LAG_KIND = types.MappingProxyType({
    InfoKind.PRICE: InputKind.PRICE, InfoKind.VOLUME: InputKind.PRICE, InfoKind.MARKET_STATE: InputKind.PRICE,
    InfoKind.FILING: InputKind.FILING, InfoKind.EARNINGS: InputKind.FUNDAMENTAL, InfoKind.LABEL: InputKind.LABEL,
    InfoKind.METADATA: InputKind.FEATURE, InfoKind.CORPORATE_ACTION: InputKind.EVENT, InfoKind.EVENT: InputKind.EVENT,
    InfoKind.FEATURE: InputKind.FEATURE, InfoKind.CONFIG: InputKind.CONFIG, InfoKind.LEARNED_STATE: InputKind.MEMORY,
    InfoKind.RESEARCH_RESULT: InputKind.MEMORY, InfoKind.PATTERN_STATUS: InputKind.MEMORY, InfoKind.PATTERN_HEALTH: InputKind.MEMORY,
    InfoKind.EXPERIMENT_RESULT: InputKind.MEMORY})
CLOSE_HOUR = 16                            # a stamp after the 16:00 close is first usable at the next session
MAX_WALK = 50_000
_ISO_FULL = re.compile(r"^(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])(?:[T ][0-9:.+-Z]*)?$")
_PARTIAL = re.compile(r"^\s*(?:19|20)\d{2}(?:[-/](?:0?[1-9]|1[0-2]))?\s*$")
_ISO_INSIDE = re.compile(r"(?<!\d)((?:19|20)\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])(?!\d)")


class AmbiguousTimestamp(FirewallBreach):
    """A timestamp that cannot be pinned to one session (section 29: no timestamp ambiguity)."""


class SharedStateError(FirewallBreach):
    """An object reaches into the other namespace's state (section 29: no implicit shared state)."""


# ------------------------------------------------------------------------------------------------ time
def min_lag_days(kind: InfoKind) -> int:
    ik = LAG_KIND.get(kind)
    return 0 if ik is None else int(DEFAULT_RULES[ik].min_lag_days)


def moment(x, what: str = "timestamp") -> dt.date:
    """The session a stamp is first usable at. Whole dates are used as is; a time after the close moves to the next day; an
    aware time is read in exchange time. Refused: missing values, numbers (seconds? ms? an ordinal?), and partial dates such as
    '2008' or '2008-03' (which day?)."""
    if x is None or (isinstance(x, float) and np.isnan(x)) or x is pd.NaT:
        raise AmbiguousTimestamp(f"{what}: missing")
    if isinstance(x, (bool, int, float, np.integer, np.floating)):
        raise AmbiguousTimestamp(f"{what}: bare number {x!r} has no unit (epoch? ordinal? year?)")
    if isinstance(x, str):
        s = x.strip()
        if not s or _PARTIAL.match(s):
            raise AmbiguousTimestamp(f"{what}: {x!r} does not name one day")
        if not _ISO_FULL.match(s):
            raise AmbiguousTimestamp(f"{what}: {x!r} is not an ISO date (day-first vs month-first cannot be told apart)")
        x = pd.Timestamp(s)
    if isinstance(x, np.datetime64):
        x = pd.Timestamp(x)
    if isinstance(x, dt.datetime):
        t = pd.Timestamp(x)
        if t.tzinfo is not None:
            t = t.tz_convert("America/New_York").tz_localize(None)
        d = t.date()
        after = (t.hour, t.minute, t.second, t.microsecond) > (CLOSE_HOUR, 0, 0, 0)
        return d + dt.timedelta(days=1) if after else d
    if isinstance(x, dt.date):
        return x
    raise AmbiguousTimestamp(f"{what}: unsupported type {type(x).__name__}")


def optional_moment(x, what: str) -> dt.date | None:
    if x is None or x == "":
        return None
    return moment(x, what)


def payload_dates(obj: Any, _depth: int = 0, _budget: list | None = None) -> list[dt.date]:
    """Every date a payload holds: date/datetime objects, ISO dates inside strings, the dated index or date columns of frames.
    A date buried in a nested dict or a free-text note still dates the payload."""
    budget = _budget if _budget is not None else [MAX_WALK]
    budget[0] -= 1
    if budget[0] < 0 or _depth > 10:
        raise FirewallBreach("payload too large or too deep to date: cannot show it is point-in-time")
    out: list[dt.date] = []
    if obj is None or isinstance(obj, (bool, int, float, np.number)):
        return out
    if isinstance(obj, (dt.date, np.datetime64, pd.Timestamp)):
        return [moment(obj, "payload date")]
    if isinstance(obj, str):
        return [dt.date(int(a), int(b), int(c)) for a, b, c in _ISO_INSIDE.findall(obj)]
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        idx = frame_dates(obj)
        idx = idx[~idx.isna()] if len(idx) else idx
        if len(idx):
            out += [idx.min().date(), idx.max().date()]
        if isinstance(obj, pd.DataFrame):
            for c in obj.columns:
                if pd.api.types.is_datetime64_any_dtype(obj[c]):
                    col = obj[c].dropna()
                    if len(col):
                        out += [pd.Timestamp(col.min()).date(), pd.Timestamp(col.max()).date()]
        return out
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind == "M" and obj.size:
            v = obj[~np.isnat(obj)]
            return [pd.Timestamp(v.min()).date(), pd.Timestamp(v.max()).date()] if v.size else out
        if obj.dtype.kind in "OU":
            for v in obj.ravel()[:2000]:
                out += payload_dates(v, _depth + 1, budget)
        return out
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            out += payload_dates(k, _depth + 1, budget)
            out += payload_dates(v, _depth + 1, budget)
        return out
    if isinstance(obj, (list, tuple, set, frozenset)):
        for v in obj:
            out += payload_dates(v, _depth + 1, budget)
        return out
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            out += payload_dates(getattr(obj, f.name), _depth + 1, budget)
        return out
    return out


def frame_timestamp_findings(frame: pd.DataFrame | pd.Series, name: str = "frame") -> list[str]:
    """Section 29 'no timestamp ambiguity' for dated frames: an undated or unreadable index, missing stamps, duplicated or
    out-of-order dates, and intraday stamps (which session is a 17:30 bar?) are each named. An empty frame is fine."""
    if len(frame) == 0:
        return []
    idx = frame.index
    lvl = idx.get_level_values(0) if isinstance(idx, pd.MultiIndex) else idx
    if not isinstance(lvl, pd.DatetimeIndex):
        dated = frame_dates(frame)
        if len(dated) == 0 or dated.isna().all():
            return [f"{name}: index is not dated, so no row can be placed in time"]
        lvl = dated
    out = []
    n_nat = int(lvl.isna().sum())
    if n_nat:
        out.append(f"{name}: {n_nat} rows have no timestamp")
    ok = lvl[~lvl.isna()]
    if ok.tz is not None:
        ok = ok.tz_convert("America/New_York").tz_localize(None)
    if not isinstance(idx, pd.MultiIndex):
        if ok.has_duplicates:
            out.append(f"{name}: {int(ok.duplicated().sum())} duplicated timestamps (which value was known?)")
        if not ok.is_monotonic_increasing:
            out.append(f"{name}: timestamps out of order")
    intraday = ok[(ok.hour != 0) | (ok.minute != 0) | (ok.second != 0)]
    if len(intraday) and len(intraday.normalize().unique()) != len(intraday):
        out.append(f"{name}: several intraday stamps share a session: the bar's close time is ambiguous")
    late = intraday[(intraday.hour > CLOSE_HOUR) | ((intraday.hour == CLOSE_HOUR) & ((intraday.minute > 0) | (intraday.second > 0)))]
    if len(late):
        out.append(f"{name}: {len(late)} stamps after the {CLOSE_HOUR}:00 close are dated to their calendar day, not the next session")
    return out


# ------------------------------------------------------------------------------------------------ freezing
def deep_freeze(obj: Any, _depth: int = 0) -> Any:
    """A read-only deep copy: dicts become MappingProxyType, lists tuples, sets frozensets, arrays read-only copies, frames
    copies whose arrays are read-only where pandas allows. A store object, a module, a function or any other live object is
    refused: it would carry state across the wall by reference."""
    if _depth > 12:
        raise FirewallBreach("payload nested deeper than 12 levels")
    if obj is None or isinstance(obj, (bool, int, float, str, bytes, np.number, dt.date, pd.Timestamp)):
        return obj
    if isinstance(obj, NamespaceStore):
        raise SharedStateError(f"payload holds a reference to the {obj.namespace} store")
    if isinstance(obj, np.ndarray):
        a = obj.copy()
        a.setflags(write=False)
        return a
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.copy(deep=True)
    if isinstance(obj, Mapping):
        return types.MappingProxyType({k: deep_freeze(v, _depth + 1) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return tuple(deep_freeze(v, _depth + 1) for v in obj)
    if isinstance(obj, (set, frozenset)):
        return frozenset(deep_freeze(v, _depth + 1) for v in obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type) and getattr(type(obj), "__dataclass_params__").frozen:
        return obj
    if isinstance(obj, _StrEnum):
        return obj
    raise SharedStateError(f"payload holds a live {type(obj).__name__} object: only plain data may be filed")


def thaw(obj: Any) -> Any:
    """A mutable deep copy of a frozen payload (for research-side analysis; never writes back into the store)."""
    if isinstance(obj, Mapping):
        return {k: thaw(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [thaw(v) for v in obj]
    if isinstance(obj, frozenset):
        return set(thaw(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return obj.copy()
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.copy(deep=True)
    return obj


def _payload_digest(payload: Any) -> str:
    def plain(o):
        if isinstance(o, (pd.DataFrame, pd.Series)):
            return {"frame": pd.util.hash_pandas_object(o, index=True).astype("int64").tolist(),
                    "cols": [str(c) for c in (o.columns if isinstance(o, pd.DataFrame) else [o.name])]}
        if isinstance(o, np.ndarray):
            return {"array": o.tolist(), "dtype": str(o.dtype)}
        if isinstance(o, Mapping):
            return {str(k): plain(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [plain(v) for v in o]
        if isinstance(o, (set, frozenset)):
            return sorted((plain(v) for v in o), key=canonical_json)
        return o
    return stable_hash(plain(payload), 24)


# ------------------------------------------------------------------------------------------------ the information object
@dataclasses.dataclass(frozen=True)
class InfoObject:
    """One dated fact with provenance. `observed_at` is the date the fact describes (a bar's session, a label's horizon close,
    a status's as-of date); `available_at` is when it became public (None = observed + the kind's minimum lag). `tags` is
    research-side audit metadata (real year, ticker, run) and is legal only in MATURED_RESEARCH_STATE."""
    object_id: str
    kind: InfoKind
    namespace: Namespace
    observed_at: Any
    payload: Mapping[str, Any]
    provenance: Provenance
    available_at: Any = None
    origin: str = ""                        # dotted module that produced it
    parents: tuple[str, ...] = ()
    cache_key: str = ""
    tags: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "kind", InfoKind.parse(self.kind))
        object.__setattr__(self, "namespace", Namespace.parse(self.namespace))
        object.__setattr__(self, "parents", tuple(str(p) for p in self.parents))

    # ---- dates
    def evidence_dates(self) -> list[dt.date]:
        """Every date this object rests on: what it describes, when it was published, what its payload holds and what its
        provenance says it has seen."""
        ds = [moment(self.observed_at, f"{self.object_id}.observed_at")]
        av = optional_moment(self.available_at, f"{self.object_id}.available_at")
        if av is not None:
            ds.append(av)
        ds += payload_dates(self.payload)
        for name in ("learned_at", "outcomes_seen_through"):
            v = optional_moment(getattr(self.provenance, name), f"{self.object_id}.provenance.{name}")
            if v is not None:
                ds.append(v)
        return ds

    def evidence_through(self) -> dt.date:
        return max(self.evidence_dates())

    def evidence_from(self) -> dt.date:
        return min([moment(self.observed_at, "observed_at")] + payload_dates(self.payload))

    def knowable_at(self) -> dt.date:
        """The first session at which this object may be used. Outcomes (MATURING kinds) and anything the provenance says was
        LEARNED need the day after their newest evidence; published facts need their lag; same-close facts their own close."""
        obs = moment(self.observed_at, f"{self.object_id}.observed_at")
        first = obs + dt.timedelta(days=min_lag_days(self.kind))
        av = optional_moment(self.available_at, f"{self.object_id}.available_at")
        if av is not None:
            first = max(first, av)
        pay = payload_dates(self.payload)
        if self.kind in MATURING:
            return max([first, obs + dt.timedelta(days=1)] + [d + dt.timedelta(days=1) for d in self.evidence_dates()])
        first = max([first] + pay)
        for name in ("learned_at", "outcomes_seen_through"):
            v = optional_moment(getattr(self.provenance, name), name)
            if v is not None:
                first = max(first, v + dt.timedelta(days=1))
        return first

    def usable_at(self, now) -> bool:
        return self.knowable_at() <= as_date(now)

    # ---- checks
    def validate(self) -> list[str]:
        errs: list[str] = []
        if not self.object_id or not isinstance(self.object_id, str):
            errs.append("object_id missing")
        if not isinstance(self.payload, Mapping):
            errs.append("payload must be a mapping")
        errs += [f"provenance: {e}" for e in self.provenance.check()] if isinstance(self.provenance, Provenance) \
            else ["provenance missing"]
        try:
            obs = moment(self.observed_at, "observed_at")
            av = optional_moment(self.available_at, "available_at")
            if av is not None and av < obs + dt.timedelta(days=min_lag_days(self.kind)):
                errs.append(f"available_at {av} is earlier than physically possible for {self.kind} observed {obs} "
                            f"(minimum lag {min_lag_days(self.kind)} days)")
            declared = max([obs] + ([av] if av is not None else []) +
                           [d for d in (optional_moment(self.provenance.learned_at, "learned_at"),
                                        optional_moment(self.provenance.outcomes_seen_through, "outcomes_seen_through")) if d])
            if isinstance(self.payload, Mapping):
                for k, v in self.payload.items():
                    if isinstance(v, (pd.DataFrame, pd.Series)):
                        errs += [f"timestamp: {e}" for e in frame_timestamp_findings(v, f"payload.{k}")]
            late = [d for d in payload_dates(self.payload) if d > declared]
            if late:
                errs.append(f"payload holds dates through {max(late)}, after everything the object declares ({declared}): "
                            "it smuggles later information")
        except FirewallBreach as e:
            errs.append(f"timestamp: {e}")
        if self.kind in RESEARCH_ONLY and self.namespace != Namespace.MATURED_RESEARCH:
            errs.append(f"{self.kind} may exist only in {Namespace.MATURED_RESEARCH}")
        if self.tags and self.namespace != Namespace.MATURED_RESEARCH:
            errs.append("audit tags (real year / ticker / run) are legal only in the research namespace")
        if self.object_id in self.parents:
            errs.append("object lists itself as a parent")
        return errs

    def digest(self) -> str:
        return stable_hash({"id": self.object_id, "kind": self.kind, "ns": self.namespace, "obs": str(self.observed_at),
                            "av": str(self.available_at), "payload": _payload_digest(self.payload), "prov": self.provenance,
                            "origin": self.origin, "parents": self.parents, "cache": self.cache_key,
                            "tags": _payload_digest(self.tags)}, 24)

    def all_parents(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(p.split("@", 1)[0] for p in self.parents + tuple(self.provenance.parents)
                                   if p and p.split("@", 1)[0] != self.object_id))

    def with_namespace(self, ns: Namespace, **changes) -> "InfoObject":
        return dataclasses.replace(self, namespace=ns, **changes)


# ------------------------------------------------------------------------------------------------ crossing tickets
@dataclasses.dataclass(frozen=True)
class CrossingTicket:
    """Proof that the firewall inspected THIS content at THIS decision date. Single use; the signature binds every field, so a
    ticket cannot be moved to other content or reused on a later day."""
    ticket_id: str
    object_id: str
    object_digest: str
    decided_now: str
    release_id: str
    signature: str

    def body(self) -> dict:
        return {"t": self.ticket_id, "o": self.object_id, "d": self.object_digest, "n": self.decided_now, "r": self.release_id}


def sign_ticket(secret: str, object_id: str, object_digest: str, decided_now, release_id: str) -> CrossingTicket:
    if not secret:
        raise FirewallBreach("a ticket needs a signing secret")
    now = as_date(decided_now).isoformat()
    tid = "T" + stable_hash({"o": object_id, "d": object_digest, "n": now, "r": release_id}, 16)
    body = {"t": tid, "o": object_id, "d": object_digest, "n": now, "r": release_id}
    return CrossingTicket(tid, object_id, object_digest, now, release_id, stable_hash({"secret": secret, **body}, 32))


def ticket_valid(secret: str, t: CrossingTicket) -> bool:
    return bool(secret) and stable_hash({"secret": secret, **t.body()}, 32) == t.signature


# ------------------------------------------------------------------------------------------------ stores
@dataclasses.dataclass(frozen=True)
class StoreEntry:
    seq: int
    object_id: str
    digest: str
    prev: str
    link: str


class NamespaceStore:
    """Append-only store of one namespace. Objects are validated, deep-frozen and chained on the way in; an id can never be
    rewritten (history is immutable). Reads hand out the frozen objects; nothing inside can be mutated in place."""
    namespace: Namespace = Namespace.MATURED_RESEARCH

    def __init__(self, name: str = ""):
        self.name = name or str(self.namespace)
        self.token = stable_hash({"ns": self.namespace, "name": self.name, "id": id(self)}, 12)
        self._objs: dict[str, InfoObject] = {}
        self._chain: list[StoreEntry] = []

    # ---- writing
    def _admit(self, obj: InfoObject) -> None:
        """Namespace-specific admission rule; raise FirewallBreach to refuse."""

    def put(self, obj: InfoObject) -> str:
        if not isinstance(obj, InfoObject):
            raise FirewallBreach(f"{self.name}: only InfoObject records may be filed, not {type(obj).__name__}")
        if obj.namespace != self.namespace:
            raise FirewallBreach(f"{self.name}: object {obj.object_id} belongs to {obj.namespace}, not {self.namespace}")
        errs = obj.validate()
        if errs:
            raise FirewallBreach(f"{self.name}: object {obj.object_id} refused: " + "; ".join(errs[:3]))
        for p in obj.all_parents():
            if p not in self._objs:
                raise FirewallBreach(f"{self.name}: object {obj.object_id} names parent {p!r} that is not in this store "
                                     "(lineage must be auditable inside one namespace)")
        self._admit(obj)
        frozen = dataclasses.replace(obj, payload=deep_freeze(obj.payload), tags=deep_freeze(obj.tags))
        dg = frozen.digest()
        old = self._objs.get(obj.object_id)
        if old is not None:
            if old.digest() == dg:
                return dg
            raise FirewallBreach(f"{self.name}: object id {obj.object_id} already filed with different content (history is immutable)")
        prev = self._chain[-1].link if self._chain else "genesis"
        link = stable_hash({"prev": prev, "id": obj.object_id, "digest": dg}, 24)
        self._chain.append(StoreEntry(len(self._chain), obj.object_id, dg, prev, link))
        self._objs[obj.object_id] = frozen
        return dg

    def put_many(self, objs: Iterable[InfoObject]) -> list[str]:
        return [self.put(o) for o in objs]

    # ---- reading
    def get(self, object_id: str) -> InfoObject:
        if object_id not in self._objs:
            raise KeyError(object_id)
        return self._objs[object_id]

    def __contains__(self, object_id) -> bool:
        return object_id in self._objs

    def __len__(self) -> int:
        return len(self._objs)

    def ids(self) -> tuple[str, ...]:
        return tuple(e.object_id for e in self._chain)

    def objects(self, kind: InfoKind | None = None) -> list[InfoObject]:
        return [self._objs[i] for i in self.ids() if kind is None or self._objs[i].kind == kind]

    def holds(self, obj: InfoObject) -> bool:
        """True only if this exact content was filed here (same id AND same digest)."""
        cur = self._objs.get(obj.object_id)
        return cur is not None and cur.digest() == dataclasses.replace(obj, payload=deep_freeze(obj.payload),
                                                                        tags=deep_freeze(obj.tags)).digest()

    def head(self) -> str:
        return self._chain[-1].link if self._chain else "genesis"

    def verify(self) -> list[str]:
        """Re-derive the chain from the stored objects; any edit through a back door shows up as a broken link."""
        errs, prev = [], "genesis"
        for e in self._chain:
            obj = self._objs.get(e.object_id)
            if obj is None:
                errs.append(f"seq {e.seq}: object {e.object_id} vanished")
                continue
            if obj.digest() != e.digest:
                errs.append(f"seq {e.seq}: object {e.object_id} content changed after filing")
            if e.prev != prev or stable_hash({"prev": prev, "id": e.object_id, "digest": e.digest}, 24) != e.link:
                errs.append(f"seq {e.seq}: chain link broken")
            prev = e.link
        if len(self._chain) != len(self._objs):
            errs.append(f"{len(self._objs)} objects but {len(self._chain)} chain entries")
        return errs

    # ---- lineage
    def lineage(self, object_id: str, max_depth: int = 64) -> tuple[list[str], list[str], bool]:
        """(ancestors, missing parents, cycle) of an object, walking both declared and provenance parents."""
        seen: list[str] = []
        missing: list[str] = []
        cycle = False
        stack = [(object_id, (), 0)]
        while stack:
            k, path, depth = stack.pop()
            if k in path or depth > max_depth:
                cycle = True
                continue
            obj = self._objs.get(k)
            if obj is None:
                missing.append(k)
                continue
            if k != object_id and k not in seen:
                seen.append(k)
            stack += [(p, path + (k,), depth + 1) for p in obj.all_parents()]
        return seen, sorted(set(missing)), cycle

    def effective_knowable(self, object_id: str) -> tuple[dt.date, str]:
        """(first usable date, the object that sets it): an object is no older than its youngest ancestor."""
        obj = self.get(object_id)
        best, who = obj.knowable_at(), object_id
        anc, missing, cycle = self.lineage(object_id)
        if missing or cycle:
            raise FirewallBreach(f"lineage of {object_id} is not auditable (missing={missing}, cycle={cycle})")
        for a in anc:
            k = self._objs[a].knowable_at()
            if k > best:
                best, who = k, a
        return best, who

    def evidence_span(self, object_id: str) -> tuple[dt.date, dt.date]:
        """Earliest and newest real date the object or any ancestor rests on (trusted side: which years it is filed under)."""
        anc, _, _ = self.lineage(object_id)
        objs = [self.get(object_id)] + [self._objs[a] for a in anc if a in self._objs]
        return min(o.evidence_from() for o in objs), max(o.evidence_through() for o in objs)

    def manifest(self) -> dict:
        kinds: dict[str, int] = {}
        for o in self._objs.values():
            kinds[str(o.kind)] = kinds.get(str(o.kind), 0) + 1
        return {"namespace": str(self.namespace), "name": self.name, "n": len(self), "head": self.head(),
                "kinds": dict(sorted(kinds.items())), "label": LABEL}


class ResearchStore(NamespaceStore):
    """MATURED_RESEARCH_STATE. May hold real dates, years, tickers and audit tags (it is the trusted side). Every object must say
    what outcomes it has seen, so its maturity can be checked; nothing here reaches a decision except through the firewall."""
    namespace = Namespace.MATURED_RESEARCH

    def _admit(self, obj: InfoObject) -> None:
        if not (obj.provenance.outcomes_seen_through or obj.provenance.learned_at):
            raise FirewallBreach(f"{self.name}: research object {obj.object_id} does not record what outcomes it has seen")

    def matured_record(self, object_id: str) -> MaturedRecord:
        """The core.MaturedRecord view: matured_at = the newest evidence date of the object and its whole ancestry."""
        obj = self.get(object_id)
        _, through = self.evidence_span(object_id)
        return MaturedRecord(object_id, through.isoformat(), dict(obj.payload), obj.provenance)

    def filed_years(self, object_id: str) -> tuple[int, ...]:
        a, b = self.evidence_span(object_id)
        return tuple(range(a.year, b.year + 1))

    def by_year(self) -> dict[int, list[str]]:
        out: dict[int, list[str]] = {}
        for i in self.ids():
            for y in self.filed_years(i):
                out.setdefault(y, []).append(i)
        return dict(sorted(out.items()))


class LiveStore(NamespaceStore):
    """LIVE_POINT_IN_TIME_STATE. A forward-only clock; nothing may be filed before it is knowable at the clock (so a future-derived
    cache or feature cannot even be stored), research-only kinds are refused, and an object whose origin is the research package
    enters only with a valid, unused CrossingTicket from the firewall."""
    namespace = Namespace.LIVE_POINT_IN_TIME
    RESEARCH_ORIGIN = "engine.research"

    def __init__(self, name: str = "", verifier: Callable[[CrossingTicket], bool] | None = None):
        super().__init__(name)
        self.clock: dt.date | None = None
        self._verifier = verifier
        self._used: set[str] = set()
        self._tickets: dict[str, str] = {}

    def advance(self, now) -> dt.date:
        d = moment(now, "clock")
        if self.clock is not None and d < self.clock:
            raise FirewallBreach(f"live clock cannot run backwards ({self.clock} -> {d})")
        self.clock = d
        return d

    def _need_clock(self) -> dt.date:
        if self.clock is None:
            raise FirewallBreach("live store has no clock: call advance(now) first (no implicit clock)")
        return self.clock

    def _admit(self, obj: InfoObject) -> None:
        now = self._need_clock()
        if obj.kind in RESEARCH_ONLY:
            raise FirewallBreach(f"{obj.kind} may never enter live state")
        k = obj.knowable_at()
        for p in obj.all_parents():
            k = max(k, self.effective_knowable(p)[0])
        if k > now:
            raise FirewallBreach(f"live object {obj.object_id} is first knowable {k}, after the clock {now}: future-derived state")
        if obj.origin.startswith(self.RESEARCH_ORIGIN) and obj.object_id not in self._tickets:
            raise FirewallBreach(f"object {obj.object_id} comes from {obj.origin} without a firewall ticket")

    def accept(self, obj: InfoObject, ticket: CrossingTicket) -> str:
        """File a research-origin object that the firewall released. The ticket must be signed, unused, for this content and for
        a decision date the clock has reached."""
        now = self._need_clock()
        if self._verifier is None:
            raise FirewallBreach("live store has no ticket verifier: it cannot accept released research")
        if ticket.ticket_id in self._used:
            raise FirewallBreach(f"ticket {ticket.ticket_id} already used (replayed ticket)")
        if not self._verifier(ticket):
            raise FirewallBreach(f"ticket {ticket.ticket_id} signature invalid")
        if ticket.object_id != obj.object_id or ticket.object_digest != obj.digest():
            raise FirewallBreach(f"ticket {ticket.ticket_id} was issued for other content")
        if as_date(ticket.decided_now) > now:
            raise FirewallBreach(f"ticket decided at {ticket.decided_now}, after the clock {now}")
        self._tickets[obj.object_id] = ticket.ticket_id
        try:
            dg = self.put(obj)
        except FirewallBreach:
            self._tickets.pop(obj.object_id, None)
            raise
        self._used.add(ticket.ticket_id)
        return dg

    def view(self, now=None) -> list[InfoObject]:
        """What the trader may read at `now` (default: the clock). Asking beyond the clock is itself a breach."""
        clock = self._need_clock()
        d = clock if now is None else moment(now, "view")
        if d > clock:
            raise FirewallBreach(f"view at {d} is after the live clock {clock}")
        return [o for o in self.objects() if o.knowable_at() <= d]


# ------------------------------------------------------------------------------------------------ shared-state audit
@dataclasses.dataclass(frozen=True)
class SharedRef:
    kind: str                # container | store | module_state
    where: str
    detail: str

    def __str__(self):
        return f"{self.kind} at {self.where}: {self.detail}"


def _containers(obj: Any, path: str, out: dict[int, str], depth: int = 0, budget: list | None = None) -> None:
    budget = budget if budget is not None else [MAX_WALK]
    budget[0] -= 1
    if budget[0] < 0 or depth > 12 or obj is None or isinstance(obj, (bool, int, float, str, bytes, np.number, dt.date)):
        return
    if isinstance(obj, (dict, list, set, np.ndarray, pd.DataFrame, pd.Series, types.MappingProxyType)):
        out.setdefault(id(obj), path)
    if isinstance(obj, np.ndarray) and obj.base is not None:
        out.setdefault(id(obj.base), path + ".base")
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            _containers(v, f"{path}.{k}", out, depth + 1, budget)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for i, v in enumerate(obj):
            _containers(v, f"{path}[{i}]", out, depth + 1, budget)


def shared_references(a: NamespaceStore, b: NamespaceStore) -> list[SharedRef]:
    """Mutable containers reachable from BOTH stores' objects (one list object filed twice would let a write on one side appear
    on the other). deep_freeze copies on the way in, so this is empty unless something bypassed `put`."""
    if a is b or a._objs is b._objs:
        return [SharedRef("store", a.name, "the two namespaces are the same store object")]
    ca: dict[int, str] = {}
    cb: dict[int, str] = {}
    for o in a.objects():
        _containers(o.payload, f"{a.name}:{o.object_id}", ca)
        _containers(o.tags, f"{a.name}:{o.object_id}.tags", ca)
    for o in b.objects():
        _containers(o.payload, f"{b.name}:{o.object_id}", cb)
        _containers(o.tags, f"{b.name}:{o.object_id}.tags", cb)
    return [SharedRef("container", ca[k], f"same object also at {cb[k]}") for k in sorted(set(ca) & set(cb), key=str)]


def reachable_stores(obj: Any, max_depth: int = 6, max_nodes: int = 20_000) -> list[SharedRef]:
    """Walk a live object graph (attributes, __slots__, containers, closures of bound callables) and report every NamespaceStore
    it can reach: a trader-side object that holds the research store can read it whenever it likes."""
    out: list[SharedRef] = []
    seen: set[int] = set()
    budget = [max_nodes]
    todo: list[tuple[Any, str, int]] = [(obj, "<root>", 0)]
    while todo:
        x, path, depth = todo.pop()
        budget[0] -= 1
        if budget[0] < 0:
            out.append(SharedRef("store", path, f"graph larger than {max_nodes} nodes: cannot show it is clean"))
            break
        if id(x) in seen or depth > max_depth or x is None or isinstance(x, (bool, int, float, str, bytes, np.number, dt.date)):
            continue
        seen.add(id(x))
        if isinstance(x, NamespaceStore):
            out.append(SharedRef("store", path, f"reaches the {x.namespace} store {x.name!r}"))
            continue
        if isinstance(x, (types.ModuleType, type)):
            continue
        if isinstance(x, Mapping):
            todo += [(v, f"{path}.{k}", depth + 1) for k, v in list(x.items())[:2000]]
        elif isinstance(x, (list, tuple, set, frozenset)):
            todo += [(v, f"{path}[{i}]", depth + 1) for i, v in enumerate(list(x)[:2000])]
        elif isinstance(x, (types.FunctionType, types.MethodType)):
            fn = getattr(x, "__func__", x)
            if getattr(x, "__self__", None) is not None:
                todo.append((x.__self__, f"{path}.__self__", depth + 1))
            for i, c in enumerate(getattr(fn, "__closure__", None) or ()):
                try:
                    todo.append((c.cell_contents, f"{path}.<closure{i}>", depth + 1))
                except ValueError:
                    continue
            for k, v in (getattr(fn, "__defaults__", None) and enumerate(fn.__defaults__) or ()):
                todo.append((v, f"{path}.<default{k}>", depth + 1))
        elif hasattr(x, "__dict__") or hasattr(type(x), "__slots__"):
            attrs = dict(getattr(x, "__dict__", {}) or {})
            for s in getattr(type(x), "__slots__", ()) or ():
                if hasattr(x, s):
                    attrs[s] = getattr(x, s)
            todo += [(v, f"{path}.{k}", depth + 1) for k, v in attrs.items()]
    return out


_MUTABLE_CALLS = frozenset({"dict", "list", "set", "defaultdict", "OrderedDict", "Counter", "deque", "bytearray"})
_MUTABLE_NODES = (ast.Dict, ast.List, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)


def module_state_violations(source: str, filename: str = "<source>") -> list[SharedRef]:
    """Static check for implicit shared state in one module: module-level mutable containers (a dict at import time is a cache
    every caller shares) and functions that rebind module globals. Frozen forms (tuple, frozenset, MappingProxyType) pass."""
    tree = ast.parse(source, filename=filename)
    out: list[SharedRef] = []
    for node in tree.body:
        targets: list[str] = []
        value = None
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            targets, value = [node.target.id], node.value
        if value is None or not targets or targets == ["__all__"]:
            continue
        mutable = isinstance(value, _MUTABLE_NODES)
        if isinstance(value, ast.Call):
            f = value.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
            mutable = name in _MUTABLE_CALLS
        if mutable:
            out.append(SharedRef("module_state", f"{filename}:{node.lineno}", f"module-level mutable {', '.join(targets)}"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            out.append(SharedRef("module_state", f"{filename}:{node.lineno}", f"function rebinds globals {', '.join(node.names)}"))
    return out


def package_state_violations(paths: Iterable[str | Path]) -> list[SharedRef]:
    out: list[SharedRef] = []
    for p in paths:
        p = Path(p)
        out += module_state_violations(p.read_text(encoding="utf-8"), p.as_posix())
    return out


# ------------------------------------------------------------------------------------------------ a run's pair of namespaces
class NamespacePair:
    """Exactly one live and one research store per run, created together so that neither can be substituted for the other.
    `check()` is the section-29 separation audit: distinct stores, no shared containers, chains intact, no research-only kind in
    live state and no research-origin object in live state without a ticket."""

    def __init__(self, run_name: str, verifier: Callable[[CrossingTicket], bool] | None = None):
        self.run_name = run_name
        self.research = ResearchStore(f"{run_name}:research")
        self.live = LiveStore(f"{run_name}:live", verifier)
        if self.research.token == self.live.token:
            raise FirewallBreach("namespace tokens collide")

    def check(self) -> list[SharedRef]:
        out = shared_references(self.live, self.research)
        for name, st in (("live", self.live), ("research", self.research)):
            out += [SharedRef("store", name, e) for e in st.verify()]
        for o in self.live.objects():
            if o.kind in RESEARCH_ONLY:
                out.append(SharedRef("store", f"live:{o.object_id}", f"research-only kind {o.kind} in live state"))
            if o.origin.startswith(LiveStore.RESEARCH_ORIGIN) and o.object_id not in self.live._tickets:
                out.append(SharedRef("store", f"live:{o.object_id}", "research-origin object without a ticket"))
            if o.object_id in self.research and self.research.get(o.object_id).payload is o.payload:
                out.append(SharedRef("container", f"live:{o.object_id}", "payload object is the research store's own"))
        return out

    def require_separated(self) -> "NamespacePair":
        bad = self.check()
        if bad:
            raise SharedStateError("namespaces are not separated: " + "; ".join(str(b) for b in bad[:3]))
        return self

    def summary(self) -> dict:
        return {"run": self.run_name, "live": self.live.manifest(), "research": self.research.manifest(),
                "live_clock": str(self.live.clock), "tickets_used": len(self.live._used), "label": LABEL}


def iter_objects(stores: Sequence[NamespaceStore]) -> Iterator[tuple[str, InfoObject]]:
    for st in stores:
        for o in st.objects():
            yield st.name, o


def store_frame(store: NamespaceStore) -> pd.DataFrame:
    """One row per object (trusted-side report): kind, first usable date, evidence span, parents, origin."""
    rows = []
    for o in store.objects():
        try:
            k, who = store.effective_knowable(o.object_id)
            a, b = store.evidence_span(o.object_id)
        except FirewallBreach as e:
            k, who, a, b = None, f"unauditable: {e}", None, None
        rows.append({"object_id": o.object_id, "kind": str(o.kind), "knowable_at": k, "set_by": who, "evidence_from": a,
                     "evidence_through": b, "n_parents": len(o.all_parents()), "origin": o.origin})
    return pd.DataFrame(rows, columns=["object_id", "kind", "knowable_at", "set_by", "evidence_from", "evidence_through",
                                       "n_parents", "origin"])


PROVENANCE_FIELDS = ("code_hash", "data_hash", "config_hash", "experiment_id", "run_id", "outcomes_seen_through")


def provenance_census(store: NamespaceStore) -> pd.DataFrame:
    """Section 29 'every information object needs provenance': per object, which provenance fields are recorded, how deep its
    lineage runs, and whether its created/learned ordering is plausible (a record cannot be written before what it learned)."""
    rows = []
    for o in store.objects():
        p = o.provenance
        anc, missing, cycle = store.lineage(o.object_id)
        try:
            created = pd.Timestamp(p.created_real).date() if p.created_real else None
        except (ValueError, TypeError):
            created = None
        learned = optional_moment(p.learned_at, "learned_at") if p.learned_at else None
        row = {"object_id": o.object_id, "kind": str(o.kind), "n_ancestors": len(anc), "missing_parents": len(missing),
               "cycle": cycle, "created_parseable": created is not None,
               "written_before_learned": bool(created is not None and learned is not None and created < learned)}
        row.update({f"has_{f}": bool(getattr(p, f)) for f in PROVENANCE_FIELDS})
        rows.append(row)
    cols = ["object_id", "kind", "n_ancestors", "missing_parents", "cycle", "created_parseable", "written_before_learned"] +         [f"has_{f}" for f in PROVENANCE_FIELDS]
    return pd.DataFrame(rows, columns=cols)


def provenance_completeness(store: NamespaceStore) -> dict:
    """Shares of objects carrying each provenance field, plus the objects whose provenance is implausible. Empty store -> n=0 and
    no shares (never a 100% on nothing)."""
    df = provenance_census(store)
    if df.empty:
        return {"n": 0, "shares": {}, "implausible": []}
    shares = {c[4:]: float(df[c].mean()) for c in df.columns if c.startswith("has_")}
    bad = df.loc[df["written_before_learned"] | ~df["created_parseable"] | df["cycle"] | (df["missing_parents"] > 0), "object_id"]
    return {"n": int(len(df)), "shares": shares, "implausible": sorted(bad.tolist())}
