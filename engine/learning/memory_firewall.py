"""Memory contamination firewall (contract C62 section 28; checklist H01-H04, A13 support; sections 62 tests 7 and 12).

Every learned item must be able to answer: *could this item have existed at the exact decision timestamp?* The answer is
computed from its Provenance and its ancestry, never from what the item says about itself, and it is REJECT (never a
silent accept) whenever the answer cannot be shown. Checks, per item:
  * provenance present, complete, and internally consistent (created / learned / outcomes-seen ordering);
  * learned_at and the newest outcome the item has ever seen are strictly before `now` (an outcome that matures ON now
    is not yet known);
  * the whole ancestry is admissible - a child derived from a future parent is tainted through the parent, parents must
    exist in the store, a child cannot pre-date its parent, and the graph must be acyclic;
  * the data snapshot it could access does not reach `now`; the code that made it existed when it was written;
  * it never saw any sealed evaluation window, and was not produced by an experiment that ran an evaluation;
  * it is not an answer lookup table: no outcome labels in its payload, no ticker/date identity keys as its contexts.
Plus store-level tools: as-of views over versions, retrieval-log audits, tamper-evident snapshots, memory-bank frames
and `memory_future_invariance` - the strongest test: rebuild the memory with all data after `now` destroyed and demand a
bit-identical result.

Reuses engine.blind_gates.check_memory_bank_causality for DataFrame banks. IMPLEMENTED - NOT VALIDATED."""
from __future__ import annotations

import dataclasses
import datetime as dt
import re
from collections import Counter
from typing import Any, Callable, Iterable, Mapping, Sequence, cast

import numpy as np
import pandas as pd

from .core import FirewallBreach, Provenance, as_date, stable_hash
from .firewalls import Finding, LayerName, Severity, fail, info, overlap_days, parse_window, warn

L = LayerName.MEMORY
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_TICKER = re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$")
IDENTITY_DIMS = frozenset({"ticker", "symbol", "cusip", "isin", "permno", "date", "day", "week", "month", "year",
                           "episode", "episode_id", "timestamp", "as_of"})
LABEL_NAMES = re.compile(r"(^y$|label|target|forward|fwd|outcome|realized|realised|answer|truth|y_true|next_ret|ret_next)", re.I)


# ---------------------------------------------------------------- policy and registries
@dataclasses.dataclass(frozen=True)
class MemoryPolicy:
    require_code_hash: bool = True
    require_data_hash: bool = True
    require_experiment_id: bool = True
    require_seen_through: bool = False         # if False a missing outcomes_seen_through falls back to learned_at
    taint_through_parents: bool = True
    forbid_identity_contexts: bool = True
    forbid_label_payload: bool = True
    max_identity_key_share: float = 0.2        # share of payload keys that look like tickers/dates before it is a lookup table
    min_payload_keys_for_share: int = 8
    max_age_days: int | None = None            # None = no age limit (old knowledge is not contamination)


@dataclasses.dataclass(frozen=True)
class CodeRegistry:
    """code_hash -> ISO wall-clock time the code first existed. A record written before its code existed is forged."""
    first_seen: Mapping[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class DataRegistry:
    """data_hash -> newest REAL market date the snapshot contains. This is what the item's process could have read."""
    through: Mapping[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class MemoryEnvironment:
    code: CodeRegistry | None = None
    data: DataRegistry | None = None
    evaluation_experiments: frozenset = frozenset()    # experiment ids that ran an evaluation: nothing may be learned from them
    eval_window: Any = None                            # (start, end) of the window being decided; items must predate its start


# ---------------------------------------------------------------- adapting arbitrary items
def _get(obj, name, default=None):
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _d(x) -> dt.date | None:
    if x is None or x == "":
        return None
    try:
        return as_date(x)
    except (ValueError, TypeError):
        return None


@dataclasses.dataclass(frozen=True)
class ItemView:
    """The parts of an item the firewall reads, normalised. Built with `view`."""
    knowledge_id: str
    version: int
    has_provenance: bool
    created_real: str
    learned_at: dt.date | None
    seen_through: dt.date | None
    code_hash: str
    data_hash: str
    experiment_id: str
    parents: tuple[str, ...]
    sealed_windows: tuple
    contexts: Mapping[str, Any]
    anti_contexts: Mapping[str, Any]
    payload: Any
    provenance_errors: tuple[str, ...]


def bare_id(ref) -> str:
    """The canonical parent reference: the bare knowledge id.  Older writers recorded 'id@vN'; the version is dropped because
    ancestry is resolved as-of `now` (views_as_of picks the version that existed then), which is the safe reading."""
    return str(ref).split("@", 1)[0]


def canonical_parents(kid, parents) -> tuple[str, ...]:
    """Parents in canonical form: bare ids, in first-seen order, without duplicates and without a self-reference (a new version
    lists its predecessor, which is the same id; that is not an ancestor)."""
    return tuple(dict.fromkeys(b for b in (bare_id(p) for p in parents or ()) if b and b != str(kid)))


def view(item) -> ItemView:
    kid = _get(item, "knowledge_id")
    if kid is None or kid == "":
        raise FirewallBreach("memory item without a knowledge_id cannot be audited")
    prov = _get(item, "provenance")
    if prov is None:
        return ItemView(str(kid), int(_get(item, "version", 0) or 0), False, "", None, None, "", "", "", (), (),
                        _get(item, "contexts", {}) or {}, _get(item, "anti_contexts", {}) or {}, _get(item, "payload"), ())
    p = Provenance(created_real=str(_get(prov, "created_real", "") or ""), learned_at=str(_get(prov, "learned_at", "") or ""),
                   code_hash=str(_get(prov, "code_hash", "") or ""), data_hash=str(_get(prov, "data_hash", "") or ""),
                   config_hash=str(_get(prov, "config_hash", "") or ""), experiment_id=str(_get(prov, "experiment_id", "") or ""),
                   outcomes_seen_through=str(_get(prov, "outcomes_seen_through", "") or ""),
                   sealed_windows=tuple(_get(prov, "sealed_windows", ()) or ()),
                   parents=canonical_parents(kid, _get(prov, "parents", ()) or ()))
    try:
        errs = tuple(p.check())
    except (ValueError, TypeError) as e:
        errs = (f"provenance dates unparseable: {e}",)
    learned = _d(p.learned_at)
    seen = _d(p.outcomes_seen_through) or learned
    return ItemView(str(kid), int(_get(item, "version", 0) or 0), True, p.created_real, learned, seen, p.code_hash, p.data_hash,
                    p.experiment_id, p.parents, p.sealed_windows,
                    _get(item, "contexts", {}) or {}, _get(item, "anti_contexts", {}) or {}, _get(item, "payload"), errs)


def item_digest(item) -> str:
    """Content digest of an item: dataclasses hash by value, anything else by its normalised view + version."""
    if dataclasses.is_dataclass(item) and not isinstance(item, type):
        return stable_hash(item)
    v = view(item)
    return stable_hash([v.knowledge_id, v.version, str(v.learned_at), str(v.seen_through), v.code_hash, v.data_hash,
                        v.experiment_id, v.parents, dict(v.contexts)])


# ---------------------------------------------------------------- ancestry
@dataclasses.dataclass(frozen=True)
class Ancestry:
    effective_seen: dt.date | None             # newest outcome seen by the item OR any ancestor
    depth: int
    missing: tuple[str, ...]
    cycle: bool
    tainted_by: str | None                     # ancestor that pushes effective_seen furthest


def ancestry(kid: str, views: Mapping[str, ItemView], max_depth: int = 64) -> Ancestry:
    """Walk parents. The effective knowledge date of an item is the newest outcome ANY ancestor has seen."""
    missing: list[str] = []
    cyc = False
    best: list = [None, None]
    depth_seen = [0]

    def walk(k: str, stack: tuple[str, ...], depth: int):
        nonlocal cyc
        depth_seen[0] = max(depth_seen[0], depth)
        v = views.get(k)
        if v is None:
            missing.append(k)
            return
        if k in stack:
            cyc = True
            return
        if depth > max_depth:
            cyc = True                                       # a chain this deep is treated as a cycle: fail closed
            return
        cand = max([d for d in (v.seen_through, v.learned_at) if d is not None], default=None)
        if cand is not None and (best[0] is None or cand > best[0]):
            best[0], best[1] = cand, k
        for p in v.parents:
            walk(p, stack + (k,), depth + 1)

    walk(kid, (), 0)
    return Ancestry(best[0], depth_seen[0], tuple(sorted(set(missing))), cyc, best[1])


# ---------------------------------------------------------------- identity-lookup / label detection
def _looks_like_identity(v) -> bool:
    if isinstance(v, (tuple, list)):
        return any(_looks_like_identity(x) for x in v)
    if isinstance(v, (dt.date, pd.Timestamp)):
        return True
    if isinstance(v, str):
        return bool(_ISO_DATE.match(v)) or bool(_TICKER.match(v))
    return False


def identity_key_share(mapping: Any, depth: int = 0) -> tuple[float, int]:
    """Share of a payload's KEYS (at every nesting depth) that are tickers or dates or tuples containing them, and the
    number of keys inspected. A memorised table hides one level down ({'table': {(ticker, date): value}})."""
    n, hit = _count_identity_keys(mapping, depth)
    return (hit / n if n else 0.0), n


def _count_identity_keys(obj: Any, depth: int) -> tuple[int, int]:
    if depth > 6:
        return 0, 0
    if isinstance(obj, Mapping):
        n, hit = len(obj), sum(_looks_like_identity(k) for k in obj)
        for v in obj.values():
            a, b = _count_identity_keys(v, depth + 1)
            n, hit = n + a, hit + b
        return n, hit
    if isinstance(obj, (list, tuple)):
        n = hit = 0
        for v in obj[:200]:
            a, b = _count_identity_keys(v, depth + 1)
            n, hit = n + a, hit + b
        return n, hit
    return 0, 0


def label_fields(payload: Any, path: str = "payload", depth: int = 0) -> list[str]:
    """Names inside a payload that store outcomes (labels, forward returns, answers), searched recursively."""
    out: list[str] = []
    if depth > 6:
        return out
    if isinstance(payload, Mapping):
        for k, v in payload.items():
            if isinstance(k, str) and LABEL_NAMES.search(k):
                out.append(f"{path}.{k}")
            out.extend(label_fields(v, f"{path}.{k}", depth + 1))
    elif isinstance(payload, pd.DataFrame):
        out.extend(f"{path}[{c}]" for c in payload.columns if LABEL_NAMES.search(str(c)))
    elif isinstance(payload, pd.Series) and payload.name is not None and LABEL_NAMES.search(str(payload.name)):
        out.append(f"{path}<{payload.name}>")
    elif isinstance(payload, (list, tuple)):
        for i, v in enumerate(payload[:50]):
            out.extend(label_fields(v, f"{path}[{i}]", depth + 1))
    return out


def context_identity_keys(contexts: Mapping[str, Any]) -> list[str]:
    """Context dimensions that name an identity (ticker/date/year/episode) instead of a transferable situation."""
    return sorted(str(k) for k in contexts if str(k).lower() in IDENTITY_DIMS)


# ---------------------------------------------------------------- the per-item question
@dataclasses.dataclass(frozen=True)
class Existence:
    """Answer to 'could this item have existed at `now`?' with every reason behind a no."""
    knowledge_id: str
    version: int
    could_exist: bool
    findings: tuple[Finding, ...]
    effective_seen: dt.date | None = None

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(f.check for f in self.findings if f.is_fail)


def could_exist_at(item, now, views: Mapping[str, ItemView] | None = None, env: MemoryEnvironment | None = None,
                   sealed_windows: Sequence = (), policy: MemoryPolicy | None = None) -> Existence:
    """Judge one item at `now`. `views` is the store (by knowledge_id) used to resolve ancestry."""
    policy = policy or MemoryPolicy()
    env = env or MemoryEnvironment()
    v = view(item)
    views = dict(views or {})
    views[v.knowledge_id] = v                              # the record being judged, not a newer version of the same id
    nowd = as_date(now)
    out: list[Finding] = []
    kid = v.knowledge_id
    if not v.has_provenance:
        out.append(fail(L, "no-provenance", kid, "item has no provenance: it cannot show it could exist at the decision time"))
        return Existence(kid, v.version, False, tuple(out))
    for e in v.provenance_errors:
        out.append(fail(L, "provenance-invalid", kid, e))
    if v.learned_at is None:
        out.append(fail(L, "learned-at-missing", kid, "learned_at missing or unparseable"))
    elif v.learned_at >= nowd:
        out.append(fail(L, "learned-after-now", kid, f"learned_at {v.learned_at} is not before now {nowd}",
                        learned_at=str(v.learned_at)))
    if v.seen_through is not None and v.seen_through >= nowd:
        out.append(fail(L, "saw-future-outcomes", kid, f"item has seen outcomes through {v.seen_through}, not before now {nowd}",
                        seen_through=str(v.seen_through)))
    if policy.require_seen_through and not _get(_get(item, "provenance"), "outcomes_seen_through", ""):
        out.append(fail(L, "seen-through-missing", kid, "outcomes_seen_through not recorded"))
    for flag, val, name in ((policy.require_code_hash, v.code_hash, "code_hash"), (policy.require_data_hash, v.data_hash, "data_hash"),
                            (policy.require_experiment_id, v.experiment_id, "experiment_id")):
        if flag and not val:
            out.append(fail(L, f"{name}-missing", kid, f"provenance.{name} not recorded: what {name.split('_')[0]} made it is unknown"))
    if v.learned_at is not None and policy.max_age_days is not None and (nowd - v.learned_at).days > policy.max_age_days:
        out.append(warn(L, "too-old", kid, f"item is {(nowd - v.learned_at).days} days old (policy {policy.max_age_days})"))
    anc = None
    if policy.taint_through_parents:
        anc = ancestry(kid, views)
        for m in anc.missing:
            out.append(fail(L, "parent-missing", kid, f"ancestor {m!r} is not in the store: lineage cannot be audited"))
        if anc.cycle:
            out.append(fail(L, "lineage-cycle", kid, "the parent graph loops (or is implausibly deep)"))
        if anc.effective_seen is not None and anc.effective_seen >= nowd and (v.seen_through is None or anc.effective_seen > v.seen_through):
            out.append(fail(L, "tainted-by-parent", kid, f"ancestor {anc.tainted_by!r} saw outcomes through {anc.effective_seen}, "
                            f"not before now {nowd}", ancestor=anc.tainted_by))
        for p in v.parents:
            pv = views.get(p)
            if pv is not None and pv.learned_at is not None and v.learned_at is not None and pv.learned_at > v.learned_at:
                out.append(fail(L, "child-predates-parent", kid, f"learned {v.learned_at} but parent {p!r} was learned {pv.learned_at}"))
    windows = list(sealed_windows) + list(v.sealed_windows)
    if env.eval_window is not None:
        windows.append(env.eval_window)
    for w in windows:
        try:
            ws, we = parse_window(w)
        except (ValueError, KeyError, TypeError):
            out.append(fail(L, "sealed-window-unparseable", kid, f"window {w!r} cannot be parsed"))
            continue
        for label, day in (("learned_at", v.learned_at), ("outcomes_seen_through", v.seen_through),
                           ("ancestor outcomes", anc.effective_seen if anc else None)):
            if day is not None and ws.date() <= day <= we.date():
                out.append(fail(L, "saw-sealed-window", kid, f"{label} {day} falls inside sealed window {ws.date()}..{we.date()}"))
                break
        if env.eval_window is not None and w is env.eval_window and v.seen_through is not None and v.seen_through >= ws.date():
            out.append(fail(L, "not-before-eval-window", kid, f"saw outcomes through {v.seen_through}, at/after the evaluation start {ws.date()}"))
    if env.evaluation_experiments and v.experiment_id in env.evaluation_experiments:
        out.append(fail(L, "learned-from-evaluation", kid, f"produced by evaluation experiment {v.experiment_id!r}: "
                        "test results may not alter training state"))
    if env.data is not None and v.data_hash:
        thr = env.data.through.get(v.data_hash)
        if thr is None:
            out.append(fail(L, "data-snapshot-unknown", kid, f"data hash {v.data_hash!r} is not registered: what it could read is unknown"))
        else:
            td = _d(thr)
            if td is not None and td >= nowd:
                out.append(fail(L, "data-reaches-now", kid, f"data snapshot runs through {td}, not before now {nowd}"))
            elif td is not None and v.learned_at is not None and td > v.learned_at:
                out.append(warn(L, "data-beyond-learned", kid, f"snapshot runs through {td}, later than learned_at {v.learned_at}"))
    if env.code is not None and v.code_hash:
        fs = env.code.first_seen.get(v.code_hash)
        if fs is None:
            out.append(fail(L, "code-unregistered", kid, f"code hash {v.code_hash!r} never registered"))
        elif v.created_real and str(fs) > v.created_real:
            out.append(fail(L, "code-postdates-record", kid, f"record written {v.created_real} before its code existed ({fs})"))
    if policy.forbid_identity_contexts:
        keys = context_identity_keys(v.contexts) + context_identity_keys(v.anti_contexts)
        if keys:
            out.append(fail(L, "identity-context", kid, f"contexts keyed by identity dimension(s) {sorted(set(keys))}: "
                            "an answer lookup, not transferable knowledge", keys=sorted(set(keys))))
    if policy.forbid_label_payload and v.payload is not None:
        labs = label_fields(v.payload)
        if labs:
            out.append(fail(L, "outcome-labels-stored", kid, f"payload stores outcome-like fields {labs[:4]}", fields=labs[:8]))
        share, n = identity_key_share(v.payload)
        if n >= policy.min_payload_keys_for_share and share > policy.max_identity_key_share:
            out.append(fail(L, "answer-lookup-table", kid, f"{share:.0%} of {n} payload keys are tickers/dates: a memorised table",
                            share=float(share), n_keys=n))
    return Existence(kid, v.version, not any(f.is_fail for f in out), tuple(out), anc.effective_seen if anc else v.seen_through)


# ---------------------------------------------------------------- the store audit
@dataclasses.dataclass(frozen=True)
class MemoryAuditReport:
    now: str
    existences: tuple[Existence, ...]
    findings: tuple[Finding, ...]

    @property
    def admissible(self) -> tuple[str, ...]:
        return tuple(e.knowledge_id for e in self.existences if e.could_exist)

    @property
    def rejected(self) -> tuple[str, ...]:
        return tuple(e.knowledge_id for e in self.existences if not e.could_exist)

    @property
    def contamination_rate(self) -> float:
        return len(self.rejected) / len(self.existences) if self.existences else 0.0

    def reasons(self) -> Counter:
        return Counter(r for e in self.existences for r in e.reasons)

    def require_clean(self) -> "MemoryAuditReport":
        if self.rejected:
            raise FirewallBreach(f"memory contamination at now={self.now}: {len(self.rejected)}/{len(self.existences)} items rejected "
                                 f"({dict(self.reasons().most_common(3))})")
        return self

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"knowledge_id": e.knowledge_id, "version": e.version, "could_exist": e.could_exist,
                              "reasons": ",".join(e.reasons), "effective_seen": e.effective_seen} for e in self.existences],
                            columns=["knowledge_id", "version", "could_exist", "reasons", "effective_seen"])


def views_as_of(items: Iterable, now) -> dict[str, ItemView]:
    """id -> the version of that id that existed at `now` (newest with learned_at before now; if none, the oldest). Ancestry is
    resolved through these, so a parent's LATER version cannot taint or excuse a child."""
    by: dict[str, list[ItemView]] = {}
    for it in items:
        try:
            v = view(it)
        except FirewallBreach:
            continue
        by.setdefault(v.knowledge_id, []).append(v)
    nowd = as_date(now)
    out = {}
    for kid, vs in by.items():
        vs = sorted(vs, key=lambda x: x.version)
        past = [x for x in vs if x.learned_at is not None and x.learned_at < nowd]
        out[kid] = past[-1] if past else vs[0]
    return out


def audit_store(items: Iterable, now, store: Iterable | None = None, sealed_windows: Sequence = (),
                policy: MemoryPolicy | None = None, env: MemoryEnvironment | None = None) -> MemoryAuditReport:
    """Audit every item at `now`. `store` (default: the items) supplies ancestry. Duplicate (id, version) pairs with
    different content are themselves a finding: history is immutable, so two records cannot share a version."""
    items = list(items)
    pool = list(store) if store is not None else items
    views = views_as_of(pool, now)
    out: list[Finding] = []
    digests: dict[tuple, str] = {}
    for it in pool:
        try:
            v = view(it)
        except FirewallBreach as e:
            out.append(fail(L, "unidentifiable-item", "?", str(e)))
            continue
        key = (v.knowledge_id, v.version)
        dg = item_digest(it)
        if key in digests and digests[key] != dg:
            out.append(fail(L, "history-rewritten", v.knowledge_id, f"two different records share version {v.version}"))
        digests[key] = dg
    exists = []
    for it in items:
        try:
            ex = could_exist_at(it, now, views, env, sealed_windows, policy)
        except FirewallBreach as err:
            ex = Existence("?", 0, False, (fail(L, "unidentifiable-item", "?", str(err)),))
        exists.append(ex)
        out.extend(ex.findings)
    return MemoryAuditReport(str(as_date(now)), tuple(exists), tuple(out))


# ---------------------------------------------------------------- as-of views (time travel)
def as_of_view(items: Iterable, now, policy: MemoryPolicy | None = None) -> list:
    """The memory as it stood at `now`: for every knowledge_id the newest version that could exist then. Versions
    created later are invisible; an id with no admissible version is absent. Retirement is a state of an admissible
    version, so retired items stay visible (retire is not delete)."""
    items = list(items)
    views = views_as_of(items, now)
    best: dict[str, Any] = {}
    for it in items:
        v = view(it)
        if could_exist_at(it, now, views, None, (), policy).could_exist:
            if v.knowledge_id not in best or v.version >= view(best[v.knowledge_id]).version:
                best[v.knowledge_id] = it
    return [best[k] for k in sorted(best)]


def admissibility_monotone(items: Iterable, nows: Sequence, policy: MemoryPolicy | None = None) -> list[str]:
    """Property: once an item may exist at time t it may exist at every later time (knowledge does not un-happen).
    Returns violations; a non-empty list means the audit logic itself is inconsistent."""
    items = list(items)
    order = sorted(nows, key=as_date)
    seen_ok: set[str] = set()
    bad = []
    for n in order:
        views = views_as_of(items, n)
        for it in items:
            kid = f"{view(it).knowledge_id}@{view(it).version}"
            ok = could_exist_at(it, n, views, None, (), policy).could_exist
            if kid in seen_ok and not ok:
                bad.append(f"{kid} admissible earlier but rejected at {as_date(n)}")
            if ok:
                seen_ok.add(kid)
    return bad


# ---------------------------------------------------------------- retrieval log
def audit_retrieval_log(log: Iterable[Mapping], items: Iterable, policy: MemoryPolicy | None = None) -> list[Finding]:
    """Each entry {now, knowledge_id[, version]}: the item retrieved at that time must have been admissible then.
    Catches a retrieval that used a memory built later than the decision it informed."""
    items = list(items)
    by: dict[tuple[str, int | None], Any] = {}
    for i in items:
        v = view(i)
        by[(v.knowledge_id, v.version)] = i
        by.setdefault((v.knowledge_id, None), i)
    out: list[Finding] = []
    for n, e in enumerate(log):
        kid, ver, now = e.get("knowledge_id"), e.get("version"), e.get("now")
        if now is None or kid is None:
            out.append(fail(L, "retrieval-entry-malformed", f"entry {n}", "retrieval log entry needs now and knowledge_id"))
            continue
        it = by.get((kid, ver)) or by.get((kid, None))
        if it is None:
            out.append(fail(L, "retrieved-unknown-item", kid, f"entry {n}: retrieved item {kid!r} is not in the store"))
            continue
        ex = could_exist_at(it, now, views_as_of(items, now), None, (), policy)
        if not ex.could_exist:
            out.append(fail(L, "retrieved-before-existence", kid, f"entry {n}: retrieved at {as_date(now)} but {ex.reasons}"))
    return out


# ---------------------------------------------------------------- snapshots
@dataclasses.dataclass(frozen=True)
class MemorySnapshot:
    """Tamper-evident record of exactly which items (and versions) a run was allowed to use."""
    taken_at: str
    digests: Mapping[str, str]                 # "id@version" -> content digest

    @property
    def root(self) -> str:
        return stable_hash(sorted(self.digests.items()), 32)


def snapshot_memory(items: Iterable, taken_at) -> MemorySnapshot:
    dg = {}
    for it in items:
        v = view(it)
        dg[f"{v.knowledge_id}@{v.version}"] = item_digest(it)
    return MemorySnapshot(str(as_date(taken_at)), dg)


def verify_snapshot(snap: MemorySnapshot, items: Iterable, now) -> list[Finding]:
    """A snapshot is valid at `now` only if it was taken before it, still matches the items byte for byte, and no item
    was added or edited after it was taken."""
    out: list[Finding] = []
    if as_date(snap.taken_at) >= as_date(now):
        out.append(fail(L, "snapshot-from-future", snap.taken_at, f"snapshot taken {snap.taken_at}, not before now {as_date(now)}"))
    cur = snapshot_memory(items, snap.taken_at).digests
    for k in sorted(set(cur) - set(snap.digests)):
        out.append(fail(L, "item-added-after-snapshot", k, "item is not in the snapshot"))
    for k in sorted(set(snap.digests) - set(cur)):
        out.append(fail(L, "item-vanished", k, "snapshot item is missing now (history may not be deleted)"))
    for k in sorted(set(cur) & set(snap.digests)):
        if cur[k] != snap.digests[k]:
            out.append(fail(L, "item-edited-after-snapshot", k, "content changed since the snapshot (history is immutable)"))
    return out


# ---------------------------------------------------------------- DataFrame memory banks
def audit_bank_frame(bank: pd.DataFrame | None, now, learned_col: str = "learned_at", seen_col: str = "outcomes_seen_through") -> list[Finding]:
    """A memory bank kept as a table: one row per lesson. Every row needs a learned date strictly before `now`; a row whose
    window ended on/after `now` is a lesson from this run's own future (engine.blind_gates.check_memory_bank_causality)."""
    if bank is None:
        return [fail(L, "bank-missing", "bank", "no memory bank supplied")]
    if not len(bank):
        return [info(L, "bank-empty", "bank", "memory bank is empty")]
    out: list[Finding] = []
    nowd = pd.Timestamp(as_date(now))
    for col in (learned_col,):
        if col not in bank:
            return [fail(L, "bank-no-dates", "bank", f"bank has no {col} column, so its causality cannot be shown")]
    t = pd.to_datetime(bank[learned_col], errors="coerce")
    if t.isna().any():
        out.append(fail(L, "bank-null-dates", "bank", f"{int(t.isna().sum())} lessons without a parseable {learned_col}"))
    if (t >= nowd).any():
        out.append(fail(L, "bank-future-lessons", "bank", f"{int((t >= nowd).sum())} lessons learned on/after {nowd.date()}",
                        n=int((t >= nowd).sum())))
    if seen_col in bank:
        s = pd.to_datetime(bank[seen_col], errors="coerce")
        if (s >= nowd).any():
            out.append(fail(L, "bank-saw-future", "bank", f"{int((s >= nowd).sum())} lessons saw outcomes on/after {nowd.date()}"))
        if (s < t).any():
            out.append(fail(L, "bank-seen-before-learned", "bank", "outcomes_seen_through precedes learned_at"))
    if "real_end" in bank:
        try:
            from engine.blind_gates import check_memory_bank_causality
            for f in check_memory_bank_causality(bank, nowd):
                if f.severity == "fail":
                    out.append(fail(L, "bank-window-end", "bank", f.message))
        except ImportError as e:                 # fail CLOSED (C69 audit, 29 Sep): an unchecked bank must block, never pass
            out.append(fail(L, "bank-causality-check-missing", "bank", f"engine.blind_gates unavailable ({e}); window-end causality unchecked"))
    return out


# ---------------------------------------------------------------- the strongest test: rebuild without the future
def scramble_future(data, now, seed: int):
    """Copy of a date-indexed (or (date, ticker)-indexed) DataFrame/Series whose rows after `now` are permuted among
    themselves and perturbed, so any dependence on the future changes the result."""
    idx = data.index.get_level_values(0) if isinstance(data.index, pd.MultiIndex) else data.index
    fut = np.asarray(pd.DatetimeIndex(idx) > pd.Timestamp(as_date(now)))
    out = data.copy()
    if not fut.any():
        return out
    rng = np.random.default_rng(seed)
    vals = out.to_numpy(dtype=float, copy=True) if isinstance(out, pd.DataFrame) else out.to_numpy(dtype=float, copy=True)
    fv = vals[fut]
    fv = fv[rng.permutation(len(fv))] * (1.0 + rng.normal(0, 0.5, fv.shape)) + rng.normal(0, np.nanstd(fv) or 1.0, fv.shape)
    vals[fut] = fv
    if isinstance(out, pd.DataFrame):
        return pd.DataFrame(vals, index=out.index, columns=out.columns)
    return pd.Series(vals, index=out.index, name=out.name)


def memory_future_invariance(build: Callable[[Any], Any], data, now, seed: int = 0, trials: int = 3) -> list[Finding]:
    """`build(data)` returns the memory (items or any hashable artefact) learned from `data`. If the memory differs when
    everything after `now` is scrambled versus deleted, the builder read the future: the item could not exist at `now`."""
    idx = data.index.get_level_values(0) if isinstance(data.index, pd.MultiIndex) else data.index
    dates = pd.DatetimeIndex(idx)
    nowt = pd.Timestamp(as_date(now))
    if not (dates > nowt).any():
        return [fail(L, "no-future-to-scramble", "probe", "data holds nothing after now: the invariance probe would prove nothing")]
    truncated = data[np.asarray(dates <= nowt)]
    base = stable_hash(_artifact_key(build(truncated)))
    out: list[Finding] = []
    for k in range(trials):
        got = stable_hash(_artifact_key(build(scramble_future(data, now, seed * 1009 + k))))
        if got != base:
            out.append(fail(L, "memory-depends-on-future", "probe",
                            f"memory built with the post-{nowt.date()} data scrambled differs from the truncated build (trial {k})", trial=k))
    return out


def _artifact_key(obj):
    if isinstance(obj, (list, tuple)):
        return [_artifact_key(o) for o in obj]
    if isinstance(obj, pd.DataFrame):
        return [list(map(str, obj.columns)), obj.round(12).to_numpy(dtype=object).tolist()]
    if isinstance(obj, pd.Series):
        return [str(obj.name), obj.round(12).tolist()]
    if isinstance(obj, np.ndarray):
        return np.round(obj.astype(float), 12).tolist() if obj.dtype.kind in "fc" else obj.tolist()
    return obj


# ---------------------------------------------------------------- summaries
def contamination_by_layer(report: MemoryAuditReport) -> pd.DataFrame:
    """Reject reasons ranked, with how many distinct items each touched."""
    rows: dict[str, set[str]] = {}
    for e in report.existences:
        for r in set(e.reasons):
            rows.setdefault(r, set()).add(e.knowledge_id)
    df = pd.DataFrame([{"reason": r, "items": len(s)} for r, s in rows.items()], columns=["reason", "items"])
    return df.sort_values(["items", "reason"], ascending=[False, True]).reset_index(drop=True)


def age_profile(items: Iterable, now, bins: Sequence[int] = (0, 30, 90, 365, 1825, 10 ** 6)) -> pd.Series:
    """How old the admissible memory is (in days at `now`): a memory that is all recent, or all ancient, tells you what
    the learner can and cannot recall."""
    ages = []
    nowd = as_date(now)
    for it in items:
        v = view(it)
        if v.learned_at is not None and v.learned_at < nowd:
            ages.append((nowd - v.learned_at).days)
    if not ages:
        return pd.Series(dtype=int)
    return pd.cut(pd.Series(ages), bins=list(bins), right=False, include_lowest=True).value_counts().sort_index()


def sealed_window_exposure(items: Iterable, windows: Sequence) -> pd.DataFrame:
    """For each sealed window, which items saw outcomes inside it or later (candidates for contamination when the window
    is opened). Used before opening a window to prove nothing already learned from it."""
    rows = []
    for w in windows:
        ws, we = parse_window(w)
        for it in items:
            v = view(it)
            day = v.seen_through or v.learned_at
            if day is not None and day >= ws.date():
                rows.append({"window": f"{ws.date()}..{we.date()}", "knowledge_id": v.knowledge_id, "seen_through": day,
                             "inside": day <= we.date()})
    return pd.DataFrame(rows, columns=["window", "knowledge_id", "seen_through", "inside"])


def windows_overlapping(items: Iterable, window) -> list[str]:
    """Items whose own sealed windows overlap `window`: they were created while it was supposed to be sealed and unseen."""
    hits = []
    for it in items:
        v = view(it)
        if any(overlap_days(w, window) > 0 for w in v.sealed_windows):
            hits.append(v.knowledge_id)
    return sorted(hits)


# ---------------------------------------------------------------- hidden label storage (obfuscated names)
def payload_numbers(payload: Any, limit: int = 20000, depth: int = 0) -> np.ndarray:
    """Every finite number found in a payload (nested dicts/lists/arrays/frames), flattened, capped at `limit`."""
    vals: list[np.ndarray] = []
    total = [0]

    def take(a):
        a = np.asarray(a, dtype=float).ravel()
        a = a[np.isfinite(a)]
        room = limit - total[0]
        if room > 0 and len(a):
            vals.append(a[:room])
            total[0] += min(len(a), room)

    def walk(o, d):
        if d > 6 or total[0] >= limit:
            return
        if isinstance(o, (bool, np.bool_)):
            return
        if isinstance(o, (int, float, np.integer, np.floating)):
            take([o])
        elif isinstance(o, (np.ndarray, pd.Series)):
            if np.asarray(o).dtype.kind in "fiu":
                take(np.asarray(o))
        elif isinstance(o, pd.DataFrame):
            take(o.select_dtypes(include=[np.number]).to_numpy())
        elif isinstance(o, Mapping):
            for v in o.values():
                walk(v, d + 1)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v, d + 1)

    walk(payload, depth)
    return np.concatenate(vals) if vals else np.array([], dtype=float)


def hidden_label_overlap(payload: Any, outcomes: Sequence[float] | pd.Series, decimals: int = 6, min_numbers: int = 20,
                         max_share: float = 0.3) -> Finding | None:
    """Catch labels stored under an innocent name: the share of the payload's numbers that appear EXACTLY among the
    outcome values. Honest statistics (means, thresholds, weights) almost never coincide with raw outcomes to six
    decimals; a stored answer column does. Returns a FAIL finding or None."""
    nums = payload_numbers(payload)
    y = np.asarray(outcomes, dtype=float)
    y = y[np.isfinite(y)]
    if len(nums) < min_numbers or len(y) < min_numbers:
        return None
    pool = set(np.round(y, decimals).tolist())
    hit = float(np.mean([round(float(v), decimals) in pool for v in nums]))
    if hit > max_share:
        return fail(L, "hidden-label-storage", "payload", f"{hit:.0%} of {len(nums)} payload numbers equal raw outcome values exactly",
                    share=hit, n_numbers=int(len(nums)))
    return None


# ---------------------------------------------------------------- enforcement: a memory that refuses contaminated writes
@dataclasses.dataclass(frozen=True)
class Tombstone:
    """Retirement marker (section 13: retire is not delete). The item itself stays; this says it is no longer used."""
    knowledge_id: str
    retired_at: str
    reason: str


class FirewalledMemory:
    """An append-only item store that enforces the memory firewall on the way in and on the way out.

    * `add(item, now)` rejects (FirewallBreach) any item that could not exist at `now`, any second record for an existing
      (id, version), and any record whose parents are not already stored.
    * `retrieve(now)` returns only what could exist at `now` (newest admissible version per id, not retired at `now`) and
      logs every retrieval so `audit_retrieval_log` can re-check it later.
    * Nothing is ever removed: rejected writes go to `quarantine`, retirements are tombstones."""

    def __init__(self, policy: MemoryPolicy | None = None, env: MemoryEnvironment | None = None):
        self.policy, self.env = policy, env
        self._items: list = []
        self._keys: dict[tuple, str] = {}
        self.tombstones: list[Tombstone] = []
        self.quarantine: list[tuple[Any, str, tuple[str, ...]]] = []          # (item, rejected_at, reasons)
        self.log: list[dict] = []

    def __len__(self):
        return len(self._items)

    def _views(self, now=None) -> dict[str, ItemView]:
        return views_as_of(self._items, now if now is not None else "9999-12-31")

    def add(self, item, now) -> Existence:
        v = view(item)
        key = (v.knowledge_id, v.version)
        if key in self._keys:
            self.quarantine.append((item, str(as_date(now)), ("version-exists",)))
            raise FirewallBreach(f"{key} already stored: history is immutable, write a new version")
        views = self._views(now)
        for p in v.parents:
            if p not in views:
                self.quarantine.append((item, str(as_date(now)), ("parent-missing",)))
                raise FirewallBreach(f"{v.knowledge_id}: parent {p!r} is not in the store")
        ex = could_exist_at(item, now, views, self.env, (), self.policy)
        if not ex.could_exist:
            self.quarantine.append((item, str(as_date(now)), ex.reasons))
            raise FirewallBreach(f"{v.knowledge_id} rejected at now={as_date(now)}: {', '.join(ex.reasons)}")
        self._items.append(item)
        self._keys[key] = item_digest(item)
        return ex

    def retire(self, knowledge_id: str, now, reason: str) -> Tombstone:
        if knowledge_id not in self._views():
            raise FirewallBreach(f"cannot retire unknown item {knowledge_id!r}")
        if not reason:
            raise FirewallBreach("retirement needs a reason (an unexplained retirement is a lost lesson)")
        t = Tombstone(knowledge_id, str(as_date(now)), reason)
        self.tombstones.append(t)
        return t

    def is_retired(self, knowledge_id: str, now) -> bool:
        return any(t.knowledge_id == knowledge_id and as_date(t.retired_at) < as_date(now) for t in self.tombstones)

    def retrieve(self, now, predicate: Callable[[Any], bool] | None = None, include_retired: bool = False) -> list:
        got = []
        for it in as_of_view(self._items, now, self.policy):
            kid = view(it).knowledge_id
            if not include_retired and self.is_retired(kid, now):
                continue
            if predicate is None or predicate(it):
                got.append(it)
                self.log.append({"now": str(as_date(now)), "knowledge_id": kid, "version": view(it).version})
        return got

    def audit(self, now) -> MemoryAuditReport:
        return audit_store(self._items, now, policy=self.policy, env=self.env)

    def snapshot(self, taken_at) -> MemorySnapshot:
        return snapshot_memory(self._items, taken_at)

    def verify_log(self) -> list[Finding]:
        return audit_retrieval_log(self.log, self._items, self.policy)

    def requalify(self, now) -> list:
        """Re-audit quarantined items: an item rejected only because it was too new may be admissible once `now` has
        moved past its evidence. Returns items that now pass; they are added, and removed from quarantine."""
        back, keep = [], []
        for item, when, reasons in self.quarantine:
            if "version-exists" in reasons or "parent-missing" in reasons:
                keep.append((item, when, reasons))
                continue
            try:
                self.add(item, now)
                back.append(item)
            except FirewallBreach:
                keep.append((item, when, reasons))
        self.quarantine = keep
        return back


# ---------------------------------------------------------------- timelines and reports
def admissibility_timeline(items: Iterable, nows: Sequence, policy: MemoryPolicy | None = None) -> pd.DataFrame:
    """Rows = decision dates, columns = items, cells = could-exist. Reading it shows exactly when each lesson becomes usable."""
    items = list(items)
    order = sorted(nows, key=as_date)
    data: dict[str, list[bool]] = {}
    for n in order:
        vs = views_as_of(items, n)
        for i in items:
            data.setdefault(f"{view(i).knowledge_id}@{view(i).version}", []).append(could_exist_at(i, n, vs, None, (), policy).could_exist)
    return pd.DataFrame(data, index=pd.Index([pd.Timestamp(as_date(n)) for n in order], name="now"))


def taint_sources(report: MemoryAuditReport, items: Iterable) -> pd.Series:
    """Which ancestor taints the most descendants: fix the root, not each child."""
    views = {view(i).knowledge_id: view(i) for i in items}
    counts: Counter = Counter()
    for e in report.existences:
        if "tainted-by-parent" in e.reasons:
            anc = ancestry(e.knowledge_id, views)
            if anc.tainted_by and anc.tainted_by != e.knowledge_id:
                counts[anc.tainted_by] += 1
    return pd.Series(dict(counts), dtype=int).sort_values(ascending=False)


def existence_markdown(report: MemoryAuditReport, limit: int = 20) -> str:
    """Reviewer-facing account of an audit: totals, reject reasons, and the first rejected items with their reasons."""
    n, bad = len(report.existences), len(report.rejected)
    lines = [f"# Memory firewall audit at now={report.now}",
             f"{n} items audited, {n - bad} admissible, {bad} rejected (contamination rate {report.contamination_rate:.1%})", ""]
    for reason, cnt in report.reasons().most_common():
        lines.append(f"- {reason}: {cnt} item(s)")
    shown = 0
    for e in report.existences:
        if not e.could_exist and shown < limit:
            lines.append(f"- REJECT {e.knowledge_id} v{e.version}: " + "; ".join(f.message for f in e.findings if f.is_fail))
            shown += 1
    lines.append("")
    lines.append("This audit is IMPLEMENTED - NOT VALIDATED.")
    return "\n".join(lines)


def planted_contamination_suite(make_clean: Callable[[], Any], now) -> dict[str, Existence]:
    """Self-test: derive one item per known contamination kind from a clean item factory and return each verdict.
    A clean item must be admitted; every planted variant must be rejected (used by tests and startup self-check)."""
    clean = make_clean()
    prov = _get(clean, "provenance")
    nd = as_date(now)
    fut = str(nd + dt.timedelta(days=30))
    variants = {
        "clean": clean,
        "learned_in_future": _with_prov(clean, prov, learned_at=fut, outcomes_seen_through=fut),
        "saw_future_outcomes": _with_prov(clean, prov, outcomes_seen_through=fut),
        "no_code_hash": _with_prov(clean, prov, code_hash=""),
        "no_experiment": _with_prov(clean, prov, experiment_id=""),
    }
    return {k: could_exist_at(v, now) for k, v in variants.items()}


def _with_prov(item, prov, **changes):
    new = dataclasses.replace(prov, **changes) if dataclasses.is_dataclass(prov) else {**dict(prov), **changes}
    if dataclasses.is_dataclass(item):
        return dataclasses.replace(item, provenance=new)
    out = dict(item)
    out["provenance"] = new
    return out


# ---------------------------------------------------------------- leak channel 4: learned state trained on the future
@dataclasses.dataclass(frozen=True)
class TrainingWindow:
    """A window a piece of learned state was fitted on, in REAL (referee-side) dates."""
    window_id: str
    real_start: str
    real_end: str

    def bounds(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return pd.Timestamp(self.real_start).normalize(), pd.Timestamp(self.real_end).normalize()


@dataclasses.dataclass(frozen=True)
class LearnedState:
    """State that shapes later decisions without being a KnowledgeObject: a search basis (cfg + meta), tuned defaults, a
    memory bank. Leak channel 4 (state/research/leak_audit): such state trained on windows from the played window's
    future, or on the played window itself, is a leak that no per-item provenance would show."""
    version: str
    kind: str                                  # basis_cfg | basis_meta | defaults | memory_bank
    trained_on: tuple[TrainingWindow, ...] = ()
    tuned_years: tuple[int, ...] = ()          # calendar years whose OUTCOMES chose these defaults


def audit_state_lineage(states: Sequence[LearnedState], played: Sequence[Mapping], allow_same_window: bool = False) -> list[Finding]:
    """`played` = [{id, real_start, version}]: which state version each window was played with. A window's state may be
    trained only on windows that ENDED before its real start; a rerun of the same real window may not use state trained on
    its own first run unless the caller explicitly allows it (C54 vs C56). Built on engine.leak_audit.BasisLineage."""
    from engine.leak_audit import BasisLineage, TrainedOn
    out: list[Finding] = []
    lin = BasisLineage()
    by_version = {}
    for st in states:
        try:
            lin.register(st.version, st.kind, {"tuned_years": st.tuned_years},
                         [TrainedOn(w.window_id, *w.bounds()) for w in st.trained_on])
            by_version[st.version] = st
        except (ValueError, TypeError) as e:
            out.append(fail(L, "state-window-unparseable", st.version, f"training window dates cannot be read: {e}"))
    if not states:
        out.append(info(L, "no-learned-state", "state", "no learned state supplied"))
    for p in played:
        if p.get("version") not in by_version:
            out.append(fail(L, "state-version-unknown", str(p.get("id")), f"played with state version {p.get('version')!r} that has no lineage record"))
    for v in lin.violations([dict(p) for p in played if p.get("version") in by_version], allow_same_window):
        st = by_version[v["version"]]
        start = next(pd.Timestamp(p["real_start"]) for p in played if p["id"] == v["window"])
        for wid in v["trained_on_late"]:
            w = next(x for x in st.trained_on if x.window_id == wid)
            same = w.bounds()[0] == start.normalize()
            out.append(fail(L, "state-trained-on-same-window" if same else "state-trained-on-future-window", v["window"],
                            f"state {st.version} was trained on window {wid} ({w.real_start}..{w.real_end}), "
                            + ("the very window being played" if same else f"which had not ended before {start.date()}"),
                            state=st.version, trained_on=wid))
    starts = [pd.Timestamp(p["real_start"]) for p in played if p.get("real_start") is not None]
    for st in states:
        if st.tuned_years and starts:
            from engine.leak_audit import defaults_contamination
            res = defaults_contamination(st.tuned_years, starts)
            if res["contaminated_share"] > 0:
                hit = [s for s in starts if any(y in set(st.tuned_years) for y in range(s.year, (s + pd.DateOffset(months=12)).year + 1))]
                out.append(fail(L, "defaults-tuned-on-window", st.version,
                                f"defaults were tuned on outcomes of years {sorted(st.tuned_years)[:6]}; {len(hit)} of {len(starts)} played windows "
                                "touch those years (in-sample for their own defaults)", share=float(res["contaminated_share"])))
    return out


def eligible_state(states: Sequence[LearnedState], real_start, allow_same_window: bool = False) -> LearnedState | None:
    """The newest state whose training windows ALL ended before `real_start`; None means 'use the untrained defaults'."""
    from engine.leak_audit import BasisLineage, TrainedOn
    lin = BasisLineage()
    for st in states:
        lin.register(st.version, st.kind, {}, [TrainedOn(w.window_id, *w.bounds()) for w in st.trained_on])
    rec = lin.basis_for(real_start, allow_same_window)
    return next((s for s in states if rec is not None and s.version == rec["version"]), None)


def future_training_share(states: Sequence[LearnedState], played: Sequence[Mapping]) -> dict:
    """Measure of channel 4: over played windows, the mean share of their state's training windows that end on/after the
    window's start, and the share of played windows touched by at least one such window."""
    by = {s.version: s for s in states}
    shares, touched = [], 0
    for p in played:
        st = by.get(cast(str, p.get("version")))
        if st is None or not st.trained_on:
            continue
        start = pd.Timestamp(p["real_start"])
        late = sum(w.bounds()[1] >= start for w in st.trained_on)
        shares.append(late / len(st.trained_on))
        touched += late > 0
    if not shares:
        return {"n_played": 0, "mean_future_share": float("nan"), "share_of_windows_touched": float("nan")}
    return {"n_played": len(shares), "mean_future_share": float(np.mean(shares)), "share_of_windows_touched": touched / len(shares)}


# ---------------------------------------------------------------- enforcing causality on a memory bank table
def causal_bank(bank: pd.DataFrame, window_start, end_col: str = "real_end") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a lesson table into (usable, excluded): a lesson is usable for a window only if its own window ENDED before
    the window's start. Excluded rows are returned, not dropped silently, so the caller can log what was refused."""
    if bank is None or not len(bank):
        return (bank if bank is not None else pd.DataFrame()), pd.DataFrame(columns=getattr(bank, "columns", []))
    if end_col not in bank:
        raise FirewallBreach(f"bank has no {end_col} column: its causality cannot be shown")
    ends = pd.to_datetime(bank[end_col], errors="coerce")
    ok = (ends < pd.Timestamp(window_start)) & ends.notna()
    return bank[ok], bank[~ok]


def provenance_completeness(items: Iterable) -> pd.DataFrame:
    """One row per item: which provenance fields are present. Sorting by `n_missing` shows where the audit is blind."""
    rows = []
    for it in items:
        v = view(it)
        present = {"provenance": v.has_provenance, "learned_at": v.learned_at is not None, "seen_through": v.seen_through is not None,
                   "code_hash": bool(v.code_hash), "data_hash": bool(v.data_hash), "experiment_id": bool(v.experiment_id),
                   "sealed_windows": bool(v.sealed_windows)}
        rows.append({"knowledge_id": v.knowledge_id, **present, "n_missing": sum(not b for b in present.values())})
    return pd.DataFrame(rows).sort_values("n_missing", ascending=False).reset_index(drop=True) if rows else pd.DataFrame(
        columns=["knowledge_id", "n_missing"])


def snapshot_diff(a: MemorySnapshot, b: MemorySnapshot) -> dict[str, list[str]]:
    """What changed between two snapshots: added, removed and edited items (edited = same id@version, different content:
    a violation of immutable history)."""
    added = sorted(set(b.digests) - set(a.digests))
    removed = sorted(set(a.digests) - set(b.digests))
    edited = sorted(k for k in set(a.digests) & set(b.digests) if a.digests[k] != b.digests[k])
    return {"added": added, "removed": removed, "edited": edited}


def plant_future_item(make_clean: Callable[[], Any], now, days_ahead: int = 30):
    """A copy of a clean item whose evidence matures `days_ahead` days AFTER now: the canonical planted memory leak
    (section 62 test 7/11). could_exist_at must reject it."""
    clean = make_clean()
    prov = _get(clean, "provenance")
    fut = str(as_date(now) + dt.timedelta(days=days_ahead))
    return _with_prov(clean, prov, learned_at=fut, outcomes_seen_through=fut)


def plant_hidden_answer_table(n: int = 40, seed: int = 0) -> dict:
    """A payload that is an answer lookup: {(ticker, date): forward return}. `label_fields` cannot see it by name (the values
    are innocently keyed), `identity_key_share` and `hidden_label_overlap` can."""
    rng = np.random.default_rng(seed)
    tickers = [f"T{chr(65 + i % 26)}{chr(65 + (i // 26) % 26)}" for i in range(n)]
    dates = pd.bdate_range("2019-01-02", periods=n)
    return {"table": {(t, str(d.date())): float(v) for t, d, v in zip(tickers, dates, rng.normal(0, 0.05, n))}}


# ---------------------------------------------------------------- retirement is not deletion (section 13)
def audit_retirements(items: Iterable, tombstones: Iterable[Tombstone], now, retrieval_log: Iterable[Mapping] = ()) -> list[Finding]:
    """A retired item stays in the store; the tombstone says why and when. Checks: every tombstone names an existing item,
    carries a reason, is dated before `now`; no item was DELETED (a tombstone for an id that has no record); and no retired item
    was retrieved after its retirement date."""
    items = list(items)
    ids = {view(i).knowledge_id for i in items}
    nowd = as_date(now)
    out: list[Finding] = []
    seen: dict[str, str] = {}
    for t in tombstones:
        if t.knowledge_id not in ids:
            out.append(fail(L, "tombstone-without-item", t.knowledge_id, "an item was retired but no record of it exists: it was deleted"))
        if not t.reason:
            out.append(fail(L, "tombstone-without-reason", t.knowledge_id, "retirement carries no reason (a lost lesson)"))
        d = _d(t.retired_at)
        if d is None or d >= nowd:
            out.append(fail(L, "tombstone-not-in-past", t.knowledge_id, f"retirement dated {t.retired_at!r} is not before now {nowd}"))
        if t.knowledge_id in seen and seen[t.knowledge_id] != t.retired_at:
            out.append(warn(L, "retired-twice", t.knowledge_id, f"retired on {seen[t.knowledge_id]} and again on {t.retired_at}"))
        seen.setdefault(t.knowledge_id, t.retired_at)
    for e in retrieval_log:
        kid, when = e.get("knowledge_id"), _d(e.get("now"))
        if kid in seen and when is not None and _d(seen[kid]) is not None and when > cast(dt.date, _d(seen[kid])):
            out.append(fail(L, "retired-item-retrieved", str(kid), f"retrieved on {when}, after its retirement on {seen[kid]}"))
    return out


def contamination_score(report: MemoryAuditReport, weights: Mapping[str, float] | None = None) -> float:
    """Severity-weighted contamination in [0, 1]: a future-outcome leak counts more than a missing hash. 0 = clean store."""
    w = {"saw-future-outcomes": 1.0, "tainted-by-parent": 1.0, "learned-after-now": 1.0, "saw-sealed-window": 1.0,
         "learned-from-evaluation": 1.0, "outcome-labels-stored": 0.9, "answer-lookup-table": 0.9, "data-reaches-now": 0.9}
    w.update(weights or {})
    if not report.existences:
        return 0.0
    per_item = []
    for e in report.existences:
        per_item.append(max((w.get(r, 0.4) for r in e.reasons), default=0.0))
    return float(np.mean(per_item))


def merge_stores(a: Sequence, b: Sequence) -> list:
    """Union of two item lists that respects immutable history: an (id, version) present in both must have the same content
    digest, otherwise the merge is refused (someone rewrote history in one of the stores)."""
    merged: dict[tuple, Any] = {}
    digests: dict[tuple, str] = {}
    for it in list(a) + list(b):
        v = view(it)
        key = (v.knowledge_id, v.version)
        dg = item_digest(it)
        if key in digests:
            if digests[key] != dg:
                raise FirewallBreach(f"cannot merge: {key} has different content in the two stores (history rewritten)")
            continue
        digests[key] = dg
        merged[key] = it
    return [merged[k] for k in sorted(merged)]


# ---------------------------------------------------------------- lineage graph
def lineage_edges(items: Iterable) -> dict[str, tuple[str, ...]]:
    """child id -> parent ids, for the newest version of each id."""
    return {v.knowledge_id: v.parents for v in views_as_of(items, "9999-12-31").values()}


def descendants_of(items: Iterable, roots: Iterable[str]) -> set[str]:
    """Every item that derives (directly or transitively) from any of `roots`. When a root is rejected, its descendants are
    suspect even if their own provenance looks clean: they inherited what the root learned."""
    edges = lineage_edges(items)
    children: dict[str, set[str]] = {}
    for c, ps in edges.items():
        for p in ps:
            children.setdefault(p, set()).add(c)
    out: set[str] = set()
    stack = list(roots)
    while stack:
        cur = stack.pop()
        for ch in children.get(cur, ()):
            if ch not in out:
                out.add(ch)
                stack.append(ch)
    return out


def quarantine_closure(report: MemoryAuditReport, items: Iterable) -> set[str]:
    """Rejected items plus all their descendants: the set that must not be retrieved until the root is re-qualified."""
    rejected = set(report.rejected)
    return rejected | descendants_of(items, rejected)


def lineage_depth(items: Iterable) -> dict[str, int]:
    """Longest ancestor chain per item (0 for a root). A deep chain is a long path along which contamination can travel."""
    edges = lineage_edges(items)
    memo: dict[str, int] = {}

    def depth(k: str, stack: frozenset) -> int:
        if k in memo:
            return memo[k]
        if k in stack or k not in edges:
            return 0
        d = 0 if not edges[k] else 1 + max(depth(p, stack | {k}) for p in edges[k])
        memo[k] = d
        return d
    return {k: depth(k, frozenset()) for k in edges}


STRICT_POLICY = MemoryPolicy(require_seen_through=True, max_age_days=None)
LENIENT_POLICY = MemoryPolicy(require_code_hash=False, require_data_hash=False, require_experiment_id=False, forbid_identity_contexts=False)


# ---------------------------------------------------------------- human-readable answers
def explain_existence(ex: Existence) -> str:
    """The answer to 'could this item have existed at the decision time?' in one paragraph, with the reasons behind a no."""
    if ex.could_exist:
        seen = f" (newest outcome seen, including ancestors: {ex.effective_seen})" if ex.effective_seen else ""
        warns = [f.message for f in ex.findings if f.severity == Severity.WARN]
        return f"{ex.knowledge_id} v{ex.version}: YES, it could have existed{seen}." + (f" Warnings: {'; '.join(warns)}." if warns else "")
    return f"{ex.knowledge_id} v{ex.version}: NO - REJECT. " + " ".join(f"[{f.check}] {f.message}." for f in ex.findings if f.is_fail)


def validate_policy(policy: MemoryPolicy) -> list[str]:
    """A policy that switches off provenance requirements is legal for research replays and must be visible: returns the list of
    protections it disables so a report can print them next to any result."""
    off = []
    for name, label in (("require_code_hash", "code hash"), ("require_data_hash", "data hash"), ("require_experiment_id", "experiment id"),
                        ("taint_through_parents", "ancestry taint"), ("forbid_identity_contexts", "identity-context ban"),
                        ("forbid_label_payload", "stored-label ban")):
        if not getattr(policy, name):
            off.append(label)
    if not 0 < policy.max_identity_key_share <= 1:
        off.append("identity-key share out of range")
    return off


def earliest_use_date(item, store: Iterable | None = None) -> dt.date | None:
    """The first decision date on which this item may be used: one day after the newest outcome it or any ancestor has seen.
    None if its provenance is too incomplete to say (it may not be used at all)."""
    v = view(item)
    if not v.has_provenance or v.learned_at is None:
        return None
    pool = {v.knowledge_id: v, **views_as_of(list(store) if store is not None else [item], "9999-12-31")}
    pool[v.knowledge_id] = v
    anc = ancestry(v.knowledge_id, pool)
    if anc.missing or anc.cycle or anc.effective_seen is None:
        return None
    return anc.effective_seen + dt.timedelta(days=1)


def usable_from(items: Iterable) -> pd.Series:
    """earliest_use_date for every item (NaT when it can never be shown clean): when each lesson comes into play."""
    items = list(items)
    return pd.Series({view(i).knowledge_id: pd.Timestamp(d) if (d := earliest_use_date(i, items)) else pd.NaT for i in items}, dtype="datetime64[ns]")
