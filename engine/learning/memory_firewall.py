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
from typing import Any, Callable, Iterable, Mapping, Sequence

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
                   sealed_windows=tuple(_get(prov, "sealed_windows", ()) or ()), parents=tuple(_get(prov, "parents", ()) or ()))
    try:
        errs = tuple(p.check())
    except (ValueError, TypeError) as e:
        errs = (f"provenance dates unparseable: {e}",)
    learned = _d(p.learned_at)
    seen = _d(p.outcomes_seen_through) or learned
    return ItemView(str(kid), int(_get(item, "version", 0) or 0), True, p.created_real, learned, seen, p.code_hash, p.data_hash,
                    p.experiment_id, tuple(str(x) for x in p.parents), p.sealed_windows,
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


def identity_key_share(mapping: Any) -> tuple[float, int]:
    """Share of a payload's KEYS that are tickers or dates (or tuples containing them), and how many keys there were."""
    if not isinstance(mapping, Mapping) or not len(mapping):
        return 0.0, 0
    n = len(mapping)
    return sum(_looks_like_identity(k) for k in mapping) / n, n


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
    views.setdefault(v.knowledge_id, v)
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


def audit_store(items: Iterable, now, store: Iterable | None = None, sealed_windows: Sequence = (),
                policy: MemoryPolicy | None = None, env: MemoryEnvironment | None = None) -> MemoryAuditReport:
    """Audit every item at `now`. `store` (default: the items) supplies ancestry. Duplicate (id, version) pairs with
    different content are themselves a finding: history is immutable, so two records cannot share a version."""
    items = list(items)
    pool = list(store) if store is not None else items
    views: dict[str, ItemView] = {}
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
        old = views.get(v.knowledge_id)
        if old is None or v.version >= old.version:
            views[v.knowledge_id] = v
    exists = []
    for it in items:
        try:
            e = could_exist_at(it, now, views, env, sealed_windows, policy)
        except FirewallBreach as err:
            e = Existence("?", 0, False, (fail(L, "unidentifiable-item", "?", str(err)),))
        exists.append(e)
        out.extend(e.findings)
    return MemoryAuditReport(str(as_date(now)), tuple(exists), tuple(out))


# ---------------------------------------------------------------- as-of views (time travel)
def as_of_view(items: Iterable, now, policy: MemoryPolicy | None = None) -> list:
    """The memory as it stood at `now`: for every knowledge_id the newest version that could exist then. Versions
    created later are invisible; an id with no admissible version is absent. Retirement is a state of an admissible
    version, so retired items stay visible (retire is not delete)."""
    items = list(items)
    views = {}
    for it in items:
        v = view(it)
        cur = views.get(v.knowledge_id)
        if cur is None or v.version >= cur.version:
            views[v.knowledge_id] = v
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
    views = {view(i).knowledge_id: view(i) for i in items}
    seen_ok: set[str] = set()
    bad = []
    for n in order:
        for it in items:
            kid = view(it).knowledge_id
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
    views = {view(i).knowledge_id: view(i) for i in items}
    by = {}
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
        ex = could_exist_at(it, now, views, None, (), policy)
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
        except ImportError:
            pass
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
    rows = {}
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
