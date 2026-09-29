"""Bible Phase 0.2 + Phase 30: read-side of the experiment registry and the experiment memory.

engine.improve.log_experiment appends to state/experiments.jsonl and never edits. This module queries that log
(filter / sort / compare), audits it against the Phase 0.2 field list (missing provenance, duplicate ids, orphans,
non-reproducible repeats) and keeps the experiment memory: what was tried, so it is not tried again (Phase 30).
It tolerates a damaged log: unparseable lines are counted and reported, never silently dropped."""
import hashlib
import json
import math
from pathlib import Path

from . import config as K

REQUIRED = ("experiment_id", "timestamp", "git_commit", "canon_hash", "blueprint_version", "config_hash",
            "data_snapshot", "seed", "window_ids", "model_params", "train_range", "validation_range", "test_range",
            "metrics", "gates", "outcome", "reason")
# field -> names it may be stored under (log_experiment writes t / canon_sha; later writers use the long names)
ALIASES = {"timestamp": ("timestamp", "t"), "canon_hash": ("canon_hash", "canon_sha"),
           "window_ids": ("window_ids", "windows"), "model_params": ("model_params", "params"),
           "validation_range": ("validation_range", "val_range"), "reason": ("reason", "adoption_reason", "why")}
OUTCOMES = ("adopt", "reject", "continue_testing")
# Phase 0.2: every experiment must be reproducible; these identify "the same experiment"
REPRO_KEY = ("config_hash", "data_snapshot", "seed", "code_hash")


def _get(rec, field):
    for n in ALIASES.get(field, (field,)):
        v = rec.get(n)
        if v is not None:
            return v
    return None


def _empty(v):
    return v is None or (isinstance(v, (str, list, dict, tuple)) and len(v) == 0)


def _num(v):
    if isinstance(v, bool):
        return None
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _dig(rec, path):
    """'metrics.sharpe' -> nested lookup; a flat key of that exact name or an aliased top-level field also works."""
    if path in rec:
        return rec[path]
    cur = rec
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return _get(rec, path) if "." not in path else None
    return cur


class Registry:
    def __init__(self, path=None):
        self.path = Path(path) if path else K.STATE / "experiments.jsonl"
        self.records, self.bad_lines = [], []
        self.load()

    def load(self):
        self.records, self.bad_lines = [], []
        if not self.path.exists():
            return self
        for i, line in enumerate(self.path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                if not isinstance(r, dict):
                    raise ValueError("not an object")
                r["_line"] = i
                self.records.append(r)
            except ValueError as e:
                self.bad_lines.append((i, str(e)))
        return self

    def __len__(self):
        return len(self.records)

    def append(self, rec):
        """Append-only write with a uniqueness check: an existing experiment_id is never reused or replaced."""
        eid = rec.get("experiment_id")
        if not eid:
            raise ValueError("record needs an experiment_id")
        if any(r.get("experiment_id") == eid for r in self.records):
            raise ValueError(f"experiment_id {eid} already registered; the registry is append-only")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        return self.load()

    def get(self, eid):
        hits = [r for r in self.records if r.get("experiment_id") == eid]
        return hits[-1] if hits else None

    # ---- query ----
    def query(self, where=None, since=None, until=None, sort=None, desc=False, limit=None):
        """where: {field: value | callable | {'gt','lt','ge','le','in','ne'}}; fields may be dotted.
        since/until are ISO timestamps compared with the record timestamp. A record missing a filtered field
        never matches (a missing value is not a zero)."""
        out = []
        for r in self.records:
            ts = _get(r, "timestamp")
            if (since or until) and ts is None:
                continue
            if (since and str(ts) < since) or (until and str(ts) > until):
                continue
            if all(self._match(_dig(r, k), c) for k, c in (where or {}).items()):
                out.append(r)
        if sort:
            have = [r for r in out if _dig(r, sort) is not None]
            rest = [r for r in out if _dig(r, sort) is None]
            have.sort(key=lambda r: _dig(r, sort), reverse=desc)
            out = have + rest
        return out[:limit] if limit is not None else out

    @staticmethod
    def _match(v, cond):
        if v is None:
            return False
        if callable(cond):
            return bool(cond(v))
        if not isinstance(cond, dict):
            return v == cond
        x = _num(v)
        for op, c in cond.items():
            if op == "in" and v not in c:
                return False
            if op == "ne" and v == c:
                return False
            if op in ("gt", "lt", "ge", "le"):
                if x is None:
                    return False
                if (op == "gt" and not x > c) or (op == "lt" and not x < c) or \
                        (op == "ge" and not x >= c) or (op == "le" and not x <= c):
                    return False
        return True

    def best(self, metric, n=5, where=None, higher_is_better=True):
        rows = [r for r in self.query(where) if _num(_dig(r, metric)) is not None]
        rows.sort(key=lambda r: _num(_dig(r, metric)), reverse=higher_is_better)
        return rows[:n]

    def compare(self, ids, metrics):
        """Side-by-side table plus each experiment's delta to the first id. Unknown ids are listed under 'missing';
        a metric an experiment lacks is None, never 0."""
        table, missing = {}, []
        for i in ids:
            r = self.get(i)
            if r is None:
                missing.append(i)
                continue
            table[i] = {m: _num(_dig(r, m)) for m in metrics}
        delta = {}
        if table:
            ref = table[next(iter(table))]
            for i, row in table.items():
                delta[i] = {m: (None if row[m] is None or ref[m] is None else row[m] - ref[m]) for m in metrics}
        return {"table": table, "delta_vs_first": delta, "missing": missing}

    # ---- audit ----
    def audit(self):
        """Phase 0.2 completeness and integrity; `ok` is True only when there are no findings at all."""
        missing_by_field, dup_ids, orphans, bad_outcome, seen = {}, [], [], [], {}
        for r in self.records:
            for f in REQUIRED:
                if _empty(_get(r, f)):
                    missing_by_field.setdefault(f, []).append(r["_line"])
            eid = r.get("experiment_id")
            if eid:
                if eid in seen:
                    dup_ids.append((eid, seen[eid], r["_line"]))
                seen[eid] = r["_line"]
            oc = _get(r, "outcome")
            if oc is None:
                orphans.append(r["_line"])
            elif str(oc).lower() not in OUTCOMES:
                bad_outcome.append((r["_line"], oc))
        unrepro = self.reproducibility_conflicts()
        findings = {"n_records": len(self.records), "bad_lines": self.bad_lines, "missing_fields": missing_by_field,
                    "duplicate_ids": dup_ids, "orphans_no_outcome": orphans, "unknown_outcomes": bad_outcome,
                    "reproducibility_conflicts": unrepro}
        findings["ok"] = bool(self.records) and not (self.bad_lines or missing_by_field or dup_ids or orphans
                                                      or bad_outcome or unrepro)
        findings["completeness"] = ({f: 1 - len(missing_by_field.get(f, [])) / len(self.records) for f in REQUIRED}
                                    if self.records else {})
        return findings

    def reproducibility_conflicts(self, tol=1e-9):
        """Records with the same (config, data, seed, code) must report the same metrics; a difference means the
        experiment is not reproducible or a hidden input is missing from the record."""
        groups = {}
        for r in self.records:
            key = tuple(r.get(k) for k in REPRO_KEY)
            if None in key or not isinstance(r.get("metrics"), dict):
                continue
            groups.setdefault(key, []).append(r)
        out = []
        for key, rs in groups.items():
            for r in rs[1:]:
                a, b = rs[0]["metrics"], r["metrics"]
                diffs = {m: (a[m], b[m]) for m in a.keys() & b.keys()
                         if _num(a[m]) is not None and _num(b[m]) is not None
                         and abs(_num(a[m]) - _num(b[m])) > tol * max(1.0, abs(_num(a[m])))}
                if diffs:
                    out.append({"key": dict(zip(REPRO_KEY, key)), "lines": (rs[0]["_line"], r["_line"]), "diffs": diffs})
        return out

    def summary(self):
        oc, events = {}, {}
        for r in self.records:
            o = str(_get(r, "outcome") or "none").lower()
            oc[o] = oc.get(o, 0) + 1
            e = r.get("event", "?")
            events[e] = events.get(e, 0) + 1
        return {"n": len(self), "outcomes": oc, "events": events, "bad_lines": len(self.bad_lines)}


# ---------------- Phase 30: experiment memory ----------------
QUESTIONS = ("what_changed", "why_changed", "data_used", "data_unseen", "baseline", "improved", "worsened",
             "statistically_meaningful", "risk_changed", "survived_another_window", "adopted")


def fingerprint(change):
    """Canonical hash of a proposed change, so the same idea with keys in another order is recognised as a repeat."""
    def norm(x):
        if isinstance(x, dict):
            return {str(k): norm(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
        if isinstance(x, (list, tuple)):
            return [norm(v) for v in x]
        return round(x, 6) if isinstance(x, float) else x
    return hashlib.sha256(json.dumps(norm(change), sort_keys=True, default=str).encode()).hexdigest()[:16]


class ExperimentMemory:
    """Append-only lessons file. One entry per experiment answers the twelve Phase 30 questions (the twelfth,
    the reason for a rejection, is required when adopted is False). `orphans` lists registry experiments with no
    complete entry; `already_tried` blocks a repeat of a rejected idea."""

    def __init__(self, path):
        self.path = Path(path)
        self.entries = []
        if self.path.exists():
            self.entries = [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]

    @staticmethod
    def missing_answers(entry):
        """False / 0 are real answers ('nothing improved'); None, '' and empty containers are not."""
        miss = [q for q in QUESTIONS if _empty(entry.get(q))]
        if entry.get("adopted") is False and _empty(entry.get("if_rejected_why")):
            miss.append("if_rejected_why")
        return miss

    def record(self, experiment_id, change, answers, now):
        entry = {"experiment_id": experiment_id, "recorded_at": str(now), "fingerprint": fingerprint(change),
                 "change": change, **answers}
        miss = self.missing_answers(entry)
        if miss:
            raise ValueError(f"experiment {experiment_id} unanswered: {miss}")
        if any(e["experiment_id"] == experiment_id for e in self.entries):
            raise ValueError(f"experiment {experiment_id} already has a memory entry")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        self.entries.append(entry)
        return entry

    def already_tried(self, change):
        fp = fingerprint(change)
        hits = [e for e in self.entries if e["fingerprint"] == fp]
        rejected = [e for e in hits if e.get("adopted") is False]
        adopted = [e for e in hits if e.get("adopted") is True]
        return {"tried": len(hits), "blocked": bool(rejected) and not adopted,
                "reasons": [e.get("if_rejected_why") for e in rejected], "ids": [e["experiment_id"] for e in hits]}

    def orphans(self, registry):
        have = {e["experiment_id"] for e in self.entries if not self.missing_answers(e)}
        return sorted({r["experiment_id"] for r in registry.records if r.get("experiment_id")} - have)

    def lessons(self, adopted=None):
        return [e for e in self.entries if adopted is None or e.get("adopted") is adopted]
