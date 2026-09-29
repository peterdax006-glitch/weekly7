"""Bible Phase 5 - long-term pattern bank (canon C43; "no pattern is trusted forever, none deleted for one bad regime").

Persistent, versioned, hash-chained store of PatternLifecycle records. Each commit writes a NEW file `bank_vNNNNNN.json`
(never overwrites) holding the whole bank, its SHA-256 and the hash of its parent, under a cross-process lock file.
A blind window may only read what earlier windows knew: `read(as_of)` returns each record as it stood strictly before
`as_of` (state, scope, effect, discovery, evidence windows, failures and rescopes all rebuilt from dated histories).
Evidence from repeated runs is merged by set-union keyed on (window_end, run_id), so re-committing a run is idempotent;
relevance decays with the age of the last validation, so a pattern that is never retested drifts toward zero trust."""
import contextlib
import copy
import hashlib
import json
import os
import time
from math import erf, sqrt

import numpy as np
import pandas as pd

from .pattern_lifecycle import ALLOWED, Lifecycle, Panel, TRADEABLE, _iso, _num

SCHEMA = 1
BANK_DEFAULT = {"half_life_years": 3.0, "max_prior": 500, "lock_timeout_s": 30.0, "lock_stale_s": 120.0,
                "min_trust": 0.05, "store_noise": False}
NOISE_STATES = ("rejected", "duplicate", "no_gain")
TRACKED = {"effect": 0.0, "scope": None, "discovery": {}, "watch_count": 0, "discard_reason": None}
PRIOR_STATES = ("active", "watch", "rescoped", "discarded")       # rejected/duplicate/no_gain were noise, not regime losses


class BankError(RuntimeError):
    pass


class BankCorrupt(BankError):
    pass


class BankLockTimeout(BankError):
    pass


def _clean(o):
    """JSON-safe copy: numpy scalars to python, non-finite floats to None (allow_nan stays off, so the hash is stable)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return _num(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    return o


def canon_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@contextlib.contextmanager
def file_lock(path, timeout=30.0, stale=120.0, poll=0.05):
    """Cross-process lock: atomic O_EXCL create of a lock file. A lock older than `stale` seconds is presumed dead
    (its owner crashed) and broken; a live holder past `timeout` raises BankLockTimeout rather than corrupting the bank."""
    t0 = time.monotonic()
    while True:
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(path) > stale:
                    os.remove(path)
                    continue
            except OSError:
                continue                                    # holder released between our exists-check and stat
            if time.monotonic() - t0 > timeout:
                raise BankLockTimeout(f"could not acquire {path} within {timeout}s")
            time.sleep(poll)
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            os.remove(path)


# ---------------------------------------------------------------- evidence maths (pure)
def decay(age_days, half_life_years):
    return float(0.5 ** (max(float(age_days), 0.0) / 365.25 / half_life_years))


def _ncdf(z):
    return 0.5 * (1 + erf(z / sqrt(2)))


def summarize(rec, as_of, half_life_years=BANK_DEFAULT["half_life_years"]):
    """Pooled evidence for one (already as-of-filtered) record.
    relevance: decay of the last validation's age. modernity: decay-weighted share of windows in which the pattern
    held in its own direction. pooled_t: decay-weighted MEAN of window t-statistics (not Stouffer - retests on growing
    panels overlap, so summing them would invent independence). confidence = Phi(pooled_t). trust multiplies the three."""
    now = pd.Timestamp(as_of)
    sign = float(np.sign(rec.get("effect") or 0.0)) or 1.0
    wins = [w for w in rec.get("windows", []) if pd.Timestamp(w["window_end"]) <= now]
    if not wins:
        age = (now - pd.Timestamp(rec["first_seen"])).days
        return {"n_windows": 0, "relevance": decay(age, half_life_years), "modernity": 0.0, "pooled_t": 0.0,
                "pooled_effect": None, "confidence": 0.5, "trust": 0.0, "last_validation": None, "n_reviews": 0}
    ages = np.array([(now - pd.Timestamp(w["window_end"])).days for w in wins], float)
    d = 0.5 ** (np.maximum(ages, 0) / 365.25 / half_life_years)
    wt = d * np.sqrt(np.array([max(w.get("n") or 1, 1) for w in wins], float))
    t = np.array([w["t"] if w.get("t") is not None else 0.0 for w in wins], float)
    hit = (sign * t > 0).astype(float)
    ms = [(w["m"], x) for w, x in zip(wins, wt) if w.get("m") is not None]
    pooled_t = float((wt * sign * t).sum() / wt.sum())
    relevance = float(d.max())                              # the newest window is the one that sets currency
    modernity = float((d * hit).sum() / d.sum())
    conf = _ncdf(pooled_t)
    return {"n_windows": len(wins), "relevance": relevance, "modernity": modernity, "pooled_t": pooled_t,
            "pooled_effect": float(sum(m * x for m, x in ms) / sum(x for _, x in ms)) if ms else None,
            "confidence": conf, "trust": relevance * modernity * conf,
            "last_validation": max(w["window_end"] for w in wins),
            "n_reviews": sum(w.get("source") == "review" for w in wins)}


# ---------------------------------------------------------------- record shaping and merging
def _track(rec, as_of):
    """Append a dated entry to <field>_history whenever a tracked field differs from its last recorded value."""
    for f, default in TRACKED.items():
        h = rec.setdefault("hist_" + f, [])
        val = rec.get(f, default)
        if not h or h[-1]["v"] != val:
            h.append({"as_of": _iso(as_of), "v": copy.deepcopy(val)})


def _union(a, b, key):
    seen, out = set(), []
    for x in list(a) + list(b):
        k = key(x)
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


def _hist_key(e):
    return (e["as_of"], e["kind"], e["frm"], e["to"], e["reason"])


def merge_records(a, b):
    """Union of two bank records of the SAME pattern. Lists merge by dated keys; scalars come from the record whose
    state changed last (ties: b). Never drops evidence, so merging is order-insensitive for everything but ties."""
    if a is None:
        return copy.deepcopy(b)
    if a["id"] != b["id"]:
        raise BankError("merge of different patterns")
    base = copy.deepcopy(b if pd.Timestamp(b["changed"]) >= pd.Timestamp(a["changed"]) else a)
    base["history"] = sorted(_union(a.get("history", []), b.get("history", []), _hist_key), key=lambda e: e["as_of"])
    base["windows"] = sorted(_union(a.get("windows", []), b.get("windows", []),
                                    lambda w: (w["window_end"], w["run_id"])), key=lambda w: (w["window_end"], w["run_id"]))
    base["failures"] = _union(a.get("failures", []), b.get("failures", []), lambda x: (x["as_of"], x["reason"]))
    base["rescopes"] = _union(a.get("rescopes", []), b.get("rescopes", []),
                              lambda x: (x["as_of"], x["source"], json.dumps(x["scope"], sort_keys=True)))
    base["runs"] = sorted(_union(a.get("runs", []), b.get("runs", []), lambda x: (x["as_of"], x["run_id"])),
                          key=lambda x: (x["as_of"], x["run_id"]))
    for f in TRACKED:
        h = {e["as_of"]: e for e in a.get("hist_" + f, []) + b.get("hist_" + f, [])}
        base["hist_" + f] = [h[k] for k in sorted(h)]
    base["first_seen"] = min(a["first_seen"], b["first_seen"], key=pd.Timestamp)
    base["version"] = max(a.get("version", 0), b.get("version", 0))
    return base


def view_record(rec, cutoff):
    """The record as an earlier window would have seen it: only entries dated strictly before `cutoff`. None if the
    pattern did not yet exist."""
    c = pd.Timestamp(cutoff)
    before = lambda e, k="as_of": pd.Timestamp(e[k]) < c
    hist = [e for e in rec.get("history", []) if before(e)]
    trans = [e for e in hist if e["kind"] == "transition"]
    if not trans:
        return None
    out = copy.deepcopy(rec)
    out["history"], out["state"], out["changed"] = hist, trans[-1]["to"], trans[-1]["as_of"]
    for f, default in TRACKED.items():
        h = [e for e in rec.get("hist_" + f, []) if before(e)]
        out[f] = copy.deepcopy(h[-1]["v"]) if h else copy.deepcopy(default)
        out["hist_" + f] = h
    out["windows"] = [w for w in rec.get("windows", []) if before(w, "window_end")]
    out["runs"] = [x for x in rec.get("runs", []) if before(x)]
    out["failures"] = [x for x in rec.get("failures", []) if before(x)]
    out["rescopes"] = [x for x in rec.get("rescopes", []) if before(x)]
    out["version"] = len(trans)
    lv = out["windows"][-1] if out["windows"] else None
    out["last_validation"] = None if lv is None else {
        "as_of": lv["window_end"], "verdict": lv.get("verdict"), "m_long": lv.get("m"), "t_long": lv.get("t"),
        "n_long": lv.get("n"), "m_recent": lv.get("m_recent"), "t_recent": lv.get("t_recent"),
        "n_recent": lv.get("n_recent"), "scope": out["scope"]}
    return out


def _window(rec, run_id, source, as_of):
    """An evidence window exists only for a check made AT this as_of; a stale last_validation is not re-counted."""
    lv = rec.get("last_validation")
    if lv and lv.get("as_of") == _iso(as_of):
        return {"window_end": lv["as_of"], "run_id": run_id, "source": source, "m": lv.get("m_long"),
                "t": lv.get("t_long"), "n": lv.get("n_long"), "m_recent": lv.get("m_recent"),
                "t_recent": lv.get("t_recent"), "n_recent": lv.get("n_recent"), "verdict": lv.get("verdict"),
                "scoped": lv.get("scope") is not None}
    d = rec.get("discovery") or {}
    if d.get("as_of") == _iso(as_of) and d.get("t_conf") is not None:
        return {"window_end": d["as_of"], "run_id": run_id, "source": "miner", "m": rec.get("effect"), "t": d["t_conf"],
                "n": None, "m_recent": None, "t_recent": None, "n_recent": None, "verdict": "miner", "scoped": False}
    return None


# ---------------------------------------------------------------- the bank
class PatternBank:
    def __init__(self, root, params=None):
        self.root = os.fspath(root)
        self.p = {**BANK_DEFAULT, **(params or {})}
        os.makedirs(self.root, exist_ok=True)
        self.lock_path = os.path.join(self.root, "bank.lock")
        self.corrupt = []

    # ----- storage
    def _file(self, v):
        return os.path.join(self.root, f"bank_v{v:06d}.json")

    def versions(self):
        out = []
        for f in os.listdir(self.root):
            if f.startswith("bank_v") and f.endswith(".json") and f[6:-5].isdigit():
                out.append(int(f[6:-5]))
        return sorted(out)

    def _read_version(self, v):
        try:
            with open(self._file(v), "rb") as fh:
                env = json.loads(fh.read().decode("utf-8"))
            payload, digest = env["payload"], env["sha256"]
        except (OSError, ValueError, KeyError, UnicodeDecodeError) as e:
            raise BankCorrupt(f"v{v} unreadable: {e}") from e
        if canon_hash(payload) != digest:
            raise BankCorrupt(f"v{v} integrity hash mismatch")
        if payload.get("schema") != SCHEMA or payload.get("version") != v:
            raise BankCorrupt(f"v{v} schema/version field wrong")
        return payload, digest

    def head(self):
        """(version, payload, digest) of the newest version that verifies; corrupt newer ones are recorded, skipped and
        never overwritten. An empty bank is version 0."""
        self.corrupt = []
        for v in reversed(self.versions()):
            try:
                payload, digest = self._read_version(v)
                return v, payload, digest
            except BankCorrupt as e:
                self.corrupt.append(str(e))
        return 0, {"schema": SCHEMA, "version": 0, "parent": None, "records": {}, "noise": {}}, None

    def _write(self, payload):
        payload = _clean(payload)
        digest = canon_hash(payload)
        data = json.dumps({"payload": payload, "sha256": digest}, sort_keys=True, separators=(",", ":")).encode("utf-8")
        final = self._file(payload["version"])
        tmp = final + f".tmp{os.getpid()}"
        with open(tmp, "wb") as fh:                          # binary: no CRLF translation, hash stays portable
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        if os.path.exists(final):
            os.remove(tmp)
            raise BankError(f"refusing to overwrite {final}")
        os.replace(tmp, final)
        return digest

    def records(self):
        return copy.deepcopy(self.head()[1]["records"])

    def noise_ledger(self):
        """Per run: how many never-confirmed candidates (rejected / duplicate / no_gain) were seen and not stored
        individually. They are counted, so nothing vanishes unrecorded; the miner re-derives them if they ever qualify."""
        return copy.deepcopy(self.head()[1].get("noise", {}))

    def verify(self):
        """Every version checked: hash valid, and each parent pointer equals the hash of the previous version."""
        bad, breaks, prev = [], [], None
        for v in self.versions():
            try:
                payload, digest = self._read_version(v)
            except BankCorrupt as e:
                bad.append(str(e))
                prev = None
                continue
            if prev is not None and payload.get("parent") != prev:
                breaks.append(f"v{v} parent hash does not match the previous version")
            prev = digest
        return {"ok": not bad and not breaks, "versions": len(self.versions()), "corrupt": bad, "chain_breaks": breaks}

    # ----- writing
    def commit(self, life, as_of, run_id, since=0):
        """Fold a Lifecycle (records + log[since:]) into the bank as one new version. Runs under the file lock and
        merges against whatever head exists at that moment, so concurrent writers lose nothing."""
        new_log = {}
        for e in life.log[since:]:
            new_log.setdefault(e["id"], []).append({k: e[k] for k in ("kind", "as_of", "frm", "to", "reason", "evidence")})
        with file_lock(self.lock_path, self.p["lock_timeout_s"], self.p["lock_stale_s"]):
            _, payload, digest = self.head()
            top = max(self.versions() or [0])
            recs = copy.deepcopy(payload["records"])
            noise = copy.deepcopy(payload.get("noise", {}))
            tally = {}
            for rid, r in life.records.items():
                if r["state"] in NOISE_STATES and rid not in recs and not self.p["store_noise"]:
                    tally[r["state"]] = tally.get(r["state"], 0) + 1      # counted in the ledger, not stored one by one
                    continue
                r = copy.deepcopy(r)
                r.setdefault("history", [])
                r["history"] = _union(r["history"], [dict(e, id=rid) for e in new_log.get(rid, [])], _hist_key)
                r["history"] = [{k: v for k, v in e.items() if k != "id"} for e in r["history"]]
                r.setdefault("windows", [])
                w = _window(r, run_id, "review", as_of)
                if w:
                    r["windows"] = _union(r["windows"], [w], lambda x: (x["window_end"], x["run_id"]))
                r["runs"] = _union(r.get("runs", []), [{"as_of": _iso(as_of), "run_id": run_id}],
                                   lambda x: (x["as_of"], x["run_id"]))
                _track(r, as_of)
                recs[rid] = merge_records(recs.get(rid), _clean(r))
            if tally:
                noise[run_id] = {"as_of": _iso(as_of), **tally}
            out = {"schema": SCHEMA, "version": top + 1, "parent": digest, "records": recs, "noise": noise}
            self._write(out)
            return top + 1

    def ingest_miner(self, frame, as_of, run_id, params=None):
        """Record one PatternMiner run: read the bank as of `as_of`, fold the frame in, commit."""
        life = self._life_at(as_of, params)
        n0 = len(life.log)
        res = life.ingest(frame, as_of, run_id)
        v = self.commit(life, as_of, run_id, since=n0)
        return {"version": v, **res}

    def retest(self, X, y, as_of, run_id, params=None):
        """Retest the whole bank on a new window (Bible: every blind window retests the bank). Reads only entries
        before `as_of`; the panel excludes any row whose outcome closed after `as_of`."""
        life = self._life_at(as_of, params)
        n0 = len(life.log)
        summ = life.review(Panel.build(X, y, as_of, params), as_of)
        bad = life.invariants()
        if bad:
            raise BankError("lifecycle invariants violated: " + "; ".join(bad[:3]))
        v = self.commit(life, as_of, run_id, since=n0)
        return {"version": v, **summ}

    def _life_at(self, as_of, params=None):
        recs = self.read(as_of)
        log = []
        for r in recs:
            log += [dict(e, id=r["id"], name=r["name"]) for e in r["history"]]
        log.sort(key=lambda e: e["as_of"])
        for i, e in enumerate(log):
            e["seq"] = i
        return Lifecycle(recs, log, params)

    # ----- reading
    def read(self, as_of, strict=True):
        """Records as they stood before `as_of` (strict) - the only read a blind window may use."""
        cutoff = pd.Timestamp(as_of) + (pd.Timedelta(0) if strict else pd.Timedelta(nanoseconds=1))
        out = []
        for rid, rec in sorted(self.head()[1]["records"].items()):
            v = view_record(rec, cutoff)
            if v is not None:
                out.append(v)
        return out

    def summary(self, as_of):
        """One row per pattern with decayed relevance, modernity, pooled evidence, confidence and trust."""
        rows = []
        for r in self.read(as_of):
            s = summarize(r, pd.Timestamp(as_of) - pd.Timedelta(nanoseconds=1), self.p["half_life_years"])
            rows.append({"id": r["id"], "name": r["name"], "state": r["state"], "effect": r["effect"], "scope": r["scope"],
                         "failures": len(r["failures"]), "rescopes": len(r["rescopes"]), **s})
        return pd.DataFrame(rows)

    def trusted(self, as_of, min_trust=None):
        """Tradeable patterns that a blind retest has validated at least once and whose decayed trust clears the bar.
        Discovery evidence alone (the miner's own confirmation half) never qualifies; age erodes what remains."""
        S = self.summary(as_of)
        if S.empty:
            return S
        bar = self.p["min_trust"] if min_trust is None else min_trust
        return S[S["state"].isin([s for s, w in TRADEABLE.items() if w > 0]) & (S["trust"] > bar)
                 & (S["n_reviews"] > 0)].reset_index(drop=True)

    def prior_frame(self, as_of):
        """Frame for PatternMiner.fit(prior=...): the miner re-tests each entry as a candidate on its own window.
        Includes discarded patterns (a failure in one regime is not a deletion), best trust first, capped."""
        S = self.summary(as_of)
        if S.empty:
            return pd.DataFrame(columns=["names", "effect", "p_real", "state", "trust"])
        recs = {r["id"]: r for r in self.read(as_of)}
        S = S[S["state"].isin(PRIOR_STATES)].sort_values(["trust", "id"], ascending=[False, True]).head(self.p["max_prior"])
        return pd.DataFrame({"names": [recs[i]["names"] for i in S["id"]], "effect": S["effect"].values,
                             "p_real": [recs[i]["discovery"].get("p_real") for i in S["id"]],
                             "state": S["state"].values, "trust": S["trust"].values}).reset_index(drop=True)

    def history(self, pattern_id, as_of=None):
        """Full dated transition history of one pattern (for audit), optionally as an earlier window saw it."""
        recs = self.head()[1]["records"]
        if pattern_id not in recs:
            raise KeyError(pattern_id)
        r = recs[pattern_id] if as_of is None else view_record(recs[pattern_id], as_of)
        return [] if r is None else copy.deepcopy(r["history"])

    def audit(self):
        """Promises checked on the stored bank itself: every record has a transition history ending in its state, only
        legal moves, a discard reason if discarded, and nothing was ever removed between versions."""
        problems, prev_ids = [], set()
        for v in self.versions():
            try:
                ids = set(self._read_version(v)[0]["records"])
            except BankCorrupt:
                continue
            gone = prev_ids - ids
            if gone:
                problems.append(f"v{v} dropped {len(gone)} pattern(s)")
            prev_ids = ids
        for rid, r in self.head()[1]["records"].items():
            tr = [e for e in r.get("history", []) if e["kind"] == "transition"]
            if not tr or tr[-1]["to"] != r["state"]:
                problems.append(f"{r['name']}: state {r['state']} not backed by history")
            problems += [f"{r['name']}: illegal {e['frm']}->{e['to']}" for e in tr if e["to"] not in ALLOWED.get(e["frm"], ())]
            if r["state"] == "discarded" and not r["discard_reason"]:
                problems.append(f"{r['name']}: discarded without reason")
            if r["state"] in ("failed", "cause_search"):
                problems.append(f"{r['name']}: stuck in {r['state']}")
        return problems
