"""Bible Phase 0.4 (immutable baseline snapshot) and Phase 35 (champion / challenger).

Baseline: the existing system is measured once, before it is modified, and frozen with a content hash; it can
be read and verified but never rewritten.

Board: Champion, Challenger, Candidate, Rejected. A challenger is never promoted because it is newer. `evaluate`
runs the six Phase 35 tests and every one of them fails closed - a missing gate, a missing metric or an
exception inside a check is a failure, never a pass. A failed promotion leaves the champion untouched and is
logged; a replaced champion is retired into history, not deleted."""
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from . import config as K

BASELINE_FIELDS = ("blind_window_results", "mover_accuracy", "model_results", "pattern_results", "analog_results",
                   "weekly_distribution", "max_drawdown", "worst_weeks", "turnover", "costs", "missed_winners")
# Phase 0.4 says "direction accuracy where available": it may be recorded as None but the key must be present
BASELINE_OPTIONAL = ("direction_accuracy",)

REQUIRED_GATES = ("parity", "retester", "future_scramble", "time_fence", "worker_health", "reproducibility")

# (metric, tier, direction, tolerance). Tier 1 is the highest priority (Blueprint: ~7%/week, then risk, then positives).
# direction 'target' = closer to `target` is better; 'up' / 'down' = larger / smaller is better.
OBJECTIVES = (("mean_week", 1, "target", 0.07, 0.003), ("share_in_band", 1, "up", None, 0.02),
              ("max_drawdown", 2, "up", None, 0.01), ("worst_week", 2, "up", None, 0.01),
              ("p05_week", 2, "up", None, 0.005), ("positive_week_pct", 3, "up", None, 0.02))

DEFAULT_POLICY = {"min_windows": 3, "min_weeks": 100, "min_improvement_sigma": 2.0,
                  "max_drawdown_floor": -0.35, "worst_week_floor": -0.20, "max_catastrophic": 0}


class ChampionError(RuntimeError):
    pass


def _atomic_write(p, text):
    p = Path(p)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)


def _canon(o):
    return json.dumps(o, sort_keys=True, default=str, allow_nan=False)


def _num(v):
    if isinstance(v, bool) or v is None:
        return None
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


# ---------------- Phase 0.4: immutable baseline ----------------
def freeze_baseline(path, results, now, provenance=None):
    """Write the baseline once. Refuses if it exists or if any Phase 0.4 measurement is absent: a partial baseline
    would let a later change be compared against something that was never measured."""
    p = Path(path)
    if p.exists():
        raise ChampionError("baseline already frozen; it is immutable")
    miss = [f for f in BASELINE_FIELDS if results.get(f) is None] + \
           [f for f in BASELINE_OPTIONAL if f not in results]
    if miss:
        raise ChampionError(f"baseline incomplete: {miss}")
    body = {"frozen_at": str(now), "results": results, "provenance": provenance or {}}
    body["sha256"] = hashlib.sha256(_canon({k: body[k] for k in ("frozen_at", "results", "provenance")}).encode()).hexdigest()
    p.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(p, json.dumps(body, indent=1, sort_keys=True, default=str))
    os.chmod(p, 0o444)
    return body["sha256"]


def load_baseline(path):
    """Read the baseline and verify its hash; a tampered file raises rather than returning wrong numbers."""
    b = json.loads(Path(path).read_text(encoding="utf-8"))
    want = hashlib.sha256(_canon({k: b[k] for k in ("frozen_at", "results", "provenance")}).encode()).hexdigest()
    if want != b.get("sha256"):
        raise ChampionError("baseline hash mismatch: file was altered after freezing")
    return b


def vs_baseline(baseline, metrics):
    """Delta of numeric scalar metrics against the frozen baseline results (None when either side is missing)."""
    base = baseline["results"]
    return {k: (None if _num(base.get(k)) is None or _num(v) is None else _num(v) - _num(base[k]))
            for k, v in metrics.items()}


# ---------------- Phase 35: the checks ----------------
def _check_gates(ch, pol):
    g = ch.get("gates") or {}
    bad = [n for n in REQUIRED_GATES if g.get(n) is not True]
    return not bad, "all gates pass" if not bad else f"gate not passed or missing: {bad}"


def _check_evidence(ch, pol):
    ev = ch.get("evidence") or {}
    wins = set(ev.get("windows") or [])
    trained = set(ev.get("train_windows") or [])
    weeks = ev.get("n_weeks") or 0
    if wins & trained:
        return False, f"evidence windows overlap training: {sorted(wins & trained)}"
    if len(wins) < pol["min_windows"]:
        return False, f"{len(wins)} independent windows < {pol['min_windows']}"
    if weeks < pol["min_weeks"]:
        return False, f"{weeks} weeks < {pol['min_weeks']}"
    return True, f"{len(wins)} windows, {weeks} weeks, none seen in training"


def _score(name, direction, target, v):
    return -abs(v - target) if direction == "target" else v


def _check_priority(ch, champ, pol):
    """No metric of tier <= the best improvement's tier may be worse by more than its tolerance, and there must be at
    least one real improvement. Compares only metrics present on both sides; a metric the champion has but the
    challenger lacks counts as harm (it cannot be shown unharmed)."""
    if champ is None:
        return True, "no champion to displace"
    cm, hm = champ.get("metrics") or {}, ch.get("metrics") or {}
    gains, harms, lost = [], [], []
    for name, tier, direction, target, tol in OBJECTIVES:
        a, b = _num(cm.get(name)), _num(hm.get(name))
        if a is None:
            continue
        if b is None:
            lost.append((tier, name))
            continue
        d = _score(name, direction, target, b) - _score(name, direction, target, a)
        if d > tol:
            gains.append((tier, name, d))
        elif d < -tol:
            harms.append((tier, name, d))
    if lost:
        return False, f"challenger lacks metrics the champion has: {[n for _, n in lost]}"
    if not gains:
        return False, "no objective improved beyond tolerance"
    # a gain at tier k never licenses a loss at any tier <= k (equal or higher priority)
    lowest = max(t for t, _, _ in gains)
    blocking = [h for h in harms if h[0] <= lowest]
    if blocking:
        return False, f"higher-priority objective harmed: {[(n, round(d, 4)) for _, n, d in blocking]}"
    return True, f"improved {[n for _, n, _ in gains]}, nothing of equal or higher priority harmed"


def _check_risk(ch, pol):
    m = ch.get("metrics") or {}
    dd, ww, cat = _num(m.get("max_drawdown")), _num(m.get("worst_week")), _num(m.get("catastrophic_losses"))
    if dd is None or ww is None or cat is None:
        return False, "risk metrics missing (max_drawdown, worst_week, catastrophic_losses)"
    bad = []
    if dd < pol["max_drawdown_floor"]:
        bad.append(f"max_drawdown {dd:.3f} < {pol['max_drawdown_floor']}")
    if ww < pol["worst_week_floor"]:
        bad.append(f"worst_week {ww:.3f} < {pol['worst_week_floor']}")
    if cat > pol["max_catastrophic"]:
        bad.append(f"catastrophic {cat:g} > {pol['max_catastrophic']}")
    return not bad, "within risk limits" if not bad else "; ".join(bad)


def _check_repro(ch, pol):
    r = ch.get("reproduced")
    return r is True, "reproduced" if r is True else f"reproducibility not shown ({r!r})"


def _check_blind(ch, pol):
    b = ch.get("blind") or {}
    ok = b.get("passed") is True and bool(b.get("windows"))
    return ok, "blind validation passed" if ok else "blind validation missing or failed"


def _check_basis(ch, pol):
    """Canon (memory rule): never promote on P&L alone. The recorded basis must include something other than P&L."""
    basis = set(ch.get("basis") or [])
    ok = bool(basis - {"pnl"})
    return ok, f"basis {sorted(basis)}" if ok else f"promotion basis is {sorted(basis) or 'empty'}: P&L alone never promotes"


class Ledger:
    """Append-only promotion ledger with a hash chain: each row carries the sha256 of the previous row, so an edited,
    deleted or reordered row is detected by `verify`."""

    def __init__(self, path):
        self.path = Path(path)

    def rows(self):
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def append(self, event, eid, now, **detail):
        rows = self.rows()
        prev = rows[-1]["hash"] if rows else "0" * 64
        body = {"seq": len(rows), "t": str(now), "event": event, "id": eid, "detail": detail, "prev": prev}
        body["hash"] = hashlib.sha256(_canon(body).encode()).hexdigest()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(body, sort_keys=True, default=str) + "\n")
        return body

    def verify(self):
        prev, probs = "0" * 64, []
        for i, r in enumerate(self.rows()):
            core = {k: r[k] for k in ("seq", "t", "event", "id", "detail", "prev")}
            if r["seq"] != i:
                probs.append(f"row {i}: seq {r['seq']}")
            if r["prev"] != prev:
                probs.append(f"row {i}: chain broken")
            if hashlib.sha256(_canon(core).encode()).hexdigest() != r["hash"]:
                probs.append(f"row {i}: content altered")
            prev = r["hash"]
        return {"ok": not probs, "problems": probs}


LEDGER_EVENTS = ("promoted", "promotion_refused", "rolled_back", "shadow_confirmed", "retired")


class Board:
    """State file layout: {champion: rec|None, challengers: {id: rec}, candidates: {id: rec}, rejected: {id: rec},
    history: [events]}. A rec is {id, metrics, gates, evidence, blind, reproduced, ...}."""

    def __init__(self, path, policy=None):
        self.path = Path(path)
        self.policy = {**DEFAULT_POLICY, **(policy or {})}
        self.ledger = Ledger(self.path.with_suffix(".ledger.jsonl"))
        if self.path.exists():
            self.s = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.s = {"champion": None, "challengers": {}, "candidates": {}, "rejected": {}, "history": []}
        self.s.setdefault("retired", [])
        self.s.setdefault("shadow", None)

    def _save(self):
        _atomic_write(self.path, json.dumps(self.s, indent=1, sort_keys=True, default=str))

    def _log(self, event, eid, now, **kw):
        self.s["history"].append({"t": str(now), "event": event, "id": eid, **kw})
        if event in LEDGER_EVENTS:
            self.ledger.append(event, eid, now, **kw)

    def _known(self, eid):
        return (self.s["champion"] or {}).get("id") == eid or any(eid in self.s[k] for k in
                                                                  ("challengers", "candidates", "rejected"))

    def add_candidate(self, rec, now):
        eid = rec.get("id")
        if not eid:
            raise ChampionError("record needs an id")
        if self._known(eid):
            raise ChampionError(f"{eid} is already on the board")
        self.s["candidates"][eid] = rec
        self._log("candidate", eid, now)
        self._save()

    def nominate(self, eid, now):
        """Candidate -> challenger. Rejected experiments cannot be nominated (they must return as a new id)."""
        if eid in self.s["rejected"]:
            raise ChampionError(f"{eid} was rejected: {self.s['rejected'][eid]['reason']}")
        if eid not in self.s["candidates"]:
            raise ChampionError(f"{eid} is not a candidate")
        self.s["challengers"][eid] = self.s["candidates"].pop(eid)
        self._log("nominated", eid, now)
        self._save()

    def update_evidence(self, eid, now, **fields):
        """Attach new evidence (metrics, gates, evidence, blind, reproduced) to a candidate or challenger."""
        for k in ("candidates", "challengers"):
            if eid in self.s[k]:
                self.s[k][eid].update(fields)
                self._log("evidence", eid, now, fields=sorted(fields))
                self._save()
                return
        raise ChampionError(f"{eid} is not open for evidence")

    def reject(self, eid, reason, now):
        if not reason:
            raise ChampionError("a rejection needs a reason (Phase 30)")
        for k in ("candidates", "challengers"):
            if eid in self.s[k]:
                rec = self.s[k].pop(eid)
                self.s["rejected"][eid] = {**rec, "reason": reason}
                self._log("rejected", eid, now, reason=reason)
                self._save()
                return
        raise ChampionError(f"{eid} cannot be rejected from its current role")

    def evaluate(self, eid):
        """Run the six tests on a challenger. Returns {'promote': bool, 'checks': {name: {'ok', 'detail'}}}.
        Any exception inside a check is recorded as a failure of that check."""
        ch = self.s["challengers"].get(eid)
        if ch is None:
            raise ChampionError(f"{eid} is not a challenger")
        champ = self.s["champion"]
        p = self.policy
        tests = (("required_gates", lambda: _check_gates(ch, p)), ("independent_evidence", lambda: _check_evidence(ch, p)),
                 ("priority_not_harmed", lambda: _check_priority(ch, champ, p)), ("risk_constraints", lambda: _check_risk(ch, p)),
                 ("reproducibility", lambda: _check_repro(ch, p)), ("blind_validation", lambda: _check_blind(ch, p)),
                 ("not_pnl_alone", lambda: _check_basis(ch, p)))
        checks = {}
        for name, fn in tests:
            try:
                ok, detail = fn()
            except Exception as e:
                ok, detail = False, f"check raised {type(e).__name__}: {e}"
            checks[name] = {"ok": bool(ok), "detail": detail}
        return {"promote": all(c["ok"] for c in checks.values()), "checks": checks}

    def promote(self, eid, now):
        res = self.evaluate(eid)
        if not res["promote"]:
            self._log("promotion_refused", eid, now, failed=[n for n, c in res["checks"].items() if not c["ok"]])
            self._save()
            return res
        old = self.s["champion"]
        if old is not None:
            self._log("retired", old["id"], now, replaced_by=eid)
            self.s["retired"].append(old)
        self.s["champion"] = self.s["challengers"].pop(eid)
        # the 10-session live shadow starts now; until it is confirmed the promotion is provisional
        self.s["shadow"] = {"new": eid, "old": (old or {}).get("id"), "since": str(now), "diffs": [],
                            "status": "pending" if old is not None else "no_predecessor"}
        self._log("promoted", eid, now, replaced=(old or {}).get("id"),
                  checks={n: c["detail"] for n, c in res["checks"].items()})
        self._save()
        return res

    def record_shadow_session(self, new_ret, old_ret, now):
        """One live session of the promoted champion and its predecessor's shadow, same-day returns. NaN is refused
        (a missing session is not a zero) so the tripwire never counts a day that was not observed."""
        sh = self.s["shadow"]
        if not sh or sh["status"] != "pending":
            raise ChampionError("no pending shadow period")
        if _num(new_ret) is None or _num(old_ret) is None:
            raise ChampionError("shadow session returns must be finite")
        sh["diffs"].append([str(now), float(new_ret), float(old_ret)])
        self._save()

    def shadow_review(self, now, sessions=10, t_stop=-1.28, min_gap=0.0005):
        """Tripwire (Blueprint M: a 10-session tripwire rolls back a champion that underperforms its predecessor).
        Fewer than `sessions` observations -> 'pending', nothing happens. At `sessions` or more: roll back when the
        mean paired difference (new - old) is below -min_gap AND its t-statistic is below t_stop; otherwise confirm.
        P&L only ever triggers a ROLLBACK here; it can never trigger a promotion."""
        sh = self.s["shadow"]
        if not sh or sh["status"] != "pending":
            return {"status": (sh or {}).get("status", "none")}
        d = np.array([n - o for _, n, o in sh["diffs"]], float)
        if len(d) < sessions:
            return {"status": "pending", "sessions": len(d), "needed": sessions}
        mean, sd = d.mean(), d.std(ddof=1)
        t = mean / (sd / np.sqrt(len(d))) if sd > 0 else (-np.inf if mean < 0 else np.inf if mean > 0 else 0.0)
        res = {"sessions": len(d), "mean_diff": float(mean), "t": float(t)}
        if mean < -min_gap and t < t_stop:
            return {**res, **self.rollback(now, f"shadow underperformed predecessor: mean {mean:+.4f}, t {t:.2f}")}
        sh["status"] = "confirmed"
        self._log("shadow_confirmed", sh["new"], now, **{k: v for k, v in res.items() if np.isfinite(v)})
        self._save()
        return {**res, "status": "confirmed"}

    def rollback(self, now, reason):
        """Restore the most recently retired champion. The rolled-back champion goes to `rejected` with the reason."""
        if not reason:
            raise ChampionError("a rollback needs a reason")
        if not self.s["retired"] or self.s["champion"] is None:
            raise ChampionError("nothing to roll back to")
        bad, prev = self.s["champion"], self.s["retired"].pop()
        self.s["rejected"][bad["id"]] = {**bad, "reason": f"rolled back: {reason}"}
        self.s["champion"] = prev
        if self.s["shadow"]:
            self.s["shadow"]["status"] = "rolled_back"
        self._log("rolled_back", bad["id"], now, restored=prev["id"], reason=reason)
        self._save()
        return {"status": "rolled_back", "restored": prev["id"], "reason": reason}

    def roles(self):
        return {"champion": (self.s["champion"] or {}).get("id"), "challengers": sorted(self.s["challengers"]),
                "candidates": sorted(self.s["candidates"]), "rejected": sorted(self.s["rejected"])}


def default_paths():
    return {"baseline": K.STATE / "baseline.json", "board": K.STATE / "champion_board.json"}


class ChallengerQueue:
    """Ordered waiting list for the (single) challenger slot. Priority = pre-registered expected gain (never realised
    P&L); ties break by arrival. With an engine.experiment_memory.TriedIndex, repeats and previously-blocked ideas are
    refused at submission."""

    def __init__(self, path, tried=None, max_len=25):
        self.path, self.tried, self.max_len = Path(path), tried, max_len
        self.items = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []

    def _save(self):
        _atomic_write(self.path, json.dumps(self.items, indent=1, sort_keys=True, default=str))

    def submit(self, eid, cfg, expected_gain, now):
        if any(i["id"] == eid for i in self.items):
            raise ChampionError(f"{eid} already queued")
        if _num(expected_gain) is None:
            raise ChampionError("expected_gain must be finite and stated up front")
        if len(self.items) >= self.max_len:
            raise ChampionError("queue full")
        if self.tried is not None:
            chk = self.tried.check(cfg)
            if chk["block"]:
                raise ChampionError(f"blocked by prior negative result: {chk['reasons']}")
            if chk["verdict"] in ("repeat", "near_duplicate"):
                raise ChampionError(f"duplicate of earlier attempt: {chk['exact'] or chk['close']}")
            key = self.tried.space.key(cfg)
        else:
            key = hashlib.sha256(_canon(cfg).encode()).hexdigest()[:16]
        if any(i["key"] == key for i in self.items):
            raise ChampionError("same configuration already queued")
        self.items.append({"id": eid, "cfg": cfg, "key": key, "expected_gain": float(expected_gain), "t": str(now),
                           "seq": len(self.items)})
        self._save()

    def order(self):
        return sorted(self.items, key=lambda i: (-i["expected_gain"], i["seq"]))

    def next_challenger(self, board, now, make_record):
        """Move the top item into the board as candidate -> challenger, only while no challenger is open.
        make_record(item) returns the board record. Returns the id, or None."""
        if board.s["challengers"] or not self.items:
            return None
        top = self.order()[0]
        board.add_candidate(make_record(top), now)
        board.nominate(top["id"], now)
        self.items = [i for i in self.items if i["id"] != top["id"]]
        self._save()
        return top["id"]
