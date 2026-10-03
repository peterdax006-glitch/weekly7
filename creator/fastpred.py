"""FAST-RESOLVING PROSPECTIVE TOPICS (owner, 2 Oct 2026: Nupen's opinion must become TRUSTWORTHY, measured). The slow topics (git_fixed, package
verdicts) resolve in days, so no topic could earn the trust gate overnight. These topics are resolved by Nupen's OWN activity within minutes:

  drill_beats_best     before a drill variant job runs: P(its select Brier beats the best so far of its source)          resolved when the job ends
  test_file_passes     before a test file runs in an evaluation: P(it passes)                                              resolved when pytest ends
  test_file_slow       ... P(it takes longer than TEST_SLOW_S)                                                              resolved when pytest ends
  stage_slow           at a cycle stage start: P(the stage takes longer than STAGE_SLOW_S)                                  resolved when the stage ends
  judgment_correct     before a judgment call: P(the model's answer is closer to the truth than the statistical predictor) resolved after the call
  plan_first           when the swarm plans k packages: P(this one reaches a verdict first)                                 resolved by the first verdict

PROSPECTIVE DISCIPLINE: `begin` appends the prediction (timestamped, with its p, its baselines and the key) to state/creator/thinking/
fast_predictions.jsonl BEFORE the outcome exists; `end` appends the outcome. The file is append-only; folding the file accepts an outcome only if
its prediction line EARLIER in the file exists, was never resolved, and the outcome time is not before the prediction time: anything else is
ignored (an outcome written before its prediction is rejected). A prediction is never edited, and begin is idempotent per id.
LEARNER: the per-key decayed frequency shrunk toward the topic's overall rate, trained only on outcomes resolved before the prediction time
(bootstrapped from earlier resolved drill rows for drill_beats_best, walk-forward replay marked mode='replay'). Baselines: the overall rate and
the last value. The gate is thinking.score/trust_of, unchanged (n>=50, >=10 prospective, CI over the best baseline, ECE). Hooks call
`begin_safe/end_safe`: no-ops when the module is missing or NUPEN_FASTPRED=0 or under pytest without NUPEN_FASTPRED_STATE; best effort, never raise.
Loaded on demand (registry 'fastpred')."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

from creator import thinking as T

TOPICS = ("drill_beats_best", "test_file_passes", "test_file_slow", "stage_slow", "judgment_correct", "plan_first")
TEST_SLOW_S = 20.0               # a test file taking longer than this (seconds) is "slow"
STAGE_SLOW_S = 30.0              # a cycle stage taking longer than this is "slow"
DECAY = 0.97
K = 4.0                          # observations the topic-wide rate is worth when estimating one key
_lock = threading.Lock()


# ------------------------------------------------------------------------------------------------ where, and whether
def state_dir() -> Path:
    env = os.environ.get("NUPEN_FASTPRED_STATE")
    return Path(env) if env else Path(__file__).resolve().parents[1] / "state" / "creator"


def enabled() -> bool:
    if os.environ.get("NUPEN_FASTPRED") == "0":
        return False
    return not ("PYTEST_CURRENT_TEST" in os.environ and not os.environ.get("NUPEN_FASTPRED_STATE"))


def store(state: Path) -> Path:
    return Path(state) / "thinking" / "fast_predictions.jsonl"


# ------------------------------------------------------------------------------------------------ the append-only store, folded incrementally
class _Fold:
    """The store folded in file order; re-reads only the bytes appended since the last call (cheap on the hot path)."""

    def __init__(self, path: Path) -> None:
        self.path, self.offset = path, 0
        self.preds: dict[str, dict[str, Any]] = {}
        self.outs: dict[str, dict[str, Any]] = {}
        self.rejected = 0
        self._tail = b""

    def refresh(self) -> None:
        try:
            size = self.path.stat().st_size
        except OSError:
            return
        if size < self.offset:                                    # truncated or replaced: start over
            self.__init__(self.path)                              # type: ignore[misc]
        if size == self.offset:
            return
        with self.path.open("rb") as f:
            f.seek(self.offset)
            data = self._tail + f.read()
        self.offset = size
        *lines, self._tail = data.split(b"\n")
        for raw in lines:
            try:
                r = json.loads(raw.decode("utf-8", "replace"))
            except ValueError:
                continue
            if not isinstance(r, dict):
                continue
            if r.get("kind") == "pred" and r.get("id") and r["id"] not in self.preds:
                self.preds[r["id"]] = r
            elif r.get("kind") == "out":
                p = self.preds.get(r.get("id", ""))
                if p is None or r["id"] in self.outs or float(r.get("at", 0)) < float(p["made_at"]) or r.get("y") not in (0, 1):
                    self.rejected += 1                            # before its prediction, twice, or time-travelling: ignored
                else:
                    self.outs[r["id"]] = r


_folds: dict[str, _Fold] = {}


def fold(state: Path) -> _Fold:
    k = str(store(state))
    with _lock:
        f = _folds.get(k)
        if f is None:
            f = _folds[k] = _Fold(store(state))
        f.refresh()
        return f


def _append(state: Path, rec: dict[str, Any]) -> None:
    p = store(state)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("ab") as f:                                       # one write of one line: appends from several processes interleave whole lines
        f.write((json.dumps(rec, sort_keys=True) + "\n").encode("utf-8"))


# ------------------------------------------------------------------------------------------------ the learner
def history(state: Path, topic: str, fo: Optional[_Fold] = None) -> list[tuple[float, str, int, float]]:
    """Resolved events (resolved_at, key, y, group_size) in resolution order: bootstrap rows plus resolved live predictions."""
    fo = fo or fold(state)
    ev = [(float(o["at"]), fo.preds[i]["key"], int(o["y"]), float(fo.preds[i].get("k", 1))) for i, o in fo.outs.items() if fo.preds[i]["topic"] == topic]
    ev += [(t, k, y, 1.0) for t, k, y in bootstrap(state, topic)]
    return sorted(ev)


def estimate(ev: list[tuple[float, str, int, float]], key: str, prior_k: float = 0.0, at: Optional[float] = None) -> tuple[float, float, float]:
    """(p, base, last) for `key` from events resolved strictly before `at`. base = overall Laplace rate (or 1/prior_k for plan_first), last = last value."""
    tr = [e for e in ev if at is None or e[0] < at]
    ys = [e[2] for e in tr]
    base = (1.0 / prior_k) if prior_k else (sum(ys) + 1.0) / (len(ys) + 2.0)
    mine = [e[2] for e in tr if e[1] == key]
    n = len(mine)
    s = sum(y * DECAY ** (n - 1 - j) for j, y in enumerate(mine))
    w = sum(DECAY ** (n - 1 - j) for j in range(n))
    p = (s + K * base) / (w + K)
    last = 0.5 if not ys else (0.75 if ys[-1] else 0.25)
    return min(0.98, max(0.02, p)), min(0.98, max(0.02, base)), last


_boot: dict[tuple[str, str], tuple[float, list[tuple[float, str, int]]]] = {}


def bootstrap(state: Path, topic: str) -> list[tuple[float, str, int]]:
    """Earlier resolved rows that train a topic (cached 5 min). Only drill_beats_best has any: drill_runs.jsonl, each row (a select Brier) beating
    the best earlier row of its source and data digest."""
    if topic != "drill_beats_best":
        return []
    k = (str(state), topic)
    c = _boot.get(k)
    if c and time.time() - c[0] < 300:
        return c[1]
    rows = [r for r in T._jsonl(Path(state) / "thinking" / "drill_runs.jsonl") if r.get("select", {}).get("n") and r.get("at")]
    rows.sort(key=lambda r: r["at"])
    best: dict[tuple[str, str], float] = {}
    out: list[tuple[float, str, int]] = []
    for r in rows:
        g = (r["source"], r.get("digest", ""))
        b = r["select"]["brier"]
        if g in best:
            out.append((T._ts(r["at"]), _drill_key(r["source"], bool(r.get("search"))), int(b < best[g])))
        best[g] = min(b, best.get(g, b))
    _boot[k] = (time.time(), out)
    return out


def _drill_key(source: str, search: bool) -> str:
    return f"{source}|{'search' if search else 'grid'}"


# ------------------------------------------------------------------------------------------------ begin / end
def begin(state: Path, topic: str, subject: str, key: str, now: Optional[float] = None, group: str = "", k: int = 1,
          extra: Optional[dict[str, Any]] = None) -> Optional[str]:
    """Write the prediction BEFORE the outcome exists and return its id. Idempotent: a subject already predicted for this topic returns the same id
    and writes nothing."""
    if topic not in TOPICS:
        raise ValueError(f"unknown fast topic {topic!r}")
    state = Path(state)
    pid = f"{topic}:{subject}"
    fo = fold(state)
    if pid in fo.preds:
        return pid
    t = time.time() if now is None else now
    p, base, last = estimate(history(state, topic, fo), key, float(k) if topic == "plan_first" else 0.0, at=t)
    rec: dict[str, Any] = {"kind": "pred", "id": pid, "topic": topic, "subject": subject, "key": key, "made_at": t, "p": round(p, 6),
                           "base": round(base, 6), "last": last, "mode": "live", "group": group, "k": k}
    if extra:
        rec["extra"] = extra
    _append(state, rec)
    return pid


def end(state: Path, pid: Optional[str], y: int, now: Optional[float] = None) -> bool:
    """Write the outcome. Rejected (False, nothing written) when there is no earlier prediction, it is already resolved, or `now` precedes it."""
    if pid is None:
        return False
    state = Path(state)
    fo = fold(state)
    pr = fo.preds.get(pid)
    t = time.time() if now is None else now
    if pr is None or pid in fo.outs or t < float(pr["made_at"]) or y not in (0, 1):
        return False
    _append(state, {"kind": "out", "id": pid, "y": int(y), "at": t})
    fold(state)
    return True


def begin_safe(topic: str, subject: str, key: str, **kw: Any) -> Optional[str]:
    """The hook entry: never raises, a no-op when disabled."""
    try:
        if enabled():
            return begin(kw.pop("state", None) or state_dir(), topic, subject, key, **kw)
    except Exception:                                             # noqa: BLE001 - a hook never breaks the hooked path
        pass
    return None


def end_safe(pid: Optional[str], y: int, state: Optional[Path] = None) -> None:
    try:
        if pid is not None and enabled():
            end(state or state_dir(), pid, y)
    except Exception:                                             # noqa: BLE001
        pass


# ------------------------------------------------------------------------------------------------ hook helpers (one per topic)
def tests_begin(files: list[str]) -> list[tuple[str, str, str]]:
    """Before a test run: predict pass and slowness per file. Returns [(file, pass_id, slow_id)] for tests_end."""
    run = f"{time.time():.3f}-{os.getpid()}"
    out = []
    for f in files:
        fk = f.replace("\\", "/").split("::", 1)[0]
        out.append((fk, begin_safe("test_file_passes", f"{run}:{fk}", fk) or "", begin_safe("test_file_slow", f"{run}:{fk}", fk) or ""))
    return out


def tests_end(pending: list[tuple[str, str, str]], cases: dict[str, Any], seconds: float, crashed: bool) -> None:
    """After it: a file passed when none of its cases is bad; its duration is the sum of its case seconds (the run's when no case is known)."""
    if crashed:
        return                                                     # a crashed/timed-out run says nothing about the files: the predictions stay unscored
    for fk, pid, sid in pending:
        mine = [c for cid, c in cases.items() if cid.replace("\\", "/").split("::", 1)[0] == fk]
        if not mine:
            continue
        end_safe(pid or None, int(not any(c.outcome.bad for c in mine)))
        end_safe(sid or None, int(sum(float(c.seconds) for c in mine) > TEST_SLOW_S))


def plan_begin(subjects: list[tuple[str, str]]) -> None:
    """At a planning decision: (package_id, requirement kind) of every package planned together; they race to a verdict."""
    if len(subjects) < 2:
        return
    g = f"{time.time():.3f}-{os.getpid()}"
    for pkg, key in subjects:
        begin_safe("plan_first", pkg, key, group=g, k=len(subjects))


def plan_verdict(pkg: str) -> None:
    """A package reached a verdict: the first of its group gets 1, the rest 0 (they were beaten)."""
    try:
        if not enabled():
            return
        state = state_dir()
        fo = fold(state)
        pid = f"plan_first:{pkg}"
        pr = fo.preds.get(pid)
        if pr is None or pid in fo.outs:
            return
        sib = [i for i, p in fo.preds.items() if p["topic"] == "plan_first" and p.get("group") == pr.get("group")]
        if any(i in fo.outs and fo.outs[i]["y"] == 1 for i in sib):
            end(state, pid, 0)
            return
        end(state, pid, 1)
        for i in sib:
            if i != pid and i not in fo.outs:
                end(state, i, 0)
    except Exception:                                             # noqa: BLE001
        pass


def judgment_resolve(state: Path, pid: Optional[str], topic: str, subject: str, p_model: Optional[float], y_true: int,
                     repo: Optional[Path] = None) -> None:
    """After a judgment call: y = 1 when the model's answer is closer to the truth than the statistical predictor's on the same subject."""
    try:
        if pid is None or not enabled():
            return
        from creator import judgment as J
        c = _stat.get(topic)
        if c is None or time.time() - c[0] > 600:
            c = _stat[topic] = (time.time(), J._stat_preds(topic, Path(state), repo or Path(__file__).resolve().parents[1]))
        sp = c[1].get(subject)
        if sp is None:
            return
        closer = p_model is not None and abs(p_model - y_true) < abs(sp.p - y_true)
        end(Path(state), pid, int(closer))
    except Exception:                                             # noqa: BLE001
        pass


_stat: dict[str, tuple[float, Any]] = {}


def drill_prior_best(state: Path, source: str, digest: str) -> Optional[float]:
    rows = [r["select"]["brier"] for r in T._jsonl(Path(state) / "thinking" / "drill_runs.jsonl")
            if r.get("source") == source and r.get("digest") == digest and r.get("select", {}).get("n")]
    return min(rows) if rows else None


# ------------------------------------------------------------------------------------------------ the gate
def preds(state: Path) -> dict[str, list[T.Pred]]:
    """Per topic: walk-forward replay of the bootstrap (mode 'replay') plus resolved live predictions (mode 'live')."""
    out: dict[str, list[T.Pred]] = {t: [] for t in TOPICS}
    fo = fold(state)
    boot = bootstrap(state, "drill_beats_best")
    seen: list[tuple[float, str, int, float]] = []
    for t, k, y in boot:
        p, b, la = estimate(seen, k, at=t)
        out["drill_beats_best"].append(T.Pred("drill_beats_best", f"boot:{t:.0f}", t, p, b, la, y, "replay"))
        seen.append((t, k, y, 1.0))
    for i, o in fo.outs.items():
        r = fo.preds[i]
        out[r["topic"]].append(T.Pred(r["topic"], r["subject"], r["made_at"], r["p"], r["base"], r["last"], int(o["y"]), "live", {"key": r["key"]}))
    return out


def trust_section(state: Path) -> dict[str, Any]:
    """For trust.json: every fast topic through the SAME gate (thinking.score + thinking.trust_of), plus how many are waiting for their outcome."""
    fo = fold(state)
    ps = preds(state)
    out: dict[str, Any] = {}
    for t in TOPICS:
        sc = T.score(ps[t])
        ok, why = T.trust_of(sc) if sc.get("n") else (False, ["no scored predictions"])
        out[t] = {"trusted": ok, "why_not": why, "score": sc,
                  "pending": sum(1 for i, p in fo.preds.items() if p["topic"] == t and i not in fo.outs)}
    out["_rejected_outcome_lines"] = fo.rejected                   # type: ignore[assignment]
    return out
