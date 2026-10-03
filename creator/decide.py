"""TRUSTED PREDICTIONS IN NUPEN'S DECISIONS (owner, 3 Oct 2026: "none of this changes what Nupen does unless trusted predictions get wired
into its decisions, for example ranking packages by likely success or testing risky changes more deeply").

One question decides whether a prediction may act: `use(state, topic, fn)` returns fn() ONLY when the topic is trusted RIGHT NOW (the
trust report state/creator/trust.json - a protected path Nupen's workers cannot write - is fresh, says trusted, AND its stored score passes
thinking.trust_of again here), else None. Untrusted topics change nothing: every hook below returns its "no opinion" value and writes nothing,
so behaviour is exactly the old one.

SAFETY (the only two things a prediction may do):
  REORDER   schedule.schedule: ready gaps of the same priority class (capacity first, critical path first - unchanged) are ordered by
            predicted value per predicted cost (P(adopted) from 'verdict', expected seconds from 'duration'). Dependencies, the file/component
            exclusion and the slot count are applied afterwards exactly as before: a reorder can never start blocked or conflicting work.
  STRICTER  kernel.execute -> sandbox.evaluate: a candidate whose changed files carry a high predicted P(a fix touches them within 20 commits)
            ('git_fixed', top band ~25%) runs the FULL test suite instead of the affected-test selection. The selection only grows (every old
            test keeps its reason); an empty selection is never widened (empty = no report = rejected today, and more tests must not turn a
            rejection into a run). Nothing here can skip, shorten or weaken a check, change a verdict or touch the trust gate.

MEASUREMENT: every prediction-driven decision is a line in state/creator/thinking/decisions.jsonl with what it did AND what the old rule would
have done. A deterministic hash of the decision unit (package id; for a scheduling round its ready set and 10-minute bucket) puts HOLDOUT_SHARE
of the decisions on the old rule (arm 'holdout'), the rest act (arm 'treated'). `summary(state)` joins outcomes (adoption, regressions caught,
cycle seconds, wasted seconds) and compares the arms with 95% CIs; thinking.trust() copies it into trust.json['decisions'].
Kill switch: NUPEN_DECIDE_OFF=1. Loaded on demand (registry 'decide'); imports only the standard library at module level."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, TypeVar

X = TypeVar("X")
HOLDOUT_SHARE = 0.30                 # share of decisions kept on the old rule (the comparison arm)
TRUST_MAX_AGE_S = 12 * 3600.0        # an older trust report is not "trusted right now"
RISK_BAND = (0.20, 0.30)             # the stricter-testing band flags about this share of commits (chosen from the predictions' own spread)
RISK_TOPIC = "git_fixed"
RANK_TOPICS = ("verdict", "duration")
DECISIONS = "decisions.jsonl"


# ------------------------------------------------------------------------------------------------ the one "use a prediction?" helper
def _off() -> bool:
    return os.environ.get("NUPEN_DECIDE_OFF", "").strip() not in ("", "0")


def trust_report(state: Path, now: Optional[float] = None) -> Optional[dict[str, Any]]:
    """state/trust.json when it exists and is fresh, else None."""
    p = Path(state) / "trust.json"
    try:
        rep = json.loads(p.read_text(encoding="utf-8"))
        at = dt.datetime.fromisoformat(str(rep["at"]).replace("Z", "+00:00"))
        at = at if at.tzinfo else at.replace(tzinfo=dt.timezone.utc)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    t = time.time() if now is None else now
    return rep if isinstance(rep, dict) and 0 <= t - at.timestamp() <= TRUST_MAX_AGE_S + 300 else None


def _entry(rep: Mapping[str, Any], topic: str) -> Optional[tuple[bool, dict[str, Any]]]:
    """(the report's verdict, the score the gate judged) for a package topic, a drill source or a fast topic."""
    for sec, key in (("topics", "score"), ("drills", "heldout"), ("fast_topics", "score")):
        e = (rep.get(sec) or {}).get(topic)
        if isinstance(e, dict):
            return bool(e.get("trusted")), dict(e.get(key) or {})
    return None


def trusted_now(state: Path, topic: str, now: Optional[float] = None) -> bool:
    """The topic is trusted right now: fresh report, its verdict says trusted, and the same gate (thinking.trust_of) passes on its score."""
    if _off():
        return False
    rep = trust_report(state, now)
    e = _entry(rep, topic) if rep is not None else None
    if e is None or not e[0]:
        return False
    from creator import thinking as T                                  # on demand
    return T.trust_of(e[1])[0]


def use(state: Path, topic: str, fn: Callable[[], X], now: Optional[float] = None) -> Optional[X]:
    """The prediction (fn()) ONLY when its topic is trusted right now; otherwise None and fn is never called."""
    return fn() if trusted_now(state, topic, now) else None


def holdout(unit: str) -> bool:
    """Deterministic randomisation: True puts this decision on the OLD rule (comparison arm), for HOLDOUT_SHARE of units."""
    h = int(hashlib.sha256(f"nupen-decide:{unit}".encode()).hexdigest()[:12], 16)
    return h / float(16 ** 12) < HOLDOUT_SHARE


def log_path(state: Path) -> Path:
    return Path(state) / "thinking" / DECISIONS


def record(state: Path, row: dict[str, Any]) -> None:
    p = log_path(state)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"at": round(time.time(), 3), **row}, sort_keys=True, default=str) + "\n")


# ------------------------------------------------------------------------------------------------ REORDER: value per cost from trusted predictions
def rank_scores(state: Path, steps: Sequence[str], now: Optional[float] = None) -> Optional[dict[str, dict[str, float]]]:
    """step -> {'p_adopted', 'est_s'} for the trusted ranking topics only (a missing key = that topic is not trusted); None when none is."""
    use_v = trusted_now(state, "verdict", now)
    use_d = trusted_now(state, "duration", now)
    if not (use_v or use_d):
        return None
    from creator import thinking as T
    t = time.time() if now is None else now
    items = T.load_items(Path(state))
    tr = T._train(items, t)
    out: dict[str, dict[str, float]] = {}
    for s in sorted(set(steps)):
        d: dict[str, float] = {}
        if use_v:
            d["p_adopted"] = float(T.outcome_dist(tr, s)["ADOPTED"])
        if use_d:
            d["est_s"] = float(math.exp(T.duration_model(tr, T.Item("?", t, req=f"X.{s}"), False)[0]))
        out[s] = d
    return out


def ev_per_cost(importance: float, step_scores: Mapping[str, float], old_success: float, old_cost: float) -> float:
    """Expected value per expected second; a topic that is not trusted keeps the old estimate."""
    p = step_scores.get("p_adopted", old_success)
    c = step_scores.get("est_s", old_cost)
    return importance * p / max(c, 1.0)


# ------------------------------------------------------------------------------------------------ STRICTER: deeper testing for predicted risk
def _candidate_commit(files: Sequence[str], diff: str, message: str, t: float) -> dict[str, Any]:
    add = sum(1 for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
    dele = sum(1 for ln in diff.splitlines() if ln.startswith("-") and not ln.startswith("---"))
    return {"h": "candidate0", "t": t, "s": message, "files": {f.replace("\\", "/") for f in files}, "lines": add + dele, "add": add, "del": dele,
            "au": "", "body": ""}


def risk_band(ps: Sequence[float], band: tuple[float, float] = RISK_BAND) -> float:
    """The lowest P such that 'P >= it' flags a share of `ps` inside the band (the share closest to its middle); with ties that cannot
    land inside, the largest share below the band's top. inf when nothing can be flagged."""
    xs = sorted(ps, reverse=True)
    n = len(xs)
    if not n:
        return math.inf
    best, best_d = math.inf, math.inf
    fallback = math.inf
    for v in sorted(set(xs), reverse=True):
        share = sum(1 for x in xs if x >= v) / n
        if share <= band[1]:
            fallback = v
        if band[0] <= share <= band[1] and abs(share - sum(band) / 2) < best_d:
            best, best_d = v, abs(share - sum(band) / 2)
    return best if best < math.inf else fallback


def git_fixed_risk(state: Path, files: Sequence[str], diff: str, message: str, now: Optional[float] = None,
                   journal: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """P(a fix touches these files within FIX_WINDOW later commits) for the candidate, by the chosen git_fixed variant fit on the cached history
    (no git call), and the risk threshold from the same model's walk-forward predictions of the last 300 resolved commits."""
    from creator import drillsources as DS
    t = time.time() if now is None else now
    b = DS.best_variant(Path(state), RISK_TOPIC)
    commits = DS._read_cache(DS.cache_path(Path(state)))
    if b is None or not commits:
        return None
    commits = sorted(commits, key=lambda c: c["t"]) + [_candidate_commit(files, diff, message, max(t, commits[-1]["t"] + 1.0))]
    jr = journal if journal is not None else Path.home() / "Masterstock" / "JOURNAL.md"
    items = DS.apply_variant(DS._git_events(commits, DS.FIX_WINDOW, "fixed"), b["variant"], jr)
    v = b["variant"]
    cand = [p for p in DS.predict_open(items, RISK_TOPIC, v) if p.subject == "candidate0"]
    if not cand:
        return None
    hist = DS.walk_forward(items, RISK_TOPIC, float(v["decay"]), float(v["k"]), str(v.get("agg", "mean")), float(v.get("cap", 0.02)))[-300:]
    thr = risk_band([p.p for p in hist])
    return {"p": round(cand[0].p, 4), "threshold": round(thr, 4) if thr < math.inf else None, "variant": v, "n_ref": len(hist),
            "ref_flag_share": round(sum(1 for p in hist if p.p >= thr) / len(hist), 3) if hist else None,
            "high": bool(cand[0].p >= thr)}


def risk_gate(state: Path, package_id: str, files: Sequence[str], diff: str, message: str, now: Optional[float] = None,
              journal: Optional[Path] = None) -> str:
    """The kernel's call before the sandbox evaluation. '' = test as today. A non-empty reason = run the FULL suite (the treated arm of a
    high-risk candidate). Untrusted topic: '' and nothing is written. Any failure here also returns '' (today's testing, never less)."""
    try:
        risk = use(state, RISK_TOPIC, lambda: git_fixed_risk(state, files, diff, message, now, journal), now)
    except Exception as e:                                             # noqa: BLE001 - a broken predictor never changes a cycle
        record(state, {"kind": "risk", "unit": package_id, "error": f"{type(e).__name__}: {e}"[:300]})
        return ""
    if risk is None:
        return ""
    arm = "holdout" if holdout(package_id) else "treated"
    deeper = bool(risk["high"]) and arm == "treated"
    why = f"{RISK_TOPIC} P={risk['p']:.2f} >= {risk['threshold']} (top band)" if risk["high"] else ""
    record(state, {"kind": "risk", "unit": package_id, "arm": arm, "topic": RISK_TOPIC, "prediction": risk, "files": sorted(files)[:50],
                   "did": "full_suite" if deeper else "selection", "old_rule": "selection", "changed": deeper, "flagged": bool(risk["high"])})
    return why if deeper else ""


# ------------------------------------------------------------------------------------------------ outcomes and the arm comparison
def _jsonl(p: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if p.is_file():
        for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                x = json.loads(ln)
            except ValueError:
                continue
            if isinstance(x, dict):
                out.append(x)
    return out


def _rate(xs: Sequence[int]) -> dict[str, Any]:
    """Proportion with a Wilson 95% interval."""
    n = len(xs)
    if not n:
        return {"n": 0}
    p, z = sum(xs) / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {"n": n, "rate": round(p, 4), "ci95": [round(c - h, 4), round(c + h, 4)]}


def _diff(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    """mean(a) - mean(b) with a normal-approximation 95% CI (Welch standard error)."""
    if len(a) < 2 or len(b) < 2:
        return {"n": [len(a), len(b)]}
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    se = math.sqrt(va / len(a) + vb / len(b))
    d = ma - mb
    return {"n": [len(a), len(b)], "diff": round(d, 4), "ci95": [round(d - 1.96 * se, 4), round(d + 1.96 * se, 4)]}


def _outcomes(state: Path) -> dict[str, dict[str, Any]]:
    """package -> outcome facts from kernel_log.jsonl (the last line per package)."""
    out: dict[str, dict[str, Any]] = {}
    for k in _jsonl(Path(state) / "kernel_log.jsonl"):
        if k.get("package"):
            det = k.get("details") or {}
            out[k["package"]] = {"adopted": int(k.get("outcome") == "ADOPTED"), "seconds": float(k.get("seconds") or 0.0),
                                 "regression_caught": int(det.get("regression") not in (None, "", "CLEAN") or bool(det.get("weakening"))),
                                 "rolled_back": int("rollback" in str(k.get("reason", "")).lower()), "req": k.get("requirement", ""),
                                 "outcome": k.get("outcome")}
    return out


def _arms(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by: dict[str, list[dict[str, Any]]] = {"treated": [], "holdout": []}
    for r in rows:
        by.setdefault(r["arm"], []).append(r)
    res: dict[str, Any] = {}
    for arm, rs in by.items():
        res[arm] = {"n": len(rs), "adoption": _rate([r["adopted"] for r in rs]),
                    "regression_caught": _rate([r["regression_caught"] for r in rs]), "rolled_back": _rate([r["rolled_back"] for r in rs]),
                    "mean_cycle_s": round(sum(r["seconds"] for r in rs) / len(rs), 1) if rs else None,
                    "wasted_s": round(sum(r["seconds"] for r in rs if not r["adopted"]), 1)}
    t, h = by["treated"], by["holdout"]
    res["treated_minus_holdout"] = {m: _diff([float(r[m]) for r in t], [float(r[m]) for r in h])
                                    for m in ("adopted", "regression_caught", "rolled_back", "seconds")}
    return res


def summary(state: Path) -> dict[str, Any]:
    """The section other reports read: decisions per kind and arm, and outcome comparisons (treated vs holdout) with 95% CIs. For 'risk' only
    FLAGGED candidates are compared (the arms differ only there); for 'rank' each round's first pick is joined to the package that was
    written for its requirement at or after the decision."""
    state = Path(state)
    rows = [r for r in _jsonl(log_path(state)) if r.get("arm") in ("treated", "holdout")]
    outc = _outcomes(state)
    out: dict[str, Any] = {"holdout_share": HOLDOUT_SHARE, "decisions": len(rows), "log": str(log_path(state))}
    risk = [r for r in rows if r.get("kind") == "risk"]
    joined = [{**outc[r["unit"]], "arm": r["arm"]} for r in risk if r.get("flagged") and r["unit"] in outc]
    out["risk"] = {"decisions": len(risk), "flagged": sum(1 for r in risk if r.get("flagged")),
                   "made_stricter": sum(1 for r in risk if r.get("changed")), "resolved_flagged": len(joined), "arms": _arms(joined)}
    rank = [r for r in rows if r.get("kind") == "rank"]
    jr: list[dict[str, Any]] = []
    if rank:
        try:
            from creator import thinking as T
            items = T.load_items(state)
        except Exception:                                                # noqa: BLE001 - no ledger: nothing to join
            items = []
        for r in rank:
            req = (r.get("picks_req") or [None])[0]
            it = next((i for i in items if i.req == req and i.created >= float(r.get("at", 0)) - 60 and i.pkg in outc), None)
            if it is not None:
                jr.append({**outc[it.pkg], "arm": r["arm"]})
    out["rank"] = {"decisions": len(rank), "changed": sum(1 for r in rank if r.get("changed")), "resolved": len(jr), "arms": _arms(jr)}
    return out
