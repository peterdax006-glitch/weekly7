"""THE LEARNING LOOP: test on the data, measure what was learned and kept, diagnose the misses, improve, re-measure (owner, 3 Oct 2026: "we are
having it collect all this data which is great but if we arent testing on the data to see what it remembers and then improving the system
afterward what good was collecting the data in the first place").

One `run()` (a thinking-focus filler job every LEARN_EVERY_S; the frozen benchmark at most every BENCH_EVERY_S when RAM allows) does:
  1 PROGRESS    every skill's score with its 95% CI, time-stamped, appended to thinking/progress.jsonl: drill held-out gain per source, drill live
                (prospective) gain, the thinking topics, judgment per topic (and vs the statistical predictor), the fast prospective topics, the
                anticipation rate, any plug-in skill (h27 reasoning drills: a trust.json section or thinking/skill_scores/*.json), and every frozen
                benchmark run per part and model role. `curve()` classifies each skill's series improving / plateau / regressing (late third vs
                early third with CIs; the benchmark PAIRED on its frozen items) -> thinking/progress_curve.json.
  2 RETENTION   rolling probes (thinking/retention.jsonl): FRESH items - new since the last probe and in the walk-forward's held-out tail, so no variant
                selection has used them - are scored with the current predictor BEFORE any search learns from them; a newly acquired repository's items are
                probed the same way (score before trained on). Later the same items are re-scored (paired) under whatever predictor exists then:
                kept gains, forgetting and transfer (fresh vs held-out gain) are numbers.
  3 DIAGNOSIS   from the recorded predictions: where the score is lost (excess Brier over the running base rate) by source, feature family, outcome
                calibration bucket, confident predictions (p <= 0.2 or >= 0.8: overconfidence shows as loss there), judgment topic/model, prospective topic/key, and regressing skills; ranked by the
                conservative loss (n x CI lower bound) -> thinking/diagnosis.json.
  4 IMPROVE     the top actionable weakness becomes (a) a set of drill variants queued for the EXISTING measured search (thinking/learn_queue.jsonl,
                consumed by drillsources.drill_filler: selection still only on the select part, held-out never chooses), or (b) a goal proposal
                (goals 'weakness' source, module creator/goal_think_*.py so the thinking focus allows it) carrying the evidence and the metric that
                must move. Every improvement is re-measured later and gets a verdict in thinking/improvements.jsonl: helped / no_effect /
                hurt_flag_revert / inconclusive / rejected - never silently.
  5 NEED DATA   when the limit is data (search converged and no fresh items) an event goes to thinking/need_data.jsonl (+ need_data.json, latest)
                for the acquisition job to read.
MEASURING CODE IS READ-ONLY HERE: this module never writes trust.json, the ledger, the benchmark directory or any scoring code; every write goes
through `_out()`, which only knows the files in FILES (plus the goal proposal log through goals' own append)."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from creator import thinking as T

LEARN_EVERY_S = 1800.0                 # cheap parts (the swarm filler's cadence)
BENCH_EVERY_S = 6 * 3600.0             # the frozen benchmark (~10 min, model servers)
BENCH_MIN_FREE_GB = 8.0                # ... only when this much RAM is free
TRUST_STALE_S = 2 * 3600.0             # trust.json older than this is recomputed (read-only) instead of read
MIN_GROUP_N = 15                       # a diagnosis group needs this many scored predictions
FRESH_WINDOW_S = 86400.0             # new resolved items in this window mean the search still has new data to learn from
MIN_FRESH = 5                          # a fresh probe needs this many never-selected-on items
PROBE_MAX = 200
RESCORE_AFTER_S = 1800.0
RESCORE_STALE_S = 86400.0              # an unchanged predictor's probe is re-scored once a day (new history can still move its predictions)
VERIFY_AFTER_S = 2 * 3600.0            # an improvement is judged no earlier than this (and once its queued variants have run)
NEED_DATA_EVERY_S = 6 * 3600.0
MAX_OPEN_GOALS = 3                     # learn-loop goal proposals pending at once (the teacher builds them; more would only queue)
DIAG_SOURCES = ("git_fixed", "git_churn", "journal_persist", "plan_choice", "x_git_fixed", "x_git_churn")   # research_bool: a 60k-file walk, left out
FILES = {"progress": "progress.jsonl", "curve": "progress_curve.json", "retention": "retention.jsonl", "diagnosis": "diagnosis.json",
         "improvements": "improvements.jsonl", "queue": "learn_queue.jsonl", "need_data": "need_data.jsonl", "need_data_now": "need_data.json",
         "run": "learnloop_run.json", "experiments": "experiments.json"}


# ------------------------------------------------------------------------------------------------ io (the only writer)
def _out(state: Path, key: str) -> Path:
    return Path(state) / "thinking" / FILES[key]


def _append(state: Path, key: str, recs: Iterable[dict[str, Any]]) -> int:
    rows = list(recs)
    if rows:
        p = _out(state, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")
    return len(rows)


def _write(state: Path, key: str, obj: Any) -> None:
    p = _out(state, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(p)


def _read(state: Path, key: str) -> list[dict[str, Any]]:
    return T._jsonl(_out(state, key))


def _iso(t: float) -> str:
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat(timespec="seconds")


def _ts(s: Any) -> float:
    try:
        return dt.datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def mean_ci(xs: Sequence[float]) -> list[float]:
    """[mean, lo, hi] (normal 95%); a single value has an infinite interval."""
    n = len(xs)
    if not n:
        return [0.0, -math.inf, math.inf]
    m = sum(xs) / n
    if n < 2:
        return [m, -math.inf, math.inf]
    se = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n)
    return [m, m - 1.96 * se, m + 1.96 * se]


def _r(x: Optional[float], nd: int = 4) -> Optional[float]:
    return None if x is None or not math.isfinite(x) else round(x, nd)


def wilson(k: int, n: int) -> list[float]:
    if not n:
        return [0.0, 1.0]
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


# ------------------------------------------------------------------------------------------------ 1 progress: what every skill scores now
def trust_report(state: Path, now: float) -> dict[str, Any]:
    """The latest trust.json (written by thinking.run every 10 min) or, when stale/missing, the same report recomputed WITHOUT writing it."""
    p = Path(state) / "trust.json"
    try:
        tj = json.loads(p.read_text(encoding="utf-8"))
        if now - _ts(tj.get("at")) <= TRUST_STALE_S:
            return dict(tj)
    except (OSError, ValueError):
        pass
    return T.trust(Path(state), write=False)


def _score_row(skill: str, sc: Any, kind: str, at: Any, extra: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
    if not isinstance(sc, dict) or not sc.get("n") or sc.get("gain_vs_best") is None:
        return None
    return {"skill": skill, "kind": kind, "n": sc["n"], "value": sc.get("gain_vs_best"), "ci": sc.get("gain_ci95"), "brier": sc.get("brier"),
            "ece": sc.get("ece"), "n_live": sc.get("n_live"), "measured_at": at, **(extra or {})}


KNOWN_SECTIONS = ("anticipation", "at", "topics", "drills", "judgment", "fast_topics", "drills_error", "live", "experiments")


def skill_rows(tj: dict[str, Any]) -> list[dict[str, Any]]:
    """Every scored skill in a trust report, as {skill, value (gain over the best baseline, higher = better), ci, n, ...}."""
    at = tj.get("at")
    rows: list[Optional[dict[str, Any]]] = []
    for t, v in (tj.get("topics") or {}).items():
        rows.append(_score_row(f"thinking:{t}", (v or {}).get("score"), "replay+live", at))
    for s, v in (tj.get("drills") or {}).items():
        if isinstance(v, dict):
            rows.append(_score_row(f"drill:{s}:heldout", v.get("heldout"), "heldout", at, {"variant": v.get("variant"), "search": v.get("search")}))
            rows.append(_score_row(f"drill:{s}:live", v.get("live"), "prospective", at))
    for t, v in (tj.get("judgment") or {}).items():
        if not isinstance(v, dict):
            continue
        rows.append(_score_row(f"judgment:{t}", v.get("judge_calibrated"), "judge", at, {"strategy": v.get("best_strategy")}))
        g = v.get("gain_vs_statistical")
        if isinstance(g, list) and len(g) == 3:
            rows.append({"skill": f"judgment:{t}:vs_statistical", "kind": "judge_vs_statistical", "n": v.get("n"), "value": g[0], "ci": g[1:],
                         "measured_at": at})
    for t, v in (tj.get("fast_topics") or {}).items():
        if isinstance(v, dict):
            rows.append(_score_row(f"fast:{t}", v.get("score"), "prospective", at))
    ant = tj.get("anticipation") or {}
    if ant.get("eligible_directives"):
        k, n = int(ant.get("anticipated") or 0), int(ant["eligible_directives"])
        rows.append({"skill": "anticipation:rate", "kind": "anticipation", "n": n, "value": round(k / n, 4), "ci": wilson(k, n), "measured_at": at})
    dr = ant.get("drill") or {}
    if dr.get("n") and dr.get("gain") is not None:
        rows.append({"skill": "anticipation:drill", "kind": "anticipation", "n": dr["n"], "value": dr["gain"], "ci": dr.get("gain_ci95"), "measured_at": at})
    for sec, body in tj.items():                                   # PLUG-INS (h27 reasoning drills etc.): any other section of scored entries
        if sec in KNOWN_SECTIONS or not isinstance(body, dict):
            continue
        for name, v in body.items():
            if isinstance(v, dict):
                sc = v.get("score") if isinstance(v.get("score"), dict) else v.get("heldout") if isinstance(v.get("heldout"), dict) else v
                rows.append(_score_row(f"{sec}:{name}", sc, "plugin", at))
    return [r for r in rows if r is not None]


def plugin_rows(state: Path) -> list[dict[str, Any]]:
    """thinking/skill_scores/*.json: {"at": iso, "skills": {name: {"n", "value", "ci"}}} - the simple reader other jobs plug in through."""
    d = Path(state) / "thinking" / "skill_scores"
    out: list[dict[str, Any]] = []
    for f in sorted(d.glob("*.json")) if d.is_dir() else []:
        try:
            obj = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for name, v in (obj.get("skills") or {}).items():
            if isinstance(v, dict) and v.get("n") and v.get("value") is not None:
                out.append({"skill": f"{f.stem}:{name}", "kind": "plugin", "n": v["n"], "value": v["value"], "ci": v.get("ci"), "measured_at": obj.get("at")})
    return out


def bench_dir(state: Path) -> Path:
    return Path(state) / "thinkbench"


def bench_runs(state: Path) -> list[tuple[str, dict[str, Any]]]:
    """Every saved frozen-benchmark run (baseline_* / now_*), oldest first. Read only."""
    out = []
    d = bench_dir(state)
    for f in sorted(d.glob("*_2*.json")) if d.is_dir() else []:
        if not f.name.startswith(("baseline_", "now_")):
            continue
        try:
            res = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append((f.name, res))
    out.sort(key=lambda x: _ts(x[1].get("at")))
    return out


def bench_role(res: dict[str, Any]) -> str:
    return str(((res.get("versions") or {}).get("local_model") or {}).get("role") or "fast")


def bench_rows(state: Path, seen: set[str]) -> list[dict[str, Any]]:
    rows = []
    for name, res in bench_runs(state):
        if name in seen:
            continue
        role = bench_role(res)
        for part, p in (res.get("parts") or {}).items():
            if isinstance(p, dict) and p.get("n") and p.get("gain") is not None:
                # part a uses no model: its series is the predictor's, whatever the role
                skill = f"bench:{part}" if part == "a_prediction" else f"bench:{part}:{role}"
                rows.append({"skill": skill, "kind": "bench", "n": p["n"], "value": p["gain"], "ci": p.get("gain_ci95"), "score": p.get("score"),
                             "baseline": p.get("baseline"), "measured_at": res.get("at"), "bench_file": name, "items_hash": res.get("items_hash")})
    return rows


def drill_history(state: Path) -> list[dict[str, Any]]:
    """The drills' past, replayed from drill_runs.jsonl (file order = completion order): after every run, the variant the search would have chosen
    then (best on the select part at that row's data) and its held-out score - a learning curve from before this loop existed. The held-out
    tail grows with the data, so successive points are NOT paired."""
    from creator import drillsources as D
    best: dict[tuple[str, str], dict[str, Any]] = {}
    last: dict[str, Any] = {}
    out = []
    for r in T._jsonl(D.runs_path(Path(state))):
        s, dg, sel = r.get("source"), r.get("digest"), (r.get("select") or {})
        if not s or not sel.get("n") or not (r.get("heldout") or {}).get("n"):
            continue
        k = (str(s), str(dg))
        if k not in best or sel["brier"] < best[k]["select"]["brier"]:
            best[k] = r
        h = best[k]["heldout"]
        if last.get(s) == (dg, h.get("gain_vs_best"), h.get("n")):
            continue
        last[s] = (dg, h.get("gain_vs_best"), h.get("n"))
        row = _score_row(f"drill:{s}:heldout", h, "heldout", r.get("at"), {"variant": best[k]["variant"], "backfill": True})
        if row is not None:
            out.append(row)
    return out


def measure(state: Path, now: float, tj: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
    """Append the skills whose measurement changed since the last record (the first time also the drills' replayed past); returns the new rows."""
    prev = _read(state, "progress")
    if not prev:
        _append(state, "progress", [{"at": _iso(now), "t": now, **r} for r in drill_history(state)])
        prev = _read(state, "progress")
    last: dict[str, dict[str, Any]] = {}
    for r in prev:
        last[r["skill"]] = r
    seen_bench = {r["bench_file"] for r in prev if r.get("bench_file")}
    tj = tj if tj is not None else trust_report(state, now)
    new = []
    for r in skill_rows(tj) + plugin_rows(state):
        lr = last.get(r["skill"])
        if lr is not None and (lr.get("measured_at") == r.get("measured_at") or (lr.get("value") == r.get("value") and lr.get("n") == r.get("n"))):
            continue
        new.append({"at": _iso(now), "t": now, **r})
    new += [{"at": _iso(now), "t": now, **r} for r in bench_rows(state, seen_bench)]
    _append(state, "progress", new)
    return new


def classify(points: Sequence[tuple[float, float, Optional[Sequence[Any]]]], min_points: int = 3) -> dict[str, Any]:
    """points = (time, value, ci95 or None), value higher = better. Late third vs early third of the series; each third's uncertainty is the mean
    standard error of its points (successive measurements overlap in data, so they are NOT treated as independent), or the series' own scatter.
    improving: the change's 95% lower bound > 0; regressing: upper bound < 0; else plateau."""
    pts = sorted((float(t), float(v), ci) for t, v, ci in points if v is not None and math.isfinite(float(v)))
    if len(pts) < min_points:
        return {"trend": "insufficient", "points": len(pts)}
    k = max(1, len(pts) // 3)
    early, late = pts[:k], pts[-k:]

    def se_of(g: Sequence[tuple[float, float, Any]]) -> Optional[float]:
        s = [(float(ci[1]) - float(ci[0])) / 3.92 for _t, _v, ci in g
             if ci and len(ci) == 2 and ci[0] is not None and ci[1] is not None and math.isfinite(float(ci[0])) and math.isfinite(float(ci[1]))]
        return sum(s) / len(s) if s else None
    vals = [v for _t, v, _c in pts]
    mv = sum(vals) / len(vals)
    scatter = math.sqrt(sum((v - mv) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
    se_e, se_l = se_of(early), se_of(late)
    se_e = scatter / math.sqrt(k) if se_e is None else se_e
    se_l = scatter / math.sqrt(k) if se_l is None else se_l
    d = sum(v for _t, v, _c in late) / k - sum(v for _t, v, _c in early) / k
    s = math.sqrt(se_e ** 2 + se_l ** 2)
    lo, hi = d - 1.96 * s, d + 1.96 * s
    trend = "improving" if lo > 0 else "regressing" if hi < 0 else "plateau"
    span = pts[-1][0] - pts[0][0]
    return {"trend": trend, "points": len(pts), "first": _r(pts[0][1]), "last": _r(pts[-1][1]), "change": _r(d), "change_ci95": [_r(lo), _r(hi)],
            "span_h": round(span / 3600.0, 2), "paired": False}


def _paired(base: dict[str, Any], now: dict[str, Any]) -> dict[str, Any]:
    """The benchmark's own paired comparison (creator.thinkbench.compare_part), read-only; a minimal copy when that module cannot load."""
    try:
        from creator import registry as REG
        return dict(REG.get("thinkbench").compare_part(base, now))
    except Exception:                                               # noqa: BLE001
        bi, ni = base.get("per_item") or {}, now.get("per_item") or {}
        ids = sorted(set(bi) & set(ni))
        sign = 1.0 if now.get("higher_is_better") else -1.0
        m = mean_ci([sign * (ni[i] - bi[i]) for i in ids])
        v = "SIGNIFICANTLY BETTER" if m[1] > 0 else "SIGNIFICANTLY WORSE" if m[2] < 0 else "no significant change"
        return {"n_paired": len(ids), "change": _r(m[0]), "change_ci95": [_r(m[1]), _r(m[2])], "verdict": v}


def bench_curve(state: Path) -> dict[str, dict[str, Any]]:
    """Per benchmark part and model role: the first and the latest run compared PAIRED on the frozen items (and the latest vs the one before)."""
    by: dict[str, list[dict[str, Any]]] = {}
    for _name, res in bench_runs(state):
        role = bench_role(res)
        for part, p in (res.get("parts") or {}).items():
            if isinstance(p, dict) and p.get("n"):
                by.setdefault(f"bench:{part}" if part == "a_prediction" else f"bench:{part}:{role}", []).append(p)
    out: dict[str, dict[str, Any]] = {}
    for skill, ps in by.items():
        if len(ps) < 2:
            out[skill] = {"trend": "insufficient", "points": len(ps), "paired": True}
            continue
        c = _paired(ps[0], ps[-1])
        v = str(c.get("verdict", ""))
        trend = "improving" if v.startswith("SIGNIFICANTLY BETTER") else "regressing" if v.startswith("SIGNIFICANTLY WORSE") else "plateau"
        out[skill] = {"trend": trend, "points": len(ps), "paired": True, "first_vs_last": c, "previous_vs_last": _paired(ps[-2], ps[-1]),
                      "first": ps[0].get("gain"), "last": ps[-1].get("gain"), "change": c.get("change"), "change_ci95": c.get("change_ci95")}
    return out


def curve(state: Path, now: float) -> dict[str, Any]:
    series: dict[str, list[tuple[float, float, Any]]] = {}
    for r in _read(state, "progress"):
        if r.get("kind") != "bench" and r.get("value") is not None:
            series.setdefault(r["skill"], []).append((_ts(r.get("measured_at")) or float(r.get("t", 0.0)), float(r["value"]), r.get("ci")))
    skills = {s: classify(pts) for s, pts in series.items()}
    skills.update(bench_curve(state))
    groups: dict[str, list[str]] = {"improving": [], "plateau": [], "regressing": [], "insufficient": []}
    for s, c in sorted(skills.items()):
        groups[c["trend"]].append(s)
    answer = (f"Is Nupen learning from the data? {len(groups['improving'])} skills improving, {len(groups['plateau'])} on a plateau, "
              f"{len(groups['regressing'])} regressing ({len(groups['insufficient'])} with too few measurements). Regressing: "
              f"{', '.join(groups['regressing']) or 'none'}.")
    rep = {"at": _iso(now), "answer": answer, "summary": groups, "skills": skills}
    _write(state, "curve", rep)
    return rep


# ------------------------------------------------------------------------------------------------ predictions with their items (shared)
class Data:
    """Per drill source: items, the best variant, its walk-forward predictions aligned to the items (loaded once per run)."""

    def __init__(self, state: Path, repo: Path, journal: Path, research: Path) -> None:
        self.state, self.repo, self.journal, self.research = Path(state), Path(repo), Path(journal), Path(research)
        self._c: dict[str, Optional[dict[str, Any]]] = {}

    def get(self, source: str) -> Optional[dict[str, Any]]:
        if source not in self._c:
            self._c[source] = self._load(source)
        return self._c[source]

    def _load(self, source: str) -> Optional[dict[str, Any]]:
        from creator import drillsources as D
        b = D.best_variant(self.state, source)
        if b is None:
            return None
        try:
            raw = D.load(source, self.state, self.repo, self.journal, self.research)
        except Exception:                                           # noqa: BLE001 - a source that cannot be read now is skipped this round
            return None
        return {"items": raw, "variant": b["variant"], "preds": predict(raw, source, b["variant"], self.journal)}


def predict(raw: Sequence[Any], source: str, variant: dict[str, Any], journal: Path) -> list[tuple[Any, T.Pred]]:
    """(raw item, walk-forward prediction under `variant`) for every resolved item. The prediction of an item uses only items resolved strictly
    before it was created (drillsources.walk_forward); the raw item keeps the unmasked feature keys for the diagnosis."""
    from creator import drillsources as D
    items = D.apply_variant(list(raw), variant, journal)
    preds = D.walk_forward(items, source, float(variant["decay"]), float(variant["k"]), str(variant.get("agg", "mean")), float(variant.get("cap", 0.02)))
    by: dict[tuple[str, float], list[Any]] = {}
    for it in raw:
        if it.resolved is not None:
            by.setdefault((it.subject, it.created), []).append(it)
    out = []
    for p in preds:
        lst = by.get((p.subject, p.made_at))
        if lst:
            out.append((lst.pop(0), p))
    return out


def last_run_time(state: Path, source: str) -> float:
    """When a drill run of this source last finished: items resolved after it were never seen by any variant selection."""
    from creator import drillsources as D
    ts = [_ts(r.get("at")) for r in T._jsonl(D.runs_path(Path(state))) if r.get("source") == source and r.get("select", {}).get("n")]
    return max(ts) if ts else 0.0


# ------------------------------------------------------------------------------------------------ 2 retention
def _probe_id(source: str, kind: str, subjects: Sequence[str]) -> str:
    return "RP-" + hashlib.sha1(f"{source}|{kind}|{'|'.join(subjects)}".encode()).hexdigest()[:10]


def _brier_of(pairs: Sequence[tuple[Any, T.Pred]], subjects: set[str]) -> dict[str, list[float]]:
    return {f"{it.subject}@{it.created}": [p.p, float(p.outcome or 0), p.base] for it, p in pairs if it.subject in subjects}


def select_cut(pairs: Sequence[tuple[Any, T.Pred]]) -> float:
    """Creation time of the last item in the SELECT part (the first SELECT_SPLIT of the time order): the search chooses variants on those items
    only, so anything created later has never been learned from by a selection."""
    from creator import drillsources as D
    cut = int(len(pairs) * D.SELECT_SPLIT)
    return float(pairs[cut - 1][1].made_at) if cut > 0 else -math.inf


def fresh_items(pairs: Sequence[tuple[Any, T.Pred]], learned_until: float, probed: set[str]) -> list[tuple[Any, T.Pred]]:
    """Never learned from by a variant selection (created after the select part's last item) and not in any probe yet."""
    return [(it, p) for it, p in pairs if it.resolved is not None and p.made_at > learned_until and f"{it.subject}@{it.created}" not in probed]


def retention(state: Path, data: Data, now: float, sources: Sequence[str]) -> dict[str, Any]:
    """New probes (fresh items, new repositories) scored BEFORE they are learned from, and paired re-scores of old probes."""
    from creator import drillsources as D
    ev = _read(state, "retention")
    probes = {e["id"]: e for e in ev if e.get("event") == "probe"}
    last_rescore: dict[str, dict[str, Any]] = {}
    for e in ev:
        if e.get("event") == "rescore":
            last_rescore[e["probe"]] = e
    probed = {k for e in probes.values() for k in e.get("scores", {})}
    known_repos = {e.get("repo") for e in probes.values() if e.get("kind") == "repo"} | {e.get("repo") for e in ev if e.get("event") == "repo_seen"}
    new: list[dict[str, Any]] = []
    for s in sources:
        d = data.get(s)
        if d is None:
            continue
        pairs, variant = d["preds"], d["variant"]
        lu = select_cut(pairs)
        if D.SOURCES.get(s) == "gitx":                                 # a newly acquired repository: probe it before any run trained on it
            repos = sorted({k[2:] for it, _p in pairs for k in it.keys if k.startswith("r:")})
            for repo in repos:
                if repo in known_repos:
                    continue
                try:
                    built = _repo_cache_mtime(repo)
                except (OSError, ImportError):
                    built = 0.0
                if built and built > last_run_time(state, s):          # its history appeared after the last selection: score it untrained
                    rp = [(it, p) for it, p in pairs if f"r:{repo}" in it.keys][-PROBE_MAX:]
                    if len(rp) >= MIN_FRESH:
                        new.append(_probe(s, "repo", rp, variant, lu, now, repo=repo))
                        known_repos.add(repo)
                        continue
                new.append({"event": "repo_seen", "source": s, "repo": repo, "at": _iso(now), "note": "already trained on when first seen: no before-score"})
                known_repos.add(repo)
        fr = fresh_items(pairs, lu, probed)[-PROBE_MAX:]
        if len(fr) >= MIN_FRESH:
            new.append(_probe(s, "fresh", fr, variant, lu, now))
            probed |= {f"{it.subject}@{it.created}" for it, _p in fr}
    for pid, pr in probes.items():                                     # re-score old probes under today's predictor (paired, same items)
        d = data.get(pr["source"])
        if d is None or now - _ts(pr["at"]) < RESCORE_AFTER_S:
            continue
        lr = last_rescore.get(pid)
        sig = json.dumps(d["variant"], sort_keys=True)
        if lr is not None and (lr.get("sig") == sig and now - _ts(lr["at"]) < RESCORE_STALE_S or now - _ts(lr["at"]) < RESCORE_AFTER_S):
            continue                                                   # the predictor has not changed (re-checked daily as data grows)
        now_s = _brier_of(d["preds"], {k.split("@")[0] for k in pr["scores"]})
        both = [k for k in pr["scores"] if k in now_s]
        if not both:
            continue
        diff = [T.brier(pr["scores"][k][0], int(pr["scores"][k][1])) - T.brier(now_s[k][0], int(now_s[k][1])) for k in both]   # > 0: better now
        m = mean_ci(diff)
        new.append({"event": "rescore", "probe": pid, "source": pr["source"], "kind": pr["kind"], "at": _iso(now), "sig": sig, "variant": d["variant"],
                    "n": len(both), "gain_since_probe": _r(m[0]), "gain_ci95": [_r(m[1]), _r(m[2])],
                    "brier_now": _r(sum(T.brier(now_s[k][0], int(now_s[k][1])) for k in both) / len(both)),
                    "selected_since": d["variant"] != pr["variant"]})
    _append(state, "retention", new)
    return summary_retention(_read(state, "retention"))


def _repo_cache_mtime(repo_name: str) -> float:
    from creator import device as DEV
    return (DEV.runtime_dir() / "thinking" / f"git_history_x_{repo_name}.jsonl").stat().st_mtime


def _probe(source: str, kind: str, pairs: Sequence[tuple[Any, T.Pred]], variant: dict[str, Any], learned_until: float, now: float,
           repo: str = "") -> dict[str, Any]:
    scores = {f"{it.subject}@{it.created}": [p.p, int(p.outcome or 0), p.base] for it, p in pairs}
    g = [T.brier(v[2], int(v[1])) - T.brier(v[0], int(v[1])) for v in scores.values()]
    m = mean_ci(g)
    subj = sorted(scores)
    return {"event": "probe", "id": _probe_id(source, kind, subj), "source": source, "kind": kind, "repo": repo, "at": _iso(now),
            "learned_until": _iso(learned_until) if learned_until > 0 else None, "variant": variant, "n": len(scores), "scores": scores,
            "brier": _r(sum(T.brier(v[0], int(v[1])) for v in scores.values()) / len(scores)), "gain_vs_base": _r(m[0]), "gain_ci95": [_r(m[1]), _r(m[2])],
            "never_selected_on": all(bool(p.made_at > learned_until) for _it, p in pairs) if kind == "fresh" else None}


def summary_retention(ev: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per source: fresh-item gain (the before-learning test: does it transfer to data nothing was selected on?) and the paired change of
    re-scored probes (kept / forgot)."""
    out: dict[str, Any] = {}
    latest: dict[str, dict[str, Any]] = {}
    for e in ev:
        if e.get("event") == "rescore":
            latest[e["probe"]] = e
    for e in ev:
        if e.get("event") != "probe":
            continue
        o = out.setdefault(e["source"], {"probes": 0, "fresh_n": 0, "fresh_gain": [], "rescored": 0, "kept": [], "repos": []})
        o["probes"] += 1
        o["fresh_n"] += e["n"]
        o["fresh_gain"].append(e.get("gain_vs_base") or 0.0)
        if e.get("kind") == "repo":
            o["repos"].append({"repo": e.get("repo"), "before_gain": e.get("gain_vs_base"), "before_ci95": e.get("gain_ci95"),
                               "after": (latest.get(e["id"]) or {}).get("gain_since_probe")})
        r = latest.get(e["id"])
        if r is not None:
            o["rescored"] += 1
            o["kept"].append(r.get("gain_since_probe") or 0.0)
    for o in out.values():
        o["fresh_gain_mean"] = _r(sum(o["fresh_gain"]) / len(o["fresh_gain"])) if o["fresh_gain"] else None
        o["change_since_probe_mean"] = _r(sum(o["kept"]) / len(o["kept"])) if o["kept"] else None
        o["verdict"] = ("no re-score yet" if not o["kept"] else "forgetting: probes score worse now" if o["change_since_probe_mean"] < -0.002
                        else "kept or improved")
        del o["fresh_gain"], o["kept"]
    return out


# ------------------------------------------------------------------------------------------------ 3 diagnosis: where the score is lost
def _group(acc: dict[tuple[str, str], dict[str, Any]], kind: str, where: str, loss: float, p: float, y: int, extra: Optional[dict[str, Any]] = None) -> None:
    g = acc.setdefault((kind, where), {"kind": kind, "where": where, "losses": [], "ps": [], "ys": [], **(extra or {})})
    g["losses"].append(loss)
    g["ps"].append(p)
    g["ys"].append(y)


def drill_groups(source: str, pairs: Sequence[tuple[Any, T.Pred]], acc: dict[tuple[str, str], dict[str, Any]]) -> None:
    """Excess Brier over the running base rate per item (> 0 = score lost there), grouped by source, feature family=value (key position i, which
    the search's mask can drop), outcome class, calibration bucket and overconfident misses."""
    for it, p in pairs:
        y = int(p.outcome or 0)
        loss = T.brier(p.p, y) - T.brier(p.base, y)
        _group(acc, "source", source, loss, p.p, y, {"source": source})
        for i, k in enumerate(it.keys):
            fam = k.split(":", 1)[0] if ":" in k else k.split("|", 1)[0]
            _group(acc, "feature", f"{source}|{k}", loss, p.p, y, {"source": source, "pos": i, "family": fam})
        _group(acc, "calibration", f"{source}|p{min(4, int(p.p * 5))}", loss, p.p, y, {"source": source})
        if abs(p.p - 0.5) >= 0.3:                                  # confident predictions (selected on p only, never on the outcome)
            _group(acc, "confident", f"{source}|{'high' if p.p >= 0.5 else 'low'}", loss, p.p, y, {"source": source})


def judgment_groups(state: Path, repo: Path, tj: dict[str, Any], acc: dict[tuple[str, str], dict[str, Any]]) -> None:
    """The local model's calibrated answers (its best strategy) vs the statistical predictor on the same subjects: loss = judge - statistical."""
    try:
        from creator import judgment as J
    except Exception:                                               # noqa: BLE001
        return
    rows = T._jsonl(J.path(Path(state)))
    for topic, sec in (tj.get("judgment") or {}).items():
        if not isinstance(sec, dict) or not sec.get("best_strategy"):
            continue
        sid = json.dumps(sec["best_strategy"], sort_keys=True)
        try:
            stat = J._stat_preds(topic, Path(state), Path(repo))
        except Exception:                                           # noqa: BLE001
            continue
        by_model: dict[str, dict[str, dict[str, Any]]] = {}
        for r in rows:
            if r.get("topic") == topic and r.get("p") is not None and json.dumps(r.get("strategy"), sort_keys=True) == sid:
                by_model.setdefault(str(r.get("model") or J.LEGACY_TAG), {})[r["subject"]] = r
        for model, mine in by_model.items():
            for r, p, _c in J.calibrated(list(mine.values())):
                sp = stat.get(r["subject"])
                if sp is None:
                    continue
                y = int(r["y"])
                loss = T.brier(p, y) - T.brier(sp.p, y)
                _group(acc, "judgment", f"{topic}|{model}", loss, p, y, {"topic": topic, "model": model})
                _group(acc, "judgment_calibration", f"{topic}|{model}|p{min(4, int(p * 5))}", loss, p, y, {"topic": topic, "model": model})
                if abs(p - 0.5) >= 0.3:
                    _group(acc, "judgment_confident", f"{topic}|{model}|{'high' if p >= 0.5 else 'low'}", loss, p, y, {"topic": topic, "model": model})


def live_groups(state: Path, acc: dict[tuple[str, str], dict[str, Any]]) -> None:
    """Prospective predictions (made before their outcome existed): thinking topics, live drills and the fast topics (per topic and key)."""
    th = Path(state) / "thinking"
    for name, topic_field in (("predictions.jsonl", "topic"), ("drill_live.jsonl", "source")):
        recs: dict[str, dict[str, Any]] = {}
        for r in T._jsonl(th / name):
            if "resolved" in r and r.get("id") in recs:
                recs[r["id"]]["y"] = r["resolved"]
            elif "p" in r:
                recs[r["id"]] = dict(r)
        for r in recs.values():
            if r.get("y") is None or r.get("mode") != "live":
                continue
            y = int(r["y"])
            _group(acc, "live", f"{name.split('.')[0]}|{r.get(topic_field)}", T.brier(r["p"], y) - T.brier(r["base"], y), r["p"], y)
    fast: dict[str, dict[str, Any]] = {}
    for r in T._jsonl(th / "fast_predictions.jsonl"):
        if r.get("kind") == "pred":
            fast[r["id"]] = dict(r)
        elif r.get("kind") == "out" and r.get("id") in fast:
            fast[r["id"]]["y"] = r.get("y")
    for r in fast.values():
        if r.get("y") is None:
            continue
        y = int(r["y"])
        loss = T.brier(r["p"], y) - T.brier(r["base"], y)
        _group(acc, "live", f"fast|{r.get('topic')}", loss, r["p"], y)
        _group(acc, "live_key", f"fast|{r.get('topic')}|{r.get('key')}", loss, r["p"], y)


def rank(acc: dict[tuple[str, str], dict[str, Any]], min_n: int = MIN_GROUP_N) -> list[dict[str, Any]]:
    """Weaknesses ranked by the CONSERVATIVE score lost: n x (95% lower bound of the mean excess loss); only groups that lose score with
    confidence come first, then the rest by total loss. Calibration groups also report (mean p - event rate)."""
    out = []
    for (kind, where), g in acc.items():
        n = len(g["losses"])
        if n < min_n:
            continue
        m = mean_ci(g["losses"])
        pbar, ybar = sum(g["ps"]) / n, sum(g["ys"]) / n
        w = {k: v for k, v in g.items() if k not in ("losses", "ps", "ys")}
        w.update(n=n, loss_total=_r(m[0] * n), loss_mean=_r(m[0]), loss_ci95=[_r(m[1]), _r(m[2])], loss_conservative=_r(m[1] * n),
                 significant=bool(m[1] > 0), mean_p=_r(pbar), event_rate=_r(ybar), miscalibration=_r(pbar - ybar),
                 id="W-" + hashlib.sha1(f"{kind}|{where}".encode()).hexdigest()[:10])
        if (w["loss_total"] or 0.0) > 0:
            out.append(w)
    out.sort(key=lambda w: (not w["significant"], -(w["loss_conservative"] if w["significant"] else w["loss_total"])))
    return out


def regressions(cur: dict[str, Any]) -> list[dict[str, Any]]:
    """Skills whose progress curve regresses: a weakness of their own (the benchmark ones are paired on frozen items)."""
    out = []
    for s, c in (cur.get("skills") or {}).items():
        if c.get("trend") == "regressing":
            ch = float(c.get("change") or 0.0)
            out.append({"kind": "regression", "where": s, "id": "W-" + hashlib.sha1(f"regression|{s}".encode()).hexdigest()[:10], "n": c.get("points"),
                        "change": c.get("change"), "change_ci95": c.get("change_ci95"), "paired": c.get("paired"), "significant": True,
                        "loss_total": _r(-ch), "detail": c.get("first_vs_last")})
    return out


def limits(state: Path, data: Data, sources: Sequence[str], now: float) -> dict[str, Any]:
    """Per drill source: is the limit the DATA? (search converged/stopped and no fresh items since the last selection)."""
    from creator import drillsources as D
    rep = D.search_report(Path(state), sources)
    queued = pending_queue(state)
    out = {}
    for s in sources:
        st = rep.get(s) or {}
        d = data.get(s)
        fresh = len([1 for it, _p in (d["preds"] if d else []) if it.resolved is not None and now - it.resolved <= FRESH_WINDOW_S])
        converged = bool(st.get("stopped")) or int(st.get("consecutive_fails") or 0) >= D.SEARCH_STOP_K // 2
        out[s] = {"search": st, "fresh_items": fresh, "converged": converged, "queued": sum(1 for q in queued if q["source"] == s),
                  "data_limited": converged and fresh < MIN_FRESH and not any(q["source"] == s for q in queued)}
    return out


def diagnose(state: Path, repo: Path, data: Data, tj: dict[str, Any], cur: dict[str, Any], sources: Sequence[str], now: float) -> dict[str, Any]:
    acc: dict[tuple[str, str], dict[str, Any]] = {}
    for s in sources:
        d = data.get(s)
        if d is not None:
            drill_groups(s, d["preds"], acc)
    judgment_groups(state, repo, tj, acc)
    live_groups(state, acc)
    ranked = rank(acc)
    ranked = regressions(cur) + ranked                              # a regressing skill is the first thing to look at
    lim = limits(state, data, sources, now)
    rep = {"at": _iso(now), "measure": "excess Brier over the running base rate (judgment: over the statistical predictor), summed over the group; "
           "ranked by n x 95% lower bound", "weaknesses": ranked[:40], "n_groups": len(ranked), "limits": lim,
           "note": "benchmark items are never diagnosed (they only measure); this report only reads the predictions"}
    _write(state, "diagnosis", rep)
    return rep


# ------------------------------------------------------------------------------------------------ 4 improve, then re-measure
def queue_rows(state: Path) -> list[dict[str, Any]]:
    return _read(state, "queue")


def pending_queue(state: Path) -> list[dict[str, Any]]:
    """Queued variants whose run has not been recorded at the source's latest data (what drillsources.drill_filler should hand out)."""
    from creator import drillsources as D
    rows = T._jsonl(D.runs_path(Path(state)))
    ran = {(r.get("source"), json.dumps(r.get("variant"), sort_keys=True)) for r in rows}
    return [q for q in queue_rows(state) if (q["source"], json.dumps(q["variant"], sort_keys=True)) not in ran]


def queued_variants(state: Path) -> list[tuple[str, dict[str, Any]]]:
    """The reader drillsources.drill_filler uses: (source, variant) still to run. The variants are ordinary search variants; the search's own
    selection (select part only) decides whether any of them is adopted - the held-out tail never chooses."""
    return [(q["source"], dict(q["variant"])) for q in pending_queue(state)]


def variants_for(w: dict[str, Any], best: dict[str, Any], improvement: str) -> list[dict[str, Any]]:
    """Parameter-level remedies for a drill weakness, as variants of the current best (the search's own knobs)."""
    from creator import drillsources as D
    base = {k: v for k, v in best.items() if k not in ("search", "learn")}
    base.setdefault("mask", 0)
    base.setdefault("extra", "none")
    base.setdefault("agg", "mean")
    base.setdefault("cap", 0.02)
    vs: list[dict[str, Any]] = []
    if w["kind"] == "feature":
        m = int(base["mask"]) | (1 << int(w["pos"]))
        vs += [{**base, "mask": m}, {**base, "mask": m, "agg": "logit"}, {**base, "mask": m, "k": float(base["k"]) * 2}]
        vs += [{**base, "extra": e} for e in D.EXTRAS if e != base["extra"]]
    elif w["kind"] in ("calibration", "confident"):
        vs += [{**base, "cap": c} for c in D.CAPS if c > float(base["cap"])]
        vs += [{**base, "k": float(base["k"]) * f} for f in (2.0, 4.0)] + [{**base, "agg": "mean" if base["agg"] == "logit" else "logit"}]
    elif w["kind"] == "source":
        vs += [{**base, "decay": d} for d in (0.9, 0.95, 0.99, 1.0) if d != float(base["decay"])]
        vs += [{**base, "extra": e} for e in D.EXTRAS if e != base["extra"]]
    seen, out = set(), []
    for v in vs:
        v.update(search=1, learn=improvement)
        key = json.dumps({k: x for k, x in v.items() if k != "learn"}, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out


def skill_value(state: Path, skill: str, tj: dict[str, Any]) -> Optional[dict[str, Any]]:
    for r in skill_rows(tj) + plugin_rows(state):
        if r["skill"] == skill:
            return {"value": r.get("value"), "ci": r.get("ci"), "n": r.get("n")}
    return None


def _skill_of(w: dict[str, Any]) -> str:
    if w.get("source"):
        return f"drill:{w['source']}:heldout"
    if w.get("topic"):
        return f"judgment:{w['topic']}:vs_statistical"
    if w.get("kind") in ("live", "live_key"):                          # prospective groups map to their progress skill
        f = str(w.get("where")).split("|")
        return {"fast": f"fast:{f[1]}", "drill_live": f"drill:{f[1]}:live", "predictions": f"thinking:{f[1]}"}.get(f[0], str(w.get("where")))
    return str(w.get("where"))


def improvements(state: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for e in _read(state, "improvements"):
        if e.get("event") == "applied":
            out[e["id"]] = dict(e)
        elif e.get("event") == "verdict" and e.get("id") in out:
            out[e["id"]]["verdict"] = e
    return out


def propose_goal(state: Path, w: dict[str, Any], skill: str, before: Optional[dict[str, Any]], why: str, now: float) -> Optional[str]:
    """A bigger remedy: a goal proposal (goals 'weakness' source) for a THINKING module (creator/goal_think_*.py, allowed by the thinking focus),
    with the evidence and the metric that must move. Never a duplicate of an open/approved one for the same weakness."""
    from creator import goals as GO
    subject = f"think_{w['kind']}_{w['where']}"
    title = f"improve thinking where it loses score: {w['kind']} {w['where']}"
    rationale = (f"learn loop diagnosis: {w['kind']} {w['where']} loses {w.get('loss_total')} Brier in all over {w.get('n')} predictions "
                 f"(mean {w.get('loss_mean')}, 95% {w.get('loss_ci95')}); {why}. Metric that must move: {skill} "
                 f"(now {before}) - its gain CI must rise, measured by the unchanged gate")
    p = GO._make("weakness", subject, title, rationale, [w["id"], skill], int(w.get("n") or 10), 1.0, _iso(now))
    slug = GO._slug(f"think_{w['kind']}_{w['where']}")
    key = f"weakness:{subject}"
    d = {**p.to_dict(), "key": key, "id": GO._pid(key), "focus": "thinking", "kind": "thinking", "metric": {"skill": skill, "before": before},
         "spec": {**p.spec, "modules": [f"creator/goal_{slug}.py"], "tests": [f"tests/test_creator_goal_{slug}.py"]}}
    if any(x["key"] == key for x in GO.listing(state)):
        return None
    GO._append(state, d)
    return str(d["id"])


def improve(state: Path, diag: dict[str, Any], tj: dict[str, Any], data: Optional[Data], now: float) -> dict[str, Any]:
    """Verify the open improvements, then act on the top weakness that has no open improvement: queue drill variants (small, applied through the
    measured search), or - when it is not a drill weakness or the small fix was already tried without effect - a goal proposal."""
    from creator import drillsources as D
    done = []
    imps = improvements(state)
    for iid, imp in imps.items():
        if "verdict" not in imp:
            v = verify(state, imp, tj, now)
            if v is not None:
                _append(state, "improvements", [v])
                imp["verdict"] = v
                done.append(v)
    open_skills = {imp["skill"] for imp in imps.values() if "verdict" not in imp}
    tried = {imp["weakness"]["id"]: imp for imp in imps.values()}
    applied: list[dict[str, Any]] = []                                 # at most one small fix and one goal proposal per round
    open_goals = sum(1 for i in imps.values() if i.get("action") == "goal_proposal" and "verdict" not in i)
    for w in diag.get("weaknesses", []):
        used = {a["action"] for a in applied}
        if {"queue_variants", "goal_proposal"} <= used:
            break
        if not w.get("significant"):
            continue
        skill = _skill_of(w)
        if skill in open_skills:
            continue
        before = skill_value(state, skill, tj)
        iid = "IM-" + hashlib.sha1(f"{w['id']}|{now}".encode()).hexdigest()[:10]
        prev = tried.get(w["id"])
        small_ok = w["kind"] in ("feature", "calibration", "confident", "source") and w.get("source") and (
            prev is None or (prev.get("verdict") or {}).get("verdict") == "helped")
        if small_ok and "queue_variants" in used:
            continue
        if small_ok:
            b = D.best_variant(Path(state), w["source"])
            if b is None:
                continue
            vs = variants_for(w, dict(b["variant"]), iid)
            ran = {(r.get("source"), json.dumps({k: x for k, x in (r.get("variant") or {}).items() if k != "learn"}, sort_keys=True))
                   for r in T._jsonl(D.runs_path(Path(state)))}
            vs = [v for v in vs if (w["source"], json.dumps({k: x for k, x in v.items() if k != "learn"}, sort_keys=True)) not in ran]
            if not vs:
                small_ok = False
            else:
                _append(state, "queue", [{"source": w["source"], "variant": v, "improvement": iid, "at": _iso(now)} for v in vs])
                rec = {"event": "applied", "id": iid, "at": _iso(now), "t": now, "action": "queue_variants", "weakness": w, "skill": skill,
                       "before": before, "variants": vs, "best_before": b["variant"], "verify_after": _iso(now + VERIFY_AFTER_S)}
                _append(state, "improvements", [rec])
                applied.append(rec)
                open_skills.add(skill)
                continue
        if not small_ok:
            if open_goals >= MAX_OPEN_GOALS or "goal_proposal" in used:
                continue
            why = ("the parameter-level fix was tried: " + str((prev.get("verdict") or {}).get("verdict"))) if prev else \
                "no parameter of the existing searches reaches it"
            try:
                pid = propose_goal(state, w, skill, before, why, now)
            except Exception as e:                                   # noqa: BLE001
                pid, why = None, f"{why}; proposal failed: {type(e).__name__}: {e}"
            if pid is None:
                continue                                            # already proposed: the next weakness
            rec = {"event": "applied", "id": iid, "at": _iso(now), "t": now, "action": "goal_proposal", "proposal": pid, "weakness": w,
                   "skill": skill, "before": before, "verify_after": _iso(now + VERIFY_AFTER_S)}
            _append(state, "improvements", [rec])
            applied.append(rec)
            open_skills.add(skill)
            open_goals += 1
    return {"applied": applied, "verified": done}


def verify(state: Path, imp: dict[str, Any], tj: dict[str, Any], now: float) -> Optional[dict[str, Any]]:
    """Re-measure an improvement once due. Queued variants: helped = the search adopted one of them and the held-out gain rose; no_effect = none was
    adopted (nothing changed); hurt_flag_revert = adopted but the held-out gain fell below the previous value's CI. Goal proposals: rejected,
    or measured once approved and VERIFY_AFTER_S has passed."""
    if now < _ts(imp.get("verify_after")):
        return None
    after = skill_value(state, imp["skill"], tj)
    before = imp.get("before") or {}
    rec: dict[str, Any] = {"event": "verdict", "id": imp["id"], "at": _iso(now), "skill": imp["skill"], "before": before, "after": after}
    if imp["action"] == "queue_variants":
        from creator import drillsources as D
        src = imp["weakness"]["source"]
        if any(q["improvement"] == imp["id"] for q in pending_queue(state)):
            if now - float(imp.get("t", now)) < 4 * VERIFY_AFTER_S:
                return None                                        # its variants have not all run yet
            rec["note"] = "some queued variants never ran"
        b = D.best_variant(Path(state), src) or {}
        adopted = (b.get("variant") or {}).get("learn") == imp["id"]
        rec["adopted"] = adopted
        bv, av = before.get("value"), (after or {}).get("value")
        bci = before.get("ci") or [None, None]
        if not adopted:
            rec["verdict"] = "no_effect"
        elif bv is None or av is None:
            rec["verdict"] = "inconclusive"
        elif bci[0] is not None and av < float(bci[0]):
            rec["verdict"] = "hurt_flag_revert"
            rec["flag"] = f"held-out gain fell from {bv} to {av} after the search adopted {b.get('variant')}: revert by excluding it (teacher decides)"
        elif av > bv:
            rec["verdict"] = "helped"
        else:
            rec["verdict"] = "inconclusive"
        return rec
    from creator import goals as GO
    st = next((p for p in GO.listing(state) if p["id"] == imp.get("proposal")), None)
    if st is None or st["status"] == "PENDING":
        return None
    if st["status"] == "REJECTED":
        rec["verdict"] = "rejected"
        return rec
    if now - _ts(st.get("decided_at")) < VERIFY_AFTER_S:
        return None
    bv, av = before.get("value"), (after or {}).get("value")
    bci = before.get("ci") or [None, None]
    rec["verdict"] = ("inconclusive" if bv is None or av is None else "hurt_flag_revert" if bci[0] is not None and av < float(bci[0])
                      else "helped" if bci[1] is not None and av > float(bci[1]) else "inconclusive")
    return rec


# ------------------------------------------------------------------------------------------------ 5 need data
def need_data(state: Path, diag: dict[str, Any], now: float) -> Optional[dict[str, Any]]:
    """When the limit is data: the top weakness's source (or every source) has a converged search and no fresh items. Recorded at most every
    NEED_DATA_EVERY_S unless the set of data-limited sources changes; need_data.json always holds the latest state."""
    lim = diag.get("limits") or {}
    queued = {q["source"] for q in pending_queue(state)}               # a remedy queued since the diagnosis is still something to learn
    limited = sorted(s for s, v in lim.items() if v.get("data_limited") and s not in queued)
    top = next((w for w in diag.get("weaknesses", []) if w.get("source")), None)
    trigger = bool(limited) and (len(limited) == len(lim) or (top is not None and top["source"] in limited))
    cur = {"at": _iso(now), "need_data": trigger, "data_limited_sources": limited, "top_weakness": top and {k: top.get(k) for k in ("id", "kind", "where", "loss_total")},
           "want": "more resolved items like these sources' (new commits/repositories with history, new journal entries, plan rows)" if trigger else ""}
    _write(state, "need_data_now", cur)
    if not trigger:
        return None
    prev = _read(state, "need_data")
    if prev and prev[-1].get("data_limited_sources") == limited and now - _ts(prev[-1]["at"]) < NEED_DATA_EVERY_S:
        return None
    ev = {"event": "need_data", **cur, "evidence": {s: lim[s] for s in limited}}
    _append(state, "need_data", [ev])
    return ev


# ------------------------------------------------------------------------------------------------ the frozen benchmark on a cadence
def experiments(state: Path) -> dict[str, Any]:
    """The trial and error made visible (creator.trialerror.experiments_section, written to thinking/experiments.json); returns the compact
    per-source summary: variants tried, kept, the select-vs-held-out rank agreement, and per feature family its best held-out Brier."""
    from creator import registry as REG
    sec = REG.get("trialerror").experiments_section(state)
    _write(state, "experiments", sec)
    return {s: {"tried": d.get("tried"), "kept": sum(1 for x in d.get("latest") or [] if x.get("kept")), "rho": d.get("select_predicts_heldout_rho"),
                "families": {f: g.get("heldout_of_best_select") for f, g in (d.get("families") or {}).items()}}
            for s, d in (sec.get("drills") or {}).items()}


def bench_due(state: Path, now: float, last_launch: float) -> bool:
    runs = bench_runs(state)
    last = max([_ts(r.get("at")) for _n, r in runs] + [last_launch, 0.0])
    return now - last >= BENCH_EVERY_S


def free_gb() -> float:
    try:
        import psutil
        return float(psutil.virtual_memory().available) / 1e9
    except Exception:                                               # noqa: BLE001
        return 0.0


def launch_bench(state: Path, repo: Path) -> int:
    """The benchmark as its own low-priority process (it starts and stops its own model servers; ~10 min); returns its pid."""
    flags = 0x00004000 | 0x00000008 if os.name == "nt" else 0              # BELOW_NORMAL | DETACHED
    p = subprocess.Popen([sys.executable, str(Path(repo) / "scripts" / "nupen_thinkbench.py"), "--compare", "--model", "thinker", "--state", str(state)],
                         cwd=str(repo), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
    return int(p.pid)


# ------------------------------------------------------------------------------------------------ one round
def run(state: Path, repo: Optional[Path] = None, journal: Optional[Path] = None, research: Optional[Path] = None, now: Optional[float] = None,
        bench: bool = True, sources: Sequence[str] = DIAG_SOURCES, launcher: Optional[Callable[[Path, Path], int]] = None) -> dict[str, Any]:
    """measure -> curve -> retention -> diagnosis -> improve/verify -> need-data -> (benchmark when due). Never raises: every step's error is
    recorded in learnloop_run.json and the rest still runs."""
    state = Path(state)
    repo = Path(repo) if repo is not None else Path(__file__).resolve().parents[1]
    journal = Path(journal) if journal is not None else Path.home() / "Masterstock" / "JOURNAL.md"
    research = Path(research) if research is not None else repo / "state" / "research"
    now = time.time() if now is None else now
    out: dict[str, Any] = {"at": _iso(now), "errors": {}}
    try:
        prev_run = json.loads(_out(state, "run").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        prev_run = {}
    tj: dict[str, Any] = {}
    data = Data(state, repo, journal, research)

    def step(name: str, f: Callable[[], Any]) -> Any:
        try:
            return f()
        except Exception as e:                                      # noqa: BLE001 - measuring never stops the swarm
            out["errors"][name] = f"{type(e).__name__}: {e}"
            return None
    tj = step("trust", lambda: trust_report(state, now)) or {}
    out["measured"] = len(step("measure", lambda: measure(state, now, tj)) or [])
    cur = step("curve", lambda: curve(state, now)) or {}
    out["curve"] = cur.get("summary")
    out["answer"] = cur.get("answer")
    out["retention"] = step("retention", lambda: retention(state, data, now, sources))
    diag = step("diagnosis", lambda: diagnose(state, repo, data, tj, cur, sources, now)) or {}
    out["top_weaknesses"] = [{k: w.get(k) for k in ("id", "kind", "where", "n", "loss_total", "loss_ci95", "significant")} for w in diag.get("weaknesses", [])[:5]]
    imp = step("improve", lambda: improve(state, diag, tj, data, now)) or {}
    out["improvements"] = [{k: a.get(k) for k in ("id", "action", "skill", "proposal", "before")} for a in imp.get("applied") or []]
    out["verified"] = imp.get("verified")
    out["need_data"] = step("need_data", lambda: need_data(state, diag, now)) if diag else None
    out["experiments"] = step("experiments", lambda: experiments(state))
    out["bench_launched_at"] = prev_run.get("bench_launched_at", 0.0)
    if bench and bench_due(state, now, float(prev_run.get("bench_launched_at") or 0.0)) and free_gb() >= BENCH_MIN_FREE_GB:
        pid = step("bench", lambda: (launcher or launch_bench)(state, repo))
        if pid:
            out["bench_launched_at"], out["bench_pid"] = now, pid
    _write(state, "run", out)
    return out


if __name__ == "__main__":                                          # python -m creator.learnloop [state] : one round, no benchmark
    print(json.dumps(run(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "state" / "creator", bench=False),
                     indent=1, default=str)[:6000])
