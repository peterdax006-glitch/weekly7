"""THINKING DRILLS and the TRUST GATE (owner, 2 Oct 2026: "train thinking until its opinion is trust worthy ... until it can reliably self manage
everything else"). Thinking is judged by CALIBRATION and plan quality against ground truth from Nupen's own history, never by eloquence.

Drills: before the outcome is known, predict it with a probability and score it against what happened, with strictly time-ordered data
(a prediction at time t sees only records RESOLVED before t; the no-leak property is tested). Topics:
  verdict      P(this package is ADOPTED)             also a 5-class distribution (ADOPTED/REJECTED/ERROR/CANCELLED/DEFERRED)
  duration     P(this cycle takes longer than SLOW_S)  also a point estimate scored in log-seconds
  constraint   P(the top limiter stays the same at the next constraint snapshot)
  cost         P(build takes longer than SLOW_S) estimated from the SPEC only (make-or-ask), also log-seconds
  goal_value   P(a goal proposal is approved)
Baselines for every topic: the running base rate and the last value. Scores: Brier, log-loss. A topic is TRUSTED only when `trust()` says so
(enough scored predictions incl. prospective ones, beats the best baseline with a 95% CI, calibrated). Nothing is auto-enabled by it:
`independent(state, topic)` is the check a caller (e.g. goals.approve by Nupen itself) must pass. Records: state/creator/thinking/predictions.jsonl,
report state/creator/trust.json (both protected paths: Nupen's workers cannot write them). The predictors LEARN: every fit uses all outcomes
resolved so far (decayed frequency / shrunk means), so each new outcome changes the next prediction. Loaded on demand (registry 'thinking')."""
from __future__ import annotations

import datetime as dt
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

CLASSES = ("ADOPTED", "REJECTED", "ERROR", "CANCELLED", "DEFERRED")
TOPICS = ("verdict", "duration", "constraint", "cost", "goal_value")
SLOW_S = 1800.0                    # "slow" cycle / expensive build: > 30 minutes
ASK_S = 8 * 3600.0                 # a build estimated beyond this is days of work: MAKE-or-ASK says ASK (hand the teacher a spec)
MIN_N = 50                         # scored predictions before a topic can be trusted
MIN_LIVE = 10                      # ... of which this many made prospectively (before the outcome existed), not replayed
ECE_TOL = 0.10                     # reliability tolerance
DECAY = 0.97                       # older outcomes count a bit less (regimes change)
SHRINK_K = 3.0                     # how many observations the parent estimate is worth (outcome classes)
DUR_K = 8.0                        # ... for durations, which are noisy (the recency term was tried and removed: it scored worse than the median)


# ------------------------------------------------------------------------------------------------ loading history
def _ts(s: str) -> float:
    d = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    return (d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)).timestamp()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                x = json.loads(line)
            except ValueError:
                continue
            if isinstance(x, dict):
                out.append(x)
    return out


def _local_ts(s: str) -> float:
    """Timestamps of the constraint/goal logs are local naive times (MDT, UTC-6); ledger ones are UTC. Naive ones are shifted to UTC."""
    return _ts(s) if ("+" in s[10:] or s.endswith("Z")) else _ts(s) + 6 * 3600


@dataclass
class Item:
    """One package: what is known at plan time (spec) and, once resolved, what happened."""
    pkg: str
    created: float                  # the package was written (prediction time)
    req: str = ""                   # e.g. K07.exists, EFF.size
    spec_len: int = 0               # size of the written spec (chars)
    n_files: int = 0
    resolved: Optional[float] = None
    outcome: str = ""               # one of CLASSES once resolved
    seconds: float = 0.0
    usd: float = 0.0

    @property
    def kind(self) -> str:
        return self.req.split(".")[-1] if self.req else "?"


_REQ = re.compile(r"\(([A-Z][A-Z0-9]*\.[a-z_]+)\)")


def load_items(state: Path) -> list[Item]:
    """Packages joined from the ledger (WorkPackage = spec at plan time; StrategyOutcome = resolution time) and kernel_log (outcome, seconds)."""
    wp: dict[str, tuple[str, float, dict[str, Any]]] = {}
    res: dict[str, float] = {}
    for r in _jsonl(state / "ledger.jsonl"):
        d = r.get("data") or {}
        ts = (r.get("provenance") or {}).get("timestamp")
        if not ts:
            continue
        if r.get("rtype") == "WorkPackage" and d.get("package_id") and d["package_id"] not in wp:
            wp[d["package_id"]] = (r.get("id", ""), _ts(ts), d)
        elif r.get("rtype") == "StrategyOutcome":
            res[d.get("subject_id", "")] = _ts(ts)
    klog = {k["package"]: k for k in _jsonl(state / "kernel_log.jsonl") if k.get("package")}
    items: list[Item] = []
    for pkg, (wid, t, d) in wp.items():
        m = _REQ.search(str(d.get("why_it_exists", "")))
        k = klog.get(pkg)
        it = Item(pkg, t, req=(k or {}).get("requirement") or (m.group(1) if m else ""),
                  spec_len=len(json.dumps(d, default=str)), n_files=len(d.get("outputs") or []))
        if k and k.get("outcome") in CLASSES:
            it.outcome, it.seconds, it.usd = k["outcome"], float(k.get("seconds") or 0), float(k.get("usd") or 0)
            it.resolved = max(res.get(wid) or (t + it.seconds), t)
        items.append(it)
    items.sort(key=lambda i: i.created)
    return items


# ------------------------------------------------------------------------------------------------ predictors (learn from resolved history)
def outcome_dist(train: Sequence["Item"], req_kind: str) -> dict[str, float]:
    """P(outcome class) for a package of this kind: decayed frequency of the kind, shrunk toward the decayed overall frequency."""
    n = len(train)
    tot = dict.fromkeys(CLASSES, 0.0)
    per = dict.fromkeys(CLASSES, 0.0)
    for i, it in enumerate(train):
        w = DECAY ** (n - 1 - i)
        tot[it.outcome] += w
        if it.kind == req_kind:
            per[it.outcome] += w
    nt, nk = sum(tot.values()), sum(per.values())
    glob = {c: (tot[c] + 0.5) / (nt + 0.5 * len(CLASSES)) for c in CLASSES}
    return {c: (per[c] + SHRINK_K * glob[c]) / (nk + SHRINK_K) for c in CLASSES}


def _mean_sd(xs: Sequence[float]) -> tuple[float, float]:
    if not xs:
        return math.log(SLOW_S), 1.0
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1) if len(xs) > 1 else 1.0
    return m, math.sqrt(v)


def _cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _p_slow(mu: float, sd: float) -> float:
    return min(0.98, max(0.02, 1 - _cdf((math.log(SLOW_S) - mu) / max(sd, 0.3))))


def duration_model(train: Sequence[Item], it: Item, use_recent: bool) -> tuple[float, float]:
    """(mean, sd) of log-seconds: kind mean shrunk to the overall mean; with use_recent also pulled toward the last few cycles (a time series)."""
    xs = [(i.kind, math.log(max(i.seconds, 1.0))) for i in train if i.seconds > 0]
    if not xs:
        return math.log(SLOW_S), 1.0
    gm, gsd = _mean_sd([x for _k, x in xs])
    ks = [x for k, x in xs if k == it.kind]
    mu = (sum(ks) + DUR_K * gm) / (len(ks) + DUR_K)
    if use_recent:
        rec = [x for _k, x in xs[-4:]]
        mu = 0.5 * mu + 0.5 * (sum(rec) / len(rec))
    return mu, gsd


def cost_model(train: Sequence[Item], it: Item) -> tuple[float, float]:
    """Build cost from the SPEC alone (kind, spec size): kind mean plus a ridge-shrunk slope on log spec size. No recency, no outcome."""
    xs = [(i.kind, math.log(max(i.seconds, 1.0)), math.log(max(i.spec_len, 1))) for i in train if i.seconds > 0]
    if not xs:
        return math.log(SLOW_S), 1.0
    gm, gsd = _mean_sd([y for _k, y, _s in xs])
    sm = sum(s for _k, _y, s in xs) / len(xs)
    sxx = sum((s - sm) ** 2 for _k, _y, s in xs)
    sxy = sum((s - sm) * (y - gm) for _k, y, s in xs)
    slope = sxy / (sxx + 1.0)
    ks = [y - slope * (s - sm) for k, y, s in xs if k == it.kind]
    mu = (sum(ks) + DUR_K * gm) / (len(ks) + DUR_K) + slope * (math.log(max(it.spec_len, 1)) - sm)
    return mu, gsd


def make_or_ask(mu_log_s: float) -> str:
    """The make-or-ask rule: a build the spec says takes days (> ASK_S) is handed to the teacher as a precise spec; anything else Nupen makes."""
    return "ASK" if math.exp(mu_log_s) > ASK_S else "MAKE"


# ------------------------------------------------------------------------------------------------ predictions and scoring
@dataclass
class Pred:
    topic: str
    subject: str
    made_at: float
    p: float                                    # model P(event)
    base: float                                 # baseline: running base rate
    last: float                                 # baseline: last value
    outcome: Optional[int] = None
    mode: str = "replay"                        # "live" = made before the outcome existed
    extra: dict[str, Any] = field(default_factory=dict)


def _laplace(xs: Sequence[int]) -> float:
    return (sum(xs) + 1.0) / (len(xs) + 2.0)


def _last(xs: Sequence[int]) -> float:
    return 0.5 if not xs else (0.75 if xs[-1] else 0.25)


def _train(items: Sequence[Item], t: float) -> list[Item]:
    """The ONLY history a prediction made at time t may see: items resolved strictly before t, in resolution order."""
    return sorted((i for i in items if i.resolved is not None and i.resolved < t), key=lambda i: i.resolved or 0.0)


def predict_item(topic: str, items: Sequence[Item], it: Item, at: Optional[float] = None) -> Pred:
    """One prediction for `it` made at `at` (default: it.created) from the items resolved before then; no later record is read."""
    t = it.created if at is None else at
    tr = _train(items, t)
    if topic == "verdict":
        ev = [int(i.outcome == "ADOPTED") for i in tr]
        d = outcome_dist(tr, it.kind)
        return Pred(topic, it.pkg, t, d["ADOPTED"], _laplace(ev), _last(ev), extra={"dist": d})
    ev = [int(i.seconds > SLOW_S) for i in tr if i.seconds > 0]
    mu, sd = duration_model(tr, it, False) if topic == "duration" else cost_model(tr, it)
    return Pred(topic, it.pkg, t, _p_slow(mu, sd), _laplace(ev), _last(ev),
                extra={"log_s": round(mu, 4), "est_s": round(math.exp(mu), 1), "decision": make_or_ask(mu)})


def _truth(topic: str, it: Item) -> int:
    return int(it.outcome == "ADOPTED") if topic == "verdict" else int(it.seconds > SLOW_S)


def replay(items: Sequence[Item], topic: str) -> list[Pred]:
    """Walk forward through history: predict each resolved package at its creation time, then score it against what happened."""
    out = []
    for it in items:
        if it.resolved is None or (topic in ("duration", "cost") and it.seconds <= 0):
            continue
        p = predict_item(topic, items, it)
        p.outcome = _truth(topic, it)
        p.extra["actual_s"] = it.seconds
        out.append(p)
    return out


def replay_constraint(state: Path) -> list[Pred]:
    """Top-limiter persistence at the next constraint snapshot; the model is the decayed persistence rate of that limiter."""
    snaps = sorted((_local_ts(s["at"]), (s.get("ranked") or [{}])[0].get("name", "?")) for s in _jsonl(state / "constraints.jsonl")
                   if s.get("event") == "snapshot" and s.get("ranked"))
    out: list[Pred] = []
    hist: list[tuple[str, int]] = []
    for i in range(len(snaps) - 1):
        t, cur = snaps[i]
        ev = [h for _n, h in hist]
        same = [h for n, h in hist if n == cur]
        sp = sum(h * DECAY ** (len(same) - 1 - j) for j, h in enumerate(same))
        sw = sum(DECAY ** (len(same) - 1 - j) for j in range(len(same)))
        gp = _laplace(ev)
        o = int(snaps[i + 1][1] == cur)
        out.append(Pred("constraint", f"after:{cur}@{t:.0f}", t, min(0.98, max(0.02, (sp + SHRINK_K * gp) / (sw + SHRINK_K))), gp, _last(ev), o,
                        extra={"top": cur}))
        hist.append((cur, o))
    return out


def replay_goal_value(state: Path) -> list[Pred]:
    """P(a proposal is approved) from the proposer's past approvals; resolved by an approve/reject event."""
    ev = _jsonl(state / "goal_proposals.jsonl")
    made = {e["id"]: e for e in ev if e.get("event") == "proposal"}
    done = sorted((e for e in ev if e.get("event") in ("approve", "reject") and e.get("id") in made), key=lambda e: _local_ts(e["at"]))
    out: list[Pred] = []
    hist: list[int] = []
    for e in done:
        o = int(e["event"] == "approve")
        t = _local_ts(made[e["id"]].get("created", e["at"]))
        out.append(Pred("goal_value", e["id"], t, min(0.98, max(0.02, _laplace(hist))), _laplace(hist), _last(hist), o))
        hist.append(o)
    return out


def brier(p: float, o: int) -> float:
    return (p - o) ** 2


def logloss(p: float, o: int) -> float:
    p = min(1 - 1e-6, max(1e-6, p))
    return -math.log(p if o else 1 - p)


def score(preds: Sequence[Pred]) -> dict[str, Any]:
    """Brier / log-loss of the model and both baselines, the paired CI of (best baseline - model), and reliability (ECE over 5 bins)."""
    ps = [p for p in preds if p.outcome is not None]
    n = len(ps)
    if not n:
        return {"n": 0}
    o = [int(p.outcome or 0) for p in ps]
    sc = {"model": [brier(p.p, y) for p, y in zip(ps, o)], "base": [brier(p.base, y) for p, y in zip(ps, o)],
          "last": [brier(p.last, y) for p, y in zip(ps, o)]}
    mean = {k: sum(v) / n for k, v in sc.items()}
    best = "base" if mean["base"] <= mean["last"] else "last"
    d = [b - m for b, m in zip(sc[best], sc["model"])]
    dm = sum(d) / n
    se = math.sqrt(sum((x - dm) ** 2 for x in d) / (n - 1) / n) if n > 1 else float("inf")
    ece = 0.0
    for b in range(5):
        sel = [(p.p, y) for p, y in zip(ps, o) if min(4, int(p.p * 5)) == b]
        if sel:
            ece += len(sel) / n * abs(sum(p for p, _ in sel) / len(sel) - sum(y for _, y in sel) / len(sel))
    return {"n": n, "n_live": sum(1 for p in ps if p.mode == "live"), "brier": round(mean["model"], 4),
            "brier_base_rate": round(mean["base"], 4), "brier_last_value": round(mean["last"], 4), "best_baseline": best,
            "logloss": round(sum(logloss(p.p, y) for p, y in zip(ps, o)) / n, 4),
            "logloss_base_rate": round(sum(logloss(p.base, y) for p, y in zip(ps, o)) / n, 4),
            "gain_vs_best": round(dm, 4), "gain_ci95": [round(dm - 1.96 * se, 4), round(dm + 1.96 * se, 4)] if n > 1 else [None, None],
            "ece": round(ece, 4), "event_rate": round(sum(o) / n, 3)}


def log_mae(items: Sequence[Item], topic: str) -> dict[str, Any]:
    """Point-estimate quality of the seconds estimate (duration/cost): mean |log error| vs the running median and the last value."""
    errs: dict[str, list[float]] = {"model": [], "median": [], "last": []}
    for it in items:
        if it.resolved is None or it.seconds <= 0:
            continue
        tr = [i for i in _train(items, it.created) if i.seconds > 0]
        if not tr:
            continue
        y = math.log(it.seconds)
        mu = duration_model(tr, it, False)[0] if topic == "duration" else cost_model(tr, it)[0]
        ls = sorted(math.log(i.seconds) for i in tr)
        errs["model"].append(abs(mu - y))
        errs["median"].append(abs(ls[len(ls) // 2] - y))
        errs["last"].append(abs(math.log(tr[-1].seconds) - y))
    return {"n": len(errs["model"]), **{f"log_mae_{k}": round(sum(v) / len(v), 4) for k, v in errs.items() if v}}


# ------------------------------------------------------------------------------------------------ record store (live predictions)
def _store(state: Path) -> Path:
    return state / "thinking" / "predictions.jsonl"


def stored(state: Path) -> dict[str, dict[str, Any]]:
    """id -> prediction record with later resolution lines folded in."""
    out: dict[str, dict[str, Any]] = {}
    for r in _jsonl(_store(state)):
        if "resolved" in r:
            if r.get("id") in out:
                out[r["id"]].update(outcome=r["resolved"], resolved_at=r.get("at"))
        else:
            out[r["id"]] = r
    return out


def _append(state: Path, rec: dict[str, Any]) -> None:
    p = _store(state)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")


def live_pass(state: Path, now: float, max_new: int = 20, horizon_s: float = 86400.0) -> dict[str, int]:
    """Record BEFORE-the-fact predictions for packages written in the last horizon and not yet resolved (verdict, duration, cost), and resolve the
    ones whose outcome has since arrived. Idempotent: a package is predicted once per topic, and never after its outcome is known."""
    items = load_items(state)
    known = stored(state)
    new = resolved = 0
    for it in items:
        if it.resolved is None and now - it.created <= horizon_s and it.req:
            for topic in ("verdict", "duration", "cost"):
                pid = f"{topic}:{it.pkg}"
                if pid in known or new >= max_new:
                    continue
                p = predict_item(topic, items, it, at=now)
                _append(state, {"id": pid, "topic": topic, "subject": it.pkg, "made_at": now, "p": p.p, "base": p.base, "last": p.last,
                                "mode": "live", "outcome": None, "extra": p.extra})
                new += 1
    by_pkg = {i.pkg: i for i in items if i.resolved is not None}
    for pid, rec in known.items():
        it2 = by_pkg.get(rec["subject"])
        if rec.get("outcome") is None and it2 is not None and not (rec["topic"] in ("duration", "cost") and it2.seconds <= 0):
            _append(state, {"id": pid, "resolved": _truth(rec["topic"], it2), "at": it2.resolved})
            resolved += 1
    return {"new": new, "resolved": resolved}


def all_preds(state: Path) -> dict[str, list[Pred]]:
    """Per topic: replayed history (walk-forward, no leak) plus stored live predictions that have been resolved."""
    items = load_items(state)
    out: dict[str, list[Pred]] = {t: replay(items, t) for t in ("verdict", "duration", "cost")}
    out["constraint"] = replay_constraint(state)
    out["goal_value"] = replay_goal_value(state)
    for rec in stored(state).values():
        if rec.get("mode") == "live" and rec.get("outcome") is not None and rec["topic"] in out:
            out[rec["topic"]].append(Pred(rec["topic"], rec["subject"], rec["made_at"], rec["p"], rec["base"], rec["last"], rec["outcome"], "live",
                                          rec.get("extra", {})))
    return out


# ------------------------------------------------------------------------------------------------ the trust gate
def trust_of(sc: dict[str, Any]) -> tuple[bool, list[str]]:
    """trusted = enough scored predictions (incl. prospective ones) AND beats the best baseline (95% CI lower bound > 0) AND calibrated."""
    why = []
    if sc.get("n", 0) < MIN_N:
        why.append(f"only {sc.get('n', 0)} scored predictions, need {MIN_N}")
    if sc.get("n_live", 0) < MIN_LIVE:
        why.append(f"only {sc.get('n_live', 0)} prospective predictions, need {MIN_LIVE}")
    lo = (sc.get("gain_ci95") or [None])[0]
    if lo is None or lo <= 0:
        why.append(f"does not beat the best baseline ({sc.get('best_baseline', '?')}) with confidence: gain CI lower bound {lo}")
    if sc.get("ece", 1.0) > ECE_TOL:
        why.append(f"not calibrated: reliability error {sc.get('ece')} > {ECE_TOL}")
    return (not why), why


def trust(state: Path, write: bool = True, owner_dir: Optional[Path] = None) -> dict[str, Any]:
    """Score every topic and report state/creator/trust.json. Pure measurement: it enables nothing. FIRST in the report: the anticipation rate
    (creator.anticipation: the share of the owner's directives Nupen had already proposed before they were given), the top-level metric."""
    from creator import anticipation as ANT                          # on demand: the drills do not need it
    preds = all_preds(state)
    items = load_items(state)
    rep: dict[str, Any] = {"anticipation": ANT.summary(ANT.report(state, owner_dir, write=write)),
                           "at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "topics": {}}
    for t in TOPICS:
        sc = score(preds.get(t, []))
        ok, why = trust_of(sc) if sc.get("n") else (False, ["no scored predictions"])
        rep["topics"][t] = {"trusted": ok, "why_not": why, "score": sc}
        if t in ("duration", "cost"):
            rep["topics"][t]["point_estimate"] = log_mae(items, t)
    try:                                                              # on demand: best variant per drill source (held-out score), when drills ran
        from creator import registry as REG
        rep["drills"] = REG.get("drillsources").trust_section(state)
        rep["judgment"] = REG.get("judgment").trust_section(state)
        rep["fast_topics"] = REG.get("fastpred").trust_section(state)     # fast-resolving prospective topics, same gate (creator.fastpred)
    except Exception as e:                                           # noqa: BLE001 - a missing drill module never breaks the trust report
        rep["drills_error"] = f"{type(e).__name__}: {e}"
    try:                                                              # what trusted predictions DID (creator.decide): arms and outcomes
        from creator import registry as REG2
        rep["decisions"] = REG2.get("decide").summary(state)
    except Exception as e:                                           # noqa: BLE001 - never breaks the trust report
        rep["decisions_error"] = f"{type(e).__name__}: {e}"
    rep["reasoning"] = reasoning_section(state)
    try:                                                              # the trial and error made visible (creator.trialerror): what was tried,
        from creator import registry as REG3                          # why, select / held-out, kept or dropped; a reader only, never the gate
        rep["experiments"] = REG3.get("trialerror").experiments_section(state)
    except Exception as e:                                           # noqa: BLE001 - never breaks the trust report
        rep["experiments"] = {"error": f"{type(e).__name__}: {e}"}
    if write:
        state.mkdir(parents=True, exist_ok=True)
        (state / "trust.json").write_text(json.dumps(rep, indent=1, sort_keys=True), encoding="utf-8")
    return rep


def reasoning_section(state: Path) -> dict[str, Any]:
    """The reasoning drills' report (creator.reasondrills: per-strategy accuracy on fresh multiple-choice questions, CIs vs chance and vs the
    plain prompt, the epoch curve, the fresh-question supply). A reader only: it is not a trust topic and never enters the gate."""
    try:
        from creator import registry as REG
        return dict(REG.get("reasondrills").report_section(state))
    except Exception as e:                                           # noqa: BLE001 - a missing/broken module never breaks the trust report
        return {"error": f"{type(e).__name__}: {e}"}


def independent(state: Path, topic: str) -> bool:
    """May Nupen decide on its own in this topic? Only when its predictions there are trusted NOW (recomputed, never read from a file it could
    write). E.g. goals.approve by Nupen itself requires independent(state, 'goal_value'). Nothing calls this yet: no autonomy is auto-enabled."""
    rep = trust(state, write=False)
    if topic in TOPICS:
        return bool(rep["topics"][topic]["trusted"])
    return bool((rep.get("fast_topics") or {}).get(topic, {}).get("trusted"))        # the fast prospective topics (creator.fastpred), same gate


def run(state: Path, now: Optional[float] = None, write: bool = True) -> dict[str, Any]:
    """One drill round (what the swarm's thinking work calls): live predictions and resolutions, then the trust report."""
    t = dt.datetime.now(dt.timezone.utc).timestamp() if now is None else now
    lp = live_pass(state, t) if write else {}
    rep = trust(state, write=write)
    rep["live"] = lp
    return rep
