"""JUDGMENT DRILLS WITH THE LOCAL MODEL (owner, 2 Oct 2026: Nupen works solely on thinking until its opinion is trustworthy).

The statistical predictors (creator.thinking, creator.drillsources) count; this module asks the LOCAL language model (creator.generator.LocalModel,
a warm pooled server when one is loaded; nothing leaves the machine) to JUDGE: it reads a case exactly as it stood BEFORE the outcome and answers
with a probability and one line of reasoning. Cases: `verdict` (a kernel package: will the cycle be ADOPTED?) and `git_fixed` (a commit: will a later
fix/revert touch one of its files within FIX_WINDOW commits?).

NO FUTURE INFORMATION: a case shows only its setup features and the history of items RESOLVED strictly before the case was created (base rates, the
last outcomes, few-shot examples drawn from those items). Calibration (Platt scaling) is fitted at SCORING time on the strategy's own earlier records
whose outcome was known before the case was created - never on the case itself or anything later.

STRATEGY SEARCH (successive halving): every prompt strategy (shots 0/2/5 x with/without base-rate hint) answers the same first subjects; after 8
common subjects the worse half is dropped, after 20 down to two, after 40 to one. Cases are taken newest first so each has a long history behind it.

Results append to state/creator/thinking/judgment.jsonl; `trust_section` scores them against the statistical predictor on the SAME subjects and
the gate (thinking.trust_of). RAM-heavy (a model server), CPU-moderate: `judgment_filler` hands out batches like the other fillers, at most one
batch per server slot at a time, so the resource-aware admission decides when they run."""
from __future__ import annotations

import datetime as dt
import json
import math
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from creator import drillsources as D
from creator import thinking as T

TOPICS = ("verdict", "git_fixed")
STRATEGIES = [{"shots": s, "hint": h} for s in (0, 2, 5) for h in (0, 1)]
HALVING = ((8, 4), (20, 2), (40, 1))              # (common subjects scored, strategies kept)
MAX_SUBJECTS = 300
BATCH = 4                                          # model calls per job (one server lease each)
MIN_CAL = 15                                       # earlier records needed before Platt scaling replaces the raw probability
PROB = re.compile(r"PROBABILITY\s*[:=]\s*(0(?:\.\d+)?|1(?:\.0+)?|\.\d+)", re.I)


def path(state: Path) -> Path:
    return state / "thinking" / "judgment.jsonl"


@dataclass
class Case:
    topic: str
    subject: str
    created: float
    resolved: float
    y: int
    group: str                  # the feature the hint/shots group by (package kind, commit message class)
    text: str                   # the setup, as known at creation


def _iso(t: float) -> str:
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%d %H:%M")


def load_cases(topic: str, state: Path, repo: Path) -> list[Case]:
    """All RESOLVED cases of a topic in creation order, each with its setup text (never the outcome)."""
    out: list[Case] = []
    if topic == "verdict":
        for it in T.load_items(state):
            if it.resolved is None or not it.outcome:
                continue
            out.append(Case(topic, it.pkg, it.created, it.resolved, int(it.outcome == "ADOPTED"), it.kind,
                            f"A development package for requirement kind '{it.kind}' (requirement {it.req or '?'}) is planned at {_iso(it.created)} UTC. "
                            f"Its written spec has {it.spec_len} characters and it lists {it.n_files} output files. Will the kernel ADOPT its result?"))
    elif topic == "git_fixed":
        items = D.git_items(Path(repo), D.FIX_WINDOW, "fixed", state)
        subj = {c["h"][:10]: c["s"] for cs in D._GIT_CACHE.values() for c in cs}
        for bi in items:
            if bi.resolved is None:
                continue
            out.append(Case(topic, bi.subject, bi.created, bi.resolved, bi.y, bi.keys[0],
                            f"A commit made at {_iso(bi.created)} UTC with message '{subj.get(bi.subject, '?')[:90]}' ({bi.keys[1]}, {bi.keys[2]}, {bi.keys[3]}; "
                            f"numbers are log2 buckets of files and changed lines). Will a later fix or revert commit touch one of its files within "
                            f"{D.FIX_WINDOW} commits?"))
    return out


def history(cases: Sequence[Case], c: Case) -> list[Case]:
    """The only history a case may see: cases resolved strictly before it was created."""
    return sorted((x for x in cases if x.resolved < c.created), key=lambda x: x.resolved)


def build_prompt(strategy: dict[str, Any], cases: Sequence[Case], c: Case) -> list[dict[str, str]]:
    hist = history(cases, c)
    sys_msg = ("You are a careful forecaster for a software-development system. You estimate the probability that the event in the question "
               "happens, from the case and the past record only. Answer with ONE short line of reasoning, then a final line "
               "'PROBABILITY: 0.xx'. Do not hedge at exactly 0.5 unless the record gives no information.")
    lines: list[str] = []
    if strategy.get("hint") and hist:
        same = [x for x in hist if x.group == c.group]
        lines.append(f"Record so far: {len(hist)} resolved cases, event rate {sum(x.y for x in hist) / len(hist):.2f}; "
                     f"last 8 outcomes (1 = event): {[x.y for x in hist[-8:]]}.")
        if same:
            lines.append(f"Cases of this kind ('{c.group}'): {len(same)}, event rate {sum(x.y for x in same) / len(same):.2f}.")
    msgs: list[dict[str, str]] = [{"role": "system", "content": sys_msg}]
    k = int(strategy.get("shots", 0))
    if k and hist:
        pool = [x for x in hist if x.group == c.group][-k:]
        if len(pool) < k:
            pool = (pool + [x for x in reversed(hist) if x not in pool])[:k]
        for x in pool:
            msgs.append({"role": "user", "content": x.text})
            msgs.append({"role": "assistant", "content": f"PROBABILITY: {'0.85' if x.y else '0.15'}"})
    msgs.append({"role": "user", "content": ("\n".join(lines) + "\n" if lines else "") + c.text})
    return msgs


def parse(reply: str) -> Optional[float]:
    m = PROB.findall(reply)
    if not m:
        return None
    return min(0.97, max(0.03, float(m[-1])))


# ------------------------------------------------------------------------------------------------ calibration and scoring
def _logit(p: float) -> float:
    p = min(0.999, max(0.001, p))
    return math.log(p / (1 - p))


def platt_fit(xs: Sequence[float], ys: Sequence[int], lam: float = 1.0, iters: int = 30) -> tuple[float, float]:
    """p = sigmoid(a*logit(raw) + b), ridge-pulled toward the identity (a=1, b=0); Newton steps on the penalised log-loss."""
    a, b = 1.0, 0.0
    for _ in range(iters):
        ga, gb = lam * (a - 1.0), lam * b
        haa, hab, hbb = lam, 0.0, lam
        for x, y in zip(xs, ys):
            q = 1 / (1 + math.exp(-(a * x + b)))
            r = q - y
            w = max(q * (1 - q), 1e-6)
            ga += r * x
            gb += r
            haa += w * x * x
            hab += w * x
            hbb += w
        det = haa * hbb - hab * hab
        if abs(det) < 1e-12:
            break
        da, db = (hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det
        a, b = a - da, b - db
        if abs(da) + abs(db) < 1e-6:
            break
    return a, b


def calibrated(recs: Sequence[dict[str, Any]]) -> list[tuple[dict[str, Any], float, bool]]:
    """Walk the records in creation order; each case's calibrated p uses a Platt fit on records of the SAME strategy whose outcome was resolved
    before this case was created. Returns (record, p, calibrated?)."""
    rs = sorted((r for r in recs if r.get("p") is not None), key=lambda r: r["created"])
    out = []
    for r in rs:
        prior = [x for x in rs if x["resolved"] < r["created"]]
        if len(prior) >= MIN_CAL:
            a, b = platt_fit([_logit(x["p"]) for x in prior], [x["y"] for x in prior])
            out.append((r, min(0.97, max(0.03, 1 / (1 + math.exp(-(a * _logit(r["p"]) + b))))), True))
        else:
            out.append((r, r["p"], False))
    return out


def _stat_preds(topic: str, state: Path, repo: Path) -> dict[str, T.Pred]:
    if topic == "verdict":
        preds = T.replay(T.load_items(state), "verdict")
    else:
        preds = D.walk_forward(D.git_items(Path(repo), D.FIX_WINDOW, "fixed", state), "git_fixed")
    return {p.subject: p for p in preds}


def _mean_ci(d: Sequence[float]) -> list[float]:
    n = len(d)
    if n < 2:
        return [0.0, 0.0]
    m = sum(d) / n
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n)
    return [round(m - 1.96 * se, 4), round(m + 1.96 * se, 4)]


def score_topic(topic: str, state: Path, repo: Path) -> dict[str, Any]:
    """Per strategy: the judge's Brier (raw and Platt-calibrated) against the statistical predictor and the baselines on the SAME subjects."""
    recs = [r for r in T._jsonl(path(state)) if r.get("topic") == topic and r.get("p") is not None]
    if not recs:
        return {}
    stat = _stat_preds(topic, state, repo)
    out: dict[str, Any] = {}
    for sid in sorted({json.dumps(r["strategy"], sort_keys=True) for r in recs}):
        mine: dict[str, dict[str, Any]] = {}
        for r in recs:
            if json.dumps(r["strategy"], sort_keys=True) == sid:
                mine[r["subject"]] = r                          # a repeated call overwrites: the latest answer stands
        cal = [(r, p, c) for r, p, c in calibrated(list(mine.values())) if r["subject"] in stat]
        if not cal:
            continue

        def mk(f: Callable[[dict[str, Any], float], float]) -> list[T.Pred]:
            return [T.Pred(topic, r["subject"], r["created"], f(r, p), stat[r["subject"]].base, stat[r["subject"]].last, r["y"]) for r, p, _c in cal]
        raw, cl = mk(lambda r, p: r["p"]), mk(lambda r, p: p)
        sp, bl = mk(lambda r, p: stat[r["subject"]].p), mk(lambda r, p: (p + stat[r["subject"]].p) / 2)
        d_raw = [T.brier(s.p, s.outcome or 0) - T.brier(j.p, j.outcome or 0) for s, j in zip(sp, raw)]     # > 0: the judge is better
        d_cal = [T.brier(s.p, s.outcome or 0) - T.brier(j.p, j.outcome or 0) for s, j in zip(sp, cl)]
        out[sid] = {"n": len(cal), "n_calibrated": sum(1 for _r, _p, c in cal if c), "judge_raw": T.score(raw), "judge_calibrated": T.score(cl),
                    "statistical": T.score(sp), "blend": T.score(bl),
                    "gain_vs_statistical_raw": [round(sum(d_raw) / len(d_raw), 4), *_mean_ci(d_raw)],
                    "gain_vs_statistical_calibrated": [round(sum(d_cal) / len(d_cal), 4), *_mean_ci(d_cal)]}
    return out


def trust_section(state: Path, repo: Optional[Path] = None) -> dict[str, Any]:
    """For trust.json: per topic the best strategy (lowest calibrated Brier), its comparison with the statistical predictor on the same subjects
    and the gate's verdict. Judgment is not trusted unless it also beats the statistical predictor with confidence."""
    repo = repo or Path(__file__).resolve().parents[1]
    out: dict[str, Any] = {}
    for t in TOPICS:
        sc = score_topic(t, state, repo)
        if not sc:
            continue
        sid, b = min(sc.items(), key=lambda kv: kv[1]["judge_calibrated"].get("brier", 9.0))
        ok, why = T.trust_of(b["judge_calibrated"])
        beats = b["gain_vs_statistical_calibrated"][1] > 0
        if not beats:
            why.append("does not beat the statistical predictor on the same subjects (95% CI lower bound <= 0)")
        out[t] = {"trusted": ok and beats, "why_not": why, "best_strategy": json.loads(sid), "n": b["n"], "judge_calibrated": b["judge_calibrated"],
                  "statistical": b["statistical"], "gain_vs_statistical": b["gain_vs_statistical_calibrated"], "strategies_tried": len(sc)}
    return out


# ------------------------------------------------------------------------------------------------ the strategy search and the job factory
def alive(topic_recs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Successive halving over the strategies from the records so far (compared on subjects every surviving strategy has answered)."""
    live = list(STRATEGIES)
    by: dict[str, dict[str, dict[str, Any]]] = {}
    for r in topic_recs:
        by.setdefault(json.dumps(r["strategy"], sort_keys=True), {})[r["subject"]] = r
    for cutoff, keep in HALVING:
        if len(live) <= keep:
            continue
        common = set.intersection(*[set(by.get(json.dumps(s, sort_keys=True), {})) for s in live])
        if len(common) < cutoff:
            break

        def brier(s: dict[str, Any]) -> float:
            rs = by[json.dumps(s, sort_keys=True)]
            return sum(T.brier(rs[k]["p"] if rs[k].get("p") is not None else 0.5, rs[k]["y"]) for k in common) / len(common)
        live = sorted(live, key=brier)[:keep]
    return live


def _append(state: Path, row: dict[str, Any]) -> None:
    p = path(state)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def run_batch(state: Path, batch: Sequence[tuple[Case, dict[str, Any]]], cases: Sequence[Case], make_llm: Callable[[], Any]) -> int:
    """Ask the model each (case, strategy) in the batch under one server lease and record the answers (the outcome is stored beside the answer only
    AFTER the answer exists; the prompt never contained it)."""
    n = 0
    with make_llm() as llm:
        for c, s in batch:
            reply = llm.chat(build_prompt(s, cases, c), max_tokens=120, temperature=0.2, seed=0, timeout=300.0)
            _append(state, {"topic": c.topic, "subject": c.subject, "strategy": s, "created": c.created, "resolved": c.resolved, "y": c.y,
                            "p": parse(reply), "reply": reply[:300], "at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
            n += 1
    return n


def judgment_filler(state: Path, repo: Path, max_servers: int = 1, llm_factory: Optional[Callable[[], Any]] = None,
                    ) -> Callable[[], Optional[Callable[[], None]]]:
    """next_job() hands out one batch of BATCH (case, strategy) calls, newest cases first, only strategies still alive; None when there is nothing
    to do or `max_servers` batches are already in flight (a batch waits for a model-server slot, so more would only hold threads)."""
    state = Path(state)
    lock = threading.Lock()
    flight = {"n": 0}
    taken: set[tuple[str, str, str]] = set()
    cache: dict[str, list[Case]] = {}

    def make_llm() -> Any:
        if llm_factory is not None:
            return llm_factory()
        from creator import generator as G
        return G.LocalModel()

    def pending(topic: str) -> list[tuple[Case, dict[str, Any]]]:
        if topic not in cache:
            cache[topic] = load_cases(topic, state, Path(repo))
        cases = cache[topic]
        recs = [r for r in T._jsonl(path(state)) if r.get("topic") == topic and "subject" in r]
        done = {(r["subject"], json.dumps(r["strategy"], sort_keys=True)) for r in recs}
        live = alive(recs)
        out: list[tuple[Case, dict[str, Any]]] = []
        for c in list(reversed(cases))[:MAX_SUBJECTS]:
            if len(history(cases, c)) < 3:
                continue
            for s in live:
                k = (c.subject, json.dumps(s, sort_keys=True))
                if k not in done and (topic, *k) not in taken:
                    out.append((c, s))
            if len(out) >= BATCH:
                break
        return out

    def next_job() -> Optional[Callable[[], None]]:
        with lock:
            if flight["n"] >= max_servers:
                return None
            for topic in TOPICS:
                try:
                    todo = pending(topic)
                except Exception:                                  # noqa: BLE001 - a topic without data is skipped
                    continue
                if todo:
                    batch = todo[:BATCH]
                    for c, s in batch:
                        taken.add((topic, c.subject, json.dumps(s, sort_keys=True)))
                    flight["n"] += 1
                    break
            else:
                return None

        def job() -> None:
            try:
                run_batch(state, batch, cache[topic], make_llm)
            except Exception as e:                                 # noqa: BLE001 - a filler never stops the swarm
                _append(state, {"topic": topic, "error": f"{type(e).__name__}: {e}"[:300]})
            finally:
                with lock:
                    flight["n"] -= 1
        return job
    return next_job
