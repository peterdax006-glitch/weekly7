"""JUDGMENT DRILLS WITH THE LOCAL MODEL (owner, 2 Oct 2026: Nupen works solely on thinking until its opinion is trustworthy).

The statistical predictors (creator.thinking, creator.drillsources) count; this module asks the LOCAL language model (creator.generator.LocalModel,
a warm pooled server when one is loaded; nothing leaves the machine) to JUDGE: it reads a case exactly as it stood BEFORE the outcome and answers
with a probability and one line of reasoning. Cases: `verdict` (a kernel package: will the cycle be ADOPTED?) and `git_fixed` (a commit: will a later
fix/revert touch one of its files within FIX_WINDOW commits?).
PUBLIC topics `pub_git_fixed` / `pub_git_churn` (creator.publiccases, 3 Oct 2026): ~23,000 commits of public open-source repositories,
searched in ROUNDS of fresh subjects (open_round) with hundreds of the same subjects per strategy and the winners re-tested out of sample.

NO FUTURE INFORMATION: a case shows only its setup features and the history of items RESOLVED strictly before the case was created (base rates, the
last outcomes, few-shot examples drawn from those items). Calibration (Platt scaling) is fitted at SCORING time on the strategy's own earlier records
whose outcome was known before the case was created - never on the case itself or anything later.

STRATEGY SEARCH (successive halving): every prompt strategy (shots 0/2/5 x with/without base-rate hint) answers the same first subjects; after 8
common subjects the worse half is dropped, after 20 down to two, after 40 to one. Cases are taken newest first so each has a long history behind it.

Results append to state/creator/thinking/judgment.jsonl; `trust_section` scores them against the statistical predictor on the SAME subjects and
the gate (thinking.trust_of). RAM-heavy (a model server), CPU-moderate: `judgment_filler` hands out batches like the other fillers, at most one
batch per server slot at a time, so the resource-aware admission decides when they run."""
from __future__ import annotations

import bisect
import datetime as dt
import json
import math
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from creator import drillsources as D
from creator import reasonmethods as RM
from creator import thinking as T

TOPICS = ("verdict", "git_fixed")
STRATEGIES = [{"shots": s, "hint": h} for s in (0, 2, 5) for h in (0, 1)]
HALVING = ((8, 4), (20, 2), (40, 1))              # (common subjects scored, strategies kept)
# Reasoning methods (creator.reasonmethods): retrieval memory, structured template, self-consistency, teacher traces, and the model-free kNN control.
EXTRA_STRATEGIES: list[dict[str, Any]] = [
    {"shots": 0, "hint": 1, "retrieve": 4}, {"shots": 2, "hint": 1, "retrieve": 4}, {"shots": 0, "hint": 1, "retrieve": 4, "structured": 1},
    {"shots": 0, "hint": 1, "structured": 1}, {"shots": 0, "hint": 1, "retrieve": 4, "samples": 3}, {"shots": 0, "hint": 1, "traces": 3},
    {"shots": 0, "hint": 1, "retrieve": 4, "traces": 2, "structured": 1}, {"shots": 0, "hint": 1, "retrieve": 6, "plans": 1}, {"knn": 8},
]
ALL_STRATEGIES = STRATEGIES + EXTRA_STRATEGIES
HALVING_ALL = ((8, 10), (16, 6), (24, 3), (40, 2), (60, 1))   # the wider pool is cut more gently: new methods get a fair number of subjects first
# PUBLIC COMMIT TOPICS (creator.publiccases; 3 Oct 2026: the search chose on noise with 8-76 subjects per strategy). Tens of thousands of public
# commits: each strategy now gets hundreds of the SAME subjects before it can be dropped, and the search runs in ROUNDS of fresh subjects so a
# winner is re-tested out of sample (pinned into the next round) instead of being trusted once.
PUB_TOPICS = ("pub_git_fixed", "pub_git_churn")
ALL_TOPICS = TOPICS + PUB_TOPICS
REF_STRATEGY: dict[str, Any] = {"shots": 0, "hint": 1}
OWN_STRATEGIES: list[dict[str, Any]] = [{"shots": 0, "hint": 1, "own": 6}, {"shots": 0, "hint": 1, "retrieve": 4, "own": 4}]
# teacher traces and plan notes are about Nupen's own packages: on public commits they would only repeat the plain prompt, so they are left out
PUB_STRATEGIES: list[dict[str, Any]] = ([s for s in ALL_STRATEGIES if not s.get("traces") and not s.get("plans")] + OWN_STRATEGIES)
PUB_HALVING = ((40, 8), (100, 4), (200, 2), (400, 1))
ROUND_SIZE = 400                                   # fresh subjects per search round (the last cutoff: the winner answers all of them)
PUB_SPAN = 2000                                    # per repository: the recent cases a round walks forward through first
PUB_MIN_HISTORY = 50                               # a public case is asked only with at least this many cases resolved before it
PUB_BATCH = 6                                      # public cases are short: more calls per server lease
PUB_REFRESH_S = 1800.0                             # new public repositories / fetched commits are read at most this often
SAMPLE_T = 0.7
MAX_SUBJECTS = 300
BATCH = 4                                          # model calls per job (one server lease each)
REFIT_ALWAYS = 200                                 # below this many prior records the Platt fit is redone for every case
MIN_CAL = 15                                       # earlier records needed before Platt scaling replaces the raw probability
PROB = re.compile(r"PROBABILITY\s*[:=]\s*(0(?:\.\d+)?|1(?:\.0+)?|\.\d+)", re.I)


def path(state: Path) -> Path:
    return state / "thinking" / "judgment.jsonl"


LEGACY_TAG = "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"        # records written before the thinking model existed carry no 'model'


def active_tag() -> str:
    """The model whose answers are scored and extended now: the THINKING model when one is configured, else the fast default."""
    from creator import device as DEV
    from creator import generator as G
    p = DEV.think_model_path()
    return (p or G.DEFAULT_MODEL).name


def _mine(rows: Sequence[dict[str, Any]], tag: str) -> list[dict[str, Any]]:
    """Records of one model (a verdict of the 1.5B never counts toward the thinker's trust, and vice versa)."""
    return [r for r in rows if r.get("model", LEGACY_TAG) == tag]


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
    elif topic in PUB_TOPICS:                                          # public repositories only (publiccases refuses anything else)
        from creator import registry as REG
        for repo, bi, mcls, text in REG.get("publiccases").raw_cases(topic):
            if bi.resolved is not None:
                out.append(Case(topic, bi.subject, bi.created, bi.resolved, bi.y, f"{repo}/{mcls}", text))
    elif topic == "git_fixed":
        try:
            items = D.git_items(Path(repo), D.FIX_WINDOW, "fixed", state)
        except D.GitCacheMissing:                                      # no parsed history yet (the git_cache job builds it): no cases yet
            return out
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


def own_index(recs: Sequence[dict[str, Any]], cases: Sequence[Case]) -> RM.Index:
    """MEMORY OF ITS OWN ANSWERS (owner, 3 Oct 2026: Nupen learns from itself): every subject this model has answered, as 'the case -> my forecast,
    what happened', known when the outcome resolved (the search shows a record only to cases created strictly after that)."""
    text = {c.subject: c for c in cases}
    by: dict[str, list[float]] = {}
    for r in recs:
        if r.get("p") is not None and r.get("subject") in text and "knn" not in (r.get("strategy") or {}):
            by.setdefault(r["subject"], []).append(float(r["p"]))
    out = []
    for k, ps in by.items():
        c = text[k]
        out.append(RM.Rec(c.resolved, "answer", c.topic, c.text[:260],
                          f"my forecast was {sum(ps) / len(ps):.2f}; the event {'happened' if c.y else 'did not happen'}", c.y))
    return RM.Index(out)


def own_block(own: Optional[RM.Index], strategy: dict[str, Any], c: Case) -> str:
    k = int(strategy.get("own", 0))
    if not k or own is None:
        return ""
    got = [r for r in own.search(c.text, c.created, k, kinds=("answer",), topic=c.topic) if r.known < c.created]
    return ("Your own earlier forecasts on similar cases (all resolved before this one), to correct your bias:\n"
            + "\n".join("- " + RM.fmt(r, 200) for r in got) + "\n") if got else ""


def build_prompt(strategy: dict[str, Any], cases: Sequence[Case], c: Case, index: Optional[RM.Index] = None,
                 own: Optional[RM.Index] = None) -> list[dict[str, str]]:
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
    if strategy.get("structured"):
        sys_msg += " " + RM.STRUCTURED
    msgs: list[dict[str, str]] = [{"role": "system", "content": sys_msg}]
    msgs += RM.trace_messages(index, strategy, c) if index is not None else []
    block = RM.retrieved_block(index, strategy, c) if index is not None else ""
    if block:
        lines.append(block.rstrip())
    mine = own_block(own, strategy, c)
    if mine:
        lines.append(mine.rstrip())
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
    by_res = sorted(rs, key=lambda r: r["resolved"])
    xs: list[float] = []
    ys: list[int] = []
    j, fit, fit_n = 0, (1.0, 0.0), -1
    out = []
    for r in rs:
        while j < len(by_res) and by_res[j]["resolved"] < r["created"]:     # the prior set only grows as creation time grows
            xs.append(_logit(by_res[j]["p"]))
            ys.append(by_res[j]["y"])
            j += 1
        n = len(xs)
        if n >= MIN_CAL:
            if n != fit_n and (n < REFIT_ALWAYS or n >= fit_n * 1.02):      # long records: refit when the prior grew by 2% (O(n) fits, not O(n^2))
                fit, fit_n = platt_fit(xs, ys), n
            a, b = fit
            out.append((r, min(0.97, max(0.03, 1 / (1 + math.exp(-(a * _logit(r["p"]) + b))))), True))
        else:
            out.append((r, r["p"], False))
    return out


_STAT_CACHE: dict[str, dict[str, T.Pred]] = {}


def _stat_preds(topic: str, state: Path, repo: Path) -> dict[str, T.Pred]:
    if topic == "verdict":
        preds = T.replay(T.load_items(state), "verdict")
    elif topic in PUB_TOPICS:                                          # ~23,000 items: walked once per change of the public caches
        from creator import registry as REG
        PC = REG.get("publiccases")
        key = f"{topic}|{PC.signature()}"
        if key not in _STAT_CACHE:
            if len(_STAT_CACHE) > 4:
                _STAT_CACHE.clear()
            _STAT_CACHE[key] = {p.subject: p for p in D.walk_forward(PC.items(topic), topic)}
        return _STAT_CACHE[key]
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


def _per_item(rows: Sequence[dict[str, Any]]) -> Optional[float]:
    return round(sum(float(r.get("seconds") or 0) for r in rows) / len(rows), 2) if rows else None


def score_topic(topic: str, state: Path, repo: Path, tag: Optional[str] = None) -> dict[str, Any]:
    """Per strategy: the judge's Brier (raw and Platt-calibrated) against the statistical predictor and the baselines on the SAME subjects."""
    recs = [r for r in _mine(_records(state), tag or active_tag()) if r.get("topic") == topic and r.get("p") is not None]
    if not recs:
        return {}
    stat = _stat_preds(topic, state, repo)
    out: dict[str, Any] = {}
    per: dict[str, dict[str, float]] = {}                       # strategy -> subject -> calibrated p (for the paired comparison with the default)
    ys: dict[str, int] = {}
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
                    "gain_vs_statistical_calibrated": [round(sum(d_cal) / len(d_cal), 4), *_mean_ci(d_cal)],
                    "accuracy": round(sum(1 for r, p, _c in cal if (p >= 0.5) == bool(r["y"])) / len(cal), 4),
                    "seconds_per_item": _per_item([r for r, _p, _c in cal if not r.get("gpu_pulse")]),         # this PC only:
                    "gpu_seconds_per_item": _per_item([r for r, _p, _c in cal if r.get("gpu_pulse")]),     # GPU pulse rows apart
                    "tokens_per_item": round(sum(float(r.get("tokens") or 0) for r, _p, _c in cal) / len(cal), 1)}
        out[sid]["brier_ci"] = _mean_ci([T.brier(p, r["y"]) for r, p, _c in cal])
        per[sid] = {r["subject"]: p for r, p, _c in cal}
        ys.update({r["subject"]: r["y"] for r, _p, _c in cal})
        multi = [r for r, _p, _c in cal if len(r.get("ps") or []) > 1]
        if multi:                                               # self-consistency: Brier of the mean of the first m samples, and the vote
            nmax = min(len(r["ps"]) for r in multi)
            out[sid]["by_samples"] = {str(m): round(sum(T.brier(sum(r["ps"][:m]) / m, r["y"]) for r in multi) / len(multi), 4) for m in range(1, nmax + 1)}
            out[sid]["majority_brier"] = round(sum(T.brier(RM.majority(r["ps"]) or 0.5, r["y"]) for r in multi) / len(multi), 4)
    # the DEFAULT is what the original grid's best strategy does today: every strategy is compared to it on the subjects both answered
    grid = {json.dumps(g, sort_keys=True) for g in STRATEGIES}
    base = [(sid, sum(T.brier(p, ys[k]) for k, p in per[sid].items()) / len(per[sid])) for sid in per if sid in grid]
    if base:
        dsid = min(base, key=lambda x: x[1])[0]
        for sid, o in out.items():
            both = sorted(set(per[sid]) & set(per[dsid]))
            o["default_strategy"] = json.loads(dsid)
            o["gain_vs_default"] = [*RM.paired_gain([per[dsid][k] for k in both], [per[sid][k] for k in both], [ys[k] for k in both]), len(both)]
    return out


def trust_section(state: Path, repo: Optional[Path] = None) -> dict[str, Any]:
    """For trust.json: per topic the best strategy (lowest calibrated Brier), its comparison with the statistical predictor on the same subjects
    and the gate's verdict. Judgment is not trusted unless it also beats the statistical predictor with confidence."""
    repo = repo or Path(__file__).resolve().parents[1]
    out: dict[str, Any] = {}
    for t in ALL_TOPICS:
        try:
            sc = score_topic(t, state, repo)
        except Exception as e:                                         # noqa: BLE001 - one topic's missing data never hides the others
            if t in PUB_TOPICS:
                out[t] = {"trusted": False, "why_not": [f"unscored: {type(e).__name__}: {e}"[:200]]}
                continue
            raise
        if not sc:
            continue
        top = max(v["n"] for v in sc.values())
        fair = {k: v for k, v in sc.items() if v["n"] >= 0.8 * top}      # a strategy answered on few subjects is not compared with one answered on many
        sid, b = min(fair.items(), key=lambda kv: kv[1]["judge_calibrated"].get("brier", 9.0))
        ok, why = T.trust_of(b["judge_calibrated"])
        beats = b["gain_vs_statistical_calibrated"][1] > 0
        if not beats:
            why.append("does not beat the statistical predictor on the same subjects (95% CI lower bound <= 0)")
        out[t] = {"trusted": ok and beats, "why_not": why, "best_strategy": json.loads(sid), "n": b["n"], "judge_calibrated": b["judge_calibrated"],
                  "statistical": b["statistical"], "gain_vs_statistical": b["gain_vs_statistical_calibrated"], "strategies_tried": len(sc),
                  "strategies": strategy_table(sc)}
        if t in PUB_TOPICS:                                            # public replay: measures and trains the judge, never grants trust by itself
            out[t]["trusted"] = False
            out[t]["replay_only"] = True
            out[t]["why_not"] = why + ["public replay topic: it measures the judge, it never grants trust on its own"]
            out[t]["rounds"] = rounds_report(state, t)
    return out


def strategy_table(sc: dict[str, Any]) -> dict[str, Any]:
    """Every strategy with its n, calibrated Brier and that Brier's 95% CI, and its paired gains (vs statistical, vs the default prompt), most
    answered first: a choice made on few subjects shows as a wide interval."""
    return {k: {"n": v["n"], "brier_calibrated": v["judge_calibrated"].get("brier"), "brier_ci": v.get("brier_ci"),
                "brier_raw": v["judge_raw"].get("brier"), "gain_vs_statistical": v["gain_vs_statistical_calibrated"],
                "gain_vs_default": v.get("gain_vs_default")} for k, v in sorted(sc.items(), key=lambda kv: -kv[1]["n"])}


# ------------------------------------------------------------------------------------------------ the strategy search and the job factory
def alive(topic_recs: Sequence[dict[str, Any]], pool: Optional[Sequence[dict[str, Any]]] = None,
          halving: Optional[Sequence[tuple[int, int]]] = None) -> list[dict[str, Any]]:
    """Successive halving over the strategies from the records so far (compared on subjects every surviving strategy has answered).
    Default: the original grid; the filler passes ALL_STRATEGIES / HALVING_ALL so the reasoning methods compete too."""
    live = list(STRATEGIES if pool is None else pool)
    by: dict[str, dict[str, dict[str, Any]]] = {}
    for r in topic_recs:
        by.setdefault(json.dumps(r["strategy"], sort_keys=True), {})[r["subject"]] = r
    for cutoff, keep in (HALVING if halving is None else halving):
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


def ranked(recs: Sequence[dict[str, Any]], pool: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """`pool` sorted by raw Brier on the subjects all of them answered (pool order when there are none)."""
    by: dict[str, dict[str, dict[str, Any]]] = {}
    for r in recs:
        by.setdefault(json.dumps(r["strategy"], sort_keys=True), {})[r["subject"]] = r
    sets = [set(by.get(json.dumps(s, sort_keys=True), {})) for s in pool]
    common = set.intersection(*sets) if sets else set()
    if not common:
        return list(pool)

    def brier(s: dict[str, Any]) -> float:
        rs = by[json.dumps(s, sort_keys=True)]
        return sum(T.brier(rs[k]["p"] if rs[k].get("p") is not None else 0.5, rs[k]["y"]) for k in common) / len(common)
    return sorted(pool, key=brier)


def rounds_path(state: Path) -> Path:
    return state / "thinking" / "judgment_rounds.jsonl"


def load_rounds(state: Path, topic: str, tag: str) -> list[dict[str, Any]]:
    return [r for r in T._jsonl(rounds_path(state)) if r.get("topic") == topic and r.get("model") == tag]


def _uniq(ss: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out = []
    for s in ss:
        k = json.dumps(s, sort_keys=True)
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out


def round_state(rnd: dict[str, Any], recs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Where one round stands: its records, the strategies still alive (halving over the round's own subjects), the strategies asked (alive +
    pinned), what is left to ask, and the winner once nothing is."""
    subs = set(rnd["subjects"])
    mine = [r for r in recs if r.get("round") == rnd["round"] and r.get("subject") in subs]
    live = alive(mine, PUB_STRATEGIES, PUB_HALVING)
    ask = _uniq(live + list(rnd.get("pinned", [])))
    done = {(r["subject"], json.dumps(r["strategy"], sort_keys=True)) for r in mine}
    todo = [(k, s) for k in rnd["subjects"] for s in ask if (k, json.dumps(s, sort_keys=True)) not in done]
    winner = ranked(mine, live)[0] if not todo and live else None
    return {"recs": mine, "live": live, "ask": ask, "todo": todo, "winner": winner}


def open_round(state: Path, topic: str, tag: str, cases: Sequence[Case], recs: Sequence[dict[str, Any]], rounds: Sequence[dict[str, Any]],
               size: Optional[int] = None) -> Optional[dict[str, Any]]:
    """A new round on FRESH subjects: never in an earlier round and never answered by this model. Per repository the round walks FORWARD
    through its most recent PUB_SPAN cases (oldest unused first), so every later round - and every newly fetched commit - comes after the
    answers already given and can learn from them (own-answer memory sees only what resolved before); when a repository's recent span is used
    up it goes back in time, newest unused first. Repositories are interleaved, so every round spans the projects and a new repository joins
    at once. The winners of the last two closed rounds are PINNED - they answer every subject of the new round, an out-of-sample re-test - with
    the reference prompt and the model-free kNN control. None when no fresh subject is left."""
    used = {k for r in rounds for k in r["subjects"]} | {r["subject"] for r in recs if "subject" in r}
    res = sorted(c.resolved for c in cases)
    by_repo: dict[str, list[Case]] = {}
    for c in sorted(cases, key=lambda c: c.created):
        if bisect.bisect_left(res, c.created) >= PUB_MIN_HISTORY:
            by_repo.setdefault(c.subject.split(":")[0], []).append(c)
    per: dict[str, list[Case]] = {}
    for name, cs in by_repo.items():
        cut = max(0, len(cs) - PUB_SPAN)
        per[name] = [c for c in cs[cut:] if c.subject not in used] + [c for c in reversed(cs[:cut]) if c.subject not in used]
    size = ROUND_SIZE if size is None else size
    pick: list[str] = []
    queues = [per[k] for k in sorted(per)]
    while len(pick) < size and any(queues):
        for q in queues:
            if q and len(pick) < size:
                pick.append(q.pop(0).subject)
    if not pick:
        return None
    winners: list[dict[str, Any]] = []
    for prev in rounds:
        w = round_state(prev, recs)["winner"]
        if w is not None:
            winners = _uniq(winners + [w])[-2:]
    row = {"topic": topic, "model": tag, "round": len(rounds), "subjects": pick, "pinned": _uniq(winners + [REF_STRATEGY, {"knn": 8}]),
           "previous_winners": winners, "opened": time.time()}
    p = rounds_path(state)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def rounds_report(state: Path, topic: str, tag: Optional[str] = None) -> list[dict[str, Any]]:
    """Per round: subjects, answers, strategies alive, the winner, and the OUT-OF-SAMPLE re-test of the previous winners on this round's fresh
    subjects (paired raw-Brier gain over the reference prompt [mean, CI low, CI high, n]) beside the gain that strategy showed in the round
    that chose it (selection on noise shows as an in-sample gain that does not survive)."""
    tag = tag or active_tag()
    recs = [r for r in _mine(_records(state), tag) if r.get("topic") == topic and "subject" in r and r.get("p") is not None]
    rounds = load_rounds(state, topic, tag)

    def gain(mine: Sequence[dict[str, Any]], s: dict[str, Any]) -> list[float]:
        a = {r["subject"]: r for r in mine if r["strategy"] == REF_STRATEGY}
        b = {r["subject"]: r for r in mine if r["strategy"] == s}
        both = sorted(set(a) & set(b))
        return [*RM.paired_gain([a[k]["p"] for k in both], [b[k]["p"] for k in both], [a[k]["y"] for k in both]), len(both)]
    states = [round_state(r, recs) for r in rounds]
    out: list[dict[str, Any]] = []
    for i, (rnd, st) in enumerate(zip(rounds, states)):
        out.append({"round": rnd["round"], "subjects": len(rnd["subjects"]), "answers": len(st["recs"]), "alive": len(st["live"]),
                    "winner": st["winner"], "winner_gain_vs_reference_in_sample": gain(st["recs"], st["winner"]) if st["winner"] else None,
                    "retest": [{"strategy": w, "out_of_sample_gain_vs_reference": gain(st["recs"], w),
                                "in_sample_gain_when_chosen": next((gain(states[j]["recs"], w) for j in range(i - 1, -1, -1)
                                                                    if states[j]["winner"] == w), None)}
                               for w in rnd.get("previous_winners", [])]})
    return out


_REC_CACHE: dict[str, Any] = {"key": None, "rows": []}


def _records(state: Path) -> list[dict[str, Any]]:
    """judgment.jsonl, re-read only when it changed (the filler asks often; the file grows to thousands of rows)."""
    p = path(state)
    try:
        st = p.stat()
    except OSError:
        return []
    key = (str(p), st.st_size, st.st_mtime_ns)
    if _REC_CACHE["key"] != key:
        _REC_CACHE["rows"] = T._jsonl(p)
        _REC_CACHE["key"] = key
    return list(_REC_CACHE["rows"])


def _append(state: Path, row: dict[str, Any]) -> None:
    p = path(state)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _fast_mod() -> Any:
    try:
        from creator import registry as REG
        return REG.optional("fastpred")
    except Exception:                                                  # noqa: BLE001
        return None


class _Skip(Exception):
    """The thinking model has no RAM right now: the batch is dropped silently and offered again later."""


def chat_text(llm: Any, messages: Sequence[dict[str, str]], **kw: Any) -> str:
    """The answer without a reasoning model's <think> block (fakes in tests only need .chat)."""
    from creator import generator as G
    return G.THINK_BLOCK.sub("", str(llm.chat(G.prepare_messages(messages, getattr(llm, "model", "")), **kw))).strip()


def run_batch(state: Path, batch: Sequence[tuple[Case, dict[str, Any]]], cases: Sequence[Case], make_llm: Callable[[], Any],
              index: Optional[RM.Index] = None, own: Optional[RM.Index] = None, extra: Optional[dict[str, Any]] = None) -> int:
    """Ask the model each (case, strategy) in the batch under one server lease and record the answers (the outcome is stored beside the answer only
    AFTER the answer exists; the prompt never contained it). `extra` fields (the search round) are stored on every row."""
    n = 0
    t_end = dt.datetime.now(dt.timezone.utc)
    with make_llm() as llm:
        tag = Path(str(getattr(llm, "model", ""))).name or active_tag()
        for c, s in batch:
            fp = _fast_mod()
            pid = None
            try:                                                   # FAST-PREDICTION HOOK: P(the model beats the statistical predictor here), BEFORE the call
                if fp is not None and fp.enabled():
                    pid = fp.begin_safe("judgment_correct", f"{c.topic}:{c.subject}:{json.dumps(s, sort_keys=True)}:{time.time():.0f}",
                                        f"{c.topic}|{s.get('shots')}|{s.get('hint')}", state=state)
            except Exception:                                      # noqa: BLE001
                pid = None
            t0 = time.monotonic()
            row: dict[str, Any] = {"topic": c.topic, "subject": c.subject, "strategy": s, "created": c.created, "resolved": c.resolved, "y": c.y,
                                   "model": tag, **(extra or {})}
            if isinstance(getattr(llm, "pulse", None), str) and llm.pulse:     # answered on a rented GPU (creator.gpupulse): timings kept apart
                row["gpu_pulse"] = llm.pulse
            if "knn" in s:                                          # the model-free control: no call, no cost
                hist = history(cases, c)
                base = sum(x.y for x in hist) / len(hist) if hist else 0.5
                row.update(p=RM.knn_probability(index, c, int(s["knn"]), base) if index is not None else None, reply="knn", tokens=0)
            else:
                msgs = build_prompt(s, cases, c, index, own)
                ps, first, toks = RM.sample(lambda m, **kw: chat_text(llm, m, **kw), msgs, parse, int(s.get("samples", 1)),
                                            300 if s.get("structured") else 120)
                row.update(p=RM.aggregate(ps), reply=first[:300], tokens=toks)
                if len(ps) > 1:
                    row["ps"] = [round(x, 4) for x in ps]
            row.update(seconds=round(time.monotonic() - t0, 2), at=t_end.isoformat(timespec="seconds"))
            _append(state, row)
            if pid and fp is not None:
                fp.judgment_resolve(state, pid, c.topic, c.subject, row.get("p"), c.y)
            n += 1
    return n


def judgment_filler(state: Path, repo: Path, max_servers: int = 1, llm_factory: Optional[Callable[[], Any]] = None,
                    topics: Sequence[str] = ALL_TOPICS) -> Callable[[], Optional[Callable[[], None]]]:
    """next_job() hands out one batch of (case, strategy) calls; None when there is nothing to do or `max_servers` batches are already in flight
    (a batch waits for a model-server slot, so more would only hold threads). Topics take turns (round robin) so the public topics' long
    searches never starve Nupen's own; Nupen's topics go newest case first over the strategies still alive, the public ones round by round
    (open_round): when a round's search has converged a new round opens on fresh subjects with the winners pinned for an out-of-sample re-test.
    New public repositories / fetched commits are read every PUB_REFRESH_S (creator.publiccases.refresh_all, inside a public job or on its own
    when nothing else is to do) and the cases reload when the caches changed."""
    state = Path(state)
    from creator import device as DEV
    cfg = DEV.settings()
    if llm_factory is None and DEV.think_model_path(cfg) is not None:        # a bigger thinking model: only as many at once as RAM allows
        max_servers = max(1, min(max_servers, int(cfg.get("think_servers", 1)) or 1))
    lock = threading.Lock()
    flight = {"n": 0}
    taken: set[tuple[str, str, str]] = set()
    cache: dict[str, list[Case]] = {}
    idx: dict[str, RM.Index] = {}
    sig: dict[str, Any] = {"checked": 0.0, "value": None, "refreshed": 0.0, "refreshing": False}
    turn = {"i": 0}

    def make_llm() -> Any:
        if llm_factory is not None:
            return llm_factory()
        from creator import generator as G
        lm = G.thinker()                                       # the THINKING model (device 'think_model'); the fast one when none fits
        if Path(str(lm.model)).name != active_tag():           # RAM said no: wait for room rather than pile up answers of the wrong model
            raise _Skip()
        return lm

    def pending(topic: str) -> list[tuple[Case, dict[str, Any]]]:
        if topic not in cache:
            cache[topic] = load_cases(topic, state, Path(repo))
        cases = cache[topic]
        recs = [r for r in _mine(_records(state), active_tag()) if r.get("topic") == topic and "subject" in r]
        done = {(r["subject"], json.dumps(r["strategy"], sort_keys=True)) for r in recs}
        live = alive(recs, ALL_STRATEGIES, HALVING_ALL)
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

    def public_changed() -> None:
        """Reload the public cases when a cache grew or a repository appeared (checked at most once a minute)."""
        now = time.time()
        if now - sig["checked"] < 60.0 and all(t in cache for t in topics if t in PUB_TOPICS):
            return
        sig["checked"] = now
        from creator import registry as REG
        v = REG.get("publiccases").signature()
        if v != sig["value"]:
            sig["value"] = v
            for t in PUB_TOPICS:
                cache.pop(t, None)
                idx.pop(t, None)

    def pending_pub(topic: str) -> tuple[list[tuple[Case, dict[str, Any]]], Optional[int]]:
        public_changed()
        if topic not in cache:
            cache[topic] = load_cases(topic, state, Path(repo))
        cases = cache[topic]
        if not cases:
            return [], None
        cmap = {c.subject: c for c in cases}
        tag = active_tag()
        recs = [r for r in _mine(_records(state), tag) if r.get("topic") == topic and "subject" in r]
        rounds = load_rounds(state, topic, tag)
        todo = [(k, s) for k, s in round_state(rounds[-1], recs)["todo"] if k in cmap] if rounds else []
        if not todo:                                               # converged (or first time): a fresh round, winners pinned
            rnd = open_round(state, topic, tag, cases, recs, rounds)
            if rnd is None:
                return [], None
            rounds = list(rounds) + [rnd]
            todo = [(k, s) for k, s in round_state(rnd, recs)["todo"] if k in cmap]
        out = [(cmap[k], s) for k, s in todo if (topic, k, json.dumps(s, sort_keys=True)) not in taken][:PUB_BATCH]
        return out, int(rounds[-1]["round"])

    def refresh_due() -> bool:
        return not sig["refreshing"] and time.time() - sig["refreshed"] >= PUB_REFRESH_S and any(t in PUB_TOPICS for t in topics)

    def refresh() -> None:
        try:
            from creator import registry as REG
            REG.get("publiccases").refresh_all()
        except Exception as e:                                     # noqa: BLE001
            _append(state, {"topic": "public_refresh", "error": f"{type(e).__name__}: {e}"[:300]})
        finally:
            sig["refreshed"], sig["refreshing"], sig["checked"] = time.time(), False, 0.0

    def next_job() -> Optional[Callable[[], None]]:
        with lock:
            if flight["n"] >= max_servers:
                return None
            order = [topics[(turn["i"] + j) % len(topics)] for j in range(len(topics))]
            batch: list[tuple[Case, dict[str, Any]]] = []
            extra: Optional[dict[str, Any]] = None
            topic = ""
            for topic in order:
                try:
                    if topic in PUB_TOPICS:
                        batch, rnd = pending_pub(topic)
                        extra = {"round": rnd}
                    else:
                        batch, extra = pending(topic)[:BATCH], None
                except Exception:                                  # noqa: BLE001 - a topic without data is skipped
                    continue
                if batch:
                    break
            if not batch:
                if refresh_due():                                  # nothing to ask: read new public repositories / commits on its own
                    sig["refreshing"] = True
                    return refresh
                return None
            turn["i"] = (list(topics).index(topic) + 1) % len(topics)
            keys = [(topic, c.subject, json.dumps(s, sort_keys=True)) for c, s in batch]
            taken.update(keys)
            flight["n"] += 1
            do_refresh = topic in PUB_TOPICS and refresh_due()
            if do_refresh:
                sig["refreshing"] = True

        def job() -> None:
            skipped = False
            try:
                if do_refresh:
                    refresh()
                if topic not in idx:
                    idx[topic] = RM.Index(RM.build_corpus(state, cache[topic]))
                own = None
                if topic in PUB_TOPICS and any(s.get("own") for _c, s in batch):
                    own = own_index([r for r in _mine(_records(state), active_tag()) if r.get("topic") == topic], cache[topic])
                run_batch(state, batch, cache[topic], make_llm, idx[topic], own, extra)
            except _Skip:
                skipped = True
            except Exception as e:                                 # noqa: BLE001 - a filler never stops the swarm
                _append(state, {"topic": topic, "error": f"{type(e).__name__}: {e}"[:300]})
            finally:
                with lock:
                    flight["n"] -= 1
                    if skipped:                                    # no RAM for the thinking model: offered again later
                        taken.difference_update(keys)
        return job
    return next_job
