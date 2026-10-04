"""THE SELF-TEACHING TRUST GATE (owner, 3 Oct 2026: "it should be able to collect and drill data where it can teach itself efficently ... we need it
to be the level where you would trust it to teach itself").

A MEASUREMENT, never a switch: `gate(state)` reads Nupen's real records (read-only: it never writes trust.json, the ledger, drill rows, caches or
any scoring code) and decides whether the self-teaching loop - collect data, drill, update itself - may run WITHOUT supervision. It answers six
questions, each a check in the style of creator.thinking.trust_of (pass / why_not), and every failing check names its fix:

  a GROUNDED    every source a TRUSTED skill learns from has outcomes computed by a deterministic checker from recorded facts (git history,
                tests, kernel verdicts, known answers), not by a model and not by Nupen's own unverified decisions; banked worked examples
                carry their verification (model, pulse) and none is a frozen-benchmark question.
  b LEAK-PROOF  canaries: the walk-forward actually in use may never see an item's own outcome (an outcome resolved at the SAME instant as the
                prediction is a leak: 'strictly before' is the contract) nor a later one; the real records are scanned for such ties; the
                frozen benchmark's items stay outside variant selection and outside the diagnosis that steers the search; its content hash is
                the pinned one.
  c HELD-OUT    every adopted update is judged on data it was not selected on: no regression flagged by the learn loop is still the adopted
                variant (auto-revert), every 'helped' verdict is significant (the after value above the before value's 95% CI), and no
                trusted judgment topic picked its best strategy on the subjects it is scored on.
  d NO DRIFT    a minimum share of the data that resolved recently comes from OUTSIDE Nupen (public upstreams), outside data was acquired
                recently, and no learning source is Nupen's own unverified output.
  e CHOOSING    the next thing to learn is chosen by expected gain per CPU-hour, and the predictions that do the choosing (P(a variant beats the
                best), fastpred 'drill_beats_best') pass the same trust gate (n, prospective n, CI, ECE).
  f EFFICIENCY  drill jobs record what they cost (CPU seconds, wall seconds), so gain per CPU-hour can be computed; it is reported per source.
  g OWN RULER   Nupen cannot edit its measuring code: every measuring module and record is a protected path (creator.sandbox.PROTECTED), and the
                frozen benchmark directory too.

SELF_TEACH_TRUSTED = all checks pass. Nothing calls this to enable anything yet (like thinking.independent); the report goes to
state/creator/thinking/selfteach_gate.json and .md (a protected path), or to `out_dir`. Loaded on demand (registry 'selfteach'); imports only
creator.thinking at module level."""
from __future__ import annotations

import datetime as dt
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from creator import thinking as T

# ------------------------------------------------------------------------------------------------ thresholds (trust_of style)
PINNED_BENCH_HASH = "1f520b202ac4"     # the frozen thinking benchmark's items hash (thinkbench items.json); an edited/refrozen set fails b
MAX_TIE_SHARE = 0.0                    # resolved items whose outcome is known at their own prediction instant (resolved <= created)
MAX_BENCH_IN_SELECT = 0.0              # frozen benchmark items inside a source's current SELECT part (variant selection learns from them)
MIN_HELPED_SIGNIFICANT = 1.0           # share of 'helped' verdicts whose after value is above the before value's 95% CI
MAX_OPEN_REGRESSIONS = 0               # 'hurt_flag_revert' improvements whose variant is still the adopted best
FRESH_WINDOW_S = 86400.0
MIN_OUTSIDE_SHARE = 0.20               # of the prospective (live) predictions resolved in the window: outside (public upstream) share
ACQUIRE_WINDOW_S = 7 * 86400.0         # at least one successful public acquisition in this window
MIN_COST_RECORDED = 0.90               # drill rows at the latest data that carry cpu_s
CHOOSER_TOPIC = "drill_beats_best"     # creator.fastpred: P(this variant beats its source's best) - the prediction that would choose

# How each drill source's OUTCOME is produced (static: it follows from the code in creator.drillsources). 'checker' = deterministic from
# recorded external facts; 'self_output' = the label is a decision Nupen's own code made (predicting itself); 'mixed' = files Nupen's own
# research code wrote, provenance per field unknown.
SOURCE_GROUNDING: dict[str, tuple[str, str]] = {
    "git_fixed": ("checker", "a later commit with a fix word touches one of its files within 20 commits (git history)"),
    "git_churn": ("checker", "a file of it changes again within 10 commits (git history)"),
    "x_git_fixed": ("checker", "same rule on other projects' / public upstreams' git history"),
    "x_git_churn": ("checker", "same rule on other projects' / public upstreams' git history"),
    "journal_persist": ("checker", "keyword topic of the owner's next journal entry (owner-written facts; keyword classifier)"),
    "plan_choice": ("self_output", "whether Nupen's own scheduler chose the node: Nupen predicting its own decisions"),
    "research_bool": ("mixed", "top-level yes/no fields of research JSON written by Nupen's own research code"),
}

# The measuring code and records Nupen must never edit (check g). Predictor knobs live in data (variants), not in these files, so protecting
# them leaves the self-improvement path (learn queue variants, goal_think_* modules) open.
MEASURING_PATHS = (
    "creator/thinking.py", "creator/thinkbench.py", "creator/learnloop.py", "creator/drillsources.py", "creator/fastwalk.py",
    "creator/judgment.py", "creator/fastpred.py", "creator/decide.py", "creator/trialerror.py", "creator/selfteach.py", "creator/gpuselfteach.py",
    "state/creator/trust.json", "state/creator/thinking/drill_runs.jsonl", "state/creator/thinking/selfteach_gate.json",
    "state/creator/thinkbench/items.json", "state/creator/thinkbench/baseline_x.json",
)


def _now() -> float:
    return time.time()


def _iso(t: float) -> str:
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat(timespec="seconds")


def _ts(s: Any) -> float:
    if isinstance(s, (int, float)):
        return float(s)
    try:
        return T._ts(str(s))
    except ValueError:
        return 0.0


def _check(why: list[str], fix: list[str], evidence: dict[str, Any]) -> dict[str, Any]:
    return {"pass": not why, "why_not": why, "fix": fix if why else [], "evidence": evidence}


def _trust_json(state: Path) -> dict[str, Any]:
    try:
        return dict(json.loads((Path(state) / "trust.json").read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return {}


# ------------------------------------------------------------------------------------------------ b: leak canaries
class _Item:
    """A minimal drill item (the walk-forward reads keys / created / resolved / y / subject)."""

    def __init__(self, keys: tuple[str, ...], created: float, resolved: Optional[float], y: int, subject: str) -> None:
        self.keys, self.created, self.resolved, self.y, self.subject, self.meta = keys, created, resolved, y, subject, None


def _canary_items(tie: bool, later: bool = False) -> list[_Item]:
    """20 history items (shared key 'k', outcome 0) and one TARGET with its own key 'k' and outcome 1. The target resolves at its creation
    instant (tie), 1 second after (honest) or - `later` - another item with outcome 1 resolves after the target's creation."""
    hist = [_Item(("k",), float(i), float(i) + 0.5, 0, f"h{i}") for i in range(20)]
    tgt = _Item(("k",), 100.0, 100.0 if tie else 101.0, 1, "target")
    out = hist + [tgt]
    if later:
        out.append(_Item(("k",), 99.0, 100.5, 1, "after"))       # created before, resolved after the target's prediction instant
    return out


def leak_canaries() -> dict[str, Any]:
    """Run the walk-forward drillsources.walk_forward actually uses (compiled when numba is on) and the reference on the canaries. The target's
    prediction must be identical whether its own outcome resolves at the same instant or later, and a later outcome must change nothing."""
    from creator import drillsources as D
    res: dict[str, Any] = {}
    fns = {"in_use": D.walk_forward, "reference": D.walk_forward_reference}
    for name, fn in fns.items():
        def p_of(items: Sequence[Any]) -> float:
            return next(p.p for p in fn(items, "canary", 1.0, 3.0) if p.subject == "target")
        honest, tied = p_of(_canary_items(False)), p_of(_canary_items(True))
        base_honest = p_of(_canary_items(False, later=True))
        res[name] = {"p_honest": round(honest, 6), "p_tied": round(tied, 6), "own_outcome_at_same_instant_leaks": abs(tied - honest) > 1e-12,
                     "later_outcome_leaks": abs(base_honest - honest) > 1e-12}
    res["leak"] = any(v["own_outcome_at_same_instant_leaks"] or v["later_outcome_leaks"] for v in res.values() if isinstance(v, dict))
    return res


def tie_shares(items_by_source: Mapping[str, Sequence[Any]]) -> dict[str, dict[str, Any]]:
    """Per source: resolved items whose outcome is known at (or before) their own creation instant - the walk-forward processes a resolution
    before a creation at equal time, so each of them is predicted WITH its own outcome."""
    out = {}
    for s, items in items_by_source.items():
        res = [it for it in items if it.resolved is not None]
        k = sum(1 for it in res if float(it.resolved) <= float(it.created))
        out[s] = {"resolved": len(res), "ties": k, "share": round(k / len(res), 4) if res else 0.0}
    return out


def select_cut(items: Sequence[Any], split: float) -> float:
    """Creation time of the last item of the SELECT part (the first `split` of the resolved items in creation order), as drillsources'
    split_score cuts the walk-forward's predictions."""
    res = sorted(float(it.created) for it in items if it.resolved is not None)
    k = int(len(res) * split)
    return res[k - 1] if k > 0 else -math.inf


def bench_contamination(state: Path, frozen: dict[str, Any], items_by_source: Mapping[str, Sequence[Any]], split: float) -> dict[str, Any]:
    """Per source: frozen part-a items inside the SELECT part that drillsources does NOT exclude (variants would be chosen on them), and what
    the learn loop's diagnosis (learnloop.diag_pairs, also used by trialerror.react) is handed: frozen items and held-out-tail items in it."""
    from creator import drillsources as D
    from creator import learnloop as LL
    out: dict[str, Any] = {}
    for s, items in items_by_source.items():
        fz = [x for x in frozen.get("a") or [] if x.get("source") == s]
        if not fz:
            continue
        cut = select_cut(items, split)
        excluded = D.frozen_subjects(Path(state), s)
        in_sel = sum(1 for x in fz if float(x.get("created") or 0) <= cut and str(x.get("subject")) not in excluded)
        res = [it for it in items if it.resolved is not None]
        pairs = [(it, T.Pred(s, it.subject, float(it.created), 0.5, 0.5, 0.5, int(it.y))) for it in res]
        diag = LL.diag_pairs(Path(state), s, pairs)
        fzs = {str(x.get("subject")) for x in fz}
        out[s] = {"frozen": len(fz), "in_select_now": in_sel, "excluded_by_drills": len(excluded), "share": round(in_sel / len(fz), 4),
                  "in_diagnosis_input": sum(1 for it, _p in diag if str(it.subject) in fzs),
                  "heldout_in_diagnosis_input": sum(1 for it, _p in diag if float(it.created) > cut)}
    return out


def load_drill_items(state: Path, journal: Optional[Path]) -> dict[str, list[Any]]:
    """The cheap drill sources, READ ONLY: Nupen's git history from its persistent cache (never refreshed here), the journal, the plan log.
    research_bool (a 60k-file walk) and the x sources (hundreds of thousands of commits) are not loaded: their ties follow from the code."""
    from creator import drillsources as D
    out: dict[str, list[Any]] = {}
    cache = D._read_cache(D.cache_path(Path(state)))
    if cache:
        cache.sort(key=lambda c: c["t"])
        out["git_fixed"] = D._git_events(cache, D.FIX_WINDOW, "fixed")
        out["git_churn"] = D._git_events(cache, D.CHURN_WINDOW, "churn")
    if journal is not None and Path(journal).is_file():
        out["journal_persist"] = D.journal_items(Path(journal))
    out["plan_choice"] = D.plan_items(Path(state))
    return out


def frozen_bench(state: Path) -> tuple[Optional[dict[str, Any]], str]:
    try:
        from creator import thinkbench as TB
        return TB.load_frozen(Path(state)), ""
    except Exception as e:                                           # noqa: BLE001 - a missing/edited frozen set is a failing check, not a crash
        return None, f"{type(e).__name__}: {e}"


# ------------------------------------------------------------------------------------------------ the checks
def check_grounded(state: Path, tj: dict[str, Any], frozen: Optional[dict[str, Any]]) -> dict[str, Any]:
    why, fix = [], []
    drills = tj.get("drills") or {}
    used = {s: SOURCE_GROUNDING.get(s, ("unknown", "not in the grounding table"))[0] for s in drills} or \
        {s: g[0] for s, g in SOURCE_GROUNDING.items()}
    unverified = sorted(s for s, g in used.items() if g != "checker")
    trusted_unverified = sorted(s for s in unverified if (drills.get(s) or {}).get("trusted"))
    if trusted_unverified:
        why.append(f"trusted skills learn from unverified sources: {trusted_unverified}")
        fix.append("never grant trust on a self_output/mixed source: add the grounding table to drillsources.trust_section's veto")
    bank = T._jsonl(Path(state) / "thinking" / "trace_bank.jsonl")
    unverif_bank = [r.get("qid") for r in bank if not (r.get("model") and r.get("gpu_pulse") and r.get("trace"))]
    frozen_ids = {str(q.get("id")) for q in (frozen or {}).get("c") or []}
    bank_frozen = [r.get("qid") for r in bank if str(r.get("qid")) in frozen_ids]
    if unverif_bank:
        why.append(f"{len(unverif_bank)} banked worked examples carry no verification record (model, pulse)")
        fix.append("bank_add only after the known-answer check (gpupulse.traces does this); drop rows without model/gpu_pulse")
    if bank_frozen:
        why.append(f"{len(bank_frozen)} banked worked examples are frozen-benchmark questions")
        fix.append("filter trace_bank rows through reasondrills.Frozen.collides before banking")
    return _check(why, fix, {"sources": {s: {"grounding": used[s], "label": SOURCE_GROUNDING.get(s, ("?", "?"))[1],
                                             "trusted": bool((drills.get(s) or {}).get("trusted"))} for s in sorted(used)},
                             "unverified_sources": unverified, "trace_bank_rows": len(bank), "trace_bank_unverified": len(unverif_bank),
                             "trace_bank_frozen": len(bank_frozen)})


def check_leakproof(state: Path, frozen: Optional[dict[str, Any]], frozen_err: str, items: dict[str, list[Any]]) -> dict[str, Any]:
    from creator import drillsources as D
    why, fix = [], []
    can = leak_canaries()
    if can["leak"]:
        why.append("canary: the walk-forward predicts an item WITH its own outcome when the outcome resolves at the prediction instant "
                   "(events sort resolve-before-create at equal time)")
        fix.append("order creations before resolutions at equal time (event kind create=0, resolve=1) in drillsources.walk_forward_reference "
                   "AND fastwalk; version it (digest suffix) so every drill row is recomputed; thinkbench part a keeps its baseline semantics "
                   "or is re-baselined by the teacher")
    ties = tie_shares(items)
    ties["research_bool"] = {"resolved": None, "ties": None, "share": 1.0, "note": "created == resolved == file mtime by construction"}
    bad = {s: v["share"] for s, v in ties.items() if (v.get("share") or 0.0) > MAX_TIE_SHARE}
    if bad and can["leak"]:                                          # ties are harmless once the canary proves create-before-resolve
        why.append(f"sources with outcomes known at their own prediction instant: {bad}")
        fix.append("plan_choice / research_bool: resolve strictly after creation (plan: at the next plan row; research: drop or date the "
                   "outcome later) - with the walk-forward fix above these items stop seeing their own label")
    split = float(D.SELECT_SPLIT)
    if frozen is None:
        why.append(f"the frozen benchmark does not load: {frozen_err}")
        fix.append("restore state/creator/thinkbench/items.json (never refreeze without the teacher)")
        contam: dict[str, Any] = {}
    else:
        h = str(frozen.get("hash", ""))
        if not h.startswith(PINNED_BENCH_HASH) or D.frozen_hash_ok(Path(state)) is False:
            why.append(f"frozen benchmark hash {h[:12]} is not the pinned {PINNED_BENCH_HASH}")
            fix.append("the frozen items changed: restore them; a refreeze is a teacher decision that also updates PINNED_BENCH_HASH")
        contam = bench_contamination(state, frozen, items, split)
        sel = {s: v["share"] for s, v in contam.items() if v["share"] > MAX_BENCH_IN_SELECT}
        if sel:
            why.append(f"frozen benchmark items are now inside the drill search's SELECT part (variants are chosen on them): {sel}")
            fix.append("exclude frozen part-a subjects from drillsources.split_score's select part (and from learnloop/trialerror diagnosis) - "
                       "or pin the select/held-out cut per source at freeze time")
        diag = {s: [v["in_diagnosis_input"], v["heldout_in_diagnosis_input"]] for s, v in contam.items()
                if v["in_diagnosis_input"] or v["heldout_in_diagnosis_input"]}
        if diag:
            why.append(f"the learn loop's diagnosis (and trialerror.react) reads frozen benchmark items and the held-out tail to choose what to "
                       f"try next: {diag}")
            fix.append("learnloop.drill_groups / trialerror.react: group only pairs with made_at <= select_cut and subject not in frozen part a")
    return _check(why, fix, {"canaries": can, "ties": ties, "bench_hash": (frozen or {}).get("hash", "")[:12], "pinned": PINNED_BENCH_HASH,
                             "bench_contamination": contam, "select_split": split})


def check_heldout(state: Path, tj: dict[str, Any]) -> dict[str, Any]:
    from creator import drillsources as D
    why, fix = [], []
    imps: dict[str, dict[str, Any]] = {}
    for e in T._jsonl(Path(state) / "thinking" / "improvements.jsonl"):
        if e.get("event") == "applied":
            imps[e["id"]] = dict(e)
        elif e.get("event") == "verdict" and e.get("id") in imps:
            imps[e["id"]]["verdict"] = e
    verdicts = [i["verdict"] for i in imps.values() if "verdict" in i]
    helped = [v for v in verdicts if v.get("verdict") == "helped"]

    def significant(v: dict[str, Any]) -> bool:
        hi = ((v.get("before") or {}).get("ci") or [None, None])[1]
        av = (v.get("after") or {}).get("value")
        return hi is not None and av is not None and float(av) > float(hi)
    sig = [v["id"] for v in helped if significant(v)]
    share = len(sig) / len(helped) if helped else 1.0
    if share < MIN_HELPED_SIGNIFICANT:
        why.append(f"{len(helped) - len(sig)} of {len(helped)} 'helped' verdicts are not significant (after value inside the before CI)")
        fix.append("learnloop.verify (queue_variants): 'helped' only when the after value is above the before CI upper bound, else "
                   "'inconclusive' - the same rule its goal-proposal branch already uses")
    open_reg = []
    for i in imps.values():
        v = i.get("verdict") or {}
        if v.get("verdict") == "hurt_flag_revert" and i.get("action") == "queue_variants":
            b = D.best_variant(Path(state), str((i.get("weakness") or {}).get("source")))
            if b is not None and (b.get("variant") or {}).get("learn") == i["id"]:
                open_reg.append({"id": i["id"], "skill": i.get("skill"), "flag": v.get("flag")})
    if len(open_reg) > MAX_OPEN_REGRESSIONS:
        why.append(f"{len(open_reg)} regressions flagged by the learn loop are still the adopted variant (no auto-revert)")
        fix.append("auto-revert: best_variant skips variants whose improvement id has a hurt_flag_revert verdict (record the exclusion)")
    judged = []
    for t, v in (tj.get("judgment") or {}).items():
        if isinstance(v, dict) and int(v.get("strategies_tried") or 0) > 1:
            judged.append({"topic": t, "strategies": v.get("strategies_tried"), "trusted": bool(v.get("trusted"))})
    biased_trusted = [j["topic"] for j in judged if j["trusted"]]
    if biased_trusted:
        why.append(f"judgment topics trusted on the subjects their best strategy was picked on: {biased_trusted}")
        fix.append("judgment.trust_section: pick the strategy on earlier subjects, score it on later ones (time split) before trust")
    drills = tj.get("drills") or {}
    return _check(why, fix, {"verdicts": {k: sum(1 for v in verdicts if v.get("verdict") == k)
                                          for k in sorted({str(v.get("verdict")) for v in verdicts})},
                             "helped_significant": sig, "open_regressions": open_reg, "judgment_selection_on_eval": judged,
                             "drill_trust_on_heldout": {s: {"trusted": v.get("trusted"), "heldout_ci": (v.get("heldout") or {}).get("gain_ci95")}
                                                        for s, v in drills.items() if isinstance(v, dict)}})


def check_drift(state: Path, tj: dict[str, Any], now: float) -> dict[str, Any]:
    why, fix = [], []
    th = Path(state) / "thinking"

    def resolved_in_window(path: Path) -> int:
        n = 0
        for r in T._jsonl(path):
            if "resolved" in r and now - _ts(r.get("at")) <= FRESH_WINDOW_S:
                n += 1
        return n
    own, outside = resolved_in_window(th / "drill_live.jsonl"), resolved_in_window(th / "drill_live_x.jsonl")
    share = outside / (own + outside) if own + outside else 0.0
    if share < MIN_OUTSIDE_SHARE:
        why.append(f"outside data share of recently resolved live predictions {share:.2f} < {MIN_OUTSIDE_SHARE}")
        fix.append("keep public acquisition/fetch jobs running (publicdata) so outside outcomes keep arriving")
    acq = [a for a in T._jsonl(th / "public_acquisitions.jsonl") if a.get("ok") and now - _ts(a.get("t") or a.get("at")) <= ACQUIRE_WINDOW_S]
    if not acq:
        why.append(f"no successful public acquisition in the last {ACQUIRE_WINDOW_S / 86400:.0f} days")
        fix.append("publicdata.acquire_one: acquisition stalled - check need_data.json and the fetch errors in drill_runs.jsonl")
    drills = tj.get("drills") or {}
    self_src = sorted(s for s in drills if SOURCE_GROUNDING.get(s, ("unknown", ""))[0] in ("self_output", "unknown"))
    if self_src:
        why.append(f"learning sources that are Nupen's own decisions: {self_src} (it predicts itself; never a trust basis)")
        fix.append("keep them as replay-only (no trust, no learn-loop goals) or drop them from DIAG_SOURCES")
    return _check(why, fix, {"window_h": FRESH_WINDOW_S / 3600, "resolved_live_own": own, "resolved_live_outside": outside,
                             "outside_share": round(share, 4), "acquisitions_7d": len(acq), "self_output_sources": self_src})


def check_choosing(state: Path, tj: dict[str, Any]) -> dict[str, Any]:
    from creator import drillsources as D
    why, fix = [], []
    chooser = ((tj.get("fast_topics") or {}).get(CHOOSER_TOPIC) or {})
    sc = chooser.get("score") or {}
    ok, why_c = T.trust_of(sc) if sc.get("n") else (False, ["no scored predictions"])
    if not ok:
        why.append(f"the gain predictor ({CHOOSER_TOPIC}) is not trusted: {why_c}")
        fix.append("let fastpred resolve more drill_beats_best predictions; it gates the chooser below")
    uses = bool(getattr(D, "CHOOSE_BY_GAIN_PER_CPU_H", False))
    if not uses:
        why.append("the drill filler picks the next job by fixed grid / learn queue / round robin, not by expected gain per CPU-hour")
        fix.append("drill_filler: order candidate jobs by P(beats best) x expected Brier gain / predicted CPU-hours (cpu_s of the source's "
                   "recent rows), once the cost is recorded (check f) and the gain predictor is trusted")
    imps = [e for e in T._jsonl(Path(state) / "thinking" / "improvements.jsonl") if e.get("event") == "applied"]
    with_pred = sum(1 for e in imps if e.get("predicted_gain") is not None)
    if imps and with_pred < len(imps):
        why.append(f"{len(imps) - with_pred} of {len(imps)} learn-loop improvements recorded no predicted gain: predicted vs actual cannot be "
                   "calibrated")
        fix.append("learnloop.improve: record predicted_gain (e.g. the weakness's conservative loss share) so verify can score it")
    return _check(why, fix, {"chooser_topic": CHOOSER_TOPIC, "chooser_trusted": ok, "chooser_score": {k: sc.get(k) for k in
                             ("n", "n_live", "gain_ci95", "ece", "brier", "brier_base_rate")}, "chooses_by_gain_per_cpu_h": uses,
                             "improvements": len(imps), "improvements_with_predicted_gain": with_pred})


def check_efficiency(state: Path, now: float) -> dict[str, Any]:
    from creator import drillsources as D
    why, fix = [], []
    rows = [r for r in T._jsonl(D.runs_path(Path(state))) if (r.get("select") or {}).get("n")]
    recent = [r for r in rows if now - _ts(r.get("at")) <= FRESH_WINDOW_S]
    costed = [r for r in recent if r.get("cpu_s") is not None]
    share = len(costed) / len(recent) if recent else 0.0
    per: dict[str, dict[str, Any]] = {}
    for r in costed:
        p = per.setdefault(str(r.get("source")), {"jobs": 0, "cpu_s": 0.0, "wall_s": 0.0, "items": 0})
        p["jobs"] += 1
        p["cpu_s"] += float(r.get("cpu_s") or 0.0)
        p["wall_s"] += float(r.get("wall_s") or 0.0)
        p["items"] += int(r.get("items") or 0)
    for s, p in per.items():
        b = D.best_variant(Path(state), s)
        g = ((b or {}).get("heldout") or {}).get("gain_vs_best")
        p["cpu_h"] = round(p["cpu_s"] / 3600.0, 4)
        p["items_per_cpu_s"] = round(p["items"] / p["cpu_s"], 1) if p["cpu_s"] else None
        p["best_heldout_gain_per_cpu_h"] = round(float(g) / p["cpu_h"], 4) if g is not None and p["cpu_h"] else None
    if share < MIN_COST_RECORDED:
        why.append(f"only {share:.0%} of the last day's drill rows record their CPU cost (need {MIN_COST_RECORDED:.0%})")
        fix.append("drillsources.compute_row records cpu_s / wall_s (added on h58); rows written by the running swarm get it once merged")
    disk: dict[str, float] = {}
    for name in ("drill_runs.jsonl", "drill_live_x.jsonl", "fast_predictions.jsonl", "retention.jsonl", "git_history.jsonl", "trace_bank.jsonl"):
        fp = Path(state) / "thinking" / name
        disk[name] = round(fp.stat().st_size / 1e6, 2) if fp.exists() else 0.0
    res = T._jsonl(Path(state) / "resource_samples.jsonl")[-50:]
    return _check(why, fix, {"rows_last_day": len(recent), "rows_costed": len(costed), "cost_share": round(share, 4), "per_source": per,
                             "disk_mb": disk, "resource_samples": len(res),
                             "cpu_pct_mean": round(sum(float(r.get("cpu") or 0) for r in res) / len(res), 1) if res else None,
                             "free_gb_min": min((float(r.get("free_gb") or 0) for r in res), default=None)})


def check_own_ruler() -> dict[str, Any]:
    from creator import focus as F
    from creator import sandbox as S
    why, fix = [], []
    open_paths = [p for p in MEASURING_PATHS if not S.is_protected(p)]
    if open_paths:
        why.append(f"measuring code/records a Nupen work package could change: {open_paths}")
        fix.append("add them to creator/sandbox.py PROTECTED (the owner adopts measuring-code changes, as for devbench/ledger)")
    planned = sorted(set(F.THINKING_MODULES) & set(MEASURING_PATHS))
    return _check(why, fix, {"measuring_paths": list(MEASURING_PATHS), "unprotected": open_paths,
                             "focus_lists_as_thinking_work": planned,
                             "note": "a focus-listed measuring module is still planned as thinking work; its packages are refused at adoption"})


# ------------------------------------------------------------------------------------------------ the gate
ORDER = (("a_grounded", "collected data verified / grounded"), ("b_leakproof", "leak-proof (canaries, ties, frozen bench outside training)"),
         ("c_heldout", "every update judged held-out, significant gains only, regressions reverted"),
         ("d_no_drift", "fresh outside data, never only its own outputs"), ("e_choosing", "chooses by expected gain per CPU-hour, calibrated"),
         ("f_efficiency", "cost tracked: gain per CPU-hour / RAM / disk"), ("g_own_ruler", "cannot edit its own measuring code"))


def gate(state: Path, journal: Optional[Path] = None, now: Optional[float] = None, out_dir: Optional[Path] = None, write: bool = True) -> dict[str, Any]:
    """Compute every check from the real records; SELF_TEACH_TRUSTED only when all pass. Writes selfteach_gate.json/.md to `out_dir`
    (default state/creator/thinking) when `write`. Never raises for a missing record: a check that cannot be computed fails with the reason."""
    state = Path(state)
    now = _now() if now is None else now
    journal = journal if journal is not None else Path.home() / "Masterstock" / "JOURNAL.md"
    tj = _trust_json(state)
    frozen, ferr = frozen_bench(state)
    checks: dict[str, Any] = {}

    def run(name: str, f: Any) -> None:
        try:
            checks[name] = f()
        except Exception as e:                                       # noqa: BLE001 - an uncomputable check fails, never crashes the gate
            checks[name] = _check([f"could not be computed: {type(e).__name__}: {e}"[:300]], ["fix the record or the gate"], {})
    items: dict[str, list[Any]] = {}
    try:
        items = load_drill_items(state, journal)
    except Exception as e:                                           # noqa: BLE001
        checks["_load_error"] = f"{type(e).__name__}: {e}"
    run("a_grounded", lambda: check_grounded(state, tj, frozen))
    run("b_leakproof", lambda: check_leakproof(state, frozen, ferr, items))
    run("c_heldout", lambda: check_heldout(state, tj))
    run("d_no_drift", lambda: check_drift(state, tj, now))
    run("e_choosing", lambda: check_choosing(state, tj))
    run("f_efficiency", lambda: check_efficiency(state, now))
    run("g_own_ruler", check_own_ruler)
    ok = all(checks[k]["pass"] for k, _d in ORDER)
    rep = {"at": _iso(now), "self_teach_trusted": ok, "trust_json_at": tj.get("at"),
           "failing": [k for k, _d in ORDER if not checks[k]["pass"]], "checks": checks,
           "thresholds": {"MAX_TIE_SHARE": MAX_TIE_SHARE, "MAX_BENCH_IN_SELECT": MAX_BENCH_IN_SELECT, "MIN_HELPED_SIGNIFICANT": MIN_HELPED_SIGNIFICANT,
                          "MAX_OPEN_REGRESSIONS": MAX_OPEN_REGRESSIONS, "MIN_OUTSIDE_SHARE": MIN_OUTSIDE_SHARE, "MIN_COST_RECORDED": MIN_COST_RECORDED,
                          "PINNED_BENCH_HASH": PINNED_BENCH_HASH, "chooser": f"thinking.trust_of on fastpred '{CHOOSER_TOPIC}'"}}
    if write:
        d = Path(out_dir) if out_dir is not None else state / "thinking"
        d.mkdir(parents=True, exist_ok=True)
        (d / "selfteach_gate.json").write_text(json.dumps(rep, indent=1, sort_keys=True, default=str), encoding="utf-8")
        (d / "selfteach_gate.md").write_text(markdown(rep), encoding="utf-8")
    return rep


def markdown(rep: dict[str, Any]) -> str:
    lines = [f"# Self-teaching trust gate ({rep['at']})", "",
             f"**SELF_TEACH_TRUSTED: {'YES' if rep['self_teach_trusted'] else 'NO'}** - failing: {', '.join(rep['failing']) or 'none'}", "",
             "| check | pass | why not |", "|---|---|---|"]
    for k, desc in ORDER:
        c = rep["checks"].get(k) or {}
        lines.append(f"| {k} - {desc} | {'yes' if c.get('pass') else 'NO'} | {'; '.join(c.get('why_not') or []) or '-'} |")
    lines += ["", "## Fixes named by the failing checks", ""]
    for k, _desc in ORDER:
        for f in (rep["checks"].get(k) or {}).get("fix") or []:
            lines.append(f"- **{k}**: {f}")
    lines += ["", "## Evidence", ""]
    for k, _desc in ORDER:
        lines += [f"### {k}", "", "```json", json.dumps((rep["checks"].get(k) or {}).get("evidence"), indent=1, sort_keys=True, default=str)[:4000],
                  "```", ""]
    return "\n".join(lines)


if __name__ == "__main__":                                          # python -m creator.selfteach [state] [out_dir]
    import sys
    st = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "state" / "creator"
    od = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    r = gate(st, out_dir=od)
    print(json.dumps({"self_teach_trusted": r["self_teach_trusted"], "failing": r["failing"],
                      "why_not": {k: r["checks"][k]["why_not"] for k in r["failing"]}}, indent=1))
