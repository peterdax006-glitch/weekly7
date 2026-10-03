"""Write state/creator/BLUEPRINT.md: Nupen's vision, capability map, limiting factors, ranked roadmap with a MAKE-or-ASK decision per item, and the
current trust scores. On demand (python scripts/nupen_blueprint.py [--state DIR] [--out FILE]); read-only on the state except the one output file.

Every number is computed from the ledger / state files at run time; the only literals are the documented thresholds of creator.thinking
(ASK_S, MIN_N, ...). An ASK item carries a precise spec for the teacher. The mechanism (named, NOT triggered here): creator.goals.propose ->
teacher approves (creator.goals.approve) -> the packages Nupen cannot build itself reach the teacher as one file each in state/creator/handoffs/
(scripts/creator_swarm.announce, creator.kernel.HandoffWorker). This script never sends a handoff."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from creator import thinking as T  # noqa: E402


def _json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _latest(rows: list[dict[str, Any]], event: str) -> Optional[dict[str, Any]]:
    sel = [r for r in rows if r.get("event") == event]
    return sel[-1] if sel else None


def _hours(s: float) -> str:
    return f"{s / 3600:.1f} h" if s < 48 * 3600 else f"{s / 86400:.1f} days"


def build(state: Path, owner_dir: Optional[Path] = None) -> str:
    items = T.load_items(state)
    done = [i for i in items if i.resolved is not None]
    kl = T._jsonl(state / "kernel_log.jsonl")
    cons = T._jsonl(state / "constraints.jsonl")
    snap = _latest(cons, "snapshot") or {}
    sm = _json(state / "selfmodel.json") or {}
    obj = [r for r in T._jsonl(state / "ledger.jsonl") if r.get("rtype") == "Objective"]
    statement = (obj[0].get("data", {}).get("statement") if obj else "") or "(no objective in the ledger)"
    current_goal = (obj[-1].get("data", {}).get("statement") if len(obj) > 1 else "")
    rep = T.trust(state, write=False, owner_dir=owner_dir)

    secs = [i.seconds for i in done if i.seconds > 0]
    med_s = statistics.median(secs) if secs else T.SLOW_S
    recent = done[-20:]
    adopt_rate = max(0.05, sum(1 for i in recent if i.outcome == "ADOPTED") / len(recent)) if recent else 0.05
    out: list[str] = []
    w = out.append

    w("# Nupen blueprint")
    w(f"\nGenerated {dt.datetime.now().isoformat(timespec='minutes')} from `{state}` (ledger head {str((sm.get('active') or {}).get('ledger_head', ''))[:12]}). "
      "Every number below is computed from the ledger and state files; nothing is typed in.\n")
    an = rep["anticipation"]
    w("## 0. Anticipation rate (the top-level trust metric)\n")
    w("Share of the owner's directives that Nupen had already proposed, in writing, before the owner said them. The goal: the owner's talking is a waste.\n")
    w(f"- **Anticipation rate: {an.get('anticipated')} of {an.get('directives')} directives ({an.get('rate')})**; of the {an.get('eligible_directives')} given after "
      f"Nupen began writing proposals: {an.get('rate_eligible')}. Backlog of never-anticipated directives (training targets): {an.get('backlog')}.")
    dr = an.get("drill") or {}
    if dr.get("n"):
        w(f"- Prediction drill (next directive's content words from what existed just before it, precision@10): model {dr['precision_at_10_model']} vs "
          f"most-frequent-words baseline {dr['precision_at_10_baseline']}, gain {dr['gain']} {dr['gain_ci95']} over {dr['n']} directives.")
    th = an.get("thresholds") or {}
    w(f"- Matching rule: an earlier Nupen artefact must share >= {th.get('min_shared')} distinctive words the owner had not already said and reach match score "
      f"{th.get('match_score')}; matched pairs and the backlog (with the signals that preceded each directive) are in thinking/anticipation.json.\n")
    w("## 1. Vision\n")
    w(f"Objective (ledger): {statement}\n")
    h = (snap.get("headline") or {}).get("current") or {}
    if h:
        w(f"Measured improvement rate now: {h.get('adopted')} adopted of {h.get('cycles')} cycles in the last "
          f"{(snap.get('headline') or {}).get('window_h')} h; meta rate {(snap.get('headline') or {}).get('meta_rate_per_day')} per day. "
          f"History: {len(done)} resolved cycles, {sum(1 for i in done if i.outcome == 'ADOPTED')} adopted "
          f"(recent-20 adoption rate {adopt_rate:.0%}), median cycle {_hours(med_s)}.\n")
    w("Current owner priority (focus): thinking until its opinion is trustworthy, then self-management of everything else. "
      "Focus file says: `" + (json.dumps(_json(state / "focus.json")) if (state / "focus.json").exists() else "(none)") + "`.\n")

    w("## 2. Capability map\n")
    w("| id | capability | state | certainty | lines | tests (last result) | what limits it |")
    w("|---|---|---|---|---|---|---|")
    caps = sm.get("capabilities") or []
    for c in caps:
        res = c.get("last_results") or {}
        npass = sum(1 for v in res.values() if v == "PASS")
        lim = c.get("why") or ""
        if c.get("missing_modules"):
            lim = "missing " + ", ".join(c["missing_modules"])
        elif c.get("tests_missing"):
            lim = "no tests: " + ", ".join(c["tests_missing"])
        w(f"| {c.get('id')} | {c.get('name')} | {c.get('state')} | {c.get('uncertainty')} | {c.get('meaningful')} | {npass}/{len(res)} PASS | {lim} |")
    lims: dict[str, int] = {}
    for x in sm.get("limitations") or []:
        lims[x.get("kind", "?")] = lims.get(x.get("kind", "?"), 0) + 1
    if lims:
        w("\nSelf-model limitations: " + ", ".join(f"{k} {v}" for k, v in sorted(lims.items())) + ".")
    w("\nMeasured strength of Nupen's own judgement (the thinking drills, section 5): see the trust table; a capability whose own prediction "
      "is untrusted is not yet allowed to approve itself.\n")

    w("## 3. Limiting factors (constraint loop)\n")
    ranked = snap.get("ranked") or []
    if ranked:
        w("| rank | factor | value | unit | loss | score | remedy |")
        w("|---|---|---|---|---|---|---|")
        for k, r in enumerate(ranked[:6], 1):
            w(f"| {k} | {r.get('name')} | {r.get('value')} | {r.get('unit')} | {r.get('loss')} | {r.get('score')} | {r.get('remedy')} |")
        for m in snap.get("messages") or []:
            w(f"\n- {m}")
    else:
        w("No constraint snapshot yet (run scripts/nupen_constraints.py).")
    w("")

    # ---- roadmap: candidates with leverage and cost
    cand: list[dict[str, Any]] = []
    props = [r for r in T._jsonl(state / "goal_proposals.jsonl") if r.get("event") == "proposal"]
    for r in ranked:
        if (r.get("score") or 0) < 0.05:
            continue
        pk = 1
        for pr in props:
            if pr.get("source") == "constraint" and r.get("name", "") in str(pr.get("key", "")):
                pk = int((pr.get("cost") or {}).get("packages") or 1)
        est = pk * med_s / adopt_rate
        cand.append({"title": f"relieve limiting factor {r.get('name')} ({r.get('unit')})", "leverage": float(r.get("score") or 0),
                     "packages": pk, "est_s": est, "kind": "constraint", "remedy": r.get("remedy"), "metric": f"{r.get('name')} loss {r.get('loss')} -> 0",
                     "basis": f"{pk} package(s) x median cycle {_hours(med_s)} / recent adoption rate {adopt_rate:.0%}"})
    deps = sm.get("dependencies") or {}
    mod_of = {m: c for c in caps for m in c.get("present_modules", [])}
    users: dict[str, int] = {}
    for src, ds in deps.items():
        for d in ds:
            if d in mod_of and mod_of.get(src) is not mod_of[d]:
                users[mod_of[d]["id"]] = users.get(mod_of[d]["id"], 0) + 1
    mx = max(users.values()) if users else 1
    for c in caps:
        if c.get("state") != "TESTED" or c.get("tests_missing") or c.get("missing_modules"):
            lev = 0.3 * users.get(c["id"], 0) / mx
            cand.append({"title": f"{c['id']} {c['name']}: bring from {c.get('state')} to TESTED", "leverage": lev, "packages": 1,
                         "est_s": med_s / adopt_rate, "kind": "capability", "remedy": "work",
                         "metric": f"{c['id']}.tested passes; used by {users.get(c['id'], 0)} other capability module edges",
                         "basis": f"1 package x median cycle {_hours(med_s)} / recent adoption rate {adopt_rate:.0%}"})
    for t, v in rep["topics"].items():
        if v["trusted"]:
            continue
        sc = v["score"]
        need = max(0, T.MIN_N - sc.get("n", 0)) + max(0, T.MIN_LIVE - sc.get("n_live", 0))
        pt = [p.made_at for p in T.all_preds(state).get(t, [])]
        rate = (len(pt) / ((max(pt) - min(pt)) / 3600)) if len(pt) > 1 and max(pt) > min(pt) else 0.0
        est = (need / rate * 3600) if rate > 0 and need else 0.0
        cand.append({"title": f"earn trust in topic '{t}': " + "; ".join(v["why_not"][:2]), "leverage": 1.0, "packages": 0, "est_s": est, "kind": "trust",
                     "remedy": "drill", "metric": f"trust.json topics.{t}.trusted == true",
                     "basis": f"{need} more scored predictions at this topic's observed {rate:.2f}/hour (drills cost no model calls)" +
                     (" - its data source is slow: a denser source (more snapshots / goal decisions) is the lever" if est > T.ASK_S else "")})
    cand.sort(key=lambda c: (-c["leverage"], c["est_s"]))

    w("## 4. Roadmap (ranked by leverage on self-development capacity)\n")
    w("Leverage: constraint items = that factor's score; trust items = 1.0 (they gate independence, the owner's stated priority); capability items = "
      "0.3 x the share of cross-capability imports into it. Cost = packages x median measured cycle / recent adoption rate (an expected time to a "
      f"landed change); MAKE-or-ASK threshold {_hours(T.ASK_S)}.\n")
    w("| # | item | leverage | cost estimate | decision | basis |")
    w("|---|---|---|---|---|---|")
    asks = []
    for k, c in enumerate(cand, 1):
        dec = "ASK" if c["est_s"] > T.ASK_S and c["kind"] != "trust" else "MAKE"   # a drill accumulates evidence on its own: waiting is not building
        c["decision"] = dec
        if dec == "ASK":
            asks.append((k, c))
        w(f"| {k} | {c['title']} | {c['leverage']:.2f} | {_hours(c['est_s'])} | **{dec}** | {c['basis']} |")
    w("")
    if asks:
        w("### Handoff specs (ASK items)\n")
        for k, c in asks:
            w(f"- **#{k} {c['title']}**: done = `{c['metric']}`; audit clean; no regression. Hand over via goals.propose -> teacher approve -> "
              f"handoffs/ (not sent by this script). Estimated {_hours(c['est_s'])} for Nupen alone against the teacher's measured turnaround.")
    else:
        w("No item is estimated beyond the MAKE-or-ASK threshold from measured history; Nupen builds all of them itself. "
          "(The drills' cost topic is what keeps this honest: see its score below.)")
    w("")

    w("## 5. Trust scores (creator.thinking, computed now)\n")
    w("| topic | trusted | n (live) | Brier | best baseline | baseline Brier | gain vs best [95% CI] | calibration error | why not |")
    w("|---|---|---|---|---|---|---|---|---|")
    for t, v in rep["topics"].items():
        s = v["score"]
        if not s.get("n"):
            w(f"| {t} | no | 0 | - | - | - | - | - | {'; '.join(v['why_not'])} |")
            continue
        bb = s["brier_base_rate"] if s["best_baseline"] == "base" else s["brier_last_value"]
        w(f"| {t} | {'YES' if v['trusted'] else 'no'} | {s['n']} ({s['n_live']}) | {s['brier']} | {s['best_baseline']} | {bb} | {s['gain_vs_best']} "
          f"{s['gain_ci95']} | {s['ece']} | {'; '.join(v['why_not'])} |")
    for t in ("duration", "cost"):
        pe = rep["topics"][t].get("point_estimate") or {}
        if pe.get("n"):
            w(f"\nSeconds estimate '{t}' (mean |log error|, lower is better): model {pe.get('log_mae_model')}, running median {pe.get('log_mae_median')}, "
              f"last value {pe.get('log_mae_last')} over {pe['n']} cycles.")
    w("\nIndependence is granted per topic only when trusted (`creator.thinking.independent(state, topic)`); nothing is enabled by this report.\n")
    return "\n".join(out)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", default=str(ROOT / "state" / "creator"))
    ap.add_argument("--out", default="")
    ap.add_argument("--owner-dir", default="", help="folder with OWNER_MESSAGES.md and JOURNAL.md (default: Masterstock)")
    a = ap.parse_args(argv)
    state = Path(a.state)
    text = build(state, Path(a.owner_dir) if a.owner_dir else None)
    out = Path(a.out) if a.out else state / "BLUEPRINT.md"
    out.write_text(text, encoding="utf-8")
    from creator import anticipation
    anticipation.snapshot_blueprint(state, text)               # a dated copy of every blueprint: later directives are matched against earlier ones
    print(f"wrote {out} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
