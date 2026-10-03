"""Nupen's thinking benchmark: freeze, baseline, morning comparison. Read-only toward everything except state/creator/thinkbench/.

  python scripts/nupen_thinkbench.py --baseline [--no-model]   freeze the items (once), run them, write baseline_<ISO>.json
  python scripts/nupen_thinkbench.py --compare [--no-model]    re-run the SAME items now, print the morning report (plain English first), write now_<ISO>.json
  python scripts/nupen_thinkbench.py --freeze                  only freeze the items
Options: --state DIR (default <repo>/state/creator), --owner-dir DIR (default ~/Masterstock), --hours N (verdict window, default 12), --workers 1|2,
  --model fast|thinker, --think-model GGUF (measure a candidate thinker without touching the live setting), --repo DIR (with a --state copy)."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from creator import thinkbench as TB  # noqa: E402
from creator import thinking as T  # noqa: E402

NAMES = {"a_prediction": "Prediction (will it happen?)", "b_judgment": "Judgment (local model forecasts)", "c_reasoning": "Reasoning (multiple choice from history)",
         "d_planning": "Planning (what to do next)"}


def _ci(x: Any) -> str:
    return "n/a" if not x or x[0] is None else f"[{x[0]:+.3f}, {x[1]:+.3f}]" if len(x) > 1 and x[1] is not None else "[n/a]"


def part_line(k: str, p: dict[str, Any]) -> str:
    sc = f"{p['score']:.3f}" if p.get("score") is not None else "n/a"
    bl = f"{p['baseline']:.3f}" if p.get("baseline") is not None else "n/a"
    gain = f"{p['gain']:+.3f} {_ci(p.get('gain_ci95'))}" if p.get("gain") is not None else "n/a"
    return f"  {NAMES.get(k, k):42s} n={p['n']:<4d} score={sc}  baseline={bl}  gain over baseline={gain}   ({p['metric']})"


def extras(state: Path, owner_dir: Path, hours: float) -> list[str]:
    out: list[str] = []
    try:
        from creator import anticipation as A
        r = A.report(state, owner_dir, write=False)
        out.append(f"Anticipation rate (owner directives Nupen had already prepared): {r.get('rate')} ({r.get('anticipated')} of {r.get('eligible_directives')} eligible, "
                   f"{r.get('directives')} directives in all)")
    except Exception as e:                                           # noqa: BLE001
        out.append(f"Anticipation: unavailable ({type(e).__name__})")
    try:
        tj = json.loads((state / "trust.json").read_text(encoding="utf-8"))
        for k, v in tj.items():
            if isinstance(v, dict) and k not in ("anticipation", "drills", "judgment"):
                out.append(f"Trust topic {k}: trusted={v.get('trusted')} n={v.get('n')} {v.get('why_not') or ''}"[:200])
        for sec in ("drills", "judgment"):
            for k, v in (tj.get(sec) or {}).items():
                if isinstance(v, dict):
                    out.append(f"Trust {sec}/{k}: trusted={v.get('trusted')} n={v.get('n')}"[:200])
    except Exception as e:                                           # noqa: BLE001
        out.append(f"trust.json: unavailable ({type(e).__name__})")
    try:
        m = importlib.import_module("creator.goal_constraint_learning_signal")
        try:
            out.append(f"Learning-signal assess: {str(m.assess(state))[:300]}")
        except TypeError:
            out.append(f"Learning-signal assess: {str(m.assess())[:300]}")
    except Exception as e:                                           # noqa: BLE001
        out.append(f"Learning-signal assess: unavailable ({type(e).__name__}: module not in this checkout or not callable)")
    now = time.time()
    items = [i for i in T.load_items(state) if i.resolved is not None and now - i.resolved <= hours * 3600]
    cnt: dict[str, int] = {}
    for i in items:
        cnt[i.outcome] = cnt.get(i.outcome, 0) + 1
    out.append(f"Kernel verdicts in the last {hours:g} h: {sum(cnt.values())} " + str(dict(sorted(cnt.items()))))
    try:
        from creator import device as DEV
        cur = json.loads((DEV.runtime_dir() / "lmckpt" / "current.json").read_text(encoding="utf-8"))
        out.append(f"Language model: {cur.get('bpb'):.3f} bits/byte (95% {cur.get('lo'):.3f}-{cur.get('hi'):.3f}) at step {cur.get('step')}, {cur.get('tokens')} tokens, "
                   f"{cur.get('params')} parameters (lower is better)")
        log = (state / "lm_train.log").read_text(encoding="utf-8", errors="replace").splitlines()
        out.append("LM training log, last line: " + (log[-1] if log else "-"))
    except Exception as e:                                           # noqa: BLE001
        out.append(f"Language model: unavailable ({type(e).__name__})")
    return out


def summary_sentences(cmp: dict[str, Any]) -> list[str]:
    better = [k for k, v in cmp["parts"].items() if v["verdict"] == "SIGNIFICANTLY BETTER"]
    worse = [k for k, v in cmp["parts"].items() if v["verdict"] == "SIGNIFICANTLY WORSE"]
    s = []
    if not cmp["same_items"]:
        s.append("WARNING: the item sets differ, so this comparison is not valid.")
    if better:
        s.append("Nupen thinks measurably better than at the baseline in: " + ", ".join(NAMES.get(k, k) for k in better) + ".")
    if worse:
        s.append("Nupen is measurably WORSE in: " + ", ".join(NAMES.get(k, k) for k in worse) + ".")
    if not better and not worse:
        s.append("No part moved by more than the noise of this small benchmark: Nupen does not (yet) measurably think better or worse than at the baseline.")
    s.append("A change counts as real only when the 95% interval of the paired item-by-item difference excludes zero; with these item counts that needs a sizable change.")
    return s


def _same_model(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Same model file and size (the 'role' label and mtime are not the model)."""
    return (a.get("file"), a.get("bytes")) == (b.get("file"), b.get("bytes"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--state", default=str(ROOT / "state" / "creator"))
    ap.add_argument("--owner-dir", default=str(Path.home() / "Masterstock"))
    ap.add_argument("--hours", type=float, default=12.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--model", choices=("fast", "thinker"), default="fast",
                    help="fast = the fixed benchmark model (comparable runs); thinker = device think_model (labelled in versions)")
    ap.add_argument("--think-model", default="",
                    help="with --model thinker: measure THIS gguf as the thinker (a candidate); the live device setting is not touched")
    ap.add_argument("--repo", default="", help="the repository the drills read (default: two levels above --state); use with a --state copy")
    a = ap.parse_args()
    state, owner = Path(a.state), Path(a.owner_dir)
    repo = Path(a.repo) if a.repo else state.parents[1]
    if a.think_model:
        if a.model != "thinker" or not Path(a.think_model).is_file():
            print("--think-model needs --model thinker and an existing .gguf file")
            return 2
        TB.THINK_MODEL_OVERRIDE = Path(a.think_model)

    def prog(i: int, n: int) -> None:
        if i % 10 == 0 or i == n:
            print(f"  model calls {i}/{n}", flush=True)
    if a.freeze or a.baseline or a.compare:
        frozen = TB.freeze(state, repo, owner)
        print(f"frozen items: a={len(frozen['a'])} b={len(frozen['b'])} c={len(frozen['c'])} d={len(frozen['d'])} hash={frozen['hash'][:12]}")
    if a.freeze and not (a.baseline or a.compare):
        return 0
    if a.baseline:
        res = TB.run(state, repo, owner, frozen, use_model=not a.no_model, workers=a.workers, progress=prog, model=a.model)
        p = TB.save(state, res, "baseline")
        print(f"baseline written: {p}  (runtime {res['runtime_s']} s)")
        for k, v in res["parts"].items():
            print(part_line(k, v))
        return 0
    if a.compare:
        bases = sorted(TB.bench_dir(state).glob("baseline_*.json"))
        if not bases:
            print("no baseline_*.json: run --baseline first")
            return 2
        base = json.loads(bases[0].read_text(encoding="utf-8"))
        res = TB.run(state, repo, owner, frozen, use_model=not a.no_model, workers=a.workers, progress=prog, model=a.model)
        p = TB.save(state, res, "now")
        cmp = TB.compare(base, res)
        print("\n=== NUPEN THINKING REPORT ===")
        for line in summary_sentences(cmp):
            print(line)
        print(f"\nBaseline {bases[0].name} vs now {p.name} (runtime {res['runtime_s']} s)")
        for k, v in cmp["parts"].items():
            print(f"  {NAMES.get(k, k):42s} baseline={v['score_before']}  now={v['score_now']}  paired change={v['change']} {_ci(v['change_ci95'])} "
                  f"over {v['n_paired']} items -> {v['verdict']}")
        print("\nNow, in detail:")
        for k, v in res["parts"].items():
            print(part_line(k, v))
        mv, bv = res["versions"], base.get("versions", {})
        changed = [f for f, h in mv["code_sha"].items() if bv.get("code_sha", {}).get(f) != h]
        print(f"\nWhat changed under the benchmark: code {changed or 'nothing'}; model same={_same_model(mv.get('local_model', {}), bv.get('local_model', {}))} (now {mv.get('local_model', {}).get('file')}); "
              f"procedure same={mv['procedure_sha'] == bv.get('procedure_sha')}; judgment strategies now {mv['judgment_strategies']}")
        print("\nOther signals:")
        for line in extras(state, owner, a.hours):
            print("  " + line)
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
