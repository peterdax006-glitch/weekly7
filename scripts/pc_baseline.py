"""P0.9 PC baseline of the full coding pipeline (MASTER_BLUEPRINT 12 / 0.9): the denominator of the 'dozens of times faster' north star.

  python scripts/lowprio.py python scripts/pc_baseline.py suite  --bakeoff-dir <agent_bakeoff dir> --out <suite dir>     (once; suite stays OUTSIDE the repo)
  python scripts/lowprio.py python scripts/pc_baseline.py run    --suite <suite dir> --out <run dir> [--limit N] [--only id ...]
  python scripts/lowprio.py python scripts/pc_baseline.py calib  --heldout <heldout_CALIB.jsonl> --out <run dir> [--limit N]
  python scripts/pc_baseline.py report --out <run dir>
  R2/R4 switches of the fast path (any coder slot; the model is the CODER slot spec):
    ... run --fast --repair none|line|regen|line+regen|regen+line --repair-rounds N [--coder-moe | --coder-gguf <file>] [--lean-prefill]

The suite = the agent bake-off's held-out tasks (function tasks from the export eval split, feature/bug tasks on the minishop app) with their
hidden tests; nothing here trains on them. `run` drives every task through creator.team with the REAL slot actors of creator.slotteam:
SPEC -> LOCATE -> PACK -> PLAN -> CODE -> SAFETY -> VALIDATE -> (DEBUG_PINPOINT -> DEBUG_FIX -> SAFETY -> VALIDATE)x<=2 -> CONFIDENCE, scored
afterwards by the hidden tests. ONE llama-server runs at a time, so the steps run stage by stage (THINKER, CODER, CHECKER) over all tasks;
the per-step times are the same as in a resident-slot team and the three server loads are reported separately. Every dispatched step is one
event on the metrics bus (creator.slowpath)."""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from creator import repair as RP  # noqa: E402
from creator import router as RT  # noqa: E402
from creator import slotteam as SL  # noqa: E402
from creator import slowpath as SP  # noqa: E402
from creator import team as TM  # noqa: E402

MAX_DEBUG = 2
MAX_FAST_DEBUG = SL.MAX_FAST_DEBUG
Z_LINE = "Z = 1.96" + chr(10) * 2
CONF_THRESHOLD = 0.5
FN_PRE = "from __future__ import annotations\n\nimport math\nfrom typing import Any, Optional, Sequence\n\n"
CHECK = r'''
import json, math, sys
sys.path.insert(0, ".")
tests = json.loads(open(sys.argv[1], encoding="utf-8").read()); name = sys.argv[2]
def norm(x): return json.loads(json.dumps(x, default=list))
def close(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try: return (a is None) == (b is None) and math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-9)
        except Exception: return False
    if isinstance(a, list) and isinstance(b, list): return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict): return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    return a == b
ok, fails = 0, []
try:
    import solution
    f = getattr(solution, name)
except Exception as e:
    print(json.dumps({"passed": 0, "total": len(tests), "fails": ["import: " + repr(e)[:200]]})); sys.exit(0)
for t in tests:
    try: got = norm(f(*t["args"]))
    except Exception as e: got = "raised " + type(e).__name__
    if close(got, t["expect"]): ok += 1
    else: fails.append(f"{t['args']!r:.60} -> {got!r:.40} != {t['expect']!r:.40}")
print(json.dumps({"passed": ok, "total": len(tests), "fails": fails[:3]}))
'''
EXAMPLES_TEST = '''import json, math
import pytest
from solution import {name}
CASES = json.loads({cases!r})
def _norm(x): return json.loads(json.dumps(x, default=list))
def _close(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try: return (a is None) == (b is None) and math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-9)
        except Exception: return False
    if isinstance(a, list) and isinstance(b, list): return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    return a == b
@pytest.mark.parametrize("case", CASES)
def test_example(case):
    got = _norm({name}(*case["args"]))
    assert _close(got, case["expect"]), f"{{case['args']!r:.100}} -> {{got!r:.60}} != {{case['expect']!r:.60}}"
'''


def stratified(tasks: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """n tasks spread evenly over (family, router class) strata, deterministic (sorted by stratum then id; every k-th)."""
    if n >= len(tasks):
        return tasks
    key = lambda t: (t["family"], RT.task_class(RT.features(t)), t["id"])          # noqa: E731
    srt = sorted(tasks, key=key)
    pick = sorted({int((i + 0.5) * len(srt) / n) for i in range(n)})
    return [srt[i] for i in pick]


def runtime() -> Path:
    from creator import generator as G
    return G.RUNTIME


def jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


# ---------------------------------------------------------------------------------------------------------------- suite
def build_suite(a: argparse.Namespace) -> int:
    bb = Path(a.bakeoff_dir)
    sys.path.insert(0, str(bb))
    from app_tasks import APP_TASKS                                                   # type: ignore[import-not-found]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "app_base").exists():
        shutil.rmtree(out / "app_base")
    shutil.copytree(bb / "app_base", out / "app_base", ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    tasks: list[dict[str, Any]] = []
    for t in APP_TASKS:
        if t["family"] == "app":
            tasks.append({"id": t["id"], "family": "app", "request": t["prompt"], "query": t["prompt"], "accept": t["accept"]})
    gd = runtime() / "gpuday"
    base = [r for r in jsonl(gd / "export" / "rl_tasks.jsonl") if r["split"] == "eval"]
    seen = {r["name"] for r in base}
    plus = [r for r in jsonl(gd / "export_plus" / "rl_tasks.jsonl") if r["split"] == "eval" and r["name"] not in seen]
    plus = [r for r in plus if re.search(r'"""[^"]{20,}', r["prompt"]) and len(r["tests"]) >= 4]
    plus.sort(key=lambda r: r["id"])
    for r in base + plus[: a.extra_fn]:
        m = re.search(r"\n\n(def .*?)(\n\nExamples:\n.*)$", r["prompt"], re.S)
        if not m:
            continue
        sig, ex = m.group(1), m.group(2).strip().split("\n", 1)[1]
        tasks.append({"id": "fn." + r["name"], "family": "fn", "name": r["name"], "query": f"{r['name']} {sig[:300]}",
                      "request": f"Write the Python function `{r['name']}` described below (standard library only; return the function definition "
                                 f"in one ```python block).\n\n{sig}",
                      "examples": ex + "\n(`null` means None; lists may stand for tuples.)", "stub": FN_PRE + (Z_LINE if re.search(r"=\s*Z[,)]", sig) else "") + sig.rstrip() + "\n    raise NotImplementedError\n",
                      "tests": r["tests"], "n_visible": ex.count("\n") + 1 - 1})
    (out / "tasks.json").write_text(json.dumps({"tasks": tasks}, indent=1), encoding="utf-8")
    print(f"suite: {len(tasks)} tasks ({sum(t['family'] == 'app' for t in tasks)} app, {sum(t['family'] == 'fn' for t in tasks)} fn) -> {out}")
    return 0


# ---------------------------------------------------------------------------------------------------------------- workspaces
def make_ws(task: dict[str, Any], suite: Path, ws: Path) -> None:
    if ws.exists():
        shutil.rmtree(ws, ignore_errors=True)
    if task["family"] == "fn":
        (ws / "tests").mkdir(parents=True)
        (ws / "solution.py").write_text(task["stub"], encoding="utf-8")
        vis = task["tests"][: max(1, min(2, len(task["tests"])))]
        (ws / "tests" / "test_examples.py").write_text(EXAMPLES_TEST.format(name=task["name"], cases=json.dumps(vis)), encoding="utf-8")
    else:
        shutil.copytree(suite / "app_base", ws, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))


def grade(task: dict[str, Any], suite: Path, ws: Path, py: str) -> dict[str, Any]:
    """The hidden tests, run after the pipeline stopped (grader time is not pipeline time)."""
    t0 = time.perf_counter()
    try:
        if task["family"] == "fn":
            tf = ws.parent / (ws.name + "_hidden.json")
            tf.write_text(json.dumps(task["tests"]), encoding="utf-8")
            cf = ws.parent / "grade_fn.py"
            cf.write_text(CHECK, encoding="utf-8")
            p = subprocess.run([py, str(cf), str(tf), task["name"]], cwd=ws, capture_output=True, text=True, timeout=90)
            d = json.loads(p.stdout.strip().splitlines()[-1])
            res = {"passed": d["passed"] == d["total"], "cases": f"{d['passed']}/{d['total']}", "detail": d["fails"]}
        else:
            for f in (suite / "app_base" / "tests").glob("*.py"):
                shutil.copy2(f, ws / "tests" / f.name)
            (ws / "tests" / "test_zz_acceptance.py").write_text(task["accept"], encoding="utf-8")
            p = subprocess.run([py, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], cwd=ws, capture_output=True, text=True, timeout=120)
            res = {"passed": p.returncode == 0, "detail": [x for x in p.stdout.splitlines() if x.strip()][-4:]}
    except Exception as e:                                                          # noqa: BLE001 - a hang or crash is a fail
        res = {"passed": False, "detail": [f"grader: {type(e).__name__}"]}
    res["grade_s"] = round(time.perf_counter() - t0, 2)
    return res


# ---------------------------------------------------------------------------------------------------------------- the run
class Run:
    def __init__(self, a: argparse.Namespace) -> None:
        self.suite, self.out = Path(a.suite), Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)
        all_t = json.loads((self.suite / "tasks.json").read_text(encoding="utf-8"))["tasks"]
        if a.only:
            all_t = [t for t in all_t if t["id"] in a.only]
        self.tasks = all_t[: a.limit] if a.limit else all_t
        if getattr(a, "subset", 0):
            self.tasks = stratified(all_t, int(a.subset))
        self.state_f = self.out / "run_state.json"
        self.st: dict[str, Any] = json.loads(self.state_f.read_text(encoding="utf-8")) if self.state_f.exists() else {"tasks": {}, "loads": []}
        SP.set_state(self.out / "state")
        specs = SL.default_specs(threads=int(getattr(a, "threads", 6) or 6))
        if getattr(a, "coder_moe", False):
            specs = SL.moe_specs(specs, threads=int(getattr(a, "threads", 6) or 6))
        if getattr(a, "coder_gguf", ""):                                       # any other coder model: same slot, prompts and grammars
            from creator import modelpool as MP
            old = specs["CODER"]
            specs["CODER"] = MP.SlotSpec("CODER", Path(a.coder_gguf), threads=old.threads, ctx=old.ctx, max_tokens=old.max_tokens, grammars=old.grammars)
        SL.LEAN = bool(getattr(a, "lean_prefill", False))
        SL.PREFIX_FIRST = bool(getattr(a, "prefix_first", False))
        self.conf_mode = getattr(a, "conf", "auto") or "auto"             # auto: CONFIDENCE only where the tests did not decide; always: every task (the old fast path)
        self.route = bool(getattr(a, "route", False))
        self.class_caps: dict[str, int] = json.loads(Path(a.class_caps).read_text(encoding="utf-8")) if getattr(a, "class_caps", "") else {}
        self.router = RT.Router({"C06": 58.0, "C17": 22.0}, Path(a.router_state) if getattr(a, "router_state", "") else None)
        self.pack_k = int(getattr(a, "pack_k", 0) or 0)
        self.pack_tokens = int(getattr(a, "pack_tokens", 0) or 0)
        self.repair = getattr(a, "repair", "none") or "none"
        self.repair_rounds = int(getattr(a, "repair_rounds", 3))
        self.max_fast_debug = int(a.max_fast_debug) if getattr(a, "max_fast_debug", None) is not None else MAX_FAST_DEBUG
        self.runner = SL.SlotRunner(specs, self.out / "server")
        self.teams: dict[str, SL.MeasuredTeam] = {}
        self.py = sys.executable
        self.fast = bool(getattr(a, "fast", False))
        self.reuse_db = getattr(a, "reuse_db", "") or ""

    def save(self) -> None:
        self.st["loads"] = self.runner.loads + [x for x in self.st.get("loads", []) if x not in self.runner.loads]
        self.state_f.write_text(json.dumps(self.st, indent=1), encoding="utf-8")

    def sink(self, actor: str, **kw: Any) -> None:
        SP.event(actor, **kw)

    def team(self, t: dict[str, Any]) -> SL.MeasuredTeam:
        tid = t["id"]
        if tid in self.teams:
            return self.teams[tid]
        from creator.tools import index as IX
        ws = self.out / "ws" / tid
        ts = self.st["tasks"].setdefault(tid, {"ids": {}, "done": [], "debug": 0})
        if "ws" not in ts.get("done", []) and not ts.get("ws"):
            make_ws(t, self.suite, ws)
            ts["ws"] = True
        tm = SL.MeasuredTeam(self.out / "tasks" / tid, SL.actors(self.runner), goal_id=tid, sink=self.sink)
        tm.fast = self.fast
        (self.out / "idx").mkdir(parents=True, exist_ok=True)
        tm.ctx = SL.ToolContext(ws=ws, ix=IX.Index(ws, db=self.out / "idx" / f"{tid}.sqlite"), python=self.py,
                                default_path="solution.py" if t["family"] == "fn" else "")
        if not tm.ctx.orig:
            tm.ctx.snapshot()
        if self.pack_k:
            tm.ctx.k = self.pack_k
        if self.pack_tokens:
            tm.ctx.pack_tokens = self.pack_tokens
        if "task" not in ts["ids"]:
            ts["ids"]["task"] = tm.board.put(tid, "task", json.dumps(t, sort_keys=True))
        self.teams[tid] = tm
        return tm

    def env(self, tid: str, step: str, inputs: list[str], success: str = "") -> TM.Envelope:
        prof = SL.PROFILES.get(step)
        budget = {"max_tok": prof.max_tokens if prof else 400, "max_s": prof.max_s if prof else 150}
        if self.fast and step in ("CODE", "DEBUG_FIX"):
            budget["max_tok"] = self.class_caps.get(self.cur_class) or SL.CODE_CAP["fn" if self.cur_family == "fn" else "app"]
        elif self.fast and step == "CONFIDENCE":
            budget["max_s"] = 20
        return TM.Envelope(tid, step, inputs=inputs, success_test=success, budget=budget)

    cur_family = ""
    cur_class = ""
    cur_slot = ""

    def step(self, tm: SL.MeasuredTeam, step: str, inputs: list[str], kind: str, success: str = "") -> tuple[str, str]:
        env = self.env(tm.goal() or "", step, inputs, success)
        if self.cur_slot and step == "CODE":
            env.constraints = [SL.COUPLED_SLOT + self.cur_slot]
        out = tm.dispatch(env)
        return tm.board.put(tm.goal() or "", kind, out), out

    # --- reuse (R3): proven code before any generation; function tasks only
    def stage_reuse(self, t: dict[str, Any]) -> None:
        ts = self.st["tasks"].setdefault(t["id"], {"ids": {}, "done": [], "debug": 0})
        if "reuse" in ts or t["family"] != "fn":
            return
        from creator import slowpath as _sp
        from creator.tools import reuse as RU
        t0 = time.perf_counter()
        if not hasattr(self, "_rix"):
            self._rix = RU.ReuseIndex(self.reuse_db, RU.default_guard(ROOT))
        r = RU.prepare(self._rix, t, ROOT, python=self.py)
        ts["reuse"] = {k: v for k, v in r.items() if k != "code"}
        if r["mode"] in ("direct", "adapt"):
            t["stub"] = r["code"]                                       # the starting code every later step (and make_ws) sees
        if r["mode"] == "direct":
            tm = self.team(t)
            (tm.ctx.ws / "solution.py").write_text(r["code"], encoding="utf-8")
            ts.update({"visible_first": True, "visible_pass": True, "stop_reason": "reuse-direct", "confidence": None, "verdict": None})
            ts["done"] += ["think", "code", "check"]
        _sp.event("reuse", goal_id=t["id"], step="REUSE", wall_s=time.perf_counter() - t0, model=False)
        self.save()

    # --- stages
    def stage_think(self, t: dict[str, Any]) -> None:
        tm = self.team(t)
        ts = self.st["tasks"][t["id"]]
        if "think" in ts["done"]:
            return
        ids = ts["ids"]
        if self.fast:                       # routing: the code tools extract the spec; SPEC / PLAN are not run (no THINKER model at all)
            if t["family"] != "fn":
                ids["locate"], _ = self.step(tm, "LOCATE", [ids["task"]], "locate")
                ids["pack"], _ = self.step(tm, "PACK", [ids["task"], ids["locate"]], "pack")
            ids.setdefault("pack", ids["task"])
            ts["done"].append("think")
            self.save()
            return
        ids["spec"], _ = self.step(tm, "SPEC", [ids["task"]], "spec")
        ids["locate"], _ = self.step(tm, "LOCATE", [ids["task"]], "locate")
        ids["pack"], _ = self.step(tm, "PACK", [ids["task"], ids["locate"]], "pack")
        ids["plan"], _ = self.step(tm, "PLAN", [ids["task"], ids["spec"], ids["pack"]], "plan")
        ts["done"].append("think")
        self.save()

    def stage_code(self, t: dict[str, Any], coder: str = "C17") -> None:
        """One CODE attempt chain. coder C06 = the router's cheap first attempt (base 0.6B, no debug rounds): a visible pass finishes the task,
        a failure leaves it for the 1.7B stage, which starts again from the pristine workspace."""
        tm = self.team(t)
        ts = self.st["tasks"][t["id"]]
        if "code" in ts["done"]:
            return
        ids, c, gid = ts["ids"], tm.ctx, t["id"]
        self.cur_family = t["family"]
        self.cur_class = ts.get("route", {}).get("class") or RT.task_class(RT.features(t))
        self.cur_slot = "CODER06" if coder == "C06" else ""
        small = coder == "C06"
        ts["class"] = self.cur_class
        make_ws(t, self.suite, c.ws)                                  # a resumed run starts the task from the pristine workspace
        c.snapshot()
        ids["diff"], code = self.step(tm, "CODE", [ids["task"], ids.get("spec", ids["task"]), ids.get("plan", ids["task"]), ids["pack"]], "diff")
        answers = [code]
        ok, why, val = False, "", ""
        for rnd in range(MAX_DEBUG + 1):
            sid, safe = self.step(tm, "SAFETY", [ids["diff"]], "safety")
            if safe.startswith("BLOCK") or not answers[-1].strip():
                why = "blocked" if safe.startswith("BLOCK") else "empty answer"
                break
            vid, val = self.step(tm, "VALIDATE", [ids["diff"]], "result", success="visible tests")
            if rnd == 0:
                ts["visible_first"] = val.startswith("PASS")
            if val.startswith("PASS"):
                ok = True
                break
            if rnd == (0 if small else self.max_fast_debug if self.fast else MAX_DEBUG):
                if self.fast and self.repair != "none" and not ok and not small:
                    ok, val, why = self.repair_loop(t, tm, ts, c, val)
                break
            ids["failure"] = tm.board.put(gid, "failure", val)
            c.tb_lines, c.pin_scores = [], []
            if val.startswith("FAIL FAILED"):
                ids["pinpoint"], _ = self.step(tm, "DEBUG_PINPOINT", [ids["failure"]], "pinpoint")
            else:
                ids["pinpoint"] = tm.board.put(gid, "pinpoint", "")
            if self.fast and not SL.debug_worthwhile(c):
                why = "not pinned: no debug"
                break
            ts["debug"] = rnd + 1
            ids["code"] = tm.board.put(gid, "code", self.current_code(t, c) or (tm.board.get(ids["pack"]) or {}).get("body", ""))
            ids["diff"], fix = self.step(tm, "DEBUG_FIX", [ids["task"], ids["code"], ids["failure"], ids["pinpoint"]], "diff")
            answers.append(fix)
            if not fix.strip():
                why = "empty fix"
                break
        ts["decided"] = bool(SL.tests_decided(c, val) or why in ("blocked", "empty answer"))
        ts["lines_added"] = sum(1 for x in c.unified_diff().splitlines() if x.startswith("+") and not x.startswith("+++") and x[1:].strip())
        ts["visible_pass"], ts["stop_reason"] = ok, why or ("pass" if ok else "debug rounds used")
        if self.fast:
            pin = (tm.board.get(ids["pinpoint"]) or {}).get("body", "") if ids.get("pinpoint") else ""
            ids["answer"] = tm.board.put(gid, "answer", SL.confidence_summary(t, c.unified_diff(), "PASS" if ok else (val or why), pin, ts.get("debug", 0)))
        else:
            ids["answer"] = tm.board.put(gid, "answer", "\n".join(answers)[:3000])
        if small:
            last = tm.last_call or {}
            ts["c06"] = {"visible_pass": ok, "lines_added": ts["lines_added"], "stop": ts["stop_reason"], "decided": ts["decided"]}
            self.router.observe(self.cur_class, "C06", ok)
            if not ok:
                self.cur_slot = ""
                self.save()
                return
        ts["coder"] = "C06" if small else "C17"
        ts["done"].append("code")
        self.save()

    def repair_loop(self, t: dict[str, Any], tm: SL.MeasuredTeam, ts: dict[str, Any], c: SL.ToolContext, val: str) -> tuple[bool, str, str]:
        """R2: test-driven repair on the VISIBLE examples only. Each round = one strategy (creator.repair.plan_rounds): 'line' edits the best
        untried pinpointed line (prefilled SEARCH, <=72 tokens), 'regen' runs CODE again with the failing example in the prompt. A round that
        does not lower the failing-test score is rolled back; two rounds without progress stop the loop."""
        gid, ids, fn = t["id"], ts["ids"], t["family"] == "fn"
        rounds, tried = RP.plan_rounds(self.repair, self.repair_rounds), set()
        best, best_snap, hist, log = RP.score(val), RP.py_snapshot(c.ws), [RP.score(val)], []
        ts["repair"] = {"strategy": self.repair, "rounds": log, "fixed": False}
        ok, why, regens = False, "repair rounds used", 0
        for strat in rounds:
            if strat == "line" and best >= 1000:                         # nothing applied yet: there is no suspect line, only regeneration helps
                if "regen" not in self.repair:
                    why = "no applied code to pinpoint"
                    break
                strat = "regen"
            ids["failure"] = tm.board.put(gid, "failure", val)
            row: dict[str, Any] = {"strategy": strat, "before": best}
            if strat == "line":
                c.tb_lines, c.pin_scores = [], []
                ids["pinpoint"], pin = self.step(tm, "DEBUG_PINPOINT", [ids["failure"]], "pinpoint")
                cands = RP.candidates(pin, c.ws, tried)
                if not cands:
                    row["result"] = "no candidate"
                    log.append(row)
                    why = "no pinpointed line"
                    break
                f, ln, txt = cands[0]
                tried.add((f, ln))
                tg = tm.board.put(gid, "target", json.dumps({"file": f, "line": ln, "text": txt, "fail": val[:600], "with_file": not fn}, sort_keys=True))
                ids["diff"], fix = self.step(tm, "DEBUG_FIX", [ids["task"], tg], "diff")
                row["target"] = f"{f}:{ln}"
            else:
                make_ws(t, self.suite, c.ws)
                c.snapshot()
                regens += 1
                hint = tm.board.put(gid, "hint", RP.regen_hint(val, regens))
                self.cur_family = t["family"]
                ids["diff"], fix = self.step(tm, "CODE", [ids["task"], ids.get("spec", ids["task"]), ids.get("plan", ids["task"]), ids["pack"], hint], "diff")
            _, safe = self.step(tm, "SAFETY", [ids["diff"]], "safety")
            if safe.startswith("BLOCK") or not fix.strip():
                new, nval = 1000, "FAIL apply: blocked or empty"
            else:
                _, nval = self.step(tm, "VALIDATE", [ids["diff"]], "result", success="visible tests")
                new = RP.score(nval)
            row.update(after=new, result="pass" if new == 0 else ("better" if new < best else "no gain"))
            log.append(row)
            hist.append(new)
            if new < best:
                best, best_snap, val = new, RP.py_snapshot(c.ws), nval
            else:
                RP.py_restore(c.ws, best_snap)
            if new == 0:
                ok, why = True, "pass after repair"
                ts["repair"]["fixed"] = True
                break
            if RP.stop_early(hist):
                why = "no progress"
                break
        return ok, val, why

    @staticmethod
    def current_code(t: dict[str, Any], c: SL.ToolContext) -> str:
        if t["family"] == "fn":
            return (c.ws / "solution.py").read_text(encoding="utf-8")
        files = sorted({ln[6:].split("\t")[0].strip() for ln in c.unified_diff().splitlines() if ln.startswith("+++ b/")})
        return "\n\n".join(f"FILE: {f}\n```\n{(c.ws / f).read_text(encoding='utf-8')[:5000]}\n```" for f in files)

    def stage_check(self, t: dict[str, Any]) -> None:
        tm = self.team(t)
        ts = self.st["tasks"][t["id"]]
        if "check" in ts["done"]:
            return
        ids = ts["ids"]
        if self.conf_mode == "auto" and ts.get("decided"):              # the tests decided: no second model call (and, if all are decided, no CHECKER server)
            ts["confidence"], ts["verdict"] = None, "tests"
            ts["done"].append("check")
            self.save()
            return
        _, out = self.step(tm, "CONFIDENCE", [ids["task"], ids.get("plan", ids["task"]), ids["pack"], ids["answer"]], "confidence")
        m = re.search(r"([01](?:\.\d+)?)", out)
        ts["confidence"] = float(m.group(1)) if m else None
        ts["verdict"] = None if ts["confidence"] is None else ("correct" if ts["confidence"] >= CONF_THRESHOLD else "incorrect")
        ts["done"].append("check")
        self.save()

    def run(self) -> int:
        t_all = time.perf_counter()
        if self.reuse_db:
            for t in self.tasks:
                self.stage_reuse(t)
        if self.route:
            for t in self.tasks:
                ts = self.st["tasks"].setdefault(t["id"], {"ids": {}, "done": [], "debug": 0})
                if "route" not in ts:
                    r = self.router.route(t, {"C06", "C17"})
                    ts["route"] = {"class": r["class"], "start": r["start"], "expected_s": r["expected_s"], "p_solve": r["p_solve"]}
            self.save()
        stages = [("think", "THINKER", self.stage_think)]
        if self.route:
            stages.append(("code06", "CODER06", lambda t: self.stage_code(t, "C06")))
        stages += [("code", "CODER", self.stage_code), ("check", "CHECKER", self.stage_check)]
        for name, slot, fn in stages:
            done_name = "code" if name == "code06" else name
            todo = [t for t in self.tasks if done_name not in self.st["tasks"].get(t["id"], {}).get("done", [])]
            if name == "code06":
                todo = [t for t in todo if self.st["tasks"][t["id"]].get("route", {}).get("start") == "C06" and "c06" not in self.st["tasks"][t["id"]]]
            if name == "check" and self.conf_mode == "auto":
                todo = [t for t in todo if not self.st["tasks"][t["id"]].get("decided")]
            if not todo:
                continue
            import psutil
            self.st.setdefault("noise", []).append({"stage": name, "cpu_pct_before": psutil.cpu_percent(interval=3.0), "t": round(time.time())})
            if self.fast and name == "think":
                for t in todo:
                    fn(t)
                continue
            self.runner.use(slot)
            self.save()
            print(f"[{name}] {len(todo)} tasks on {slot}", flush=True)
            for t in todo:
                t0 = time.perf_counter()
                fn(t)
                print(f"  {t['id']:28s} {time.perf_counter() - t0:6.1f}s", flush=True)
            self.runner.stop()
        for t in self.tasks:
            ts = self.st["tasks"][t["id"]]
            if "graded" not in ts:
                ts["graded"] = grade(t, self.suite, self.team(t).ctx.ws, self.py)
                self.save()
                print(f"  graded {t['id']:24s} passed={ts['graded']['passed']}", flush=True)
        self.st["session_wall_s"] = round(self.st.get("session_wall_s", 0) + time.perf_counter() - t_all, 1)
        self.save()
        SP.close_handles()
        return 0


# ---------------------------------------------------------------------------------------------------------------- calibration
def ece(pairs: list[tuple[float, int]], bins: int = 10) -> float:
    n = len(pairs)
    tot = 0.0
    for b in range(bins):
        sel = [(p, y) for p, y in pairs if (min(int(p * bins), bins - 1) == b)]
        if sel:
            tot += len(sel) / n * abs(sum(p for p, _ in sel) / len(sel) - sum(y for _, y in sel) / len(sel))
    return tot


def run_calib(a: argparse.Namespace) -> int:
    rows = jsonl(Path(a.heldout))
    if a.limit:
        step = max(1, len(rows) // a.limit)
        rows = rows[::step][: a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runner = SL.SlotRunner(SL.default_specs(), out / "server")
    res_f = out / "calib_results.jsonl"
    done = {json.loads(x)["id"] for x in res_f.read_text(encoding="utf-8").splitlines()} if res_f.exists() else set()
    runner.use("CHECKER")
    t0 = time.perf_counter()
    with res_f.open("a", encoding="utf-8") as fh:
        for i, r in enumerate(rows):
            if r["id"] in done:
                continue
            m = r["messages"]
            c = runner.complete("CHECKER", SL.chatml(m[0]["content"], m[1]["content"]), "confidence", 12, 150.0)
            mm = re.search(r"([01](?:\.\d+)?)", c["content"])
            fh.write(json.dumps({"id": r["id"], "y": r["meta"]["y"], "p": float(mm.group(1)) if mm else None, "role": m[1]["content"].split("\n")[0][24:],
                                 "in_tok": c.get("in_tok"), "prompt_n": c.get("prompt_n"), "prompt_s": c.get("prompt_s"), "gen_s": c.get("gen_s"),
                                 "wall_s": round(c["wall_s"], 3), "error": c.get("error")}) + "\n")
            fh.flush()
            if i % 20 == 0:
                print(f"  calib {i + 1}/{len(rows)} {time.perf_counter() - t0:.0f}s", flush=True)
    runner.stop()
    return 0


def calib_summary(out: Path, heldout: str = "") -> dict[str, Any]:
    f = out / "calib_results.jsonl"
    if not f.exists():
        return {}
    rs = [r for r in jsonl(f) if r["p"] is not None]
    if heldout:
        roles = {h["id"]: h["messages"][1]["content"].splitlines()[0][len("ROLE OF THE ANSWER: "):] for h in jsonl(Path(heldout))}
        for r in rs:
            r["role"] = roles.get(r["id"], r["role"])
    pairs = [(r["p"], int(r["y"])) for r in rs]
    if not pairs:
        return {}
    ybar = sum(y for _, y in pairs) / len(pairs)
    byr: dict[str, list[tuple[float, int]]] = {}
    for r in rs:
        byr.setdefault(r["role"], []).append((r["p"], int(r["y"])))
    return {"n": len(pairs), "rate": round(ybar, 4), "ece10": round(ece(pairs, 10), 4), "ece5": round(ece(pairs, 5), 4),
            "brier": round(sum((p - y) ** 2 for p, y in pairs) / len(pairs), 4), "brier_base": round(sum((ybar - y) ** 2 for _, y in pairs) / len(pairs), 4),
            "mean_p": round(sum(p for p, _ in pairs) / len(pairs), 4), "by_role": {k: {"n": len(v), "ece5": round(ece(v, 5), 4)} for k, v in byr.items()},
            "mean_wall_s": round(statistics.mean(r["wall_s"] for r in rs), 2), "mean_in_tok": round(statistics.mean(r["in_tok"] or 0 for r in rs), 0),
            "mean_prompt_s": round(statistics.mean(r["prompt_s"] or 0 for r in rs), 2), "errors": sum(1 for r in jsonl(f) if r.get("error"))}


# ---------------------------------------------------------------------------------------------------------------- report
def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round((c - w) / d, 3), round((c + w) / d, 3)


def events(out: Path) -> list[dict[str, Any]]:
    SP.flush()
    rows: list[dict[str, Any]] = []
    for f in sorted((out / "state" / "metrics").glob("events-*.jsonl")):
        for ln in f.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                rows.append(json.loads(ln))
            except ValueError:
                pass
    return rows


def summarize(out: Path) -> dict[str, Any]:
    st = json.loads((out / "run_state.json").read_text(encoding="utf-8"))
    ev = [e for e in events(out) if e.get("goal_id") in st["tasks"]]
    tasks = {k: v for k, v in st["tasks"].items() if "graded" in v}
    n = len(tasks)
    solved = [k for k, v in tasks.items() if v["graded"]["passed"]]
    per_task = {k: sum(e["wall_s"] for e in ev if e["goal_id"] == k) for k in tasks}
    cpu_task = {k: sum(e.get("cpu_s", 0) for e in ev if e["goal_id"] == k) for k in tasks}
    loads = st.get("loads", [])
    load_s = sum(x["load_s"] for x in loads)
    total_wall = sum(per_task.values())
    steps: dict[str, dict[str, float]] = {}
    for e in ev:
        s = steps.setdefault(f"{e.get('step')}", {"n": 0, "wall_s": 0.0, "cpu_s": 0.0, "in_tok": 0, "out_tok": 0, "prompt_s": 0.0, "gen_s": 0.0,
                                                  "model": 0, "cache_hits": 0, "truncated": 0})
        s["n"] += 1
        s["wall_s"] += e["wall_s"]
        s["cpu_s"] += e.get("cpu_s", 0) or 0
        s["in_tok"] += e.get("in_tok", 0)
        s["out_tok"] += e.get("out_tok", 0)
        s["prompt_s"] += e.get("prompt_s", 0) or 0
        s["gen_s"] += e.get("gen_s", 0) or 0
        s["model"] += 1 if e.get("model") else 0
        s["cache_hits"] += 1 if e.get("cache_hit") else 0
        s["truncated"] += 1 if e.get("outcome") == "truncated" else 0
    sinks: list[tuple[str, float]] = []
    for k, s in steps.items():
        if s["model"]:
            other = max(0.0, s["wall_s"] - s["prompt_s"] - s["gen_s"])
            sinks += [(f"{k} prompt read (model)", s["prompt_s"]), (f"{k} generation (model)", s["gen_s"]), (f"{k} http+grammar overhead", other)]
        else:
            sinks.append((f"{k} (code tool)", s["wall_s"]))
    sinks.append(("model server loads (3 swaps, once per stage)", load_s))
    sinks.sort(key=lambda x: -x[1])
    nev, nmodel = len(ev), sum(1 for e in ev if e.get("model"))
    conf = [(v["confidence"], int(v["graded"]["passed"])) for v in tasks.values() if v.get("confidence") is not None]
    lo, hi = wilson(len(solved), n)
    wl = sum(tasks[k].get("lines_added", 0) for k in solved)
    gen_tok = sum(e.get("out_tok", 0) for e in ev if e.get("model"))
    good_tok = sum(e.get("out_tok", 0) for e in ev if e.get("model") and e.get("goal_id") in solved)
    trunc = sum(1 for e in ev if e.get("outcome") == "truncated")
    over = {"gt10": sum(1 for v in per_task.values() if v > 10), "gt100": sum(1 for v in per_task.values() if v > 100),
            "max": round(max(per_task.values()), 1) if per_task else 0, "p90": round(sorted(per_task.values())[int(0.9 * (n - 1))], 1) if n else 0}
    mev = [e for e in ev if e.get("model")]
    pin_tok, proc_tok = sum(e.get("in_tok", 0) for e in mev), sum(e.get("prompt_n") or e.get("in_tok", 0) for e in mev)
    prompt_s = sum(e.get("prompt_s", 0) or 0 for e in mev)
    rep_rows = [(k, v["repair"]) for k, v in tasks.items() if v.get("repair")]
    r4 = {"prompt_tokens": pin_tok, "prompt_tokens_processed": proc_tok, "prompt_reuse_share": round(1 - proc_tok / pin_tok, 4) if pin_tok else 0,
          "prompt_read_s": round(prompt_s, 1), "prompt_read_share_of_wall": round(prompt_s / total_wall, 4) if total_wall else 0,
          "prompt_read_share_of_model_wall": round(prompt_s / max(1e-9, sum(e["wall_s"] for e in mev)), 4) if mev else 0,
          "loop_aborts": sum(1 for e in ev if e.get("looped") or e.get("first_looped")), "loop_retries": sum(1 for e in ev if e.get("retried")),
          "tokens_wasted_in_aborted_first_calls": sum(e.get("first_out_tok", 0) for e in ev if e.get("retried"))}
    r2 = {"strategy": next((v["strategy"] for _, v in rep_rows), "none"), "tasks_with_repair_rounds": len(rep_rows),
          "rounds": sum(len(v["rounds"]) for _, v in rep_rows),
          "fixed_visible": [k for k, v in rep_rows if v["fixed"]],
          "fixed_and_hidden_pass": [k for k, v in rep_rows if v["fixed"] and tasks[k]["graded"]["passed"]],
          "by_strategy": {s_: {"rounds": sum(1 for _, v in rep_rows for r_ in v["rounds"] if r_["strategy"] == s_),
                               "to_pass": sum(1 for _, v in rep_rows for r_ in v["rounds"] if r_["strategy"] == s_ and r_.get("result") == "pass")}
                          for s_ in ("line", "regen")}}
    by_slot: dict[str, dict[str, float]] = {}
    for e in mev:
        b_ = by_slot.setdefault(str(e.get("slot")), {"calls": 0, "wall_s": 0.0, "prompt_s": 0.0, "gen_s": 0.0, "in_tok": 0, "prompt_n": 0, "out_tok": 0})
        b_["calls"] += 1
        b_["wall_s"] += e["wall_s"]
        b_["prompt_s"] += e.get("prompt_s", 0) or 0
        b_["gen_s"] += e.get("gen_s", 0) or 0
        b_["in_tok"] += e.get("in_tok", 0)
        b_["prompt_n"] += e.get("prompt_n") or e.get("in_tok", 0)
        b_["out_tok"] += e.get("out_tok", 0)
    by_slot = {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in by_slot.items()}
    cls_tok: dict[str, list[int]] = {}
    cls_trunc: dict[str, int] = {}
    for e in ev:
        if e.get("step") == "CODE" and e.get("model") and e.get("slot") in ("CODER", None):
            c_ = tasks.get(e["goal_id"], {}).get("class", "?")
            if e.get("outcome") == "truncated":
                cls_trunc[c_] = cls_trunc.get(c_, 0) + 1
            else:
                cls_tok.setdefault(c_, []).append(int(e.get("out_tok", 0)))
    q = lambda xs, f: sorted(xs)[min(len(xs) - 1, int(f * len(xs)))]          # noqa: E731
    out_tok_by_class = {c_: {"n": len(v), "p50": q(v, 0.5), "p90": q(v, 0.9), "p95": q(v, 0.95), "max": max(v), "truncated": cls_trunc.get(c_, 0)} for c_, v in cls_tok.items()}
    c06 = [(k, v) for k, v in tasks.items() if "c06" in v]
    route = {"routed_c06": sum(1 for v in tasks.values() if v.get("route", {}).get("start") == "C06"), "c06_attempted": len(c06),
             "c06_visible_pass": sum(1 for _, v in c06 if v["c06"]["visible_pass"]),
             "c06_finished_and_hidden_pass": sum(1 for k, v in c06 if v["c06"]["visible_pass"] and v["graded"]["passed"]),
             "escalated_to_c17": sum(1 for _, v in c06 if not v["c06"]["visible_pass"]),
             "escalated_solved": sum(1 for k, v in c06 if not v["c06"]["visible_pass"] and v["graded"]["passed"]),
             "by_class": {c_: {"n": sum(1 for v in tasks.values() if v.get("class") == c_), "solved": sum(1 for v in tasks.values() if v.get("class") == c_ and v["graded"]["passed"])}
                          for c_ in sorted({v.get("class", "?") for v in tasks.values()})}}
    conf_skipped = sum(1 for v in tasks.values() if v.get("verdict") == "tests")
    return {"by_slot": by_slot, "route": route, "out_tok_by_class": out_tok_by_class, "confidence_skipped": conf_skipped, "r4": r4, "r2": r2, "working_lines": wl, "working_lines_per_hour": round(wl / (total_wall / 3600), 1) if total_wall else 0,
            "useful_token_ratio": round(good_tok / gen_tok, 4) if gen_tok else 0, "truncated_calls": trunc, "over": over,
            "n_tasks": n, "solved": len(solved), "pass_rate": round(len(solved) / n, 4) if n else 0, "pass_ci95": [lo, hi],
            "visible_first_pass": sum(1 for v in tasks.values() if v.get("visible_first")), "visible_final_pass": sum(1 for v in tasks.values() if v.get("visible_pass")),
            "tasks_with_debug": sum(1 for v in tasks.values() if v.get("debug")), "debug_rounds": sum(v.get("debug", 0) for v in tasks.values()),
            "by_family": {f: {"n": sum(1 for k in tasks if k.startswith(f)), "solved": sum(1 for k in solved if k.startswith(f))} for f in ("app", "fn")},
            "s_per_task_mean": round(total_wall / n, 1) if n else 0, "s_per_task_median": round(statistics.median(per_task.values()), 1) if n else 0,
            "s_per_solved": round(total_wall / len(solved), 1) if solved else None,
            "s_per_solved_with_loads": round((total_wall + load_s) / len(solved), 1) if solved else None,
            "cpu_s_per_task_mean": round(sum(cpu_task.values()) / n, 1) if n else 0, "steps": nev, "model_steps": nmodel,
            "qwen_share": round(nmodel / nev, 4) if nev else 0, "server_load_s": round(load_s, 1), "loads": loads,
            "step_table": {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in sorted(steps.items(), key=lambda x: -x[1]["wall_s"])},
            "top_sinks": [(k, round(v, 1)) for k, v in sinks[:12]], "total_pipeline_wall_s": round(total_wall, 1),
            "team_confidence": {"n": len(conf), "ece5": round(ece(conf, 5), 4) if conf else None,
                                "verdict_acc": round(sum(1 for p, y in conf if (p >= CONF_THRESHOLD) == bool(y)) / len(conf), 3) if conf else None},
            "per_task": {k: {"s": round(per_task[k], 1), "passed": tasks[k]["graded"]["passed"], "visible_first": tasks[k].get("visible_first"),
                             "debug": tasks[k].get("debug"), "lines": tasks[k].get("lines_added"), "conf": tasks[k].get("confidence"), "stop": tasks[k].get("stop_reason")} for k in tasks}}


def report(a: argparse.Namespace) -> int:
    out = Path(a.out)
    s = summarize(out)
    s["calib"] = calib_summary(out / "calib", a.heldout)
    (out / "summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in s.items() if k not in ("per_task", "step_table")}, indent=1))
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("suite")
    s.add_argument("--bakeoff-dir", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--extra-fn", type=int, default=16)
    r = sub.add_parser("run")
    r.add_argument("--suite", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--only", nargs="*", default=[])
    r.add_argument("--fast", action="store_true", help="the P0.10 fast path (routing, prefilled edit, <=1 gated debug, short confidence)")
    r.add_argument("--reuse-db", default="", help="R3: path of the reuse index (creator.tools.reuse); on -> function tasks start from proven code")
    r.add_argument("--coder-moe", action="store_true", help="R10: serve CODER from Qwen3-30B-A3B (MoE) instead of the merged 1.7B")
    r.add_argument("--threads", type=int, default=6)
    r.add_argument("--repair", default="none", choices=list(RP.STRATEGIES), help="R2 test-driven repair after CODE (fast path only)")
    r.add_argument("--repair-rounds", type=int, default=3)
    r.add_argument("--lean-prefill", action="store_true", help="R4 variant: stub once in the prompt, one-line SEARCH prefill (fewer prompt tokens)")
    r.add_argument("--coder-gguf", default="", help="serve CODER from this GGUF (any coder model; same prompts and grammars)")
    r.add_argument("--max-fast-debug", type=int, default=None, help="override the fast-path DEBUG_FIX rounds (default SL.MAX_FAST_DEBUG)")
    r.add_argument("--subset", type=int, default=0, help="a deterministic stratified subset of N tasks (family x router class)")
    r.add_argument("--conf", default="auto", choices=("auto", "always"), help="auto: CONFIDENCE only for tasks whose tests did not decide; always: the old path")
    r.add_argument("--route", action="store_true", help="R5: router -> base 0.6B first on easy function tasks, escalate to the 1.7B on a visible failure")
    r.add_argument("--router-state", default="", help="router counts file (measured per-class pass counts)")
    r.add_argument("--prefix-first", action="store_true", help="R4: fixed text first, task-specific text last in the CODE prompt (longer shared KV prefix)")
    r.add_argument("--class-caps", default="", help="JSON {router class: CODE max_tokens} from the measured output distribution")
    r.add_argument("--pack-k", type=int, default=0, help="app tasks: located hits packed (default ToolContext.k)")
    r.add_argument("--pack-tokens", type=int, default=0, help="app tasks: pack token budget (default ToolContext.pack_tokens)")
    c = sub.add_parser("calib")
    c.add_argument("--heldout", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    p.add_argument("--heldout", default="")
    a = ap.parse_args(argv)
    return {"suite": build_suite, "run": lambda x: Run(x).run(), "calib": run_calib, "report": report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
