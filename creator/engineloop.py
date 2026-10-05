"""The engine loop: the constraint engine's DECISION -> the adopt cycle (build -> verify -> adopt) -> the measured outcome back into the
engine's calibration (MASTER_BLUEPRINT 7.2-7.4, 9.1). This module only WIRES the existing pieces to the real functions:

  constraints.engine_cycle   detect, rank, decide (WIP 1) -> a decision, or none
  adopt.run_goal             the fallback ladder around adopt.run_candidate (build by the team, verify, adopt)
  RealEnv                    adopt.Env bound to the real checks: a sandbox of the repo, the fast loop (testrun.quick_cycle: patch, static
                             checks, affected tests), the COLD protected suite as the verdict (safety.check_tree, new failures vs the
                             base), git merge / revert in the sandbox machinery (sandbox.adopt / sandbox.rollback), the post-merge
                             suite (safety.suite_failure)
  constraints.mark_done      frees the WIP slot and feeds predicted-vs-actual saving back into the calibration
  adopt.earn                 after a verified adoption the class may climb the autonomy ladder (A0 -> A1 -> A2; A3 is the owner's)

Nothing here runs by itself and everything heavy is imported inside functions. EVERY entry point refuses while the stop file
(`<state>/NUPEN_STOP`) exists. Standard library only at import time."""
from __future__ import annotations

import re
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

STOP_FILE = "NUPEN_STOP"


def stopped(state: Path) -> bool:
    """True while the owner's off switch exists (state/creator/NUPEN_STOP): nothing in this module starts."""
    return (Path(state) / STOP_FILE).exists()


def candidate_from(decision: Mapping[str, Any]) -> dict[str, Any]:
    """The engine's decision as the adopt cycle's candidate: predicted = per-event saving (c0 - c1), f = events per day, B = build cost."""
    ev = decision.get("evidence")
    return {"id": re.sub(r"[^A-Za-z0-9_.-]+", "_", str(decision["key"])), "key": str(decision["key"]), "title": f"{decision.get('kind', '')}: {decision.get('fix', '')}"[:200],
            "claim": (ev if isinstance(ev, str) else str(ev or ""))[:300] or str(decision.get("fix", ""))[:300],
            "cls": str(decision.get("cls") or "tool"), "kind": decision.get("kind", ""),
            "predicted": max(0.0, float(decision.get("c0") or 0.0) - float(decision.get("c1") or 0.0)),
            "f": float(decision.get("f_per_day") or 0.0), "B": float(decision.get("build_s") or 0.0)}


class _Committed:
    """sandbox.adopt commits the sandbox itself; the protected suite needs the commit earlier. This proxy turns the second commit into
    'the commit that already exists'."""

    def __init__(self, sb: Any, rev: str):
        self._sb, self._rev = sb, rev

    def commit(self, message: str) -> str:
        return self._rev

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sb, name)


class RealEnv:
    """adopt.Env bound to the real functions for ONE candidate. `env` is the adopt.Env; call close() when the candidate is finished.
    cfg: a creator.kernel.KernelConfig (repo, state, scratch, hide, omit, pytest). suite_files / runner / warm are overridable for
    tests; by default the suite is safety.suite_files() and the warm test worker serves the fast loop."""

    def __init__(self, cfg: Any, team: Any, by: str, *, bench: Optional[Callable[[Mapping[str, Any], str], list[float]]] = None,
                 tracked: Optional[Callable[[str], Mapping[str, Mapping[str, Any]]]] = None, suite_files: Optional[Sequence[str]] = None,
                 runner: Any = None, warm: bool = True, impact_dir: str = "", clock: Callable[[], float] = time.time):
        from creator import adopt as A
        self.cfg, self.team, self.by = cfg, team, by
        self._bench, self._tracked, self._files, self._runner = bench, tracked, suite_files, runner
        self._warm, self._impact = warm, impact_dir
        self.sb: Any = None
        self.box: Any = None
        self.cid = "cand"
        self.rev = ""
        self.merged = False
        self.env = A.Env(state=Path(cfg.state), team=team, by=by, check=self.check, bench=self.bench, protected=self.protected,
                         tracked=self.tracked, merge=self.merge, revert=self.revert, post_merge=self.post_merge, clock=clock)

    # -- sandbox -----------------------------------------------------------------------------------------------------
    def _open(self) -> Any:
        if self.sb is None:
            from creator import sandbox as S
            from creator.tools import toolbox as TB
            c = self.cfg
            self.sb = S.Sandbox.open(c.repo, "HEAD", c.scratch, label="adopt", hide=getattr(c, "hide", ()), omit=getattr(c, "omit", ()))
            self.box = TB.ToolBox(self.sb.path, Path(c.state), f"adopt_{self.sb.id}", "creator")
        return self.sb

    def close(self) -> None:
        if self.box is not None:
            self.box.close()
        if self.sb is not None:
            from creator import sandbox as S
            if self.merged:
                self.sb.close()                                        # the branch stays: the adopted change is inspectable
            else:
                S.discard(self.sb)
        self.sb = self.box = None

    # -- the Env functions -------------------------------------------------------------------------------------------------
    def check(self, patch: str) -> dict[str, Any]:
        """Apply in the candidate sandbox + static checks + affected tests (the warm fast loop). A failed check is undone so the next debug
        round starts from the base again."""
        from creator import testrun as T
        sb = self._open()
        junit = Path(self.cfg.state) / "evidence" / "adopt" / f"{sb.id}_{int(time.time() * 1000) % 10**9}.xml"
        junit.parent.mkdir(parents=True, exist_ok=True)
        r = T.quick_cycle(self.box.files, sb.path, patch, junit, warm=self._warm, impact_dir=self._impact)
        if r.get("ok"):
            return {"ok": True, "why": "", "stage_s": r.get("stage_s")}
        run = r.get("run") or {}
        why = (r.get("why") or str((r.get("apply") or {}).get("error") or (r.get("apply") or {}).get("why") or "")
               or (f"tests {run.get('status')}: {run.get('counts')} {'; '.join(run.get('problems') or [])}"[:300] if run else "check failed"))
        if r.get("undo"):
            self.box.files.undo_patch(r["undo"])
        return {"ok": False, "why": why}

    def bench(self, spec: Mapping[str, Any], which: str) -> list[float]:
        """Cost samples per event: 'before' on the unchanged repo, 'after' in the candidate sandbox. Run by VERIFY, never by the builder."""
        if self._bench is not None:
            return list(self._bench(spec, which))
        from creator import adopt as A
        argv = [sys.executable if a == "python" else a for a in spec["argv"]]
        root = self.cfg.repo if which == "before" else self._open().path
        return A.run_bench_command(Path(root), argv, int(spec.get("repeats", 5)), float(spec.get("timeout", 120.0)))

    def tracked(self, which: str) -> Mapping[str, Mapping[str, Any]]:
        return self._tracked(which) if self._tracked is not None else {}

    def _suite_files(self) -> tuple[str, ...]:
        if self._files is not None:
            return tuple(self._files)
        from creator import safety as SF
        return SF.suite_files()

    def protected(self) -> dict[str, Any]:
        """The cold protected suite on the committed candidate tree; only failures the base tree does not have count."""
        from creator import safety as SF
        from creator import sandbox as S
        sb = self._open()
        self.rev = sb.commit(f"adopt candidate {self.cid}")
        files = self._suite_files()
        after = SF.check_tree(self.cfg, sb.path, self.rev, files, f"adopt_{sb.id}", self._runner)
        if not SF._ok(after):
            return {"status": str(after.get("status")), "failed": [f"no result: {after.get('problems')}"]}
        failed = list(after.get("failed") or [])
        if failed:
            base_rev = sb.base
            base = SF._cache(self.cfg.state).get(SF.tree_key(self.cfg.repo, base_rev, files, self.cfg.pytest.python))
            if base is None:                                               # base never run: run only the failing files on it
                bsb = S.Sandbox.open(self.cfg.repo, base_rev, self.cfg.scratch, label="adoptbase", omit=getattr(self.cfg, "omit", ()))
                try:
                    base = (self._runner or SF.run_suite)(bsb.path, sorted({c.split("::", 1)[0] for c in failed}),
                                                          Path(self.cfg.state) / "evidence" / "adopt" / f"{sb.id}_base.xml",
                                                          SF.policy(self.cfg.state).suite_timeout, self.cfg.pytest.python, 1)
                finally:
                    S.discard(bsb)
            failed = sorted(set(failed) - set(base.get("failed") or []))
        return {"status": str(after.get("status")), "failed": failed, "cached": bool(after.get("cached"))}

    def merge(self, patch: str, cand: Mapping[str, Any]) -> str:
        from creator import sandbox as S
        sb = self._open()
        res = S.adopt(_Committed(sb, self.rev or sb.commit(f"adopt {cand['id']}")), types.SimpleNamespace(verdict="ADOPT"),
                      f"{cand['id']}: {cand.get('title', '')}"[:200])
        self.merged = True
        return res.merge_commit

    def revert(self, commit: str, why: str) -> str:
        from creator import sandbox as S
        return S.rollback(self.cfg.repo, commit, why)

    def post_merge(self, commit: str) -> str:
        from creator import safety as SF
        return SF.suite_failure(self.cfg, commit, types.SimpleNamespace(details={}, package=self.cid), self._runner)


def trust_signals(state: Path, by: str, result: Mapping[str, Any]) -> tuple[bool, Optional[float]]:
    """(gate_open, ECE) for adopt.earn: the coding trust gate for the paths of the adopted diff, and the calibration (ECE) of the worker's
    recorded confidences."""
    from creator import adopt as A
    from creator import codetrust as CT
    from creator import safety as SF
    pol = SF.policy(state)
    src = pol.trust_records.get(by, "")
    recs = CT._jsonl(Path(src)) if src else []
    pairs = [(float(r["confidence"]), int(bool(r["passed"]))) for r in recs if r.get("confidence") is not None and "passed" in r]
    ece = CT.ece(pairs) if pairs else None
    diff = Path(str(result.get("artifact", ""))) / "candidate.diff"
    paths = A._paths(diff.read_text(encoding="utf-8")) if diff.is_file() else []
    return (SF.trust(pol, by, paths, "")[0] if paths else False), ece


def engine_step(state: Path, cfg: Any, team: Any, by: str, *, now: Optional[float] = None, rng: Any = None, days: float = 7.0,
                bench: Any = None, tracked: Any = None, suite_files: Optional[Sequence[str]] = None, runner: Any = None, warm: bool = True,
                impact_dir: str = "", fingerprints: Optional[Callable[[], Mapping[str, str]]] = None,
                clock: Callable[[], float] = time.time) -> dict[str, Any]:
    """ONE pass: decide -> adopt cycle -> mark_done (calibration) -> earn. Returns {'decision', 'result', 'level'} or {'skipped': reason}."""
    from creator import adopt as A
    from creator import constraints as C
    state = Path(state)
    if stopped(state):
        return {"skipped": f"{STOP_FILE} exists"}
    rec = C.engine_cycle(state, now, days, rng)
    dec = rec.get("decision")
    if not dec:
        return {"skipped": rec.get("why_none") or "no decision", "record": rec}
    cand = candidate_from(dec)
    real = RealEnv(cfg, team, by, bench=bench, tracked=tracked, suite_files=suite_files, runner=runner, warm=warm, impact_dir=impact_dir, clock=clock)
    real.cid = cand["id"]
    res: dict[str, Any] = {}
    actual: Optional[float] = None
    try:
        res = A.run_goal(real.env, cand["id"], lambda rung: dict(cand, attempt=rung if rung != "start" else ""),
                         fingerprints or (lambda: {}), kind=str(dec.get("kind", "")))
        v = res.get("verdict")
        if res.get("outcome") in ("ADOPTED", "PROPOSED", "REFUSED", "ROLLED_BACK") and v:
            actual = float(v.get("saving") or 0.0) * cand["f"] if res["outcome"] != "ROLLED_BACK" else 0.0   # per day, the unit of the prediction
    finally:
        real.close()
        C.mark_done(state, str(dec["key"]), actual, dec)                    # always frees the WIP slot; a measured outcome recalibrates
    level = None
    if res.get("outcome") == "ADOPTED":
        gate_open, ece = trust_signals(state, by, res)
        level = A.earn(state, str(res.get("cls", cand["cls"])), gate_open=gate_open, ece=ece, now=clock())
    return {"decision": dec, "result": res, "actual_saving_day": actual, "level": level}


# ------------------------------------------------------------------------------------------------ the live trigger (gated)
ENGINE_ON = "ENGINE_ON"                  # the owner's explicit opt-in file: state/creator/ENGINE_ON
LAST_STEP = "engine/last_step.json"
_WIP = threading.Lock()                  # WIP 1: one engine cycle at a time (constraints.engine_cycle holds the persistent WIP slot as well)


def enabled(state: Path) -> bool:
    """Gates 1 and 2: the stop file is absent AND the owner created ENGINE_ON. (Gate 3, the autonomy ladder, is enforced inside the adopt
    cycle: every class starts at A0 = proposal + benchmark only; only adopt.earn / the owner raise it, a regression lowers it.)"""
    s = Path(state)
    return not stopped(s) and (s / ENGINE_ON).exists()


def slot_team(state: Path, chat: Optional[Callable[[str, str], str]] = None) -> Any:
    """The builder team for the live engine: code tools answer from files, the slot roles (CHECKER / THINKER / CODER) ask the local model.
    chat(role, prompt) -> text; default = creator.generator.LocalModel (leases a warm pool server). The envelope carries ids; the prompt
    is rebuilt here from the board content."""
    from creator import team as TM

    def ask(role: str) -> Callable[[Any, Any], str]:
        def fn(env: Any, team: Any) -> str:
            ctx = []
            for ref in env.inputs:
                row = team.board.get(ref) if TM.input_kind(ref) == "board" else None
                ctx.append(str(row["body"]) if row else str(ref))
            prompt = f"{role} step {env.step}.\n" + "\n---\n".join(ctx + list(env.constraints) + [env.success_test, env.prior])
            if chat is not None:
                return chat(role, prompt)
            from creator import generator as G
            with G.LocalModel(startup_s=120.0) as llm:
                return str(llm.chat_text([{"role": "user", "content": prompt}], max_tokens=1500))
        return fn

    def locate(env: Any, team: Any) -> str:
        return "none"                                                  # no index wired here: the builder works from the task text
    return TM.Team(Path(state) / "engine" / "team", [TM.actor(r, ask(r)) for r in TM.SLOTS] + [TM.actor("locate", locate)], log=TM.slowpath_log())


def maybe_step(state: Path, cfg: Any, *, team_factory: Optional[Callable[[Path], Any]] = None, by: str = "engine", every_s: float = 3600.0,
               now: Optional[float] = None, spawn: bool = True, **kw: Any) -> str:
    """The swarm's hourly hook (next to constraints.maybe_run). Returns why nothing started, or 'started' / 'ran'.
    Gates: NUPEN_STOP absent, ENGINE_ON present, at most one cycle at a time, at most one per `every_s`. The cycle runs on an idle-priority
    daemon thread (spawn=True) so the swarm round never waits for it; spawn=False runs it inline (tests). Never raises."""
    import json
    state = Path(state)
    try:
        if stopped(state):
            return f"{STOP_FILE} exists"
        if not (state / ENGINE_ON).exists():
            return f"no {ENGINE_ON} (the owner has not turned the engine on)"
        t = time.time() if now is None else now
        f = state / LAST_STEP
        try:
            if t - float(json.loads(f.read_text(encoding="utf-8")).get("at", 0.0)) < every_s:
                return "cadence: ran within the last interval"
        except (OSError, ValueError):
            pass
        if not _WIP.acquire(blocking=False):
            return "WIP 1: an engine cycle is already running"
    except Exception as e:                                           # noqa: BLE001
        return f"error: {type(e).__name__}"

    def work() -> None:
        try:
            if stopped(state):
                return
            rec: dict[str, Any] = {"at": t}
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(rec), encoding="utf-8")          # stamped first: a crash does not make the next round retry at once
            res = engine_step(state, cfg, (team_factory or slot_team)(state), by, now=now, **kw)
            out = res.get("result") or {}
            f.write_text(json.dumps({"at": t, "skipped": res.get("skipped"), "outcome": out.get("outcome"), "id": out.get("id"),
                                     "level": res.get("level")}, default=str), encoding="utf-8")
        except Exception as e:                                       # noqa: BLE001 - the engine never costs a swarm round
            try:
                f.write_text(json.dumps({"at": t, "error": f"{type(e).__name__}: {e}"[:300]}), encoding="utf-8")
            except OSError:
                pass
        finally:
            _WIP.release()
    if not spawn:
        work()
        return "ran"
    try:
        from creator import resources as RS
        th = threading.Thread(target=RS.idle_thread(work), name="engine-step", daemon=True)
        th.start()
    except Exception:                                                # noqa: BLE001
        _WIP.release()
        return "error: could not start"
    return "started"
