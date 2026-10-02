"""The Creator assesses itself from evidence: run its declared tests, build the self-model, compile the self-objective's requirement
ladder, sync gaps into the development ledger, and write state/creator/STATUS.json (C77 sec 6 reconnaissance, secs 11/41/42)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from creator import gaps as G  # noqa: E402
from creator import objective as O  # noqa: E402
from creator import schedule as SCH  # noqa: E402
from creator import selfmodel as SM  # noqa: E402
from creator.audit import checks as AUD  # noqa: E402
from creator.ledger import Ledger  # noqa: E402

STATE = ROOT / "state" / "creator"
LEDGER = STATE / "ledger.jsonl"


def _oversight(led: Ledger) -> dict:
    from creator import oversight as OV
    OV.knowledge_gaps(led)
    return OV.summary(led)


def _efficiency() -> dict:
    from creator import efficiency as E
    act = E.activation(ROOT)
    return {"package_ast_nodes": act["package_nodes"], "active_ast_nodes": act["active_nodes"],
            "active_fraction": act["fraction"], "peak_memory_mb": E.peak_memory_mb(ROOT, replicates=1)[0]}


def _curriculum() -> dict:
    from creator import curriculum as CUR
    lessons = CUR.LessonLog(STATE / "lessons.jsonl").lessons()
    r = CUR.Router()
    kinds = sorted({les.task_kind for les in lessons})
    return {**CUR.student_scores(lessons), "lessons": len(lessons),
            "handed_over": {k: r.owners(lessons, k) for k in kinds if r.handed_over(lessons, k)}}


def _goals() -> dict:
    from creator import goals as GO
    pend = GO.pending(STATE)
    return {"pending": len(pend), "items": [{"id": p["id"], "source": p["source"], "title": p["title"], "value": p["value"]} for p in pend]}


def main(argv: list[str]) -> int:
    t0 = time.time()
    specs = SM.load_capabilities()
    tests = sorted({t for s in specs for t in s.tests if (ROOT / t).is_file()})
    ev = SM.collect_test_evidence(ROOT, tests, STATE / "test_evidence.json") if "--no-tests" not in argv \
        else SM.load_test_evidence(STATE / "test_evidence.json")
    led = Ledger(LEDGER, evidence_root=ROOT)
    model = SM.build(ROOT, scope=("creator", "tests", "scripts"), capabilities=specs, test_evidence=ev, ledger=led)
    SM.save(model, STATE / "selfmodel.json")
    oid = O.self_objective(led)
    O.compile_capabilities(led, oid, specs)
    rep = G.sync(led, model)
    summ = G.summary(led)
    aud = AUD.audit(led, model)
    attacks = AUD.adversary()
    (STATE / "AUDIT.json").write_text(json.dumps({"audit": aud.to_dict(), "adversary": [a.__dict__ for a in attacks]}, indent=1),
                                      encoding="utf-8")
    caps = [{"id": c.id, "name": c.name, "state": c.state, "uncertainty": c.uncertainty, "meaningful": c.meaningful,
             "floor": c.floor, "why": c.why, "retired": O.retired_reason(c.id)} for c in model.capabilities]
    status = {"at": summ["at"], "selfmodel_digest": model.digest(), "versions": dict(model.versions), "capabilities": caps,
              "requirements": summ, "sync": {"opened": len(rep.opened), "closed": len(rep.closed), "regressed": list(rep.regressed)},
              "audit": aud.to_dict()["counts"], "audit_errors": dict(aud.errors),
              "adversary": {"attacks": len(attacks), "caught": sum(a.caught for a in attacks),
                            "uncaught": [a.name for a in attacks if not a.caught]},
              "efficiency": _efficiency(), "claude_dependence": O.claude_dependence(led), "oversight": _oversight(led), "failures": AUD.failure_report(led), "curriculum": _curriculum(), "goal_proposals": _goals(), "plan": SCH.last_plan(LEDGER),
              "ledger_records": led.verify(), "seconds": round(time.time() - t0, 1)}
    (STATE / "STATUS.json").write_text(json.dumps(status, indent=1, default=str), encoding="utf-8")
    for c in caps:
        print(f"{c['id']} {c['state']:<12} {c['uncertainty']:<9} {c['meaningful']:>5}/{c['floor']:<5} {c['name']}"
              + (f" [RETIRED: {c['retired']}]" if c["retired"] else ""))
    print(json.dumps({k: summ[k] for k in ("requirements", "by_status", "open_gaps", "unblocked")}))
    print("EFFICIENCY", status["efficiency"], "CLAUDE DEPENDENCE", status["claude_dependence"])
    print("CURRICULUM (Nupen = student, Claude = teacher; teacher_share must fall)", status["curriculum"])
    print("SYNC", status["sync"], "AUDIT", status["audit"], aud.errors or "", "ADVERSARY", status["adversary"])
    from creator import constraints as CON
    for i, m in enumerate(CON.top_constraints(STATE), 1):
        print(f"CONSTRAINT {i}: {m['name']} loss {m['loss']} score {m['score']} ({m['value']} {m['unit']}) remedy={m['remedy']}")
    gp = status["goal_proposals"]
    print(f"GOAL PROPOSALS pending approval: {gp['pending']} (python scripts/nupen_goals.py list)")
    for g in gp["items"][:5]:
        print(f"  {g['id']} ({g['source']}, value {g['value']}) {g['title']}")
    plan = status["plan"]
    if plan["at"]:
        print(f"PLAN {plan['at']} critical path {plan['critical_path_s']:.0f}s: {' -> '.join(plan['critical_path']) or '-'}; chosen {plan['chosen']}")
        for r in plan["reasons"]:
            print(f"  {'RUN ' if r['chosen'] else 'hold'} {r['node']} {r['component']}.{r['step']}: {r['why']}")
    for f in aud.findings[:10]:
        print("FINDING", f.severity, f.check, f.subject, f.detail[:100])
    for g in summ["next"]:
        print("NEXT", g["requirement_key"], g["importance"], g["description"][:100])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
