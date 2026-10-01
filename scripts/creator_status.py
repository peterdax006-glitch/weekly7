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
from creator import selfmodel as SM  # noqa: E402
from creator.ledger import Ledger  # noqa: E402

STATE = ROOT / "state" / "creator"
LEDGER = STATE / "ledger.jsonl"


def main(argv: list[str]) -> int:
    t0 = time.time()
    specs = SM.load_capabilities()
    tests = sorted({t for s in specs for t in s.tests if (ROOT / t).is_file()})
    ev = SM.collect_test_evidence(ROOT, tests, STATE / "test_evidence.json") if "--no-tests" not in argv \
        else SM.load_test_evidence(STATE / "test_evidence.json")
    led = Ledger(LEDGER, evidence_root=ROOT)
    model = SM.build(ROOT, scope=("creator", "tests"), capabilities=specs, test_evidence=ev, ledger=led)
    SM.save(model, STATE / "selfmodel.json")
    oid = O.self_objective(led)
    O.compile_capabilities(led, oid, specs)
    rep = G.sync(led, model)
    summ = G.summary(led)
    caps = [{"id": c.id, "name": c.name, "state": c.state, "uncertainty": c.uncertainty, "meaningful": c.meaningful,
             "floor": c.floor, "why": c.why} for c in model.capabilities]
    status = {"at": summ["at"], "selfmodel_digest": model.digest(), "versions": dict(model.versions), "capabilities": caps,
              "requirements": summ, "sync": {"opened": len(rep.opened), "closed": len(rep.closed), "regressed": list(rep.regressed)},
              "ledger_records": led.verify(), "seconds": round(time.time() - t0, 1)}
    (STATE / "STATUS.json").write_text(json.dumps(status, indent=1, default=str), encoding="utf-8")
    for c in caps:
        print(f"{c['id']} {c['state']:<12} {c['uncertainty']:<9} {c['meaningful']:>5}/{c['floor']:<5} {c['name']}")
    print(json.dumps({k: summ[k] for k in ("requirements", "by_status", "open_gaps", "unblocked")}))
    for g in summ["next"]:
        print("NEXT", g["requirement_key"], g["importance"], g["description"][:100])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
