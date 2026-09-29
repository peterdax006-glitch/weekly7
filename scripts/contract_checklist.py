"""Update items / budget rows of state/build/SELF_LEARNING_MASTER_CHECKLIST.json (contract C62 section 68) with evidence.

  python scripts/contract_checklist.py item I06 I07 --status IMPLEMENTED --code engine/learning/planted_world.py \
         --tests tests/test_learning_planted_world.py --note "IMPLEMENTED — NOT VALIDATED. ..."
  python scripts/contract_checklist.py budget "Calibration" --code engine/learning/calibration.py   (actual lines = ruler count)
  python scripts/contract_checklist.py summary

VALIDATED is refused unless --evidence names an existing artefact (section 59: no false completion). Stamps last_run and code_hash."""
import argparse, datetime, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "state" / "build" / "SELF_LEARNING_MASTER_CHECKLIST.json"
if "--tenhour" in sys.argv:                                    # the C69 10-hour execution checklist
    sys.argv.remove("--tenhour")
    F = ROOT / "state" / "build" / "TEN_HOUR_CHECKLIST.json"
if "--addition" in sys.argv:                                   # the C68 addition checklist
    sys.argv.remove("--addition")
    F = ROOT / "state" / "build" / "PREDICTION_ERROR_CHECKLIST.json"
if "--research" in sys.argv:                                   # the C66 checklist (RESEARCH_BRAIN_CHECKLIST.json), same rules
    sys.argv.remove("--research")
    F = ROOT / "state" / "build" / "RESEARCH_BRAIN_CHECKLIST.json"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def load():
    return json.loads(F.read_text(encoding="utf-8"))


def save(d):
    text = json.dumps(d, indent=1, ensure_ascii=False)
    json.loads(text)                                           # validate before write
    import os, time
    tmp = F.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    for attempt in range(20):                                  # readers (autorefresh, AV) may hold the file briefly
        try:
            os.replace(tmp, F)
            return
        except OSError:
            time.sleep(0.5)
    raise OSError(f"could not replace {F}; new content left in {tmp}")


def merge(old, new):
    return sorted(set(old or []) | set(new or []))


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # section names contain '→'; a cp1252 console crashed before save
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["item", "budget", "summary"])
    ap.add_argument("keys", nargs="*")
    ap.add_argument("--status")
    ap.add_argument("--code", nargs="*", default=[])
    ap.add_argument("--tests", nargs="*", default=[])
    ap.add_argument("--evidence", nargs="*", default=[])
    ap.add_argument("--note")
    ap.add_argument("--validation")
    ap.add_argument("--downgrade", action="store_true")
    a = ap.parse_args()
    d = load()
    if a.kind == "summary":
        c = {}
        for it in d["items"]:
            c[it["status"]] = c.get(it["status"], 0) + 1
        print("items:", dict(sorted(c.items())))
        under = [b for b in d["line_budget"] if (b.get("actual_lines") or 0) < b["minimum_lines"]]
        print(f"budget rows under minimum: {len(under)}/{len(d['line_budget'])}")
        return
    if a.status and a.status not in d["allowed_status"]:
        sys.exit(f"status {a.status} not allowed: {d['allowed_status']}")
    if a.status == "VALIDATED" and not [e for e in a.evidence if (ROOT / e).exists()]:
        sys.exit("VALIDATED needs --evidence pointing at an existing artefact")
    from engine.learning.core import current_code_hash
    stamp = {"last_run": datetime.datetime.now().isoformat(timespec="seconds"), "code_hash": current_code_hash()}
    if a.kind == "item":
        rows = {it["id"]: it for it in d["items"]}
    else:
        rows = {b["section"]: b for b in d["line_budget"]}
    missing = [k for k in a.keys if k not in rows]
    if missing:
        sys.exit(f"unknown keys: {missing}")
    for k in a.keys:
        r = rows[k]
        if a.status and r.get("status") in ("VALIDATED", "FAILED") and a.status != r["status"] and not a.downgrade:
            print(f"{k}: kept {r['status']} (earned by evidence; pass --downgrade with a reason in --note to change it)")
            a_status, r_status = None, r["status"]
        else:
            a_status = a.status
        if a_status:
            r["status"] = a_status
        r["code_paths"] = merge(r.get("code_paths"), a.code)
        if "tests" in r or a.kind == "item":
            r["tests"] = merge(r.get("tests"), a.tests)
        if "evidence" in r or a.kind == "item":
            r["evidence"] = merge(r.get("evidence"), a.evidence)
        if a.note:
            r["notes"] = a.note if not r.get("notes") or a.note in r["notes"] else a.note + " | earlier: " + r["notes"]
        if a.validation:
            r["validation_result"] = a.validation
        if a.kind == "budget" and r["code_paths"]:
            from contract_lines import meaningful
            r["actual_lines"] = sum(meaningful(ROOT / p) for p in r["code_paths"] if (ROOT / p).exists())
        r.update(stamp) if a.kind == "item" else None
        print(k, "->", r.get("status"), r.get("actual_lines", ""))
    save(d)


if __name__ == "__main__":
    main()
