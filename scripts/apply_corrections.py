"""Apply an auditor's corrections file (list of {checklist, id, apply_status, needs_downgrade_flag, note_only, evidence, reason})
through scripts/contract_checklist.py, so every rule of that tool (VALIDATED needs an existing artefact; downgrades need
--downgrade) still applies. C75 Phase 0 (30 Sep): F20's C75_PHASE0_CORRECTIONS.json.

  python scripts/apply_corrections.py state/build/C75_PHASE0_CORRECTIONS.json --source "F20 C75 Phase 0 audit"
"""
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLAG = {"SELF_LEARNING_MASTER_CHECKLIST.json": [], "RESEARCH_BRAIN_CHECKLIST.json": ["--research"],
        "PREDICTION_ERROR_CHECKLIST.json": ["--addition"], "TEN_HOUR_CHECKLIST.json": ["--tenhour"],
        "ULTIMATE_MASTER_CHECKLIST.json": ["--ultimate"], "MASTER_EXECUTION_CHECKLIST.json": ["--master"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("corrections")
    ap.add_argument("--source", required=True)
    a = ap.parse_args(argv)
    rows = json.loads(Path(a.corrections).read_text(encoding="utf-8"))
    ok, refused = 0, []
    for r in rows:
        flag = FLAG[Path(r["checklist"]).name]
        note = f"{a.source} (30 Sep): {r['reason']}"
        cmd = [sys.executable, str(ROOT / "scripts/contract_checklist.py"), *flag, "item", r["id"], "--note", note]
        if not r.get("note_only"):
            cmd += ["--status", r["apply_status"]]
        if r.get("needs_downgrade_flag"):
            cmd.append("--downgrade")
        paths = [e.split(":", 1)[-1].strip() if e.startswith(("tests:", "evidence:", "code:")) else e for e in r.get("evidence", [])]
        files = [p.strip() for e in paths for p in e.split(",") if (ROOT / p.strip().split("::")[0]).exists()]
        if files:
            cmd += ["--evidence", *files]
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if res.returncode == 0 and "kept" not in res.stdout:
            ok += 1
        else:
            refused.append((r["checklist"], r["id"], (res.stdout + res.stderr).strip()[-200:]))
    print(f"applied {ok}/{len(rows)}")
    for x in refused:
        print("REFUSED/KEPT", *x)
    return 0


if __name__ == "__main__":
    sys.exit(main())
