import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import audit_registry as A


def test_audit_script_reports_and_never_writes_the_log(tmp_path):
    log = tmp_path / "e.jsonl"
    rows = [{"event": "x", "t": "2026-01-01"}, {"event": "x", "t": "2026-01-02", "experiment_id": "E1", "seed": 3},
            {"event": "y", "experiment_id": "E1"}]
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
    before = log.read_bytes()
    out = tmp_path / "res" / "audit.json"
    r = A.main(log, out)
    assert log.read_bytes() == before
    assert r["n_records"] == 3 and not r["ok"] and r["orphans_no_outcome"] == 3
    assert r["duplicate_ids"] and r["bad_lines"] and r["records_fully_complete"] == 0
    assert r["missing_counts"]["experiment_id"] == 1 and r["by_event"]["x"]["records"] == 2
    assert json.loads(out.read_text())["n_records"] == 3


def test_audit_script_on_missing_log(tmp_path):
    r = A.main(tmp_path / "nope.jsonl", tmp_path / "a.json")
    assert r["n_records"] == 0 and not r["ok"]
