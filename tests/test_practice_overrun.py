"""practice.overrun_sets (h62): a direct-test set that keeps timing out and never finished stops costing the practice budget."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import practice as P  # noqa: E402


def _row(tests: list[str], reason: str, seconds: float, at: float, metric: str = "size") -> dict:
    return {"metric": metric, "tests": tests, "reason": reason, "seconds": seconds, "stage": "ok",
            "at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(at))}


def test_only_sets_that_timed_out_twice_and_never_finished_are_skipped(tmp_path: Path) -> None:
    now = time.time()
    to = "inconclusive: TIMEOUT: ('pytest exceeded 45.0s',)"
    rows = [_row(["tests/test_slow.py"], to, 45.6, now - 60), _row(["tests/test_slow.py"], to, 45.2, now - 30),
            _row(["tests/test_slow.py"], to, 45.2, now - 30, metric="activation"),             # one count per candidate
            _row(["tests/test_once.py"], to, 45.1, now - 30),                                     # once is not enough
            _row(["tests/test_mixed.py"], to, 45.1, now - 90), _row(["tests/test_mixed.py"], to, 45.1, now - 80),
            _row(["tests/test_mixed.py"], "", 12.0, now - 70),                                    # it finished once: never skipped
            _row(["tests/test_old.py"], to, 45.1, now - 9 * 86400), _row(["tests/test_old.py"], to, 45.1, now - 9 * 86400),
            _row(["tests/test_short.py"], to, 20.0, now - 10), _row(["tests/test_short.py"], to, 20.0, now - 10)]   # a shorter timeout then
    out = tmp_path / "rows.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows) + "not json\n", encoding="utf-8")
    assert P.overrun_sets(out, 45.0, now) == {frozenset({"tests/test_slow.py"}): 2}
    assert P.overrun_sets(tmp_path / "missing.jsonl", 45.0, now) == {}
