"""CR K11: the debug component."""
from __future__ import annotations

from creator import debug as D

TB = '''Traceback (most recent call last):
  File "app/main.py", line 10, in run
    helper(x)
  File "app/util.py", line 3, in helper
    return d["k"]
KeyError: 'k'
'''


def test_reproduce_failure_and_success():
    bad = D.reproduce(D.python_cmd("-c", "raise ValueError('boom')"))
    assert bad.reproduced and bad.returncode != 0 and "ValueError" in bad.output
    ok = D.reproduce(D.python_cmd("-c", "print(1)"))
    assert not ok.reproduced and ok.returncode == 0


def test_reproduce_empty_and_missing_and_timeout():
    assert D.reproduce([]).reproduced
    miss = D.reproduce(["definitely-not-a-binary-xyz"])
    assert miss.reproduced and miss.returncode == -1 and miss.output
    slow = D.reproduce(D.python_cmd("-c", "import time; time.sleep(5)"), timeout=0.3)
    assert slow.reproduced and "Timeout" in slow.output


def test_localise_innermost_first_and_prefer():
    locs = D.localise(TB)
    assert (locs[0].file, locs[0].line, locs[0].function) == ("app/util.py", 3, "helper")
    assert D.localise(TB, prefer="main")[0].file == "app/main.py"
    assert D.localise("") == [] and D.localise("nothing here") == []


def test_localise_short_form():
    assert D.localise("tests/test_a.py:12: AssertionError")[0].line == 12


def test_classify():
    assert D.classify(TB) == "KEY"
    assert D.classify("ModuleNotFoundError: No module named 'x'") == "IMPORT"
    assert D.classify("E   assert 1 == 2") == "ASSERTION"
    assert D.classify("") == "UNKNOWN"
    assert D.classify("all fine") == "UNKNOWN"


def test_diagnose_full():
    d = D.diagnose(TB)
    assert d.classification == "KEY" and d.exception == "KeyError"
    assert "app/util.py:3" in d.root_cause
    assert d.hypotheses and d.evidence
    assert d.to_dict()["classification"] == "KEY"


def test_diagnose_empty_and_unknown():
    d = D.diagnose("")
    assert d.classification == "UNKNOWN" and d.locations == [] and "no failure output" in d.root_cause
    assert D.diagnose("weird output").root_cause.startswith("undetermined")


def test_propose_repairs():
    props = D.propose_repairs(D.diagnose(TB))
    assert props and props[0].rank == 0 and props[0].target == "app/util.py:3"
    assert [p.rank for p in props] == list(range(len(props)))
    assert D.propose_repairs(D.Diagnosis("UNKNOWN")) == []


def test_repair_experiment_fixed(tmp_path):
    f = tmp_path / "t.py"
    f.write_text("raise SystemExit(1)")
    r = D.repair_experiment(D.python_cmd(str(f)), lambda: f.write_text("pass"))
    assert r["before"].reproduced and not r["after"].reproduced and r["fixed"]


def test_repair_experiment_not_fixed_reverts(tmp_path):
    f = tmp_path / "t.py"
    f.write_text("raise SystemExit(1)")
    calls = []
    r = D.repair_experiment(D.python_cmd(str(f)), lambda: None, lambda: calls.append(1))
    assert not r["fixed"] and calls == [1]


def test_repair_experiment_apply_raises():
    calls = []

    def boom():
        raise RuntimeError("nope")

    r = D.repair_experiment(D.python_cmd("-c", "raise SystemExit(1)"), boom, lambda: calls.append(1))
    assert not r["fixed"] and "RuntimeError" in r["error"] and calls == [1]
