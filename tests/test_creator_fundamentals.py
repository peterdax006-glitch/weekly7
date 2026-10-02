"""Fundamentals catalogue: complete, every computed check flags a planted violation and passes a clean change, principles_for is
kind-appropriate, the report feeds the status line and the constraint loop."""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

from creator import constraints as CON
from creator import fundamentals as FU

CLEAN = '''
def add(a, b):
    """Add."""
    return a + b
'''


def run(kind: str, path: str, old: str | None, new: str | None, corpus: dict | None = None, extra: list | None = None, notes: str = ""):
    ch = [FU.Change(path, old, new)] + (extra or [])
    return FU.run_checks(ch, kind, corpus or {}, notes)


def ids(rep) -> set[str]:
    return {v.principle for v in rep.violations}


TEST_CH = FU.Change("tests/test_x.py", None, "def test_a():\n    assert 1\n")


def test_catalogue_complete():
    seen = set()
    for p in FU.CATALOGUE:
        assert p.id not in seen
        seen.add(p.id)
        assert p.name and p.statement and p.why and p.where
        assert set(p.where) <= {FU.PLANNER, FU.STUDENT, FU.GUARD, FU.AUDIT, FU.TESTGEN}
        assert p.check is not None or p.no_check_reason, p.id
    need = {"small", "responsib", "dry", "yagni", "fails without", "weaken", "swallowed", "determinism", "secrets", "measure",
            "naming", "interface"}
    text = " ".join(p.name.lower() for p in FU.CATALOGUE)
    assert all(n in text for n in need)
    assert len(FU.catalogue()) == len(FU.CATALOGUE)


def test_clean_change_passes_every_check():
    old = CLEAN
    new = CLEAN + '\n\ndef sub(a, b):\n    """Sub."""\n    return a - b\n\n\nUSE = sub(2, 1)\n'
    rep = run("feature", "creator/m.py", old, new, extra=[TEST_CH])
    assert rep.total == 0, rep.violations
    assert not rep.errors


def test_f01_small_change():
    big = "\n".join(f"x{i} = {i}" for i in range(600))
    assert "F01" in ids(run("repair", "creator/m.py", None, big, extra=[TEST_CH]))
    assert "F01" not in ids(run("repair", "creator/m.py", None, "x = 1\n", extra=[TEST_CH]))


def test_f02_complexity_and_params():
    body = "".join(f"    if x == {i}:\n        r += {i}\n" for i in range(20))
    new = f"def f(x):\n    r = 0\n{body}    return r\n"
    assert "F02" in ids(run("feature", "creator/m.py", None, new, extra=[TEST_CH]))
    many = "def g(a, b, c, d, e, f, g, h, i):\n    return a\n"
    assert "F02" in ids(run("feature", "creator/m.py", None, many, extra=[TEST_CH]))
    old = new
    assert "F02" not in ids(run("feature", "creator/m.py", old, old.replace("r += 0", "r += 100"), extra=[TEST_CH]))   # not grown


DUP = "def f(x):\n    y = x + 1\n    z = y * 2\n    return z - 3\n"


def test_f03_dry():
    corp = {"creator/other.py": DUP.replace("def f", "def g")}
    assert "F03" in ids(run("feature", "creator/m.py", None, DUP, corp, extra=[TEST_CH]))
    assert "F03" not in ids(run("feature", "creator/m.py", None, DUP.replace("- 3", "- 4"), corp, extra=[TEST_CH]))


def test_f04_dead_public():
    assert "F04" in ids(run("feature", "creator/m.py", "", "def orphan():\n    return 1\n", {"creator/o.py": "x = 1\n"}, extra=[TEST_CH]))
    used = {"tests/test_o.py": "from creator.m import orphan\ndef test_o():\n    orphan()\n"}
    assert "F04" not in ids(run("feature", "creator/m.py", "", "def orphan():\n    return 1\n", used, extra=[TEST_CH]))
    assert "F04" not in ids(run("feature", "creator/m.py", "", "def _p():\n    return 1\n", extra=[TEST_CH]))


def test_f05_test_present():
    assert "F05" in ids(run("feature", "creator/m.py", "def a():\n    return 1\n", "def a():\n    return 2\n"))
    assert "F05" not in ids(run("feature", "creator/m.py", "def a():\n    return 1\n", "def a():\n    return 2\n", extra=[TEST_CH]))


def test_f06_weakening_reuses_audit():
    old = "def test_a():\n    assert f() == 3\n    assert g() == 4\n"
    new = "def test_a():\n    assert f() == 3\n"
    ch = FU.Change("tests/test_a.py", old, new)
    assert "F06" in ids(FU.run_checks([ch], "feature"))
    assert "F06" not in ids(FU.run_checks([FU.Change("tests/test_a.py", old, old + "    assert h()\n")], "feature"))


def test_f07_swallowed():
    bad = "def f():\n    try:\n        g()\n    except Exception:\n        pass\n"
    bare = "def f():\n    try:\n        g()\n    except:\n        return 1\n"
    good = "def f():\n    try:\n        g()\n    except ValueError as e:\n        raise RuntimeError('x') from e\n"
    assert "F07" in ids(run("repair", "creator/m.py", None, bad, extra=[TEST_CH]))
    assert "F07" in ids(run("repair", "creator/m.py", None, bare, extra=[TEST_CH]))
    assert "F07" not in ids(run("repair", "creator/m.py", None, good, extra=[TEST_CH]))
    assert "F07" not in ids(run("repair", "creator/m.py", bad, bad + "\n# c\n", extra=[TEST_CH]))        # pre-existing, not new


def test_f08_determinism():
    bad = "import time\n\ndef f():\n    return time.time()\n"
    inj = "import time\n\ndef f(now=None):\n    return now or time.time()\n"
    rnd = "import random\n\ndef f():\n    return random.random()\n"
    assert "F08" in ids(run("feature", "creator/m.py", None, bad, extra=[TEST_CH]))
    assert "F08" in ids(run("feature", "creator/m.py", None, rnd, extra=[TEST_CH]))
    assert "F08" not in ids(run("feature", "creator/m.py", None, inj, extra=[TEST_CH]))


def test_f09_secrets():
    bad1 = 'P = "C:\\\\Users\\\\Peter\\\\x"\n'
    bad2 = 'API_KEY = "abcdefgh12345678"\n'
    assert "F09" in ids(run("feature", "creator/m.py", None, bad1, extra=[TEST_CH]))
    assert "F09" in ids(run("feature", "creator/m.py", None, bad2, extra=[TEST_CH]))
    assert "F09" not in ids(run("feature", "creator/m.py", None, 'P = "relative/x"\n', extra=[TEST_CH]))


def test_f10_unmeasured_claim():
    bad = "# this is much faster now\nx = 1\n"
    assert "F10" in ids(run("feature", "creator/m.py", None, bad, extra=[TEST_CH]))
    assert "F10" in ids(run("feature", "creator/m.py", None, "x = 1\n", extra=[TEST_CH], notes="I optimised the loop"))
    assert "F10" not in ids(run("efficiency", "creator/m.py", None, bad, extra=[TEST_CH]))
    assert "F10" not in ids(run("feature", "creator/m.py", None, "# the counter\nx = 1\n", extra=[TEST_CH]))


def test_f11_readability():
    long = "def f():\n" + "".join(f"    x{i} = {i}\n" for i in range(70)) + "    return x0\n"
    deep = "def g(a):\n" + "".join("    " * (i + 1) + "if a:\n" for i in range(6)) + "    " * 7 + "return 1\n"
    assert "F11" in ids(run("feature", "creator/m.py", None, long, extra=[TEST_CH]))
    assert "F11" in ids(run("feature", "creator/m.py", None, deep, extra=[TEST_CH]))
    assert "F11" not in ids(run("feature", "creator/m.py", None, CLEAN, extra=[TEST_CH]))


def test_f12_interface_stability():
    old = "def keep():\n    return 1\n\n\ndef gone():\n    return 2\n"
    new = "def keep():\n    return 1\n"
    corp = {"creator/user.py": "from creator.m import gone\nx = gone()\n"}
    assert "F12" in ids(run("shrink", "creator/m.py", old, new, corp, extra=[TEST_CH]))
    assert "F12" not in ids(run("shrink", "creator/m.py", old, new, {"creator/user.py": "x = 1\n"}, extra=[TEST_CH]))


def test_principles_for_kinds():
    repair, shrink, cov = FU.principles_for("repair"), FU.principles_for("shrink"), FU.principles_for("coverage")
    assert "Reproduce before fixing" in repair and "Reproduce before fixing" not in shrink
    assert "DRY" in shrink and "DRY" not in cov
    assert "200 changed lines" in repair and "400 changed lines" in shrink
    assert FU.principles_for("something-new").startswith("Engineering fundamentals for this feature")
    assert all(len(FU.principles_for(k)) < 3000 for k in FU.ALL_KINDS)


def test_broken_check_is_a_note_not_a_failure():
    rep = FU.run_checks([FU.Change("creator/m.py", None, "def (:\n")], "feature")
    assert rep.total >= 0 and isinstance(rep.to_dict(), dict)


def test_status_line_and_constraint(tmp_path: Path):
    now = dt.datetime(2026, 10, 2, 12, 0, 0)
    assert FU.status_line(tmp_path).startswith("fundamentals: 0 violations over 0 candidates")
    for i, tot in enumerate((4, 2)):
        d = tmp_path / "cycles" / f"P{i}"
        d.mkdir(parents=True)
        (d / "fundamentals.json").write_text(json.dumps({"total": tot, "counts": {"F02": tot}}), encoding="utf-8")
        t = (now - dt.timedelta(hours=1)).timestamp()
        os.utime(d / "fundamentals.json", (t, t))
    assert "6 violations over 2 candidates (F02:6)" in FU.status_line(tmp_path)
    m = CON.fundamentals_metric(tmp_path, now, 24.0)
    assert m.value == 3.0 and m.loss == 0.15 and m.kind == "information"
    rep = CON.measure_all(tmp_path, now)
    assert any(x["name"] == "fundamentals_violations" for x in rep["ranked"])


def test_evaluate_candidate_never_raises(tmp_path: Path):
    rep = FU.evaluate_candidate(tmp_path / "missing", "HEAD", ["creator/x.py"], "feature")
    assert rep.total == 0


def test_a_hung_git_cannot_hang_the_candidate_evaluation(tmp_path, monkeypatch):
    """Validator 6: `git show` had no timeout, so a locked repository hung the kernel cycle that 'never raises'."""
    import subprocess
    seen = {}

    def hang(cmd, **kw):
        seen.update(kw)
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout") or 0)
    (tmp_path / "m.py").write_text("X = 1\n", encoding="utf-8")
    monkeypatch.setattr(FU.subprocess, "run", hang)
    rep = FU.evaluate_candidate(tmp_path, "HEAD", ["m.py"], "feature")
    assert seen.get("timeout") and "evaluate" in rep.errors
