"""Creator student: learns rewrite templates from adopted lessons and applies them to unseen code."""
import json
from pathlib import Path
from types import SimpleNamespace

from creator.student import LessonStudent

EAGER1 = "import json\n\n\ndef dump(x):\n    return json.dumps(x)\n\n\ndef other():\n    return 1\n"
LAZY1 = "\n\ndef dump(x):\n    import json\n    return json.dumps(x)\n\n\ndef other():\n    return 1\n"
EAGER2 = "import math\n\n\ndef area(r):\n    \"\"\"circle\"\"\"\n    return math.pi * r * r\n"
LAZY2 = "\n\ndef area(r):\n    \"\"\"circle\"\"\"\n    import math\n    return math.pi * r * r\n"
THIRD = "import textwrap\n\nCONST = 3\n\n\ndef wrap(s):\n    return textwrap.fill(s, 5)\n\n\ndef twice(s):\n    return wrap(s) + wrap(s)\n"


def _lesson(i, b, a, adopted=True, kind="activation"):
    return {"lesson_id": f"L{i}", "package_id": f"P{i}", "component": "creator.x", "task_kind": kind, "objective": "o", "context": "",
            "files_before": {"creator/m.py": b}, "files_after": {"creator/m.py": a}, "reasoning": "", "solver": "claude",
            "adopted": adopted, "verdict": "ok"}


def _write(tmp_path, rows, raw=()):
    p = tmp_path / "l.jsonl"
    p.write_text("\n".join([*(json.dumps(r) for r in rows), *raw]) + "\n", encoding="utf-8")
    return p


def _pkg(rel):
    return SimpleNamespace(outputs=(rel,), task_kind="activation")


def test_learns_lazy_import_and_applies_to_unseen(tmp_path):
    p = _write(tmp_path, [_lesson(1, EAGER1, LAZY1), _lesson(2, EAGER2, LAZY2)], raw=["{corrupt"])
    st = LessonStudent(p)
    assert st.skipped == 1 and st.support("lazy_import") == 2
    assert st.can_attempt("activation") and not st.can_attempt("other_kind")
    wd = tmp_path / "wd"
    (wd / "creator").mkdir(parents=True)
    (wd / "creator" / "t.py").write_text(THIRD, encoding="utf-8")
    r = st(None, _pkg("creator/t.py"), wd)
    assert r.claimed_done and r.by == "self-student-v1"
    new = (wd / "creator" / "t.py").read_text(encoding="utf-8")
    assert not new.startswith("import textwrap") and new.count("import textwrap") == 1
    a, b = {}, {}
    exec(THIRD, a)
    exec(new, b)
    s = "hello wonderful world"
    assert a["wrap"](s) == b["wrap"](s) and a["twice"](s) == b["twice"](s) and a["CONST"] == b["CONST"]


def test_nothing_matches_and_non_adopted_ignored(tmp_path):
    wd = tmp_path / "wd"
    (wd / "creator").mkdir(parents=True)
    (wd / "creator" / "t.py").write_text(THIRD, encoding="utf-8")
    st = LessonStudent(_write(tmp_path, [_lesson(1, EAGER1, LAZY1, adopted=False), _lesson(2, EAGER2, LAZY2, adopted=None)]))
    assert not st.templates and not st.can_attempt("activation")
    r = st(None, _pkg("creator/t.py"), wd)
    assert not r.claimed_done and "no learned template" in r.notes
    assert (wd / "creator" / "t.py").read_text(encoding="utf-8") == THIRD
    assert not LessonStudent(tmp_path / "missing.jsonl").templates


def test_other_templates(tmp_path):
    rm_b = "def _dead(x):\n    return x\n\n\ndef live():\n    return 1\n"
    rm_a = "def live():\n    return 1\n"
    in_b = "def f(a):\n    t = a + 1\n    return t\n"
    in_a = "def f(a):\n    return (a + 1)\n"
    g_b = "def s(xs):\n    for x in xs:\n        print(x)\n    return 0\n"
    g_a = "def s(xs):\n    if not xs:\n        return 0\n    for x in xs:\n        print(x)\n    return 0\n"
    st = LessonStudent(_write(tmp_path, [_lesson(1, rm_b, rm_a), _lesson(2, in_b, in_a), _lesson(3, g_b, g_a)]))
    assert {"remove_unused", "inline_temp", "empty_guard"} <= set(st.templates)
    new, done = st.apply_source("def _gone(q):\n    return q\n\n\ndef keep():\n    y = 2 * 3\n    return y\n")
    assert "_gone" not in new and "inline_temp" in done and "remove_unused" in done
    ns = {}
    exec(new, ns)
    assert ns["keep"]() == 6
