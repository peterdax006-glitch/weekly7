"""creator/jellyc.py (Open-Jellycore in WSL) with a fake runner, and its optional hook in shortcutgen.validate. No WSL needed."""
from __future__ import annotations

from pathlib import Path

from creator import jellyc as J
from creator import shortcutgen as S

GOOD = 'import Shortcuts #Color: blue, #Icon: shortcuts\nspeakText(text: "hi")\n'


def test_wsl_path():
    assert J.wsl_path(Path("C:/Users/x/a b/f.jelly")) == "/mnt/c/Users/x/a b/f.jelly"


def test_parse_success_and_errors():
    ok = J.parse(0, "Successfully Compiled Shortcut\n")
    assert ok == {"available": True, "ok": True, "errors": [], "warnings": [], "output": "Successfully Compiled Shortcut\n"}
    tab = J.parse(1, "warning\t3\tMissing parameter rate in speakText\tInclude it\nerror\t7\tUnknown function fooBar\t\nFound 1 errors\n")
    assert not tab["ok"] and tab["errors"] == ["line 7: Unknown function fooBar"]
    assert tab["warnings"] == ["line 3: Missing parameter rate in speakText (Include it)"]
    warn_only = J.parse(0, "warning\t-\tMissing parameter wait in speakText\t\nSuccessfully Compiled Shortcut\n")
    assert warn_only["ok"] and warn_only["errors"] == [] and len(warn_only["warnings"]) == 1
    bad = J.parse(0, "Unknown function fooBar Check the documentation\nType mismatch on line 3 No Recovery Strategies Available\nFound 2 errors\n")
    assert not bad["ok"] and len(bad["errors"]) == 2 and bad["errors"][0].startswith("Unknown function fooBar")
    crash = J.parse(1, "")
    assert not crash["ok"] and "exited 1" in crash["errors"][0]


def test_compile_code_through_fake_wsl():
    calls: list[list[str]] = []

    def run(cmd, timeout):
        calls.append(cmd)
        assert cmd[:6] == ["wsl.exe", "-d", "Ubuntu", "-u", "root", "--"]
        if cmd[6] == "test":
            return 0, ""
        src = cmd[-1]
        assert src.startswith("/mnt/") and src.endswith("/in.jelly")
        return 0, J.OK_LINE
    r = J.compile_code(GOOD, run=run)
    assert r["available"] and r["ok"] and len(calls) == 2


def test_missing_compiler_is_not_an_error():
    r = J.compile_code(GOOD, run=lambda c, t: (1, "no such file"))
    assert r["available"] is False and r["ok"] is None
    assert S.validate(GOOD, compiler=lambda code: r) == []


def test_validate_uses_compiler_only_after_static_rules(monkeypatch):
    seen: list[str] = []

    def comp(code):
        seen.append(code)
        return {"available": True, "ok": False, "errors": ["Unknown function speakText"]}
    assert S.validate(GOOD, compiler=comp) == ["jelly compiler: Unknown function speakText"]
    assert S.validate("urlContents(url: \"x\")", compiler=comp)[0].startswith("line 1: urlContents is forbidden")
    assert len(seen) == 1                                                                 # forbidden code never reaches the compiler
    assert S.validate(GOOD) == []                                                         # default: static only
    monkeypatch.setenv("NUPEN_JELLY_CHECK", "1")
    monkeypatch.setattr(J, "compile_code", lambda code: {"available": True, "ok": True, "errors": []})
    assert S.validate(GOOD) == []


NUPEN2 = Path(__file__).resolve().parents[1] / "scripts" / "nupen2.jelly"


def test_nupen2_avoids_what_open_jellycore_cannot_parse():
    import re
    src = NUPEN2.read_text(encoding="utf-8")
    assert not re.search(r"headers: \{", src), "dictionary literal: write headers as a JSON string"
    assert not re.search(r"duration: \d", src), "bare time span: write duration: \"10 min\""
    assert "PASTE_TOKEN" in src and "PASTE_PC_ADDRESS" in src                            # the template keeps its placeholders


def test_real_compiler_accepts_nupen2_and_generated_code():
    import pytest
    if not J.available():
        pytest.skip("Open-Jellycore not built in WSL on this machine")
    r = J.compile_file(NUPEN2)
    assert r["ok"] and r["errors"] == [], r["errors"]
    made = S.make("make a shortcut that sets a timer for 10 minutes and says done")
    assert made["ok"], made
    assert S.validate(made["code"], compiler=J.compile_code) == []
