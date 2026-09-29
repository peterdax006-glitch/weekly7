"""Bible PHASE 31 (code quality firewall) and PHASE 32 (test pyramid) tooling: scripts/quality_gate.py and
scripts/test_inventory.py. Every check is proven against a PLANTED defect in a throw-away source tree (a check that cannot
fail is worthless), against the clean case, and against the empty/degenerate case; the gate's exit-code contract, its
baseline handling and its fail-closed behaviour are tested as well. Synthetic trees under tmp_path only; no network."""
import importlib.util
import json
import sys
import textwrap
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


G = _load("_qg_under_test", "quality_gate.py")
TI = _load("_ti_under_test", "test_inventory.py")

def tree(tmp_path, engine=None, scripts=None, tests=None, extra=None):
    """Write a source tree; every value is dedented source text. Returns the root."""
    for grp, files in (("engine", engine or {}), ("scripts", scripts or {}), ("tests", tests or {})):
        d = tmp_path / grp
        d.mkdir(parents=True, exist_ok=True)
        for name, src in files.items():
            p = d / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(textwrap.dedent(src), encoding="utf-8")
    for name, src in (extra or {}).items():
        (tmp_path / name).write_text(textwrap.dedent(src), encoding="utf-8")
    return tmp_path


def run(root, only, **cfg):
    c = G.load_config(root)
    c.update(cfg)
    ctx = G.build_context(root, c)
    ctx.inventory = TI                                     # the module under test (maps may be monkeypatched)
    findings, _ = G.run_checks(ctx, only)
    return findings


def msgs(findings, severity=None):
    return [f.msg for f in findings if severity is None or f.severity == severity]


DOC = '"""Bible phase 1 canon C1."""\n'


# ------------------------------------------------------------------ syntax / docstrings / config
def test_syntax_error_is_caught_and_clean_tree_passes(tmp_path):
    root = tree(tmp_path, engine={"good.py": DOC + "x = 1\n", "bad.py": "def f(:\n    pass\n"})
    out = run(root, ["syntax"])
    assert [f.path for f in out] == ["engine/bad.py"] and out[0].severity == "error"
    root2 = tree(tmp_path / "ok", engine={"good.py": DOC + "x = 1\n"})
    assert run(root2, ["syntax"]) == []


def test_empty_tree_has_nothing_to_report(tmp_path):
    (tmp_path / "engine").mkdir()
    assert run(tmp_path, [n for n in G.CHECKS if n not in ("inventory", "tests_pass")]) == []


def test_module_docstring_rules(tmp_path):
    root = tree(tmp_path, engine={"none.py": "x = 1\n", "vague.py": '"""does things."""\nx = 1\n', "ok.py": DOC + "x = 1\n",
                                  "__init__.py": ""}, tests={"test_a.py": "def test_a():\n    assert 1 + 1 == 2\n"})
    out = run(root, ["module_docstrings"])
    by = {f.path: f.severity for f in out}
    assert by["engine/none.py"] == "error" and by["engine/vague.py"] == "warning"
    assert "engine/ok.py" not in by and "engine/__init__.py" not in by
    assert by["tests/test_a.py"] == "warning"


def test_config_override_and_malformed_config(tmp_path):
    root = tree(tmp_path, extra={"pyproject.toml": "[tool.quality_gate]\ngiant_warn = 7\n"})
    assert G.load_config(root)["giant_warn"] == 7 and G.load_config(root)["giant_error"] == G.DEFAULTS["giant_error"]
    bad = tree(tmp_path / "b", extra={"pyproject.toml": "[tool.quality_gate\n"})
    with pytest.raises(SystemExit):
        G.load_config(bad)


# ------------------------------------------------------------------ import boundaries
def test_research_module_importing_the_broker_is_an_error_in_every_import_form(tmp_path):
    root = tree(tmp_path, engine={
        "broker.py": DOC + "def submit():\n    pass\n",
        "live.py": DOC + "from . import broker\n",                                   # live owns the order path: fine
        "iso.py": DOC + "x = 1\n",
        "a.py": DOC + "from . import broker\n",
        "b.py": DOC + "from .broker import submit\n",
        "c.py": DOC + "from engine.broker import submit\n",
        "d.py": DOC + "import engine.broker\n",
        "e.py": DOC + "def go():\n    from . import broker\n    return broker\n",       # a lazy import still counts
        "clean.py": DOC + "from . import iso\n"})
    out = run(root, ["import_boundaries"])
    assert sorted(f.path for f in out) == [f"engine/{n}.py" for n in "abcde"]


def test_live_import_needs_a_reviewed_coupling(tmp_path):
    root = tree(tmp_path, engine={"live.py": DOC + "x = 1\n", "improve.py": DOC + "from . import live\n",
                                  "other.py": DOC + "from . import live\n"})
    out = run(root, ["import_boundaries"])
    assert [f.path for f in out] == ["engine/other.py"]
    out2 = run(root, ["import_boundaries"], known_live_couplings={"improve": ["live"], "other": ["live"]})
    assert out2 == []


def test_alpaca_trading_sdk_only_in_the_broker_module(tmp_path):
    root = tree(tmp_path, engine={
        "broker.py": DOC + "from alpaca.trading.client import TradingClient\n",
        "tick.py": DOC + "from alpaca.trading.client import TradingClient\n",
        "md.py": DOC + "from alpaca.data.historical import StockHistoricalDataClient\n"},
        scripts={"research.py": "from alpaca.trading.requests import MarketOrderRequest\n"})
    out = run(root, ["import_boundaries"])
    assert sorted(f.path for f in out) == ["engine/tick.py", "scripts/research.py"]


def test_scripts_may_not_reach_live_unless_listed(tmp_path):
    root = tree(tmp_path, engine={"live.py": DOC + "x = 1\n", "broker.py": DOC + "x = 1\n"},
                scripts={"sim.py": "from engine import live\n", "livesim_cycle.py": "from engine import live, broker\n",
                         "fine.py": "import json\n"})
    out = run(root, ["import_boundaries"])
    assert [f.path for f in out] == ["scripts/sim.py"]


def test_engine_may_not_import_its_callers(tmp_path):
    root = tree(tmp_path, engine={"m.py": DOC + "import scripts.helper\n"})
    assert msgs(run(root, ["import_boundaries"])) == ["engine imports 'scripts' (a caller of engine)"]


def test_top_level_import_cycle_detected_and_lazy_import_breaks_it(tmp_path):
    cyc = tree(tmp_path, engine={"a.py": DOC + "from . import b\n", "b.py": DOC + "from . import c\n",
                                 "c.py": DOC + "from . import a\n", "d.py": DOC + "from . import a\n"})
    out = run(cyc, ["import_boundaries"])
    assert len(out) == 1 and "a -> b -> c -> a" in out[0].msg
    lazy = tree(tmp_path / "l", engine={"a.py": DOC + "from . import b\n", "b.py": DOC + "def f():\n    from . import a\n    return a\n"})
    assert run(lazy, ["import_boundaries"]) == []


def test_cycle_finder_on_raw_graphs():
    assert G._cycles({"a": {"b"}, "b": {"a"}, "c": {"a"}}) == [["a", "b"]]
    assert G._cycles({"a": {"b"}, "b": {"c"}}) == []
    assert G._cycles({}) == [] and G._cycles({"a": {"a"}}) == []          # a self edge is not a multi-module cycle


# ------------------------------------------------------------------ exceptions / prints / randomness / clock
def test_bare_except_is_an_error_and_swallowed_exception_a_warning(tmp_path):
    root = tree(tmp_path, engine={"m.py": DOC + textwrap.dedent('''
        def a():
            try:
                1 / 0
            except:
                return 1

        def b():
            try:
                1 / 0
            except Exception:
                pass

        def c():
            try:
                1 / 0
            except ZeroDivisionError:
                pass

        def d():
            try:
                1 / 0
            except Exception as e:
                raise RuntimeError("x") from e
        ''')})
    out = run(root, ["bare_except"])
    assert [(f.severity, f.line) for f in out] == [("error", 6), ("warning", 12)]


def test_print_and_debugger_left_in_engine(tmp_path):
    root = tree(tmp_path, engine={
        "m.py": DOC + "def f():\n    print('x')\n    breakpoint()\n",
        "data.py": DOC + "def f():\n    print('progress')\n",                        # downloader: allowed by config
        "n.py": DOC + "import pdb\n"})
    out = run(root, ["print_debug"])
    assert sorted((f.path, f.line) for f in out) == [("engine/m.py", 3), ("engine/m.py", 4), ("engine/n.py", 2)]


def test_print_in_main_block_is_allowed_but_library_print_is_not(tmp_path):
    """The command-line entry point may print; the same module's library code still may not (the exemption must
    not widen into the whole file)."""
    root = tree(tmp_path, engine={
        "u.py": DOC + "def f():\n    print('lib')\n\n\nif __name__ == \"__main__\":\n    print('cli')\n"})
    out = run(root, ["print_debug"])
    assert [(f.path, f.line) for f in out] == [("engine/u.py", 3)]


def test_network_client_and_url_calls_in_tests(tmp_path):
    root = tree(tmp_path, tests={
        "test_a.py": "import requests\n\ndef test_a():\n    assert requests\n",
        "test_b.py": "from urllib.request import urlopen\n\ndef test_b():\n    assert urlopen('http://x')\n",
        "test_c.py": "import socket\n\ndef test_c():\n    assert socket\n",
        "test_ok.py": "import json\n\ndef test_ok():\n    assert json\n"})
    out = run(root, ["network_in_tests"])
    assert sorted({f.path for f in out}) == ["tests/test_a.py", "tests/test_b.py", "tests/test_c.py"]
    assert sum(f.path == "tests/test_b.py" and "network module" in f.msg for f in out) == 1      # reported once per line


def test_tests_must_not_touch_caches_or_sealed_windows(tmp_path):
    root = tree(tmp_path, tests={
        "test_a.py": "from engine import config as K\n\ndef test_a():\n    assert K.CACHE\n",
        "test_b.py": "def test_b():\n    assert open('data/cache/x.parquet')\n",
        "test_c.py": "def test_c():\n    p = 'state\\\\livesim\\\\y'\n    assert p\n",
        "test_ok.py": "def test_ok(tmp_path):\n    assert tmp_path\n"})
    out = run(root, ["tests_touch_data"])
    assert sorted(f.path for f in out) == ["tests/test_a.py", "tests/test_b.py", "tests/test_c.py"]
    assert run(root, ["tests_touch_data"], test_allowed_data=["tests/test_a.py", "tests/test_b.py", "tests/test_c.py"]) == []


def test_unseeded_and_global_randomness(tmp_path):
    src = DOC + textwrap.dedent('''
        import random
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier

        def bad(n):
            a = np.random.rand(n)
            b = np.random.default_rng().normal(size=n)
            c = np.random.default_rng(None)
            d = random.random()
            e = RandomForestClassifier()
            f = np.random.RandomState()
            return a, b, c, d, e, f

        def good(n, seed):
            rng = np.random.default_rng(seed)
            g = np.random.default_rng(3)
            h = random.Random(5).random()
            i = RandomForestClassifier(random_state=1)
            return rng.normal(size=n), g, h, i
        ''')
    root = tree(tmp_path, engine={"m.py": src}, tests={"test_t.py": "import numpy as np\n\ndef test_t():\n    assert np.random.rand() >= 0\n"})
    out = run(root, ["randomness"])
    lines = sorted(f.line for f in out if f.path == "engine/m.py")
    assert lines == [8, 9, 10, 11, 12, 13]                                   # exactly the six in bad(), none in good()
    assert [f.severity for f in out if "RandomForest" in f.msg] == ["warning"]
    assert any(f.path == "tests/test_t.py" and f.severity == "error" for f in out)


def test_wall_clock_in_research_code_is_a_warning(tmp_path):
    root = tree(tmp_path, engine={"m.py": DOC + "import time\nfrom datetime import datetime\n\ndef f():\n    return datetime.now(), time.time()\n",
                                  "live.py": DOC + "from datetime import datetime\n\ndef f():\n    return datetime.now()\n"})
    out = run(root, ["wall_clock"])
    assert {f.path for f in out} == {"engine/m.py"} and {f.severity for f in out} == {"warning"} and len(out) == 2


# ------------------------------------------------------------------ test hygiene
def test_tests_that_cannot_fail_are_errors(tmp_path):
    src = textwrap.dedent('''
        import pytest

        def test_empty():
            pass

        def test_no_assert():
            x = 1 + 1

        def test_assert_true():
            assert True

        def test_real():
            assert 1 + 1 == 2

        def test_raises():
            with pytest.raises(ZeroDivisionError):
                1 / 0

        def test_approx():
            assert 0.1 + 0.2 == pytest.approx(0.3)

        def assert_close(a, b):
            assert a == b

        def test_helper():
            assert_close(1, 1)

        def test_direct_raise():
            raise AssertionError("x")

        @pytest.mark.skip
        def test_skipped():
            assert 2 > 1
        ''')
    root = tree(tmp_path, tests={"test_h.py": src})
    out = run(root, ["test_hygiene"])
    flagged = {f.msg.split(":")[0] for f in out if f.severity == "error"}
    assert flagged == {"test_empty", "test_no_assert", "test_assert_true"}
    assert any(f.severity == "warning" and "skip" in f.msg for f in out)


# ------------------------------------------------------------------ structure checks
def test_mutable_default_arguments(tmp_path):
    root = tree(tmp_path, engine={"m.py": DOC + "def a(x=[]):\n    return x\n\ndef b(x={}):\n    return x\n\ndef c(x=None):\n    return x\n\ndef d(x=list()):\n    return x\n"})
    assert sorted(f.msg.split(":")[0] for f in run(root, ["mutable_defaults"])) == ["a", "b", "d"]


def test_global_statement_flagged(tmp_path):
    root = tree(tmp_path, engine={"m.py": DOC + "_C = None\n\ndef f():\n    global _C\n    _C = 1\n"})
    assert [f.severity for f in run(root, ["hidden_global_state"])] == ["warning"]


def test_giant_function_thresholds(tmp_path):
    body = "\n".join(f"    x{i} = {i}" for i in range(60))
    root = tree(tmp_path, engine={"m.py": DOC + f"def big():\n{body}\n    return 1\n\ndef small():\n    return 1\n"})
    assert run(root, ["giant_functions"]) == []                                                   # under default 120
    out = run(root, ["giant_functions"], giant_warn=40, giant_error=1000)
    assert [(f.severity, f.msg.split()[0]) for f in out] == [("warning", "big")]
    out = run(root, ["giant_functions"], giant_warn=40, giant_error=50)
    assert [f.severity for f in out] == ["error"]


def _fn(name, params, op="+", n=10):
    """Source of a function with an n-line body; params are (a, b)."""
    a, b = params
    lines = "\n".join(f"    y{i} = {a} * {i} {op} {b}" for i in range(n))
    return f"def {name}({a}, {b}):\n{lines}\n    return y{n - 1}\n"


def test_duplicate_implementation_across_files(tmp_path):
    root = tree(tmp_path, engine={
        "one.py": DOC + _fn("score", ("a", "b")),
        "two.py": DOC + _fn("rank", ("p", "q")),                 # renamed copy of score
        "three.py": DOC + _fn("diff", ("a", "b"), op="-")})      # different logic
    out = run(root, ["duplicates"])
    assert [(f.path, f.severity) for f in out] == [("engine/two.py", "warning")]
    assert "rank duplicates score in engine/one.py" in out[0].msg


def test_duplicates_inside_one_file_and_short_bodies_are_ignored(tmp_path):
    root = tree(tmp_path, engine={"same.py": DOC + _fn("s1", ("a", "b")) + "\n\n" + _fn("s2", ("a", "b")),
                                  "s1.py": DOC + _fn("t", ("a", "b"), n=2), "s2.py": DOC + _fn("u", ("c", "d"), n=2)})
    assert run(root, ["duplicates"]) == []


def test_magic_constant_needs_a_reason(tmp_path):
    root = tree(tmp_path, engine={"m.py": DOC + textwrap.dedent('''
        BARE = 0.37
        SAME_LINE = 12          # weeks of history: shorter windows are too noisy
        # the vendor caps requests at 200 per call
        ABOVE = 200
        NAME = "text"
        ZERO = 0
        _PRIVATE = 99
        lower = 5
        TABLE = {"a": 0.5,
                 "b": 0.7}
        ''')})
    out = run(root, ["magic_constants"])
    assert sorted(f.msg.split()[0] for f in out) == ["BARE", "TABLE"]


# ------------------------------------------------------------------ baseline, verdict, exit codes
def _F(check="c", sev="error", path="p.py", line=1, msg="m"):
    return G.Finding(check, sev, path, line, msg)


def test_finding_identity_survives_line_moves():
    assert _F(line=3).key() == _F(line=300).key() and _F(msg="x").key() != _F(msg="y").key()


def test_decide_baseline_strict_and_stale_entries():
    e, w = _F(msg="err"), _F(sev="warning", msg="warn")
    v = G.decide([e, w], set(), strict=False)
    assert v["failing"] == [e] and v["exit_code"] == 1
    assert G.decide([w], set(), strict=False)["exit_code"] == 0
    assert G.decide([w], set(), strict=True)["exit_code"] == 1
    v = G.decide([e, w], {e.key()}, strict=False)
    assert v["exit_code"] == 0 and v["baselined"] == [e]
    new = _F(msg="brand new")
    assert G.decide([e, new], {e.key()}, strict=False)["failing"] == [new]          # a new finding is never waived
    assert G.decide([], {e.key()}, strict=False)["stale_baseline"] == [e.key()]


def test_baseline_roundtrip(tmp_path):
    p = tmp_path / "b" / "baseline.json"
    G.write_baseline(p, [_F(msg="a"), _F(msg="a", line=9), _F(msg="b")])
    assert G.read_baseline(p) == {_F(msg="a").key(), _F(msg="b").key()}
    assert G.read_baseline(None) == set()


def test_a_crashing_check_fails_the_gate_closed(tmp_path, monkeypatch):
    def boom(ctx):
        raise RuntimeError("kaput")
    monkeypatch.setitem(G.CHECKS, "syntax", boom)
    root = tree(tmp_path, engine={"good.py": DOC + "x = 1\n"})
    findings, _ = G.run_checks(G.build_context(root), ["syntax"])
    assert findings[0].severity == "error" and "crashed" in findings[0].msg
    assert G.decide(findings, set(), False)["exit_code"] == 1


def test_unknown_check_name_is_refused(tmp_path):
    root = tree(tmp_path, engine={"good.py": DOC + "x = 1\n"})
    with pytest.raises(SystemExit):
        G.run_checks(G.build_context(root), ["no_such_check"])


def test_main_exit_codes_and_json_report(tmp_path):
    clean = tree(tmp_path / "c", engine={"good.py": DOC + "x = 1\n"})
    dirty = tree(tmp_path / "d", engine={"bad.py": DOC + "def f():\n    try:\n        pass\n    except:\n        pass\n"})
    out = tmp_path / "r.json"
    only = "bare_except,syntax,print_debug"
    assert G.main(["--root", str(clean), "--only", only, "--json", str(out), "--quiet"]) == 0
    assert json.loads(out.read_text())["exit_code"] == 0
    assert G.main(["--root", str(dirty), "--only", only, "--json", str(out), "--quiet"]) == 1
    rep = json.loads(out.read_text())
    assert rep["exit_code"] == 1 and rep["findings"][0]["check"] == "bare_except" and rep["failing"]
    base = tmp_path / "base.json"
    assert G.main(["--root", str(dirty), "--only", only, "--write-baseline", str(base)]) == 0
    assert G.main(["--root", str(dirty), "--only", only, "--baseline", str(base), "--json", str(out), "--quiet"]) == 0


def test_strict_mode_turns_warnings_into_failures(tmp_path):
    root = tree(tmp_path, engine={"m.py": DOC + "def f():\n    try:\n        pass\n    except Exception:\n        pass\n"})
    args = ["--root", str(root), "--only", "bare_except", "--json", str(tmp_path / "o.json"), "--quiet"]
    assert G.main(args) == 0 and G.main(args + ["--strict"]) == 1


def test_run_tests_option_flags_failing_and_slow_files(tmp_path):
    root = tree(tmp_path, tests={"test_pass.py": "def test_ok():\n    assert 1\n", "test_fail.py": "def test_no():\n    assert 0\n"})
    out = run(root, ["tests_pass"])
    assert [(f.path, f.severity) for f in out] == [("tests/test_fail.py", "error")]
    slow = run(root, ["tests_pass"], test_seconds=0.0001)
    assert {("tests/test_pass.py", "warning"), ("tests/test_fail.py", "error")} == {(f.path, f.severity) for f in slow}


def test_real_repository_parses_cleanly():
    """The gate run over the actual repo: syntax is the one check that must be zero today (others are reported, not asserted)."""
    root = SCRIPTS.parent
    findings, stats = G.run_checks(G.build_context(root), ["syntax"])
    assert findings == [] and stats["syntax"]["errors"] == 0


# ------------------------------------------------------------------ test inventory
BIBLE = textwrap.dedent('''
    # PHASE 1 - FIRST THING
    text
    # PHASE 2 - SECOND THING
    text
    # PHASE 38 - MASTER CHECKLIST
    ## ALGORITHM
    * [x] A1 Alpha item
    * [ ] A2b Beta item
    ## LIVE
    * [~] L1 Gamma item
    # PHASE 39 - DEFINITION OF DONE
    ''').replace(" - ", " — ")


@pytest.fixture
def small_maps(monkeypatch):
    monkeypatch.setattr(TI, "PHASE_MODULES", {1: ["alpha"], 2: ["beta"]})
    monkeypatch.setattr(TI, "CHECKLIST_MAP", {"A1": ([1], []), "A2b": ([2], []), "L1": ([], ["beta"])})
    monkeypatch.setattr(TI, "REQUIRED_LEVELS", {1: (1, 6), 2: (1, 4)})


def suite_tree(tmp_path, tests):
    root = tree(tmp_path, engine={"alpha.py": DOC, "beta.py": DOC}, tests=tests, extra={"BIBLE.md": BIBLE})
    return root


def test_bible_parsing_finds_phases_items_and_sections():
    b = TI.parse_bible(BIBLE)
    assert sorted(b["phases"]) == [1, 2, 38, 39] and b["phases"][2]["title"] == "SECOND THING"
    assert [(i["id"], i["state"], i["section"]) for i in b["items"]] == [
        ("A1", "x", "ALGORITHM"), ("A2b", "", "ALGORITHM"), ("L1", "~", "LIVE")]
    assert TI.parse_bible("nothing here") == {"phases": {}, "items": []}


@pytest.mark.parametrize("text, phases, items", [
    ("Bible Phases 21-23 and T11", {21, 22, 23}, {"T11"}),
    ("PHASES 6 and 34", {6, 34}, set()),
    ("phases 3, 5-7 and 9", {3, 5, 6, 7, 9}, set()),
    ("(L2-L4) A13/T17", set(), {"L2", "L3", "L4", "A13", "T17"}),
    ("item A2b and V2b", set(), {"A2b", "V2b"}),
    ("no tags at all, version 3.11", set(), set()),
])
def test_tag_extraction(text, phases, items):
    p, i, _ = TI.extract_tags(text)
    assert (p, i) == (phases, items)


def test_inventory_reports_untested_items_and_direct_versus_module_evidence(tmp_path, small_maps):
    root = suite_tree(tmp_path, {
        "test_alpha.py": '"""Bible Phase 1: alpha."""\nfrom engine import alpha\n\ndef test_a():\n    assert alpha\n',
        "test_orphan.py": '"""Unrelated."""\ndef test_o():\n    assert 1\n'})
    inv = TI.run(root, write=False)
    st = {r["id"]: r["status"] for r in inv["items"]}
    assert st == {"A1": "direct", "A2b": "none", "L1": "none"}
    assert inv["gaps"]["items_untested"] == ["A2b", "L1"] and inv["gaps"]["phases_untested"] == [2]
    assert inv["gaps"]["orphan_test_files"] == ["tests/test_orphan.py"]
    # adding a test that merely imports beta (no tag) upgrades the beta items to module-level evidence
    (root / "tests" / "test_beta.py").write_text('"""Beta."""\nfrom engine import beta\n\ndef test_b():\n    assert beta\n')
    inv = TI.run(root, write=False)
    st = {r["id"]: r["status"] for r in inv["items"]}
    assert st["A2b"] == "module" and st["L1"] == "module" and inv["gaps"]["phases_untested"] == []
    (root / "tests" / "test_beta.py").write_text('"""Beta, checklist item A2b, Phase 2."""\nfrom engine import beta\n\ndef test_b():\n    assert beta\n')
    assert {r["id"]: r["status"] for r in TI.run(root, write=False)["items"]}["A2b"] == "direct"


def test_inventory_config_drift_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(TI, "PHASE_MODULES", {1: ["alpha", "ghost"], 9: []})
    monkeypatch.setattr(TI, "CHECKLIST_MAP", {"A1": ([1], []), "Z9": ([], []), "L1": ([], ["phantom"])})
    root = suite_tree(tmp_path, {})
    probs = TI.run(root, write=False)["gaps"]["config_problems"]
    joined = " | ".join(probs)
    assert "engine/ghost.py" in joined and "phase 9" in joined and "Z9" in joined
    assert "A2b has no CHECKLIST_MAP entry" in joined and "engine/phantom.py" in joined


def test_levels_are_assigned_from_names_location_and_module_use(tmp_path, small_maps):
    body = textwrap.dedent('''
        """Bible Phase 1."""
        from engine import alpha, beta

        def test_plain():
            assert alpha

        def test_scramble_future_is_caught():
            assert alpha

        def test_two_modules_together():
            assert alpha and beta

        def test_twice_is_reproducible():
            assert alpha

        def test_walk_forward_chronology():
            assert alpha

        def test_blind_disguise():
            assert alpha
        ''')
    root = suite_tree(tmp_path, {"test_lv.py": body, "integration/test_deep.py": '"""Phase 2."""\nfrom engine import beta\n\ndef test_x():\n    assert beta\n'})
    f = {s.path: s for s in TI.scan_suite(root / "tests", root)}
    lv = f["tests/test_lv.py"].tests
    assert lv["test_plain"] == [1] and 6 in lv["test_scramble_future_is_caught"] and 4 in lv["test_scramble_future_is_caught"]
    assert 2 in lv["test_two_modules_together"] and 7 in lv["test_twice_is_reproducible"]
    assert 4 in lv["test_walk_forward_chronology"] and 5 in lv["test_blind_disguise"]
    assert f["tests/integration/test_deep.py"].tests["test_x"] == [2]


def test_missing_levels_are_listed_per_phase(tmp_path, small_maps):
    root = suite_tree(tmp_path, {"test_a.py": '"""Bible Phase 1."""\ndef test_only_unit():\n    assert 1\n'})
    gaps = TI.run(root, write=False)["gaps"]["level_gaps"]
    assert gaps[1] == [6] and gaps[2] == [1, 4]          # phase 2 has no test at all, phase 1 lacks the adversarial level


def test_unparseable_test_file_is_skipped_not_fatal(tmp_path, small_maps):
    root = suite_tree(tmp_path, {"test_broken.py": "def test_x(:\n", "test_ok.py": '"""Phase 1."""\ndef test_y():\n    assert 1\n'})
    assert [s.path for s in TI.scan_suite(root / "tests", root)] == ["tests/test_ok.py"]


def test_inventory_writes_reports_and_cli_exit_code(tmp_path, small_maps):
    root = suite_tree(tmp_path, {"test_a.py": '"""Bible Phase 1."""\ndef test_a():\n    assert 1\n'})
    assert TI.main(["--root", str(root)]) == 0
    rep = json.loads((root / "state" / "quality" / "test_inventory.json").read_text())
    assert rep["n_tests"] == 1 and (root / "state" / "quality" / "test_inventory.md").read_text().startswith("# Test inventory")
    assert TI.main(["--root", str(root), "--fail-on-gap"]) == 1                # A2b has no test


def test_gate_inventory_check_fails_on_untested_item_and_passes_when_covered(tmp_path, small_maps):
    root = suite_tree(tmp_path, {"test_a.py": '"""Bible Phase 1."""\nfrom engine import alpha\n\ndef test_a():\n    assert alpha\n'})
    out = run(root, ["inventory"])
    errs = [f for f in out if f.severity == "error"]
    assert any("A2b" in f.msg for f in errs) and any("phase 2" in f.msg for f in errs)
    (root / "tests" / "test_b.py").write_text('"""Bible Phase 2, L1."""\nfrom engine import beta\n\ndef test_b():\n    assert beta\n')
    assert [f for f in run(root, ["inventory"]) if f.severity == "error"] == []
    (root / "BIBLE.md").unlink()
    assert "BIBLE.md not found" in run(root, ["inventory"])[0].msg


def test_real_bible_checklist_is_fully_mapped():
    """The inventory's own tables must cover the real Bible: no checklist id unmapped, no mapped module missing."""
    root = SCRIPTS.parent
    inv = TI.run(root, write=False)
    assert inv["gaps"]["config_problems"] == []
    assert len(inv["items"]) >= 40 and len(inv["phases"]) >= 46
