"""creator.diskcache: persisted content-keyed caches must equal recomputation, invalidate on content/code change, survive
corruption and concurrent writers."""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

from creator import diskcache as DC
from creator import selfmodel as SM
from creator import testrun as T


@pytest.fixture(autouse=True)
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "cache"
    monkeypatch.setenv("WEEKLY7_CREATOR_CACHE", str(d))
    monkeypatch.setattr(SM, "_COMPONENT_CACHE", {})
    monkeypatch.setattr(T, "_IMPORTS_CACHE", {})
    return d


def _tree(root: Path) -> None:
    (root / "pkg").mkdir()
    (root / "pkg" / "__init__.py").write_text("")
    (root / "pkg" / "a.py").write_text('"""A."""\nimport os\nfrom . import b\n\ndef f(x: int) -> int:\n    return x\n')
    (root / "pkg" / "b.py").write_text("def g():\n    pass\n")
    (root / "pkg" / "bad.py").write_text("def (:\n")


def _view(root: Path) -> tuple[object, object]:
    g = T.ImportGraph.build(root)
    comps = {r: SM.scan_component(root, r) for r in ("pkg/a.py", "pkg/b.py", "pkg/bad.py")}
    return ({k: sorted(v) for k, v in g.imports.items()}, dict(g.unparsable)), comps


def _files(d: Path) -> list[Path]:
    return [p for p in d.rglob("*.pkl")]


def test_hit_equals_recompute(tmp_path: Path, cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _tree(tmp_path)
    cold = _view(tmp_path)
    assert _files(cache)
    monkeypatch.setattr(SM, "_COMPONENT_CACHE", {})            # simulate a fresh process: only the disk survives
    monkeypatch.setattr(T, "_IMPORTS_CACHE", {})
    monkeypatch.setattr(SM, "_scan_component_uncached", lambda *a: pytest.fail("recomputed"))
    assert _view(tmp_path) == cold
    monkeypatch.undo()                                          # real analysis again, cache disabled: same answer
    monkeypatch.setenv("WEEKLY7_CREATOR_CACHE", "off")
    monkeypatch.setattr(SM, "_COMPONENT_CACHE", {})
    monkeypatch.setattr(T, "_IMPORTS_CACHE", {})
    assert _view(tmp_path) == cold


def test_content_change_invalidates(tmp_path: Path) -> None:
    _tree(tmp_path)
    before = _view(tmp_path)
    (tmp_path / "pkg" / "b.py").write_text("import json\ndef g():\n    return 1\n")
    after = _view(tmp_path)
    assert after != before
    assert "json" in T.ImportGraph.build(tmp_path).imports["pkg.b"]
    assert SM.scan_component(tmp_path, "pkg/b.py").stubs == ()


def test_code_salt_change_invalidates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _tree(tmp_path)
    _view(tmp_path)
    monkeypatch.setattr(SM, "_COMPONENT_CACHE", {})
    monkeypatch.setattr(SM, "_COMPONENT_SALT", ["changed-analysis-code"])
    calls: list[str] = []
    real = SM._scan_component_uncached
    monkeypatch.setattr(SM, "_scan_component_uncached", lambda full, rel, raw: (calls.append(rel), real(full, rel, raw))[1])
    SM.scan_component(tmp_path, "pkg/a.py")
    assert calls == ["pkg/a.py"]
    assert DC.salt_of([DC.key_of]) != DC.salt_of([DC.get])
    assert DC.salt_of([DC.key_of]) != DC.salt_of([DC.key_of], [b"ruler"])


def test_corrupt_entry_ignored(tmp_path: Path, cache: Path) -> None:
    _tree(tmp_path)
    cold = _view(tmp_path)
    for p in _files(cache):
        p.write_bytes(p.read_bytes()[:7])                        # truncated
    (_files(cache)[0]).write_bytes(b"not a pickle at all")
    import creator.selfmodel as sm
    sm._COMPONENT_CACHE.clear()
    T._IMPORTS_CACHE.clear()
    assert _view(tmp_path) == cold


def test_concurrent_writers(cache: Path) -> None:
    errs: list[BaseException] = []

    def work(n: int) -> None:
        try:
            for i in range(60):
                DC.put("k", "salt", DC.key_of("same"), ("v", i, n))
                v = DC.get("k", "salt", DC.key_of("same"))
                assert v is None or v[0] == "v"
        except BaseException as e:
            errs.append(e)
    ts = [threading.Thread(target=work, args=(n,)) for n in range(2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs
    assert DC.get("k", "salt", DC.key_of("same"))[0] == "v"
    assert not list(cache.rglob("*.tmp"))
