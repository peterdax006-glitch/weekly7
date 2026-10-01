"""Throughput caches must return exactly what the uncached computation returns, and must see an edit at once."""
from __future__ import annotations

from pathlib import Path

from creator import selfmodel as S
from creator import testrun as T


def _graph_view(g: T.ImportGraph) -> tuple[object, ...]:
    return (g.modules, g.imports, g.unparsable, g.nonmodule_tests)


def test_import_graph_cache_equals_uncached(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("from . import a\n")
    (tmp_path / "pkg" / "a.py").write_text("import os\nfrom pkg import b\n")
    (tmp_path / "pkg" / "bad.py").write_text("def (:\n")
    T._IMPORTS_CACHE.clear()
    cold = _graph_view(T.ImportGraph.build(tmp_path))
    warm = _graph_view(T.ImportGraph.build(tmp_path))
    assert cold == warm
    (tmp_path / "pkg" / "a.py").write_text("import json\n")                      # an edit is seen on the next build
    edited = T.ImportGraph.build(tmp_path)
    assert "json" in edited.imports["pkg.a"] and "os" not in edited.imports["pkg.a"]
    (tmp_path / "pkg" / "bad.py").write_text("x = 1\n")                         # an unparsable file fixed is seen
    assert "pkg/bad.py" not in T.ImportGraph.build(tmp_path).unparsable
    mutated = T.ImportGraph.build(tmp_path)                                    # callers may mutate their copy
    mutated.imports["pkg.a"].add("zzz")
    assert "zzz" not in T.ImportGraph.build(tmp_path).imports["pkg.a"]


def test_scan_component_cache_equals_uncached_and_sees_edit(tmp_path: Path) -> None:
    f = tmp_path / "m.py"
    f.write_text('"""doc"""\n\n\ndef f(x):\n    return x\n')
    S._COMPONENT_CACHE.clear()
    cold = S.scan_component(tmp_path, "m.py")
    assert S.scan_component(tmp_path, "m.py") == cold
    assert cold == S._scan_component_uncached(f, "m.py", f.read_bytes())
    f.write_text('"""doc"""\n\n\ndef g(x):\n    return x\n\n\ndef h():\n    ...\n')
    new = S.scan_component(tmp_path, "m.py")
    assert new != cold and {i.name for i in new.interfaces} == {"g", "h"}
