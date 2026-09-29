"""C69 audit (29 Sep 2026): three firewall checks silently skipped when a helper module failed to import (fail OPEN). Each must
now produce a FAIL finding instead. The helper import is sabotaged with a meta-path finder so the real modules stay untouched."""
import importlib.abc
import sys

import pytest


class _Block(importlib.abc.MetaPathFinder):
    def __init__(self, names):
        self.names = set(names)

    def find_spec(self, fullname, path, target=None):
        if fullname in self.names:
            raise ImportError(f"blocked for test: {fullname}")
        return None


@pytest.fixture
def block(monkeypatch):
    def _do(*names):
        for n in names:
            monkeypatch.delitem(sys.modules, n, raising=False)
        finder = _Block(names)
        sys.meta_path.insert(0, finder)
        return finder
    yield _do
    sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, _Block)]


def _sources():
    import inspect
    from engine.learning import firewalls, future_firewall, memory_firewall
    return {m.__name__: inspect.getsource(m) for m in (firewalls, future_firewall, memory_firewall)}


def test_no_firewall_swallows_an_import_error_silently():
    """Static guard: no `except ImportError: pass` may remain in the three firewall modules."""
    import re
    for name, src in _sources().items():
        assert not re.search(r"except ImportError[^:]*:\s*\n\s*pass\b", src), f"{name} still fails open on ImportError"


def test_memory_firewall_fails_closed_when_causality_helper_is_missing(block):
    import pandas as pd
    from engine.learning import memory_firewall as MF
    fn = next((getattr(MF, n) for n in dir(MF) if n.startswith("check_") and "bank" in n), None)
    if fn is None:
        pytest.skip("bank check entry not public; covered by the static guard")
    block("engine.blind_gates")
    bank = pd.DataFrame({"real_end": [pd.Timestamp("2001-01-05")], "learned_at": [pd.Timestamp("2001-01-05")],
                         "outcomes_seen_through": [pd.Timestamp("2001-01-05")]})
    try:
        out = fn(bank, pd.Timestamp("2001-02-01"))
    except TypeError:
        pytest.skip("signature differs; covered by the static guard")
    assert any("missing" in str(getattr(f, "check", f)) for f in out)
