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
    """F22 (Firewall 5): this test used to look for a `check_*bank*` function that does not exist and pytest.skip() - it had never run.
    The bank entry is memory_firewall.audit_bank_frame; it is exercised directly, with no skip path."""
    from engine.learning import memory_firewall as MF
    bank = pd.DataFrame({"real_end": [pd.Timestamp("2001-01-05")], "learned_at": [pd.Timestamp("2001-01-05")],
                         "outcomes_seen_through": [pd.Timestamp("2001-01-05")]})
    now = pd.Timestamp("2001-02-01")
    assert not [f for f in MF.audit_bank_frame(bank, now) if f.check == "bank-causality-check-missing"]    # null: helper present
    block("engine.blind_gates")
    out = MF.audit_bank_frame(bank, now)
    missing = [f for f in out if f.check == "bank-causality-check-missing"]
    assert len(missing) == 1 and missing[0].severity.value == "FAIL", out
