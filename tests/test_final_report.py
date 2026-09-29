"""Phase 46 report generator: the distribution section reports the whole distribution (not just a mean) correctly,
audits PASS only with complete evidence (missing evidence is never a pass), and checklist states are counted."""
import importlib.util

import numpy as np
import pytest

from engine import config as K

spec = importlib.util.spec_from_file_location("final_report", K.ROOT / "scripts" / "final_report.py")
FR = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FR)


def test_distribution_on_known_weeks():
    w = [0.10, -0.05, 0.02, 0.07, -0.20]
    d = FR.distribution(w)
    assert d["weeks"] == 5 and d["worst"] == -0.20 and d["best"] == 0.10
    assert d["median"] == pytest.approx(0.02) and d["mean"] == pytest.approx(np.mean(w))
    eq = np.cumprod(1 + np.array(w))
    assert d["drawdown_chained"] == pytest.approx((eq / np.maximum.accumulate(eq) - 1).min())
    assert d["share_in_band_5_10"] == pytest.approx(3 / 5)              # 0.10, -0.05, 0.07


def test_distribution_empty_and_nan():
    assert FR.distribution([]) is None
    assert FR.distribution([np.nan, None]) is None


def test_audit_needs_complete_evidence():
    assert FR.audit({"a": True, "b": True})["verdict"] == "PASS"
    assert FR.audit({"a": True, "b": False})["verdict"].startswith("FAIL")
    v = FR.audit({"a": True, "b": None})["verdict"]
    assert v.startswith("INCOMPLETE") and "b" in v                       # missing evidence is never a pass
    assert FR.audit({})["verdict"] == FR.NO


def test_checklist_counts_all_states():
    txt = "- [x] a\n- [~] b\n- [~] c\n- [!] d\n- [ ] e\n- [?] f\nnot an item\n"
    c = FR.checklist_counts(txt)
    assert c == {"validated": 1, "implemented/testing": 2, "failed": 1, "not started": 1, "unproven": 1}


def test_config_hash_is_order_independent_and_sensitive():
    assert FR.cfg_hash({"a": 1, "b": 2}) == FR.cfg_hash({"b": 2, "a": 1}) != FR.cfg_hash({"a": 1, "b": 3})
