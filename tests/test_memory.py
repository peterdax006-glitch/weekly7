"""Bible Phase 9 (canon C34): factor-weighted memory. Each weighting factor is checked in isolation - recency,
market similarity, long-term source discount, shrinkage, shock cut - plus bank round-trip and the empty case."""
import numpy as np
import pandas as pd
import pytest

from engine.memory import CTX, Memory, context_of

Z = np.zeros(len(CTX))


def ctx(v):
    c = np.zeros(len(CTX)); c[0] = v
    return c


def test_empty_memory_knows_nothing():
    m, se, n = Memory().estimate(("k", 3), 10, Z)
    assert m == 0.0 and np.isinf(se) and n == 0.0


def test_recency_halves_weight_per_half_life():
    M = Memory({"mem_half_life": 4.0})
    M.record("a", 0, Z, 0.01)
    e = M.ep[0]
    assert M.weight(e, 4, Z) == pytest.approx(0.5 * M.weight(e, 0, Z))
    assert M.weight(e, 8, Z) == pytest.approx(0.25 * M.weight(e, 0, Z))


def test_recent_outcomes_dominate():
    M = Memory({"mem_half_life": 2.0, "mem_shrink": 0.0, "mem_shock_k": 1e9})
    for w in range(10):
        M.record("a", w, Z, -0.02)
    for w in range(10, 14):
        M.record("a", w, Z, +0.02)
    assert M.estimate("a", 14, Z)[0] > 0                               # the latest regime wins


def test_similar_markets_count_more():
    M = Memory({"mem_bandwidth": 0.5, "mem_shrink": 0.0, "mem_shock_k": 1e9})
    for w in range(6):
        M.record("a", w, ctx(-1.0 if w % 2 else 1.0), -0.03 if w % 2 else 0.03)   # +3% in calm, -3% in fear
    calm = M.estimate("a", 6, ctx(1.0))[0]
    fear = M.estimate("a", 6, ctx(-1.0))[0]
    assert calm > 0 > fear


def test_shrinkage_pulls_thin_evidence_toward_zero():
    lo = Memory({"mem_shrink": 0.0}); hi = Memory({"mem_shrink": 20.0})
    for M in (lo, hi):
        M.record("a", 0, Z, 0.05)
    assert abs(hi.estimate("a", 1, Z)[0]) < abs(lo.estimate("a", 1, Z)[0])


def test_long_term_episodes_discounted_by_prior_scale():
    lt = pd.DataFrame({"arm": [repr(("k", 3))], "ctx": [list(Z)], "outcome": [0.04]})
    M = Memory({"mem_prior_scale": 0.3}, long_term=lt)
    M.record(("k", 3), 0, Z, 0.04)
    own = [e for e in M.ep if e[4] == 0][0]; old = [e for e in M.ep if e[4] == 1][0]
    assert M.weight(old, 0, Z) == pytest.approx(0.3 * M.weight(own, 0, Z))


def test_long_term_arm_text_round_trips_to_tuple():
    M = Memory()
    M.record(("k", 3, 0.5), 0, Z, 0.01)
    ex = M.export()
    M2 = Memory(long_term=ex)
    assert M2.ep[0][0] == ("k", 3, 0.5)                                # stored as repr, read back as the same arm


def test_export_contains_only_this_windows_episodes():
    lt = pd.DataFrame({"arm": [repr("a")], "ctx": [list(Z)], "outcome": [0.01]})
    M = Memory(long_term=lt)
    M.record("a", 0, Z, 0.02)
    ex = M.export()
    assert len(ex) == 1 and ex["outcome"].iloc[0] == 0.02             # the long-term row is not re-exported


def test_shock_detector_marks_a_break_and_cuts_old_evidence():
    M = Memory({"mem_shock_k": 1.0, "mem_shock_cut": 0.1})
    rng = np.random.default_rng(0)
    for w in range(30):
        M.record("a", w, Z, 0.01 + rng.normal(0, 0.002))
    for w in range(30, 36):
        M.record("a", w, Z, -0.05)                                     # the arm stops working
    assert "a" in M.breaks and M.breaks["a"] >= 30
    old = M.ep[5]
    assert M.weight(old, 36, Z) < 0.2 * 0.5 ** ((36 - 5) / M.p["mem_half_life"]) + 1e-12


def test_no_shock_on_stable_arm():
    M = Memory()
    rng = np.random.default_rng(1)
    for w in range(60):
        M.record("a", w, Z, rng.normal(0.01, 0.005))
    assert "a" not in M.breaks


def test_arms_do_not_leak_into_each_other():
    M = Memory({"mem_shrink": 0.0})
    M.record("a", 0, Z, 0.05)
    M.record("b", 0, Z, -0.05)
    assert M.estimate("a", 1, Z)[0] > 0 > M.estimate("b", 1, Z)[0]


def test_context_of_reads_market_columns_and_fills_missing():
    snap = pd.DataFrame({"m_vix": [0.7, 0.7], "x": [1, 2]})
    c = context_of(snap)
    assert c[0] == pytest.approx(0.7) and (c[1:] == 0).all()
