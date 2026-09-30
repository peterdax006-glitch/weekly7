"""F21 (C75 phase 17, type safety). The type-only edits must not change a result; these tests pin the few places where a type fix made
an implicit failure explicit (a value that used to be None at a use site now raises a named error) and the places where the narrowing
refactor touched a formula. Synthetic data only, runs in milliseconds."""
from __future__ import annotations

import numpy as np
import pytest

from engine.research import autopsy as AU
from engine.research import break_research as BR
from engine.research import cross_section as CS
from engine.research import multiscale as MS
from engine.research import regimes as RG


def test_kmeans_without_a_start_is_refused_not_returned_as_none():
    """Found by the type check: n_init < 1 skipped the loop and returned None from a function typed to return a tuple."""
    X = np.random.default_rng(0).normal(size=(30, 2))
    with pytest.raises(ValueError, match="n_init"):
        CS.kmeans(X, 3, seed=1, n_init=0)
    labels, centroids, inertia = CS.kmeans(X, 3, seed=1, n_init=2)
    assert labels.shape == (30,) and centroids.shape == (3, 2) and inertia >= 0


def test_required_share_names_a_missing_measurement():
    assert CS._required(0.25) == 0.25
    with pytest.raises(ValueError):
        CS._required(None)


def test_empty_autopsy_has_no_parts_to_render():
    empty = AU.Autopsy("2026-01-05", "2026-01-06", "h", True, None, None, None, None, None)
    with pytest.raises(AU.AutopsyError):
        empty.parts()


def test_benjamini_hochberg_keeps_missing_p_values_missing():
    """The refactor sorted on a dict of the known p-values; None entries must stay None and not count toward the number of tests."""
    q = MS.benjamini_hochberg([0.01, None, 0.04])
    assert q[1] is None
    assert q[0] == pytest.approx(0.02) and q[2] == pytest.approx(0.04)
    assert MS.benjamini_hochberg([None, None]) == [None, None]
    assert MS.benjamini_hochberg([]) == []


def test_scale_effect_absolute_helpers_match_the_properties_they_replace():
    eff = MS.ScaleEffect("d1", 5, 100, 50, 10.0, -0.02, 0.005, -4.0, 0.0, 0.0, 0.5, -4.0, "2026-01-01")
    assert eff.abs_t == 4.0
    assert eff.abs_per_session == abs(eff.per_session)
    unmeasured = MS.ScaleEffect("d1", 5, 0, 0, 0.0, None, None, None, None, None, None, None, "2026-01-01", status="INSUFFICIENT")
    with pytest.raises(ValueError):
        unmeasured.abs_t
    with pytest.raises(ValueError):
        unmeasured.abs_per_session


def test_state_effect_reports_measured_pair_or_refuses():
    measured = RG.StateEffect("vol", "high", 40, 0.01, 0.004, 2.5)
    assert measured.effect_se() == (0.01, 0.004)
    with pytest.raises(ValueError):
        RG.StateEffect("vol", "low", 3, 0.02, None, None).effect_se()


def test_pool_states_shrinks_the_thin_state_most():
    """pool_states was rewritten around (state, effect, se) triples; the DerSimonian-Laird shrinkage must be unchanged in kind."""
    states = [RG.StateEffect("a", "x", 100, 0.010, 0.002, 5.0), RG.StateEffect("a", "y", 100, 0.012, 0.002, 6.0),
              RG.StateEffect("a", "thin", 5, 0.080, 0.050, 1.6)]
    pooled, tau2, mu = RG.pool_states(states)
    assert len(pooled) == 3 and tau2 >= 0 and mu is not None
    by = {p.state: p for p in pooled}
    assert abs(by["thin"].effect - mu) < abs(0.080 - mu)            # the thin state moved toward the pooled mean
    assert by["thin"].weight < by["x"].weight                        # and kept less of its own estimate than a well-measured state
    assert RG.pool_states(states[:1]) == ([], 0.0, None)


def test_fit_threshold_abstains_when_a_side_is_too_small():
    """Typed `tuple | None`: too few positives or negatives is an abstention, not a fabricated threshold."""
    x = np.arange(10, dtype=float)
    none_pos = np.zeros(10, bool)
    rows = np.arange(10)
    assert BR.fit_threshold(x, 1, none_pos, ~none_pos, rows, {}) is None
