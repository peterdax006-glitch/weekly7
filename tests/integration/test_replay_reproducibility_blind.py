"""Bible PHASE 32 levels 5 and 7 on the whole chain features -> pattern miner -> session replay (Phases 21-23, 33;
checklist T13, T16): two complete runs (fit, snapshots, replay) must agree exactly, and a disguised market (every ticker
renamed and reshuffled) must produce the same trades up to the renaming. Both are compared against planted-effect
markets so a chain that ignored its inputs (constant picks) would still be visible in the other tests."""
import pytest

from pipeline_world import make_world, permute_tickers, replay_chain as replay


def test_two_complete_runs_are_identical():
    w = make_world(seed=6, n_days=650)
    S1, P1, _ = replay(w, start=470, refit_every=12)
    S2, P2, _ = replay(w, start=470, refit_every=12)
    assert S1.decisions == S2.decisions and S1.orders == S2.orders and S1.days == S2.days
    assert P1.fit_log == P2.fit_log


def test_disguised_tickers_give_the_same_trades():
    """Blind/disguise (T16): rename and reshuffle every ticker; the trades must be the same up to the renaming."""
    w = make_world(seed=7, n_days=650, n_tickers=40)
    S1, _, _ = replay(w, start=470, refit_every=12)
    disguised, mp = permute_tickers(w, seed=13)
    S2, _, _ = replay(disguised, start=470, refit_every=12)
    picks = lambda S, m: [(d, sorted(m.get(t, t) for t in held)) for d, held in S.decisions]
    assert picks(S1, mp) == picks(S2, {})
    assert [v for _, v in S1.days] == pytest.approx([v for _, v in S2.days], rel=1e-9)
