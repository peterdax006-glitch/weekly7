"""Bible PHASE 32 levels 2, 4, 5, 6, 7: the whole chain features -> pattern miner -> engine.adaptive session replay on a
small synthetic market (Phases 3, 18, 21-23; checklist A2, T2, T9, T10, T11, T13, T16).

  * walk-forward replay: the miner is refit on realised history only, and the replayed decisions fill at the NEXT open;
  * the planted edge shows up in the replayed weekly returns, beats a noise-only market run through the same pipeline
    and beats a random-score control - and the comparison could fail (the controls are real runs, not constants);
  * end-to-end future scramble: prices after the cut do not change any decision or fill up to the cut for the honest
    chain, and DO change them for a chain that peeks - so the gate is capable of failing;
  * degenerate: no snapshots means no trades and flat equity; too little history still replays without error."""
import numpy as np
import pandas as pd
import pytest

from engine import adaptive as A
from pipeline_world import CFG, decision_days, make_world, replay_chain as replay, scramble_after

START = 480                                   # first replayed session; the miner sees history before it only
N_DAYS = 780


def weekly_mean(S):
    return float(np.mean(S.weeks)) if S.weeks else 0.0


@pytest.fixture(scope="module")
def planted():
    w = make_world(seed=0, n_days=N_DAYS)
    S, P, sn = replay(w)
    return w, S, P, sn


@pytest.fixture(scope="module")
def noise_run():
    w = make_world(seed=3, n_days=N_DAYS, eff=0.0)
    return replay(w)[0]


def test_walk_forward_refits_use_only_realised_history(planted):
    _, _, P, _ = planted
    fits = [(a, last) for a, last, n in P.fit_log if last is not None]
    assert len(fits) >= 5
    assert [a for a, _ in fits] == sorted(a for a, _ in fits)          # chronological
    assert all(last <= a for a, last in fits)                            # every label realised by its decision date


def test_pattern_decisions_fill_at_the_next_open(planted):
    w, S, _, _ = planted
    days = list(w["Close"].index[START:].strftime("%Y-%m-%d"))
    decided = {d for d, _ in S.decisions}
    assert S.orders and decided
    for day, tk, dv, reason in S.orders:
        assert days.index(day) > 0 and days[days.index(day) - 1] in decided, (day, reason)


def test_replay_trades_only_listed_tickers_within_the_cash_budget(planted):
    w, S, _, _ = planted
    assert {o[1] for o in S.orders} <= set(w["Close"].columns)
    assert min(v for _, v in S.days) > 0 and np.isfinite([v for _, v in S.days]).all()


def test_planted_edge_beats_noise_market_and_random_control(planted, noise_run):
    w, S, P, sn = planted
    rng = np.random.default_rng(11)
    rand_snaps = {d: s.assign(mu_raw=rng.random(len(s))) for d, s in sn.items()}      # same stocks, random scores
    S_rand, _, _ = replay(w, snaps=rand_snaps)
    edge = weekly_mean(S)
    assert edge > weekly_mean(noise_run) + 0.008, (edge, weekly_mean(noise_run))
    assert edge > weekly_mean(S_rand) + 0.008, (edge, weekly_mean(S_rand))


def test_end_to_end_future_scramble_leaves_honest_decisions_unchanged():
    w = make_world(seed=5, n_days=760)
    cut = decision_days(w["Close"], w["Close"].index[START])[14]
    S1, _, _ = replay(w, end=cut, refit_every=6)
    S2, _, _ = replay(scramble_after(w, cut, seed=8), end=cut, refit_every=6)
    upto = lambda S: ([d for d in S.decisions if pd.Timestamp(d[0]) <= cut],
                      [o for o in S.orders if pd.Timestamp(o[0]) <= cut])
    assert upto(S1)[0] and upto(S1) == upto(S2)


def test_end_to_end_scramble_exposes_a_peeking_chain():
    """Control: the same gate run on a chain whose score reads the next week's return must show a difference."""
    w = make_world(seed=5, n_days=760)
    cut = decision_days(w["Close"], w["Close"].index[START])[14]
    fresh = {**CFG, "exit_q": 2.0}                  # never keep a holding: every pick comes straight from today's scores
    S1, _, _ = replay(w, end=cut, refit_every=6, leak="peek", cfg=fresh)
    S2, _, _ = replay(scramble_after(w, cut, seed=8), end=cut, refit_every=6, leak="peek", cfg=fresh)
    dec = lambda S: [d for d in S.decisions if pd.Timestamp(d[0]) <= cut]
    assert dec(S1) != dec(S2)


def test_no_snapshots_means_no_trades_and_flat_equity():
    w = make_world(seed=8, n_days=200, n_tickers=10)
    S = A.replay(CFG, {}, w["Close"].iloc[100:], 0.0, {}, opens=w["Open"].iloc[100:])
    assert S.orders == [] and S.decisions == []
    assert {v for _, v in S.days} == {1000.0}


def test_too_little_history_replays_with_a_no_signal_miner():
    w = make_world(seed=8, n_days=260, n_tickers=20)
    S, P, _ = replay(w, start=80, end=w["Close"].index[200], refit_every=100)
    assert P.fit_log and P.fit_log[0][1] is None                          # the miner was never asked
    assert S.orders and np.isfinite(S.result()["year_return"])
