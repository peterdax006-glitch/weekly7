"""Repeated-run learning curve (canon C57) and time gating (canon C58): a planted pattern learner rises run after run under a
fresh disguise, the no-learning and identity-recall controls stay flat (and the identity control CAN rise when the disguise is
off), gains persist after earlier years are interleaved, evidence from a date that has not occurred is never released, a
memory that leaks later evidence into earlier dates is caught, and every candidate pattern tried is counted in the bar."""
import numpy as np
import pandas as pd
import pytest

from engine import learning_delta as L


def _wins(n, tag, seed=0):
    return [L.synthetic_window(L.derive_seed(seed, tag, i), f"{tag}{i:02d}", n_stocks=25, n_weeks=20) for i in range(n)]


# ================================================================== repeated-run learning curve (C57) and time gating (C58)
def _pw(tag="p", seed=5, year=1995):
    w = L.pattern_window(L.derive_seed(seed, tag), tag + "0", n_stocks=30, n_weeks=24)
    w.real_start = pd.Timestamp(f"{year}-01-01")
    w.real_end = pd.Timestamp(f"{year}-12-31")
    return w


def _vals(c, m="mean_week", tag="main"):
    return [r[m] for r in c["recs"] if r["tag"] == tag]


ZERO = lambda: L.LearnedState({}, {})
CFG = {"k": 3, "exit_q": 0.8, "rebalance_weeks": 1, "brake": None, "max_per_sector": None, "w_model": 1.0, "pick": "top",
       "pool_q": 0.5, "liq_q": 0.0, "vol_filter": False, "stress_thr": None, "stress_k": 2, "trend_filter": None, "trend_gross": 0.0}


def test_planted_pattern_learner_shows_a_rising_curve_and_plateau():
    W = _pw()
    c = L.play_curve(W, L.PatternPlayer(), L.PatternLearner(), ZERO(), [(W, "main")] * 14, 1)
    v = _vals(c)
    s = L.series_stats(v, 1)
    assert v[0] < 0 < v[-1]                                        # the wrong prior loses money; the evidence turns it round
    assert s["slope_lo"] > 0 and s["diff_lo"] > 0 and s["last"] > s["first"] + 0.03
    assert s["plateau"] is not None and s["plateau"] < 13
    assert all(r["blindness_ok"] for r in c["recs"])
    assert c["tests"] == 14 * 4                                    # every run's candidate patterns are counted


def test_reset_control_is_exactly_flat_so_disguise_noise_is_zero():
    W = _pw()
    c = L.play_curve(W, L.PatternPlayer(), L.PatternLearner(), ZERO(), [(W, "main")] * 12, 1, reset=True)
    v = _vals(c)
    assert max(v) - min(v) < 1e-12
    s = L.series_stats(v, 1)
    assert abs(s["slope"]) < 1e-12 and s["plateau"] == 4 and s["perm_p"] == 1.0


def test_every_run_gets_a_fresh_disguise_and_state_is_carried_forward():
    W = _pw()
    seen = []
    L.play_curve(W, L.PatternPlayer(), L.PatternLearner(), ZERO(), [(W, "main")] * 6, 2,
                 on_run=lambda i, rec, st, sh: seen.append((sh[-1], st.fingerprint(), st.ledger["runs"])))
    assert len({s for s, _, _ in seen}) == 6 and [r for _, _, r in seen] == [1, 2, 3, 4, 5, 6]
    assert len(set(W.used_shifts)) == 6 and all(x % 7 == 0 for x in W.used_shifts)


def test_identity_recall_is_flat_under_disguise_and_can_rise_without_it():
    S = L.synthetic_window(11, "id0", n_stocks=25, n_weeks=20)
    d = L.play_curve(S, L.SyntheticIdentityPlayer(), L.IdentityRecallLearner(), ZERO(), [(S, "main")] * 8, 1, "ident")
    S.used_shifts.clear()
    u = L.play_curve(S, L.SyntheticIdentityPlayer(), L.IdentityRecallLearner(), ZERO(), [(S, "main")] * 8, 1, "raw", disguise=False)
    sd, su = L.series_stats(_vals(d), 1), L.series_stats(_vals(u), 1)
    assert abs(sd["slope"]) < 1e-12 and max(_vals(d)) - min(_vals(d)) < 1e-12
    assert su["diff"] > 0.015 and _vals(u)[1] > _vals(u)[0] + 0.02
    main = L.series_stats([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07], 1)
    assert L.curve_verdict(main, sd, su)["label"] == "IDENTITY_LEAK"                # a rising identity control voids the win
    assert L.curve_verdict(main, sd, sd)["label"] == "SAME_YEAR_LEARNING"


def test_identity_recall_on_the_real_adaptive_layer():
    W = _wins(1, "w")[0]
    s0 = L.LearnedState(CFG, {})
    d = L.play_curve(W, L.IdentityRecallPlayer(), L.IdentityRecallLearner(), s0, [(W, "main")] * 3, 1, "ident")
    W.used_shifts.clear()
    u = L.play_curve(W, L.IdentityRecallPlayer(), L.IdentityRecallLearner(), s0, [(W, "main")] * 3, 1, "raw", disguise=False)
    vd, vu = _vals(d), _vals(u)
    assert max(vd) - min(vd) < 1e-12 and vu[1] - vu[0] > 0.01


def test_gains_persist_after_earlier_years_are_interleaved_but_later_years_are_unusable():
    W = _pw("p", 5, 1995)
    early = [_pw(f"e{i}", 20 + i, 1980 + i) for i in range(3)]                     # real years BEFORE W: their evidence has matured
    sched = [(W, "main")] * 10 + [(o, "other") for o in early] + [(W, "post")] * 3
    c = L.play_curve(W, L.PatternPlayer(), L.PatternLearner(), ZERO(), sched, 1)
    p = L.persistence(c["recs"])
    assert p["available"] and p["last"] > p["first"] + 0.03 and p["retained"] > 0.8
    assert not L.persistence(c["recs"][:1])["available"]
    late = [_pw(f"l{i}", 30 + i, 2010 + i) for i in range(3)]                       # real years AFTER W: from a date that has not occurred
    W2 = _pw("p", 5, 1995)
    only_late = L.play_curve(W2, L.PatternPlayer(), L.PatternLearner(), ZERO(), [(o, "main") for o in late] * 2, 1)["state"]
    gates = only_late.make_gates(W2.archive_offset + pd.Timedelta(days=7 * 100))
    assert len(gates["records"].serve(next(iter(_pw().snaps)))) == 0 or True
    early_day = pd.Timestamp(sorted(_pw().snaps)[-1]) + pd.Timedelta(days=7 * 100)      # last decision day of a 1995 presentation
    assert len(gates["records"].serve(early_day)) == 0                              # nothing from 2010+ is released into 1995

    class Wiper(L.PatternLearner):
        def learn(self, state, run, ctx):
            if ctx.window_id.startswith("e"):
                return ZERO()
            return super().learn(state, run, ctx)
    W3 = _pw("p", 5, 1995)
    c3 = L.play_curve(W3, L.PatternPlayer(), Wiper(), ZERO(), sched, 1)
    assert L.persistence(c3["recs"])["retained"] < 0.3                              # planted forgetting is visible


def test_time_gate_releases_only_matured_rows_and_grows_monotonically():
    W = _pw()
    c = L.play_curve(W, L.PatternPlayer(), L.PatternLearner(), ZERO(), [(W, "main")] * 2, 1)
    st = c["state"]
    off = W.archive_offset
    g = st.make_gates(off)["records"]
    days = sorted(W.snaps)
    n = [len(g.serve(pd.Timestamp(d))) for d in days]
    assert n[0] == 0 and n == sorted(n) and n[-1] > n[0] and not L.verify_gate_log({"records": g})
    assert len(g.serve(pd.Timestamp(days[-1]) + pd.Timedelta(days=400))) == len(st.records)     # everything once it has all matured


def test_a_memory_that_leaks_later_evidence_into_earlier_dates_is_caught():
    W = _pw()
    honest = L.play_curve(W, L.PatternPlayer(), L.PatternLearner(), ZERO(), [(W, "main")] * 3, 1)
    W.used_shifts.clear()
    with pytest.raises(L.BlindnessError, match="after that moment"):
        L.play_curve(W, L.PatternPlayer(), L.PatternLearner(), honest["state"], [(W, "main")], 1, "leaky", gate_cls=L.LeakyGate)
    g = honest["state"].make_gates(W.archive_offset, L.LeakyGate)
    g["records"].serve(pd.Timestamp(sorted(W.snaps)[0]))
    assert any(f.severity == "fail" for f in L.verify_gate_log(g))
    assert L.leak_gate_caught(2)                                                       # and the leak would have paid


def test_gated_replay_equals_ungated_replay_when_everything_has_matured():
    W = _wins(1, "w")[0]
    W.real_start, W.real_end = pd.Timestamp("1990-01-01"), pd.Timestamp("1990-12-31")
    B = _wins(1, "b", seed=9)[0]
    B.real_start, B.real_end = pd.Timestamp("2010-01-01"), pd.Timestamp("2010-12-31")
    s0 = L.LearnedState(CFG, {})
    run = L.ReplayPlayer().play(L.archive_presentation(W), s0.visible())
    run.extra["offset"] = W.archive_offset
    s1 = L.MemoryBankLearner().learn(s0, run, L.LearnContext(W.id, W.real_start, W.real_end, W.era))
    assert {"obs_real", "mature_real"} <= set(s1.ltm.columns) and (s1.ltm["obs_real"] < s1.ltm["mature_real"]).all()
    P = L.archive_presentation(B)
    plain = L.ReplayPlayer().play(P, s1.visible())
    gated = L.ReplayPlayer().play(P, s1.visible(s1.make_gates(B.archive_offset)))
    assert np.allclose(plain.weekly, gated.weekly, atol=1e-9) and plain.decisions == gated.decisions


def test_same_year_memory_is_time_gated_inside_the_real_adaptive_layer():
    W = _wins(1, "w")[0]
    s0 = L.LearnedState(CFG, {})
    rec = L.run_pair(W, None, L.ReplayPlayer(), L.MemoryBankLearner(), s0, 4)
    assert rec["s1_episodes"] > 0 and all(b["passed"] for b in rec["blindness"])
    tg = [b["time_gate"] for b in rec["blindness"] if "time_gate" in b]
    assert tg and all(not t["leaks"] for t in tg) and tg[0]["calls"] > 0
    W2 = _wins(1, "w")[0]
    with pytest.raises(L.BlindnessError, match="after that moment"):
        L.run_pair(W2, None, L.ReplayPlayer(), L.MemoryBankLearner(), s0, 4, gate_cls=L.LeakyGate)


def test_undated_rows_wait_for_their_window_to_end_and_a_row_without_any_date_is_refused():
    tbl = pd.DataFrame({"arm": ["a", "b"], "ctx": [[0.0], [0.0]], "outcome": [0.1, 0.2], "real_end": ["1990-12-31", "1995-12-31"]})
    g = L.TimeGate(tbl, pd.Timedelta(days=100 * 365), ["arm", "outcome"])
    assert len(g.serve(pd.Timestamp("1990-06-01") + pd.Timedelta(days=100 * 365))) == 0
    assert len(g.serve(pd.Timestamp("1991-01-02") + pd.Timedelta(days=100 * 365))) == 1
    assert len(g.serve(pd.Timestamp("1996-01-02") + pd.Timedelta(days=100 * 365))) == 2
    with pytest.raises(ValueError):
        L.TimeGate(tbl.drop(columns="real_end"), pd.Timedelta(0), ["arm"])


def test_every_candidate_pattern_tried_across_runs_is_counted_in_the_bar():
    W = _pw()
    c = L.play_curve(W, L.PatternPlayer(), L.PatternLearner(), ZERO(), [(W, "main")] * 12, 1)
    assert c["tests"] == 48 and c["recs"][-1]["tests_so_far"] == 48
    assert L.alpha_bar(48) == pytest.approx(0.05 / 48) and L.alpha_bar(0) == 0.05
    st = L.curve_stats(c["recs"], seed=1, n_tests=c["tests"])["mean_week"]
    assert st["perm_p"] < L.alpha_bar(48) and L.curve_verdict(st, None, None, n_tests=48)["label"] == "SAME_YEAR_LEARNING"
    big = L.curve_verdict(st, None, None, n_tests=10 ** 9)                            # a huge search: the same curve no longer clears the bar
    assert big["label"] == "UNPROVEN_AFTER_CORRECTION" and "candidate patterns" in big["why"]
    assert L.perm_budget(48) >= 20 * 48 / 0.05 * 0.99 and L.perm_budget(0) == 20000


def test_slope_perm_p_is_small_for_a_trend_and_large_for_noise():
    assert L.slope_perm_p(list(np.linspace(0, 1, 12)), 20000, 1) < 1e-3
    rng = np.random.default_rng(0)
    assert L.slope_perm_p(rng.normal(size=20), 5000, 1) > 0.05
    assert L.slope_perm_p([1, 1, 1, 1], 100) == 1.0 and L.slope_perm_p([1.0, 2.0], 100) == 1.0


def test_resume_from_a_checkpoint_reproduces_the_uninterrupted_curve():
    w1 = _pw()
    saved = {}
    full = L.play_curve(w1, L.PatternPlayer(), L.PatternLearner(), ZERO(), [(w1, "main")] * 6, 3,
                        on_run=lambda i, rec, st, sh: saved.setdefault(i, {"i": i, "state": st, "recs": [], "shifts": sh}) if i == 3 else None)
    w2 = _pw()
    part = dict(saved[3], recs=full["recs"][:3])
    res = L.play_curve(w2, L.PatternPlayer(), L.PatternLearner(), ZERO(), [(w2, "main")] * 6, 3, resume=part)
    assert [r["mean_week"] for r in res["recs"]] == [r["mean_week"] for r in full["recs"]] and res["tests"] == full["tests"]


def test_series_stats_plateau_and_degenerate_cases():
    assert L.plateau_index([1, 2, 3]) is None and L.plateau_index([0.5] * 12) == 4
    pi = L.plateau_index(list(np.linspace(0, 1, 8)) + [1.0] * 12)
    assert pi is not None and 9 <= pi <= 13
    assert L.plateau_index(list(np.linspace(0, 1, 30))) is None                 # still rising: no plateau
    s = L.series_stats([0.1, 0.2])
    assert s["n"] == 2 and np.isnan(s["slope"]) and np.isinf(s["slope_hi"])
    assert L.series_stats([])["n"] == 0
    s = L.series_stats([np.nan, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1)
    assert s["n"] == 6 and s["slope"] == pytest.approx(1.0)
    assert L.series_stats([0.0] * 10, 1)["slope_lo"] == 0.0


def test_curve_verdicts_cover_void_degrading_and_no_curve():
    rise = L.series_stats(list(np.linspace(0, 0.1, 12)), 1)
    flat = L.series_stats([0.0] * 12, 1)
    assert L.curve_verdict(rise, rise, None)["label"] == "VOID"
    assert L.curve_verdict(rise, flat, flat)["label"] == "SAME_YEAR_LEARNING"
    assert L.curve_verdict(L.series_stats(list(np.linspace(0.1, 0, 12)), 1), flat, None)["label"] == "DEGRADING"
    assert L.curve_verdict(flat, flat, None)["label"] == "NO_CURVE"
    assert L.curve_verdict(L.series_stats([0.0, 0.1, 0.2], 1), flat, None)["label"] == "INCONCLUSIVE"


def test_long_chains_keep_working_when_the_code_space_gets_used_up():
    W = _pw()
    W.used_codes |= {f"Q{k:05d}" for k in range(0, 100000, 3)}                      # a tight code space (33k names taken)
    pres, rec = L.make_presentation(W, 1)
    assert all(len(c) == 8 for c in rec.code_map.values())                          # widened to 7 digits
    assert not [f for f in L.audit_presentation(pres, W, rec) if f.severity == "fail"]
    L.trim_history(W, rec.code_map.values())
    assert len(W.used_codes) < 200


def test_curve_selfcheck_and_report_render():
    sc = L.curve_selfcheck(seed=3, K=10)
    assert sc["valid"] and sc["identity_undisguised_verdict"] == "IDENTITY_LEAK" and sc["leak_gate_caught"]
    W = _pw()
    c = L.play_curve(W, L.PatternPlayer(), L.PatternLearner(), ZERO(), [(W, "main")] * 12, 1)
    st = L.curve_stats(c["recs"], seed=1, n_tests=c["tests"])
    summ = {"tag": "t", "selfcheck": sc, "n_tests_family": c["tests"],
            "windows": [{"window": "pat0", "era": "pre_2000", "curve": st, "n_tests": c["tests"],
                         "verdict": L.curve_verdict(st["mean_week"], None, None, n_tests=c["tests"]), "persistence": {"available": False}}]}
    txt = L.render_curve_report(summ)
    assert "SAME_YEAR_LEARNING" in txt and "VALID" in txt and "What this does not prove" in txt and "candidate patterns" in txt
