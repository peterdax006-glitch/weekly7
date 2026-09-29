"""Learning-delta harness (canon C54/C55): a generalising learner shows a gain on the same year AND on a different year, a
memoriser shows a same-year gain and no transfer and is FLAGGED, a non-learner shows zero, and the blindness audit fails
when a window id, real date, real ticker, repeated shift or lineage field leaks. Synthetic worlds only."""
import json

import numpy as np
import pandas as pd
import pytest

from engine import learning_delta as L
from engine import livesim


def _wins(n, tag, seed=0):
    return [L.synthetic_window(L.derive_seed(seed, tag, i), f"{tag}{i:02d}", n_stocks=25, n_weeks=20) for i in range(n)]


def _recs(player, learner, n=6, seed=3, s0=None):
    s0 = s0 or L.LearnedState({}, {})
    W, B = _wins(n, "w"), _wins(n, "b", seed=9)
    return [L.run_pair(w, b, player, learner, s0, L.derive_seed(seed, "x"), log_fn=None) for w, b in zip(W, B)]


# ------------------------------------------------------------------ the three planted learners
def test_generalising_learner_gains_on_same_year_and_transfer():
    agg = L.aggregate(_recs(L.SyntheticPlayer(), L.SyntheticGeneraliser()), seed=1)
    m = agg["metrics"]["mean_week"]
    assert m["same"]["lo"] > 0 and m["transfer"]["lo"] > 0 and m["effect"]["lo"] > 0
    assert agg["verdict"]["label"] == "GENERALISING"
    assert abs(m["noise"]["mean"]) < 1e-12                 # order-preserving disguise moves nothing for an identity-free player


def test_memoriser_is_flagged_with_large_same_year_gain_and_zero_transfer():
    agg = L.aggregate(_recs(L.SyntheticMemoriserPlayer(), L.MemoriserLearner()), seed=1)
    m = agg["metrics"]["mean_week"]
    assert m["same"]["lo"] > 0.02
    assert abs(m["transfer"]["mean"]) < 1e-12
    assert m["gap"]["lo"] > 0
    assert agg["verdict"]["label"] == "MEMORISATION"


def test_non_learner_shows_zero_delta_everywhere():
    recs = _recs(L.SyntheticPlayer(), L.NullLearner())
    agg = L.aggregate(recs, seed=1)
    for m in L.METRICS:
        row = agg["metrics"][m]
        for k in ("same", "noise", "effect", "transfer"):
            v = row[k]["mean"]
            assert np.isnan(v) or abs(v) < 1e-12, (m, k, v)
    assert agg["verdict"]["label"] == "NO_EFFECT"


def test_harm_is_labelled_harmful():
    class Poisoner:
        name = "poison"

        def learn(self, state, run, ctx):
            return L.LearnedState({}, {}, None, {**state.extra, "w": -1.0})
    s0 = L.LearnedState({}, {}, None, {"w": 1.0})
    agg = L.aggregate(_recs(L.SyntheticPlayer(), Poisoner(), s0=s0), seed=1)
    assert agg["verdict"]["label"] == "HARMFUL"


def test_selfcheck_is_valid_and_catches_the_leak_probe():
    r = L.harness_selfcheck(seed=4, n_windows=6)
    assert r["valid"] and r["leak_detected"]
    assert r["memoriser"]["verdict"] == "MEMORISATION" and r["generaliser"]["verdict"] == "GENERALISING"


# ------------------------------------------------------------------ disguise and blindness audit
def test_disguise_is_fresh_order_preserving_and_faithful():
    w = _wins(1, "d")[0]
    p1, r1 = L.make_presentation(w, 1)
    p2, r2 = L.make_presentation(w, 2)
    for p, r in ((p1, r1), (p2, r2)):
        old = sorted(r.code_map)
        assert [r.code_map[c] for c in old] == sorted(r.code_map.values())      # alphabetical tie-breaks survive
        assert r.shift_days % 7 == 0 and r.shift_days != 0
        assert not set(r.code_map.values()) & set(w.closes.columns)
        assert min(pd.Timestamp(k) for k in p.snaps).year >= L.DISGUISE_MIN_YEAR
    assert not set(r1.code_map.values()) & set(r2.code_map.values()) and r1.shift_days != r2.shift_days
    inv = {v: k for k, v in r1.code_map.items()}
    back = p1.closes.rename(columns=inv)
    assert np.allclose(back.pct_change().to_numpy()[1:], w.closes.pct_change().to_numpy()[1:])
    k0 = next(iter(w.snaps))
    assert np.array_equal(p1.snaps[str((pd.Timestamp(k0) + pd.Timedelta(days=r1.shift_days)).date())]["sig"].to_numpy(), w.snaps[k0]["sig"].to_numpy())


def test_random_relabel_option_breaks_order_and_audit_requires_order():
    w = _wins(1, "d")[0]
    p, r = L.make_presentation(w, 5, order_preserving=False)
    new = [r.code_map[c] for c in sorted(r.code_map)]
    assert new != sorted(new)
    f = L.audit_presentation(p, w, r, kind="t")                  # order preservation required by default
    assert any(x.severity == "fail" and "order" in x.message for x in f)
    assert not any(x.severity == "fail" for x in L.audit_presentation(p, w, r, kind="t", order_preserving_required=False))


def test_audit_fails_when_a_window_id_leaks_in_attrs():
    w = _wins(1, "d")[0]
    p, r = L.make_presentation(w, 1)
    assert not [f for f in L.audit_presentation(p, w, r) if f.severity == "fail"]
    p.snaps[next(iter(p.snaps))].attrs["src"] = f"loaded from {w.id}"
    assert any(f.severity == "fail" for f in L.audit_presentation(p, w, r))


def test_audit_fails_on_real_date_real_ticker_and_repeated_shift():
    w = _wins(1, "d")[0]
    p, r = L.make_presentation(w, 1)
    real = p.closes.copy()
    real.index = real.index - pd.Timedelta(days=7 * 52 * 150)
    bad = L.Presentation(p.snaps, real, p.opens, p.bps, p.divs)
    assert any("real era" in f.message for f in L.audit_presentation(bad, w, r))
    cl = p.closes.copy()
    cl.columns = [sorted(w.real_tickers)[0]] + list(cl.columns[1:])
    assert any(f.severity == "fail" for f in L.audit_presentation(L.Presentation(p.snaps, cl, p.opens, p.bps, p.divs), w, r))
    p2, r2 = L.make_presentation(w, 2)
    w.used_shifts.append(r2.shift_days)                           # pretend the same shift was already shown
    assert any("repeats an earlier" in f.message for f in L.audit_presentation(p2, w, r2))


def test_audit_fails_when_disguised_prices_are_not_the_same_year():
    w = _wins(1, "d")[0]
    p, r = L.make_presentation(w, 1)
    c = p.closes.copy()
    c.iloc[10:, 0] = c.iloc[10:, 0] * 1.05
    f = L.audit_presentation(L.Presentation(p.snaps, c, p.opens, p.bps, p.divs), w, r)
    assert any("same year" in x.message for x in f)


def test_visible_state_is_scrubbed_and_audited():
    w = _wins(1, "d")[0]
    ltm = pd.DataFrame({"arm": ["('knob', 'k', 1, 2)"], "ctx": [[0.0] * 7], "outcome": [0.01], "real_end": ["1990-12-31"], "source": [w.id]})
    st = L.LearnedState({"k": 2}, {}, ltm)
    v = st.visible()
    assert tuple(v.ltm.columns) == L.VISIBLE_LTM_COLS
    assert not L.audit_visible(v, w)
    leaky = L.Visible({}, {}, ltm, {"window_id": w.id})
    assert any(f.severity == "fail" for f in L.audit_visible(leaky, w))
    yr = L.Visible({}, {"note": f"learned in {w.real_start.year}"}, None, {})
    assert any("real year" in f.message for f in L.audit_visible(yr, w))


def test_run_pair_raises_when_a_leak_is_planted():
    w, b = _wins(1, "w")[0], _wins(1, "b", seed=9)[0]
    s0 = L.LearnedState({}, {}, None, {"w": 1.0, "run_id": "r01a"})
    with pytest.raises(L.BlindnessError):
        L.run_pair(w, b, L.SyntheticPlayer(), L.NullLearner(), s0, 1)
    # a state that carries the window id in a value
    w2 = _wins(1, "w")[0]
    s1 = L.LearnedState({}, {}, None, {"w": 1.0, "note": w2.id})
    with pytest.raises(L.BlindnessError):
        L.run_pair(w2, b, L.SyntheticPlayer(), L.NullLearner(), s1, 1)


def test_player_only_receives_whitelisted_presentation_fields():
    assert tuple(f.name for f in L.dataclasses.fields(L.Presentation)) == L.PRESENTATION_FIELDS
    assert not {"id", "window", "seed", "real_start", "real_end", "shift", "tickers"} & set(L.PRESENTATION_FIELDS)


# ------------------------------------------------------------------ the real adaptive layer on a synthetic window
def test_real_replay_is_identity_free_under_order_preserving_disguise():
    w, b = _wins(1, "w")[0], _wins(1, "b", seed=9)[0]
    s0 = L.LearnedState({"k": 3, "exit_q": 0.8, "rebalance_weeks": 1, "brake": None, "max_per_sector": None, "w_model": 1.0,
                         "pick": "hivol", "pool_q": 0.5, "liq_q": 0.0, "vol_filter": False, "stress_thr": None, "stress_k": 2,
                         "trend_filter": None, "trend_gross": 0.0}, {})
    rec = L.run_pair(w, b, L.ReplayPlayer(), L.MemoryBankLearner(), s0, 5)
    # same data, new names, new dates, unchanged state: identical result (nothing reads identity)
    for m in L.METRICS:
        a, c = rec["run1"][m], rec["run2_noise"][m]
        assert (np.isnan(a) and np.isnan(c)) or abs(a - c) < 1e-12, m
    assert rec["run1"]["n_weeks"] > 10 and rec["s1_episodes"] > 0
    assert all(x["passed"] for x in rec["blindness"])


def test_memorising_replay_player_shows_same_year_gain_only():
    w, b = _wins(1, "w")[0], _wins(1, "b", seed=9)[0]
    cfg = {"k": 3, "exit_q": 0.8, "rebalance_weeks": 1, "brake": None, "max_per_sector": None, "w_model": 1.0, "pick": "top",
           "pool_q": 0.5, "liq_q": 0.0, "vol_filter": False, "stress_thr": None, "stress_k": 2, "trend_filter": None, "trend_gross": 0.0}
    rec = L.run_pair(w, b, L.MemorisingReplayPlayer(), L.MemoriserLearner(), L.LearnedState(cfg, {}), 2)
    same = rec["run2"]["mean_week"] - rec["run1"]["mean_week"]
    trans = rec["transfer_s1"]["mean_week"] - rec["transfer_s0"]["mean_week"]
    assert same > 0.01 and abs(trans) < 1e-12


# ------------------------------------------------------------------ metrics, statistics and degenerate cases
def test_pick_hits_and_metrics_on_known_path():
    idx = pd.bdate_range("2150-01-05", periods=12)
    closes = pd.DataFrame({"A": np.linspace(100, 130, 12), "B": np.linspace(100, 70, 12), "C": 100.0}, index=idx)
    dec = [(str(idx[0].date()), ["A", "B"]), (str(idx[5].date()), ["A", "C"])]
    d, m, n = L.pick_hits(dec, closes)
    assert n == 4 and d == pytest.approx(0.5)                     # A up twice, B down, C flat (not > 0)
    assert m == pytest.approx(0.75)                               # A moves >10% in both holds, B in the first, C never
    run = L.Run(np.array([0.06, -0.08, 0.02]), np.array([1000, 1060, 975, 995.0]), dec)
    mt = L.run_metrics(run, closes)
    assert mt["in_band"] == pytest.approx(2 / 3) and mt["pos_share"] == pytest.approx(2 / 3)
    assert mt["max_dd"] == pytest.approx(975 / 1060 - 1) and mt["year_return"] == pytest.approx(-0.005)


def test_empty_and_degenerate_inputs():
    empty = L.run_metrics(L.Run(np.array([]), np.array([]), []), pd.DataFrame())
    assert empty["mean_week"] == 0.0 and empty["max_dd"] == 0.0 and np.isnan(empty["dir_hit"])
    agg = L.aggregate([], seed=0)
    assert agg["n_pairs"] == 0 and agg["verdict"]["label"] == "INCONCLUSIVE"
    assert L.boot_ci([])[3] == 0 and np.isinf(L.boot_ci([0.3])[1])
    recs = _recs(L.SyntheticPlayer(), L.SyntheticGeneraliser(), n=2)
    assert L.aggregate(recs, 1)["verdict"]["label"] == "INCONCLUSIVE"          # two pairs never settle anything
    assert L.pick_hits([], pd.DataFrame(index=pd.bdate_range("2150-01-05", periods=3)))[2] == 0


def test_bootstrap_ci_covers_a_planted_effect_and_is_deterministic():
    rng = np.random.default_rng(0)
    x = rng.normal(0.03, 0.02, 40)
    a, b = L.boot_ci(x, seed=1), L.boot_ci(x, seed=1)
    assert a == b and a[1] < 0.03 < a[2] and a[1] > 0
    z = L.boot_ci(rng.normal(0.0, 0.02, 40), seed=1)
    assert z[1] < 0 < z[2]


def test_noise_warning_appears_when_disguise_changes_results():
    row = L.aggregate(_recs(L.SyntheticPlayer(), L.SyntheticGeneraliser()), 1)["metrics"]["mean_week"]
    v = L.verdict(row, 6, noise_max_abs=0.01)
    assert any("disguise noise" in n for n in v["notes"])


# ------------------------------------------------------------------ learners, pairs, pool, reseal
def test_memory_bank_learner_adds_episodes_and_keeps_lineage_hidden():
    ep = pd.DataFrame({"arm": ["('knob', 'k', 1, 2)"] * 3, "ctx": [[0.1] * 7] * 3, "outcome": [0.01, -0.02, 0.03], "date": ["2150-01-01"] * 3})
    run = L.Run(np.zeros(1), np.ones(2), [], ep)
    ctx = L.LearnContext("w1", pd.Timestamp("1990-01-01"), pd.Timestamp("1990-12-31"), "pre_2000")
    s = L.MemoryBankLearner().learn(L.LearnedState({"k": 1}, {}), run, ctx)
    assert s.n_episodes() == 3 and "date" not in s.ltm.columns and s.ltm["real_end"].iloc[0] == "1990-12-31"
    s2 = L.MemoryBankLearner().learn(s, run, ctx)
    assert s2.n_episodes() == 6 and s.n_episodes() == 3 and s2.fingerprint() != s.fingerprint()
    assert L.MemoryBankLearner().learn(L.LearnedState({}, {}), L.Run(np.zeros(1), np.ones(2), [], None), ctx).n_episodes() == 0
    assert tuple(s2.visible().ltm.columns) == L.VISIBLE_LTM_COLS


def test_causal_bank_drops_windows_that_end_on_or_after_start():
    bank = pd.DataFrame({"arm": ["a", "b"], "ctx": [[0.0], [0.0]], "outcome": [0.1, 0.2], "real_end": ["1980-06-30", "1990-06-30"]})
    assert len(L.causal_bank(bank, "1985-01-01")) == 1 and L.causal_bank(bank, "1970-01-01") is None
    assert L.causal_bank(None, "1985-01-01") is None


def test_choose_pairs_respects_gap_uniqueness_and_prefers_later_transfer():
    pool = [{"id": f"x{y}", "real_start": pd.Timestamp(f"{y}-01-01"), "real_end": pd.Timestamp(f"{y}-12-31")} for y in range(1970, 2020, 2)]
    a = L.choose_pairs(pool, 8, seed=3)
    assert a == L.choose_pairs(pool, 8, seed=3) and len(a) == 8
    assert len({w["id"] for w, _ in a}) == 8 and len({b["id"] for _, b in a}) == 8
    assert all(abs(w["real_start"].year - b["real_start"].year) >= 3 for w, b in a)
    assert L.choose_pairs(pool[:1], 3, seed=0) == []


def test_wait_for_ram_gives_up_and_returns_when_free():
    ticks = []
    assert L.wait_for_ram(2.5, poll_s=60, give_up_s=120, avail=lambda: 1.0, sleep=ticks.append) is False and ticks == [60, 60]
    assert L.wait_for_ram(2.5, avail=lambda: 9.0, sleep=ticks.append) is True


def test_revealed_pool_never_lists_an_unrevealed_window(tmp_path):
    (tmp_path / "cycles.json").write_text(json.dumps({"cycles": [{"run_id": "r01a", "revealed_year": 2016}, {"run_id": "r01b"}]}))
    (tmp_path / "loop2.json").write_text(json.dumps({"windows": [{"run_id": "w01a", "revealed": "Sep 1971 - Aug 1972"},
                                                                  {"run_id": "w01b", "revealed": None}, {"run_id": "w02a"}]}))
    pool = L.revealed_pool(tmp_path)
    assert [p["id"] for p in pool] == ["r01a", "w01a"]
    w = pool[1]
    assert w["real_start"] == pd.Timestamp("1971-09-01") and w["real_end"] == pd.Timestamp("1972-08-31")
    assert L.revealed_pool(tmp_path / "missing") == []


def test_reseal_refuses_unrevealed_never_overwrites_and_shifts(tmp_path, monkeypatch):
    monkeypatch.setattr(livesim, "DIR", tmp_path)
    (tmp_path / "sealed_r01a.json").write_text(json.dumps({"year": 2016, "shift_days": 63714}))
    with pytest.raises(PermissionError):
        livesim.reseal_window("r01a", "x1", 1, revealed=False)
    rec = livesim.reseal_window("r01a", "x1", 1, revealed=True)
    assert rec["start"] == "2016-01-01" and rec["shift_days"] % 7 == 0 and abs(rec["shift_days"] - 63714) >= 56
    from engine import blind_gates as BG
    assert BG.seal_digest(rec) == rec["digest"] and not [f for f in BG.check_seal(rec) if f.severity == "fail"]
    with pytest.raises(FileExistsError):
        livesim.reseal_window("r01a", "x1", 2, revealed=True)
    assert livesim.reseal_window("r01a", "x2", 1, revealed=True)["shift_days"] == rec["shift_days"]      # deterministic in the seed


def test_order_preserving_codes_keep_order_and_are_fresh():
    names = [f"S{k:04d}" for k in (5, 1, 9, 3)]
    a, b = livesim.order_preserving_codes(names, 1), livesim.order_preserving_codes(names, 2)
    assert [a[n] for n in sorted(names)] == sorted(a.values()) and a != b and len(set(a.values())) == 4
    with pytest.raises(ValueError):
        livesim.order_preserving_codes(range(11), 1, width=1)


def test_render_report_names_the_verdict_and_the_void_state():
    recs = _recs(L.SyntheticPlayer(), L.SyntheticGeneraliser(), n=4)
    sc = L.harness_selfcheck(seed=2, n_windows=6)
    summ = {"tag": "t", "learner": "x", "basis_version": 1, "aggregate": L.aggregate(recs, 1), "selfcheck": sc, "memoriser_control": None,
            "blindness": {"n_audits": 3, "n_failed": 0}, "caveats": L.CAVEATS}
    txt = L.render_report(summ)
    assert "GENERALISING" in txt and "VALID" in txt and "Learning delta per metric" in txt
    sc["valid"] = False
    assert "INVALID" in L.render_report(summ)


# ------------------------------------------------------------------ permutation test, tables, luck floor
def test_signflip_p_separates_a_planted_effect_from_noise():
    rng = np.random.default_rng(1)
    assert L.signflip_p(rng.normal(0.05, 0.02, 10)) < 0.01                       # exact enumeration path
    assert L.signflip_p(rng.normal(0.0, 0.02, 30), seed=2) > 0.05                # Monte Carlo path
    assert L.signflip_p([0.0, 0.0, 0.0]) == 1.0 and np.isnan(L.signflip_p([]))
    assert L.signflip_p([0.1, 0.2, np.nan]) == L.signflip_p([0.1, 0.2])


def test_pair_table_and_by_bank_breakdown_cover_every_pair():
    recs = _recs(L.SyntheticPlayer(), L.SyntheticGeneraliser(), n=4)
    t = L.pair_table(recs)
    assert len(t) == 4 and {"run1.mean_week", "run2.mean_week", "transfer_s1.worst5"} <= set(t.columns)
    agg = L.aggregate(recs, 1)
    assert agg["by_bank"]["empty_bank"]["n"] == 4 and "with_bank" not in agg["by_bank"]
    assert agg["metrics"]["mean_week"]["same"]["p_signflip"] < 0.2
    assert L.pair_table([]).empty


def test_effect_below_the_tie_break_luck_floor_is_called_out():
    recs = _recs(L.SyntheticPlayer(), L.SyntheticGeneraliser(), n=4)
    for r in recs:                                                               # plant a big random-relabel swing, tiny effect
        r["run2_shuffle"]["mean_week"] = r["run1"]["mean_week"] + 0.5
        r["run2"]["mean_week"] = r["run1"]["mean_week"] + 1e-3
        r["run2_noise"]["mean_week"] = r["run1"]["mean_week"]
    v = L.aggregate(recs, 1)["verdict"]
    assert any("tie-break luck floor" in n for n in v["notes"])
