"""Generalising learners (canon C54/C55/C56, task B24). Synthetic worlds only.

Planted worlds through the real learning-delta harness (learning_delta.run_pair):
  * a law that holds in every year (band / lesson) must be found and must TRANSFER to later years;
  * a world with nothing to learn (null), a law that flips sign with the year (flip) and a law that exists in some years only
    (year) must leave the learner idle - a learner that adopts there is chasing noise;
  * a learner shown a window that did not end before the target began must be refused (C56).
Plus the machinery: the ledger picks exactly what the system's own pick() picks, the patch a learned state applies is the rule the
ledger scored, the gate rejects spikes / sign flips / hold-out failures, the disk ledger is invalidated by an evaluator change."""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from engine import adaptive as A
from engine import learners as LR
from engine import learning_delta as L
from engine import policy

SMALL = dict(n_hist=9, n_pairs=3, n_stocks=60, n_weeks=20)
_CACHE = {}


def planted(world, name, **kw):
    """Cache: one run per (world, learner, kwargs) shared by the tests that read it."""
    key = (world, name, tuple(sorted((k, str(v)) for k, v in kw.items())))
    if key not in _CACHE:
        _CACHE[key] = LR.planted_pairs(world, name, **{**SMALL, **kw})
    return _CACHE[key]


def transfer(recs, metric):
    return L.aggregate(recs, 1)["metrics"][metric]["transfer"]


def changed(recs):
    return sum(r["s0_fp"] != r["s1_fp"] for r in recs)


# ------------------------------------------------------------------------------------------------ fidelity of the machinery
@pytest.fixture
def no_crypto(monkeypatch):
    monkeypatch.setattr(policy, "crypto_set", lambda: set())


CFGS = [dict(LR.PLANTED_BASE),
        {**LR.PLANTED_BASE, "pick": "hivol", "pool_q": 0.7, "k": 2},
        {**LR.PLANTED_BASE, "w_model": 0.5, "w_move": 0.5, "k": 3},
        {**LR.PLANTED_BASE, "vol_filter": True, "liq_q": 0.3, "k": 6}]


def test_ledger_selection_equals_the_systems_pick(no_crypto):
    """The names a Setting holds are exactly what engine.adaptive.pick chooses for the same cfg (what is tested is what trades)."""
    w = LR.planted_window(3, "x", 1990, "band", n_stocks=60, n_weeks=6)
    snap = next(iter(w.snaps.values()))
    for cfg in CFGS:
        st = LR.Setting.make(cfg)
        cols = {c: snap[c].to_numpy(float) for c in LR.PANEL_COLS}
        ctx = {c: float(snap[c].iloc[0]) for c in LR.CTX_COLS}
        mine = sorted(snap.index[LR.select_names(cols, ctx, st)])
        theirs = sorted(A.pick(snap, LR.full_cfg(cfg) | {"ew": None}, [], {}).index)
        assert mine == theirs, cfg


def test_patch_snapshot_applies_the_scored_rule(no_crypto):
    """A learned rule reaches the player as a veto: picking from the patched snapshot equals the ledger's pick under the rule."""
    w = LR.planted_window(4, "x", 1990, "band", n_stocks=80, n_weeks=6)
    snap = next(iter(w.snaps.values()))
    st = LR.Setting.make({**LR.PLANTED_BASE, "k": 5}).with_rule(("vol20", 0.6, 1.0))
    cfg, extra = st.apply(L.LearnedState({}, {}))
    assert extra["rules"] == [["vol20", 0.6, 1.0]]
    cols = {c: snap[c].to_numpy(float) for c in LR.PANEL_COLS}
    ctx = {c: float(snap[c].iloc[0]) for c in LR.CTX_COLS}
    ledger = sorted(snap.index[LR.select_names(cols, ctx, st)])
    patched = LR.patch_snapshot(snap, extra)
    player = sorted(A.pick(patched, LR.full_cfg(cfg) | {"ew": None}, [], {}).index)
    assert ledger == player
    vetoed = patched.index[patched["ev_red_flag"] > 0]
    assert len(vetoed) > 0 and set(player).isdisjoint(vetoed)
    assert LR.patch_snapshot(snap, {}) is snap                                  # no learned state: untouched


def test_state_roundtrip_and_blindness():
    """cfg overrides, rules, move score and regime map survive apply -> from_state, and the visible state passes the C55 audit."""
    st = (LR.Setting.make(LR.PLANTED_BASE).with_cfg(k=6, w_move=0.5).with_rule(("max20", 0.0, 0.9)))
    st = dataclasses.replace(st, move_w=(("p_move", 0.6), ("vol20", 0.4)), regime=("m_vix", (16.0, 19.0), ((), (("vol20", 0.5, 1.0),), ())))
    s0 = L.LearnedState(dict(LR.PLANTED_BASE), {"half_life": 6})
    cfg, extra = st.apply(s0)
    back = LR.Setting.from_state(L.LearnedState(cfg, {}, None, extra))
    assert back == st and back.key() == st.key()
    win = LR.planted_window(1, "r01a", 1990, "null", n_stocks=20, n_weeks=6)
    findings = L.audit_visible(L.LearnedState(cfg, {}, None, extra).visible(), win)
    assert not [f for f in findings if f.severity == "fail"]
    assert all(k in LR.PATCH_KEYS for k in extra)


def test_regime_and_rule_helpers():
    reg = ("m_vix", (16.0, 20.0), ((("vol20", 0.0, 0.5),), (), (("vol20", 0.7, 1.0), ("max20", 0.0, 0.8))))
    assert [LR.regime_bucket(reg, {"m_vix": v}) for v in (10, 16, 18, 20, 30)] == [0, 1, 1, 2, 2]
    assert LR.regime_bucket(reg, {}) == -1 and LR.regime_bucket(reg, {"m_vix": float("nan")}) == -1 and LR.regime_bucket((), {"m_vix": 1}) == -1
    g = (("vol20", 0.3, 1.0), ("log_dv", 0.2, 1.0))
    eff = LR.rules_for_week(g, reg, {"m_vix": 25})
    assert ("vol20", 0.3, 1.0) not in eff and ("vol20", 0.7, 1.0) in eff and ("log_dv", 0.2, 1.0) in eff   # bucket rule replaces the global one
    assert LR.rules_for_week(g, reg, {}) == g
    cols = {c: np.arange(10.0) for c in LR.PANEL_COLS}
    cols["vol20"] = np.array([np.nan] + list(range(9)), float)
    keep = LR.rule_mask(cols, (("vol20", 0.0, 0.5),))
    assert not keep[0] and keep[1] and not keep[9]                                # NaN never survives a rule


# ------------------------------------------------------------------------------------------------ the gate
def test_gate_accepts_a_consistent_effect():
    g = LR.gate([0.02, 0.03, 0.01, 0.02, 0.04, 0.02, 0.03, 0.01, 0.02, 0.03, 0.02, 0.01])
    assert g.passed and g.n_pos == 12 and g.oos_mean > 0 and g.shrunk > 0


def test_gate_rejects_a_single_year_spike():
    d = [0.0, 0.001, -0.001, 0.0, 0.002, -0.002, 0.0, 0.001, -0.001, 0.0, 0.0, 0.5]
    g = LR.gate(d)
    assert not g.passed and g.t < 2.5


def test_gate_rejects_sign_flips_and_recent_decay():
    flip = [0.03, -0.03] * 8
    assert not LR.gate(flip).passed
    decay = [0.03] * 9 + [-0.02, -0.03, -0.02, -0.03]           # helped for years, then stopped: the hold-out third says no
    g = LR.gate(decay)
    assert not g.passed and "held-out" in g.why


def test_gate_needs_enough_years_and_handles_empty():
    assert not LR.gate([0.05] * 4).passed and "only 4 years" in LR.gate([0.05] * 4).why
    assert not LR.gate([]).passed and LR.gate([]).why == "no years"
    assert not LR.gate([np.nan, np.nan]).passed


def test_shrinkage_pulls_thin_evidence_toward_zero():
    thin, thick = LR.gate([0.02] * 8), LR.gate([0.02] * 40)
    assert thin.shrunk < thick.shrunk < 0.02 and thin.shrunk == pytest.approx(0.16 / 14)


# ------------------------------------------------------------------------------------------------ past-only (C56)
def test_history_is_past_only_and_the_guard_catches_a_leak():
    recs, lrn, book = planted("band", "band_pool", feats=("vol20",))
    for r in recs:
        assert not r["transfer_anachronistic"]
    ctx = L.LearnContext("w00", pd.Timestamp("1985-01-01"), pd.Timestamp("1985-12-31"), "x")
    ids = book.before(ctx.real_start)
    assert ids and all(book.end_of(i) < ctx.real_start for i in ids)
    assert "t00" not in ids and "w00" not in ids
    # plant the defect: the book claims a window that ends after the target began is 'before' it
    book.register("future", "1991-01-01", "1991-12-31", panel=book.panel(ids[0]))
    lrn.book.before = lambda start: ids + ["future"]
    run = L.Run(np.zeros(1), np.ones(2), [], None, {}, snaps={}, closes=pd.DataFrame())
    with pytest.raises(L.BlindnessError):
        lrn.history(ctx, run)


def test_learner_ids_all_end_before_the_target_started():
    recs, lrn, book = planted("band", "band_pool", feats=("vol20",))
    for r in recs:                                               # each pair's own record: the history it used excludes its transfer year
        assert r["transfer"].startswith("t")
    start = pd.Timestamp("1960-01-01") + pd.DateOffset(years=3 * (SMALL["n_hist"] + SMALL["n_pairs"] - 1))
    assert lrn.last_ids[-1] == f"w{SMALL['n_pairs'] - 1:02d}"
    assert all(book.end_of(i) < start for i in lrn.last_ids[:-1])


# ------------------------------------------------------------------------------------------------ planted worlds
def test_band_world_learned_and_transfers():
    recs, lrn, _ = planted("band", "band_pool", feats=("vol20",))
    assert changed(recs) == len(recs)
    tr = transfer(recs, "in_band")
    assert tr["mean"] > 0.03 and tr["lo"] >= 0.0
    assert any("vol20" in x for x in recs[-1]["lineage"])
    # the same-year gain is not larger than a real generaliser's: the law is shared by every window
    assert L.aggregate(recs, 1)["metrics"]["in_band"]["same"]["mean"] > 0


def test_lesson_world_learned_and_transfers():
    recs, lrn, _ = planted("lesson", "lessons", feats=("max20", "vol20"), min_years=8)
    assert changed(recs) == len(recs)
    tr = transfer(recs, "mean_week")
    assert tr["lo"] > 0
    assert L.aggregate(recs, 1)["headline"]["verdict"]["label"] == "TRANSFER_POSITIVE"
    assert any("max20" in x for x in recs[0]["lineage"])


def test_null_world_learner_idle():
    recs, _, _ = planted("null", "band_pool", feats=("vol20",))
    assert changed(recs) == 0
    assert transfer(recs, "mean_week")["mean"] == 0 and transfer(recs, "in_band")["mean"] == 0


def test_flip_world_learner_idle():
    recs, _, _ = planted("flip", "lessons", feats=("max20", "vol20"), min_years=8)
    assert changed(recs) == 0


def test_year_specific_law_is_not_adopted():
    recs, _, _ = planted("year", "band_pool", feats=("vol20",))
    assert changed(recs) == 0


def test_false_adoption_rate_on_null_worlds_is_low():
    """Calibration: over several independent null worlds a learner with ~10 candidates adopts (almost) never."""
    adopted = 0
    for seed in range(6):
        recs, _, _ = LR.planted_pairs("null", "band_pool", n_hist=9, n_pairs=1, seed=100 + seed, n_stocks=50, n_weeks=16, feats=("vol20",))
        adopted += changed(recs)
    assert adopted <= 1


def test_regime_map_targets_only_the_regime_where_the_law_holds():
    recs, lrn, _ = planted("regime", "regime_map", feats=("vol20",), n_hist=9)
    line = " ".join(recs[-1]["lineage"])
    assert "buckets on m_vix" in line
    if changed(recs):
        # in the low-vix bucket (bucket 0) vol is irrelevant: no rule may be adopted there
        assert "bucket 0: keep" not in line
        assert "bucket 2: keep vol20" in line


# ------------------------------------------------------------------------------------------------ the movement score
def _book_for(world, n=8, **kw):
    b = LR.Book()
    for i in range(n):
        b.register_window(LR.planted_window(50 + i, f"m{i:02d}", 1960 + 3 * i, world, n_stocks=60, n_weeks=16, **kw))
    return b, b.before("2100-01-01")


def test_move_weights_find_the_sign_stable_movement_feature():
    b, ids = _book_for("band")
    w, diag = LR.fit_move_weights(b, ids)
    d = dict(w)
    assert d.get("vol20", 0) > 0.8 and abs(sum(d.values()) - 1) < 1e-3          # weekly move scales with vol20, in every year
    assert diag["vol20"]["share_pos"] == 1.0


def test_move_weights_empty_when_nothing_moves_with_a_feature():
    b, ids = _book_for("null")
    w, diag = LR.fit_move_weights(b, ids)
    assert dict(w).get("vol20", 0) < 0.2
    assert LR.fit_move_weights(LR.Book(), []) == ((), {})


def test_mover_use_learner_adopts_the_movement_score_in_a_band_world():
    recs, lrn, _ = planted("band", "mover_use", n_hist=9, n_pairs=2)
    assert lrn.diag["vol20"]["share_pos"] == 1.0
    tr = transfer(recs, "in_band")
    assert tr["mean"] >= 0.0


# ------------------------------------------------------------------------------------------------ degenerate cases and the ledger cache
def test_no_history_leaves_the_state_alone():
    book = LR.Book()
    lrn = LR.make_learner("band_pool", book, feats=("vol20",))
    w = LR.planted_window(9, "only", 1990, "band", n_stocks=30, n_weeks=8)
    pres = L.archive_presentation(w)
    r = L.Run(np.zeros(4), np.ones(5), [], None, {}, snaps=pres.snaps, closes=pres.closes)
    s0 = L.LearnedState(dict(LR.PLANTED_BASE), {})
    s1 = lrn.learn(s0, r, L.LearnContext(w.id, w.real_start, w.real_end, w.era))
    assert s1.fingerprint() == s0.fingerprint() and "nothing passed the gate" in " ".join(s1.lineage)


def test_panel_with_no_full_week_and_empty_universe():
    w = LR.planted_window(2, "e", 1990, "null", n_stocks=20, n_weeks=4)
    p = LR.build_panel({}, w.closes, "e", w.real_start, w.real_end)
    assert p.n_weeks == 0 and LR.week_returns(p, LR.Setting.make(LR.PLANTED_BASE)).shape == (0,)
    snap = next(iter(w.snaps.values()))
    snap = snap.assign(ev_red_flag=1.0)                            # nothing eligible: the week is cash, not an exception
    cols = {c: snap[c].to_numpy(float) for c in LR.PANEL_COLS}
    assert len(LR.select_names(cols, {}, LR.Setting.make(LR.PLANTED_BASE))) == 0


def test_untradeable_names_are_flagged_and_cost_nothing():
    w = LR.planted_window(2, "e", 1990, "null", n_stocks=20, n_weeks=4)
    w.closes.iloc[:, 0] = np.nan                                   # the first name never trades
    p = LR.build_panel(w.snaps, w.closes, "e", w.real_start, w.real_end)
    assert p.n_untradeable >= p.n_weeks and all(x[0, LR.PANEL_COLS.index("ev_red_flag")] == 1 for x in p.X)


def test_ledger_cache_roundtrip_and_evaluator_invalidation(tmp_path):
    w = LR.planted_window(6, "c", 1990, "band", n_stocks=40, n_weeks=8)
    st = LR.Setting.make(LR.PLANTED_BASE)
    b1 = LR.Book(cache_dir=tmp_path)
    b1.register_window(w)
    r1 = b1.returns("c", st)
    b1.flush()
    b2 = LR.Book(cache_dir=tmp_path)
    b2.register_window(w)
    assert np.allclose(b2.returns("c", st), r1) and b2.n_evals == 0
    b3 = LR.Book(cache_dir=tmp_path)
    b3.evaluator = "changed-code"                                  # an edit to the evaluator must not reuse the old ledger
    b3.register_window(w)
    b3.returns("c", st)
    assert b3.n_evals == 1


def test_panel_file_roundtrip(tmp_path):
    w = LR.planted_window(6, "c", 1990, "band", n_stocks=30, n_weeks=6)
    p = LR.build_panel(w.snaps, w.closes, "c", w.real_start, w.real_end)
    LR.save_panel(p, tmp_path / "c.npz")
    q = LR.load_panel(tmp_path / "c.npz")
    assert q.n_weeks == p.n_weeks and np.array_equal(q.X[2], p.X[2]) and np.array_equal(q.fwd[3], p.fwd[3])
    assert q.real_end == p.real_end and np.allclose(q.ctx, p.ctx, equal_nan=True)


def test_candidate_libraries_stay_on_the_adapter_grids():
    for knob, vals in LR.GRIDS.items():
        if knob in A.STEPS:
            assert all(v in A.STEPS[knob] for v in vals), knob
    base = LR.Setting.make({**LR.PLANTED_BASE, "pick": "hivol", "pool_q": 0.7})
    cands = LR.cfg_candidates(base)
    assert base not in cands and len({c.key() for c in cands}) == len(cands)
    top = LR.cfg_candidates(LR.Setting.make(LR.PLANTED_BASE))
    assert not any(dict(c.cfg)["pool_q"] != 0.95 for c in top)                # pool_q is inert for pick='top': not searched
    assert all(r[2] - r[1] >= 0.14 for c in LR.pool_candidates(base) for r in c.rules)


def test_series_metric_and_unknown_metric():
    w = np.array([0.06, -0.07, 0.01, 0.2, -0.02])
    assert LR.series_metric(w, "in_band") == pytest.approx(0.4) and LR.series_metric(w, "mean_week") == pytest.approx(w.mean())
    assert np.isnan(LR.series_metric([], "in_band"))
    with pytest.raises(ValueError):
        LR.series_metric(w, "sharpe")
