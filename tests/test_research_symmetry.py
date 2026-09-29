"""Tests for engine/research/symmetry.py (C66 section 12): symmetry, loss-risk bank on the shared loss_pipeline storage, trend downgrade, gate input.
Synthetic data only; each mechanism has a planted case, a null case and the empty case."""
import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from engine.learning.core import FirewallBreach, Lifecycle, DecisionEffect
from engine.research import frontier as F
from engine.research import symmetry as S

NOW = "2030-01-01"


@pytest.fixture(scope="module")
def sym():
    return S.synthetic_symmetry_frame(0, "planted", n_weeks=45, per_week=50)


@pytest.fixture(scope="module")
def sym_null():
    return S.synthetic_symmetry_frame(0, "null", n_weeks=45, per_week=50)


# ================================================================ symmetry
def test_symframe_requires_unfired_universe():
    f = pd.DataFrame({"pattern_id": "p", "date": pd.to_datetime(["2020-01-06"] * 3), "ticker": list("abc"), "call": [1, 1, -1],
                      "fwd": [0.06, -0.06, 0.0]})
    f["matured_at"] = f["date"] + pd.Timedelta(days=7)
    with pytest.raises(ValueError, match="only fired rows"):
        S.SymFrame(f)
    assert len(S.SymFrame(f, require_universe=False)) == 3


def test_symframe_rejects_bad_rows(sym):
    base = sym.frame.drop(columns=S.SymFrame._derived_cols()).iloc[:200].copy()
    b = base.copy()
    b["matured_at"] = b["date"]
    with pytest.raises(ValueError):
        S.SymFrame(b, require_universe=False)
    b = base.copy()
    b.loc[b.index[0], "call"] = 2
    with pytest.raises(ValueError):
        S.SymFrame(b, require_universe=False)


def test_confusion_arithmetic():
    c = S.confusion_of(np.array([1, 1, 0, 0, 1], bool), np.array([1, 0, 0, 1, 1], bool), S.Space.WINNER)
    assert (c.tp, c.fp, c.tn, c.fn) == (2, 1, 1, 1)
    assert c.precision == pytest.approx(2 / 3) and c.recall == pytest.approx(2 / 3) and c.n == 5
    assert math.isnan(S.confusion_of(np.zeros(4, bool), np.ones(4, bool), S.Space.LOSER).precision)


def test_winner_only_pattern_is_not_trusted(sym):
    lib = S.analyse_library(sym)
    w = lib.get("winners")
    assert w.trust.verdict == S.Trust.NOT_TRUSTED
    assert w.loser.recall == 0.0 and w.winner.precision > 0.3       # finds winners, finds no losers
    kinds = {s.kind for s in w.failing_slices()}
    assert S.CaseKind.LOW_LIQUIDITY_FAILURE in kinds and S.CaseKind.LARGE_LOSS in kinds


def test_noise_pattern_is_never_trusted(sym, sym_null):
    for frame in (sym, sym_null):
        assert S.analyse_library(frame).get("noise").trust.verdict != S.Trust.TRUSTED


def test_good_pattern_gets_a_verdict_and_more_than_winners_only(sym):
    g = S.analyse_library(sym).get("good")
    assert g.loser.recall > 0.2 and g.winner.recall > 0.2 and g.hit_rate > 0.9
    assert g.errors.long_adverse_ratio < 1


def test_context_never_recorded_is_untested_not_ok(sym):
    f = sym.frame.drop(columns=S.SymFrame._derived_cols() + ["external_event", "regime_transition", "mover_outcome"])
    sf = S.SymFrame(f, sym.cfg, require_universe=False)
    p = S.analyse_pattern(sf, "good")
    assert p.slice_of(S.CaseKind.EXTERNAL_EVENT).status == S.SliceStatus.UNTESTED
    assert p.slice_of(S.CaseKind.EPISODE_SPIKED).status == S.SliceStatus.UNTESTED
    assert p.trust.verdict != S.Trust.TRUSTED or not p.trust.untested


def test_too_few_calls_is_unknown():
    sf = S.synthetic_symmetry_frame(0, "planted", n_weeks=45, per_week=50)
    small = sf.frame[sf.frame["pattern_id"] == "good"].head(300).drop(columns=S.SymFrame._derived_cols())
    p = S.analyse_pattern(S.SymFrame(small, sf.cfg, require_universe=False), "good")
    assert p.trust.verdict == S.Trust.UNKNOWN


def test_missed_movers_by_reason():
    f = pd.DataFrame({"pattern_id": "p", "date": pd.to_datetime(["2020-01-06"] * 6), "ticker": list("abcdef"),
                      "call": [1, 0, 0, -1, 0, 0], "lean": [1, 1, -1, -1, 1, 0], "margin": [0.5, -0.05, -0.5, 0.3, -0.5, 0.0],
                      "fwd": [0.08, 0.09, -0.08, 0.08, 0.10, 0.0]})
    f["matured_at"] = f["date"] + pd.Timedelta(days=7)
    sf = S.SymFrame(f, S.SymmetryConfig(), require_universe=False)
    m = S.miss_breakdown(sf.frame)
    assert m.winners_total == 4 and m.winner_caught == 1
    assert m.winner_near_miss == 1 and m.winner_wrong_way == 1 and m.winner_silent == 1
    assert m.losers_total == 1 and m.loser_silent == 1


def test_orphans_and_union_recall(sym):
    o = S.orphan_movers(sym)
    assert 0 < o["orphan_share"] < 1
    u = S.union_recall(sym)
    assert u["cum_winner_recall"].is_monotonic_increasing and u["cum_loser_recall"].is_monotonic_increasing


def test_bootstrap_confusion_interval(sym):
    f = sym.of("good")
    ci = S.bootstrap_confusion(f, np.random.default_rng(0), 150)
    assert ci["win_precision"].lo < ci["win_precision"].value < ci["win_precision"].hi
    assert math.isnan(S.bootstrap_confusion(f.iloc[:0], np.random.default_rng(0))["win_recall"].value)


def test_episode_outcomes_find_planted_kind():
    sf = S.synthetic_symmetry_frame(0, "planted", n_weeks=45, per_week=50)
    rows = S.episode_outcomes(sf, "good")
    assert len(rows) == 5 and not any(r.q < 0.01 for r in rows)      # episodes are random in the synthetic world: no false discovery
    f = sf.frame.drop(columns=S.SymFrame._derived_cols()).copy()
    f.loc[f["mover_outcome"] == "spiked", "fwd"] = 0.08
    rows = S.episode_outcomes(S.SymFrame(f, sf.cfg), "good")
    assert next(r for r in rows if r.episode == "spiked").q < 0.001
    assert S.episode_outcomes(S.SymFrame(f.iloc[:0], sf.cfg, require_universe=False)) == []


def test_dose_response_and_context_symmetry(sym):
    d = S.dose_response(sym.of("good"), "long")
    assert d.bins == 4 or d.bins == 0
    cs = S.context_symmetry(sym, "winners")
    assert any(c.dimension == "liquidity" and c.value == "low" and c.verdict == S.SliceStatus.FAILS for c in cs)
    assert not any(c.verdict == S.SliceStatus.FAILS for c in S.context_symmetry(S.synthetic_symmetry_frame(0, "null", n_weeks=45, per_week=50), "good"))


def test_pattern_overlap_and_blindness(sym):
    ov = S.pattern_overlap(sym)
    assert len(ov) == 3 and all(0 <= (o.winner_jaccard if math.isfinite(o.winner_jaccard) else 0) <= 1 for o in ov)
    assert len(S.mover_blindness(sym)) >= 1


# ---------------------------------------------------------------- loss-risk bank
@pytest.fixture(scope="module")
def mined(sym):
    return S.attach_mean_loss(S.mine_loss_risks(sym, NOW, S.MiningConfig(per_pattern=True)), sym, NOW)


def test_mining_finds_planted_loss_contexts(mined):
    assert any(("liquidity", "low") in i.context and i.oos_confirmed for i in mined)          # the planted low-liquidity weakness
    assert any(i.kind == S.RiskKind.REGIME_SHIFT_RISK and i.oos_confirmed for i in mined)


def test_mining_finds_nothing_significant_in_null_world(sym_null):
    ledger = F.TestLedger()
    items = S.mine_loss_risks(sym_null, NOW, S.MiningConfig(q_max=0.01, per_pattern=False), ledger)
    assert [i for i in items if i.oos_confirmed and i.relative_risk > 2.0] == []


def test_mining_ignores_unmatured_rows(sym):
    early = sym.frame["matured_at"].sort_values().iloc[len(sym) // 10]
    assert S.mine_loss_risks(sym, early, S.MiningConfig()) == []


def test_bank_add_is_point_in_time_and_append_only(tmp_path, mined):
    bank = S.LossRiskBank(tmp_path / "b.jsonl")
    it = mined[0]
    with pytest.raises(FirewallBreach):
        bank.add(it, it.matured_at)                                # evidence not strictly older than now
    assert bank.add(it, NOW) == "added" and bank.add(it, NOW) == "stale"
    assert bank.items(it.matured_at) == []                         # invisible at its own maturity date
    assert len(bank.items(NOW)) == 1
    assert bank.verify_chain() and len(S.LossRiskBank(tmp_path / "b.jsonl")) == 1


def test_bank_file_tamper_fails_closed(tmp_path, mined):
    p = tmp_path / "b.jsonl"
    bank = S.LossRiskBank(p)
    bank.add(mined[0], NOW)
    p.write_text(p.read_text().replace('"n":', '"n" :', 1) if False else p.read_text().replace("0.", "9.", 1))
    with pytest.raises((FirewallBreach, ValueError, Exception)):
        S.LossRiskBank(p)


def test_item_validation_blocks_opportunity_effects(mined):
    it = mined[0]
    assert it.validate() == []
    bad = dataclasses.replace(it, effects=(DecisionEffect.RANKING,))
    assert any("not allowed" in e for e in bad.validate())
    assert dataclasses.replace(it, risk_id="OP123").validate()
    with pytest.raises(ValueError):
        S.LossRiskBank().add(bad, NOW)


def test_independence_audit_catches_shared_ids(mined):
    bank = S.LossRiskBank()
    bank.add(mined[0], NOW)
    assert S.independence_audit(bank, ["other"]) == []
    assert S.independence_audit(bank, [mined[0].risk_id])


def test_query_multiplier_only_shrinks_and_retire_stops_it(mined):
    bank = S.LossRiskBank()
    for it in mined:
        if it.oos_confirmed:
            bank.add(it, NOW)
    top = max((i for i in bank.items(NOW)), key=lambda i: i.weight())
    ctx = dict(top.context)
    m, ids = S.risk_multiplier(bank, NOW, ctx, top.pattern_id if top.pattern_id != "*" else None)
    assert 0.05 <= m < 1.0 and top.risk_id in ids
    assert S.risk_multiplier(bank, NOW, {"vol": "nonexistent"})[0] == 1.0
    for i in ids:
        bank.retire(i, "test", NOW)
    assert S.risk_multiplier(bank, NOW, ctx, top.pattern_id if top.pattern_id != "*" else None)[0] == 1.0
    assert bank.get(top.risk_id).lifecycle == Lifecycle.RETIRED      # retired, still readable


def test_trader_caution_is_identity_free(mined):
    bank = S.LossRiskBank()
    for it in mined[:10]:
        bank.add(it, NOW)
    out = S.trader_caution(bank, NOW, {"vol": "high"})
    assert set(out) == {"size_multiplier", "abstain", "risk_kinds", "items_matched"}
    with pytest.raises(Exception):
        S.trader_caution(bank, NOW, {"ticker": "AAPL 2019-03-04"})


def test_prune_and_priority_and_questions(mined):
    bank = S.LossRiskBank()
    for it in mined:
        bank.add(it, NOW)
    dead = S.prune_bank(bank, NOW)
    assert all(bank.get(d).lifecycle == Lifecycle.RETIRED for d in dead)
    qs = S.research_questions(bank, NOW, "2030-01-02")
    assert all(q.problem.value == "LOSS_AVOIDANCE" for q in qs)
    assert all(S.item_priority(i) >= 0 for i in mined)


def test_avoidance_tradeoff_and_coverage(sym, mined):
    it = dataclasses.replace(mined[0], pattern_id="winners", context=(("liquidity", "low"),), measure="large_loss")
    a = S.avoidance_tradeoff(sym, it, NOW)
    assert a.losses_avoided > 0 and a.net_return_avoided > 0
    bank = S.LossRiskBank()
    for i in mined:
        bank.add(i, NOW)
    cov = S.loss_bank_coverage(bank, sym, NOW)
    assert cov["large_losses"] > 0 and cov["warned_share"] > 0.3


def test_revalidation_degrades_when_risk_disappears(sym, mined):
    it = next(i for i in mined if i.oos_confirmed and i.pattern_id == "*")
    bank = S.LossRiskBank()
    bank.add(dataclasses.replace(it, matured_at="2015-01-01"), NOW)
    calm = sym.frame.drop(columns=S.SymFrame._derived_cols()).copy()
    calm["fwd"] = np.where(calm["fwd"] < -0.15, -0.02, calm["fwd"])
    calm["fwd"] = np.where(calm["fwd"].abs() >= 0.05, calm["fwd"] * 0.5, calm["fwd"])
    new = S.revalidate_item(bank, it.risk_id, S.SymFrame(calm, sym.cfg), NOW)
    assert new.lifecycle in (Lifecycle.DEGRADED, Lifecycle.RETIRED) or new.oos_confirmed is not True


def test_walk_forward_bank_reduces_large_losses(sym):
    ev = S.evaluate_bank_walk_forward(sym, NOW, S.MiningConfig(per_pattern=False, min_ctx_n=20, min_losses=4, q_max=0.2))
    assert ev.items_used > 0 and ev.scaled_calls > 0
    assert ev.big_loss_sum_scaled < ev.big_loss_sum_unscaled and ev.helps is True
    assert S.evaluate_bank_walk_forward(sym, "2000-01-01").items_used == 0


# ---------------------------------------------------------------- state, entry point, sweep, output
def test_step_end_to_end_and_null(sym, sym_null):
    st = S.SymmetryState(mc=S.MiningConfig(per_pattern=False))
    st.add(sym)
    r = S.step(st, NOW, code_hash="t")
    assert r.independence_problems == () and r.ledger_size > 0
    assert r.library.get("winners").trust.verdict == S.Trust.NOT_TRUSTED
    rec = r.to_matured_record(st.bank)
    with pytest.raises(FirewallBreach):
        rec.gate(r.provenance.learned_at)
    assert str(rec.gate("2031-01-01")["patterns"][0]["id"]).startswith("P")
    st2 = S.SymmetryState(mc=S.MiningConfig(per_pattern=False, q_max=0.01))
    st2.add(sym_null)
    r2 = S.step(st2, NOW, code_hash="t")
    assert not [i for i in st2.bank.items(NOW) if i.relative_risk > 2.0 and i.oos_confirmed]


def test_step_on_empty_state():
    st = S.SymmetryState()
    r = S.step(st, NOW, code_hash="t")
    assert r.library.patterns == () and r.added == () and len(st.bank) == 0


def test_public_summary_hides_identity(sym):
    lib = S.analyse_library(sym)
    txt = str(S.public_summary(lib))
    assert "good" not in txt and "T0" not in txt and "2016" not in txt


def test_symmetry_sweep_visits_least_covered_and_resumes(tmp_path, sym):
    tr, led = F.CoverageTracker(tmp_path / "c.json"), F.TestLedger(tmp_path / "l.jsonl")
    sw = S.SymmetrySweep(tr, led, S.LossRiskBank(tmp_path / "b.jsonl"), mc=S.MiningConfig(per_pattern=False, min_ctx_n=30))
    sw.add_source("fam", sym)

    class Cand:
        precursor_id, source_family = "pc", "gapdown"
        frame = sym.frame[sym.frame["pattern_id"] == "good"].drop(columns=S.SymFrame._derived_cols()).head(2500)
    assert sw.add_precursors([Cand()]) == 1
    steps = sw.run("2031-01-01", 3)
    assert len({s.unit for s in steps}) == 3
    assert F.CoverageTracker(tmp_path / "c.json").counts == tr.counts
    with pytest.raises(ValueError):
        sw.add_precursors([object()])


def test_render_functions_run(sym):
    lib = S.analyse_library(sym)
    txt = S.render_library(lib)
    assert "NOT VALIDATED" in txt and "winners" in txt
    bank = S.LossRiskBank()
    assert "0 items" in S.render_bank(bank, NOW)
    assert "LOSS-FIRST" in S.loss_first_report(lib, bank, NOW)
    assert S.compare_libraries(lib, lib) == []


# ---------------------------------------------------------------- one bank, one ledger
def test_bank_is_the_shared_loss_pipeline_bank(mined):
    from engine.research import loss_pipeline as LP
    from engine.research.interactions import TrialLedger
    bank = S.LossRiskBank()
    assert isinstance(bank, LP.LossRiskBank) and isinstance(F.TestLedger(), TrialLedger)
    it = mined[0]
    bank.add(it, NOW)
    e = bank.latest()[0]                                          # visible through the shared bank's own API
    assert e.entry_id == it.risk_id and e.kind == "context" and e.lift == pytest.approx(it.relative_risk)
    assert bank.risk_score({"anything": 5.0}, NOW)["score"] == 0.0   # loss_pipeline's condition scoring ignores context entries
    with pytest.raises(FirewallBreach):
        bank.assert_disjoint([it.risk_id])
    bank.retire(it.risk_id, "t", NOW)
    assert len(bank.history(it.risk_id)) == 2 and bank.history(it.risk_id)[-1].state.value == "RETIRED"


def test_ledger_counts_come_from_trial_ledger():
    L = F.TestLedger()
    L.log("a", "fam", 0.01, "2020-01-01", "2020-02-01")
    L.log("a", "fam", 0.5, "2020-01-01", "2020-02-01")
    assert L.m_total("fam") == 1 and len(L) == 1
    L.log_null("blk", "fam", 99, "2020-01-01", "2020-02-01")
    assert len(L) == 100 and L.to_dict()["seen"]["fam"]["a"] == 1


# ---------------------------------------------------------------- trend downgrade and gate input
def _drifting_world(worsen: bool):
    """A universe with one pattern that starts sound; when `worsen`, its large-loss rate climbs week by week."""
    rng = np.random.default_rng(5)
    n_weeks, per = 60, 40
    date = np.repeat(pd.date_range("2016-01-04", periods=n_weeks, freq="7D").to_numpy(), per)
    n = len(date)
    week_idx = np.repeat(np.arange(n_weeks), per)
    fwd = rng.normal(0, 0.05, n)
    call = np.where(rng.random(n) < 0.5, np.where(fwd > 0, 1, -1), 0)
    p_bad = 0.01 + (0.14 * week_idx / n_weeks if worsen else 0.0)
    bad = (call != 0) & (rng.random(n) < p_bad)
    fwd = np.where(bad, -call * rng.uniform(0.16, 0.3, n), fwd)
    f = pd.DataFrame({"pattern_id": "drift", "date": date, "ticker": [f"T{i}" for i in range(n)], "call": call, "fwd": fwd})
    f["matured_at"] = f["date"] + pd.Timedelta(days=7)
    return S.SymFrame(f)


def test_worsening_loss_side_is_detected_and_trust_downgraded():
    sf = _drifting_world(True)
    tr = S.loss_trend(sf, "drift")
    assert tr.verdict == "WORSENING" and tr.early_late_big_loss[1] > tr.early_late_big_loss[0]
    ps = S.analyse_pattern(sf, "drift")
    down = S.trend_adjusted_trust(ps, tr)
    assert down.verdict == S.Trust.NOT_TRUSTED and any("worsening" in x for x in down.failed)


def test_stable_loss_side_is_not_downgraded():
    sf = _drifting_world(False)
    tr = S.loss_trend(sf, "drift")
    assert tr.verdict in ("STABLE", "IMPROVING")
    ps = S.analyse_pattern(sf, "drift")
    assert S.trend_adjusted_trust(ps, tr).verdict == ps.trust.verdict


def test_trend_untested_on_thin_data_never_reads_as_stable():
    sf = _drifting_world(False)
    thin = S.SymFrame(sf.frame.drop(columns=S.SymFrame._derived_cols()).head(80), require_universe=False)
    assert S.loss_trend(thin, "drift").verdict == "UNTESTED"
    assert len(S.symmetry_timeline(thin, "drift")) < 4


def test_gate_input_verdicts(sym):
    from engine.research.core import GateVerdict
    lib = S.analyse_library(sym)
    gi = {g.pattern_id: g for g in S.gate_inputs(lib, sym)}
    assert gi["winners"].verdict in (GateVerdict.QUARANTINED, GateVerdict.FAILED) and gi["winners"].critical_failures
    assert gi["noise"].verdict != GateVerdict.PROMOTE
    assert all(g.as_dict()["verdict"] for g in gi.values())
    w = _drifting_world(True)
    lw = S.analyse_library(w)
    assert S.gate_input(lw.get("drift"), S.loss_trend(w, "drift")).verdict != GateVerdict.PROMOTE


# ---------------------------------------------------------------- regime-transition and external-event slices, planted
def _event_world(kind: str):
    """One decent pattern; `kind` plants harm only inside regime transitions (long calls) or external events (short calls), or nothing."""
    rng = np.random.default_rng(9)
    n_weeks, per = 50, 60
    date = np.repeat(pd.date_range("2017-01-02", periods=n_weeks, freq="7D").to_numpy(), per)
    n = len(date)
    fwd = rng.normal(0, 0.06, n)
    call = np.where(np.abs(fwd) > 0.03, np.where(rng.random(n) < 0.8, np.sign(fwd), -np.sign(fwd)), 0).astype(int)
    trans, ev = rng.random(n) < 0.15, rng.random(n) < 0.15
    if kind == "transition":
        hurt = trans & (call > 0)
    elif kind == "event":
        hurt = ev & (call < 0)
    else:
        hurt = np.zeros(n, bool)
    fwd = np.where(hurt & (rng.random(n) < 0.6), -call * rng.uniform(0.16, 0.3, n), fwd)
    f = pd.DataFrame({"pattern_id": "p", "date": date, "ticker": [f"T{i}" for i in range(n)], "call": call, "fwd": fwd,
                      "regime_transition": trans, "external_event": ev})
    f["matured_at"] = f["date"] + pd.Timedelta(days=7)
    return S.SymFrame(f)


def test_planted_transition_harm_is_caught_on_the_long_side():
    sf = _event_world("transition")
    d = S.event_slice_detail(sf.of("p"), S.CaseKind.REGIME_TRANSITION, sf.cfg)
    assert d.p_more_losses < 0.01 and d.adverse_direction in ("LONG", "BOTH") and d.lift_in_loss_rate > 2
    assert S.analyse_pattern(sf, "p").slice_of(S.CaseKind.REGIME_TRANSITION).status == S.SliceStatus.FAILS
    e = S.event_slice_detail(sf.of("p"), S.CaseKind.EXTERNAL_EVENT, sf.cfg)
    assert e.lift_in_loss_rate < 2 and e.p_more_losses > 0.001       # the un-planted slice shows no planted-size harm


def test_planted_event_harm_is_caught_on_the_short_side():
    sf = _event_world("event")
    d = S.event_slice_detail(sf.of("p"), S.CaseKind.EXTERNAL_EVENT, sf.cfg)
    assert d.p_more_losses < 0.01 and d.adverse_direction in ("SHORT", "BOTH")
    assert S.analyse_pattern(sf, "p").slice_of(S.CaseKind.EXTERNAL_EVENT).status == S.SliceStatus.FAILS


def test_null_event_world_finds_no_harm_and_report_is_complete():
    sf = _event_world("none")
    ps = S.analyse_pattern(sf, "p")
    assert ps.slice_of(S.CaseKind.REGIME_TRANSITION).status != S.SliceStatus.FAILS
    assert ps.slice_of(S.CaseKind.EXTERNAL_EVENT).status != S.SliceStatus.FAILS
    r = S.event_slice_report(sf)
    assert len(r) == 2 and set(r["slice"]) == {"REGIME_TRANSITION", "EXTERNAL_EVENT"}
    with pytest.raises(ValueError):
        S.event_slice_detail(sf.of("p"), S.CaseKind.NEAR_MISS, sf.cfg)
    empty = S.event_slice_detail(sf.of("p").iloc[:0], S.CaseKind.EXTERNAL_EVENT, sf.cfg)
    assert empty.inside_calls == 0 and empty.adverse_direction == "NONE"


def test_bank_health_untested_without_fresh_data_and_healthy_with_it(sym, mined):
    bank = S.LossRiskBank()
    old = dataclasses.replace(mined[0], matured_at="2016-03-01", context=(("liquidity", "low"),), pattern_id="winners", measure="large_loss")
    bank.add(old, NOW)
    h = S.bank_health(bank, sym, NOW)
    assert h and h[0].status in ("HEALTHY", "FADING") and h[0].fresh_exposed > 0
    assert S.bank_health(bank, sym, "2016-04-01")[0].status == "UNTESTED" or S.bank_health(bank, sym, "2016-04-01") == []
    calm = sym.frame.drop(columns=S.SymFrame._derived_cols()).copy()
    calm["fwd"] = calm["fwd"].clip(-0.1, 0.1)
    assert S.bank_health(bank, S.SymFrame(calm, sym.cfg), NOW)[0].status in ("GONE", "FADING")
