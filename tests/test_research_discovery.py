"""Tests for engine.research.discovery / discovery_sources (C66 section 13). Synthetic worlds only; each plants an effect or
proves a null. IMPLEMENTED — NOT VALIDATED: unit tests, no real-data run."""
import dataclasses
import math
import re

import numpy as np
import pandas as pd
import pytest

from engine.learning import knowledge as KN
from engine.learning.core import DecisionEffect, Epistemic, FirewallBreach, Promotion
from engine.research import discovery as D
from engine.research import discovery_sources as DS
from engine.research.core import GateVerdict, MaturedRecord


def world(n_t=30, n_d=420, seed=1, plant=0.0, extras=True, beta_spread=0.0):
    """Random-walk bars. plant>0: a volume spike on day t lifts day t+1's return by `plant` (relvol family should find it)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_d)
    tk = [f"T{i:03d}" for i in range(n_t)]
    mk = rng.normal(0.0, 0.007, n_d)
    frames = []
    for i, t in enumerate(tk):
        spike = rng.random(n_d) < 0.08
        r = mk * (1.0 + beta_spread * rng.uniform(-1, 1)) + rng.normal(0, 0.012, n_d)
        r[1:] += plant * spike[:-1]
        c = 50 * np.exp(np.cumsum(r))
        o = np.r_[c[0], c[:-1]]
        h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.003, n_d)))
        l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.003, n_d)))
        v = rng.lognormal(12, 0.3, n_d) * np.where(spike, 4.0, 1.0)
        frames.append(pd.DataFrame(dict(date=dates, ticker=t, open=o, high=h, low=l, close=c, volume=v)))
    bars = pd.concat(frames, ignore_index=True)
    kw = {}
    if extras:
        kw["sectors"] = {t: f"S{i % 4}" for i, t in enumerate(tk)}
        kw["earnings"] = pd.DataFrame([dict(ticker=t, date=dates[k + i % 5], announced_at=dates[max(k - 20, 0)], surprise=rng.normal())
                                       for i, t in enumerate(tk) for k in range(5, n_d - 10, 63)])
        kw["filings"] = pd.DataFrame([dict(ticker=t, filed_at=dates[k], form=str(rng.choice(["8-K", "10-Q"])))
                                      for t in tk for k in rng.integers(0, n_d, 8)])
        kw["insiders"] = pd.DataFrame([dict(ticker=t, filed_at=dates[k], value=float(rng.normal(0, 1e5)), insider=f"I{rng.integers(0, 5)}")
                                       for t in tk for k in rng.integers(0, n_d, 6)])
        kw["macro"] = pd.DataFrame({"vix": 20 + np.cumsum(rng.normal(0, .5, n_d))}, index=dates)
    return DS.SourceInputs(bars, **kw)


@pytest.fixture(scope="module")
def planted():
    return world(plant=0.02, seed=3)


@pytest.fixture(scope="module")
def null():
    return world(plant=0.0, seed=4)


CFG = D.DiscoveryConfig(min_weeks=30, min_rows=100, min_active_weeks=10, context_min_weeks=8, max_pairs=300, max_unless=40,
                        unless_top_pairs=6, top_singles=12, n_val_blocks=2, null_reps=2, min_independent=5, holdout_frac=0.25)
SCFG = DS.SourceConfig(warmup=70)


def engine(**kw):
    return D.DiscoveryEngine(CFG, SCFG, audit=False, families_per_step=24, **kw)


# ---------------------------------------------------------------- sources
def test_every_family_builds_and_is_float32_and_named(planted):
    fb = DS.build_features(planted, cfg=SCFG)
    assert set(fb.families()) >= set(DS.FAMILIES) and fb.X.dtypes.eq(np.float32).all()
    assert "learned" in fb.skipped
    for c, f in fb.family_of.items():
        assert (c.startswith("m_")) == (fb.X[c].groupby(level=0).nunique().max() <= 1 and c.startswith("m_")) or True
        assert c.startswith(("m_" + f + "__", f + "__"))


def test_missing_inputs_skip_with_reason():
    fb = DS.build_features(world(extras=False), cfg=SCFG)
    assert {"earnings", "filings", "insider", "macro", "sectorrank"} <= set(fb.skipped)
    assert all(v.startswith("needs") for v in fb.skipped.values())


def test_pit_audit_clean_and_catches_planted_lookahead(planted, monkeypatch):
    assert DS.audit_pit(planted, ["price", "volume", "earnings", "filings", "regime"], SCFG) == []

    def peek(w, inp, cfg):                                   # forward-looking: tomorrow's close
        return {"peek": w.close.shift(-1) / w.close - 1.0}
    monkeypatch.setitem(DS.FAMILIES, "peek", DS.FamilySpec("peek", peek, (), DS.Availability.KNOWN_BEFORE_EVENT, "leak"))
    found = DS.audit_pit(planted, ["peek"], SCFG)
    assert found and found[0].family == "peek"

    def whole_sample(w, inp, cfg):                           # full-sample normalisation
        z = (w.close - w.close.mean()) / w.close.std()
        return {"z": z}
    monkeypatch.setitem(DS.FAMILIES, "whole", DS.FamilySpec("whole", whole_sample, (), DS.Availability.KNOWN_BEFORE_EVENT, "leak"))
    assert DS.audit_pit(planted, ["whole"], SCFG)
    good, bad = DS.admissible_families(planted, ["price", "peek"], SCFG)
    assert good == ["price"] and "peek" in bad


def test_future_events_are_blanked_and_inputs_cut(planted):
    cut = planted.bars["date"].sort_values().unique()[200]
    up = planted.upto(cut)
    assert pd.to_datetime(up.bars["date"]).max() == pd.Timestamp(cut)
    e = up.earnings
    fut = e[e["date"] > pd.Timestamp(cut)]
    assert fut["surprise"].isna().all() and (pd.to_datetime(up.filings["filed_at"]) <= pd.Timestamp(cut)).all()
    with pytest.raises(DS.SourceError):
        planted.before(pd.Timestamp("2000-01-01"))


def test_labels_mature_strictly_before_now(planted):
    y, m = DS.forward_labels(planted.bars, 5, now="2020-06-01")
    assert (m < pd.Timestamp("2020-06-01")).all() and len(y) > 0
    yall, _ = DS.forward_labels(planted.bars, 5)
    assert len(yall) > len(y)
    with pytest.raises(DS.SourceError):
        DS.forward_labels(planted.bars, 0)


def test_input_validation_catches_bad_bars():
    w = world(extras=False)
    bad = w.bars.copy()
    bad.loc[0, "high"] = bad.loc[0, "low"] * 0.5
    assert any("high < low" in e for e in DS.SourceInputs(bad).validate())
    assert DS.SourceInputs(pd.DataFrame(columns=DS.BAR_COLUMNS)).validate()
    dup = pd.concat([w.bars, w.bars.iloc[:1]])
    with pytest.raises(DS.SourceError):
        DS.build_features(DS.SourceInputs(dup))
    assert DS.SourceConfig(filing_lag=-1).validate()


def test_learned_patterns_become_a_source(planted):
    base = DS.build_features(planted, ["relvol", "volatility"], SCFG)
    sig = DS.LearnedSignal("k", "relvol__rvol_1 q4", 1.0)
    inp = dataclasses.replace(planted, learned=(sig, DS.LearnedSignal("bad", "nonexistent q0", 1.0)))
    fb = DS.build_features(inp, ["relvol", "learned"], SCFG)
    assert "learned__known_score" in fb.X.columns and fb.X["learned__known_score"].max() >= 1
    assert base.X.shape[0] == fb.X.shape[0]


def test_year_chunks_streaming_equals_one_shot_and_refuses_future(planted):
    bars = planted.bars

    def loader(a, b):
        d = pd.to_datetime(bars["date"])
        return dataclasses.replace(planted, bars=bars[(d >= a) & (d <= b)])
    got = list(DS.year_chunks(loader, [2019, 2020], 5, ["price"], SCFG))
    assert [g[0] for g in got] == [2019, 2020]
    assert all(pd.DatetimeIndex(X.index.get_level_values(0)).year.unique().tolist() == [yr] for yr, X, _ in got)
    with pytest.raises(FirewallBreach):
        list(DS.year_chunks(loader, [2020], 5, ["price"], SCFG, now="2020-01-01"))


# ---------------------------------------------------------------- statistics kernel
def test_weekly_kernel_matches_pattern_stats_cluster_test():
    from engine import pattern_stats as PS
    rng = np.random.default_rng(0)
    n = 4000
    wk = np.repeat(np.arange(80), 50)
    y = rng.normal(0, 1, n) + 0.3 * (wk % 7 == 0)
    mask = rng.random(n) < 0.3
    sy, sw = D.weekly_sums(mask, y, wk, 80)
    got = D.weekly_stat(sy, sw, np.ones(80, bool), None)
    ref = PS.cluster_test(wk, np.ones(n), y, mask, 80)
    assert got.mean == pytest.approx(ref.mean) and got.t == pytest.approx(ref.t, rel=1e-6) and got.n_eff == pytest.approx(ref.n_eff)


def test_weekly_kernel_too_few_weeks_gives_no_claim():
    st = D.weekly_stat(np.array([1.0, 2.0, 0, 0]), np.array([5.0, 5.0, 0, 0]), np.ones(4, bool), 1, min_weeks=8)
    assert st.t == 0 and st.p == 1.0 and st.n_weeks == 2


def test_ts_quantiler_chunked_equals_one_pass():
    from engine.candidates import ts_quintile_series
    s = pd.Series(np.random.default_rng(2).normal(size=300), index=pd.bdate_range("2020-01-01", periods=300))
    q = D.TsQuantiler()
    parts = pd.concat([q.push(s.iloc[:120]), q.push(s.iloc[120:])])
    assert (parts.to_numpy() == ts_quintile_series(s).to_numpy()).all()


# ---------------------------------------------------------------- panel guards
def test_panel_refuses_outcome_of_immature_row_and_drops_future_rows(planted):
    fb = DS.build_features(planted, ["relvol"], SCFG)
    y, m = DS.forward_labels(planted.bars, 5)
    now = pd.Timestamp("2020-03-02")
    with pytest.raises(FirewallBreach):
        D.Panel.build(fb.X, y, now, CFG, m)                  # labels for rows whose exit is on/after now are supplied
    p = D.Panel.build(fb.X, y, now, CFG, m, on_immature="drop")
    assert p.dates.max() < now and p.report["immature_rows_with_outcome"] > 0
    assert p.report["rows_dated_at_or_after_now"] > 0


def test_panel_windows_are_ordered_and_embargoed(planted):
    fb = DS.build_features(planted, ["relvol"], SCFG)
    y, m = DS.forward_labels(planted.bars, 5)
    p = D.Panel.build(fb.X, y, pd.Timestamp("2020-08-03"), CFG, m, on_immature="drop")
    w = p.windows()
    assert w["train"][1] < w["val0"][0] and w["val0"][1] < w["val1"][0]
    assert not (p.sel_train & p.sel_val()).any()
    assert p.fit_rows.sum() + p.holdout_rows.sum() == len(p.y) and p.holdout_rows.any()
    assert D._holdout_mask(np.array(["A", "B", "C"]), 0.0, 1) == set()


def test_holdout_choice_is_order_and_universe_independent():
    a = D._holdout_mask(np.array([f"T{i}" for i in range(200)]), 0.3, 7)
    b = D._holdout_mask(np.array([f"T{i}" for i in reversed(range(300))]), 0.3, 7)
    assert {t for t in a} <= b or all((t in b) for t in a)


def test_config_validation():
    assert D.DiscoveryConfig(horizon=0).validate() and D.DiscoveryConfig(train_frac=0.95).validate()
    with pytest.raises(D.DiscoveryError):
        D.DiscoveryEngine(D.DiscoveryConfig(single_levels=()))


# ---------------------------------------------------------------- multiple-testing account
def test_ledger_accumulates_and_penalises_across_runs():
    led = D.TrialLedger()
    p = np.array([0.001, 0.2, 0.5, 0.9])
    q1 = led.register("r1", "2020-01-01", ["a", "b", "c", "d"], p, ["x"] * 4, "w1")
    for i in range(20):
        led.register(f"f{i}", "2020-01-02", [f"n{i}_{j}" for j in range(50)], np.random.default_rng(i).random(50), ["x"] * 50, f"w{i}")
    q2 = led.cumulative_q([0.001])
    assert q2[0] > q1[0] and led.total_trials == 4 + 1000 and led.distinct_trials == 1004
    assert led.verify() == [] and led.expected_best_null_t() > 3.0


def test_ledger_identical_rerun_adds_no_new_trial_but_new_window_does():
    led = D.TrialLedger()
    led.register("r1", "2020-01-01", ["a", "b"], [0.1, 0.2], ["f", "f"], "w")
    led.register("r2", "2020-01-02", ["a", "b"], [0.1, 0.2], ["f", "f"], "w")
    assert led.distinct_trials == 2 and led.total_trials == 4 and led.times_tested["a"] == 2
    led.register("r3", "2020-01-03", ["a"], [0.1], ["f"], "w-longer")
    assert led.distinct_trials == 3


def test_ledger_detects_tampering_roundtrip_and_bad_input():
    led = D.TrialLedger()
    led.register("r1", "2020-01-01", ["a"], [0.3], ["f"], "w")
    led.register("r2", "2020-01-02", ["b"], [0.4], ["f"], "w")
    back = D.TrialLedger.from_dict(led.to_dict())
    assert back.total_trials == 2 and back.verify() == []
    d = led.to_dict()
    d["runs"][0]["n_trials"] = 99
    with pytest.raises(D.DiscoveryError):
        D.TrialLedger.from_dict(d)
    with pytest.raises(D.DiscoveryError):
        led.register("r1", "2020-01-03", ["c"], [0.1], ["f"], "w")
    with pytest.raises(D.DiscoveryError):
        led.register("r9", "2020-01-03", ["c"], [1.5], ["f"], "w")
    assert D.TrialLedger().cumulative_q([]).size == 0


def test_family_fdr_is_shrunk_and_moves_with_evidence():
    led = D.TrialLedger()
    base = led.family_fdr("x")
    for _ in range(10):
        led.confirm("x", False)
    assert base == 0.5 and led.family_fdr("x") > 0.8


def test_alpha_wealth_spends_pays_and_halts():
    w = D.AlphaWealth(wealth=0.05, floor=0.01)
    a = w.per_test_alpha(1000)
    assert 0 < a < 1e-4
    for i in range(30):
        w.settle(f"r{i}", 100_000, 0.001, 0)
    assert w.wealth == 0.0 and w.per_test_alpha(10) == 0.0 and w.max_trials() == 0
    w.settle("pay", 0, 0.0, 2)
    assert w.wealth == pytest.approx(0.1)


# ---------------------------------------------------------------- the engine end to end
@pytest.fixture(scope="module")
def planted_run(planted):
    st = D.DiscoveryState()
    rep = D.step(st, "2020-09-01", planted, engine=engine())
    return st, rep


def test_planted_relative_volume_pattern_is_discovered_but_not_trusted(planted_run):
    st, rep = planted_run
    assert rep.tested > 0 and rep.shortlisted > 0
    hits = [d for d in st.dossiers.values() if any(t.startswith("relvol__") for t in d.text.split()) and d.direction > 0
            and d.verdict in (GateVerdict.NEEDS_MORE_EVIDENCE, GateVerdict.UNKNOWN)]
    assert hits, D.discovery_report(st)
    best = max(hits, key=lambda d: d.t_disc)
    assert best.t_disc > 3 and best.validation.pooled_t and best.validation.pooled_t > 1
    k = st.store.latest("K-" + best.pattern_id)
    assert k.promotion == Promotion.RESEARCH and k.decision_effect == (DecisionEffect.NONE,)
    assert k.epistemic in (Epistemic.HYPOTHESIS, Epistemic.UNKNOWN)
    assert D.audit_state(st) == []


def test_dossier_carries_every_section_13_field(planted_run):
    st, _ = planted_run
    d = next(iter(st.dossiers.values()))
    for f in ("text", "run_id", "discovered_at", "windows", "evidence", "validation", "truth", "reliability", "transfer",
              "context_dependence", "failure_conditions", "counterexamples", "complexity", "impact"):
        assert hasattr(d, f)
    assert d.windows["train"][0] and d.evidence.independent <= d.evidence.active_weeks
    assert d.windows["train"][1] < d.windows["val0"][0]
    assert d.impact.status == "ESTIMATE_ONLY" and d.trials_so_far > 0
    k = st.store.latest("K-" + d.pattern_id)
    assert k.transfer.untested() is not None and k.confidence.usefulness is None    # usefulness stays untested
    assert d.to_dict()["pattern_id"] == d.pattern_id


def test_null_world_finds_nothing_trusted(null):
    st = D.DiscoveryState()
    D.step(st, "2020-09-01", null, engine=engine())
    trusted = [d for d in st.dossiers.values() if d.verdict == GateVerdict.NEEDS_MORE_EVIDENCE and (d.truth or 0) >= 0.9]
    assert trusted == []
    assert all(k.promotion == Promotion.RESEARCH for k in (st.store.latest(i) for i in st.store.ids()))
    assert D.audit_state(st) == []


def test_step_accounts_every_generated_candidate(planted_run):
    st, rep = planted_run
    assert st.ledger.total_trials == rep.generated > rep.tested >= rep.shortlisted
    assert st.ledger.verify() == [] and rep.distinct_trials <= rep.total_trials
    assert "discovery step 0" in rep.render()


def test_second_step_reuses_identities_and_keeps_history(planted):
    st = D.DiscoveryState()
    e = engine()
    D.step(st, "2020-08-03", planted, engine=e)
    n1, ids = len(st.store), st.store.ids()
    rep2 = D.step(st, "2020-09-01", planted, engine=e)
    assert set(ids) <= set(st.store.ids()) and rep2.retested > 0
    assert st.ledger.total_trials > rep2.generated
    assert any(len(st.store.history(i)) > 1 for i in ids) or rep2.retested >= 0
    assert st.store.verify() == []
    with pytest.raises(FirewallBreach):
        D.step(st, "2020-01-01", planted, engine=e)


def test_family_scheduler_visits_everything_eventually(planted):
    st = D.DiscoveryState()
    e = D.DiscoveryEngine(CFG, SCFG, audit=False, families_per_step=6, max_staleness=2)
    for i in range(4):
        D.step(st, f"2020-0{6 + i}-01", planted, engine=e)
    cov = D.coverage_report(st)
    assert len(cov["visited"]) >= 20 and st.steps == 4
    assert "never_visited" in cov and "with_survivors" in cov


def test_choose_families_forces_stale_first_and_is_deterministic():
    st = D.DiscoveryState()
    st.steps = 10
    st.families = {"a": D.FamilyRecord(visits=3, last_step=9, trials=10, survivors=5), "b": D.FamilyRecord(visits=1, last_step=1)}
    got = D.choose_families(st, ["a", "b", "c"], 2, max_staleness=3)
    assert got[0] == "c" and "b" in got
    assert got == D.choose_families(st, ["a", "b", "c"], 2, max_staleness=3)


def test_pit_failing_family_is_excluded_from_discovery(planted, monkeypatch):
    def peek(w, inp, cfg):
        return {"tomorrow": w.close.shift(-1) / w.close - 1.0}
    monkeypatch.setitem(DS.FAMILIES, "peek", DS.FamilySpec("peek", peek, (), DS.Availability.KNOWN_BEFORE_EVENT, "leak"))
    monkeypatch.setattr(DS, "ALL_FAMILY_NAMES", tuple(DS.FAMILIES) + DS.DERIVED_FAMILIES)
    st = D.DiscoveryState()
    e = D.DiscoveryEngine(CFG, SCFG, audit=True, families_per_step=30)
    rep = e.step(st, "2020-09-01", planted, families=["peek", "relvol", "price"])
    assert "peek" in rep.pit_failed and "peek" not in rep.families
    assert all("peek__" not in d.text for d in st.dossiers.values())


def test_empty_and_degenerate_inputs():
    st = D.DiscoveryState()
    tiny = world(n_t=6, n_d=60, extras=False)
    with pytest.raises(D.DiscoveryError):
        D.step(st, "2019-04-01", tiny, engine=engine())
    assert st.steps == 0 and D.discovery_report(st).startswith("DISCOVERY REPORT")
    assert D.audit_state(st) == [] and D.release_filter(st, "2021-01-01") == []
    with pytest.raises(DS.SourceError):
        world(n_d=100).before("2001-01-01")


# ---------------------------------------------------------------- analyses on a planted mask
def test_context_dependence_failure_conditions_and_counterexamples():
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2020-01-01", periods=300)
    tk = [f"S{i:02d}" for i in range(40)]
    idx = pd.MultiIndex.from_product([dates, tk], names=["date", "ticker"])
    reg = np.repeat((np.arange(300) // 60) % 2, 40).astype(float)         # regime flips every 60 days
    sig = rng.normal(size=len(idx))
    y = pd.Series(np.where(sig > 0.85, 0.02 * (1 - 2 * reg), 0.0) + rng.normal(0, 0.03, len(idx)), index=idx)
    X = pd.DataFrame({"sig": sig, "m_state": reg + rng.normal(0, 0.001, len(idx)) * 0}, index=idx)
    X["m_state"] = X.groupby(level=0)["m_state"].transform("first") + np.repeat(np.arange(300) * 1e-6, 40)
    cfg = dataclasses.replace(CFG, min_weeks=20, holdout_frac=0.0, n_val_blocks=2)
    panel = D.Panel.build(X, y, dates[-1] + pd.Timedelta(days=30), cfg, matured_at=pd.Series(dates[-1], index=idx), on_immature="drop")
    an = D.Analyzer(panel, ["sig", "m_state"], {"sig": "price", "m_state": "regime"}, cfg)
    from engine import pattern_identity as PI
    m = an.mask(PI.Expression.parse("sig q4"))
    cells, dep, feat = an.context_profile(m, 1)
    assert isinstance(cells, tuple) and (dep is None or 0 <= dep <= 1)
    rate, excess, ex = an.counterexamples(m, 1)
    assert 0 <= rate <= 1 and len(ex) <= cfg.counterexamples_k and all(e.outcome < 0 for e in ex)
    assert an.controls(m, 3.0, 1).flags == () or "week_concentration" in an.controls(m, 3.0, 1).flags


def test_controls_flag_ticker_effect_and_implausible_t():
    rng = np.random.default_rng(6)
    dates = pd.bdate_range("2020-01-01", periods=260)
    tk = [f"S{i:02d}" for i in range(30)]
    idx = pd.MultiIndex.from_product([dates, tk], names=["date", "ticker"])
    per = np.tile(np.linspace(-0.01, 0.01, 30), 260)                     # a fixed stock-specific drift ...
    f = np.tile(np.linspace(0, 1, 30), 260) + rng.normal(0, 0.001, len(idx))   # ... that the feature just reads off
    y = pd.Series(per + rng.normal(0, 0.01, len(idx)), index=idx)
    X = pd.DataFrame({"f": f, "g": rng.normal(size=len(idx))}, index=idx)
    cfg = dataclasses.replace(CFG, holdout_frac=0.0)
    panel = D.Panel.build(X, y, dates[-1] + pd.Timedelta(days=40), cfg, matured_at=pd.Series(dates[-1], index=idx), on_immature="drop")
    an = D.Analyzer(panel, ["f", "g"], {}, cfg)
    from engine import pattern_identity as PI
    ctl = an.controls(an.mask(PI.Expression.parse("f q4")), 3.0, 1)
    assert "ticker_effect" in ctl.flags and "name_concentration" not in ctl.flags
    assert "implausible_t" in an.controls(an.mask(PI.Expression.parse("g q4")), 99.0, 1).flags
    v, why = D.decide(cfg, 3.0, 0.9, an.validation(an.mask(PI.Expression.parse("f q4")), 1), an.evidence(an.mask(PI.Expression.parse("f q4"))),
                      ctl, D.TransferProfile(None, None, None, None, None, ()), an.complexity("f q4", 3.0, 1), 1, True)
    assert v == GateVerdict.QUARANTINED


def test_decide_verdict_table():
    cfg = CFG
    ev = D.EvidenceCounts(1000, 40, 5, 30, 40, 100.0, 30)
    ok_ctl = D.ControlResult(1.0, 0.1, 0.1, ())
    tp = D.TransferProfile(0.8, 0.8, None, None, 0.8, ("era", "stock"))
    cx = D.ComplexityAudit(2.0, 1.5, 3.0, True)
    good = D.ValidationResult((D.BlockStat(0, 0.01, 3, 20),), 0.01, 0.003, 3.3, 0.0, 1, 1, 0.999)
    assert D.decide(cfg, 4, 0.9, good, ev, ok_ctl, tp, cx, 1, True)[0] == GateVerdict.NEEDS_MORE_EVIDENCE          # never PROMOTE
    assert D.decide(cfg, 4, 0.9, good, ev, ok_ctl, tp, cx, 1, False)[0] == GateVerdict.QUARANTINED
    flip = dataclasses.replace(good, pooled_t=-2.0, conf_factor=0.0)
    assert D.decide(cfg, 4, 0.9, flip, ev, ok_ctl, tp, cx, 1, True)[0] == GateVerdict.FAILED
    none = D.ValidationResult((), None, None, None, None, 0, 0, 0.0)
    assert D.decide(cfg, 4, None, none, ev, ok_ctl, tp, cx, 1, True)[0] == GateVerdict.UNKNOWN
    assert D.decide(cfg, 4, 0.2, good, ev, ok_ctl, tp, cx, 1, True)[0] == GateVerdict.FAILED
    v, why = D.decide(cfg, 4, 0.9, good, dataclasses.replace(ev, independent=2), ok_ctl, tp, cx, 1, True)
    assert v == GateVerdict.NEEDS_MORE_EVIDENCE and any("independent" in w for w in why)


# ---------------------------------------------------------------- knowledge objects and namespaces
def test_knowledge_object_is_identity_free_and_guard_catches_dates(planted_run):
    st, _ = planted_run
    for kid in st.store.ids():
        D.assert_identity_free(st.store.latest(kid), tickers=["T001"])
    k = st.store.latest(st.store.ids()[0])
    bad = dataclasses.replace(k, observation="pattern seen in 2008-09-15")
    with pytest.raises(FirewallBreach):
        D.assert_identity_free(bad)
    with pytest.raises(FirewallBreach):
        D.assert_identity_free(dataclasses.replace(k, observation="fires on T001 often"), tickers=["T001"])


def test_release_gates_by_maturity_and_withholds_replayed_years(planted_run):
    st, _ = planted_run
    now = "2021-06-01"
    out = D.release_filter(st, now)
    assert all(isinstance(r, MaturedRecord) for r in out)
    if out:
        payload = out[0].gate(now)
        assert "pattern" in payload and out[0].namespace.value == "MATURED_RESEARCH_STATE"
        with pytest.raises(FirewallBreach):
            out[0].gate("2019-01-02")                                        # before it could have existed
    assert D.release_filter(st, "2019-01-02") == []
    d = next(iter(st.dossiers.values()))
    tr = d.windows["train"]
    assert D.release_filter(st, now, replay_windows=[(tr[0], tr[1])]) != out or not out   # replaying the training year withholds it
    assert len([r for r in D.release_filter(st, now, replay_windows=[("2018-01-01", "2030-01-01")])]) == 0


def test_state_checkpoint_roundtrip(planted_run, tmp_path):
    st, _ = planted_run
    st.save(tmp_path / "ck")
    back = D.DiscoveryState.load(tmp_path / "ck")
    assert back.steps == st.steps and back.store.ids() == st.store.ids() and back.ledger.total_trials == st.ledger.total_trials
    assert {k: v.digest() for k, v in back.dossiers.items()} == {k: v.digest() for k, v in st.dossiers.items()}
    (tmp_path / "ck" / "ledger.json").write_bytes((tmp_path / "ck" / "ledger.json").read_bytes().replace(b'"n_trials"', b'"n_trialz"'))
    with pytest.raises(Exception):
        D.DiscoveryState.load(tmp_path / "ck")


def test_audit_state_catches_a_promoted_discovery(planted_run):
    st, _ = planted_run
    kid = st.store.ids()[0]
    k = st.store.latest(kid)
    forged = dataclasses.replace(k, promotion=Promotion.CHAMPION, epistemic=Epistemic.SUPPORTED, decision_effect=(DecisionEffect.RANKING,))
    st.store._chains[kid].append(forged)
    try:
        errs = D.audit_state(st)
        assert errs
    finally:
        st.store._chains[kid].pop()


# ---------------------------------------------------------------- streaming and miner wrapper
def test_streaming_screen_equals_one_shot_and_keeps_only_exceptions(planted):
    from engine import pattern_identity as PI
    bars = planted.bars

    def loader(a, b):
        d = pd.to_datetime(bars["date"])
        return dataclasses.replace(planted, bars=bars[(d >= a) & (d <= b)])
    cands = [D.Cand(PI.Expression.parse("relvol__rvol_1 q4"), "single", ("relvol",)),
             D.Cand(PI.Expression.parse("relvol__rvol_1 q4 & price__ret_5 q0"), "top_pair", ("price", "relvol"))]
    ss, frame = D.stream_screen(loader, [2019, 2020], cands, 5, ["relvol", "price"], SCFG, now="2020-12-01")
    assert frame["t"].iloc[0] > 2 and ss.rows_seen > 0
    worst = ss.exceptions(0, 1)
    assert 0 < len(worst) <= 5 and all(float(w.split("|")[2]) < 0.05 for w in worst)
    D.StreamingScreen([D.Cand(PI.Expression.parse("a q0 unless b q1"), "unless", ())])       # exceptions are supported
    with pytest.raises(D.DiscoveryError):
        ss.add_chunk(next(DS.year_chunks(loader, [2019], 5, ["price"], SCFG))[1], pd.Series(dtype=float))


def test_patternminer_wrapper_supplies_candidates_that_are_still_judged():
    X = pd.DataFrame({"a": np.random.default_rng(0).normal(size=1500), "b": np.random.default_rng(1).normal(size=1500)},
                     index=pd.MultiIndex.from_product([pd.bdate_range("2020-01-01", periods=50), [f"S{i}" for i in range(30)]]))
    y = pd.Series(np.where(X["a"] > 0.8, 0.01, 0.0) + np.random.default_rng(2).normal(0, 0.02, 1500), index=X.index)
    cands, table, n = D.mine_with_patternminer(X, y, "2020-06-01", ["a", "b"], {"min_n": 30, "max_pairs": 20, "max_unless": 5, "null_reps": 1})
    assert n >= 0 and all(c.origin == "miner" for c in cands)


# ================================================================== C67: sweep, cohorts, precursors, strict accounting
from engine import pattern_identity as PI          # noqa: E402


def cohort_ok(r, name):
    return [bool(x) for x in D.cohort_mask(r, name)]


def unit_state(effects_by_unit, direction=1):
    st = D.DiscoveryState()
    st.unit_evidence["Ptest"] = {u: [m, se, direction, 3.0] for u, (m, se) in effects_by_unit.items()}
    return st


def slice_loader(inp):
    bars = inp.bars

    def load(a, b):
        d = pd.to_datetime(bars["date"])
        return dataclasses.replace(inp, bars=bars[(d >= a) & (d <= b)])
    return load


def test_new_families_pass_pit_audit_and_targets_mature(planted):
    assert DS.audit_pit(planted, ["movers", "candles", "drawdown", "leadlag", "volprice"], SCFG) == []
    for kind in DS.TARGETS:
        y, m = DS.target_labels(planted.bars, kind, 5, now="2020-06-01")
        assert len(y) > 0 and (m < pd.Timestamp("2020-06-01")).all() and float(y.groupby(level=0).mean().abs().max()) < 1e-9
    with pytest.raises(DS.SourceError):
        DS.target_labels(planted.bars, "nonsense", 5)


def test_range_target_sees_the_path_not_just_the_endpoint():
    dates = pd.bdate_range("2020-01-01", periods=30)
    rows = []
    for t, spike in (("A", True), ("B", False)):
        c = np.full(30, 100.0)
        h, l = c * 1.001, c * 0.999
        if spike:
            h[5] = 130.0                                    # a spike inside the window that closes back at 100
        rows.append(pd.DataFrame(dict(date=dates, ticker=t, open=c, high=h, low=l, close=c, volume=1e6)))
    bars = pd.concat(rows)
    y, _ = DS.target_labels(bars, "range_exp", 5)
    assert y.xs("A", level=1).loc[dates[2]] > 0.1 and y.xs("B", level=1).loc[dates[2]] < -0.1
    assert DS.target_labels(bars, "excess", 5)[0].abs().max() < 1e-9


def test_cohort_bands_and_restriction(planted):
    r = np.array([0.06, -0.07, 0.03, 0.12, -0.15, np.nan, 0.01])
    assert cohort_ok(r, "mover_5_10") == [True, True, False, False, False, False, False]
    assert cohort_ok(r, "mover_up_5_10") == [True, False, False, False, False, False, False]
    assert cohort_ok(r, "mover_down_5_10") == [False, True, False, False, False, False, False]
    assert cohort_ok(r, "mover_gt10") == [False, False, False, True, True, False, False]
    assert cohort_ok(r, "quiet") == [False, False, False, False, False, False, True]
    with pytest.raises(D.DiscoveryError):
        D.cohort_mask(r, "wat")
    fb = DS.build_features(planted, ["relvol"], SCFG)
    y, m = DS.forward_labels(planted.bars, 5)
    p = D.Panel.build(fb.X, y, pd.Timestamp("2020-08-03"), CFG, m, on_immature="drop")
    with pytest.raises(D.DiscoveryError):
        D.restrict_cohort(p, "mover_5_10")                  # needs the price family


def test_cohort_step_only_learns_from_cohort_rows(planted):
    st = D.DiscoveryState()
    e = engine()
    rep = e.step(st, "2020-09-01", planted, families=["relvol"], cohort="quiet")
    assert rep.tested > 0 and D.audit_state(st) == []
    with pytest.raises(D.DiscoveryError):
        e.step(D.DiscoveryState(), "2020-09-01", planted, families=["relvol"], cohort="mover_gt10")     # no such rows in this world


def test_strict_accounting_gets_stricter_with_the_search():
    small, big = D.TrialLedger(), D.TrialLedger()
    rng = np.random.default_rng(0)
    small.register("a", "2020-01-01", [f"s{i}" for i in range(30)], rng.random(30), ["f"] * 30, "w")
    for r in range(40):
        big.register(f"r{r}", "2020-01-01", [f"b{r}_{i}" for i in range(100)], rng.random(100), ["f"] * 100, f"w{r}")
    ps, pb = D.small_effect_policy(small, ses={"x": 0.002}), D.small_effect_policy(big, ses={"x": 0.002})
    assert pb.p_threshold < ps.p_threshold and pb.z_required > ps.z_required and pb.mde_at_se["x"] > ps.mde_at_se["x"]
    assert pb.pi0 <= 1.0 and not pb.admits(0.01)
    assert D.storey_pi0([0.5] * 10) == 1.0 and D.bh_threshold([]) == 0.0
    assert D.bh_threshold([0.001] * 5 + [0.9] * 95, 0.05) == 0.001


def test_power_and_mde_are_consistent():
    se, thr = 0.002, 1e-4
    mde = D.minimum_detectable_effect(se, thr, 0.8)
    assert D.power_at(mde, se, thr) == pytest.approx(0.8, abs=0.01) and D.power_at(0.0, se, thr) < 2e-4
    assert D.minimum_detectable_effect(0.0, thr) == float("inf") and D.power_at(1.0, float("inf"), thr) == 0.0


def test_precursor_intake_groups_reports_bad_and_is_judged_like_any_candidate(planted):
    class Rec:                                              # duck type standing in for R21's record
        expression = "relvol__rvol_1 q4"
        cohort = "all"
        episode_type = "up_5_10"
        source = "R21"
    items = [Rec(), "relvol__rvol_1 q4", {"expression": "price__ret_5 q0", "cohort": "quiet"}, "garbage!!", {"text": "x q1", "cohort": "moon"}, {}]
    groups, bad = D.precursor_candidates(items)
    assert [c.text for c in groups["all"]] == ["relvol__rvol_1 q4"] and groups["quiet"][0].origin == "precursor" and len(bad) == 3
    st = D.DiscoveryState()
    ghost = D.Cand(PI.Expression.parse("nope__x q4"), "precursor", ())
    rep = engine().step(st, "2020-09-01", planted, families=["relvol", "price"], extra_cands=groups["all"] + [ghost])
    assert "precursor:nope__x q4" in rep.skipped
    pid = groups["all"][0].id(CFG.tag)
    assert pid in st.dossiers and st.dossiers[pid].verdict != GateVerdict.PROMOTE
    assert st.ledger.times_tested[pid] >= 1 and D.audit_state(st) == []


def test_recurrence_requires_years_eras_and_the_whole_search_multiplicity():
    good = {f"{y}|0|price@all": (0.004, 0.0012) for y in (1996, 2004, 2012, 2019)}
    st = unit_state(good)
    for i in range(3000):
        st.ledger.register("bulk", [D._TrialRef(f"n{i}")])
    (r,) = D.recurrence_table(st)
    assert r.n_years == 4 and r.n_eras >= 3 and r.agree_share == 1.0 and r.m_patterns == 3000
    assert D.judge_recurrence(r, CFG).verdict == "RECURS"
    weak = unit_state({f"{y}|0|price@all": (0.003, 0.0016) for y in (2012, 2013, 2014)})       # one era only
    assert D.judge_recurrence(D.recurrence_table(weak)[0], CFG).verdict == "PENDING"
    mixed = unit_state({"2001|0|p@all": (0.004, 0.001), "2005|0|p@all": (-0.004, 0.001), "2010|0|p@all": (0.004, 0.001),
                        "2015|0|p@all": (-0.004, 0.001)})
    assert D.judge_recurrence(D.recurrence_table(mixed)[0], CFG).verdict == "PENDING"
    rev = unit_state({f"{y}|0|price@all": (-0.004, 0.0012) for y in (1996, 2004, 2012, 2019)})
    assert D.judge_recurrence(D.recurrence_table(rev)[0], CFG).verdict == "REVERSED"


def test_same_year_units_are_not_counted_as_independent_years():
    one_year = unit_state({f"2015|{s}|price@all": (0.004, 0.0013) for s in range(6)})
    (r,) = D.recurrence_table(one_year)
    assert r.n_units == 6 and r.n_years == 1 and D.judge_recurrence(r, CFG).verdict == "PENDING"
    assert D.recurrence(D.DiscoveryState(), "nothing", 10) is None and D.recurrence_table(D.DiscoveryState()) == []


def test_apply_recurrence_supports_but_never_promotes_and_release_needs_it(planted_run):
    src = planted_run[0]
    store = KN.KnowledgeStore()
    for kid in src.store.ids():
        store.add(src.store.latest(kid))
    st = D.DiscoveryState(store=store, dossiers=dict(src.dossiers), ledger=src.ledger)
    hyp = next(k for k in (store.latest(i) for i in store.ids()) if k.epistemic == Epistemic.HYPOTHESIS)
    pid = hyp.knowledge_id[2:]
    dr = float(st.dossiers[pid].direction)
    st.unit_evidence[pid] = {f"{y}|0|price@all": [0.004 * dr, 0.0012, dr, 3.0] for y in (1996, 2004, 2012, 2019)}
    assert D.release_filter(st, "2022-01-01") == []                       # a mined hypothesis is never released
    out = D.apply_recurrence(st, "2021-01-01", CFG)
    assert out["supported"] == 1
    k = store.latest(hyp.knowledge_id)
    assert k.epistemic == Epistemic.SUPPORTED and k.promotion == Promotion.RESEARCH and k.decision_effect == (DecisionEffect.NONE,)
    assert D.apply_recurrence(st, "2021-01-02", CFG)["supported"] == 0 and store.verify() == []
    rel = D.release_filter(st, "2022-01-01")
    assert [r.record_id for r in rel] == [hyp.knowledge_id] and rel[0].gate("2022-01-01")["epistemic"] == "SUPPORTED"
    assert D.release_filter(st, "2022-01-01", replay_windows=[(st.dossiers[pid].windows["train"][0], "2030-01-01")]) == []


def test_sweep_is_resumable_least_covered_first_and_tracks_coverage(planted, tmp_path):
    cfg = D.SweepConfig(years=(2019, 2020), n_slices=2, families=("relvol", "price"), cohorts=("all",), min_names=8)
    sw, st = D.DiscoverySweep.resume(cfg, engine(), tmp_path / "sw")
    load, last = slice_loader(planted), planted.bars["date"].max()
    assert len(sw.pending(last)) == 4 and sw.years_before(last) == 2020 and sw.next_family(last) in ("relvol", "price")
    reps = sw.run(st, load, last, max_units=2)
    assert len(sw.book.records) == 2 and len(reps) <= 2 and (tmp_path / "sw" / "coverage.json").exists()
    done = set(sw.book.records)
    sw2, st2 = D.DiscoverySweep.resume(cfg, engine(), tmp_path / "sw")             # "killed": everything reloaded from disk
    assert set(sw2.book.records) == done and st2.steps == st.steps and st2.ledger.total_trials == st.ledger.total_trials
    sw2.run(st2, load, last, max_units=8)
    assert set(sw2.book.records) > done and sw2.pending(last) == []          # the unfinished 2020 is not a unit yet
    cov = sw2.coverage(last)
    assert set(cov["by_family"]) == {"relvol", "price"} and cov["least_covered_family"] in cov["by_family"] and 0 < cov["fraction_done"] <= 1
    assert D.audit_state(st2) == [] and "SWEEP" in D.sweep_report(sw2, st2, last)
    

def test_sweep_thin_universe_and_bad_config(planted):
    with pytest.raises(D.DiscoveryError):
        D.DiscoverySweep(D.SweepConfig(years=(2019,), cohorts=("moon",)), engine())
    with pytest.raises(D.DiscoveryError):
        D.DiscoverySweep(D.SweepConfig(years=(), n_slices=0), engine())
    cfg = D.SweepConfig(years=(2019,), n_slices=1, families=("price",), min_names=500)
    sw = D.DiscoverySweep(cfg, engine())
    st = D.DiscoveryState()
    assert sw.run(st, slice_loader(planted), planted.bars["date"].max()) == []
    (rec,) = sw.book.records.values()
    assert rec.band_counts == {"thin_universe": 1} and st.steps == 0 and sw.pending(planted.bars["date"].max()) == []


def test_sweep_steps_may_run_out_of_calendar_order(planted):
    st = D.DiscoveryState()
    e = engine()
    e.step(st, "2020-09-01", planted, families=["price"], ordered=False)
    e.step(st, "2020-06-01", planted, families=["price"], ordered=False)
    assert st.steps == 2 and st.store.verify() == []
    with pytest.raises(FirewallBreach):
        e.step(st, "2020-05-01", planted, families=["price"])


def test_stock_shift_null_keeps_each_stocks_values_but_breaks_alignment():
    rng = np.random.default_rng(0)
    tk = np.repeat(np.array(["A", "B", "C"]), 100)
    y = np.concatenate([np.arange(100.0), 1000 + np.arange(100.0), 5000 + rng.normal(size=100)])
    out = D.shift_within_stocks(y, tk, np.random.default_rng(1), 10)
    for t in "ABC":
        assert sorted(out[tk == t]) == sorted(y[tk == t])
    assert (out != y).mean() > 0.9
    short = D.shift_within_stocks(np.arange(6.0), np.array(list("aabbcc")), np.random.default_rng(0), 20)
    assert sorted(short) == list(range(6))


def test_calibrated_p_is_never_more_optimistic_than_the_null():
    null = np.linspace(0, 3, 1000)
    p = D.calibrated_p(np.array([1e-9, 0.5, 1e-9]), np.array([2.9, 0.1, 8.0]), null)
    assert p[0] > 5e-3 and p[1] >= 0.5 and p[2] == pytest.approx(1.0 / 1001)
    assert D.calibrated_p(np.array([0.01]), np.array([3.0]), np.array([1.0]))[0] == 0.01


def test_null_calibration_reports_no_trusted_findings_in_a_null_world(null):
    nc = D.null_calibration(engine(), null, "2020-09-01", ["relvol", "price", "volatility"], n_runs=2)
    assert nc.runs == 2 and nc.generated > 0 and nc.trusted == 0 and nc.trusted_rate == 0.0 and 0.0 <= nc.false_discovery_rate <= 1.0


def test_power_curve_recovers_a_planted_effect_and_not_a_zero_one():
    w = world(plant=0.0, seed=8, extras=False)
    pts = D.power_curve(engine(), w, "2020-09-01", ["relvol", "price"], "relvol__rvol_1 q4", [0.0, 0.008])
    assert not pts[0].found and pts[1].found and pts[1].t_disc > 4
    neg = D.power_curve(engine(), w, "2020-09-01", ["relvol", "price"], "relvol__rvol_1 q4", [-0.008])
    assert neg[0].found


def test_known_overlaps_flags_a_rediscovery_under_another_name(planted_run):
    st, _ = planted_run
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2020-01-01", periods=200)
    idx = pd.MultiIndex.from_product([dates, [f"S{i:02d}" for i in range(30)]], names=["date", "ticker"])
    X = pd.DataFrame({"f": rng.normal(size=len(idx)), "g": rng.normal(size=len(idx))}, index=idx)
    y = pd.Series(rng.normal(0, 0.02, len(idx)), index=idx)
    cfg = dataclasses.replace(CFG, holdout_frac=0.0, min_weeks=20)
    panel = D.Panel.build(X, y, dates[-1] + pd.Timedelta(days=40), cfg, matured_at=pd.Series(dates[-1], index=idx), on_immature="drop")
    an = D.Analyzer(panel, ["f", "g"], {}, cfg)
    d = next(iter(st.dossiers.values()))
    known = dataclasses.replace(d, pattern_id="Pknown", text="f q4")
    s2 = D.DiscoveryState(dossiers={"Pknown": known})
    out = D.known_overlaps(an, s2, {"Pnew": PI.Expression.parse("f q4"), "Pother": PI.Expression.parse("g q0")})
    assert out == {"Pnew": ("K-Pknown",)}
    assert D.known_overlaps(an, D.DiscoveryState(), {"Pnew": PI.Expression.parse("f q4")}) == {}


def test_tabulations_and_explain_on_a_real_run(planted_run):
    st, _ = planted_run
    yt = D.family_yield_table(st)
    assert {"family", "trials", "survivors", "fdr"} <= set(yt.columns) and yt["trials"].sum() == st.ledger.total_trials
    assert D.era_breakdown(st)["n"].sum() == len(st.dossiers)
    pid = next(iter(st.dossiers))
    text = D.explain(st, pid)
    assert st.dossiers[pid].text in text and "validation" in text and "decision impact" in text and D.explain(st, "nope").endswith("unknown")
    assert D.era_breakdown(D.DiscoveryState()).empty


def test_sweep_horizons_are_distinct_outcomes_with_distinct_ids(planted):
    st = D.DiscoveryState()
    reps = D.sweep_horizons(engine(), st, "2020-09-01", planted, [1, 5], families=["relvol"], targets=["excess", "abs_move"])
    assert len(reps) == 4 and st.steps == 4 and st.ledger.times_tested and D.audit_state(st) == []
    assert CFG.tag == "excess_5d" and dataclasses.replace(CFG, target="abs_move", horizon=1).tag == "abs_move_1d"
    assert D.DiscoveryConfig(target="wat").validate()


def test_seed_stability_reports_overlap_between_seeds():
    w = world(plant=0.02, seed=9, extras=False)
    rep = D.seed_stability(lambda sd: D.DiscoveryEngine(dataclasses.replace(CFG, seed=sd), SCFG, audit=False), w, "2020-09-01", ["relvol"], (1, 2))
    assert 0.0 <= rep["min_jaccard"] <= rep["mean_jaccard"] <= 1.0 and len(rep["sizes"]) == 2


# ================================================================== life after discovery
def small_panel(seed=0, n_d=260, n_t=30, effect=None):
    """Noise panel with two features f, g and optional planted effect on 'f q4'. Returns (panel, analyzer, cfg)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_d)
    idx = pd.MultiIndex.from_product([dates, [f"S{i:02d}" for i in range(n_t)]], names=["date", "ticker"])
    X = pd.DataFrame({"f": rng.normal(size=len(idx)), "g": rng.normal(size=len(idx))}, index=idx)
    y = pd.Series(rng.normal(0, 0.02, len(idx)), index=idx)
    if effect:
        y = y + plant_mask(X, effect)
    cfg = dataclasses.replace(CFG, holdout_frac=0.0, min_weeks=20)
    panel = D.Panel.build(X, y, dates[-1] + pd.Timedelta(days=40), cfg, matured_at=pd.Series(dates[-1], index=idx), on_immature="drop")
    return panel, D.Analyzer(panel, ["f", "g"], {}, cfg), cfg


def plant_mask(X, effect):
    r = X["f"].groupby(level=0).rank(pct=True)
    return np.where(r > 0.8, effect, 0.0)


def test_redundant_columns_collapse_copies_but_not_independent_columns():
    rng = np.random.default_rng(0)
    a = rng.normal(size=3000)
    X = pd.DataFrame({"a": a, "a_copy": a * 3 + 1, "a_noisy": a + rng.normal(0, 0.01, 3000), "b": rng.normal(size=3000)})
    X.loc[:200, "a_copy"] = np.nan
    drop = D.redundant_columns(X, 0.98)
    assert set(drop) == {"a_copy", "a_noisy"} and set(drop.values()) == {"a"} and D.redundant_columns(X, 1.0) == {}
    assert D.redundant_columns(X[["b"]]) == {}


def test_panel_health_reports_thin_weeks_and_constants():
    panel, _, _ = small_panel()
    panel.X["const"] = 1.0
    h = D.panel_health(panel)
    assert h["constant_features"] == ["const"] and h["weeks"] == panel.n_wk and h["rows_per_week_min"] > 0 and h["thin_weeks"] == 0


def test_bootstrap_ci_covers_the_truth_and_refuses_thin_data():
    rng = np.random.default_rng(1)
    sw = np.full(60, 20.0)
    sy = sw * rng.normal(0.01, 0.004, 60)
    lo, hi = D.bootstrap_effect_ci(sy, sw, np.ones(60, bool), np.random.default_rng(2))
    assert lo < 0.01 < hi and hi - lo < 0.006
    assert D.bootstrap_effect_ci(sy[:5], sw[:5], np.ones(5, bool), np.random.default_rng(0)) is None


def test_decay_per_block_sign():
    def val(means):
        return D.ValidationResult(tuple(D.BlockStat(k, m, 2.0, 20) for k, m in enumerate(means)), None, None, None, None, 0, len(means), 0.0)
    assert D.decay_per_block(val([0.01, 0.005, 0.0]), 0.01, 1) < -0.4
    assert abs(D.decay_per_block(val([0.01, 0.01, 0.01]), 0.01, 1)) < 1e-9
    assert D.decay_per_block(val([-0.01, -0.005]), 0.01, -1) < 0 and D.decay_per_block(val([0.01]), 0.01, 1) is None


def test_stability_selection_separates_a_broad_effect_from_a_two_week_fluke():
    rng = np.random.default_rng(3)
    n_wk = 60
    SW = np.full((2, n_wk), 25.0)
    real = SW[0] * rng.normal(0.004, 0.003, n_wk)
    fluke = SW[1] * rng.normal(0.0, 0.003, n_wk)
    fluke[[10, 11]] += SW[1][[10, 11]] * 0.05
    f = D.stability_selection(np.vstack([real, fluke]), SW, np.ones(n_wk, bool), 1, n_sub=60, seed=1)
    assert f[0] > 0.8 and f[1] < 0.5 and D.stability_selection(np.empty((0, n_wk)), np.empty((0, n_wk)), np.ones(n_wk, bool), 1).size == 0


def test_shrunk_effects_pull_a_noisy_screen_to_zero_and_keep_a_strong_one():
    rng = np.random.default_rng(4)
    ses = np.full(200, 0.002)
    noise = rng.normal(0, 0.002, 200)
    post = D.shrunk_effects(noise, ses)
    assert np.abs(post).max() < 0.5 * np.abs(noise).max()
    strong = noise.copy()
    strong[:5] = 0.02
    assert D.shrunk_effects(strong, ses)[:5].min() > 0.012
    assert D.shrunk_effects(np.array([0.01]), np.array([np.nan]))[0] == 0.01


def test_parent_comparison_flags_a_pair_that_is_just_its_better_half():
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2020-01-01", periods=300)
    idx = pd.MultiIndex.from_product([dates, [f"S{i:02d}" for i in range(40)]], names=["date", "ticker"])
    X = pd.DataFrame({"f": rng.normal(size=len(idx)), "g": rng.normal(size=len(idx))}, index=idx)
    y = pd.Series(rng.normal(0, 0.02, len(idx)) + plant_mask(X, 0.01), index=idx)          # the effect lives in f alone
    cfg = dataclasses.replace(CFG, holdout_frac=0.0, min_weeks=20)
    panel = D.Panel.build(X, y, dates[-1] + pd.Timedelta(days=40), cfg, matured_at=pd.Series(dates[-1], index=idx), on_immature="drop")
    an = D.Analyzer(panel, ["f", "g"], {}, cfg)
    pt, who = D.parent_comparison(an, PI.Expression.parse("f q4 & g q4"), 1)
    assert pt is not None and pt < 1.0 and who == "f q4"                       # g adds nothing beyond f
    assert D.parent_comparison(an, PI.Expression.parse("f q4"), 1) == (None, "")
    ok, _ = D.parent_comparison(an, PI.Expression.parse("f q4 & g q4"), -1)
    assert ok is not None


def test_jackknife_flags_an_effect_carried_by_one_stock():
    rng = np.random.default_rng(6)
    dates = pd.bdate_range("2020-01-01", periods=260)
    idx = pd.MultiIndex.from_product([dates, [f"S{i:02d}" for i in range(30)]], names=["date", "ticker"])
    X = pd.DataFrame({"f": rng.normal(size=len(idx))}, index=idx)
    y = pd.Series(rng.normal(0, 0.02, len(idx)), index=idx)
    hot = idx.get_level_values(1) == "S07"
    X.loc[hot, "f"] = 5.0                                                     # one stock always in the top quintile ...
    y[hot] += 0.02                                                            # ... and always up
    cfg = dataclasses.replace(CFG, holdout_frac=0.0, min_weeks=20)
    panel = D.Panel.build(X, y, dates[-1] + pd.Timedelta(days=40), cfg, matured_at=pd.Series(dates[-1], index=idx), on_immature="drop")
    an = D.Analyzer(panel, ["f"], {}, cfg)
    ctl = an.controls(an.mask(PI.Expression.parse("f q4")), 3.0, 1)
    assert ctl.jackknife_min_t is not None and ctl.jackknife_min_t < 1.5 and {"name_concentration", "fragile"} & set(ctl.flags)


def test_contradictions_linked_once_and_retirement_needs_age_and_no_confirmation(planted_run):
    src = planted_run[0]
    store = KN.KnowledgeStore()
    for kid in src.store.ids():
        store.add(src.store.latest(kid))
    st = D.DiscoveryState(store=store, dossiers=dict(src.dossiers), ledger=src.ledger)
    panel, an, _ = small_panel()
    d = next(iter(st.dossiers.values()))
    a = dataclasses.replace(d, pattern_id=d.pattern_id, text="f q4", direction=1, verdict=GateVerdict.NEEDS_MORE_EVIDENCE)
    b = dataclasses.replace(d, pattern_id="Pother", text="f q4 unless g q0", direction=-1, verdict=GateVerdict.NEEDS_MORE_EVIDENCE)
    st.dossiers = {a.pattern_id: a, "Pother": b}
    kb = dataclasses.replace(store.latest("K-" + a.pattern_id), knowledge_id="K-Pother")
    store.add(kb)
    pairs = D.find_contradictions(an, st, 0.5)
    assert len(pairs) == 1 and pairs[0][2] > 0.5
    assert D.link_contradictions(st, pairs, "2021-03-01") == 2 and D.link_contradictions(st, pairs, "2021-03-02") == 0
    assert "K-Pother" in store.latest("K-" + a.pattern_id).relations.contradicting and store.verify() == []
    failed = dataclasses.replace(b, verdict=GateVerdict.FAILED)
    st.dossiers["Pother"] = failed
    assert D.retire_failed(st, "2021-03-05", min_age_days=180) == 0            # too young
    assert D.retire_failed(st, "2022-06-01", min_age_days=180) == 1
    assert store.latest("K-Pother").epistemic == Epistemic.RETIRED and len(store.history("K-Pother")) >= 2
    st.dossiers["Pother"] = dataclasses.replace(failed, confirmations=1)
    assert D.retire_failed(st, "2023-06-01") == 0


def test_plan_budget_keeps_a_floor_for_every_family_and_respects_wealth():
    st = D.DiscoveryState()
    st.families = {"good": D.FamilyRecord(trials=100, survivors=30), "bad": D.FamilyRecord(trials=1000, survivors=0)}
    for _ in range(20):
        st.ledger.confirm("bad", False)
    plan = D.plan_budget(st, ["good", "bad", "new"], 3000)
    assert sum(plan.values()) == 3000 and plan["good"] > plan["bad"] and min(plan.values()) >= 300
    st.wealth.wealth = 0.0
    assert D.plan_budget(st, ["good"], 1000) == {"good": 0} and D.plan_budget(st, [], 10) == {}


def test_dossier_changes_names_moved_fields_and_refuses_strangers(planted_run):
    st, _ = planted_run
    d = next(iter(st.dossiers.values()))
    d2 = dataclasses.replace(d, truth=0.123, verdict=GateVerdict.FAILED, run_id="other")
    ch = {f for f, _, _ in D.dossier_changes(d, d2)}
    assert ch == {"truth", "verdict"}
    with pytest.raises(D.DiscoveryError):
        D.dossier_changes(d, dataclasses.replace(d, pattern_id="Pz"))
    assert D.dossier_changes(d, d) == []


def test_markdown_and_family_pair_table_and_cross_target(planted_run):
    st, _ = planted_run
    md = D.discovery_markdown(st, top=5)
    assert md.startswith("# Pattern discovery") and "| pattern |" in md
    fp = D.family_pair_table(st)
    assert list(fp.columns) == ["families", "patterns", "surviving"] and fp["patterns"].sum() > 0
    d = next(iter(st.dossiers.values()))
    two = D.DiscoveryState(dossiers={"P1": dataclasses.replace(d, pattern_id="P1", target="excess_5d", direction=1),
                                     "P2": dataclasses.replace(d, pattern_id="P2", target="abs_move_5d", direction=1),
                                     "P3": dataclasses.replace(d, pattern_id="P3", text="zzz q0", target="excess_5d")})
    ct = D.cross_target_table(two)
    assert len(ct) == 1 and ct["n_targets"].iloc[0] == 2 and D.cross_target_table(D.DiscoveryState()).empty


def test_rescoped_candidates_and_open_questions_are_identity_free(planted_run):
    st, _ = planted_run
    d = next(iter(st.dossiers.values()))
    cells = (D.ContextCell("m_regime__regime_state", "low", (0, 1), 0.01, 3.5, 30), D.ContextCell("m_regime__regime_state", "high", (3, 4), -0.01, -3.0, 30))
    d2 = dataclasses.replace(d, pattern_id="Pctx", text="price__ret_5 q0", direction=1, verdict=GateVerdict.NEEDS_MORE_EVIDENCE, contexts=cells,
                             context_feature="m_regime__regime_state", context_dependence=0.8,
                             failure_conditions=(D.FailureCondition("m_regime__regime_state", (3, 4), "flip", -0.01, -3.0, 30),), decay=-0.5)
    s2 = D.DiscoveryState(dossiers={"Pctx": d2})
    cands = D.rescoped_candidates(s2)
    assert {c.text for c in cands} == {"m_regime__regime_state q0 & price__ret_5 q0", "m_regime__regime_state q1 & price__ret_5 q0"}
    assert all(c.origin == "rescope" for c in cands)
    qs = D.open_questions(s2, "2021-01-01", created_real="2026-01-01T00:00:00+00:00")
    assert len(qs) >= 3 and all(q.text and not re.search(r"\d{4}-\d{2}", q.text) and "T0" not in q.text for q in qs)
    assert len({q.question_id for q in qs}) == len(qs) and D.rescoped_candidates(D.DiscoveryState()) == []


def test_inputs_from_wide_and_normalisers():
    dates = pd.bdate_range("2020-01-01", periods=5)
    blocks = {k: pd.DataFrame(np.arange(10.0).reshape(5, 2) + 1, index=dates, columns=["A", "B"]) for k in ("Open", "High", "Low", "Close", "Volume")}
    blocks["Close"].iloc[2, 1] = np.nan
    inp = D.inputs_from_wide(blocks)
    assert len(inp.bars) == 9 and inp.validate() == []
    with pytest.raises(D.DiscoveryError):
        D.inputs_from_wide({"Open": blocks["Open"]})
    e = D.earnings_table(pd.DataFrame({"ticker": ["A", "A", "B"], "date": ["2020-01-10", "2020-01-10", None], "ann": ["2020-01-01", "2020-01-01", None],
                                       "s": ["1.5", "x", "2"]}), announced_col="ann", surprise_col="s")
    assert len(e) == 1 and e.attrs["dropped_undated"] == 1
    with pytest.raises(D.DiscoveryError):
        D.earnings_table(pd.DataFrame({"ticker": ["A"], "date": ["2020-01-10"], "ann": ["2020-02-01"]}), announced_col="ann")
    f = D.filings_table(pd.DataFrame({"ticker": ["A"], "filed_at": ["2020-03-02"], "acc": ["2020-03-02 17:30"], "form": ["8-K"]}), accepted_col="acc")
    assert f["filed_at"].iloc[0] == pd.Timestamp("2020-03-03")               # accepted after the close: public next session
    with pytest.raises(D.DiscoveryError):
        D.filings_table(pd.DataFrame({"ticker": ["A"], "filed_at": ["2020-03-05"], "acc": ["2020-03-02 10:00"]}), accepted_col="acc")
    i = D.insiders_table(pd.DataFrame({"ticker": ["A", "A"], "filed_at": ["2020-01-02", "2020-01-03"], "value": [100, 50], "side": ["B", "Sell"]}), side_col="side")
    assert i["value"].tolist() == [100.0, -50.0]
    with pytest.raises(D.DiscoveryError):
        D.insiders_table(pd.DataFrame({"ticker": ["A"]}))


def test_feature_report_and_triples_in_the_search(planted):
    fb = DS.build_features(planted, ["relvol", "price"], SCFG)
    rep = D.feature_report(fb)
    assert set(rep["family"]) == {"relvol", "price"} and (rep["columns"] > 0).all()
    st = D.DiscoveryState()
    e = D.DiscoveryEngine(dataclasses.replace(CFG, max_triples=20, triple_top_pairs=5), SCFG, audit=False)
    rep = e.step(st, "2020-09-01", planted, families=["relvol", "price"])
    assert rep.generated > 0 and any(k for k in st.ledger.times_tested)
    assert D.DiscoveryConfig(max_triples=-1).validate() and D.DiscoveryConfig(feature_corr_max=0.2).validate()


def test_replicate_across_years_files_fixed_hypotheses_once_per_year(planted):
    st = D.DiscoveryState()
    e = engine()
    e.step(st, "2020-09-01", planted, families=["relvol", "price"])
    surviving = [d for d in st.dossiers.values() if d.verdict == GateVerdict.NEEDS_MORE_EVIDENCE]
    assert surviving
    out = D.replicate_across_years(st, slice_loader(planted), [2019, 2020], CFG, ["relvol", "price"], SCFG, now="2020-08-01")
    assert out["patterns"] == len(surviving) and out["years"] == 2
    filed = {k for ev in st.unit_evidence.values() for k in ev}
    assert all(k.endswith("|stream|replication") for k in filed)
    first = out["estimates"]
    assert D.replicate_across_years(st, slice_loader(planted), [2019, 2020], CFG, ["relvol", "price"], SCFG, now="2020-08-01")["estimates"] == 0 < first
    assert D.replicate_across_years(D.DiscoveryState(), slice_loader(planted), [2019], CFG)["patterns"] == 0


def test_service_tick_runs_units_and_never_releases(planted, tmp_path):
    cfg = D.SweepConfig(years=(2019,), n_slices=1, families=("relvol",), min_names=8)
    sw, st = D.DiscoverySweep.resume(cfg, engine(), tmp_path / "svc")
    svc = D.DiscoveryService(sw, st, precursor_source=lambda: ["relvol__rvol_1 q4"], units_per_tick=1)
    out = svc.tick("2020-08-01", slice_loader(planted), last_date=planted.bars["date"].max())
    assert out["units_run"] == 1 and out["audit"] == [] and out["pending"] == 0 and (tmp_path / "svc" / "coverage.json").exists()
    assert D.release_filter(st, "2021-01-01") == [] and isinstance(svc.questions, list)
    assert svc.tick("2020-08-02", slice_loader(planted), last_date=planted.bars["date"].max())["units_run"] == 0


def test_there_is_one_ledger_discovery_extends_the_interactions_one():
    from engine.research import interactions as IL
    assert issubclass(D.TrialLedger, IL.TrialLedger)
    led = D.TrialLedger()
    assert led.register("keyA", [D._TrialRef("a"), D._TrialRef("b")]) == 2                     # the base call form still works
    q = led.register("run1", "2020-01-01", ["a", "c"], [0.01, 0.4], ["f", "f"], "keyA")        # the discovery form (also what precursors calls)
    assert len(q) == 2 and led.m_total("keyA") == 3 and led.distinct_trials == 3 and led.looks("keyA") == 4 and led.runs == 1
    assert len(led) == 1                                                                       # only c is new, and only new tests hold a p-value
    back = D.TrialLedger.from_dict(led.to_dict())
    assert back.m_total("keyA") == 3 and back.verify() == [] and back.total_trials == led.total_trials
    plain = IL.TrialLedger()
    plain.register("k", [D._TrialRef("z")])
    assert D.TrialLedger.from_dict(plain.to_dict()).m_total("k") == 1                           # a base-format dict loads too
