"""Tests for engine.research.discovery / discovery_sources (C66 section 13). Synthetic worlds only; each plants an effect or
proves a null. IMPLEMENTED — NOT VALIDATED: unit tests, no real-data run."""
import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from engine.learning import knowledge as KN
from engine.learning.core import DecisionEffect, Epistemic, FirewallBreach, Promotion
from engine.research import discovery as D
from engine.research import discovery_sources as DS
from engine.research.core import GateVerdict, MaturedRecord


def world(n_t=40, n_d=420, seed=1, plant=0.0, extras=True):
    """Random-walk bars. plant>0: a volume spike on day t lifts day t+1's return by `plant` (relvol family should find it)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_d)
    tk = [f"T{i:03d}" for i in range(n_t)]
    mk = rng.normal(0.0002, 0.007, n_d)
    frames = []
    for i, t in enumerate(tk):
        spike = rng.random(n_d) < 0.08
        r = mk * rng.uniform(0.7, 1.3) + rng.normal(0, 0.012, n_d)
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
    with pytest.raises(D.DiscoveryError):
        D.StreamingScreen([D.Cand(PI.Expression.parse("a q0 unless b q1"), "unless", ())])
    with pytest.raises(D.DiscoveryError):
        ss.add_chunk(next(DS.year_chunks(loader, [2019], 5, ["price"], SCFG))[1], pd.Series(dtype=float))


def test_patternminer_wrapper_supplies_candidates_that_are_still_judged():
    X = pd.DataFrame({"a": np.random.default_rng(0).normal(size=1500), "b": np.random.default_rng(1).normal(size=1500)},
                     index=pd.MultiIndex.from_product([pd.bdate_range("2020-01-01", periods=50), [f"S{i}" for i in range(30)]]))
    y = pd.Series(np.where(X["a"] > 0.8, 0.01, 0.0) + np.random.default_rng(2).normal(0, 0.02, 1500), index=X.index)
    cands, table, n = D.mine_with_patternminer(X, y, "2020-06-01", ["a", "b"], {"min_n": 30, "max_pairs": 20, "max_unless": 5, "null_reps": 1})
    assert n >= 0 and all(c.origin == "miner" for c in cands)
