"""Beta-neutral outcome and beta-proxy control of engine.research.discovery (C66 section 13; the control the null-world work showed was
missing). Synthetic worlds only. IMPLEMENTED — NOT VALIDATED."""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from engine.research import discovery as D
from engine.research import discovery_sources as DS
from engine.research.core import GateVerdict
from test_research_discovery import CFG, SCFG, engine, world

DRIFT = 0.006


def factor_bars(n_t=14, n_d=300, seed=0, drift=DRIFT):
    """Every name is beta_i * market + small noise, with a strongly trending market: raw excess is all beta."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_d)
    m = rng.normal(drift, 0.008, n_d)
    beta = np.linspace(0.3, 1.7, n_t)
    rows = []
    for i in range(n_t):
        r = beta[i] * m + rng.normal(0, 0.004, n_d)
        c = 50 * np.exp(np.cumsum(r))
        o = np.r_[c[0], c[:-1]]
        rows.append(pd.DataFrame(dict(date=dates, ticker=f"B{i:02d}", open=o, high=np.maximum(o, c) * 1.001, low=np.minimum(o, c) * 0.999,
                                      close=c, volume=1e6)))
    return pd.concat(rows, ignore_index=True), beta


def fwd_market(bars, h=5):
    op = bars.pivot(index="date", columns="ticker", values="open")
    cl = bars.pivot(index="date", columns="ticker", values="close")
    return (cl.shift(-h) / op.shift(-1) - 1.0).mean(axis=1)


def test_residual_labels_remove_the_beta_that_raw_excess_keeps():
    bars, beta = factor_bars()
    ex, _ = DS.target_labels(bars, "excess", 5)
    rs, _ = DS.target_labels(bars, "resid", 5)
    mk = fwd_market(bars)
    truth = pd.Series(beta, index=[f"B{i:02d}" for i in range(len(beta))])

    def link(y):
        d = y.index.get_level_values(0)
        t = y.index.get_level_values(1)
        return float(np.corrcoef(y.to_numpy(), (truth.reindex(t).to_numpy() - 1.0) * mk.reindex(d).to_numpy())[0, 1])
    assert link(ex) > 0.6 and abs(link(rs)) < 0.25 and abs(link(rs)) < link(ex) / 3
    assert abs(float(rs.groupby(level=0).mean().abs().max())) < 1e-9
    assert len(rs) < len(ex)                                      # no beta before enough history: those rows are absent, not guessed


def test_past_betas_are_point_in_time_and_recover_the_truth():
    bars, beta = factor_bars(seed=1)
    cl = bars.pivot(index="date", columns="ticker", values="close")
    full = DS.past_betas(cl)
    part = DS.past_betas(cl.iloc[:180])
    both = full.iloc[:180]
    assert np.allclose(part.to_numpy(), both.to_numpy(), equal_nan=True)                     # nothing after the cut enters
    est = full.iloc[-1].to_numpy()
    adj = (1 - DS.BLUME) * beta + DS.BLUME
    assert np.corrcoef(est, beta)[0, 1] > 0.97 and np.abs(est - adj).max() < 0.15
    assert np.isnan(full.iloc[5]).all()


def test_residual_labels_mature_before_now_and_refuse_bad_horizon():
    bars, _ = factor_bars()
    y, m = DS.residual_labels(bars, 5, now="2019-09-01")
    assert (m < pd.Timestamp("2019-09-01")).all() and len(y) > 0
    with pytest.raises(DS.SourceError):
        DS.residual_labels(bars, 0)


@pytest.fixture(scope="module")
def beta_world():
    return world(n_t=30, seed=21, plant=0.0, extras=False, beta_spread=0.8, drift=DRIFT)


def controls_for(w, families, expr_text, target_cfg=CFG):
    """Analyzer.controls for one expression on a panel that carries both the raw excess and the beta-neutral outcome."""
    from engine import pattern_identity as PI
    now = pd.Timestamp("2020-09-01")
    inp = w.before(now)
    fb = DS.build_features(inp, families, SCFG)
    y, m = DS.forward_labels(inp.bars, 5, now)
    yr, _ = DS.residual_labels(inp.bars, 5, now)
    panel = D.Panel.build(fb.X, y, now, target_cfg, m, inp.sectors).with_resid(yr)
    an = D.Analyzer(panel, list(fb.X.columns), fb.family_of, target_cfg)
    mask = an.mask(PI.Expression.parse(expr_text))
    raw = an.stat(mask, an.sel_all)
    return an.controls(mask, raw.t, 1 if raw.mean > 0 else -1), raw


def test_control_flags_a_pure_beta_pattern_and_spares_a_genuine_one(beta_world):
    ctl, raw = controls_for(beta_world, ["correlations"], "correlations__beta_63 q4")
    assert abs(raw.t) >= 2 and "beta_proxy" in ctl.flags and ctl.resid_ratio < CFG.beta_min_ratio
    w = world(n_t=30, seed=22, plant=0.04, extras=False, beta_spread=0.8, drift=DRIFT)
    ctl, raw = controls_for(w, ["relvol"], "relvol__rvol_1 q4")
    assert raw.t >= 4 and "beta_proxy" not in ctl.flags and ctl.resid_ratio > 0.7 and ctl.resid_t >= CFG.beta_min_t


def test_engine_stops_beta_patterns_with_the_control_and_by_the_stock_shift_null(beta_world):
    fams = ["correlations", "volatility", "liquidity"]
    # a null that ignores stock persistence lets beta patterns through the screen; the ticker-effect control (which also happens to catch a
    # static beta) is switched off so that the beta control's own contribution is what is measured
    week = dataclasses.replace(CFG, null="week_shuffle", ticker_effect_min_ratio=-1.0)
    on = D.DiscoveryEngine(week, SCFG, audit=False, families_per_step=24)
    st = D.DiscoveryState()
    rep = on.step(st, "2020-09-01", beta_world, families=fams)
    proxies = [d for d in st.dossiers.values() if "beta_proxy" in d.controls.flags]
    assert rep.shortlisted > 0 and proxies
    assert all(d.verdict == GateVerdict.QUARANTINED for d in proxies)
    assert not [d for d in st.dossiers.values() if d.verdict == GateVerdict.NEEDS_MORE_EVIDENCE and "beta_proxy" in d.controls.flags]
    off = D.DiscoveryEngine(dataclasses.replace(week, beta_control=False), SCFG, audit=False, families_per_step=24)
    st_off = D.DiscoveryState()
    off.step(st_off, "2020-09-01", beta_world, families=fams)
    assert not any("beta_proxy" in d.controls.flags for d in st_off.dossiers.values())
    passed_without = [d for d in st_off.dossiers.values() if d.verdict == GateVerdict.NEEDS_MORE_EVIDENCE and d.pattern_id in {p.pattern_id for p in proxies}]
    assert passed_without, "without the control the same beta patterns must get through, or the control proves nothing"
    default = D.DiscoveryState()
    engine().step(default, "2020-09-01", beta_world, families=fams)                 # default null already refuses them
    assert [d for d in default.dossiers.values() if d.verdict == GateVerdict.NEEDS_MORE_EVIDENCE and (d.truth or 0) >= 0.9] == []


def test_zero_trusted_findings_on_residuals_in_a_beta_and_drift_world(beta_world):
    fams = ["correlations", "volatility", "liquidity"]
    for null in ("both", "week_shuffle"):
        res_engine = D.DiscoveryEngine(dataclasses.replace(CFG, target="resid", null=null), SCFG, audit=False, families_per_step=24)
        st = D.DiscoveryState()
        rep = res_engine.step(st, "2020-09-01", beta_world, families=fams)
        assert rep.tested > 0
        assert [d for d in st.dossiers.values() if d.verdict == GateVerdict.NEEDS_MORE_EVIDENCE and (d.truth or 0) >= 0.9] == []
        assert all(d.target == "resid_5d" for d in st.dossiers.values()) and D.audit_state(st) == []


def test_a_genuine_planted_effect_survives_on_residuals():
    w = world(n_t=30, seed=22, plant=0.04, extras=False, beta_spread=0.8, drift=DRIFT)
    st = D.DiscoveryState()
    D.DiscoveryEngine(dataclasses.replace(CFG, target="resid"), SCFG, audit=False, families_per_step=24).step(
        st, "2020-09-01", w, families=["relvol", "price"])
    hits = [d for d in st.dossiers.values() if "relvol__rvol_1 q4" in d.text and d.direction > 0 and d.verdict == GateVerdict.NEEDS_MORE_EVIDENCE]
    assert hits and max(d.t_disc for d in hits) > 4
    assert all("beta_proxy" not in d.controls.flags for d in hits)


def test_beta_control_config_and_missing_history():
    assert D.DiscoveryConfig(beta_window=5).validate() and "resid" in DS.TARGETS
    ok = D.DiscoveryConfig(target="resid")
    assert ok.validate() == [] and ok.tag == "resid_5d"
