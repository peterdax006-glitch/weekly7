"""Find-volatility pipeline (Bible Phase 38 V1-V5; canons C23, C24, C33): planted-effect and degenerate-case tests.
Synthetic data only; the two pipeline runs are shared module fixtures so the file stays well inside its time budget."""
import numpy as np
import pandas as pd
import pytest

from engine import fv_pipeline as F
from engine.exits import CostModel, ExitSpec, Paths, run_exit

CFG = F.FVConfig(min_train_weeks=100, refit_every=60, lgb_trees=50, policy_boot=60)
N_T, N_W = 120, 230


@pytest.fixture(scope="module")
def planted():
    bars = F.synthetic_bars(N_T, N_W, seed=3, dir_acc=0.95, event_p=0.2)
    panel = F.build_panel(bars, CFG)
    return F.walk_forward(panel, CFG)


@pytest.fixture(scope="module")
def noise():
    bars = F.synthetic_bars(N_T, N_W, seed=4, dir_acc=0.5, event_p=0.2)     # movers are announced, direction is a coin flip
    return F.walk_forward(F.build_panel(bars, CFG), CFG)


# ------------------------------------------------------------------ data and timing
def test_validate_bars_catches_planted_defects():
    b = F.synthetic_bars(10, 10, seed=1)
    assert F.validate_bars(b) == []
    b["High"].iloc[5, 2] = b["Low"].iloc[5, 2] * 0.5
    b["Close"].iloc[7, 3] = -1.0
    bad = " ".join(F.validate_bars(b))
    assert "high below low" in bad and "non-positive close" in bad
    with pytest.raises(ValueError):
        F.build_panel(b, CFG)
    assert "missing Open" in " ".join(F.validate_bars({k: v for k, v in F.synthetic_bars(5, 5).items() if k != "Open"}))


def test_gather_fills_next_open_pads_and_flattens_missing():
    b = F.synthetic_bars(6, 30, seed=2)
    b["Open"].iloc[11, 1] = np.nan          # a missing bar inside a holding window
    P = F.build_panel(b, CFG)
    k = 1                                    # second decision week
    w = P.dec[k]
    o, h, l, c, ok, prev = P.gather(np.array([k, k]), np.array([0, 1]))
    assert ok[0]                                                    # entry bar exists
    assert o[0, 0] == pytest.approx(b["Open"].to_numpy()[w + 1, 0])  # entry = open of the NEXT session, never the decision close
    assert prev[0] == pytest.approx(b["Close"].to_numpy()[w, 0])
    assert (h >= np.maximum(o, c) - 1e-9).all() and (l <= np.minimum(o, c) + 1e-9).all()
    assert np.isfinite(o).all() and np.isfinite(c).all()             # the missing bar was flattened, not propagated as NaN
    # a session that ends the week early is padded flat at its last close
    o2, h2, l2, c2, _, _ = P.gather(np.array([k]), np.array([0]))
    n_real = int(P.last[k] - w)
    if n_real < F.D_BARS:
        assert (o2[0, n_real:] == c2[0, n_real - 1]).all() and (h2[0, n_real:] == l2[0, n_real:]).all()


def test_week_ends_skip_final_incomplete_week():
    s = pd.bdate_range("2020-01-06", periods=12)         # Mon 6 Jan ... Tue 21 Jan
    we = F.week_end_sessions(s)
    assert list(s[we].dayofweek) == [4, 4]               # two complete weeks; the trailing Mon/Tue is not a decision day


def test_fill_timing_audit_on_run(planted):
    a = F.audit_fill_timing(planted)
    assert a["ok"] and a["positions"] > 0 and a["weekend_bars"] == 0 and a["max_bars"] <= F.D_BARS


# ------------------------------------------------------------------ shorts, exits, rule selection
def test_reflect_short_negates_return_exactly():
    rng = np.random.default_rng(0)
    n = 40
    o = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, 5)), 1))
    c = o * (1 + rng.normal(0, 0.01, (n, 5)))
    h, l = np.maximum(o, c) * 1.004, np.minimum(o, c) * 0.996
    p = Paths(o, h, l, c, o[:, 0], np.full(n, .02), np.full(n, .02), ["2020-01-03"] * n, ["2020-01-10"] * n, ["k"] * n, ["T"] * n)
    side = np.where(np.arange(n) % 2 == 0, 1, -1)
    Z = CostModel(0.0, 0.0)
    r = run_exit(F.reflect_short(p, side), ExitSpec(), Z)
    price_ret = p.c[:, -1] / p.o[:, 0] - 1
    assert np.allclose(r.net, side * price_ret, atol=1e-9)
    assert F.reflect_short(p, np.ones(n)) is p


def _spike_paths(seed, weeks=70, per=20, spike=True):
    """Half the positions spike to +11% on day 2 then decay to -4% by the close; the rest are noise."""
    rng = np.random.default_rng(seed)
    n = weeks * per
    is_sp = (rng.random(n) < 0.5) & spike
    steps = rng.normal(0, 0.004, (n, 5))
    path = np.cumsum(steps, 1)
    prof = np.array([0.05, 0.11, 0.06, 0.0, -0.04])
    path = np.where(is_sp[:, None], prof[None, :] + steps * 0.2, path)
    c = 50 * (1 + path)
    o = np.concatenate([np.full((n, 1), 50.0), c[:, :-1]], 1)
    h = np.maximum(o, c) * 1.002
    l = np.minimum(o, c) * 0.998
    wk = np.datetime64("2015-01-02", "D") + np.repeat(np.arange(weeks) * 7, per)
    kind = np.array(["hv_hp"] * n, object)
    return Paths(o, h, l, c, o[:, 0], np.full(n, .02), np.full(n, .02), wk, wk + 6, kind, np.array([f"T{i%50}" for i in range(n)]))


def test_fv_select_finds_planted_target_exit_and_keeps_baseline_on_noise():
    rules = F.build_rule_set(True, False, CostModel(2.0, 5.0))
    tr = _spike_paths(1)
    rule, sel = F.fv_select(tr, rules, 0, CFG)
    assert sel.reason == "selected" and "target" in rule.name or "t10" in rule.name, sel
    base_hit = (rules[0].run(tr).gross >= 0.10).mean()
    assert (rule.run(tr).gross >= 0.10).mean() > base_hit + 0.3
    # the same machinery on positions where no exit can matter must not invent a winner
    rule0, sel0 = F.fv_select(_spike_paths(2, spike=False), rules, 0, CFG)
    assert rule0.name == "week_end" and sel0.reason != "selected"


def test_fv_select_neutral_on_thin_data():
    rules = F.build_rule_set(True, True, CostModel())
    rule, sel = F.fv_select(_spike_paths(3, weeks=5), rules, 0, CFG)
    assert rule.name == "week_end" and sel.reason == "neutral:insufficient_data"


def test_fv_compare_feasibility_beats_hit_rate():
    a = dict(n=100, hit=0.9, mean=0.05, cat=0.10, cat_hi=0.2, feasible=False)
    b = dict(n=100, hit=0.2, mean=0.00, cat=0.00, cat_hi=0.03, feasible=True)
    assert F.fv_compare(a, b) == -1 and F.fv_compare(b, a) == 1
    assert F.fv_compare(b, dict(b)) == 0 and F.fv_compare(dict(n=0), b) == 0


def test_policy_learning_uses_only_finished_positions():
    p = _spike_paths(5, weeks=70)
    early = np.datetime64("2015-01-02", "D") + 30 * 7            # only ~30 weeks are finished by then
    pol = F.learn_policy(p, F.build_rule_set(True, False, CostModel()), early, 0, CFG)
    assert pol.as_of == early
    assert all(s.n_weeks <= 31 for s in pol.selections.values())


# ------------------------------------------------------------------ loss-cap filter
def _gap_paths(n, atr, gap_sd, seed=0):
    rng = np.random.default_rng(seed)
    prev = np.full(n, 50.0)
    o, c = np.empty((n, 5)), np.empty((n, 5))
    for d in range(5):
        o[:, d] = prev * (1 + rng.normal(0, gap_sd, n))
        c[:, d] = o[:, d] * (1 + rng.normal(0, 0.005, n))
        prev = c[:, d]
    h, l = np.maximum(o, c) * 1.002, np.minimum(o, c) * 0.998
    wk = np.datetime64("2015-01-02", "D") + np.repeat(np.arange(n // 10 + 1) * 7, 10)[:n]
    return Paths(o, h, l, c, o[:, 0] / (1 + 0.0), np.full(n, atr), np.full(n, atr), wk, wk + 6, np.array(["k"] * n, object), np.array(["T"] * n))


def test_loss_cap_filter_drops_gap_prone_names_and_is_inert_on_thin_history():
    train = _gap_paths(1500, 0.02, 0.05)
    f = F.LossCapFilter(CFG).fit(train)
    assert f.reason == "ok"
    calm = _gap_paths(40, 0.005, 0.01, seed=1)       # gaps are learned in ATR units (2.5 ATR sd here): a volatile name's
    wild = _gap_paths(40, 0.08, 0.01, seed=2)        # ATR is large, so the same cap sits within a couple of ATRs of it
    assert f.p_breach(wild).mean() > 0.2 > f.p_breach(calm).mean()
    assert f.keep(wild).mean() == 0 and f.keep(calm).all()
    thin = F.LossCapFilter(CFG).fit(_gap_paths(50, 0.02, 0.05))
    assert thin.model is None and thin.keep(wild).all()


# ------------------------------------------------------------------ the pipeline on planted data
def test_planted_market_reaches_the_goal_and_each_stage_is_measured(planted):
    mv = F.mover_report(planted, boot=100)
    assert mv["hit_touch"] > 0.80 and mv["hit_touch_lo"] > 0.75                  # V1 on a market where movers are knowable
    assert mv["hit_touch"] > 3 * mv["base_rate_all"]
    tab, diffs = F.ablation(planted, boot=100)
    t = tab.set_index("variant")
    md = t.loc["M+D"]
    assert md["dir_acc"] > 0.88 and md["coverage"] > 0.9                        # V2: direction found and gate is open
    assert md["share10_gross_lo"] >= 0.80                                       # V5: 8 of 10 finish +10%
    assert md["goal8_weeks"] > 0.6 and md["mean_week"] > 0.09
    assert t.loc["M", "share10_gross"] < 0.55                                    # movers alone (all long) are a coin flip
    row = diffs.set_index("to").loc["M+D"]
    assert row["verdict"] == "helps" and row["d_share10_lo"] > 0.25
    assert t["worst_position"].le(0).all() and (t["cat_wilson_hi"] > 0).all()


def test_direction_engine_opened_on_planted_and_calibrated(planted):
    o = planted.origins
    assert o["dir_open"].all()
    assert (o["dir_acc_gated"] > 0.90).all() and o["mover_auc_cal"].min() > 0.95


def test_noise_direction_makes_the_pipeline_abstain(noise):
    o = noise.origins
    assert not o["dir_open"].fillna(False).any()                                 # engine closed everywhere: no calibrated 80%
    pos = F.positions(noise, "MD")
    assert pos["taken"].sum() == 0
    s = F.summarize(noise, "MD", pos, boot=50)
    assert s["coverage"] == 0 and s["mean_week"] == 0 and s["worst_position"] == 0
    assert (noise.cands.loc[noise.cands["rank"] < 10, "reason"].str.startswith("engine_closed")).all()
    # movers alone are still found: the abstention is about direction, not detection
    assert F.mover_report(noise, boot=50)["hit_touch"] > 0.85
    # and betting anyway (no direction stage) is a coin flip, which is what the gate protected against
    m = F.summarize(noise, "M", boot=200)
    assert abs(m["mean_week"]) < 0.03 and m["share10_gross"] < 0.6


def test_stage_switches_and_orientation(planted):
    assert F.stage_key("m+d+e") == "MDE" and F.stage_key("MDES") == "MDES"
    for bad in ("D", "ME", "MX", ""):
        with pytest.raises(ValueError):
            F.stage_key(bad)
    m, md = F.positions(planted, "M"), F.positions(planted, "MD")
    assert (m["side"] == 1).all() and m.groupby("date").size().max() <= planted.cfg.n_picks
    assert set(md["side"].unique()) <= {-1, 1} and (md["side"] == -1).any()      # shorts really are taken
    assert F.positions(planted, "MDES")["taken"].sum() <= md["taken"].sum()      # the cap filter can only remove positions


def test_era_and_type_tables_add_up(planted):
    eras = F.era_table(planted, "MDES")
    assert eras["weeks"].sum() == len(planted.week_dates)
    assert eras["bets"].sum() == int(F.positions(planted, "MDES")["taken"].sum())
    ty = F.type_table(planted, "MD")
    assert ty["bets"].sum() == int(F.positions(planted, "MD")["taken"].sum())
    txt = F.format_report(planted, *F.ablation(planted, boot=50), F.mover_report(planted, boot=50), eras,
                          F.goal_verdict(F.summarize(planted, "MDES", boot=50)))
    assert "V1 movers" in txt and "No stop guarantees a floor" in txt


def test_goal_verdict_uses_the_lower_bound_and_the_cap():
    good = dict(weeks=10, share10_gross=0.9, share10_gross_lo=0.85, worst_position=-0.05)
    assert F.goal_verdict(good)["goal_met"]
    assert not F.goal_verdict({**good, "share10_gross_lo": 0.7})["goal_met"]
    v = F.goal_verdict({**good, "worst_position": -0.25})
    assert not v["goal_met"] and "breached" in v["why"]
    assert not F.goal_verdict({"weeks": 0})["goal_met"]


# ------------------------------------------------------------------ look-ahead, resume, hooks
def test_future_scramble_leaves_earlier_decisions_identical():
    cfg = F.FVConfig(min_train_weeks=100, refit_every=60, lgb_trees=30, policy_boot=30)
    bars = F.synthetic_bars(60, 170, seed=5)
    cut = bars["Close"].index[5 * 135]
    a = F.audit_no_lookahead(bars, cfg, cut, last_block=1)
    assert a["identical"] and a["n_compared"] > 300 and a["max_p_move_diff"] == 0.0


def test_scramble_audit_catches_a_planted_leak(monkeypatch):
    """A mover score that reads the realised outcome of its own week must fail the audit (a check that cannot fail is
    worthless)."""
    cfg = F.FVConfig(min_train_weeks=100, refit_every=60, lgb_trees=30, policy_boot=30)
    bars = F.synthetic_bars(60, 170, seed=5)
    leak = {}
    real_build, real_score = F.build_panel, F.MoverStage.score

    def build(b, c=cfg, start=None):
        P = real_build(b, c, start)
        leak["close"] = P.lab["close"]
        return P

    def leaky(self, XR):
        return real_score(self, XR) + 5 * leak["close"].reindex(XR.index).fillna(0).to_numpy()

    monkeypatch.setattr(F, "build_panel", build)
    monkeypatch.setattr(F.MoverStage, "score", leaky)
    a = F.audit_no_lookahead(bars, cfg, bars["Close"].index[5 * 135], last_block=1)
    assert not a["identical"] and a["max_p_move_diff"] > 0


def test_checkpoint_resume_reproduces_a_straight_run(planted, tmp_path):
    ck = tmp_path / "fv.pkl"
    bars = F.synthetic_bars(N_T, N_W, seed=3, dir_acc=0.95, event_p=0.2)
    panel = F.build_panel(bars, CFG)
    F.walk_forward(panel, CFG, ckpt=ck, last_block=1)
    assert ck.exists()
    full = F.walk_forward(panel, CFG, ckpt=ck)
    cols = ["date", "ticker", "p_move", "side_eng", "eng_bet"]
    assert full.cands[cols].reset_index(drop=True).equals(planted.cands[cols].reset_index(drop=True))
    other = F.walk_forward(panel, F.FVConfig(min_train_weeks=100, refit_every=60, lgb_trees=51), ckpt=ck, last_block=1)
    assert len(other.origins) == 1                    # a checkpoint from a different config is ignored, not resumed


class _FakePatterns:
    decision = {"deploy": True}

    def fit(self, X, y, now):
        assert X.index.get_level_values(0).max() < now + pd.Timedelta(days=1) and len(X) == len(y)
        return self

    def features(self, Xday, as_of):
        return pd.DataFrame({"pat_dir": Xday["r1"].to_numpy() * 10, "pat_mov": Xday["absr1"].to_numpy() * 10}, index=Xday.index)


class _BrokenPatterns(_FakePatterns):
    def fit(self, X, y, now):
        raise RuntimeError("boom")


def test_pattern_hook_is_optional_and_failures_are_recorded_not_hidden():
    bars = F.synthetic_bars(N_T, N_W, seed=3)
    panel = F.build_panel(bars, CFG)
    ok = F.walk_forward(panel, CFG, pattern_factory=_FakePatterns, last_block=1)
    assert "pattern_error" not in ok.origins.columns or ok.origins["pattern_error"].isna().all()
    assert len(ok.cands) > 0 and ok.origins["mover_reason"].iloc[0] == "ok"
    bad = F.walk_forward(panel, CFG, pattern_factory=_BrokenPatterns, last_block=1)
    assert "boom" in bad.origins["pattern_error"].iloc[0] and len(bad.cands) > 0


# ------------------------------------------------------------------ degenerate cases
def test_too_short_history_trades_nothing_and_every_report_survives():
    bars = F.synthetic_bars(30, 60, seed=6)
    run = F.walk_forward(F.build_panel(bars, CFG), CFG)
    assert len(run.cands) == 0 and len(run.week_dates) == 0
    tab, diffs = F.ablation(run, boot=20)
    assert (tab["weeks"] == 0).all() and diffs.empty
    assert F.mover_report(run)["weeks"] == 0
    assert F.era_table(run, "MD").empty and F.type_table(run, "MD").empty
    assert not F.goal_verdict(F.summarize(run, "MDES"))["goal_met"]
    assert F.audit_fill_timing(run)["ok"]


def test_certify_threshold_and_bootstrap_helpers():
    p = np.linspace(0.01, 0.99, 400)
    y = (p > 0.6).astype(float)                       # everything above 0.6 is a hit
    t = F.certify_threshold(p, y, 0.95, 40)
    assert t is not None and 0.55 < t < 0.8
    assert F.certify_threshold(p, np.zeros(400), 0.95, 40) is None
    assert F.certify_threshold(np.zeros(0), np.zeros(0)) is None
    rng = np.random.default_rng(0)
    x = rng.normal(0.05, 0.02, 200)
    for block in (1, 5):
        bi = F.boot_index(len(x), 300, block, np.random.default_rng(1))
        assert bi.shape == (300, 200) and bi.max() < 200
        lo, hi = F.ci(x[bi].mean(1))
        assert lo < 0.05 < hi and hi - lo < 0.02
    assert F.boot_index(0, 10, 3, rng).shape == (10, 0)


def test_save_results_writes_provenance(planted, tmp_path):
    tab, diffs = F.ablation(planted, boot=30)
    v = F.goal_verdict(F.summarize(planted, "MDES", boot=30))
    out = F.save_results(tmp_path / "fv", planted, tab, diffs, F.mover_report(planted, boot=30), F.era_table(planted, "MDES"),
                         F.type_table(planted, "MDES"), v, {"note": "test"})
    import json
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["config_hash"] == planted.cfg.hash() and prov["seed"] == planted.cfg.seed and prov["note"] == "test"
    assert (out / "report.md").exists() and (out / "candidates.parquet").exists()
