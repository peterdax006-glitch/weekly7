"""F08 open hooks (INTEGRATION B01, B07, B12; C69 ledger W-14): every hook is exercised on its production function and
shown to CHANGE behaviour - each test here fails if its hook is removed. Planted cases: same-close fills (the audit must
fail), a missing open (no substitute fill), an unpurged fit (the future scramble must fail closed), a look-ahead trust
table, a repeated grid job (it must be skipped). Empty cases: no decision dates, an empty candidate batch."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from engine import backtest as BT, checkpoint, experiment_memory as EM, pit, train as TR   # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")


# ---------------------------------------------------------------------------------------------------------- worlds
def world(n=160, nt=24, seed=0, start="2021-01-04"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    tk = [f"T{i:02d}" for i in range(nt)]
    C = pd.DataFrame(30 * np.exp(np.cumsum(rng.normal(0, 0.02, (n, nt)), axis=0)), index=idx, columns=tk)
    O = C.shift(1).fillna(C.iloc[0]) * np.exp(rng.normal(0, 0.01, C.shape))      # opens differ from the prior close
    H = np.maximum(C, O) * 1.01
    L = np.minimum(C, O) * 0.99
    stocks = {"Open": O, "High": H, "Low": L, "Close": C}
    mi = pd.MultiIndex.from_product([idx, tk], names=["date", "ticker"])
    X = pd.DataFrame({"log_dv": 19.0, "vol20": rng.uniform(0.01, 0.05, len(mi)), "max20": 0.05, "ev_red_flag": 0.0,
                      "ev_offering": 0.0, "atr_pct": 0.03, "earn_in_week": 0.0, "sig": rng.normal(0, 1, len(mi)),
                      "m_vix": 18.0, "m_spy_ma200": 0.02, "m_vix_chg5": 0.0}, index=mi)
    score = pd.Series(rng.normal(0, 1, len(mi)), index=mi)
    return stocks, X, score


def topk(stocks, X, score, **kw):
    return BT.run_topk(score, X, stocks, k=3, exit_q=0.8, **kw)


def _session_after(idx, d):
    return idx[idx.get_loc(pd.Timestamp(d)) + 1]


# ================================================================================================= B01 fills
def test_topk_fills_at_the_next_open_and_the_fill_audit_proves_it():
    stocks, X, score = world()
    eq, st = topk(stocks, X, score)
    F = BT.run_topk.fills
    assert st["fill_audit"] == "PASS" and len(F.fills) > 20
    idx, O = stocks["Close"].index, stocks["Open"]
    for f in F.fills:
        fd = pd.Timestamp(f["fill_date"])
        assert fd == _session_after(idx, f["decision_date"])                 # strictly the next session
        assert f["fill_price"] == pytest.approx(O.at[fd, f["ticker"]], rel=1e-12)   # at its open, never the close
    # the Executor's hash-chained log saw every fill
    assert F.store.log.verify() and sum(1 for r in F.store.log.frame()["op"] if r == "fill") == len(F.fills)


def test_legacy_close_fills_fail_the_audit_and_give_different_numbers():
    """Planted defect: same-close execution. The same fill_audit gate the Test loop uses must reject it."""
    stocks, X, score = world()
    eq_open, st_open = topk(stocks, X, score)
    eq_close, st_close = topk(stocks, X, score, fill="close")
    assert st_open["fill_audit"] == "PASS" and st_close["fill_audit"] == "FAIL"
    r = BT.run_topk.fills.audit()
    assert any(e.startswith("same_close_fill") for e in r["errors"])
    assert not np.allclose(eq_open.values, eq_close.values)                  # the hook changes the result


def test_a_next_open_run_whose_ledger_fails_the_audit_raises(monkeypatch):
    """If the pricing were silently reverted to the close while the ledger still claims next-open, numbers must not come out."""
    stocks, X, score = world()
    orig = BT.Fills.order

    def cheat(self, d, t, shares, close_px):
        p = orig(self, d, t, shares, close_px)
        if p is not None:
            self.fills[-1]["fill_price"] = float(close_px)
        return p
    monkeypatch.setattr(BT.Fills, "order", cheat)
    with pytest.raises(pit.FailClosed):
        topk(stocks, X, score)


def test_a_missing_open_leaves_the_order_unfilled_never_substituted():
    stocks, X, score = world()
    eq, st = topk(stocks, X, score)
    first = BT.run_topk.fills.fills[0]
    O = stocks["Open"].copy()
    O.at[pd.Timestamp(first["fill_date"]), first["ticker"]] = np.nan
    stocks2 = {**stocks, "Open": O}
    eq2, st2 = topk(stocks2, X, score)
    F2 = BT.run_topk.fills
    assert st2["fill_audit"] == "PASS" and st2["unfilled"] >= 1
    assert any(u["ticker"] == first["ticker"] and u["decision_date"] == first["decision_date"] for u in F2.unfilled)
    assert not any(f["ticker"] == first["ticker"] and f["fill_date"] == first["fill_date"] for f in F2.fills)


def test_next_open_without_opening_prices_fails_closed():
    stocks, X, score = world(n=40)
    with pytest.raises(pit.FailClosed):
        topk({k: v for k, v in stocks.items() if k != "Open"}, X, score)


def test_empty_run_topk_returns_an_empty_equity_curve():
    stocks, X, score = world(n=40)
    eq, st = topk(stocks, X, score, start="2030-01-01")
    assert eq.empty and st["fill_audit"] == "EMPTY"


# ---------------------------------------------------------------------------------------- the full backtest (run)
def full_inputs(n=70, nt=16, seed=1):
    stocks, X, score = world(n, nt, seed)
    C = stocks["Close"]
    P = pd.DataFrame({"mu_raw": score * 0.01}, index=score.index)
    market = {"Close": pd.DataFrame({"SPY": C.mean(axis=1)})}
    sic = pd.DataFrame({"ticker": C.columns, "sic": ["3571"] * (nt // 2) + ["6021"] * (nt - nt // 2)})
    return P, score, X, stocks, market, sic


def test_full_backtest_fills_next_open_and_leaves_a_checkpoint_and_report(tmp_path):
    P, score, X, stocks, market, sic = full_inputs()
    rec = {"ckpt_root": tmp_path / "ck", "report_dir": tmp_path / "rep"}
    eq, logs = BT.run(P, score, X, stocks, market, sic, n_scen=200, log=lambda *a: None, min_dv=0, record=rec)
    assert BT.run.stats["fill_audit"] == "PASS" and len(BT.run.fills.fills) > 0
    out = BT.run.record
    assert "error" not in out, out
    v = checkpoint.verify(out["bundle"])
    assert v["ok"], v
    rep = json.loads(Path(out["report_json"]).read_text())
    assert rep["gates"]["time_fence"] is True and rep["run_id"] == out["run_id"]
    assert rep["decision"] in ("CONTINUE TESTING", "REJECT")                 # thin synthetic evidence never adopts
    # B12 is the hook under test: without record nothing is written
    eq2, _ = BT.run(P, score, X, stocks, market, sic, n_scen=200, log=lambda *a: None, min_dv=0, record=False)
    assert BT.run.record is None
    assert len(list((tmp_path / "ck").iterdir())) == 1


# ================================================================================================= B01 training
def train_world(n=220, nt=40, seed=2):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-07", periods=n)
    tk = [f"S{i:02d}" for i in range(nt)]
    mi = pd.MultiIndex.from_product([idx, tk], names=["date", "ticker"])
    X = pd.DataFrame({"a": rng.normal(0, 1, len(mi)), "b": rng.normal(0, 1, len(mi)), "c": rng.normal(0, 1, len(mi)),
                      "m_vix": np.repeat(rng.normal(18, 2, n), nt)}, index=mi)
    f = pd.Series(0.02 * X["a"].values + rng.normal(0, 0.03, len(mi)), index=mi)
    y = pd.Series(np.sign(f.values) * (np.abs(f.values) > 0.03), index=mi, dtype=float)
    return X, y, f


def test_training_rows_purge_every_row_whose_label_closes_after_the_cut():
    X, y, f = train_world()
    ud = X.index.get_level_values(0).unique()
    end = ud[150]
    R = X.copy()
    Rt, yt, ft, info = TR.training_rows(R, y, f, end, sample_every=1)
    last = Rt.index.get_level_values(0).max()
    assert last == ud[150 - TR.LABEL_HORIZON]                                # label of the last row closes exactly at the cut
    assert info["purged_rows"] == TR.LABEL_HORIZON * 40 and info["train_rows"] == len(Rt)
    # the old ad-hoc cut kept these rows: their labels close inside what comes after the cut
    old = R.index.get_level_values(0) <= end
    leaked = R.index[old].difference(Rt.index)
    lc = pit.label_close_dates(leaked.get_level_values(0), TR.LABEL_HORIZON, pit.Calendar(ud))
    assert len(leaked) == 200 and (lc > end).all()
    assert ft.index.equals(Rt.index)


def test_fit_trains_only_on_purged_rows(monkeypatch):
    X, y, f = train_world()
    seen = {}

    def capture(R, yy, ff, fast=False):
        seen["R"] = R
        return {"reg": None}
    monkeypatch.setattr(TR.model, "fit_models", capture)
    end = X.index.get_level_values(0).unique()[120]
    TR._fit(X, y, f, end=end)
    ud = X.index.get_level_values(0).unique()
    lc = pit.label_close_dates(seen["R"].index.get_level_values(0), TR.LABEL_HORIZON, pit.Calendar(ud))
    assert (lc <= end).all() and TR._fit.last_rows["purged_rows"] > 0


def scramble_frames(n=200, nt=30, seed=3):
    X, y, f = train_world(n, nt, seed)
    rng = np.random.default_rng(seed)
    idx = X.index.get_level_values(0).unique()
    tk = X.index.get_level_values(1).unique()
    C = pd.DataFrame(20 * np.exp(np.cumsum(rng.normal(0, 0.02, (len(idx), len(tk))), axis=0)), index=idx, columns=tk)
    O = C.shift(1).fillna(C.iloc[0]) * np.exp(rng.normal(0, 0.01, C.shape))
    return {"X": X, "L": pd.DataFrame({"y": y, "f": f}), "closes": C, "opens": O}


def _friday(fr, k=150):
    ud = fr["X"].index.get_level_values(0).unique()
    return [d for d in ud[:k] if d.weekday() == 4][-1]


def test_future_scramble_passes_over_the_real_model_and_pattern_miner():
    fr = scramble_frames()
    rep = TR.future_scramble_gate(fr, _friday(fr), components=("model", "miner"))
    assert rep.passed, rep.summary()
    assert set(rep.variants) == {"truncated", "scrambled"}


def test_future_scramble_passes_over_the_real_memory_and_adaptive_session():
    fr = scramble_frames(n=130, nt=10)
    rep = TR.future_scramble_gate(fr, _friday(fr, 110), components=("memory", "adaptive"))
    assert rep.passed, rep.summary()
    fp = rep.detail["fingerprints"]
    assert fp["real"] == fp["scrambled"] == fp["truncated"]


def test_future_scramble_fails_closed_when_the_purge_is_removed(monkeypatch):
    """Hook removed: training rows cut by date only (the pre-B01 code). Labels that close after as_of now leak in."""
    fr = scramble_frames()

    def unpurged(R, y, f, end, sample_every=2, horizon=5, calendar=None):
        d = R.index.get_level_values(0)
        ok = np.asarray((d <= pd.Timestamp(end)) & y.notna().values & f.notna().values)
        return R[ok], y[ok], f[ok], {"train_rows": int(ok.sum())}
    monkeypatch.setattr(TR, "training_rows", unpurged)
    rep = TR.future_scramble_gate(fr, _friday(fr), components=("model",), require=False)
    assert not rep.passed and not rep.errors                                  # a real difference, not a crash
    assert any("pred" in x for x in rep.variants["scrambled"])
    with pytest.raises(pit.FailClosed):
        TR.future_scramble_gate(fr, _friday(fr), components=("model",))


def test_future_scramble_refuses_empty_or_unknown_components():
    fr = scramble_frames(n=60, nt=8)
    with pytest.raises(ValueError):
        TR.future_scramble_gate(fr, _friday(fr, 50), components=())
    with pytest.raises(ValueError):
        TR.future_scramble_gate(fr, _friday(fr, 50), components=("oracle",))


# ================================================================================================= B07 trust / direction
def trust_world(seed=4):
    stocks, X, score = world(n=200, nt=30, seed=seed)
    C = stocks["Close"]
    fwd = (C.shift(-5) / C - 1).stack(future_stack=True).reindex(X.index)
    X = X.copy()
    X["sig"] = fwd.fillna(0).values * 10 + np.random.default_rng(seed).normal(0, 0.05, len(X))   # a predictive indicator
    info = pd.DataFrame({"sic": 3571}, index=X.index)
    return stocks, X, score, fwd, info


def test_hooks_off_is_byte_identical_to_no_hooks():
    stocks, X, score = world()
    eq_none, st_none = topk(stocks, X, score)
    fills_none = [dict(f) for f in BT.run_topk.fills.fills]
    eq_off, st_off = topk(stocks, X, score, hooks=BT.ScoreHooks())
    assert eq_none.to_numpy().tobytes() == eq_off.to_numpy().tobytes()
    assert fills_none == BT.run_topk.fills.fills and BT.run_topk.hooks_log == []


def test_trust_on_changes_the_decisions():
    stocks, X, score, fwd, info = trust_world()
    now = X.index.get_level_values(0).unique()[100]
    h = BT.fit_score_hooks(X, fwd, now, trust=True, info=info, indicators=["sig"], trust_weight=1.0,
                           trust_kw={"min_weeks": 8, "min_obs": 100})
    assert h.trust_on and (h.trust.table["reliable"]).any()
    eq_off, _ = topk(stocks, X, score, start=now)
    picks_off = {(f["decision_date"], f["ticker"]) for f in BT.run_topk.fills.fills}
    eq_on, _ = topk(stocks, X, score, start=now, hooks=h)
    picks_on = {(f["decision_date"], f["ticker"]) for f in BT.run_topk.fills.fills}
    assert picks_on != picks_off and len(h.log) > 0 and all(n["trust_nonzero"] > 0 for n in h.log)


def test_a_trust_table_fitted_after_the_decision_date_is_look_ahead():
    stocks, X, score, fwd, info = trust_world()
    ud = X.index.get_level_values(0).unique()
    h = BT.fit_score_hooks(X, fwd, ud[150], trust=True, info=info, indicators=["sig"], trust_kw={"min_weeks": 8, "min_obs": 100})
    with pytest.raises(pit.LookAheadError):
        topk(stocks, X, score, start=ud[100], hooks=h)


def direction_inputs(X, fwd, strength):
    rng = np.random.default_rng(9)
    up = (fwd > 0).astype(float).where(fwd.notna())
    pattern = np.where(up.fillna(0.5).values > 0.5, 1.0, -1.0) * strength + rng.normal(0, 1, len(X))
    from engine.direction import build_inputs
    F = build_inputs(pattern=pd.Series(pattern, index=X.index), trust=pd.Series(1.0, index=X.index),
                     model=pd.Series(rng.normal(0, 1, len(X)), index=X.index))
    return F, up


def test_direction_on_trades_only_rows_the_engine_bets_up_on():
    stocks, X, score, fwd, info = trust_world()
    now = X.index.get_level_values(0).unique()[120]
    F, up = direction_inputs(X, fwd, strength=3.0)
    h = BT.fit_score_hooks(X, fwd, now, direction=True, direction_inputs=F, up=up, movers=pd.Series(True, index=X.index))
    assert h.direction.open, h.direction.reason
    eq_on, _ = topk(stocks, X, score, start=now, hooks=h)
    fills = BT.run_topk.fills.fills
    buys = [f for f in fills if f["shares"] > 0]
    assert buys
    for f in buys:
        d = pd.Timestamp(f["decision_date"])
        dec = h.direction.decide(F.xs(d, level=0).loc[[f["ticker"]]])
        assert bool(dec["bet"].iloc[0]) and dec["side"].iloc[0] == 1
    eq_off, _ = topk(stocks, X, score, start=now)
    assert not np.array_equal(eq_on.values, eq_off.values)


def test_a_closed_direction_engine_abstains_so_nothing_trades():
    """No measured edge (the real case: 51.8%) -> the engine stays closed -> ON trades nothing. OFF is unaffected."""
    stocks, X, score, fwd, info = trust_world()
    now = X.index.get_level_values(0).unique()[120]
    F, up = direction_inputs(X, fwd, strength=0.0)
    h = BT.fit_score_hooks(X, fwd, now, direction=True, direction_inputs=F, up=up, movers=pd.Series(True, index=X.index))
    assert not h.direction.open
    eq, st = topk(stocks, X, score, start=now, hooks=h)
    assert BT.run_topk.fills.fills == [] and (eq == eq.iloc[0]).all()


def test_hook_flags_without_fitted_objects_are_refused():
    with pytest.raises(ValueError):
        BT.ScoreHooks(trust_on=True).validate()
    with pytest.raises(ValueError):
        BT.ScoreHooks(direction_on=True).validate()
    assert not BT.ScoreHooks().validate().active


# ================================================================================================= B12 record / experiment memory
def test_record_major_run_writes_a_verified_bundle_and_a_report_and_never_overwrites(tmp_path):
    kw = dict(ckpt_root=tmp_path / "ck", report_dir=tmp_path / "rep")
    wk = np.array([0.01, -0.02, 0.03])
    a = BT.record_major_run("unit", "x/y", {"k": 3}, {"m": float("nan")}, {"s": 1}, 7, weekly=wk, **kw)
    b = BT.record_major_run("unit", "x/y", {"k": 3}, {"m": 1.0}, {"s": 1}, 7, weekly=wk, **kw)
    assert a["run_id"] != b["run_id"] and "/" not in a["run_id"]
    assert checkpoint.verify(a["bundle"])["ok"] and Path(a["report_json"]).exists()
    assert json.loads((Path(a["bundle"]) / "metrics.json").read_text())["m"] is None    # NaN is never written as a number
    with pytest.raises(checkpoint.CheckpointError):                          # bundles are write-once
        checkpoint.write_checkpoint(tmp_path / "ck", a["run_id"], {}, {}, {"s": 1}, {}, "now")


def test_research_loop_records_its_run(tmp_path):
    import research_loop as RL
    root = tmp_path / "loop"
    root.mkdir()
    summary = {"cycles": 1, "counters": {"experiments": 3}}
    (root / "summary.json").write_text(json.dumps(summary))
    a = argparse.Namespace(run_id="unit", seed=5, checkpoint_root=str(tmp_path / "ck"), report_dir=str(tmp_path / "rep"))
    out = RL.record_run(a, summary, root)
    assert "error" not in out, out
    assert checkpoint.verify(out["bundle"])["ok"] and json.loads(Path(out["report_json"]).read_text())["decision"] == "CONTINUE TESTING"
    assert json.loads((root / "run_record.json").read_text())["run_id"] == out["run_id"]


class _Proc:
    def __init__(self, rc):
        self.returncode = rc

    def poll(self):
        return self.returncode


def _fake_launcher(root, fail=()):
    launched = []

    def launch(tag, w, v, r):
        launched.append((tag, w))
        if (tag, w) in fail:
            return _Proc(1)
        p = Path(r) / "state" / "movers" / f"{w}{tag}"
        p.mkdir(parents=True, exist_ok=True)
        (p / "result.json").write_text("{}")
        return _Proc(0)
    return launch, launched


def test_grid_runner_skips_an_exact_repeat_and_retries_a_crash(tmp_path, capsys):
    import grid_runner as G
    idx = EM.launch_index("movers_grid", root=tmp_path / "tried")
    q1 = tmp_path / "q1.json"
    q1.write_text(json.dumps([{"tag": "_a", "variant": {"x": 1}}, {"tag": "_b", "variant": {"x": 1}}]))   # planted repeat
    launch, launched = _fake_launcher(tmp_path, fail={("_a", "m03")})
    out = G.main([str(q1), "4", "0"], launcher=launch, free=lambda: 99.0, index=idx, root=tmp_path, poll_s=0)
    assert {t for t, _ in launched} == {"_a"} and len(launched) == 13
    assert len(out["skipped"]) == 13 and all(s["why"] == "repeat within this batch" for s in out["skipped"])
    assert len(idx.rows) == 12                                                # the crashed window is not registered
    # a new tag with the same variant: every finished window is an exact repeat; the crashed one is retried
    q2 = tmp_path / "q2.json"
    q2.write_text(json.dumps([{"tag": "_c", "variant": {"x": 1}}, {"tag": "_d", "variant": {"x": 2}}]))
    launch2, launched2 = _fake_launcher(tmp_path)
    out2 = G.main([str(q2), "4", "0"], launcher=launch2, free=lambda: 99.0, index=EM.launch_index("movers_grid", root=tmp_path / "tried"),
                  root=tmp_path, poll_s=0)
    assert sorted(w for t, w in launched2 if t == "_c") == ["m03"] and sum(t == "_d" for t, _ in launched2) == 13
    assert len(out2["skipped"]) == 12


def test_filter_batch_on_an_empty_batch_and_a_novel_candidate(tmp_path):
    idx = EM.launch_index("unit", root=tmp_path)
    assert EM.filter_batch(idx, [], log=lambda *a: None) == ([], [])
    go, skip = EM.filter_batch(idx, [("e1", {"a": 1})], log=lambda *a: None)
    assert go == [("e1", {"a": 1})] and skip == []
    EM.record_launch(idx, "e1", {"a": 1})
    assert EM.record_launch(idx, "e1", {"a": 1}) is None                      # resume after a kill: no duplicate row
    assert not EM.prelaunch(idx, {"a": 1})["launch"] and EM.prelaunch(idx, {"a": 2})["launch"]
    with pytest.raises(ValueError):
        EM.launch_index("../escape", root=tmp_path)


def test_learner_search_reuses_a_finished_pair_only_for_the_same_code(tmp_path):
    import learner_search as LS
    idx = EM.launch_index("learner_search", root=tmp_path)
    we, be = {"id": "w1"}, {"id": "w9"}
    cfg = LS.pair_cfg("lessons", we, be, 1, 8, "code-A")
    assert LS.gated_pair(idx, cfg, "code-A") is None                          # novel -> run it
    f = tmp_path / "pair.json"
    f.write_text(json.dumps({"_code": "code-A", "delta": 0.1}))
    EM.record_launch(idx, "learner_search:t1:lessons:w1:w9", cfg, reason=str(f))
    assert LS.gated_pair(idx, cfg, "code-A") == {"_code": "code-A", "delta": 0.1}
    assert LS.gated_pair(idx, LS.pair_cfg("lessons", we, be, 1, 8, "code-B"), "code-B") is None
    f.unlink()
    assert LS.gated_pair(idx, cfg, "code-A") is None                          # the prior file is gone: run again


# ================================================================================================= reachability
def test_reachability_of_each_hook_is_what_the_report_says():
    import reachability as R
    hooks = [R.HookSpec("B12.research_loop.record", "F08", "engine.backtest:record_major_run", callers=("scripts.research_loop",)),
             R.HookSpec("B01.backtest.fills", "F08", "engine.backtest:Fills.order"),
             R.HookSpec("B01.backtest.audit", "F08", "engine.fill_audit:audit_session", callers=("engine.backtest",)),
             R.HookSpec("B07.score_hooks", "F08", "engine.backtest:ScoreHooks.apply"),
             R.HookSpec("B12.grid.prelaunch", "F08", "engine.experiment_memory:filter_batch", callers=("scripts.grid_runner",)),
             R.HookSpec("B12.learner_search.prelaunch", "F08", "engine.experiment_memory:prelaunch", callers=("scripts.learner_search",)),
             R.HookSpec("B01.train.purge", "F08", "engine.pit:purged_training_set", callers=("engine.train",)),
             R.HookSpec("B01.train.scramble", "F08", "engine.train:future_scramble_gate")]
    got = {v.name: v.status for v in R.Checker(hooks=hooks).hook_verdicts()}
    assert got["B12.research_loop.record"] == "REACHED"
    assert all(s in ("REACHED", "RESEARCH-ONLY") for s in got.values()), got    # nothing is dead
