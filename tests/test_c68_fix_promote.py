"""F11 (C69 ledger W-06, C68 PZ20 / PC11 / PC12 / PC16): a self-correction fix reaches PROMOTE on its OWN measured evidence, a null
twin's never does, a promoted fix that degrades is rolled back by the real monitor, and repeated confident errors escalate research
priority on a pattern group and on 'all'. Planted data only (C63).

Fast tests (seconds): synthetic correction frames through engine.research.self_correct and the real quality gate; the real
c68.error_research / questions.generate stages on planted error records. Long tests (W7_LONG_TESTS=1): the real research loop on the
planted C68 world (error_loop.plant_world with a weak sector) and its null twin, >= 6 seeds each - they call the same functions as
scripts/c68_fix_promote.py, which writes the rate tables under state/research/c68_fix_promote/."""
from __future__ import annotations

import copy
import dataclasses
import json
import os
import warnings

import numpy as np
import pandas as pd
import pytest

from engine.learning.core import Provenance
from engine.research import error_loop as EL
from engine.research import error_research as ER
from engine.research import feeds as FD
from engine.research import loop as LP
from engine.research import prediction_error as PE
from engine.research import quality_gate as QG
from engine.research import self_correct as S
from engine.research import two_stage as TS
from engine.research.core import GateVerdict

warnings.filterwarnings("ignore")
CLOCK = lambda: 1_700_000_000.0                                   # noqa: E731
LONG = os.environ.get("W7_LONG_TESTS") == "1"
FEATS = ["f_vol20", "f_atr", "f_r20", "f_r5"]


# ============================================================================================================ planted correction frames
def frame(planted: bool = True, seed: int = 0, weeks: int = 200, n: int = 40, weak: float = 0.08, start: str = "2017-01-06",
          memo: bool = False) -> pd.DataFrame:
    """A matured self-correction frame: realised gain = 0.01 + slope * r20 + noise, slope 0.25 everywhere except (planted) in SEC3
    where it is `weak`. The incumbent predicts with the POOLED least-squares slope, so its only error is the SEC3 slope. `memo` adds
    a per-name offset (a thing only a name-memoriser could 'learn')."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, periods=weeks, freq="W-FRI")
    mvol = 0.006 * np.exp(rng.normal(0, 0.3, weeks))
    sec = np.array([f"SEC{j % 5}" for j in range(n)])
    off = rng.normal(0, 0.03, n) if memo else np.zeros(n)
    out = []
    for w, d in enumerate(days):
        r20 = rng.normal(0.02, 0.1, n)
        vol = rng.uniform(0.01, 0.03, n)
        slope = np.where((sec == "SEC3") & planted, weak, 0.25)
        real = 0.01 + slope * r20 + off + rng.normal(0, 0.03, n)
        pooled = 0.8 * 0.25 + 0.2 * (weak if planted else 0.25)
        pred = 0.01 + pooled * r20 + rng.normal(0, 0.002, n)
        out.append(pd.DataFrame({"date": str(d.date()), "matured_at": str((d + pd.Timedelta(days=7)).date()), "predicted": pred,
                                 "realised": real, "ticker": [f"T{j:02d}" for j in range(n)], "sector": sec, "f_r20": r20, "f_vol20": vol,
                                 "f_atr": vol * 1.1, "f_r5": rng.normal(0, 0.03, n), "m_vol": mvol[w]}))
    return pd.concat(out, ignore_index=True)


def base_evidence(fr: pd.DataFrame) -> QG.QualityEvidence:
    last = str(fr["matured_at"].max())
    return QG.QualityEvidence(provenance=Provenance("2026-09-29T00:00:00", last, "f11", "d", "cfg", "exp", "run", 0, last))


NOW = "2021-01-08"
POL = QG.QualityPolicy(code_hash="f11")
FEC = S.FixEvidenceConfig(retirement_trigger=True)


def gate_one(fix: S.CandidateFix, fr: pd.DataFrame, now: str = NOW, fec: S.FixEvidenceConfig = FEC) -> S.CorrectionReport:
    fr = S.as_of(fr, now)
    return S.step(fr, [fix], now, base=base_evidence(fr), policy=POL, evidence=fec)


def slope() -> S.CandidateFix:
    return S.slope_fix("sector_slope", FEATS, "sector", 5)


# ============================================================================================================ the full evidence (fast)
def test_a_genuine_fix_reaches_promote_on_its_own_measured_evidence():
    """The planted SEC3 slope: the fix picks (r20, SEC3) on its training rows, beats the incumbent out of sample in unseen years, and
    every one of the twelve gates passes on MEASURED evidence - nothing forged, nothing waived."""
    rep = gate_one(slope(), frame(True, 1))
    d, b = rep.decisions[0], rep.bundles[0]
    assert rep.results[0].detail["group"] == "SEC3" and rep.results[0].detail["feature"] == "f_r20"
    assert b.full and not b.missing
    assert d.verdict == GateVerdict.PROMOTE, [(g.gate, g.state, g.detail[:120]) for g in d.gates if not g.ok]
    assert all(g.state in (QG.PASS, QG.NA) for g in d.gates) and d.outcome("calibration").state == QG.NA   # a point fix
    judged = gate_one(slope(), frame(True, 1), fec=dataclasses.replace(FEC, states_probabilities=True))   # ... and when its derived
    assert judged.decisions[0].outcome("calibration").state in (QG.PASS, QG.FAIL)                        # P(in band) is judged, it is
    assert b.parts["repl_status"] == "REPLICATED" and b.parts["planted_probe_caught"] and b.parts["abstains"]


def test_the_null_twin_is_never_promoted_and_says_why():
    """Same noise, no SEC3 slope: whatever the fix picks is noise - FAILED on out-of-sample (or at best not promoted), never PROMOTE."""
    for seed in range(4):
        rep = gate_one(slope(), frame(False, seed))
        assert not rep.promoted, (seed, rep.rejected)
        assert rep.decisions[0].verdict in (GateVerdict.FAILED, GateVerdict.NEEDS_MORE_EVIDENCE)


def test_the_evidence_ladder_spends_no_audit_on_a_fix_that_fails_its_oos_screen():
    rep = gate_one(slope(), frame(False, 3))
    b = rep.bundles[0]
    if rep.results[0].t < FEC.screen_t:
        assert not b.full and b.evidence.leak is None and rep.decisions[0].verdict == GateVerdict.FAILED
        assert "out_of_sample" in rep.decisions[0].blocking


def test_a_name_memorising_fix_is_quarantined_by_the_identity_harness():
    """A fix that learns each NAME's mean error (a lookup table) looks good out of sample on a world whose names carry fixed offsets,
    but its skill collapses once names are permuted: identity QUARANTINED (planted defect the harness must catch)."""
    def build(train, seed):
        m = (train["realised"] - train["predicted"]).groupby(train["ticker"]).mean()

        def predict(fr):
            return fr["predicted"].to_numpy(float) + fr["ticker"].map(m).fillna(0.0).to_numpy(float)
        return predict
    memo = S.CandidateFix("name_memory", S.Component.MISSING_FEATURE, (S.FixInput("ticker_mean"),), build)
    rep = gate_one(memo, frame(False, 2, memo=True))
    d, b = rep.decisions[0], rep.bundles[0]
    assert rep.results[0].t > 2 and b.full                                        # it does 'work' out of sample ...
    assert not d.promote and d.verdict == GateVerdict.QUARANTINED, d.blocking      # ... and is refused as a memoriser
    assert b.parts["lookup_share"] > 0.9


def test_a_fix_that_reads_the_outcome_is_quarantined_by_the_leak_audit():
    def build(train, seed):
        return lambda fr: fr["predicted"].to_numpy(float) + 0.8 * (fr["realised"] - fr["predicted"]).to_numpy(float)
    leaky = S.CandidateFix("peek", S.Component.MISSING_FEATURE, (S.FixInput("f_r20"),), build)
    rep = gate_one(leaky, frame(True, 0))
    d = rep.decisions[0]
    assert d.verdict == GateVerdict.QUARANTINED and "leakage" in d.blocking


def test_full_evidence_refuses_future_rows_and_bad_config():
    fr = frame(True, 0, weeks=120)
    with pytest.raises(Exception):
        S.step(fr, [slope()], "2018-06-01", base=base_evidence(fr), policy=POL, evidence=FEC)   # rows mature after now
    with pytest.raises(ValueError):
        S.full_fix_evidence(S.test_fix(slope(), S.as_of(fr, NOW), NOW), slope(), S.as_of(fr, NOW), NOW, 1, None, "f11", S.SelfCorrectConfig(),
                            S.FixEvidenceConfig(era="decade"))
    with pytest.raises(ValueError):
        S.test_fix(slope(), pd.DataFrame(columns=list(S.REQUIRED)), NOW)                     # the empty case


def test_slope_fix_is_a_pure_within_group_slope_and_abstains_without_inputs():
    fr = S.as_of(frame(True, 0, weeks=80), NOW)
    pred = slope().build(fr, 0)
    adj = pred(fr) - fr["predicted"].to_numpy(float)
    g = pred.detail["group"]
    inside = fr["sector"].to_numpy() == g
    assert np.allclose(adj[~inside], 0.0) and abs(adj[inside].mean()) < 1e-9          # no level change: a slope, nothing else
    with pytest.raises(KeyError):
        pred(fr.drop(columns=["f_r20"]))
    empty = S.slope_fix("x", ["f_r20"], "sector").build(fr.iloc[:5], 0)                  # too few rows: no change at all
    assert empty.detail == {} and np.array_equal(empty(fr), fr["predicted"].to_numpy(float))


# ============================================================================================================ monitor / rollback (fast)
def monitor_frame(gain_sign: float, weeks: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for w, d in enumerate(pd.date_range("2019-01-04", periods=weeks, freq="W-FRI")):
        real = rng.normal(0.02, 0.04, 40)
        shadow = real + rng.normal(0, 0.03, 40)
        pred = real + rng.normal(0, 0.03, 40) + gain_sign * 0.02 * np.sign(rng.normal(size=40))
        rows.append(pd.DataFrame({"date": str(d.date()), "matured_at": str((d + pd.Timedelta(days=7)).date()), "predicted": pred,
                                  "realised": real, "shadow": shadow}))
    return pd.concat(rows, ignore_index=True)


def test_the_monitor_rolls_back_a_degrading_fix_and_keeps_a_good_one():
    st = EL.new_state()
    led = EL.Ledgers(None, None, None, None, None, EL.PipelineLedger())
    for sign, rolled in ((0.0, 0), (1.0, 1)):
        st.production = {"name": "sector_slope", "since": "2019-01-01", "key": "FIX:sector_slope@2019-01-01", "window_share": 1.0,
                         "extra_features": ("r20_x_SEC3",)}
        if led.pipe.last_step(st.production["key"]) is None:
            led.pipe.add(st.production["key"], "OOS_TEST", "2018-12-28")
            led.pipe.add(st.production["key"], "PROMOTED", "2018-12-28")
        assert EL._monitor(st, monitor_frame(sign), "2019-09-01", led) == rolled
    assert st.production["name"] == "incumbent" and EL._retired(st)["sector_slope"] == "2019-09-01"
    assert [e["step"] for e in led.pipe.trail("FIX:sector_slope@2019-01-01")][-1] == "MONITORED"
    st.production = {"name": "x", "since": "2019-08-20", "window_share": 1.0, "extra_features": ()}
    assert EL._monitor(st, monitor_frame(1.0), "2019-09-01", led) == 0                    # too few post-promotion weeks: no verdict
    assert EL._monitor(st, monitor_frame(1.0).drop(columns=["shadow"]), "2019-09-01", led) == 0   # no shadow: nothing to compare


def test_promotion_applies_the_learned_interaction_and_never_a_control():
    st = EL.new_state()
    r = S.FixResult("sector_slope", S.Component.SECTOR_CONDITIONING, "2018-01-01", (2017,), 1, 1, (), (), 0.1, 0.001, 3.0, (), "",
                    {"feature": "f_r20", "group": "SEC3"})
    assert EL._apply_promotions(st, ["placebo_noise"], {}, "2019-01-04") == 0 and st.counters["placebo_promoted"] == 1
    assert EL._apply_promotions(st, ["sector_slope"], {"sector_slope": r}, "2019-01-04") == 1
    assert st.production["extra_features"] == ("r20_x_SEC3",) and st.production["name"] == "sector_slope"
    assert EL._apply_promotions(st, ["sector_slope"], {"sector_slope": r}, "2019-02-01") == 0      # already in production
    rows = pd.DataFrame({"r20": [0.1, 0.2], "sector": ["SEC3", "SEC1"]})
    assert EL.feature_matrix(rows, ("r20_x_SEC3",))[:, 0].tolist() == [0.1, 0.0]
    assert np.isnan(EL.feature_matrix(rows.drop(columns=["sector"]), ("r20_x_SEC3",))).all()      # absent input: NaN, never 0


# ============================================================================================================ escalation (PC11, PC12)
@pytest.fixture(scope="module")
def loop3(tmp_path_factory):
    EL.register()
    world = EL.plant_world(EL.C68Plant())
    keep = ("observe.panel", "evaluate.two_stage", "questions.generate")
    cfg = LP.LoopConfig(run_id="f11esc", free_gb=12.0, code_hash="f11", checkpoint="off", cadence={},
                        disabled=tuple(n for n in LP.BUILTIN_STAGES if n not in keep and n != "report.cycle"),
                        two_stage=TS.TwoStageConfig(gate_min_weeks=6, min_direction_rows=60, min_calib_rows=20))
    state, rt, _ = LP.open_loop(FD.WorldFeed(FD.InMemorySource(world), FD.FeedConfig(warm_weeks=50)), tmp_path_factory.mktemp("esc"), cfg,
                                clock=CLOCK)
    for _ in range(3):
        LP.step(state, rt)
    return state, rt


def records(kind: str, n: int = 36, seed: int = 0) -> list:
    """Planted matured error records. 'pattern': repeated confident over-predictions ('expected 8%, realised 2%') on ONE pattern while
    the other patterns are noise; 'all': the same over-prediction spread evenly over three patterns and three stock types (no single
    lens holds it: patterns, stock types, sector and market regimes all rotate); 'null': pure noise around the expectation."""
    rng = np.random.default_rng(seed)
    pats, types = ("mom_r20_top", "rev_r5_bottom", "vol_top"), ("vol_low", "vol_mid", "vol_high")
    out = []
    for i in range(n):
        d = pd.Timestamp("2016-12-20") - pd.Timedelta(days=2 * (n - i))              # every outcome matured before the loop's now
        p, s = pats[i % 3], types[(i // 3) % 3]
        bad = (kind == "all") or (kind == "pattern" and p == "mom_r20_top")
        real = 0.02 + rng.normal(0, 0.005) if bad else 0.08 + rng.normal(0, 0.02)
        sec, reg = ("UP_CALM", "FLAT_CALM", "UP_VOLATILE")[(i // 9) % 3], ("UP_CALM", "FLAT_CALM", "DOWN_CALM")[(i // 2) % 3]
        out.append(PE.ErrorRecord(f"X{kind}{i:03d}", str(d.date()), str((d + pd.Timedelta(days=7)).date()), 0.08, float(real), 0.8, 0.02, p,
                                  sec, reg, s, None))
    return out


def run_error_research(loop3, recs, monkeypatch) -> tuple:
    state, rt = loop3
    state = copy.deepcopy(state)
    st = state.modules["c68"]
    st.er = ER.new_state(st.cfg.research_cfg)
    st.cells, st.subject_pids = {}, {}
    state.bus = {}
    ctx = LP.Ctx(state, rt, state.now, state.cycle)
    led = EL._ledgers(ctx, st)
    monkeypatch.setattr(led.errors, "records", lambda outcomes, now: list(recs))
    EL.st_error_research(ctx)
    LP.st_generate(ctx)
    pri = {q.subject: q.priority for q in state.questions.values() if q.subject.startswith("prediction error")}
    return st, pri, st.er.book.escalated(state.now)


def test_repeated_confident_errors_on_a_pattern_escalate_under_the_pattern_lens(loop3, monkeypatch):
    st, pri, esc = run_error_research(loop3, records("pattern"), monkeypatch)
    lab = [e for e in esc if "pattern=mom_r20_top" in e.lenses]
    assert lab and lab[0].multiplier > 1.0 and lab[0].group.startswith("pattern=mom_r20_top"), [(e.group, e.aliases) for e in esc]
    assert st.er.book.escalated_lens("2099-01-01", "pattern")
    whole = [e.multiplier for e in esc if e.group == "all"]
    assert not whole or lab[0].multiplier > whole[0]                                # the error lives in the pattern, not the book
    _, null_pri, null_esc = run_error_research(loop3, records("null"), monkeypatch)
    assert all(e.multiplier < lab[0].multiplier for e in null_esc)                  # noise escalates less (see the null-rate test)
    assert max(pri, key=pri.get) == "prediction error pattern=mom_r20_top", sorted(pri.items(), key=lambda kv: -kv[1])[:3]
    assert pri["prediction error pattern=mom_r20_top"] > 10 * max(null_pri.values(), default=0.0)   # it reached the loop's own queue
    few, few_pri, few_esc = run_error_research(loop3, records("pattern", n=12), monkeypatch)
    assert not few_esc and "prediction error pattern=mom_r20_top" not in few_pri     # 4 repeats: not yet a pattern (PC12: it escalates)
    cw = [i for i in st.er.intensities.values() if i.confident_wrong]
    assert cw and len(st.er.investigations) == len(cw)                              # PC11: confident failures get the 15 questions


def test_repeated_errors_spread_over_every_lens_escalate_as_all(loop3, monkeypatch):
    st, pri, esc = run_error_research(loop3, records("all"), monkeypatch)
    top = next((e for e in esc if e.group == "all"), None)
    table = [(e.group, e.multiplier) for e in esc]
    assert top is not None and top.multiplier > 1.0 and top.multiplier >= max(m for _, m in table) - 1e-12, table
    assert "prediction error all" in pri
    _, null_pri, _ = run_error_research(loop3, records("null"), monkeypatch)
    assert max(pri, key=pri.get) == "prediction error all" and pri["prediction error all"] > 10 * max(null_pri.values(), default=0.0)
    _, few_pri, esc2 = run_error_research(loop3, records("all", n=12), monkeypatch)
    assert next(e for e in esc2 if e.group == "all").multiplier < top.multiplier     # more repetitions: more escalation ...
    assert few_pri["prediction error all"] < pri["prediction error all"]            # ... and a higher research priority


def test_identical_lenses_are_one_finding_named_by_the_most_actionable_lens():
    """Every error on one pattern AND one stock type: the pattern, stock-type, sector, regime and 'all' groups hold the same records.
    One finding - labelled by the pattern (before F11 the alphabet named it 'stock_type=...') - with the others as aliases."""
    book = ER.ErrorPatternBook()
    for i, r in enumerate(records("all")):
        book.add(ER.obs_from_record(dataclasses.replace(r, pattern="mom_r20_top", stock_type="vol_mid", sector="UP_CALM", regime="UP_CALM")),
                 "2099-01-01")
    esc = book.escalated("2099-01-01")
    assert len(esc) == 1, [(e.group, e.aliases) for e in esc]
    one = esc[0]                                                                      # the compound lens is the most specific name
    assert one.group == "pattern=mom_r20_top|regime=UP_CALM" and "pattern=mom_r20_top" == one.aliases[0]
    assert {"all", "stock_type=vol_mid", "sector=UP_CALM", "regime=UP_CALM"} <= set(one.aliases)
    assert ER.ErrorPatternBook().escalated("2099-01-01") == []                         # empty


def test_error_pattern_multiplicity_is_controlled_across_lenses():
    """Pure noise tested through ~25 lenses at once: before F11 17 of 40 books escalated some group (no control across groups); with
    Benjamini-Hochberg over the family (q = 0.10) about one book in ten may (measured: 10 of 100). The planted pattern and book-wide
    errors still escalate every time."""
    def n_escalated(kind: str) -> int:
        hit = 0
        for seed in range(40):
            b = ER.ErrorPatternBook(ER.ErrorConfig(window=24))
            for r in records(kind, seed=seed):
                b.add(ER.obs_from_record(r), "2099-01-01")
            hit += bool(b.escalated("2099-01-01"))
        return hit
    assert n_escalated("null") <= 6
    assert n_escalated("pattern") == 40 and n_escalated("all") == 40
    raw = ER.ErrorPatternBook(ER.ErrorConfig(window=24, fdr_q=0.999))       # the control is what removes them, not something else
    for r in records("null", seed=0):
        raw.add(ER.obs_from_record(r), "2099-01-01")
    assert raw.escalated("2099-01-01")


# ============================================================================================================ the real loop (long)
@pytest.mark.skipif(not LONG, reason="long real-loop runs: set W7_LONG_TESTS=1 (scripts/c68_fix_promote.py runs them in parallel)")
def test_long_planted_fix_is_promoted_and_the_null_twin_never(tmp_path):
    from scripts import c68_fix_promote as R
    rows = [R.run_one("planted", s, tmp_path / f"p{s}") for s in range(6)] + [R.run_one("null", s, tmp_path / f"n{s}") for s in range(6)]
    tab = pd.DataFrame(rows)
    assert (tab[tab.world == "null"]["promoted_sector_slope"] == 0).all(), tab
    assert tab[tab.world == "planted"]["promoted_sector_slope"].mean() >= 0.5, tab


@pytest.mark.skipif(not LONG, reason="long real-loop run: set W7_LONG_TESTS=1")
def test_long_a_promoted_fix_that_degrades_is_rolled_back(tmp_path):
    """The flip world: the planted give-back stops at 80% of the sample. Every seed whose fix was promoted BEFORE the flip must be
    rolled back by the real monitor after it (and not before), and must end retired with the incumbent in production."""
    from scripts import c68_fix_promote as R
    rows = [R.run_one("flip", s, tmp_path / f"f{s}") for s in range(4)]
    before = [r for r in rows if r["first_promote"] and r["first_promote"] < r["flip_date"]]
    assert before, rows
    for r in before:
        assert r["rolled_back"] >= 1 and r["final_production"] == "incumbent", r
        assert json.loads(r["retired"])["sector_slope"] >= r["flip_date"], r          # rolled back after the flip, not before
