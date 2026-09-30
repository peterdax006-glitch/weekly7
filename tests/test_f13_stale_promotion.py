"""F13 (C69 sections 12-15, 27, 28, 31; C68 forward-only regimes, self-correction, rollback; ledger W-06): a fix is never promoted on
evidence that predates a change, a change detected after a promotion re-opens it, and a fix that stops working is rolled back within a
stated bound. Planted data only (C63); every test runs in seconds.

The defect F11 found (flip seed 0): the planted error stopped at 80% of the sample and the sector-slope fix was promoted eleven weeks
later on OOS gains ~90% of which predated the stop. Here the same thing is planted in a correction frame (the SEC3 slope stops being
weak at `flip`), gated through the real quality gate, and must be held."""
from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pandas as pd
import pytest

from engine.learning.core import FirewallBreach, Provenance
from engine.research import change_points as CP
from engine.research import error_loop as EL
from engine.research import quality_gate as QG
from engine.research import self_correct as S
from engine.research.core import GateVerdict

warnings.filterwarnings("ignore")
FEATS = ["f_vol20", "f_atr", "f_r20", "f_r5"]
POL = QG.QualityPolicy(code_hash="f13")
FEC = S.FixEvidenceConfig(retirement_trigger=True)
RC = S.RegimeConfig()


def frame(seed: int = 1, weeks: int = 200, n: int = 40, weak: float = 0.08, flip: int | None = None, after: float = 0.6,
          start: str = "2017-01-06") -> pd.DataFrame:
    """Matured correction frame: realised = 0.01 + slope * r20 + noise; slope 0.25 everywhere except SEC3, where it is `weak` until
    week `flip` (None = never flips) and `after` from then on (0.6: the give-back turns into continuation - a strong weak slope
    would be quarantined by the leak firewall's implausible-IC cap, so the size of the change is planted on the far side). The
    incumbent keeps the POOLED slope, so after the flip the fix's SEC3 correction points the wrong way."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, periods=weeks, freq="W-FRI")
    sec = np.array([f"SEC{j % 5}" for j in range(n)])
    pooled = 0.8 * 0.25 + 0.2 * weak
    out = []
    for w, d in enumerate(days):
        r20 = rng.normal(0.02, 0.1, n)
        vol = rng.uniform(0.01, 0.03, n)
        w_now = weak if flip is None or w < flip else after
        slope = np.where(sec == "SEC3", w_now, 0.25)
        real = 0.01 + slope * r20 + rng.normal(0, 0.03, n)
        pred = 0.01 + pooled * r20 + rng.normal(0, 0.002, n)
        out.append(pd.DataFrame({"date": str(d.date()), "matured_at": str((d + pd.Timedelta(days=7)).date()), "predicted": pred,
                                 "realised": real, "ticker": [f"T{j:02d}" for j in range(n)], "sector": sec, "f_r20": r20, "f_vol20": vol,
                                 "f_atr": vol * 1.1, "f_r5": rng.normal(0, 0.03, n), "m_vol": 0.006}))
    return pd.concat(out, ignore_index=True)


def gate(fr: pd.DataFrame, now: str, changes=()) -> S.CorrectionReport:
    fr = S.as_of(fr, now)
    last = str(fr["matured_at"].max())
    base = QG.QualityEvidence(provenance=Provenance("2026-09-30T00:00:00", last, "f13", "d", "cfg", "exp", "run", 0, last))
    return S.step(fr, [S.slope_fix("sector_slope", FEATS, "sector", 5)], now, base=base, policy=POL, evidence=FEC, changes=changes)


def week(i: int, start: str = "2015-01-02") -> str:
    return str((pd.Timestamp(start) + pd.Timedelta(weeks=i)).date())


def series(values) -> tuple[list[str], np.ndarray]:
    v = np.asarray(values, float)
    return [week(i) for i in range(len(v))], v


# ============================================================================================================ the promotion gate
def test_the_f11_defect_planted_a_fix_is_not_promoted_on_evidence_that_predates_the_flip():
    """The SEC3 error stops at week 180; at week 190 the whole-window OOS evidence still says PROMOTE (the F11 defect) - the regime
    check holds it (STALE / TURNING / FAILS) and says why; the same frame without the flip still promotes at the same date."""
    now = week(191, "2017-01-06")
    ok = gate(frame(1), now)
    assert ok.promoted == ("sector_slope",), ok.rejected
    assert ok.regime["sector_slope"].verdict == "NO_CHANGE"
    rep = gate(frame(1, flip=180), now)
    d, chk = rep.decisions[0], rep.regime["sector_slope"]
    assert not rep.promoted and S.REGIME_GATE in d.blocking, (d.verdict, d.blocking, chk.reason)
    assert chk.verdict in ("STALE", "TURNING", "FAILS") and d.verdict in (GateVerdict.NEEDS_MORE_EVIDENCE, GateVerdict.FAILED)
    assert rep.results[0].mean_effect > 0                          # the whole-window mean alone would still have promoted it


def test_after_the_flip_the_fix_is_never_promoted_while_it_is_wrong():
    """Every monthly decision from 4 weeks after the flip until the frame ends: never PROMOTE (the post-change weeks do not show the
    gain). Honest limit: a change one or two weeks old is invisible to any detector - the re-validation after promotion covers it."""
    fr = frame(2, weeks=230, flip=170)
    verdicts = []
    for w in range(174, 228, 4):
        rep = gate(fr, week(w + 2, "2017-01-06"))
        verdicts.append((w, rep.regime["sector_slope"].verdict, rep.decisions[0].verdict.value))
        assert not rep.promoted, verdicts
    assert any(v in ("STALE", "FAILS", "TURNING") for _, v, _ in verdicts)


def test_a_regime_memory_change_inside_the_window_makes_the_evidence_stale_until_it_holds_after_it():
    """An external change (the loop's regime memory) 3 weeks before now: STALE; the same change 40 weeks back on a fix that keeps
    working: HOLDS (post-change weeks alone pass the screen); a change older than the evidence is not a change inside it."""
    d, v = series(np.random.default_rng(0).normal(1.0, 1.0, 100))
    now = week(101)
    recent = S.ChangeMark("regime_memory:MARKET_WIDE", d[97], d[99])
    chk = S.regime_check("f", d, v, now, [recent])
    assert chk.verdict == "STALE" and chk.regime_start == d[97] and chk.post_weeks == 3 and chk.stale_share > 0.9
    old = S.ChangeMark("regime_memory:MARKET_WIDE", d[60], d[62])
    assert S.regime_check("f", d, v, now, [old]).verdict == "HOLDS"
    before = S.ChangeMark("regime_memory:MARKET_WIDE", week(-10), week(-8))
    assert S.regime_check("f", d, v, now, [before]).verdict == "NO_CHANGE"
    late = S.ChangeMark("regime_memory:MARKET_WIDE", d[97], week(101))            # declared at now: not known at the decision
    assert S.regime_check("f", d, v, now, [late]).verdict == "NO_CHANGE"


def test_the_forward_cusum_finds_the_change_and_fails_a_fix_that_stopped_working():
    rng = np.random.default_rng(1)
    d, v = series(np.concatenate([rng.normal(0.8, 1.0, 80), rng.normal(-1.5, 1.0, 20)]))
    chk = S.regime_check("f", d, v, week(100))
    assert any(m.source == "fix_gain" and m.direction == -1 for m in chk.changes)
    assert chk.verdict in ("FAILS", "STALE") and not chk.ok and d.index(chk.regime_start) >= 75
    assert chk.gate_verdict in (GateVerdict.FAILED, GateVerdict.NEEDS_MORE_EVIDENCE)


def test_a_turn_too_recent_for_the_cusum_is_caught_by_the_tail_test_and_noise_is_not():
    rng = np.random.default_rng(3)
    base = rng.normal(0.8, 1.0, 94)
    d, v = series(np.concatenate([base, np.full(6, -3.0)]))
    chk = S.regime_check("f", d, v, week(100))
    assert chk.verdict in ("TURNING", "STALE", "FAILS") and not chk.ok
    held = 0
    for s in range(40):                                                          # null: a stationary gain series is (almost) never held
        d, v = series(np.random.default_rng(100 + s).normal(0.8, 1.0, 100))
        held += not S.regime_check("f", d, v, week(100)).ok
    assert held <= 3, held


def test_the_regime_check_only_ever_makes_a_verdict_stricter():
    def dec(v: GateVerdict) -> QG.QualityDecision:
        return QG.QualityDecision("fix:x", "2020-01-01", v, (), ("out_of_sample",) if v != GateVerdict.PROMOTE else (), (), (), "p", "e", "c")
    stale = S.RegimeCheck("x", "2020-01-01", "STALE", "why", 50, (), "2019-12-01", 3, 0.1, 0.5, 0.0, 0.9)
    fails = dataclasses.replace(stale, verdict="FAILS")
    fine = dataclasses.replace(stale, verdict="HOLDS")
    assert S.apply_regime(dec(GateVerdict.PROMOTE), stale).verdict == GateVerdict.NEEDS_MORE_EVIDENCE
    assert S.apply_regime(dec(GateVerdict.PROMOTE), fails).verdict == GateVerdict.FAILED
    assert S.apply_regime(dec(GateVerdict.NEEDS_MORE_EVIDENCE), fails).verdict == GateVerdict.FAILED
    for v in (GateVerdict.FAILED, GateVerdict.QUARANTINED, GateVerdict.UNKNOWN):
        assert S.apply_regime(dec(v), stale).verdict == v
    p = S.apply_regime(dec(GateVerdict.PROMOTE), fine)
    assert p.verdict == GateVerdict.PROMOTE and p.outcome(S.REGIME_GATE).state == QG.PASS and not p.blocking


def test_regime_config_refuses_a_lowered_bar():
    assert S.RegimeConfig().validate() == []
    assert S.RegimeConfig(screen_t=1.0).validate()                              # below the OOS screen: a lowered gate
    assert S.RegimeConfig(tail_t=0.5).validate() and S.RegimeConfig(reval_min_weeks=20, reval_max_weeks=13).validate()
    with pytest.raises(ValueError):
        S.regime_check("f", *series([1.0] * 30), week(40), cfg=S.RegimeConfig(min_post_weeks=1))


# ============================================================================================================ forward only
def test_change_detection_on_fix_gains_is_forward_only():
    """The detections up to week t are identical when every week after t is scrambled (shuffled, sign-flipped, replaced by noise)."""
    rng = np.random.default_rng(5)
    d, v = series(np.concatenate([rng.normal(0.8, 1.0, 60), rng.normal(-1.0, 1.0, 40)]))
    full = S.gain_changes(d, v)
    for t in (40, 65, 80):
        for mode in ("shuffle", "negate", "noise"):
            w = v.copy()
            tail = w[t + 1:]
            w[t + 1:] = rng.permutation(tail) if mode == "shuffle" else -tail if mode == "negate" else rng.normal(5, 3, len(tail))
            got = [m for m in S.gain_changes(d, w) if m.alarm_date <= d[t]]
            assert got == [m for m in full if m.alarm_date <= d[t]], (t, mode)
        chk_a = S.regime_check("f", d[: t + 1], v[: t + 1], week(t + 1))
        chk_b = S.regime_check("f", d[: t + 1], v[: t + 1], week(t + 1), [S.ChangeMark("x", d[0], week(t + 5))])   # declared later
        assert repr(chk_a.to_dict()) == repr(chk_b.to_dict())


def test_evidence_dated_at_or_after_now_is_refused():
    d, v = series([1.0] * 30)
    with pytest.raises(FirewallBreach):
        S.regime_check("f", d, v, d[-1])
    with pytest.raises(FirewallBreach):
        S.revalidate(d[:20], v[:20], d[20], d[20:], v[20:], d[-1])


# ============================================================================================================ re-validation and rollback
def test_a_change_after_promotion_triggers_revalidation_and_rolls_back_a_fix_that_stopped():
    rng = np.random.default_rng(7)
    ed, ev = series(rng.normal(0.8, 1.0, 70))
    since = week(70)
    live = np.concatenate([rng.normal(0.8, 1.0, 10), rng.normal(-1.5, 1.0, 20)])
    ld = [week(71 + i) for i in range(len(live))]
    states = []
    for j in range(1, len(live) + 1):
        r = S.revalidate(ed, ev, since, ld[:j], live[:j], week(72 + j))
        states.append(r.state)
        if r.rollback:
            break
    assert states[-1] == "ROLLBACK" and "WATCHING" in states[:10], states
    delay = len(states) - 10                                                    # weeks from the change to the rollback
    assert 0 < delay <= RC.reval_max_weeks + 10, delay


def test_a_fix_that_keeps_working_is_not_rolled_back_and_a_null_change_revalidates():
    rng = np.random.default_rng(8)
    ed, ev = series(rng.normal(0.8, 1.0, 70))
    ld = [week(71 + i) for i in range(40)]
    lv = rng.normal(0.8, 1.0, 40)
    assert not any(S.revalidate(ed, ev, week(70), ld[:j], lv[:j], week(72 + j)).rollback for j in range(1, 41))
    ext = [S.ChangeMark("regime_memory:MARKET_WIDE", ld[5], ld[7])]              # a market change that does not touch this fix
    r = S.revalidate(ed, ev, week(70), ld[:30], lv[:30], week(102), ext)
    assert r.state == "REVALIDATED" and r.change == ext[0]
    early = S.revalidate(ed, ev, week(70), ld[:9], lv[:9], week(81), ext)
    assert early.state == "REVALIDATING" and early.post_weeks == 4


def test_a_fix_that_never_re_validates_is_rolled_back_at_the_stated_bound():
    """Post-change gains hover just above zero: never t >= 1.64, never mean <= 0 -> ROLLBACK exactly at reval_max_weeks."""
    ed, ev = series(np.random.default_rng(9).normal(0.8, 1.0, 70))
    ld = [week(71 + i) for i in range(20)]
    lv = np.tile([0.3, -0.25], 10)                                               # mean 0.025 > 0, t ~ 0.3
    ext = [S.ChangeMark("regime_memory:SECTOR", ld[0], ld[1])]
    got = [S.revalidate(ed, ev, week(70), ld[:j], lv[:j], week(72 + j), ext) for j in range(2, 21)]
    first = next(r for r in got if r.rollback)
    assert first.post_weeks == RC.reval_max_weeks and "within" in first.reason


def test_empty_and_degenerate_inputs():
    empty = S.regime_check("f", [], [], "2020-01-01")
    assert empty.verdict == "NO_CHANGE" and empty.n_weeks == 0 and empty.post_weeks == 0
    assert S.revalidate([], [], "2020-01-01", [], [], "2020-02-01").state == "WATCHING"
    short = S.regime_check("f", *series([0.5, 0.4, 0.6]), week(5))               # too short for the CUSUM or the tail test
    assert short.verdict == "NO_CHANGE" and not S.gain_changes(*series([0.5, 0.4, 0.6]))
    with pytest.raises(ValueError):
        S.regime_check("f", ["2020-01-03"], [1.0, 2.0], "2020-02-01")


def test_rollback_delay_distribution_is_bounded_and_null_rollbacks_are_rare():
    pts = S.rollback_delay_curve(snr=0.8, shifts=(-3.0,), evidence_weeks=40, pre_weeks=4, max_weeks=30, n_sim=24, seed=1)
    p = pts[0]
    assert p.rollback_prob >= 0.9 and p.median_weeks is not None and p.p90_weeks <= 20, p
    assert p.false_rollback <= 0.2, p


# ============================================================================================================ the loop's monitor
def live_frame(since: str, weeks: int, flip_at: int | None, seed: int = 0) -> pd.DataFrame:
    """Matured rows since the promotion: production (predicted) beats the shadow by ~0.8 sd a week until `flip_at`, then loses."""
    rng = np.random.default_rng(seed)
    rows = []
    for w, d in enumerate(pd.date_range(since, periods=weeks, freq="W-FRI")):
        real = rng.normal(0.02, 0.04, 40)
        good = flip_at is None or w < flip_at
        shadow = real + rng.normal(0, 0.02, 40) * (1.6 if good else 1.0)
        pred = real + rng.normal(0, 0.02, 40) * (1.0 if good else 1.8)
        rows.append(pd.DataFrame({"date": str(d.date()), "matured_at": str((d + pd.Timedelta(days=7)).date()), "predicted": pred,
                                  "realised": real, "shadow": shadow}))
    return pd.concat(rows, ignore_index=True)


def test_the_loop_monitor_rolls_back_by_revalidation_and_reads_the_loops_change_points():
    since = "2019-01-04"
    g = EL.monitor_gains(live_frame(since, 40, None), since, 10 ** 6)
    ev_d = [str((pd.Timestamp(since) - pd.Timedelta(weeks=60 - i)).date()) for i in range(60)]
    ev_v = list(np.random.default_rng(1).normal(float(g.mean()), float(g.std()), 60))
    led = EL.Ledgers(None, None, None, None, None, EL.PipelineLedger())

    def run(fr: pd.DataFrame, now: str, changes=()) -> tuple[int, EL.C68State]:
        st = EL.new_state()
        st.production = {"name": "sector_slope", "since": since, "key": f"FIX:sector_slope@{since}", "window_share": 1.0,
                         "extra_features": ("r20_x_SEC3",), "evidence": (ev_d, ev_v)}
        if led.pipe.last_step(st.production["key"]) is None:
            led.pipe.add(st.production["key"], "OOS_TEST", "2018-12-28")
            led.pipe.add(st.production["key"], "PROMOTED", "2018-12-28")
        return EL._monitor(st, EL.SCX.as_of(fr, now), now, led, changes), st
    rolled, st = run(live_frame(since, 40, None), "2019-10-18")
    assert rolled == 0 and st.production["name"] == "sector_slope"
    rolled, st = run(live_frame(since, 40, 20), "2019-10-18")
    assert rolled == 1 and st.counters.get("rolled_back_revalidation") == 1 and st.monitoring[-1]["why"] == "revalidation"
    assert EL._retired(st)["sector_slope"] == "2019-10-18" and st.production["name"] == "incumbent"
    ext = [S.ChangeMark("regime_memory:MARKET_WIDE", "2019-03-01", "2019-03-08")]     # a declared change and a fix that holds after it
    rolled, st = run(live_frame(since, 40, None), "2019-10-18", ext)
    assert rolled == 0 and st.monitoring[-1]["revalidation"] == "REVALIDATED"


def test_loop_changes_read_the_regime_memory_records_declared_before_now():
    class Rec:
        def __init__(self, scope, cd, det):
            self.scope, self.change_date, self.detected_at = scope, cd, det

    class Mem:
        def records(self, now):
            return [r for r in (Rec("SECTOR", "2020-03-02", "2020-03-20"), Rec("MARKET_WIDE", "2020-06-01", "2020-06-19"))
                    if r.detected_at < str(now)]
    st = EL.new_state()
    st.memory = Mem()
    got = EL.loop_changes(st, "2020-05-01")
    assert got == [S.ChangeMark("regime_memory:SECTOR", "2020-03-02", "2020-03-20")]
    st.memory = None
    assert EL.loop_changes(st, "2020-05-01") == []


def test_promotion_stores_the_evidence_its_revalidation_is_judged_against():
    st = EL.new_state()
    r = S.FixResult("sector_slope", S.Component.SECTOR_CONDITIONING, "2019-01-01", (2018,), 100, 100, ("2019-02-01", "2019-02-08"),
                    (0.001, 0.002), 0.001, 0.0015, 3.0, ((1, 0.0015),), "dg", {"feature": "f_r20", "group": "SEC1"})
    assert EL._apply_promotions(st, ["sector_slope"], {"sector_slope": r}, "2019-03-01") == 1
    assert st.production["evidence"] == (["2019-02-01", "2019-02-08"], [0.001, 0.002])
    assert EL.C68Config().validate() == [] and EL.C68Config(regime=S.RegimeConfig(screen_t=1.0)).validate()
