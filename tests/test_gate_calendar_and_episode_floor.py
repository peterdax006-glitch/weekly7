"""F14 (C69 sections 12-14, 27, 28, 31; C66 section 42 quality gate; ledger W-06): the two structural issues F12 recorded.

(a) every December look FAILED the out-of-sample gate: the train window was the earliest 30% of a sliding frame wherever that fell, so
    in December it ran a few weeks past 1 January and made that whole calendar year 'seen' - one unseen year instead of two.
    evidence.train_cut now cuts at a calendar-year end, so the unseen-year count no longer depends on the month of the look.
(b) a very strong genuine effect could not collect 5 failure episodes in time and waited at NEEDS_MORE_EVIDENCE for ever.
    evidence.failure_floor accepts episodes OR exposure that bounds the failure rate as rare, at the look's spent alpha.
Every mechanism has a planted case it must catch (these tests fail on the pre-F14 code), a null case and the empty case."""
from __future__ import annotations

import dataclasses
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.learning import promotion as PR                                   # noqa: E402
from engine.research import evidence as EV                                    # noqa: E402
from engine.research import quality_gate as QG                                # noqa: E402

warnings.filterwarnings("ignore")
START = "2016-04-01"                    # the planted world's first decision date (a frame that starts mid-year)


# ============================================================================================================ (a) the calendar
def _frame(now: str, start: str = START, history_years: int = 2) -> pd.DataFrame:
    """The planted feed's sliding frame at `now`: calendar years now.year - 2 .. now.year, matured rows only (end < now)."""
    d = pd.date_range(start, now, freq="W-FRI")
    d = d[d.year >= pd.Timestamp(now).year - history_years]
    idx = pd.MultiIndex.from_product([d, ["A", "B"]])
    G = pd.DataFrame({"end": idx.get_level_values(0) + pd.Timedelta(days=7)}, index=idx)
    return G[pd.to_datetime(G["end"]) < pd.Timestamp(now)]


def _years(G: pd.DataFrame, cfg: EV.EvidenceConfig) -> tuple[set, set]:
    _, _, tr, te = EV.split_dates(G, cfg)
    d = pd.to_datetime(G.index.get_level_values(0))
    return set(d[tr].year), set(d[te].year)


def _oos_gate(G: pd.DataFrame, cfg: EV.EvidenceConfig, seed: int = 0) -> QG.GateOutcome:
    """The quality gate's own out-of-sample rule on the split's dates, with a strong positive effect on every test date."""
    train_end, _, tr, te = EV.split_dates(G, cfg)
    d = pd.DatetimeIndex(pd.to_datetime(G.index.get_level_values(0)))
    dates = tuple(str(x.date()) for x in d[te].unique().sort_values())
    eff = tuple(float(v) for v in np.random.default_rng(seed).normal(0.30, 0.08, len(dates)))
    t = PR.t_stat(np.asarray(eff))
    ob = QG.OOSBundle(PR.StatisticalEvidence(float(np.mean(eff)), len(eff), t, PR.one_sided_p(t), 20, float(len(eff))),
                      PR.OOSEvidence(str(train_end.date()), dates, eff, 0.3), tuple(sorted(set(d[tr].year))))
    return QG.gate_out_of_sample(ob, QG.QualityPolicy(), "2030-01-01")


def test_old_fractional_cut_splits_a_year_in_december():
    """Planted defect, reproduced with alignment off (the pre-F14 rule): December 2018 cuts train in January 2017, the gate counts 2017
    as seen and the look FAILS with one unseen year, while June 2018 had two."""
    old = EV.EvidenceConfig(align_years=False)
    tr, te = _years(_frame("2018-12-14"), old)
    assert tr & te == {2017}                                                     # a calendar year split between train and test
    assert "only 1 unseen calendar year" in _oos_gate(_frame("2018-12-14"), old).detail
    assert _oos_gate(_frame("2018-06-15"), old).state == QG.PASS


def _unseen(G: pd.DataFrame, cfg: EV.EvidenceConfig) -> tuple[int, int]:
    """(unseen calendar years, unseen test dates) of the split, counted exactly as the gate counts them."""
    _, _, tr, te = EV.split_dates(G, cfg)
    d = pd.to_datetime(G.index.get_level_values(0))
    seen = set(d[tr].year)
    dt = pd.DatetimeIndex(d[te]).unique()
    return len(set(dt.year) - seen), int((~np.isin(dt.year, list(seen))).sum())


LOOKS = ["2017-12-15", "2018-03-16", "2018-06-15", "2018-09-14", "2018-12-14", "2019-02-15", "2019-03-15", "2019-06-14", "2019-09-13",
         "2019-12-13"]


@pytest.mark.parametrize("now", LOOKS[2:])
def test_every_month_sees_two_unseen_years(now):
    """Fails on the pre-F14 code for December 2018 (one unseen year)."""
    cfg = EV.EvidenceConfig()
    G = _frame(now)
    assert _unseen(G, cfg)[0] == 2, (now, _years(G, cfg))
    o = _oos_gate(G, cfg)
    assert o.measures["unseen_years"] == 2 and "unseen calendar year" not in o.detail, (now, o.detail)   # (a short frame may still lack sample size)


@pytest.mark.parametrize("now", LOOKS)
def test_the_cut_never_loses_unseen_evidence_to_the_boundary(now):
    """The chosen cut leaves as many unseen test dates as the best in-bounds cut at a year end or the fractional target - and never
    fewer than the fractional cut alone did."""
    cfg, old = EV.EvidenceConfig(), EV.EvidenceConfig(align_years=False)
    G = _frame(now)
    d = pd.DatetimeIndex(pd.to_datetime(G.index.get_level_values(0))).unique().sort_values()
    yr, n = d.year.to_numpy(), len(d)
    best = max(EV.unseen_dates(yr, c) for c in range(2, n) if cfg.align_min_frac * n <= c <= cfg.align_max_frac * n)
    k, _ = EV.train_cut(d, cfg)
    assert EV.unseen_dates(yr, k) >= best and _unseen(G, cfg)[1] >= _unseen(G, old)[1]


def test_the_cut_moves_only_when_the_boundary_costs_evidence():
    """December 2018 (fractional cut two weeks into 2017) moves to the 2016 year end; June 2018 and the first look (December 2017)
    lose nothing at the fractional cut and keep it - so the first look's train window, and with it the replication discovery the
    ledger registers once, are exactly as before F14."""
    cfg, old = EV.EvidenceConfig(), EV.EvidenceConfig(align_years=False)
    dec = pd.to_datetime(_frame("2018-12-14").index.get_level_values(0))
    k, moved = EV.train_cut(dec, cfg)
    assert moved and pd.Timestamp(sorted(dec.unique())[k - 1]) == pd.Timestamp("2016-12-30")
    for now in ("2017-12-15", "2018-06-15", "2019-06-14"):
        d = pd.to_datetime(_frame(now).index.get_level_values(0))
        assert EV.train_cut(d, cfg) == EV.train_cut(d, old), now
        assert EV.split_dates(_frame(now), cfg)[0] == EV.split_dates(_frame(now), old)[0]


def test_december_and_june_are_judged_on_comparable_unseen_evidence():
    """The same unseen-year count in June and December, and December (later) never has fewer unseen weeks than June."""
    cfg = EV.EvidenceConfig()
    for y in (2018, 2019):
        jun, dec = _unseen(_frame(f"{y}-06-15"), cfg), _unseen(_frame(f"{y}-12-14"), cfg)
        assert jun[0] == dec[0] == 2 and dec[1] >= jun[1], (y, jun, dec)


def test_january_shortfall_is_the_feeds_calendar_frame_not_the_split():
    """Found while fixing (a): in the first week of a year the calendar frame (the pre-F16 store.upto kept calendar years
    now.year-2..now.year) has just dropped its oldest year and has no matured row of the new year, so it spans TWO calendar years and
    only one can be unseen, under any split. The split is not at fault; a rolling frame of the last 156 weeks gives two unseen years at
    the same date. FIXED in F16: engine/research/feeds.py FrameStore.upto is now that rolling window (FeedConfig.frame_weeks = 156);
    the real-FrameStore proofs are tests/test_rolling_research_frame.py. `_frame` here still models the calendar window on purpose."""
    from engine.research import feeds as FD
    cfg = EV.EvidenceConfig()
    G = _frame("2019-01-11")
    assert set(pd.to_datetime(G.index.get_level_values(0)).year) == {2017, 2018} and _unseen(G, cfg)[0] == 1
    now = pd.Timestamp("2019-01-11")
    d = pd.date_range(START, now, freq="W-FRI")
    d = d[d > now - pd.Timedelta(weeks=FD.FeedConfig().frame_weeks)]                  # the feed's own window rule
    idx = pd.MultiIndex.from_product([d, ["A", "B"]])
    rolling = pd.DataFrame({"end": idx.get_level_values(0) + pd.Timedelta(days=7)}, index=idx)
    rolling = rolling[pd.to_datetime(rolling["end"]) < now]
    assert _unseen(rolling, cfg)[0] >= 2


@pytest.mark.parametrize("now", LOOKS)
def test_the_split_keeps_the_purge_and_its_share_bounds(now):
    cfg = EV.EvidenceConfig()
    G = _frame(now)
    train_end, first_test, tr, te = EV.split_dates(G, cfg)
    d = pd.to_datetime(G.index.get_level_values(0))
    n = d.nunique()
    k, moved = EV.train_cut(d, cfg)
    if moved:
        assert cfg.align_min_frac * n <= k <= cfg.align_max_frac * n
        assert train_end.month == 12 and (train_end + pd.Timedelta(days=7)).year == train_end.year + 1   # the year's last Friday
    assert (pd.to_datetime(G["end"])[tr] < first_test).all() and first_test >= train_end + pd.Timedelta(days=cfg.purge_days)
    assert not (tr & te).any() and te.any() and tr.any()


def test_no_qualifying_year_end_keeps_the_fractional_cut():
    """Null case: a frame inside one calendar year, or one whose only year end lies outside the share bounds, is cut at orient_frac
    as before and says so (the gate then judges honestly: a split year is seen, never unseen)."""
    cfg = EV.EvidenceConfig()
    one_year = pd.date_range("2018-01-05", periods=40, freq="W-FRI")
    assert EV.train_cut(one_year, cfg) == (int(40 * cfg.orient_frac), False)
    late_end = pd.date_range("2018-03-02", periods=50, freq="W-FRI")          # 2018 year end at 44/50 = 0.88 of the frame
    assert EV.train_cut(late_end, cfg) == (15, False)
    assert EV.train_cut(late_end, EV.EvidenceConfig(align_years=False)) == (15, False)
    assert EV.train_cut([], cfg) == (2, False) and EV.unseen_dates(np.array([]), 0) == 0          # empty: nothing to move
    with pytest.raises(ValueError, match="too few"):
        EV.split_dates(_frame("2018-01-12").iloc[:4], cfg)


def test_alignment_config_is_validated():
    assert EV.EvidenceConfig().validate() == []
    assert EV.EvidenceConfig(align_min_frac=0.4).validate()                      # min above orient_frac
    assert EV.EvidenceConfig(align_max_frac=0.9).validate()
    assert EV.EvidenceConfig(max_failure_rate=0.5).validate() and EV.EvidenceConfig(max_failure_rate=0.0).validate()


# ============================================================================================================ (b) the failure floor
def test_failure_rate_upper_bound_is_clopper_pearson():
    assert EV.failure_rate_upper(0, 29, 0.05) == pytest.approx(1 - 0.05 ** (1 / 29), rel=1e-9)
    assert EV.failure_rate_upper(0, 0, 0.05) == 1.0 and EV.failure_rate_upper(5, 5, 0.05) == 1.0
    u = [EV.failure_rate_upper(k, 100, 0.05) for k in range(6)]
    assert all(a < b for a, b in zip(u, u[1:]))                                  # more failures, higher bound
    assert EV.failure_rate_upper(3, 100, 0.001) > EV.failure_rate_upper(3, 100, 0.05) > EV.failure_rate_upper(3, 400, 0.05)


_REF: list = []


def _bundle(n_periods: int, episodes: int | None, mu: float = 0.004, seed: int = 0, sid: str = "S") -> EV.Bundle:
    """The reference (clean) evidence with an OOS series of `n_periods` weeks in unseen years and `episodes` failure episodes."""
    if not _REF:
        _REF.append(QG.reference_evidence()[0])
    ev = _REF[0]
    d = pd.date_range("2018-01-05", periods=n_periods, freq="W-FRI")
    eff = np.random.default_rng(seed).normal(mu, 0.01, n_periods)
    t = PR.t_stat(eff)
    oos = QG.OOSBundle(PR.StatisticalEvidence(float(eff.mean()), n_periods, t, PR.one_sided_p(t), 8, float(n_periods)),
                       PR.OOSEvidence("2017-12-29", tuple(str(x.date()) for x in d), tuple(float(v) for v in eff), 0.0055), (2016, 2017))
    fe = dataclasses.replace(ev.failure, n_failure_episodes=episodes)
    return EV.Bundle(sid, dataclasses.replace(ev, oos=oos, failure=fe), {}, {})


def test_a_strong_rarely_failing_effect_is_no_longer_held_for_episodes():
    """Planted defect (F12 seeds 0 and 4): everything else clean, 150 test weeks, only 2 failure episodes. Pre-F14 the gate held it at
    NEEDS_MORE_EVIDENCE ('only 2 failure episodes'); now exposure bounds its failure rate as rare and it is PROMOTEd."""
    b = _bundle(150, 2)
    old = QG.step([QG.Candidate("S", b.evidence)], "2021-06-01", policy=QG.QualityPolicy(code_hash="refcode"))
    assert old.decisions[0].verdict != "PROMOTE" and "failure_behavior" in EV.blocking(old, "S")
    ff = EV.failure_floor(b, QG.QualityPolicy(), 0.05)
    assert ff.route == "exposure" and ff.required == 2 and ff.rate_upper <= 0.10
    assert EV.verdicts(EV.gate([b], "2021-06-01", "refcode"))["S"] == "PROMOTE"
    assert EV.verdicts(EV.gate([b], "2021-06-01", "refcode", looks={"S": 1}))["S"] == "PROMOTE"


def test_short_exposure_still_waits_for_episodes():
    """Null case for the relaxation: 2 episodes in 25 weeks bound the failure rate only below ~0.25 - not 'rare', so the floor stays."""
    b = _bundle(25, 2)
    ff = EV.failure_floor(b, QG.QualityPolicy(), 0.05)
    assert ff.route == "waiting" and ff.required == 5
    rep = EV.gate([b], "2021-06-01", "refcode")
    assert EV.verdicts(rep)["S"] != "PROMOTE" and "failure episodes" in EV.blocking(rep, "S")["failure_behavior"]


def test_repeated_looks_cannot_manufacture_rare():
    """The bound is taken at the look's spent alpha: evidence that is 'rare' at look 1 is not at look 60."""
    b, plan = _bundle(150, 2), EV.SequentialPlan()
    assert EV.failure_floor(b, plan.quality_policy(1, "refcode"), plan.alpha_at(1)).route == "exposure"
    late = EV.failure_floor(b, plan.quality_policy(60, "refcode"), plan.alpha_at(60))
    assert late.route == "waiting" and late.rate_upper > 0.10


def test_enough_episodes_use_the_policy_floor_and_missing_stays_missing():
    assert EV.failure_floor(_bundle(150, 12), QG.QualityPolicy(), 0.05).route == "episodes"
    miss = EV.failure_floor(_bundle(150, None), QG.QualityPolicy(), 0.05)
    assert miss.route == "missing" and miss.required == 5
    empty = EV.Bundle("E", QG.QualityEvidence(), {}, {})                         # empty evidence: never relaxed, never passed
    ff = EV.failure_floor(empty, QG.QualityPolicy(), 0.05)
    assert ff.route == "missing" and ff.periods == 0 and ff.rate_upper == 1.0
    assert EV.verdicts(EV.gate([empty], "2021-06-01", "refcode"))["E"] != "PROMOTE"


def test_other_failure_checks_still_bind_on_the_exposure_route():
    """Only the episode COUNT is relaxed: a rarely-failing rule without a retirement trigger, or one that collapses under data
    degradation, still fails the failure gate."""
    b = _bundle(150, 2)
    for bad in ({"has_retirement_trigger": False}, {"perturbation_retention": 0.2}, {"abstains_out_of_scope": False}):
        ev = dataclasses.replace(b.evidence, failure=dataclasses.replace(b.evidence.failure, **bad))
        rep = EV.gate([EV.Bundle("S", ev, {}, {})], "2021-06-01", "refcode")
        assert EV.verdicts(rep)["S"] != "PROMOTE" and "failure_behavior" in EV.blocking(rep, "S"), bad


def _episodes(eff: np.ndarray, frac: float = 0.5) -> int:
    """evidence.failure's episode rule: weeks under `frac` of the typical effect (under zero when there is none)."""
    m = float(eff.mean())
    return int((eff < (frac * m if m > 0 else 0.0)).sum())


def test_null_and_weak_rules_never_reach_the_exposure_route():
    """False-promotion control: a null or weak weekly AUC effect fails in a large share of weeks, so over 400 simulated findings of
    each at 60-200 periods the exposure route never opens; a strong effect (AUC ~0.85, the planted lv20) opens it."""
    rng, pol = np.random.default_rng(3), QG.QualityPolicy()
    for mu in (0.0, 0.01, 0.03):
        opened = 0
        for i in range(400):
            n = int(rng.integers(60, 201))
            e = rng.normal(mu, 0.08, n)
            b = _bundle(n, _episodes(e))
            b = EV.Bundle("S", dataclasses.replace(b.evidence, oos=dataclasses.replace(b.evidence.oos, oos=dataclasses.replace(
                b.evidence.oos.oos, oos_effects=tuple(float(v) for v in e)))), {}, {})
            opened += EV.failure_floor(b, pol, 0.05).route == "exposure"
        assert opened == 0, (mu, opened)
    strong = rng.normal(0.35, 0.08, 100)
    assert _episodes(strong) < 5 and EV.failure_floor(_bundle(100, _episodes(strong)), pol, 0.05).route == "exposure"


def test_mixed_bundles_keep_their_own_floor_and_order():
    rare, often, few = _bundle(150, 2, sid="R"), _bundle(150, 12, sid="O"), _bundle(25, 2, sid="F")
    rep = EV.gate([rare, often, few], "2021-06-01", "refcode")
    assert [d.subject_id for d in rep.decisions] == ["R", "O", "F"]
    v = EV.verdicts(rep)
    assert v["R"] == "PROMOTE" and v["O"] == "PROMOTE" and v["F"] != "PROMOTE"
    seq = EV.gate([rare, few], "2021-06-01", "refcode", looks={"R": 1, "F": 1})
    assert EV.verdicts(seq) == {"R": "PROMOTE", "F": EV.verdicts(rep)["F"]}
    assert EV.gate([], "2021-06-01", "refcode").decisions == ()
