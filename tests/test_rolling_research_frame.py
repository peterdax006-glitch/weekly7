"""F16 (C69 sections 12-14, 21, 27, 28; canon C56; follow-up to F14): the research frame is a ROLLING window of the last `frame_weeks`
decision dates ending at `now`, not whole calendar years.

Defect (F14 finding): FrameStore.upto kept calendar years now.year-2 .. now.year, so in the first weeks of a year the frame spanned two
calendar years and the gate could see only one unseen year - every January look was starved. These tests run the REAL FrameStore on the
planted world: the January planted case fails on the pre-F16 code (frame_weeks = 0 reproduces it), the window is past-only and fail-closed,
rows are identical to the calendar frames they replace, trimming never serves a short frame, and the empty / degenerate cases hold."""
from __future__ import annotations

import dataclasses
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.research import evidence as EV                                    # noqa: E402
from engine.research import feeds as FD                                       # noqa: E402
from engine.research.core import FirewallBreach                               # noqa: E402

warnings.filterwarnings("ignore")
PLANT = FD.PlantConfig(n_names=24, seed=2)                 # four years from 2016-01-04; smaller than the default for speed
CAL = dataclasses.replace(FD.FeedConfig(), frame_weeks=0)  # the pre-F16 calendar window, kept to reproduce the defect
JAN, JUL = "2019-01-11", "2018-07-13"


@pytest.fixture(scope="module")
def world():
    return FD.planted_world(PLANT)


def _feed(world, cfg=FD.FeedConfig()) -> FD.WorldFeed:
    return FD.WorldFeed(FD.InMemorySource(world), cfg)


def _matured(feed: FD.WorldFeed, now: str) -> pd.DataFrame:
    F = feed.store.upto(now)
    return F[pd.to_datetime(F["end"]) < pd.Timestamp(now)]


def _unseen_years(M: pd.DataFrame) -> int:
    """Unseen calendar years of the evidence split, counted exactly as quality_gate.gate_out_of_sample counts them."""
    _, _, tr, te = EV.split_dates(M, EV.EvidenceConfig())
    d = pd.to_datetime(M.index.get_level_values(0))
    return len(set(d[te].year) - set(d[tr].year))


def _dates(F: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(F.index.get_level_values(0))).unique().sort_values()


# ============================================================================================================ the planted defect
def test_january_look_gets_the_same_unseen_evidence_as_july(world):
    """Fails on the pre-F16 code: in January 2019 the calendar frame spans 2017-2018 only (one unseen year), while July 2018 had two.
    The rolling frame gives January at least the unseen years July had."""
    cal, roll = _feed(world, CAL), _feed(world)
    assert _unseen_years(_matured(cal, JAN)) == 1 < _unseen_years(_matured(cal, JUL)) == 2        # the defect, reproduced
    assert set(_dates(_matured(cal, JAN)).year) == {2017, 2018}
    assert _unseen_years(_matured(roll, JAN)) >= _unseen_years(_matured(roll, JUL)) == 2


def test_the_gate_itself_passes_the_january_out_of_sample_check(world):
    """End to end through evidence.assemble and the quality gate for the genuine planted feature: the January calendar look fails
    out-of-sample with 'only 1 unseen calendar year'; the rolling look does not."""
    spec = EV.FindingSpec("S_lv20", "lv20", 1.0, "VOLATILITY", n_tests_searched=20, has_falsifier=True)
    out = {}
    for name, cfg in (("cal", CAL), ("roll", FD.FeedConfig())):
        b = EV.assemble(_matured(_feed(world, cfg), JAN), spec, JAN, code_hash="f16", data_hash="dh", created_real="2026-09-30T00:00:00+00:00",
                        cfg=EV.EvidenceConfig(identity_boot=30))
        out[name] = EV.blocking(EV.gate([b], JAN, "f16"), "S_lv20")
    assert "unseen calendar year" in out["cal"].get("out_of_sample", ""), out["cal"]
    assert "unseen calendar year" not in out["roll"].get("out_of_sample", ""), out["roll"]


@pytest.mark.parametrize("now", ["2019-01-04", "2019-02-15", "2019-04-12", "2019-07-12", "2019-10-11"])
def test_every_month_has_two_unseen_years_and_the_same_history(world, now):
    feed = _feed(world)
    M = _matured(feed, now)
    assert _unseen_years(M) >= 2, now
    d = _dates(feed.store.upto(now))
    lo = pd.Timestamp(now) - pd.Timedelta(weeks=156)
    assert d.min() > lo and d.max() <= pd.Timestamp(now)
    if pd.Timestamp(now) - pd.Timedelta(weeks=156) > pd.Timestamp("2016-05-01"):              # window inside the data: full length
        assert 154 <= len(d) <= 157, (now, len(d))


# ============================================================================================================ past-only, fail-closed
def test_the_window_never_carries_a_row_after_now_and_the_audit_still_refuses_one(world):
    feed = _feed(world)
    for now in feed.dates()[::9]:
        F = feed.store.upto(now)
        assert _dates(F).max() <= pd.Timestamp(now)
        obs = feed.observe(now)
        assert FD.audit_stage_input("observe.panel", obs.matured, now) >= 1                   # matured rows only: passes
    now = feed.dates()[-5]
    panel = feed.observe(now).matured
    row = panel.iloc[[-1]].copy()
    row.index = pd.MultiIndex.from_tuples([(pd.Timestamp(now), row.index[0][1])], names=panel.index.names)
    with pytest.raises(FirewallBreach):
        FD.audit_stage_input("observe.panel", pd.concat([panel, row]), now)                 # a row dated at now is refused
    with pytest.raises(FirewallBreach):
        FD.audit_stage_input("observe.panel", {"end_date": str(pd.Timestamp(now) + pd.Timedelta(days=2))}, now)


def test_rows_are_identical_to_the_calendar_frames_they_replace(world):
    """The window changes WHICH rows a look reads, never a row's values: features are still computed per year with its warm-up."""
    now = "2019-06-14"
    roll, cal = _feed(world).store.upto(now), _feed(world, CAL).store.upto(now)
    common = roll.index.intersection(cal.index)
    assert len(common) > 0.6 * len(cal)
    pd.testing.assert_frame_equal(roll.loc[common], cal.loc[common])
    extra = _dates(roll.loc[roll.index.difference(cal.index)])
    assert len(extra) and extra.max() < pd.Timestamp("2017-01-01") and extra.min() > pd.Timestamp(now) - pd.Timedelta(weeks=156)


# ============================================================================================================ streaming and memory
def test_forward_walk_trims_the_oldest_year_and_never_serves_a_short_frame(world):
    """A forward walk trims the oldest held year to the window; an earlier `now` afterwards (a resumed or rewound loop) rebuilds it and
    gets exactly what a fresh store returns - observe stays a pure function of now."""
    feed = _feed(world)
    dates = feed.dates()
    for now in dates[::4]:
        feed.store.upto(now)
    last = dates[::4][-1]
    lo = pd.Timestamp(last) - pd.Timedelta(weeks=156)
    held = pd.concat(list(feed.store.frames.values()))
    assert _dates(held).min() > lo and min(feed.store.frames) == lo.year and feed.store.trimmed
    assert feed.store.built == len(feed.store.source.years()) and feed.store.rebuilt == 0      # each year built once going forward
    early = "2018-09-14"
    got = feed.store.upto(early)
    assert feed.store.rebuilt >= 1
    pd.testing.assert_frame_equal(got, _feed(world).store.upto(early))


def test_a_trimmed_frame_served_short_would_be_caught(world):
    """Planted defect: force the trimmed marker away so the store believes a trimmed frame is whole; the earlier look then loses rows,
    and the comparison with a fresh store catches it (this is the check the rebuild exists for)."""
    feed = _feed(world)
    for now in feed.dates()[::4]:
        feed.store.upto(now)
    feed.store.trimmed.clear()
    early = "2018-09-14"
    short, whole = feed.store.upto(early), _feed(world).store.upto(early)
    assert len(short) < len(whole)


def test_calendar_mode_still_evicts_as_before(world):
    feed = _feed(world, CAL)
    for now in feed.dates()[::6]:
        feed.store.upto(now)
    y = pd.Timestamp(feed.dates()[::6][-1]).year
    assert sorted(feed.store.frames) == [k for k in feed.store.source.years() if y - CAL.history_years <= k <= y]
    assert not feed.store.trimmed


# ============================================================================================================ null, empty, validation
def test_frame_weeks_is_validated():
    assert FD.FeedConfig().validate() == [] and FD.FeedConfig().frame_weeks == 156 and CAL.validate() == []
    assert FD.FeedConfig(frame_weeks=60).validate() and FD.FeedConfig(frame_weeks=600).validate()
    with pytest.raises(ValueError, match="frame_weeks"):
        FD.WorldFeed(FD.InMemorySource(FD.planted_world(FD.PlantConfig(n_names=8, n_days=300))), FD.FeedConfig(frame_weeks=10))


def test_empty_and_before_the_data(world):
    feed = _feed(world)
    assert len(feed.store.upto("2010-01-04")) == 0                                          # no year of the source in the window
    assert len(feed.store.upto("2016-01-04")) == 0                                          # first session: nothing dated <= now yet
    assert feed.store.span("2030-01-01")[2] == []                                           # window entirely after the data
    assert len(feed.store.upto("2030-01-01")) == 0
    obs = feed.observe("2010-01-04")
    assert len(obs.matured) == 0


def test_null_world_window_is_the_same_shape(world):
    """Null case: a world where nothing predicts gets the same window (the window never depends on the data's content)."""
    null = FD.planted_world(dataclasses.replace(PLANT, vol_state_sd=0.0, coincidence_rate=0.0, earnings_shock=0.0))
    a, b = _dates(_feed(world).store.upto(JAN)), _dates(_feed(null).store.upto(JAN))
    assert a.equals(b)
    assert np.isfinite(_feed(null).store.upto(JAN)["vol20"].to_numpy(float)).mean() > 0.5
