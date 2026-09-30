"""F18: the three C68 stages the clean full loops starved (c68.what_changed and c68.research_depth SKIPPED_NO_INPUT in 129/129 cycles,
c68.error_research OK 4-8 times; C68 error research depth / what-changed, C69 sections 5, 27, 31: a stage that never receives input is
shallow, not done). Through the REAL loop (LP.open_loop / LP.step): each stage runs on a world that carries its trigger and stays
SKIPPED on a world without it, and the skip reason names the upstream link that is missing. Unit cases for the repeated-surprise
wiring: a planted repeat must be asked once, a re-ask without a new surprise must not happen, and the empty tracker asks nothing.
Planted worlds only (C63)."""
from __future__ import annotations

import warnings

import pytest

from engine.learning.surprise import SurpriseTracker
from engine.research import error_loop as EL
from engine.research import feeds as FD
from engine.research import loop as LP
from engine.research import two_stage as TS

warnings.filterwarnings("ignore")
CLOCK = lambda: 1_700_000_000.0                                   # noqa: E731 - fixed wall clock: reruns are reproducible
KEEP = ("observe.panel", "evaluate.two_stage", "questions.generate", "hypotheses.trees", "gain.priority")
TS_CFG = TS.TwoStageConfig(gate_min_weeks=6, min_direction_rows=60, min_calib_rows=20)
FEED = FD.FeedConfig(warm_weeks=50)
STARVED = ("c68.what_changed", "c68.error_research", "c68.research_depth")


def loop_cfg(extra_off: tuple = ()) -> LP.LoopConfig:
    off = tuple(n for n in LP.BUILTIN_STAGES if n not in KEEP and n != "report.cycle") + tuple(extra_off)
    return LP.LoopConfig(run_id="f18", free_gb=12.0, code_hash="f18-test", checkpoint="off", cadence={}, disabled=off, two_stage=TS_CFG)


@pytest.fixture(scope="module", autouse=True)
def registered():
    EL.register()
    yield
    EL.unregister()


def run(world, root, cycles: int, extra_off: tuple = ()):
    state, rt, _ = LP.open_loop(FD.WorldFeed(FD.InMemorySource(world), FEED), root, loop_cfg(extra_off), clock=CLOCK)
    reps = [r for r in (LP.step(state, rt) for _ in range(cycles)) if r]
    return state, rt, EL.stage_table(reps)


@pytest.fixture(scope="module")
def trigger_run(tmp_path_factory):
    """The planted C68 world: trend episodes give names a knowable realisable gain inside the 5-10% band, so positions are taken,
    mature, and produce errors - the trigger of all three stages."""
    return run(EL.plant_world(EL.C68Plant()), tmp_path_factory.mktemp("f18trig"), 6)


@pytest.fixture(scope="module")
def null_world():
    """The same world without the trend drift (and without the regime switch): nothing is predictable into the band, so no position
    is ever taken and no prediction error can exist."""
    return EL.plant_world(EL.C68Plant(drift=0.0, regime_switch=False))


@pytest.fixture(scope="module")
def null_run(null_world, tmp_path_factory):
    return run(null_world, tmp_path_factory.mktemp("f18null"), 8)


def _ok(tab, stage):
    return tab[(tab.stage == stage) & (tab.status == "OK")]


# ============================================================================================================ the trigger world
def test_every_starved_stage_runs_on_a_world_that_has_its_trigger(trigger_run):
    state, rt, tab = trigger_run
    for s in STARVED:
        assert len(_ok(tab, s)) >= 1, (s, tab[tab.stage == s])
    led, st = rt.handles["c68.ledgers"], state.modules["c68"]
    assert len(led.expectations) >= 5 and len(led.errors) >= 3                       # the trigger: matured errors of real positions
    assert st.investigated and st.wcs.investigations                                   # what_changed investigated them
    assert (_ok(tab, "c68.error_research")["in"] > 0).any()                            # error_research ingested errors, not only market
    assert led.pipe.furthest().get("RESEARCH", 0) + sum(v for k, v in led.pipe.furthest().items()
                                                         if EL._ORDER.get(k, -1) > EL._ORDER["RESEARCH"]) >= 1
    error_subjects = [s for s in st.cells if not s.startswith("repeated surprise")]
    assert error_subjects and any(q.subject in error_subjects for q in state.questions.values())   # error cells reached the loop queue


# ============================================================================================================ the null worlds
def test_without_positions_what_changed_stays_skipped_and_says_why(null_run):
    state, rt, tab = null_run
    led = rt.handles["c68.ledgers"]
    assert len(led.expectations) == 0 and len(led.errors) == 0
    wc = tab[tab.stage == "c68.what_changed"]
    assert len(wc) == 8 and (wc.status == "SKIPPED_NO_INPUT").all()
    assert wc["note"].str.contains("0 position").all()                                  # the reason names the missing upstream link


def test_without_errors_error_research_never_asks_an_error_question(null_run):
    """error_research may still run on the null world - on repeated MARKET surprises, a trigger the world does have - but it never
    ingests an error and never files an error cell; research_depth never records research for a prediction."""
    state, rt, tab = null_run
    st, led = state.modules["c68"], rt.handles["c68.ledgers"]
    er = tab[tab.stage == "c68.error_research"]
    assert (er["in"].fillna(0) == 0).all()
    assert all(s.startswith("repeated surprise market|") for s in st.cells), st.cells
    assert "RESEARCH" not in led.pipe.furthest()
    # the F18 accounting defect: a run that emitted repeated-surprise events used to report SKIPPED_NO_INPUT
    # (with no error ingested, an OK run must have counted what it emitted)
    ok = _ok(tab, "c68.error_research")
    assert len(ok) >= 1 and (ok["out"] > 0).all()


def test_with_the_market_stage_off_as_well_both_error_stages_stay_skipped(null_world, tmp_path):
    """No position and no market expectation (the other source of C68 surprise): nothing reaches error_research or research_depth."""
    state, rt, tab = run(null_world, tmp_path, 5, extra_off=("c68.market_regime",))
    for s in STARVED:
        rows = tab[tab.stage == s]
        assert len(rows) == 5 and (rows.status == "SKIPPED_NO_INPUT").all(), (s, rows)
    assert state.modules["c68"].cells == {}
    assert not any(q.subject.startswith("repeated surprise") for q in state.questions.values())


# ============================================================================================================ repeated-surprise wiring
def _tracker_state(n_big: int, n_small: int = 8) -> EL.C68State:
    st = EL.new_state()
    tr = SurpriseTracker()
    day = 1
    for k in range(n_small):
        tr.observe("market|volatility", 0.0, 0.1 * (-1) ** k, f"2020-01-{day:02d}", f"2020-01-{day + 1:02d}", "2020-03-01", scale=1.0)
        day += 1
    for k in range(n_big):
        tr.observe("market|volatility", 0.0, 3.5, f"2020-01-{day:02d}", f"2020-01-{day + 1:02d}", "2020-03-01", scale=1.0)
        day += 1
    st.tracker = tr
    return st


def test_a_planted_repeated_surprise_is_asked_once_and_again_only_after_a_new_one():
    st = _tracker_state(3)
    ev: list = []
    assert EL.repeated_surprise_events(st, ev, "2020-03-01") == 1
    assert ev[0].subject == "repeated surprise market|volatility" and st.cells[ev[0].subject] == "market|volatility"
    assert ev[0].evidence_through < "2020-03-01"                                    # the last big surprise, never today
    assert EL.repeated_surprise_events(st, ev, "2020-03-08") == 0                   # nothing new in the cell: no re-ask
    st.tracker.observe("market|volatility", 0.0, -3.1, "2020-03-02", "2020-03-03", "2020-03-08", scale=1.0)
    assert EL.repeated_surprise_events(st, ev, "2020-03-08") == 1 and len(ev) == 2  # a new big surprise: asked again


def test_no_big_surprise_and_the_empty_tracker_ask_nothing():
    assert EL.repeated_surprise_events(EL.new_state(), [], "2020-03-01") == 0       # empty
    st = _tracker_state(0)
    ev: list = []
    assert EL.repeated_surprise_events(st, ev, "2020-03-01") == 0 and ev == [] and st.cells == {}   # small surprises only


def test_a_surprise_that_matures_at_now_is_not_evidence_yet():
    st = _tracker_state(2)
    ev: list = []
    assert EL.repeated_surprise_events(st, ev, "2020-01-10") == 0                   # the big ones mature on/after the 10th


def test_a_state_pickled_before_the_fix_still_works():
    st = _tracker_state(3)
    del st.__dict__["surprise_sent"]
    assert EL.repeated_surprise_events(st, [], "2020-03-01") == 1 and "market|volatility" in st.surprise_sent
