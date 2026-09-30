"""F24 (C75 Phase 1): persisted health books reload; the failed-lab and waste stages do real work on what the loop produces; the
conjunction pre-screen no longer loses XOR pairs. Synthetic data only; the loop test runs 8 cycles of the real research loop."""
from __future__ import annotations

import datetime as dt
import json
import sys
from math import comb

import numpy as np
import pandas as pd
import pytest

from engine.learning import health as hm
from engine.learning.core import Health
from engine.research import compute_manager as CM
from engine.research import failed_lab as FL
from engine.research import targets as T
from engine.research import value_accounting as VA
from engine.research import waste as WA
from engine.research.compute_manager import Problem

D0 = "2010-01-04"


# ============================================================================================================ 1. health-book round trip
def _series(parts, seed, start="2014-01-03"):
    rng = np.random.default_rng(seed)
    v = np.concatenate([rng.normal(m, 0.010, n) for n, m in parts])
    return pd.Series(v, index=pd.date_range(start, periods=len(v), freq="W-FRI"))


def _six_item_book() -> hm.HealthBook:
    """Six items in six different health states, each assessed on several dates, so the evidence carries NaN / inf / None fields."""
    good, broken = _series([(220, 0.006)], 1), _series([(150, 0.006), (60, -0.010)], 2)
    short, stale = _series([(20, 0.006)], 3), _series([(120, 0.006)], 4)
    empty = pd.Series([np.nan] * 5, index=pd.date_range("2020-01-03", periods=5, freq="W-FRI"))
    ins = [hm.HealthInput("good", good), hm.HealthInput("broken", broken), hm.HealthInput("short", short),
           hm.HealthInput("stale", stale), hm.HealthInput("empty", empty), hm.HealthInput("parked", good, retirement_state="DORMANT")]
    book = hm.HealthBook()
    for inp in ins:
        end = max(inp.values.index)
        for off in (7, 14, 21):
            for r in hm.assess([inp], end + pd.Timedelta(days=off)):
                book.record(r)
    return book


def test_six_persisted_health_books_reload_and_verify(tmp_path):
    """The defect: record_id hashed the raw evidence (NaN -> "nan") while the file held the cleaned evidence (NaN -> null), so every
    book with a NaN in it failed 'record altered' on reload. Save exactly as loop_hooks does (json.dumps, sort_keys), load, verify."""
    book = _six_item_book()
    states = {r.state for r in book.snapshot()}
    assert len(states) >= 4, states                                                   # the book really is mixed
    assert any(isinstance(v, float) and v != v for r in book.snapshot() for v in r.evidence.values()), "no NaN in evidence: test is vacuous"
    cases = {}
    for kid in book.items():                                                         # one single-item book per case = six separate files
        one = hm.HealthBook()
        for r in book.history(kid):
            one.record(r)
        p = tmp_path / f"{kid}.json"
        p.write_text(json.dumps(hm.export_book(one), sort_keys=True), encoding="utf-8")
        back = hm.import_book(json.loads(p.read_text(encoding="utf-8")))
        assert back.verify() == [] and len(back) == len(one)
        assert [r.record_id for r in back.history(kid)] == [r.record_id for r in one.history(kid)]
        cases[kid] = back.latest(kid).state
    assert len(cases) == 6 and cases["parked"] == Health.DORMANT


def test_reloaded_book_still_detects_tampering_and_old_format_is_refused(tmp_path):
    book = _six_item_book()
    data = json.loads(json.dumps(hm.export_book(book), sort_keys=True))
    hm.import_book(data)
    forged = json.loads(json.dumps(data))
    i = next(k for k, r in enumerate(forged["records"]) if r["state"] != "BROKEN")
    forged["records"][i]["state"] = "BROKEN"
    with pytest.raises(ValueError, match="altered"):
        hm.import_book(forged)
    chain_cut = json.loads(json.dumps(data))
    chain_cut["chain"] = chain_cut["chain"][1:]
    with pytest.raises(ValueError):
        hm.import_book(chain_cut)
    assert len(hm.import_book(hm.export_book(hm.HealthBook()))) == 0                  # empty book round-trips


# ============================================================================================================ 2a. failed lab
def _mgr(n=3):
    st = CM.ManagerState()
    bs = []
    for i in range(n):
        q = CM.ResearchQuestion.make(f"does feature_{i}_dispersion rank volatility movers", "t", Problem.VOLATILITY, D0, D0, "s", "f")
        bs.append(CM.make_branch(st, q, D0, "question:new_discovery"))
    return st, bs


def _run(b, st, day, effect=0.0004):
    b.runs.append({"stage": "STAGE1_CHEAP_SCREEN", "at": day, "effect": effect, "se": 0.0009, "t": effect / 0.0009, "action": "PARK", "cost": 20.0,
                   "n_obs": 400, "power": 0.3})
    b.spent["STAGE1_CHEAP_SCREEN"] = 20.0


def test_failed_lines_are_registered_once_and_reused():
    st, (b0, b1, b2) = _mgr()
    for b in (b0, b1):
        CM._record(st, b, CM.ResearchState.EXPLORING, "2010-01-11", "launch")
        _run(b, st, "2010-01-11")
    CM._record(st, b0, CM.ResearchState.FAILED, "2010-01-18", "a powered test excluded the smallest useful effect")
    CM.park(st, b1.branch_id, "2010-01-18", "needs ~80 cpu-min > cap 40: INFEASIBLE at today's data/compute; t=-0.77 below required 1.50")
    lab = FL.FailedLearnerLab()
    rep = FL.step(lab, "2010-02-01", manager=st)
    assert rep.synced["registered"] == 2 and rep.synced["failed_lines"] == 2 and rep.idle_reason == ""
    modes = {f.failure_mode.value for f in lab.registry.as_of("2010-02-01")}
    assert modes == {"NO_SKILL", "INFEASIBLE"}                                         # a refutation and a never-tested line are not the same thing
    assert FL.step(lab, "2010-02-02", manager=st).synced["registered"] == 0           # idempotent
    again = FL.question_already_failed(lab, b0.text, "VOLATILITY", "2010-02-02", b0.family)
    assert not again.decision.permits and again.statement                              # the failed line is not launched again
    reworded = FL.question_already_failed(lab, "does feature_0_dispersion rank the volatility movers", "VOLATILITY", "2010-02-02", b0.family)
    assert not reworded.decision.permits                                               # renaming does not hide it
    other = FL.question_already_failed(lab, "does turnover_acceleration predict gap reversals", "DIRECTION", "2010-02-02")
    assert other.decision.permits                                                      # an unrelated idea is not blocked
    assert lab.rediscovery_prevented("2010-02-03")["redirected"] >= 2
    assert FL.audit_lab(lab, "2010-02-03") is not None


def test_lab_does_not_see_the_future_or_unfailed_lines_and_says_why_it_is_idle():
    st, (b0, b1, b2) = _mgr()
    CM._record(st, b0, CM.ResearchState.EXPLORING, "2010-01-11", "launch")
    CM._record(st, b1, CM.ResearchState.EXPLORING, "2010-01-11", "launch")
    CM._record(st, b0, CM.ResearchState.FAILED, "2010-01-18", "powered null")
    CM.park(st, b1.branch_id, "2010-01-18", "cannot be resolved with the compute allowed")      # plain PARK: unresolved, not refuted
    lab = FL.FailedLearnerLab()
    early = FL.step(lab, "2010-01-18", manager=st)                                       # the failure is dated ON now: not yet known
    assert early.synced["registered"] == 0 and "nothing to register" in early.idle_reason
    late = FL.step(lab, "2010-01-19", manager=st)
    assert late.synced["registered"] == 1                                                # only the FAILED one; the plain park is not a failure
    blind = FL.step(FL.FailedLearnerLab(), "2010-02-01")
    assert "nothing feeds this lab" in blind.idle_reason and blind.synced == {}
    assert FL.step(FL.FailedLearnerLab(), "2010-02-01", manager=CM.ManagerState()).synced["failed_lines"] == 0


# ============================================================================================================ 2b. waste
def _ctx(day, h, rows):
    return WA.WorldContext(day, h, rows)


def test_ladder_parked_lines_are_adopted_then_revived_when_the_data_grows():
    st, bs = _mgr(4)
    for b in bs:
        CM._record(st, b, CM.ResearchState.EXPLORING, "2010-01-11", "launch")
        _run(b, st, "2010-01-11")
        CM.park(st, b.branch_id, "2010-01-11", "needs ~80 cpu-min > cap 40: INFEASIBLE at today's data/compute")
    led, book = VA.ValueLedger(), WA.DormantBook()
    assert not book.records
    r1 = WA.step(st, led, book, _ctx("2010-01-12", "h1", 1000), "2010-01-12")
    assert len(r1.adopted) == 4 and len(book.records) == 4 and not r1.revived           # adopted; the cooldown holds them for now
    assert {r.reason for r in book.records.values()} == {"INFEASIBLE_UNDERPOWERED"}
    assert book.life.state(WA.DormantBook.kid(bs[0].branch_id), "2010-01-13").value == "DORMANT"
    r2 = WA.step(st, led, book, _ctx("2010-02-20", "h1", 1000), "2010-02-20")            # same data: nothing new, nothing revived
    assert not r2.adopted and not r2.revived
    assert all(b.state is CM.ResearchState.DORMANT for b in bs)
    r3 = WA.step(st, led, book, _ctx("2010-02-27", "h2", 1400), "2010-02-27")            # +40% rows: new data -> compute goes back to the lines
    assert set(r3.revived) == {b.branch_id for b in bs}
    assert all(b.state is CM.ResearchState.QUEUED and b.frontier is CM.Stage.CHEAP_SCREEN for b in bs)
    r4 = WA.step(st, led, book, _ctx("2010-03-06", "h2", 1400), "2010-03-06")
    assert not r4.revived                                                                # the same trigger cannot revive twice


def test_waste_adoption_ignores_active_branches_and_handles_empty():
    st, bs = _mgr(2)
    CM._record(st, bs[0], CM.ResearchState.EXPLORING, "2010-01-11", "launch")
    res = WA.step(st, VA.ValueLedger(), WA.DormantBook(), _ctx("2010-01-12", "h", 10), "2010-01-12")
    assert res.adopted == () and res.parked == ()
    empty = WA.step(CM.ManagerState(), VA.ValueLedger(), WA.DormantBook(), _ctx("2010-01-12", "h", 10), "2010-01-12")
    assert empty.adopted == () and empty.verdicts == ()
    with pytest.raises(Exception):                                                        # a world dated after now is refused
        WA.step(st, VA.ValueLedger(), WA.DormantBook(), _ctx("2010-03-01", "h", 10), "2010-01-12")


# ============================================================================================================ 2c. through the real loop
@pytest.fixture(scope="module")
def loop_run(tmp_path_factory):
    sys.path.insert(0, "tests")
    import test_research_loop as TL
    from engine.research import loop as LP

    def st_sync(ctx):
        lab = ctx.mod_state("failed_lab", FL.FailedLearnerLab)
        ms = ctx.state.modules.get("compute_manager")
        if ms is None:
            raise LP.NoInput("no ladder yet")
        rep = FL.step(lab, ctx.now, manager=ms)
        return rep.synced.get("failed_lines", 0), rep.synced.get("registered", 0), rep.idle_reason

    LP.register_stage("priorities.failed_lab_sync", st_sync, after="priorities.failed_lab")
    try:
        F = TL.world()
        feed = LP.FrameFeed(F, dates=TL.dates_of(F)[30:38])
        state, reps = LP.run(feed, tmp_path_factory.mktemp("f24loop"), TL.cfg(), max_cycles=None, clock=TL.CLOCK)
    finally:
        LP.unregister_stage("priorities.failed_lab_sync")
    return state, reps


def test_real_loop_feeds_the_failed_lab_and_waste_adopts_what_the_ladder_parked(loop_run):
    state, reps = loop_run
    ms = state.modules["compute_manager"]
    parked = [b for b in ms.branches.values() if b.state is CM.ResearchState.DORMANT]
    assert parked, "the planted world parked nothing: this test would be vacuous"
    book = state.modules["waste_book"]
    assert {b.branch_id for b in parked} <= set(book.records), "waste never took over the lines the ladder parked"
    lab = state.modules["failed_lab"]
    infeasible = [b for b in parked if "INFEASIBLE" in b.dormant_reason]
    known = {t.branch_id for t in ms.log if t.at < str(reps[-1]["now"])[:10]}               # parks dated on the last day are not yet known
    distinct = {(b.text, b.family) for b in infeasible if b.branch_id in known}                                   # one learner per distinct question
    assert len(lab.registry.as_of("2100-01-01")) >= len(distinct) > 0
    by = {s["stage"]: s for s in reps[-1]["stages"]}
    assert by["priorities.failed_lab_sync"]["status"] == "OK" and by["priorities.failed_lab_sync"]["n_in"] > 0
    ranked = FL.question_already_failed(lab, infeasible[0].text, infeasible[0].problem.value, reps[-1]["now"], infeasible[0].family)
    assert not ranked.decision.permits                                                   # a registered line is refused on reuse


# ============================================================================================================ 3. XOR / interaction pre-screen
def _world(kind, seed, n=800, n_feat=20, rate=0.55, base=0.12):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        f = {f"f{i}": float(rng.uniform(0, 1)) for i in range(n_feat)}
        a, b = f["f0"] > 0.5, f["f1"] > 0.5
        hot = {"xor": a != b, "and": a and b, "null": False}[kind]
        rows.append(T.PredictionRow("p", not (rng.random() < (rate if hot else base)), f))
    return rows


def _has_pair(found):
    return any({"f0", "f1"} <= set(c.features) and not (set(c.features) - {"f0", "f1"}) for c in found)


def test_planted_xor_is_found_by_the_pair_prescreen_and_was_lost_without_it():
    with_seeds = sum(_has_pair(T.find_conjunctions(_world("xor", s))) for s in range(6))
    without = sum(_has_pair(T.find_conjunctions(_world("xor", s), max_pair_seeds=0, reachable_space=False)) for s in range(6))
    assert with_seeds >= 5, with_seeds
    assert without <= with_seeds - 3, f"old pre-screen {without}/6 vs new {with_seeds}/6: the planted case no longer exposes the defect"
    top = T.find_conjunctions(_world("xor", 0))[0]
    assert top.rate_in > 0.3 > top.rate_out and top.p_value < 0.05                        # a real failure cell, not noise


def test_planted_and_pattern_is_still_found():
    assert sum(_has_pair(T.find_conjunctions(_world("and", s))) for s in range(6)) == 6


def test_null_worlds_are_not_admitted_and_cost_is_bounded():
    admitted = sum(bool(T.find_conjunctions(_world("null", 500 + s))) for s in range(30))
    assert admitted <= 2, f"{admitted}/30 null worlds produced a region"
    st = {}
    T.find_conjunctions(_world("xor", 1), stats=st)
    assert st["pair_tests"] == 4 * comb(20, 2) and st["pair_seeds"] <= 6                  # the pair screen is O(features^2), 6 seeds evaluated
    assert st["evaluated"] <= 40000 and st["tried"] >= st["pair_tests"] + st["screened"]  # and every test of it is charged to the correction
    off = {}
    T.find_conjunctions(_world("xor", 1), stats=off, max_pair_seeds=0)
    assert off["pair_tests"] == 0 and off["pair_seeds"] == 0


def test_pair_prescreen_degenerate_inputs():
    assert T.find_conjunctions([]) == []
    one = [T.PredictionRow("p", i % 7 != 0, {"only": float(i % 10)}) for i in range(100)]
    assert isinstance(T.find_conjunctions(one), list)                                     # a single feature has no pair to screen
    assert T._interaction_seeds(one, ["only"], 3, 3, 6) == ([], 0)
    seeds, n = T._interaction_seeds(_world("xor", 2, n=300, n_feat=4), ["f0", "f1", "f2", "f3"], 3, 3, 3)
    assert n == 4 * 6 and len(seeds) <= 3 and all(len(s) == 2 for s in seeds)
    assert {s[0].feature for s in seeds} | {s[1].feature for s in seeds} >= {"f0", "f1"}
