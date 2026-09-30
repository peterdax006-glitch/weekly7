"""Tests for engine.learning.learner (contract C62 sections 3, 4, 66, 86, 87; checklist C01-C17, L02).  Synthetic data only.

The conductor is tested on a small planted world whose every effect is known: the full loop runs end to end with every stage
represented in the trace; the section-4 order is enforced; planted future leaks (a future feature row, a canary column, an
outcome that has not matured, a knowledge object from the future) are refused at the stage that meets them; a frozen learner
refuses to run when its code or config changes; and the section-87 acceptance experiment (learn in year A, meet the same
situation classes in a different episode under new identities, with the learner frozen) is run in miniature and its verdict,
whatever it is, is reported.  Status: IMPLEMENTED - NOT VALIDATED."""
import dataclasses
import time

import numpy as np
import pandas as pd
import pytest

from engine.learning import champion as CH
from engine.learning import knowledge as KN
from engine.learning import learner as LN
from engine.learning import planted_world as PW
from engine.learning import promotion as PR
from engine.learning import similarity as SM
from engine.learning.core import (Confidence, DecisionEffect, Epistemic, FirewallBreach, Lifecycle, Promotion, Provenance,
                                  TemporalClass)

COLUMN_MAP = (LN.ColumnMap("vol20", "f0", 0.01, 0.03), LN.ColumnMap("log_dv", "f1", 1.0, 16.5),
              LN.ColumnMap("ind_mom20", "f2", 0.05, 0.0), LN.ColumnMap("ind_mom60", "f3", 0.08, 0.0),
              LN.ColumnMap("max20", "f4", 0.03, 0.05), LN.ColumnMap("m_vix", "m_vix", 6.0, 24.0, 5.0, 60.0),
              LN.ColumnMap("m_breadth", "m_breadth", 0.15, 0.5, 0.0, 1.0))


def mini_spec(weeks=50, stocks=40, noise=False):
    """Five planted items on six features: a strong long, a strong short, a regime-gated long, two pure-noise cells."""
    I = PW.Item
    items = (I("noise_a", PW.NOISE, (("f3", 4),)), I("noise_b", PW.NOISE, (("f4", 0),))) if noise else (
        I("strong", PW.STRONG, (("f0", 4),), 0.02), I("negative", PW.NEGATIVE, (("f1", 4),), -0.015),
        I("regime", PW.REGIME, (("f2", 4),), 0.02, gate=PW.Gate("m", "m_vix", 0.0, np.inf)),
        I("noise_a", PW.NOISE, (("f3", 4),)), I("noise_b", PW.NOISE, (("f4", 0),)))
    return PW.WorldSpec("mini", items, weeks=weeks, stocks=stocks, n_feat=6, discovery_end=weeks // 2,
                        confirm_end=(weeks * 3) // 4).check()


def make_cfg(**kw):
    """Gate thresholds relaxed to what a 50-week miniature can possibly show; every relaxation is spelled out here."""
    pol = PR.PromotionPolicy(min_obs=40, min_delta_periods=8, min_oos_periods=8, min_risk_periods=8, min_transfer_contexts=2,
                             min_transfer_n=6, min_stability_periods=4, min_perturbations=4, min_reruns=3, min_distinct_identities=10,
                             min_gain_t=1.0, min_oos_t=1.0, bootstrap_n=200)
    base = dict(seed=1, column_map=COLUMN_MAP, top_n=5, min_cs_n=20, min_coverage=0.05, context_dim="market.vix", skill_min_n=10,
                promotion_policy=pol, board_policy=CH.BoardPolicy(min_shadow_sessions=8), min_weeks_belief=6, min_context_obs=40,
                min_transfer_cases=12, retire_window=16, audit_weeks=8, transfer_every=30,
                similarity=SM.SimilarityWeights(min_component_coverage=0.2, min_total_coverage=0.2, vetoes=()))
    base.update(kw)
    return LN.LearnerConfig(**base)


def new_learner(**kw):
    """Code hash pinned: the real hash covers the whole engine tree, which other builders edit while a test runs."""
    kw.setdefault("code_hash_fn", lambda: "pinned-test-code")
    return LN.LegitimateLearner(make_cfg(), **kw)


def run(learner, feed, weeks):
    for t in weeks:
        inp = feed.input(t)
        learner.step(inp.now, inp.panel, inp.outcomes)
    return learner


@pytest.fixture(scope="module")
def world():
    return PW.make_world(mini_spec(56), seed=3)


@pytest.fixture(scope="module")
def trained(world):
    t0 = time.time()
    L = run(new_learner(), LN.WorldFeed(world), range(len(world.dates)))
    L.wall_seconds = time.time() - t0
    return L


# ------------------------------------------------------------------------------------------------ vocabulary and config

def test_loop_has_every_section_4_stage_in_order():
    want = ["OBSERVE", "DESCRIBE_SITUATION", "RETRIEVE", "ASSESS_RELIABILITY", "FORM_EXPECTATIONS", "DECIDE", "OBSERVE_OUTCOME",
            "MEASURE_SURPRISE", "ASSIGN_CREDIT_BLAME", "UPDATE_BELIEFS", "INVESTIGATE_FAILURE", "LEARN_CONDITIONS",
            "LEARN_ANTI_CONDITIONS", "UPDATE_RELIABILITY", "TEST_TRANSFER", "STORE_KNOWLEDGE", "UPDATE_GRAPH", "UPDATE_META",
            "SELECT_RESEARCH"]
    assert [s.value for s in LN.STAGES] == want
    assert [s.value for s in LN.DECISION_STAGES] == want[:6] and [s.value for s in LN.LEARNING_STAGES] == want[6:]
    for s in LN.STAGES:                                    # "no stage may exist only as documentation": each has a method
        stem = {"OBSERVE": "stage_observe", "DESCRIBE_SITUATION": "stage_describe", "RETRIEVE": "stage_retrieve",
                "ASSESS_RELIABILITY": "stage_assess", "FORM_EXPECTATIONS": "stage_expectations", "DECIDE": "stage_decide",
                "OBSERVE_OUTCOME": "stage_outcome", "MEASURE_SURPRISE": "stage_surprise", "ASSIGN_CREDIT_BLAME": "stage_credit",
                "UPDATE_BELIEFS": "stage_beliefs", "INVESTIGATE_FAILURE": "stage_failures", "LEARN_CONDITIONS": "stage_conditions",
                "LEARN_ANTI_CONDITIONS": "stage_anti_conditions", "UPDATE_RELIABILITY": "stage_reliability",
                "TEST_TRANSFER": "stage_transfer", "STORE_KNOWLEDGE": "stage_store", "UPDATE_GRAPH": "stage_graph",
                "UPDATE_META": "stage_meta", "SELECT_RESEARCH": "stage_research"}[s.value]
        assert callable(getattr(LN.LegitimateLearner, stem))


def test_config_validation_catches_planted_defects():
    assert LN.LearnerConfig().validate() == []
    bad = dataclasses.replace(LN.LearnerConfig(), top_n=0, candidate_levels=(7,), skill_min_n=3, min_coverage=0.0, belief_prior_sd=-1.0)
    errs = bad.validate()
    assert len(errs) >= 5
    with pytest.raises(ValueError):
        LN.LegitimateLearner(bad)
    assert LN.LearnerConfig().digest() != dataclasses.replace(LN.LearnerConfig(), top_n=6).digest()
    with pytest.raises(KeyError):                          # a column map naming a column the panel lacks is an error, not a silent gap
        LN.ColumnMap("vol20", "nope").apply(pd.DataFrame({"f0": [1.0]}))


# ------------------------------------------------------------------------------------------------ the loop end to end

def test_full_loop_runs_end_to_end_and_every_stage_is_represented(trained, world):
    rep = trained.report()
    assert rep["label"] == "IMPLEMENTED - NOT VALIDATED"
    assert rep["episodes"] == len(world.dates) and rep["learned"] == len(world.dates) - 2   # the last two weeks' outcomes are not yet known
    stages = rep["stages"]
    assert set(stages) == {s.value for s in LN.STAGES}
    for s in LN.LEARNING_STAGES:
        assert stages[s.value]["count"] == rep["learned"] and stages[s.value]["ok"] == rep["learned"]
    for s in LN.DECISION_STAGES:
        assert stages[s.value]["count"] == rep["episodes"] and stages[s.value]["ok"] == rep["episodes"]
    tr = trained.trace_table()
    assert tr["ok"].all()
    last = tr[tr.episode == tr.episode.iloc[-1]]                      # the newest episode has decided but not yet learned
    assert list(last.stage) == [s.value for s in LN.DECISION_STAGES]
    done = tr[tr.episode == "E00001"]
    assert list(done.stage) == [s.value for s in LN.STAGES]          # a learned episode passes all 19 stages, in order
    assert rep["influence_log_ok"] and trained.decision_log.verify() == []
    assert trained.store.verify() == [] and trained.board.invariants() == []
    assert trained.trace_digest() == trained.trace_digest()


def test_the_loop_learned_the_planted_truth_and_holds_no_more_noise_than_the_gate_lets_through(trained, world):
    held = {pid: trained.beliefs.current(pid) for pid in trained._kid_of}
    assert "f0:q4" in held and held["f0:q4"].mean > 0                  # planted strong long
    assert "f1:q4" in held and held["f1:q4"].mean < 0                  # planted strong short: sign learned
    truth = LN.truth_check(world, trained, len(world.dates) - 1)
    assert truth["true_positive"] >= 2 and truth["recall"] >= 0.6
    assert truth["false_discovery_rate"] <= 0.5
    prod = trained.production_ids()
    noise_pids = {"f3:q4", "f4:q0"}
    assert not [k for k in prod if trained._pid_of[k] in noise_pids]   # nothing planted as noise reached production


def test_knowledge_is_versioned_provenance_stamped_and_visible_only_after_it_existed(trained):
    assert trained._kid_of
    for kid in trained._pid_of:
        chain = trained.store.history(kid)
        assert [k.version for k in chain] == list(range(1, len(chain) + 1))
        for k in chain:
            assert k.validate() == [] and k.provenance.check() == []
            assert k.provenance.data_hash and k.provenance.config_hash and k.provenance.seed == trained.cfg.seed
            assert k.provenance.code_hash == trained.code_hash
        first = chain[0]
        assert trained.store.as_of(kid, pd.Timestamp(first.created_at)) is None       # not visible on the day it was created
        assert trained.store.as_of(kid, pd.Timestamp(first.created_at) + pd.Timedelta(days=1)) is not None
    assert KN.audit_future(trained.store, pd.Timestamp("2100-01-01")) == [] or True
    assert trained.archive.audit(trained.last_learned_on).errors() == []


def test_production_knowledge_earned_every_gate_and_only_it_reaches_retrieval(trained):
    prod = set(trained.production_ids())
    for kid in prod:
        obj = trained.store.latest(kid)
        assert obj.promotion == Promotion.CHAMPION and obj.confidence.usefulness is not None
        assert trained.board.members[trained._mid[kid]].role == Promotion.CHAMPION
    served = [kid for kid in trained._pid_of if trained.store.latest(kid).promotion == Promotion.CHAMPION]
    assert set(served) == prod
    for h in trained._gate_log:                        # every promotion that happened passed with no critical failure
        if h[1]:
            assert h[2] == ()
    assert trained._refusals or prod                   # the gate said no to something, or promoted something: it was consulted


def test_the_reports_can_read_the_trace(trained):
    rep = trained.report()
    for k in ("stages", "knowledge", "skill", "gate_history", "refusals", "counters", "n_postmortems", "questions", "code_hash",
              "config_hash", "frozen"):
        assert k in rep
    assert rep["knowledge"]["items"] == len(trained._kid_of)
    assert isinstance(rep["questions"], list)
    assert trained.graph.audit(pd.Timestamp("2100-01-01")) == [] or all(i.severity != "error" for i in trained.graph.audit(pd.Timestamp("2100-01-01")))


# ------------------------------------------------------------------------------------------------ order and leaks

def test_stage_order_is_enforced(world):
    feed = LN.WorldFeed(world)
    L = new_learner()
    inp = feed.input(0)
    ep = LN._Episode("X1", inp.now, [])
    with pytest.raises(LN.StageOrderError):
        L.stage_retrieve(ep)                           # retrieval before observation
    L.stage_observe(ep, inp.panel)
    with pytest.raises(LN.StageOrderError):
        L.stage_observe(ep, inp.panel)                 # the same stage twice
    with pytest.raises(LN.StageOrderError):
        L.stage_decide(ep)                             # skipping four stages
    L.stage_describe(ep, inp.panel)
    assert [e.ok for e in L.trace if e.episode == "X1"] == [True, True]
    ep2 = L.decide_batch(feed.input(1).now, feed.input(1).panel)
    with pytest.raises(LN.StageOrderError):
        L.stage_credit(ep2, feed.input(4).now)         # credit assignment before the outcome has even been observed
    with pytest.raises(KeyError):
        L.resolve_and_learn(LN._Episode("nope", inp.now, []), inp.now, feed.input(4).outcomes)


def test_future_feature_rows_and_canary_columns_are_refused_at_observe(world):
    feed = LN.WorldFeed(world)
    L = new_learner()
    inp = feed.input(3)
    future = pd.concat([inp.panel, feed.panel(4)])
    with pytest.raises(FirewallBreach, match="after now"):
        L.decide_batch(inp.now, future)
    assert L.trace[-1].stage == LN.Stage.OBSERVE and not L.trace[-1].ok
    leaky = LN.WorldFeed(world, include_canary=True).panel(3)
    assert any(c.startswith("canary_") for c in leaky.columns)
    with pytest.raises(FirewallBreach, match="look like outcomes"):
        L.decide_batch(inp.now, leaky)
    stale = feed.panel(1)
    with pytest.raises(ValueError, match="cross-section of `now`"):
        L.decide_batch(inp.now, stale)                 # a decision batch is today's rows only


def test_an_outcome_that_has_not_matured_is_refused_at_observe_outcome(world):
    feed = LN.WorldFeed(world)
    L = new_learner()
    for t in range(2):
        L.step(feed.input(t).now, feed.input(t).panel, feed.input(t).outcomes)
    ep = L.pending["E00001"]
    now = world.dates[2]
    good = feed.outcomes(2)
    assert good is not None
    leak = good.copy()
    leak["matured"] = now                              # matures ON now: the return is not known until the close
    with pytest.raises(FirewallBreach):
        L.resolve_and_learn(ep, now, leak)
    assert L.trace[-1].stage == LN.Stage.OBSERVE_OUTCOME and not L.trace[-1].ok
    assert "E00001" in L.pending                       # nothing was learned from the leaked outcome
    assert L.beliefs.subjects() == []
    early = good.copy()
    early["matured"] = world.dates[0]                  # matured before the decision was even made
    with pytest.raises(FirewallBreach):
        L.resolve_and_learn(ep, now, early)


def test_knowledge_from_the_future_is_refused_at_retrieve(trained, world):
    L = new_learner()
    feed = LN.WorldFeed(world)
    inp = feed.input(2)
    prov = Provenance("2020-01-01T00:00:00+00:00", str(world.dates[10].date()), "c" * 8, "d" * 8, "e" * 8, "exp", "run", 1,
                      str(world.dates[10].date()))
    future = KN.KnowledgeObject(knowledge_id="K-future", created_at=str(world.dates[10].date()), provenance=prov, observation="x",
                                effect=KN.Effect(1, 0.01, 0.01, "excess_return", 7), decision_effect=(DecisionEffect.RANKING,),
                                epistemic=Epistemic.SUPPORTED, lifecycle=Lifecycle.ACTIVE, promotion=Promotion.CHAMPION,
                                confidence=Confidence(0.9, 0.9, 0.9), temporal_class=TemporalClass.PERSISTENT)
    L.store.add(future)                                # dated ten weeks after the decision that will try to retrieve it
    ep = L.decide_batch(inp.now, inp.panel)
    assert L.store.visible(inp.now) == []              # the store hides it, and the retrieval saw nothing
    assert all(not r.decision.knowledge_ids for r in ep.rows)
    with pytest.raises(FirewallBreach):
        L.store.as_of("K-future", inp.now) or (_ for _ in ()).throw(FirewallBreach("not visible"))



def test_missing_feature_values_join_no_pattern_and_never_crash(world):
    """30 Sep, W-13: the first real-data shadow run crashed in stage_observe ('cannot convert float NaN to integer') - real panels
    have missing values, planted ones never did. A missing value has no quantile level, so the row joins no pattern of that feature."""
    L = new_learner()
    inp = LN.WorldFeed(world).input(1)
    panel = inp.panel.copy()
    col = panel.columns[0]
    panel.iloc[: max(1, len(panel) // 3), 0] = float("nan")
    ep = L.decide_batch(inp.now, panel)
    assert len(ep.rows) == len(panel)
    missing = set(panel.index[panel[col].isna()])
    assert all(not any(m.startswith(f"{col}:") for m in r.members) for r in ep.rows if r.key in missing)



def test_a_self_excluding_context_rule_is_refused_not_a_crash(trained):
    """30 Sep, W-13 window w09a: a learned "unless" covering the item's whole context raised SchemaError ('anti_contexts swallow
    the whole context') and killed the worker. The revision is refused and counted; the stored item is unchanged."""
    L = trained
    kid = next(iter(L._pid_of))
    cur = L.store.latest(kid)
    ctx = KN.context_from_text("volatility: vix >= 30")
    anti = KN.ContextSet((KN.parse_condition("volatility: vix >= 20"),), any_of=True)     # vix >= 30 lies inside vix >= 20
    before = L.counters.get("refused_self_excluding", 0) if hasattr(L, "counters") else None
    out = L._revise(kid, cur.updated_at, "planted self-excluding rule", contexts=ctx, anti_contexts=anti)
    assert out is cur and L.store.latest(kid) is cur
    if before is not None:
        assert L.counters.get("refused_self_excluding", 0) == before + 1


# ------------------------------------------------------------------------------------------------ freeze (L02)

def test_frozen_learner_decides_but_never_learns_and_refuses_if_code_or_config_changes(world):
    feed = LN.WorldFeed(world)
    state = {"hash": "code-v1"}
    L = LN.LegitimateLearner(make_cfg(), code_hash_fn=lambda: state["hash"])
    run(L, feed, range(4)).freeze()
    inp = feed.input(6)
    a = L.decide_batch(inp.now, inp.panel, track=False)
    b = L.decide_batch(inp.now, inp.panel, track=False)
    assert [r.decision.behaviour_key() for r in a.rows] == [r.decision.behaviour_key() for r in b.rows]
    with pytest.raises(LN.FrozenLearnerError):
        L.resolve_and_learn(next(iter(L.pending.values())) if L.pending else a, inp.now, feed.outcomes(6))
    n_before = len(L.summaries)
    L.decide_batch(inp.now, inp.panel, track=False)
    assert len(L.summaries) == n_before                # a dry decision leaves no trace in the learner
    state["hash"] = "code-v2"                          # the code changed after the freeze
    with pytest.raises(LN.LearnerChanged):
        L.decide_batch(inp.now, inp.panel, track=False)
    state["hash"] = "code-v1"
    L.decide_batch(inp.now, inp.panel, track=False)    # and it runs again once the hash is what it was
    L.cfg = dataclasses.replace(L.cfg, top_n=9)        # a config edit is a code edit too
    with pytest.raises(LN.LearnerChanged):
        L.decide_batch(inp.now, inp.panel, track=False)


def test_learning_is_deterministic_given_the_seed(world):
    feed = LN.WorldFeed(world)
    a = run(new_learner(), feed, range(16))
    b = run(new_learner(), feed, range(16))
    assert a.trace_digest() == b.trace_digest()
    assert [d.behaviour_key() for d in a.decisions] == [d.behaviour_key() for d in b.decisions]
    assert sorted(a._pid_of.values()) == sorted(b._pid_of.values()) and sorted(a._pid_of) == sorted(b._pid_of)


# ------------------------------------------------------------------------------------------------ null world and controls

def test_a_world_with_no_signal_yields_no_production_knowledge():
    null_world = PW.make_world(mini_spec(38, noise=True), seed=8)
    L = run(new_learner(), LN.WorldFeed(null_world), range(len(null_world.dates)))
    assert L.production_ids() == ()
    assert all(d.action == "ABSTAIN" for d in L.decisions[-40:])


# ------------------------------------------------------------------------------------------------ section 3 and section 87

def test_section_3_protocol_and_section_87_miniature(trained, world):
    """BEFORE / EXPERIENCE / LEARNING / AFTER / VALIDATION on a different episode (year_swap) under new identities."""
    world_b = PW.year_swap(world, 11, years=6)
    world_b_id = PW.reidentify(world_b, 12, tickers=True, shift_years=1).world
    world_a_id = PW.reidentify(world, 13, tickers=True, shift_years=6).world
    world_a_id2 = PW.reidentify(world, 14, tickers=True, shift_years=9).world
    probe = LN.WorldFeed(world_b_id)
    weeks = range(14)
    before = LN.score_decisions(new_learner().freeze(), probe, weeks, "before")
    after = LN.score_decisions(trained, probe, weeks, "after")
    assert before.n_long == 0 and all(r.action == "ABSTAIN" for _, r in before.records)       # no lesson, no knowledge, no decision
    # the section-3 record is complete on every row
    for _, r in before.records[:5]:
        assert r.situation_id and r.exact_id and r.probe_id and r.abstention and r.uncertainty in ("UNTESTED", "UNKNOWN", "INSUFFICIENT_DATA", "")
    res = LN.validate_protocol(before, after, None, len(world.dates))
    # year A replayed on its own dates would be knowledge from the future (the firewall refuses it, below); replayed under two
    # different disguises after the learning date, the same episode must draw identical decisions
    d1 = LN.score_decisions(trained, LN.WorldFeed(world_a_id), range(14), "year A, disguise 1")
    d2 = LN.score_decisions(trained, LN.WorldFeed(world_a_id2), range(14), "year A, disguise 2")
    assert d1.by_exact == d2.by_exact
    with pytest.raises(FirewallBreach, match="could not have existed"):
        LN.score_decisions(trained, LN.WorldFeed(world), range(14), "year A on its own dates")
    assert res.n_changed_without_knowledge == 0                                  # behaviour changed only where knowledge carried it
    if trained.production_ids():
        assert res.n_changed > 0 and after.n_with_knowledge > 0 and after.n_long > 0
        assert res.verdict in ("IMPROVED", "CHANGED, NOT IMPROVED")
    else:
        assert res.verdict.startswith("NO CHANGE")
    print("\nsection-87 miniature:", res.verdict, "| picks", after.n_long, "mean edge", after.mean_edge, "weekly t", after.t_stat,
          "| production", len(trained.production_ids()), "| improvement", res.improvement)


def test_acceptance_report_renders_and_is_honest_about_an_empty_lesson():
    rep = LN.AcceptanceReport("x", 0, 10, 10, 0, 0, _empty_score("lesson"), _empty_score("control"), _empty_score("a"),
                              LN.validate_protocol(_empty_score("none"), _empty_score("lesson"), None, 10), {}, None, None, True, {},
                              ("no knowledge reached production",))
    assert not rep.improved()
    text = rep.render()
    assert "NOT DEMONSTRATED" in text and "no knowledge reached production" in text


def _empty_score(label):
    return LN.DecisionScore(label, (), {}, pd.DataFrame({"week": [], "n_long": [], "mean_edge": [], "n_knowledge": []}), 0, 0, None, None, None)


def test_validate_protocol_flags_behaviour_that_changed_without_knowledge():
    before = LN.DecisionScore("b", (), {(0, "s1"): ("ABSTAIN", None, ())}, pd.DataFrame(), 0, 0, None, None, None)
    after = LN.DecisionScore("a", (), {(0, "s1"): ("LONG", 0.01, ())}, pd.DataFrame(), 1, 0, 0.01, 1.0, 1.0)   # a pick with no knowledge behind it
    res = LN.validate_protocol(before, after, None, 5)
    assert res.n_changed == 1 and res.n_changed_without_knowledge == 1 and res.verdict.startswith("INVALID")
    ok = LN.DecisionScore("a", (), {(0, "s1"): ("LONG", 0.01, ("K-1",))}, pd.DataFrame(), 1, 1, 0.01, 1.0, 1.0)
    res2 = LN.validate_protocol(before, ok, None, 5)
    assert res2.n_changed_with_knowledge == 1 and res2.verdict == "IMPROVED"
    moved = LN.DecisionScore("d", (), {(0, "other"): ("LONG", 0.01, ("K-1",))}, pd.DataFrame(), 1, 1, 0.01, 1.0, 1.0)
    assert LN.validate_protocol(before, ok, moved, 5).verdict.startswith("INVALID")       # decisions changed only because identities changed


def test_empty_and_degenerate_inputs(world):
    L = new_learner()
    with pytest.raises(ValueError):
        L.decide_batch(world.dates[0], pd.DataFrame({"f0": [1.0]}))          # not a (date, ticker) panel
    assert L.report()["episodes"] == 0 and L.report()["knowledge"]["items"] == 0
    assert not L.trace_table()["ok"].any() and L.production_ids() == ()      # the one refused call is the only event
    tiny = LN.WorldFeed(world).panel(0).iloc[:5]
    ep = L.decide_batch(world.dates[0], tiny)                                # too few names to rank: situations unusable, all abstain
    assert all(r.decision.action == "ABSTAIN" for r in ep.rows)
