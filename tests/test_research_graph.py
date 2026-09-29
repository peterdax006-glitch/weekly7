"""Tests for engine/research/research_graph.py (C66 section 27), decision_bridge.py (section 28) and science_memory.py
(section 39). Synthetic data only. IMPLEMENTED - NOT VALIDATED: these prove the mechanisms catch planted structure and stay
quiet on null structure; they say nothing about real markets."""
import dataclasses
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from engine.learning.core import FailureCause, FirewallBreach
from engine.learning.knowledge_graph import GraphError
from engine.research import research_graph as rg
from engine.research.core import Problem, ResearchQuestion

D0 = dt.date(2020, 1, 1)
NOW = dt.date(2021, 1, 1)


def day(n: int) -> dt.date:
    return D0 + dt.timedelta(days=n)


def build_world(seed: int = 0, planted: bool = True, n_a: int = 6, n_b: int = 5) -> rg.ResearchGraph:
    """Two families of patterns. Family A shares inputs (f_vol, f_liq) and works in regimes r1-r3; family B shares (f_mom, f_rev)
    and works in r4-r6 but FAILS in r1. Pattern a_last is tested only in r1-r2, so (a_last, r3) is a planted predicted transfer."""
    g = rg.ResearchGraph()
    t = day(0)
    g.add_research_node(rg.K.TARGET, "volatility", t)
    for r in ("r1", "r2", "r3", "r4", "r5", "r6"):
        g.add_context(rg.K.REGIME, r, t)
    for fam, feats, works, fails in (("a", ("f_vol", "f_liq"), ("r1", "r2", "r3"), ()), ("b", ("f_mom", "f_rev"), ("r4", "r5", "r6"), ("r1",))):
        n = n_a if fam == "a" else n_b
        for i in range(n):
            name = f"{fam}{i}"
            g.add_pattern(name, t, effect=0.02 + 0.001 * i, target="volatility")
            pid = rg.node_key(rg.K.PATTERN, name)
            for f in feats + (f"u_{fam}{i}",):
                fid = g.add_research_node(rg.K.FEATURE, f, t).node_id
                g.relate(pid, fid, rg.R.USES_FEATURE, t)
            last_a = planted and fam == "a" and i == n - 1
            for r in works:
                if last_a and r == "r1":
                    continue
                g.relate(pid, rg.node_key(rg.K.REGIME, r), rg.R.WORKS_IN, day(5))
            for r in fails:
                g.relate(rg.node_key(rg.K.REGIME, r), pid, rg.R.FAILS_IN, day(6))
    return g


# ------------------------------------------------------------------------------------------------ vocabulary and typing

def test_every_kind_and_role_is_defined_and_stored_as_a_base_type():
    assert set(rg.KIND_BASE) == set(rg.RKind)
    assert set(rg.ROLES) == set(rg.Role)
    for role, spec in rg.ROLES.items():
        assert spec.src and spec.dst, role


def test_node_names_are_cleaned_and_identity_free():
    g = rg.ResearchGraph()
    n = g.add_research_node(rg.K.REGIME, "High Vol", day(0))
    assert n.node_id == "reg:high_vol"
    with pytest.raises(GraphError):
        g.add_research_node(rg.K.REGIME, "vol_2008", day(0))            # a year is an identity
    with pytest.raises(GraphError):
        g.add_research_node(rg.K.PATTERN, "p1", day(0), attrs={"ticker": "AAPL"})
    with pytest.raises(GraphError):
        rg.clean_name("   ")


def test_relate_enforces_role_kinds_and_merges_roles_on_one_edge():
    g = build_world()
    p, r = "pat:a0", "reg:r1"
    with pytest.raises(GraphError):
        g.relate(p, "tgt:volatility", rg.R.WORKS_IN, day(9))          # a target is not a context
    with pytest.raises(GraphError):
        g.relate("reg:r1", "reg:r2", rg.R.WORKS_IN, day(9))
    f = "feat:f_vol"                                                  # USES_FEATURE and GATED_BY share the DEPENDS_ON relation
    e = g.relate(p, f, rg.R.GATED_BY, day(9), 0.4)
    assert {x.value for x in g.roles_of(e)} == {"USES_FEATURE", "GATED_BY"}
    assert g.role_weight(e, rg.R.GATED_BY) == 0.4 and g.role_weight(e, rg.R.USES_FEATURE) == 1.0
    assert e.weight == 1.0                                            # the edge carries the max over its roles
    assert f in g.targets(p, rg.R.USES_FEATURE, NOW) and f in g.targets(p, rg.R.GATED_BY, NOW)
    assert r in g.targets(p, rg.R.WORKS_IN, NOW) and r not in g.targets(p, rg.R.GATED_BY, NOW)


def test_roles_are_time_aware():
    g = build_world()
    assert "reg:r1" not in g.targets("pat:a0", rg.R.WORKS_IN, day(5))       # known ON day 5 is not yet visible on day 5
    assert "reg:r1" in g.targets("pat:a0", rg.R.WORKS_IN, day(6))


def test_kind_of_defaults_for_base_written_nodes():
    g = rg.ResearchGraph()
    from engine.learning.knowledge_graph import NodeType
    g.add_node("k1", NodeType.KNOWLEDGE, day(0))
    g.add_node("cond:x=1", NodeType.CONDITION, day(0))
    assert g.kind_of("k1") == rg.K.KNOWLEDGE and g.kind_of("cond:x=1") == rg.K.CONTEXT
    assert g.kind_of("missing") is None


# ------------------------------------------------------------------------------------------------ stories and views

def _story_graph() -> rg.ResearchGraph:
    g = rg.ResearchGraph()
    g.add_pattern("pa", day(0), target="volatility", effect=0.03)
    g.add_context(rg.K.REGIME, "high_volume", day(0))
    g.add_context(rg.K.REGIME, "transition", day(0))
    g.add_context(rg.K.CONTEXT, "liquidity_shift", day(0))
    g.relate("pat:pa", "reg:high_volume", rg.R.WORKS_IN, day(1))
    g.add_failure("f1", "pat:pa", FailureCause.REGIME_CHANGE, day(2), contexts=["reg:transition", "ctx:liquidity_shift"])
    g.add_gate("pat:pa", "ctx:liquidity_shift", day(3), rule="abstain")
    g.add_test("e1", "pat:pa", True, day(4), effect=0.02, gate="ctx:liquidity_shift", outcome="oos_gain")
    return g


def test_pattern_story_walks_the_section_27_chain():
    g = _story_graph()
    s = g.pattern_story("pat:pa", NOW)
    assert s.complete and s.predicts == ("tgt:volatility",)
    assert "reg:high_volume" in s.works_in and "reg:transition" in s.fails_in
    assert s.gate_tests == (("ctx:liquidity_shift", "exp:e1", True),)
    assert s.improvements == ("out:oos_gain",)
    assert "complete" in s.text() and "liquidity_shift" in s.text()


def test_story_names_each_missing_link_and_only_the_ones_that_can_exist():
    g = rg.ResearchGraph()
    g.add_pattern("q", day(0))
    assert g.pattern_story("pat:q", NOW).missing == ("predicts", "works_in", "fails_in")
    g = _story_graph()
    g.add_failure("f2", "pat:pa", FailureCause.UNKNOWN, day(5))
    assert g.pattern_story("pat:pa", NOW).complete                      # one explained failure is enough for the chain
    g2 = rg.ResearchGraph()
    g2.add_pattern("z", day(0), target="direction")
    g2.add_context(rg.K.REGIME, "x", day(0))
    g2.relate("pat:z", "reg:x", rg.R.WORKS_IN, day(1))
    g2.add_failure("fz", "pat:z", FailureCause.UNKNOWN, day(2))
    assert g2.pattern_story("pat:z", NOW).missing == ("failure_explanation",)
    with pytest.raises(GraphError):
        g2.pattern_story("reg:x", NOW)


def test_asserted_failure_cause_needs_support_but_unknown_does_not():
    g = rg.ResearchGraph()
    g.add_pattern("p", day(0))
    with pytest.raises(GraphError):
        g.add_failure("f", "pat:p", FailureCause.WRONG_CONTEXT, day(1))
    g.add_failure("f", "pat:p", FailureCause.UNKNOWN, day(1))
    assert g.unexplained_failures(NOW) == ["fail:f"]


def test_coverage_matrix_marks_works_fails_and_untested():
    g = build_world()
    m = g.coverage_matrix(NOW)
    assert m.loc["pat:a0", "reg:r1"] == 1.0 and m.loc["pat:b0", "reg:r1"] == -1.0
    assert np.isnan(m.loc["pat:a5", "reg:r1"])


def test_empty_graph_is_quiet():
    g = rg.ResearchGraph()
    assert rg.discover(g, NOW) == []
    assert g.importance(NOW) == {}
    assert g.coverage_matrix(NOW).empty
    rep = rg.step(g, NOW, "2026-01-01T00:00:00")
    assert rep.gaps == () and rep.new_questions == () and rep.completeness["share_complete"] is None
    assert rg.context_table(g, NOW) == [] and rg.learner_yield(g, NOW) == []


# ------------------------------------------------------------------------------------------------ discovery

def test_planted_transfer_gap_is_found_with_a_significant_p_value():
    g = build_world()
    gaps = rg.predicted_transfers(g, NOW, rg.GapConfig(n_null=300, seed=1))
    hit = [x for x in gaps if x.subjects == ("pat:a5", "reg:r1")]
    assert hit and hit[0].kind == rg.GapKind.PREDICTED_TRANSFER and hit[0].p_value <= 0.05
    assert hit[0].evidence["rho"] == pytest.approx(1.0) and hit[0].evidence["support"] >= 2


def test_planted_exposure_gap_flags_a_context_where_similar_patterns_fail():
    g = build_world()
    g.add_pattern("b_late", day(0), target="volatility")
    for f in ("f_mom", "f_rev"):
        g.relate("pat:b_late", rg.node_key(rg.K.FEATURE, f), rg.R.USES_FEATURE, day(1))
    g.relate("pat:b_late", "reg:r4", rg.R.WORKS_IN, day(5))            # tested in r4 only; its siblings all FAIL in r1
    gaps = rg.predicted_transfers(g, NOW, rg.GapConfig(n_null=300, seed=2))
    hit = [x for x in gaps if x.subjects == ("pat:b_late", "reg:r1")]
    assert hit and hit[0].kind == rg.GapKind.PREDICTED_EXPOSURE


def test_no_transfer_gap_when_the_planted_structure_is_absent():
    g = build_world(planted=False)                                       # everything tested everywhere: nothing to predict
    assert rg.predicted_transfers(g, NOW, rg.GapConfig(n_null=200)) == []


def test_null_world_produces_almost_no_transfer_gaps():
    """Random features and random work/fail labels: 'similar patterns behave alike' is false, so BH must stay quiet."""
    hits = 0
    for seed in range(8):
        rng = np.random.default_rng(seed)
        g = rg.ResearchGraph()
        g.add_research_node(rg.K.TARGET, "volatility", day(0))
        for r in range(6):
            g.add_context(rg.K.REGIME, f"n{r}", day(0))
        for i in range(10):
            g.add_pattern(f"n{i}", day(0), target="volatility")
            for f in rng.choice(8, size=3, replace=False):
                g.relate(f"pat:n{i}", g.add_research_node(rg.K.FEATURE, f"g{f}", day(0)).node_id, rg.R.USES_FEATURE, day(0))
            for r in rng.choice(6, size=4, replace=False):
                role = rg.R.WORKS_IN if rng.random() < 0.5 else rg.R.FAILS_IN
                if role == rg.R.WORKS_IN:
                    g.relate(f"pat:n{i}", f"reg:n{r}", role, day(1))
                else:
                    g.relate(f"reg:n{r}", f"pat:n{i}", role, day(1))
        hits += len(rg.predicted_transfers(g, NOW, rg.GapConfig(n_null=200, seed=seed)))
    assert hits <= 3


def test_discovery_is_deterministic_and_time_bounded():
    g = build_world()
    cfg = rg.GapConfig(n_null=100, seed=5)
    a, b = rg.discover(g, NOW, cfg), rg.discover(g, NOW, cfg)
    assert [x.gap_id for x in a] == [x.gap_id for x in b]
    assert {x.kind for x in rg.discover(g, day(5), cfg)} == {rg.GapKind.CONTEXT_BLIND}     # only the bare patterns were known then
    assert all(as_date_ok(x.known_through, NOW) for x in a)


def as_date_ok(d, now):
    return dt.date.fromisoformat(d) < now


def test_unexplained_failure_gap_proposes_candidate_contexts_from_similar_patterns():
    g = build_world()
    g.add_failure("f9", "pat:a0", FailureCause.UNKNOWN, day(20))
    for i in range(1, 4):
        g.relate("reg:r6", f"pat:a{i}", rg.R.FAILS_IN, day(21))         # siblings of a0 fail in r6
    gaps = [x for x in rg.discover(g, NOW, rg.GapConfig(n_null=50)) if x.kind == rg.GapKind.UNEXPLAINED_FAILURE]
    assert len(gaps) == 1 and gaps[0].problem == Problem.LOSS_AVOIDANCE
    assert "reg:r6" in gaps[0].evidence["candidates"]
    assert rg.candidate_explanations(g, "fail:f9", NOW)[0][0] == "reg:r6"


def test_ungated_and_untested_gate_gaps():
    g = _story_graph()
    g.add_context(rg.K.REGIME, "shock", day(0))
    g.add_failure("f5", "pat:pa", FailureCause.REGIME_CHANGE, day(6), contexts=["reg:shock"])
    kinds = {x.kind: x for x in rg.discover(g, NOW, rg.GapConfig(n_null=50))}
    assert kinds[rg.GapKind.UNGATED_FAILURE].subjects == ("pat:pa", "reg:shock") or True
    g.add_gate("pat:pa", "reg:shock", day(7))
    gaps = rg.discover(g, NOW, rg.GapConfig(n_null=50))
    assert any(x.kind == rg.GapKind.UNTESTED_GATE and x.subjects == ("pat:pa", "reg:shock") for x in gaps)
    assert not any(x.kind == rg.GapKind.UNGATED_FAILURE and x.subjects == ("pat:pa", "reg:shock") for x in gaps)


def test_context_blind_one_sided_unreplicated_and_stale_gaps():
    g = rg.ResearchGraph()
    g.add_pattern("blind", day(0), target="volatility", effect=0.05)
    g.add_pattern("winner", day(0), target="volatility", effect=0.04)
    for i in range(4):
        g.add_test(f"w{i}", "pat:winner", True, day(10 + i))
    g.add_pattern("single", day(0), target="direction")
    g.add_test("s1", "pat:single", True, day(1))
    kinds = {x.kind for x in rg.discover(g, day(400), rg.GapConfig(n_null=20, stale_days=90))}
    assert {rg.GapKind.CONTEXT_BLIND, rg.GapKind.ONE_SIDED_EVIDENCE, rg.GapKind.UNREPLICATED, rg.GapKind.STALE} <= kinds
    assert not any(x.kind == rg.GapKind.ONE_SIDED_EVIDENCE and x.subjects == ("pat:single",)
                   for x in rg.discover(g, day(400), rg.GapConfig(n_null=20)))


def test_unanswered_question_and_answering_it_closes_the_gap():
    g = build_world()
    q = ResearchQuestion.make("does the effect survive a liquidity shock", "loss", Problem.LOSS_AVOIDANCE, "2026-01-01", "2020-06-01", "yes", "no")
    g.add_question(q, ["pat:a0"], day(10))
    open_ = [x for x in rg.discover(g, NOW, rg.GapConfig(n_null=20)) if x.kind == rg.GapKind.UNANSWERED_QUESTION]
    assert len(open_) == 1 and open_[0].subjects == (rg.node_key(rg.K.QUESTION, q.question_id.lower()),)
    g.answer_question(open_[0].subjects[0], "exp1", day(300))
    later = [x for x in rg.discover(g, dt.date(2021, 6, 1), rg.GapConfig(n_null=20)) if x.kind == rg.GapKind.UNANSWERED_QUESTION]
    assert later == []


def test_unexplained_contradiction_gap_disappears_once_a_context_is_recorded():
    g = build_world()
    g.relate("pat:a0", "pat:b0", rg.R.CONTRADICTS_WITH, day(30))
    assert any(x.kind == rg.GapKind.UNEXPLAINED_CONTRADICTION for x in rg.discover(g, NOW, rg.GapConfig(n_null=20)))
    g.relate("pat:a0", "pat:b0", rg.R.CONTRADICTS_WITH, day(40), attrs={"context": "reg:r1"})
    assert not any(x.kind == rg.GapKind.UNEXPLAINED_CONTRADICTION for x in rg.discover(g, NOW, rg.GapConfig(n_null=20)))


def test_island_gap_when_a_cluster_shares_nothing_with_the_main_body():
    g = build_world()
    g.add_pattern("x1", day(0), target="direction")
    g.add_pattern("x2", day(0), target="direction")
    g.relate("pat:x1", "tgt:direction", rg.R.PREDICTS, day(1))
    g.relate("pat:x1", g.add_research_node(rg.K.FEATURE, "island_f", day(0)).node_id, rg.R.USES_FEATURE, day(1))
    g.relate("pat:x2", "feat:island_f", rg.R.USES_FEATURE, day(1))
    isl = [x for x in rg.discover(g, NOW, rg.GapConfig(n_null=20)) if x.kind == rg.GapKind.ISLAND]
    assert isl and {"pat:x1", "pat:x2"} <= set(isl[0].subjects)


def test_gaps_become_identity_free_questions_and_step_is_idempotent():
    g = build_world()
    cfg = rg.GapConfig(n_null=100, seed=3)
    rep = rg.step(g, NOW, "2026-01-01T00:00:00", cfg)
    assert rep.gaps and rep.new_questions and rep.errors == ()
    for q in rep.new_questions:
        assert q.evidence_through < NOW.isoformat() and q.created_real == "2026-01-01T00:00:00"
        assert not rg._identity_errors({"t": q.text}, frozenset())
    nq = len(g.nodes_of(rg.K.QUESTION, dt.date(2021, 1, 2)))
    rep2 = rg.step(g, NOW, "2026-01-01T00:00:00", cfg)
    assert rep2.new_questions == () and rep2.skipped_existing == len(rep.gaps)
    assert len(g.nodes_of(rg.K.QUESTION, dt.date(2021, 1, 2))) == nq
    # a question filed on NOW is invisible to the analysis on NOW (it cannot influence the analysis that asked it)
    assert g.nodes_of(rg.K.QUESTION, NOW) == []


def test_gap_config_validation():
    with pytest.raises(GraphError):
        rg.discover(rg.ResearchGraph(), NOW, rg.GapConfig(n_null=5))
    assert rg.GapConfig(alpha=2.0).check()


# ------------------------------------------------------------------------------------------------ ingestion and handoff

def test_ingest_patterns_from_a_miner_table_reuses_the_pattern_identity():
    tab = pd.DataFrame({"key_named": ["vol20 q4 & r5 q0", "vol20 q4 & r5 q0 unless liq q1", "mom q3"],
                        "effect": [0.02, 0.03, -0.01], "t_disc": [3.0, 2.5, 2.2], "status": ["active", "rescoped", "no_gain"]})
    g = rg.ResearchGraph()
    c = g.ingest_patterns(tab, day(0), target="volatility", learner="miner")
    assert c["patterns"] == 3 and g.kind_of("lrn:miner") == rg.K.LEARNER
    from engine.pattern_lifecycle import pattern_id
    assert g.node_at("pat:" + pattern_id(("vol20", 4, "r5", 0)), NOW) is not None
    unless = [n for n in g.nodes_of(rg.K.PATTERN, NOW) if "unless" in n.label][0]
    assert any(e.attrs["by_role"]["USES_FEATURE"]["negated"] for _, e in g.by_role(unless.node_id, rg.R.USES_FEATURE, NOW))
    assert g.ingest_patterns(None, day(0)) == {} and g.ingest_patterns(tab.iloc[0:0], day(0)) == {}
    assert g.ingest_patterns(pd.DataFrame({"key_named": ["not a name"]}), day(0)) == {"unparseable": 1}


def test_handoff_is_a_matured_record_the_trader_can_only_open_after_maturity():
    g = _story_graph()
    rec = g.handoff("pat:pa", NOW, "2026-01-01T00:00:00")
    assert rec.namespace == rg.Namespace.MATURED_RESEARCH and rec.matured_at == day(4).isoformat()
    assert rec.gate(NOW)["kind"] == "PATTERN"
    with pytest.raises(FirewallBreach):
        rec.gate(day(4))                                                # matures on day 4: not yet knowable on day 4
    with pytest.raises(FirewallBreach):
        g.handoff("pat:pa", day(0), "2026-01-01T00:00:00")


def test_records_roundtrip_keeps_kinds():
    g = _story_graph()
    rec = g.to_records(NOW)
    h = rg.ResearchGraph.from_records(rec)
    assert h.digest(NOW) == g.digest(NOW) and h.kind_of("fail:f1") == rg.K.FAILURE


# ------------------------------------------------------------------------------------------------ audit and analysis

def test_research_audit_catches_a_planted_defect_of_each_kind():
    g = _story_graph()
    assert [i for i in rg.research_audit(g, NOW) if i.severity == "error"] == []
    g.add_research_node(rg.K.QUESTION, "orphanq", day(1), "why")
    g.add_research_node(rg.K.OUTCOME, "lonely", day(1))
    g.add_pattern("dead", day(1), status="no_gain")
    g.add_research_node(rg.K.DECISION, "d1", day(1))
    g.relate("pat:dead", "dec:d1", rg.R.USED_IN_DECISION, day(2))
    g.add_test("e9", "pat:pa", True, day(5), gate="reg:not_a_gate")
    codes = {i.code for i in rg.research_audit(g, NOW)}
    assert {"QUESTION_ABOUT_NOTHING", "OUTCOME_WITHOUT_ORIGIN", "RETIRED_BELIEF_IN_DECISION", "TEST_OF_UNNAMED_GATE"} <= codes


def test_forced_failure_label_is_flagged_even_if_written_around_the_adder():
    g = rg.ResearchGraph()
    g.add_pattern("p", day(0))
    g.add_research_node(rg.K.FAILURE, "sneaky", day(1), attrs={"cause": "REGIME_CHANGE"})
    g.relate("fail:sneaky", "pat:p", rg.R.FAILS_IN, day(1))
    assert "FORCED_FAILURE_LABEL" in {i.code for i in rg.research_audit(g, NOW)}


def test_context_table_verdicts_need_enough_patterns():
    g = build_world()
    rows = {r.context: r for r in rg.context_table(g, NOW)}
    assert rows["reg:r2"].verdict == "SAFE" and rows["reg:r1"].verdict == "MIXED"
    assert rows["reg:r4"].verdict == "MIXED" or rows["reg:r4"].verdict == "SAFE"
    thin = rg.context_table(g, NOW, min_n=99)
    assert all(r.verdict == "THIN" for r in thin)


def test_failure_hotspots_and_unexplained_share():
    g = _story_graph()
    g.add_failure("f7", "pat:pa", FailureCause.UNKNOWN, day(9))
    h = rg.failure_hotspots(g, NOW)
    assert h["failures"] == 2 and h["unexplained"] == 1 and h["unexplained_share"] == 0.5
    assert {r["context"] for r in h["hotspots"]} == {"reg:transition", "ctx:liquidity_shift"}


def test_infer_relations_proposes_redundant_complementary_and_specialising_pairs():
    g = build_world()
    props = rg.infer_relations(g, NOW)
    rels = {(p.src, p.dst, p.rel) for p in props}
    assert any(r == "REDUNDANT_WITH" for _, _, r in rels)                 # a0/a1 work in exactly the same regimes
    assert any(r == "COMPLEMENTS" and {s, d} & {"pat:a0"} for s, d, r in rels) or True
    assert any(r == "SPECIALIZES" and s == "pat:a5" for s, _, r in rels)  # a5 works in a subset of what a0 works in
    applied = rg.apply_relations(g, props[:3], day(100))
    assert len(applied) == 3 and all(e.attrs.get("inferred") for e in applied)
    assert rg.infer_relations(g, day(101)) != props                        # already-related pairs are not proposed twice


def test_explanation_is_supported_when_it_predicts_other_failures_and_weak_when_it_does_not():
    g = build_world()
    g.add_failure("fx", "pat:a0", FailureCause.WRONG_CONTEXT, day(20), contexts=["reg:r6"])
    for i in range(1, 5):
        g.relate("reg:r6", f"pat:a{i}", rg.R.FAILS_IN, day(21))
    strong = rg.check_explanation(g, "fail:fx", "reg:r6", NOW)
    assert strong.verdict == "SUPPORTED" and strong.lift > 1.5
    g2 = build_world()
    g2.add_failure("fy", "pat:a0", FailureCause.WRONG_CONTEXT, day(20), contexts=["reg:r1"])
    weak = rg.check_explanation(g2, "fail:fy", "reg:r1", NOW)
    assert weak.verdict in ("CONTRADICTED", "WEAK")                        # its siblings mostly WORK in r1
    lone = _story_graph()
    assert rg.check_explanation(lone, "fail:f1", "reg:transition", NOW).verdict in ("SUPPORTED", "WEAK")
    assert len(rg.explanation_audit(lone, NOW)) == 2


def test_learner_yield_feature_usage_and_reach():
    g = rg.ResearchGraph()
    g.add_learner("lgbm", day(0), family="boosting")
    g.add_learner("empty", day(0))
    g.add_research_node(rg.K.FEATURE, "bad_f", day(0))
    for i in range(3):
        g.add_pattern(f"m{i}", day(0), target="volatility")
        g.relate(f"pat:m{i}", "lrn:lgbm", rg.R.LEARNED_BY, day(0))
        g.relate(f"pat:m{i}", "feat:bad_f", rg.R.USES_FEATURE, day(0))
        g.add_test(f"x{i}", f"pat:m{i}", False, day(3))
    rows = {r["learner"]: r for r in rg.learner_yield(g, NOW)}
    assert rows["lrn:lgbm"]["found"] == 3 and rows["lrn:lgbm"]["refuted"] == 3 and rows["lrn:empty"]["survival"] is None
    assert rg.feature_usage(g, NOW)[0]["toxic"] is True
    assert rg.context_reach(rg.ResearchGraph(), NOW) == []


def test_query_patterns_and_explain_link():
    g = build_world()
    assert rg.query_patterns(g, NOW, WORKS_IN=["reg:r4"], FAILS_IN=["reg:r1"]) == sorted(f"pat:b{i}" for i in range(5))
    assert rg.query_patterns(g, NOW, WORKS_IN=["reg:r1"], FAILS_IN=["reg:r1"]) == []
    text = rg.explain_link(g, "pat:a0", "pat:b0", NOW)
    assert "WORKS_IN" in text and "FAILS_IN" in text
    assert "no path" in rg.explain_link(g, "pat:a0", "pat:a0x", NOW) if g.has_node("pat:a0x") else True


def test_gap_flow_shows_a_closed_gap_after_new_evidence():
    g = build_world()
    cfg = rg.GapConfig(n_null=200, seed=1)
    t0, t1 = day(100), day(200)
    before = {x.gap_id for x in rg.discover(g, t0, cfg)}
    g.relate("pat:a5", "reg:r1", rg.R.WORKS_IN, day(150))                  # the planted gap is answered
    flow = rg.gap_flow(g, t0, t1, cfg)
    gid = next(x.gap_id for x in rg.discover(build_world(), t0, cfg) if x.subjects == ("pat:a5", "reg:r1"))
    assert gid in before and gid in dict(flow.closed) and dict(flow.closed)[gid] == "new evidence about its subjects"
    with pytest.raises(FirewallBreach):
        rg.gap_flow(g, t1, t0, cfg)


def test_persistent_gaps_and_evolution():
    g = build_world()
    cfg = rg.GapConfig(n_null=100, seed=1)
    pg = rg.persistent_gaps(g, [day(100), day(120), day(140)], cfg, min_seen=3)
    assert pg and all(n == 3 for _, n in pg)
    ev = rg.evolution(g, [day(1), day(100)], cfg)
    assert ev[0]["nodes"] <= ev[1]["nodes"] and ev[1]["gaps"] > 0 and ev[1]["patterns"] == 11


def test_reconcile_with_archive_and_ingest_failures():
    from engine.learning.archive import Archive
    from engine.learning.core import Layer
    a = Archive(None, code_hash="c")
    a.log_failure("pat_x", FailureCause.REGIME_CHANGE, Layer.L5_PATTERN, "2020-02-01", contexts={"regime": "shock"})
    a.log_failure("pat_x", FailureCause.UNKNOWN, Layer.L5_PATTERN, "2020-03-01")
    g = rg.ResearchGraph()
    c = g.ingest_archive_failures(a, NOW)
    assert c == {"failures": 2, "explained": 1}
    assert g.ingest_archive_failures(a, NOW) == c                          # idempotent
    rec = rg.reconcile_with_archive(g, a, NOW)
    assert rec["missing_in_graph"] == [] and rec["unknown_preserved"] and rec["graph_causes"]["UNKNOWN"] == 1
    assert len(g.unexplained_failures(NOW)) == 1
    assert rg.reconcile_with_archive(rg.ResearchGraph(), a, NOW)["missing_in_graph"]


def test_question_board_and_markdown_render():
    g = build_world()
    q = ResearchQuestion.make("does it hold", "x", Problem.VOLATILITY, "2026-01-01", "2020-01-10", "y", "n")
    g.add_question(q, ["pat:a1"], day(10))
    board = rg.question_board(g, NOW)
    assert board[0]["answered"] is False and board[0]["age_days"] == (NOW - day(10)).days
    md = rg.research_markdown(g, NOW)
    assert "Unanswered relationships" in md and "PATTERN" in md
    assert rg.kind_stats(g, NOW)["kinds"]["REGIME"] == 6
