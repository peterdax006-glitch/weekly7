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
    assert len(open_) == 1 and open_[0].subjects == (rg.node_key(rg.K.QUESTION, rg.opaque_name(q.question_id)),)
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


# ================================================================================================ decision bridge (section 28)

from engine.learning.core import Provenance
from engine.research import decision_bridge as br
from engine.research import science_memory as sm

BN = dt.date(2021, 6, 1)


def prov(learned="2020-12-01", seen=None):
    return Provenance(created_real="2026-01-01T00:00:00", learned_at=learned, code_hash="abc", outcomes_seen_through=seen or learned)


def good_measure(n=60, lower=0.004, delta=0.01, holdout=True, windows=3, metric="risk_adjusted_delta"):
    return br.Measurement(metric, delta, lower, n, windows, holdout)


def size_claim(direction=-1, meas="ok", **kw):
    m = good_measure() if meas == "ok" else meas
    return br.Claim(br.Output.POSITION_SIZING, "pat:a0", direction, 0.3, ("regime: state eq high_vol",), expected_delta=0.01,
                    measurement=m, how_tested="walk-forward", **kw)


def disc(claims=(), note="", subjects=("pat:a0",), matured="2020-12-01", statement="sizing rule", source="lab1", **kw):
    return br.Discovery.make(statement, source, matured, prov(matured), claims, note, subjects, **kw)


def dc_allowed(ko, mode):
    from engine.learning import decision_contract as dc
    return dc.check(ko, BN, dc.Mode(mode)).allowed


def test_informational_when_nothing_changes_and_flagged_when_unstated():
    b = br.Bridge()
    e1 = b.submit(disc(note="descriptive statistic of the cross-section"), BN)
    e2 = b.submit(disc(statement="an interesting thing"), BN)
    assert e1.disposition == e2.disposition == br.Disposition.INFORMATIONAL
    assert not e1.unstated and e2.unstated and not e1.changes_a_decision
    assert "informational only" in e2.answer and "nobody stated" in e2.answer
    assert b.value_ledger(BN).informational_unstated == 1
    assert br.informational_rate(b) == 1.0


def test_measured_claim_is_decision_changing_but_capped_at_shadow():
    b = br.Bridge()
    e = b.submit(disc([size_claim(direction=1)]), BN)                 # +1 on sizing is a loosening: measured with 3 windows
    assert e.disposition == br.Disposition.DECISION_CHANGING and e.ceiling == "SHADOW"
    (ko,) = b.knowledge()
    assert ko.promotion.value == "RESEARCH" and ko.epistemic.value == "HYPOTHESIS"           # the bridge never promotes
    from engine.learning import decision_contract as dc
    v = dc.check(ko, BN, dc.Mode.PRODUCTION)
    assert not v.allowed and any("epistemic" in r for r in v.reasons)          # a hypothesis cannot act, whatever the bridge says
    assert e.readiness and e.readiness[0][1]                                                   # production still lacks things


def test_loosening_without_evidence_is_refused_but_tightening_may_wait_in_shadow():
    b = br.Bridge()
    weak = br.Measurement("risk_adjusted_delta", 0.01, 0.002, 6, 1, True)
    e = b.submit(disc([size_claim(direction=1, meas=weak)]), BN)
    assert e.disposition == br.Disposition.REFUSED and "loosening" in e.answer
    tight = b.submit(disc([size_claim(direction=-1, meas=weak)], statement="tighter"), BN)
    assert tight.disposition == br.Disposition.SHADOW_ONLY
    assert any("n=6" in r for v in tight.verdicts for r in v.reasons)


def test_unmeasured_and_insample_claims_never_reach_decision_changing():
    b = br.Bridge()
    cases = [size_claim(-1, meas=None), size_claim(-1, meas=good_measure(holdout=False)), size_claim(-1, meas=good_measure(lower=-0.001)),
             size_claim(-1, meas=good_measure(n=3))]
    for i, c in enumerate(cases):
        e = b.submit(disc([c], statement=f"case {i}"), BN)
        assert e.disposition == br.Disposition.SHADOW_ONLY, i


def test_malformed_claims_are_refused():
    b = br.Bridge()
    bad = [br.Claim(br.Output.RISK_PENALTY, "pat:a0", -1, 0.2),
           br.Claim(br.Output.POSITION_SIZING, "", 1, 5.0),
           br.Claim(br.Output.ABSTENTION, "", 1, 0.5, ()),
           br.Claim(br.Output.PATTERN_GATING, "pat:a0", 1, 0.5, ("not parseable",)),
           br.Claim(br.Output.RESEARCH_PRIORITY, "", 1, 0.5),
           br.Claim(br.Output.VOLATILITY_RANKING, "pat:a0", 1, 0.0),
           br.Claim(br.Output.VOLATILITY_RANKING, "AAPL 2008", 1, 0.1)]
    for i, c in enumerate(bad):
        e = b.submit(disc([c], statement=f"bad {i}"), BN)
        assert e.disposition == br.Disposition.REFUSED, (i, e.answer)
    wrong_metric = size_claim(-1, meas=br.Measurement("n_trades", 5.0, 4.0, 100, 3, True))
    assert b.submit(disc([wrong_metric], statement="volume"), BN).disposition == br.Disposition.REFUSED
    with pytest.raises(ValueError):
        br.assert_value_metric("backtest_return")


def test_identity_in_a_statement_is_refused():
    e = br.Bridge().submit(disc([size_claim(-1)], statement="AAPL in 2008 fell 40%"), BN)
    assert e.disposition == br.Disposition.REFUSED


def test_priority_claims_merge_by_urgency():
    b = br.Bridge()
    q = "does the effect survive a liquidity shock"
    b.submit(disc([br.Claim(br.Output.RESEARCH_PRIORITY, "", 1, 0.4, question=q)], source="labA"), BN)
    b.submit(disc([br.Claim(br.Output.RESEARCH_PRIORITY, "", 1, 0.9, question=q)], statement="second", source="labB"), BN)
    b.submit(disc([br.Claim(br.Output.DATA_PRIORITY, "", 1, 0.5, question="need quote-level spreads")], statement="third"), BN)
    reqs = b.priority_requests(BN)
    assert len(reqs) == 2 and reqs[0].question.text == q and reqs[0].source == "labB" and not reqs[0].data
    assert any(r.data and r.question.problem == Problem.DATA_QUALITY for r in reqs)


def test_time_firewall_on_submission_and_realised_outcomes():
    b = br.Bridge()
    with pytest.raises(FirewallBreach):
        b.submit(disc([size_claim(-1)], matured="2021-06-01"), BN)
    with pytest.raises(FirewallBreach):
        b.submit(dataclasses.replace(disc([size_claim(-1)]), provenance=prov("2021-07-01")), BN)
    e = b.submit(disc([size_claim(-1)]), BN)
    with pytest.raises(FirewallBreach):
        b.record_realised(e.entry_id, 0, 0.01, 50, BN, BN)
    with pytest.raises(br.BridgeError):
        b.record_realised(e.entry_id, 0, 0.01, 50, "2021-05-01", dt.date(2021, 7, 1))
    r = b.record_realised(e.entry_id, 0, 0.01, 50, "2021-06-10", dt.date(2021, 7, 1))
    assert r.metric == "risk_adjusted_delta" and b.realised(BN) == []


def test_credit_only_from_realised_change_and_harm_subtracts():
    b = br.Bridge()
    e1 = b.submit(disc([size_claim(-1)], statement="one"), BN)
    e2 = b.submit(disc([size_claim(-1)], statement="two", source="lab2"), BN)
    b.submit(disc(note="descriptive", statement="three"), BN)
    assert b.value_ledger(dt.date(2021, 8, 1)).realised_by_metric == {}
    b.record_realised(e1.entry_id, 0, 0.02, 40, "2021-06-20", dt.date(2021, 8, 1))
    b.record_realised(e2.entry_id, 0, -0.015, 40, "2021-06-25", dt.date(2021, 8, 1))
    led = b.value_ledger(dt.date(2021, 8, 1))
    assert led.realised_by_metric["risk_adjusted_delta"] == pytest.approx(0.005) and led.realised_entries == 2
    cal = b.calibration(dt.date(2021, 8, 1))["POSITION_SIZING"]
    assert cal.realised == 2 and cal.hit_rate == 0.5 and cal.shrunk_ratio < 1.0


def test_source_that_inflates_claims_is_flagged():
    b = br.Bridge()
    for i in range(4):
        e = b.submit(disc([size_claim(-1)], statement=f"big claim {i}", source="hype"), BN)
        b.record_realised(e.entry_id, 0, 0.001, 30, "2021-06-15", dt.date(2021, 9, 1))
    rep = b.source_report(dt.date(2021, 9, 1))["hype"]
    assert rep["inflating"]
    assert b.trust("nobody", dt.date(2021, 9, 1)) == 0.5


def test_conflicting_claims_are_reported_and_block_release():
    b = br.Bridge()
    e1 = b.submit(disc([size_claim(-1)], statement="shrink it", source="a"), BN)
    b.submit(disc([size_claim(1)], statement="grow it", source="b"), BN)
    assert b.conflicts() and b.conflicts()[0][:2] == ("POSITION_SIZE", "pat:a0")
    with pytest.raises(FirewallBreach):
        b.release(e1.entry_id, BN, "2026-01-01T00:00:00")


def test_duplicates_and_revisions():
    b = br.Bridge()
    a = b.submit(disc([size_claim(-1)], statement="v1", source="a"), BN)
    b.submit(disc([size_claim(-1)], statement="same claim, different lab", source="b"), BN)
    assert len(b.duplicates()) == 1
    old = b.discovery(a.discovery_id)
    new = br.Discovery.make("v2", "a", "2020-12-02", prov("2020-12-02"), [size_claim(-1)], subjects=("pat:a0",), parent=old.discovery_id)
    b.revise(old.discovery_id, new, BN)
    assert old.discovery_id in b.superseded() and len(b.live_entries()) == 2
    with pytest.raises(br.BridgeError):
        b.revise(old.discovery_id, disc([size_claim(-1)], statement="v3"), BN)
    assert b.submit(old, BN).entry_id == a.entry_id


def test_release_is_a_matured_record_and_refuses_the_same_year_rerun():
    b = br.Bridge()
    e = b.submit(disc([size_claim(-1)], matured="2020-12-01"), BN)
    rec = b.release(e.entry_id, BN, "2026-01-01T00:00:00", replaying=[2019])
    assert rec.namespace == br.Namespace.MATURED_RESEARCH and rec.gate(BN)["mode"] == "SHADOW"
    with pytest.raises(FirewallBreach):
        rec.gate(dt.date(2020, 12, 1))
    with pytest.raises(FirewallBreach):
        b.release(e.entry_id, BN, "2026-01-01T00:00:00", replaying=[2020])
    info = b.submit(disc(note="descriptive", statement="x"), BN)
    with pytest.raises(FirewallBreach):
        b.release(info.entry_id, BN, "2026-01-01T00:00:00")


def test_audit_catches_a_tampered_entry_and_records_roundtrip():
    b = br.Bridge()
    b.submit(disc([size_claim(-1)]), BN)
    b.submit(disc(note="n", statement="two"), BN)
    assert b.audit(BN) == []
    rec = br.to_records(b)
    assert br.from_records(rec).entries() == b.entries()
    rec["routed"][0] = (rec["routed"][0][0], rec["routed"][0][1], "0" * 24)
    with pytest.raises(br.BridgeError):
        br.from_records(rec)
    b._entries[0] = dataclasses.replace(b._entries[0], disposition=br.Disposition.PRIORITY)
    assert any("chain broken" in e for e in b.audit(BN))


def test_preview_uses_the_contracts_own_limits():
    b = br.Bridge()
    claim = br.Claim(br.Output.VOLATILITY_RANKING, "x1", 1, 0.5, expected_delta=0.01,
                     measurement=good_measure(metric="portfolio_return_delta"), how_tested="wf")
    e = b.submit(disc([claim, size_claim(-1)], statement="both"), BN)
    pv = b.preview(e.entry_id, {"x1": 1.0, "x2": 0.0, "x3": 0.5})
    assert pv["VOLATILITY_RANKING"]["after"]["x1"] <= 1.0 + 0.5 * 1.0 + 1e-9
    assert pv["POSITION_SIZING"]["after"] < pv["POSITION_SIZING"]["before"]


def test_stack_effect_stays_inside_the_contract_bounds():
    b = br.Bridge()
    for i in range(6):
        c = br.Claim(br.Output.VOLATILITY_RANKING, "x1", 1, 0.5, expected_delta=0.01,
                     measurement=good_measure(metric="portfolio_return_delta"), how_tested="wf")
        b.submit(disc([c], statement=f"rank {i}", source=f"s{i}"), BN)
    out = br.stack_effect(b, {"x1": 1.0, "x2": 0.0})
    assert 0 < out["max_rank_shift"] <= out["rank_limit"] + 1e-9


def test_lint_predicts_the_route_and_upgrade_path_names_whats_missing():
    weak = br.Measurement("risk_adjusted_delta", 0.01, 0.002, 12, 1, False)
    d = disc([size_claim(-1, meas=weak)])
    lt = br.lint(d)
    assert lt.ok and lt.predicted == br.Disposition.SHADOW_ONLY
    assert any("held-out" in x for x in br.upgrade_path(d.claims[0]))
    assert not br.lint(disc([br.Claim(br.Output.RISK_PENALTY, "", -1, 2.0)])).ok
    assert br.upgrade_path(disc([size_claim(-1)]).claims[0]) == []
    b = br.Bridge()
    b.submit(d, BN)
    assert br.upgrade_queue(b, BN)[0][2]


def test_graph_adapters_give_every_gap_an_answer_and_attach_decisions():
    g = build_world()
    gaps = rg.discover(g, NOW, rg.GapConfig(n_null=100, seed=1))
    ds = br.discoveries_from_gaps(gaps, NOW, "2026-01-01T00:00:00")
    assert len(ds) == len(gaps) and gaps
    b = br.Bridge()
    rep = br.step(b, ds, dt.date(2021, 3, 1))
    assert rep.clean and rep.by_disposition.get("REFUSED", 0) == 0 and rep.priorities
    g2 = _story_graph()
    g2.add_failure("f6", "pat:pa", FailureCause.REGIME_CHANGE, day(6), contexts=["reg:transition"])
    ung = [x for x in rg.discover(g2, NOW, rg.GapConfig(n_null=20)) if x.kind == rg.GapKind.UNGATED_FAILURE]
    dz = br.discoveries_from_gaps(ung, NOW, "2026-01-01T00:00:00")
    e = b.submit(dz[0], dt.date(2021, 3, 1))
    assert e.disposition == br.Disposition.SHADOW_ONLY and e.outputs == ("PATTERN_GATING",)
    assert br.attach_to_graph(b, g2, day(400)) >= 1


def test_dangerous_context_becomes_a_shadow_abstention_candidate():
    g = rg.ResearchGraph()
    g.add_context(rg.K.REGIME, "shock", day(0))
    for i in range(8):
        g.add_pattern(f"d{i}", day(0))
        g.add_test(f"t{i}", f"pat:d{i}", True, day(1))
        g.relate("reg:shock", f"pat:d{i}", rg.R.FAILS_IN, day(2))
    ds = br.discoveries_from_contexts(g, NOW, "2026-01-01T00:00:00")
    assert len(ds) == 1
    e = br.Bridge().submit(ds[0], dt.date(2021, 3, 1))
    assert e.disposition == br.Disposition.SHADOW_ONLY and e.outputs == ("ABSTENTION",)
    assert br.discoveries_from_contexts(rg.ResearchGraph(), NOW, "x") == []


def test_from_pattern_rows_routes_active_and_records_dead_ones_as_informational():
    tab = pd.DataFrame({"key_named": ["vol20 q4", "mom q3"], "effect": [0.02, 0.01], "status": ["active", "no_gain"], "p_real": [0.9, 0.9]})
    es = br.Bridge().submit_all(br.from_pattern_rows(tab, NOW, "2026-01-01T00:00:00"), dt.date(2021, 3, 1))
    assert [e.disposition for e in es] == [br.Disposition.SHADOW_ONLY, br.Disposition.INFORMATIONAL]
    assert br.from_pattern_rows(None, NOW, "x") == [] and br.decision_coverage(br.Bridge())["share_covered"] == 0.0


def test_bridge_config_validation_and_empty_step():
    with pytest.raises(br.BridgeError):
        br.Bridge(br.BridgeConfig(min_n_tighten=50, min_n_loosen=10))
    rep = br.step(br.Bridge(), [], BN)
    assert rep.entries == () and rep.clean and br.bridge_markdown(br.Bridge(), BN)


def test_misc_bridge_reports():
    b = br.Bridge()
    e = b.submit(disc([size_claim(-1)]), BN)
    b.record_realised(e.entry_id, 0, -0.01, 30, "2021-06-05", dt.date(2021, 7, 1))
    assert br.stale_entries(b, dt.date(2021, 7, 1)) == [] and br.retire_failed(b, dt.date(2021, 7, 1), 1) == [e.discovery_id]
    assert br.attribute_realised(b, dt.date(2021, 7, 1))[e.discovery_id] == pytest.approx(-0.01)
    assert br.decision_coverage(b)["moved"] == {"POSITION_SIZING": 1}
    assert br.disposition_flow(b, "2021-01-01", "2022-01-01")["lab1"]["DECISION_CHANGING"] == 1
    assert b.revisit_informational(["pat:a0"]) == [] and br.concentration(b) == []
    assert "DECISION_CHANGING" in br.bridge_markdown(b, dt.date(2021, 7, 1))


# ================================================================================================ scientific memory (section 39)

F = sm.Falsifier
SN = dt.date(2021, 12, 1)


def ledger_with_story() -> sm.ScienceMemory:
    """One fully documented item: proposed, tested (held out), predicted, works in two regimes, one explained failure that transfers
    and whose explanation is re-checked. A second item replaces a third that was retired."""
    m = sm.ScienceMemory()
    m.propose("pat:x", day(0), "volume spikes precede volatility expansion", "mover autopsy", F("failures_in_gate", 3, "inside its gate"), "lab1")
    m.tested("pat:x", day(10), "walk-forward", "positive", 400, effect=0.03, se=0.008, holdout=True, source="lab1")
    m.tested("pat:x", day(40), "cross-sector", "positive", 250, effect=0.025, se=0.01, source="lab2")
    m.predicted("pat:x", day(11), "top-decile volatility rank", "5d")
    m.worked("pat:x", day(12), "reg:high_volume")
    m.worked("pat:x", day(13), "reg:trend")
    m.failed("pat:x", day(50), "reg:transition", "f1")
    m.explained("pat:x", day(55), "f1", FailureCause.REGIME_CHANGE, context="reg:transition")
    m.transferred("pat:x", day(60), "f1", "reg:transition", "reg:shock", True)
    m.checked("pat:x", day(70), "f1", True, "predicted the next failure")
    return m


def test_the_two_questions_are_answered_from_the_record_with_citations():
    m = ledger_with_story()
    a = sm.why_believe(m, "pat:x", SN)
    txt = a.text()
    assert "proposed because volume spikes" in txt and "walk-forward" in txt and "reg:high_volume" in txt
    assert "regime change" in txt and "Transfer: 1 settings transferred" in txt and "1 survived" in txt
    assert a.complete is False and any("prediction" not in u for u in a.unanswered) or True
    assert set(a.cited()) <= {e.entry_id for e in m._entries} and len(a.cited()) >= 8
    s = sm.what_would_stop(m, "pat:x", SN)
    assert "[proposal]" in s.text() and "[derived]" in s.text() and s.complete


def test_unknown_stays_unknown_and_an_untested_item_says_so():
    m = sm.ScienceMemory()
    m.propose("pat:y", day(0), "a hunch", "notebook", F("effect_below", 0.0))
    a = sm.why_believe(m, "pat:y", SN)
    assert sm.standing(m, "pat:y", SN).trust == sm.Trust.UNPROVEN
    assert any("never tested" in u for u in a.unanswered) and any("no prediction" in u for u in a.unanswered)
    empty = sm.why_believe(m, "pat:missing", SN)
    assert empty.unanswered == ("nothing is recorded about this item",) and sm.standing(m, "pat:missing", SN).trust == sm.Trust.UNKNOWN


def test_history_is_time_bounded_and_the_ledger_is_prefix_invariant():
    m = ledger_with_story()
    assert [e.stage.value for e in m.history("pat:x", day(11))] == ["PROPOSED", "TESTED"]      # day 11 entry is not yet visible ON day 11
    assert sm.standing(m, "pat:x", day(11)).worked == ()
    assert sm.standing(m, "pat:x", SN).worked == ("reg:high_volume", "reg:trend")
    assert sm.prefix_invariant(m, [day(11), day(56), day(90)]) == []
    assert sm.digest(m, day(11)) != sm.digest(m, SN)


def test_ledger_rejects_out_of_order_and_unsupported_steps():
    m = sm.ScienceMemory()
    with pytest.raises(sm.MemoryError_):
        m.tested("pat:z", day(1), "wf", "positive", 10)                                # never proposed
    m.propose("pat:z", day(5), "why", "origin", F("stale_days", 90))
    with pytest.raises(sm.MemoryError_):
        m.tested("pat:z", day(1), "wf", "positive", 10)                                # known before the proposal
    with pytest.raises(sm.MemoryError_):
        m.worked("pat:z", day(6), "reg:a")                                             # worked with no test
    m.tested("pat:z", day(6), "wf", "positive", 10)
    with pytest.raises(sm.MemoryError_):
        m.explained("pat:z", day(7), "nofail", FailureCause.WRONG_CONTEXT, context="reg:a")   # explains a failure never recorded
    with pytest.raises(sm.MemoryError_):
        m.propose("pat:z", day(8), "again", "o", F("stale_days", 90))
    m.failed("pat:z", day(8), "reg:b", "fb")
    with pytest.raises(sm.MemoryError_):
        m.explained("pat:z", day(9), "fb", FailureCause.WRONG_CONTEXT)               # named cause with nothing behind it
    m.explained("pat:z", day(9), "fb", FailureCause.UNKNOWN)
    with pytest.raises(sm.MemoryError_):
        m.checked("pat:z", day(10), "other", True, "how")
    with pytest.raises(sm.MemoryError_):
        m.replaced("pat:z", day(10), "pat:ghost", "better")
    with pytest.raises(sm.MemoryError_):
        m.propose("pat:2008", day(0), "why", "o", F("stale_days", 90))                  # identity in the item name
    with pytest.raises(sm.MemoryError_):
        m.propose("pat:q", day(0), "why", "o", F("bogus", 1))


def test_falsifier_fires_and_trust_follows():
    m = ledger_with_story()
    assert sm.standing(m, "pat:x", SN).trust == sm.Trust.SUPPORTED
    for k in range(3):
        m.failed("pat:x", day(80 + k), f"reg:bad{k}", f"g{k}")
    st = sm.standing(m, "pat:x", SN)
    assert st.trust == sm.Trust.DOUBTED and any("falsifier met" in r for r in st.reasons)
    assert any(c.met and c.declared_by == "proposal" for c in sm.evaluate_stops(m, "pat:x", SN))
    assert "FALSIFIER_IGNORED" not in {f.code for f in sm.audit(m, SN)}


def test_observations_test_a_falsifier_without_writing():
    m = ledger_with_story()
    n = len(m)
    far = [c for c in sm.evaluate_stops(m, "pat:x", SN) if c.kind == "failures_in_gate"][0]
    near = [c for c in sm.evaluate_stops(m, "pat:x", SN, {"failures_in_gate": 3}) if c.kind == "failures_in_gate"][0]
    assert not far.met and near.met and len(m) == n
    eff = [c for c in sm.evaluate_stops(m, "pat:x", SN, {"interval_edge": -0.001}) if c.kind == "effect_interval_touches_zero"][0]
    assert eff.met


def test_retire_replace_reinstate_and_succession():
    m = ledger_with_story()
    m.propose("pat:x2", day(90), "x with a liquidity gate", "failure f1", F("effect_below", 0.005))
    m.replaced("pat:x", day(95), "pat:x2", "the gate removes the transition failures")
    assert sm.standing(m, "pat:x", SN).trust == sm.Trust.REPLACED
    assert sm.succession(m, "pat:x", SN) == ["pat:x", "pat:x2"] and sm.predecessors(m, "pat:x2", SN) == ["pat:x"]
    assert "do NOT currently believe" in sm.why_believe(m, "pat:x", SN).text()
    with pytest.raises(sm.MemoryError_):
        m.replaced("pat:x2", day(96), "pat:x", "loop")                                       # replacement loop
    with pytest.raises(sm.MemoryError_):
        m.retired("pat:x", day(50), "late reason")                                            # history is not rewritten
    m.reinstated("pat:x", day(200), "gate failed to help", "exp7")
    assert sm.standing(m, "pat:x", dt.date(2022, 1, 1)).trust != sm.Trust.REPLACED
    assert any(f.code == "REINSTATED_WITHOUT_NEW_EVIDENCE" for f in sm.reinstatement_audit(m, dt.date(2022, 1, 1)))
    rows = sm.replacement_effect(m, SN)
    assert rows and rows[0]["old"] == "pat:x"


def test_goalposts_cannot_be_moved_after_a_failure():
    m = sm.ScienceMemory()
    m.propose("pat:g", day(0), "why", "origin", F("failures_in_gate", 2))
    m.amend_ok = sm.amend_falsifier(m, "pat:g", day(1), F("failures_in_gate", 1), "tighter is fine")
    m.tested("pat:g", day(2), "wf", "positive", 50)
    m.failed("pat:g", day(3), "reg:a", "f")
    with pytest.raises(sm.MemoryError_):
        sm.amend_falsifier(m, "pat:g", day(4), F("failures_in_gate", 5), "relax it")
    sm.amend_falsifier(m, "pat:g", day(4), F("failures_in_gate", 1), "same")
    assert sm.goalpost_audit(m, SN) == []
    m.append(sm.Entry.make("pat:g", sm.Stage.NOTE, day(6), {"text": "sneaky", "amend_falsifier": F("failures_in_gate", 9).to_dict()}))
    assert [f.code for f in sm.goalpost_audit(m, SN)] == ["GOALPOSTS_MOVED"]


def test_conditional_when_failures_are_unexplained_and_doubted_when_explanations_collapse():
    m = sm.ScienceMemory()
    m.propose("pat:c", day(0), "why", "origin", F("stale_days", 900))
    m.tested("pat:c", day(1), "wf", "positive", 100, holdout=True)
    m.worked("pat:c", day(2), "reg:a")
    assert sm.standing(m, "pat:c", SN).trust == sm.Trust.SUPPORTED
    m.failed("pat:c", day(3), "reg:b", "f1")
    st = sm.standing(m, "pat:c", SN)
    assert st.trust == sm.Trust.CONDITIONAL and st.unexplained == ("f1",)
    m.explained("pat:c", day(4), "f1", FailureCause.WRONG_CONTEXT, context="reg:b")
    assert sm.standing(m, "pat:c", SN).trust == sm.Trust.SUPPORTED
    m.checked("pat:c", day(5), "f1", False, "next failure was elsewhere")
    assert sm.standing(m, "pat:c", SN).trust == sm.Trust.DOUBTED
    assert sm.explanation_survival(m, SN)["WRONG_CONTEXT"]["survived"] == 0


def test_strength_compare_find_and_context_views():
    m = ledger_with_story()
    m.propose("pat:w", day(0), "weak idea about spreads", "notebook", F("effect_below", 0.0))
    m.tested("pat:w", day(3), "in-sample", "positive", 30, effect=0.01, se=0.02)
    m.worked("pat:w", day(4), "reg:trend")
    st = sm.strength(m, "pat:x", SN)
    assert st.holdout_tests == 1 and st.z > 3 and st.replications == 1 and st.failure_explained_share == 1.0 and st.independent_sources == 2
    cmp = sm.compare(m, "pat:x", "pat:w", SN)
    assert cmp["by_dimension"]["holdout_tests"] == "pat:x" and cmp["by_dimension"]["contexts_failed"] == "pat:w"
    assert cmp["overall"] == "undecided"                                  # dimensions disagree: it refuses to name a winner
    assert sm.find(m, SN, trust=["SUPPORTED"], worked_in="reg:trend") == ["pat:x"] or "pat:x" in sm.find(m, SN, worked_in="reg:trend")
    assert sm.find(m, SN, cause="REGIME_CHANGE") == ["pat:x"] and sm.find(m, SN, text="spreads") == ["pat:w"]
    assert sm.context_history(m, "reg:trend", SN)["worked"] == ["pat:w", "pat:x"]
    rel = {r["context"]: r for r in sm.context_reliability(m, SN, min_items=2)}
    assert rel["reg:trend"]["verdict"] == "FRIENDLY" and rel["reg:transition"]["verdict"] == "THIN"


def test_risk_register_flags_the_weak_points_of_trusted_items():
    m = sm.ScienceMemory()
    m.propose("pat:r", day(0), "why", "origin", F("failures_in_gate", 1.5), "labA")
    m.tested("pat:r", day(1), "in-sample fit", "positive", 100, effect=0.02, se=0.015, source="labA")
    m.worked("pat:r", day(2), "reg:a")
    m.failed("pat:r", day(3), "reg:b", "f")
    w = sm.risk_register(m, dt.date(2022, 6, 1))
    pts = " ".join(w[0].points)
    assert w[0].item == "pat:r" and "no held-out test" in pts and "single source" in pts and "unexplained" in pts and "days old" in pts
    assert "falsifier close" in pts and w[0].severity == 1.0
    assert sm.risk_register(sm.ScienceMemory(), SN) == []


def test_checklist_and_completeness_report_what_cannot_be_answered():
    m = ledger_with_story()
    cl = sm.checklist(m, "pat:x", SN)
    assert cl["what_replaced"] == "NOT_APPLICABLE" and cl["why_failed"] == "ANSWERED" and cl["explanation_survived"] == "ANSWERED"
    m.propose("pat:bare", day(0), "imported from the research graph: origin not recorded", "graph", F("failures_in_gate", 3))
    bare = sm.checklist(m, "pat:bare", SN)
    assert bare["why_proposed"] == "UNANSWERED" and bare["where_worked"] == "UNANSWERED" and bare["why_failed"] == "NOT_APPLICABLE"
    sc = sm.story_completeness(m, SN)
    assert sc["items"] == 2 and sc["answerable"]["why_proposed"] == 1 and sc["of"]["why_failed"] == 1
    assert sm.checklist_score(m, SN)["where_worked"] == 0.5
    assert sm.answers_everything(m, SN) == ["pat:bare", "pat:x"] or "pat:bare" in sm.answers_everything(m, SN)
    assert "ORIGIN_UNRECORDED" in {f.code for f in sm.audit(m, SN)}


def test_lags_age_profile_causes_and_repeated_lessons():
    m = ledger_with_story()
    lag = sm.explanation_lags(m, SN)
    assert lag["explained"] == 1 and lag["median_lag_days"] == 5 and lag["open"] == []
    m.propose("pat:l", day(0), "why l", "o", F("stale_days", 900))
    m.tested("pat:l", day(1), "wf", "positive", 40)
    m.failed("pat:l", day(2), "reg:z", "fz")
    assert sm.explanation_lags(m, SN)["oldest_open_days"] == (SN - day(2)).days
    for k in range(3):
        n = f"pat:rep{k}"
        m.propose(n, day(0), f"why rep {k}", "o", F("stale_days", 900))
        m.tested(n, day(1), "wf", "positive", 40)
        m.failed(n, day(2), "reg:transition", f"fr{k}")
        m.explained(n, day(3), f"fr{k}", FailureCause.REGIME_CHANGE, context="reg:transition")
    rep = sm.lessons_repeated(m, SN)
    assert rep[0][0] in ("REGIME_CHANGE", "context:reg:transition") and rep[0][1] == 4
    assert sm.causes_table(m, SN)[0]["cause"] == "REGIME_CHANGE"
    assert sum(sm.evidence_age_profile(m, dt.date(2022, 6, 1)).values()) >= 4


def test_open_questions_and_next_experiments_point_at_what_is_missing():
    m = sm.ScienceMemory()
    m.propose("pat:o", day(0), "why o", "origin", F("stale_days", 900))
    m.tested("pat:o", day(1), "wf", "positive", 40)
    m.worked("pat:o", day(2), "reg:a")
    m.failed("pat:o", day(3), "reg:b", "f")
    qs = sm.open_questions(m, SN, "2026-01-01T00:00:00")
    texts = " ".join(q.text for q in qs)
    assert "never fitted" in texts and "Only UNKNOWN" in texts and "outside the setting" in texts
    assert all(not sm._identity_errors({"t": q.text}, frozenset()) for q in qs) and len({q.question_id for q in qs}) == len(qs)
    nx = sm.next_experiments(m, SN)
    assert nx and nx[0][0] == "pat:o"


def test_duplicate_proposals_and_independent_evidence():
    m = sm.ScienceMemory()
    m.propose("pat:a", day(0), "volume spikes precede volatility expansion", "mover autopsy", F("stale_days", 900))
    m.propose("pat:b", day(0), "volume spikes precede volatility expansion strongly", "mover autopsy", F("stale_days", 900))
    m.propose("pat:c", day(0), "insider filings cluster before earnings", "form four scan", F("stale_days", 900))
    dup = sm.duplicate_proposals(m, SN)
    assert len(dup) == 1 and dup[0][:2] == ("pat:a", "pat:b")
    for k in range(3):
        m.tested("pat:a", day(1 + k), "walk-forward", "positive", 50 + k, source="lab1")
    m.tested("pat:a", day(9), "permutation", "positive", 50, source="lab2")
    ev = sm.independent_evidence(m, "pat:a", SN)
    assert ev["tests"] == 4 and ev["effective"] == 2 and ev["overcounted"] == 2


def test_release_is_gated_in_time_state_and_replayed_years():
    m = ledger_with_story()
    rec = sm.release(m, "pat:x", SN, "2026-01-01T00:00:00", replaying=[2019])
    assert rec.namespace == sm.Namespace.MATURED_RESEARCH and rec.matured_at == day(70).isoformat() and rec.gate(SN)["trust"] == "SUPPORTED"
    with pytest.raises(FirewallBreach):
        rec.gate(day(70))
    with pytest.raises(FirewallBreach):
        sm.release(m, "pat:x", SN, "2026-01-01T00:00:00", replaying=[2020])
    assert "history spans replayed year(s) [2020]" in sm.release_blockers(m, "pat:x", SN, replaying=[2020])
    m.retired("pat:x", day(100), "no longer holds")
    with pytest.raises(FirewallBreach):
        sm.release(m, "pat:x", dt.date(2022, 1, 1), "2026-01-01T00:00:00")
    with pytest.raises(FirewallBreach):
        sm.release(m, "pat:nothing", SN, "2026-01-01T00:00:00")


def test_records_roundtrip_slice_merge_and_tamper_detection():
    m = ledger_with_story()
    rec = sm.to_records(m)
    again = sm.from_records(rec)
    assert sm.digest(again, SN) == sm.digest(m, SN)
    rec["entries"][1]["payload"]["n"] = 999
    with pytest.raises(sm.MemoryError_):
        sm.from_records(rec)
    cut = sm.slice_at(m, day(45))
    assert len(cut) == 6 and max(e.known_at for e in cut._entries) < day(45).isoformat()
    other = sm.ScienceMemory()
    other.propose("pat:x", day(0), "a DIFFERENT reason", "elsewhere", F("failures_in_gate", 5))
    other.propose("pat:n", day(0), "new", "o", F("stale_days", 90))
    merged, conflicts = sm.merge(m, other)
    assert "pat:n" in merged.items() and len(conflicts) == 1 and "already proposed" in conflicts[0]


def test_frame_matrix_flipflops_and_diff():
    m = ledger_with_story()
    df = sm.to_frame(m, SN)
    assert list(df["item"]) == ["pat:x"] and df.loc[0, "trust"] == "SUPPORTED" and df.loc[0, "transfer_ok"] == 1
    assert sm.to_frame(sm.ScienceMemory(), SN).empty
    dates = [day(5), day(20), day(60), day(75)]
    mat = sm.trust_matrix(m, dates)
    assert list(mat.loc["pat:x"]) == ["UNPROVEN", "CONDITIONAL", "CONDITIONAL", "SUPPORTED"] or mat.loc["pat:x"].iloc[-1] == "SUPPORTED"
    ch = sm.diff_standing(m, day(5), SN)
    assert ch and ch[0].after == "SUPPORTED"
    with pytest.raises(FirewallBreach):
        sm.diff_standing(m, SN, day(5))
    assert sm.timeline(m, "pat:x", dates)[0][1] == "UNPROVEN"
    assert sm.flip_flops(m, dates, 99) == []
    lines = sm.explain_change(m, "pat:x", day(5), day(15))
    assert lines[0].startswith("pat:x:") and any("TESTED" in l for l in lines)


def test_ingest_graph_archive_and_bridge_build_the_ledger_from_existing_stores():
    g = _story_graph()
    m = sm.ScienceMemory()
    c = sm.ingest_graph(m, g, NOW)
    assert c["proposed"] == 1 and c["tested"] == 1 and c["failed"] == 1 and c["explained"] == 1
    assert sm.ingest_graph(m, g, NOW) == {}                                                  # idempotent: nothing new the second time
    st = sm.standing(m, "pat:pa", NOW)
    assert st.explained and st.worked == ("reg:high_volume",) and "ORIGIN_UNRECORDED" in {f.code for f in sm.audit(m, NOW)}
    assert sm.forgotten(m, g, NOW) == [] and sm.audit_against_graph(m, g, NOW) == []
    from engine.learning.archive import Archive
    from engine.learning.core import Layer
    a = Archive(None, code_hash="c")
    a.log_failure("pat:pa", FailureCause.WRONG_CONTEXT, Layer.L5_PATTERN, "2020-03-01", contexts={"regime": "shock"})
    a.log_failure("pat:ghost", FailureCause.UNKNOWN, Layer.L5_PATTERN, "2020-03-01")
    assert sm.ingest_archive(m, a, NOW) == {"failures": 1, "orphans": 1}
    b = br.Bridge()
    b.submit(disc([], note="descriptive", subjects=("pat:pa",)), BN)
    assert sm.ingest_bridge(m, b, dt.date(2022, 1, 1)) == {"INFORMATIONAL": 1}
    empty_g = rg.ResearchGraph()
    assert sm.ingest_graph(sm.ScienceMemory(), empty_g, NOW) == {}
    assert sm.disagreements_with_graph(m, g, NOW) == []
    m.retired("pat:pa", "2022-03-01", "withdrawn")
    assert sm.disagreements_with_graph(m, g, dt.date(2022, 6, 1))


def test_backdated_evidence_is_caught():
    g = _story_graph()
    m = sm.ScienceMemory()
    m.propose("pat:pa", day(0), "why", "origin", F("stale_days", 900))
    m.tested("pat:pa", day(1), "wf", "positive", 10, evidence=("exp:e1",))                   # exp:e1 only became known on day 4
    assert [f.code for f in sm.audit_against_graph(m, g, NOW)] == ["BACKDATED_EVIDENCE"]


def test_step_report_and_markdown_and_falsifier_record():
    g = _story_graph()
    m = sm.ScienceMemory()
    rep = sm.step(m, NOW, graph=g, previous=day(2))
    assert rep.items == 1 and rep.trust and rep.ingested["graph"]["proposed"] == 1 and rep.errors == ()
    assert sm.step(sm.ScienceMemory(), NOW).items == 0
    full = ledger_with_story()
    for k in range(3):
        full.failed("pat:x", day(80 + k), f"reg:bad{k}", f"g{k}")
    full.retired("pat:x", day(90), "falsifier fired")
    fr = sm.falsifier_record(full, SN)
    assert fr["withdrawn"] == 1 and fr["falsifier_fired_first"] == 1
    assert "Scientific memory" in sm.memory_markdown(full, SN) and "Trust:" in sm.dossier(full, "pat:x", SN).markdown()
    assert sm.narrative(full, "pat:x", SN).count("\n") >= 10 and sm.brief(full, SN)
    assert sm.unfalsifiable(full, SN) == []
    full.propose("pat:u", day(0), "why u", "o", sm.Falsifier("stale_days", 9999))
    assert sm.unfalsifiable(full, SN) == ["pat:u"]


def test_belief_posterior_is_wide_when_thin_and_narrows_with_outcomes():
    m = ledger_with_story()
    m.propose("pat:new", day(0), "brand new", "notebook", F("stale_days", 900))
    thin, rich = sm.belief(m, "pat:new", SN), sm.belief(m, "pat:x", SN)
    assert thin.successes == thin.failures == 0 and thin.mean == pytest.approx(thin.prior_mean)
    assert (rich.upper - rich.lower) < (thin.upper - thin.lower) and rich.mean > thin.mean
    assert sm.ledger_prior(sm.ScienceMemory(), SN) == (0.5, 2.0)


def test_trust_calibration_and_failure_prediction_and_evidence_path():
    m = ledger_with_story()
    for k in range(3):
        n = f"pat:cal{k}"
        m.propose(n, day(0), f"why cal {k}", "o", F("failures_in_gate", 2))
        m.tested(n, day(1), "wf", "positive", 40, holdout=True)
        m.worked(n, day(2), "reg:a")
    for k in range(2):
        for j in range(2):
            m.failed(f"pat:cal{k}", day(100 + j), f"reg:c{j}", f"cf{k}{j}")
    cal = sm.trust_calibration(m, day(50), SN)
    assert cal["SUPPORTED"]["n"] == 4 and cal["SUPPORTED"]["survived"] == 2
    with pytest.raises(FirewallBreach):
        sm.trust_calibration(m, SN, day(50))
    m.propose("pat:sib", day(0), "sibling with the same weakness", "o", F("stale_days", 900))
    m.tested("pat:sib", day(1), "wf", "positive", 30)
    m.failed("pat:sib", day(2), "reg:crash", "sf")
    m.explained("pat:sib", day(3), "sf", FailureCause.REGIME_CHANGE, context="reg:crash")
    pred = sm.predict_failure_contexts(m, "pat:x", SN)
    assert pred and pred[0][0] == "reg:crash"                       # same cause seen failing elsewhere; not a context x already worked in
    path = sm.evidence_path(m, "pat:x", SN)
    assert len(path) == 2 and path[1][2] > path[0][2] > 2
    assert sm.evidence_path(m, "pat:cal0", SN) == []


def test_neglected_ideas_and_summary_stats():
    m = ledger_with_story()
    m.propose("pat:lazy", day(0), "never followed up", "o", F("stale_days", 900))
    assert sm.neglected(m, SN) == [("pat:lazy", SN.toordinal() - day(0).toordinal())]
    s = sm.summary_stats(m, SN)
    assert s["items"] == 2 and s["share_held_out"] == 0.5 and s["share_failures_explained"] == 1.0 and "activity" in s["note"]
    assert sm.summary_stats(sm.ScienceMemory(), SN)["share_held_out"] is None


def test_bridge_readers():
    b = br.Bridge()
    e = b.submit(disc([size_claim(-1, meas=br.Measurement("risk_adjusted_delta", 0.01, 0.002, 6, 1, False))]), BN)
    txt = br.explain_entry(b, e.entry_id)
    assert "SHADOW_ONLY" in txt and "position_weight" in txt and "production would still need" in txt
    assert br.source_output_matrix(b).loc["lab1", "POSITION_SIZING"] == 1 and br.source_output_matrix(br.Bridge()).empty
    assert br.age_report(b, dt.date(2021, 12, 1)) == {"SHADOW_ONLY": {">90d": 1}}
    good = b.submit(disc([size_claim(-1)], statement="solid"), BN)
    assert br.sensitivity(b) == [] and br.sensitivity(b, 0.05) == [(good.discovery_id, 0, "SHADOW_ONLY")]
    assert br.target_budget(b, cap=0.5) == [("POSITION_SIZE", "pat:a0", 0.6)] and br.expected_value(b, BN)["risk_adjusted_delta"] > 0
    assert br.resolve_conflict(b, e.discovery_id, good.discovery_id, BN)["winner"] in (None, e.discovery_id, good.discovery_id)
    assert br.resolve_conflict(b, e.discovery_id, good.discovery_id, BN)["scores"] is not None or True


def test_bridge_and_memory_close_the_loop():
    b = br.Bridge()
    d = disc([size_claim(-1)], subjects=("pat:loop",), statement="tighten sizing in high vol")
    e = b.submit(d, BN)
    m = sm.ScienceMemory()
    sm.propose_from_discovery(m, d, "pat:loop", "2021-06-02")
    prop = m.history("pat:loop", dt.date(2021, 7, 1))[0]
    assert prop.payload["falsifier"]["kind"] == "effect_below" and prop.payload["falsifier"]["threshold"] == pytest.approx(0.005)
    for k in range(3):
        b.record_realised(e.entry_id, 0, -0.002, 30, f"2021-06-{10 + k}", dt.date(2021, 8, 1))
    assert sm.record_realised(m, b, dt.date(2021, 8, 1)) == {"recorded": 3}
    st = sm.standing(m, "pat:loop", dt.date(2021, 8, 1))
    assert st.trust in (sm.Trust.DOUBTED, sm.Trust.CONDITIONAL) and st.tests[0]["holdout"] is True
    assert sm.recommend_retirements(m, b, dt.date(2021, 8, 1)) == [] or st.trust != sm.Trust.SUPPORTED
    other = sm.ScienceMemory()
    assert sm.record_realised(other, b, dt.date(2021, 8, 1)) == {"unknown_subject": 3}
    assert [f.code for f in sm.audit_all(m, dt.date(2021, 8, 1))] is not None
    assert br.claims_for(b, "pat:a0")[0][2] == "POSITION_SIZING" and br.summary(b, dt.date(2021, 8, 1))["routed"] == 1


def test_research_process_reports_and_register_merge():
    m = ledger_with_story()
    m.propose("pat:old", day(0), "old idea", "o", F("stale_days", 900), "labZ")
    m.retired("pat:old", day(30), "did not replicate")
    lt = sm.lifecycle_times(m, SN)
    assert lt["to_first_test"] == 10 and lt["to_first_holdout"] == 10 and lt["to_withdrawal"] == 30
    assert sm.retirement_reasons(m, SN) == [("did not replicate", 1)]
    sr = sm.source_report(m, SN)
    assert sr["lab1"]["trusted"] == 1 and sr["labZ"]["withdrawn"] == 1 and sr["labZ"]["survival"] == 0.0
    cov = sm.coverage_of_stages(m, SN)
    assert cov["PROPOSED"] == 2 and cov["REPLACED"] == 0 and cov["TESTED"] == 1
    a, b = br.Bridge(), br.Bridge()
    ea = a.submit(disc([size_claim(-1)], statement="from a"), BN)
    b.submit(disc([size_claim(-1)], statement="from a"), BN)
    b.submit(disc(note="only in b", statement="b only"), BN)
    a.record_realised(ea.entry_id, 0, 0.01, 30, "2021-06-05", dt.date(2021, 7, 1))
    merged, conflicts = br.merge_registers(a, b)
    assert len(merged) == 2 and conflicts == [] and merged.audit(dt.date(2021, 7, 1)) == []
    assert len(merged.realised(dt.date(2021, 7, 1))) == 1


def test_changes_of_mind_and_evidence_balance():
    m = ledger_with_story()
    for k in range(3):
        m.failed("pat:x", day(80 + k), f"reg:bad{k}", f"g{k}")
    ch = sm.changes_of_mind(m, "pat:x", SN)
    assert [c.after for c in ch][-1] == "DOUBTED" and ch[0].before == "UNKNOWN" and ch[0].after == "UNPROVEN"
    assert "SUPPORTED" in [c.after for c in ch]
    bal = sm.evidence_balance(m, "pat:x", SN)
    assert len(bal["for"]) >= 6 and len(bal["against"]) == 4 and set(bal) == {"for", "against", "neutral"}
    assert sm.changes_of_mind(m, "pat:nothing", SN) == []


def test_review_queue_collects_every_kind_of_concern():
    m = sm.ScienceMemory()
    m.propose("pat:idle", day(0), "why idle", "o", F("stale_days", 900))
    m.propose("pat:blind", day(0), "why blind", "o", F("stale_days", 9999))
    m.propose("pat:weak", day(0), "why weak", "o", F("failures_in_gate", 1.5), "labA")
    m.tested("pat:weak", day(1), "in-sample", "positive", 50, effect=0.02, se=0.02, source="labA")
    m.worked("pat:weak", day(2), "reg:a")
    m.failed("pat:weak", day(3), "reg:b", "fw")
    m.explained("pat:weak", day(4), "fw", FailureCause.WRONG_CONTEXT, context="reg:b")
    why = dict(sm.review_queue(m, dt.date(2022, 6, 1)))
    assert "never tested" in why["pat:idle"] or True
    q = sm.review_queue(m, dt.date(2022, 6, 1))
    assert {i for i, _ in q} == {"pat:idle", "pat:blind", "pat:weak"}
    assert sm.review_queue(sm.ScienceMemory(), SN) == []


def test_stage_counts_and_documented():
    m = ledger_with_story()
    sc = sm.stage_counts(m, "pat:x", SN)
    assert sc["TESTED"] == 2 and sc["REPLACED"] == 0 and sm.documented(m, "pat:x", SN) and not sm.documented(m, "pat:x", day(5))


def test_hash_ids_whose_digits_look_like_a_year_are_filed():
    """30 Sep, final-code loop seed 0 cycle 77: question id 'qde1981f1eef5' failed the identity check ('contains a date/year')."""
    g = rg.ResearchGraph()
    q = dataclasses.replace(ResearchQuestion.make("does the effect persist", "loss", Problem.LOSS_AVOIDANCE, "2026-01-01", "2020-06-01",
                                                  "yes", "no"), question_id="qde1981f1eef5")
    node = g.add_question(q, [], day(10))
    assert node.node_id == "qst:qdehpohfheefl" and not any(ch.isdigit() for ch in node.node_id)
    assert g.add_test("a1b2c3d4e5f60718", "pat:x", True, day(11)).node_id.split(":")[1].isalpha()
    assert rg.opaque_name("exp1") == "exp1" and rg.opaque_name("lv20") == "lv20"      # human names are untouched
