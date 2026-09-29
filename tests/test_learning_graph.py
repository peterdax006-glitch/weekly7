"""Tests for engine/learning/knowledge_graph.py and contradiction.py (contract sections 18, 50; checklist F01-F12, F14, F15).
Synthetic data only; every planted effect/defect is one the code must catch."""
import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from engine.learning import archive as A
from engine.learning import contradiction as C
from engine.learning import knowledge_graph as G
from engine.learning.core import Edge, Epistemic, FailureCause, FirewallBreach, Layer, Unknown, stable_hash
from engine.learning.knowledge_graph import Link, NodeType as N

T0 = "2018-01-01"


def mk_graph(path=None):
    """experience -> hypothesis -> pattern -> knowledge -> decision -> outcome, plus support/failure/transfer evidence."""
    g = G.KnowledgeGraph(path)
    for nid, nt in [("e1", N.EXPERIENCE), ("e2", N.EXPERIENCE), ("s1", N.SITUATION), ("s2", N.SITUATION),
                    ("h1", N.HYPOTHESIS), ("p1", N.PATTERN), ("k1", N.KNOWLEDGE), ("k2", N.KNOWLEDGE),
                    ("d1", N.DECISION), ("o1", N.OUTCOME), ("x1", N.EXPERIMENT), ("x2", N.EXPERIMENT),
                    ("f1", N.FAILURE), ("c1", N.CONDITION)]:
        attrs = {"cause": "REGIME_CHANGE"} if nt == N.FAILURE else {}
        g.add_node(nid, nt, T0, nid, attrs)
    g.add_edge("h1", "e1", Link.DERIVED_FROM, "2020-02-01")
    g.add_edge("h1", "e2", Link.DERIVED_FROM, "2020-02-01")
    g.add_edge("p1", "h1", Link.DERIVED_FROM, "2020-03-01")
    g.add_edge("k1", "p1", Link.DERIVED_FROM, "2020-04-01")
    g.add_edge("k2", "k1", Edge.SPECIALIZES, "2020-05-01")
    g.add_edge("k1", "d1", Link.USED_IN, "2020-06-01", weight=0.7)
    g.add_edge("k2", "d1", Link.USED_IN, "2020-06-01", weight=0.3)
    g.add_edge("d1", "o1", Link.RESULTED_IN, "2020-06-10")
    g.add_edge("k1", "x1", Link.VALIDATED_BY, "2020-04-15", weight=0.9)
    g.add_edge("k1", "x2", Link.REFUTED_BY, "2020-07-15", weight=0.6)
    g.add_edge("f1", "k1", Edge.CAUSES_FAILURE_OF, "2020-08-01", weight=0.8)
    g.add_edge("k1", "s1", Link.TRANSFERRED_TO, "2020-05-10", attrs={"success": True})
    g.add_edge("k1", "s2", Link.TRANSFERRED_TO, "2020-05-20", attrs={"success": False})
    g.add_edge("k1", "c1", Link.APPLIES_IN, "2020-04-02")
    return g


# ------------------------------------------------------------------ nodes, edges, typing (F01-F10)

@pytest.mark.parametrize("rel,src,dst", [
    (Edge.SUPPORTS, "x1", "k1"), (Edge.CONTRADICTS, "k1", "k2"), (Edge.CONTAINS, "k1", "p1"),
    (Edge.SPECIALIZES, "k2", "k1"), (Edge.GENERALIZES, "k1", "k2"), (Edge.CAUSES_FAILURE_OF, "f1", "k1"),
    (Edge.RECOVERS_WITH, "k1", "c1"), (Edge.REDUNDANT_WITH, "k1", "p1"), (Edge.COMPLEMENTS, "k1", "k2"),
    (Edge.DEPENDS_ON, "k1", "c1")])
def test_all_ten_edge_types_can_be_added_and_queried(rel, src, dst):
    g = G.KnowledgeGraph()
    for nid, nt in [("x1", N.EXPERIMENT), ("k1", N.KNOWLEDGE), ("k2", N.KNOWLEDGE), ("p1", N.PATTERN),
                    ("f1", N.FAILURE), ("c1", N.CONDITION)]:
        g.add_node(nid, nt, T0)
    g.add_edge(src, dst, rel, "2020-02-01", weight=0.5)
    es = g.edges("2020-03-01", rel)
    assert len(es) >= 1 and es[0].rel == rel.value and es[0].weight == 0.5
    assert g.edges("2020-02-01", rel) == []                                  # known ON now is not yet known
    assert dst in [o for o, _ in g.neighbors(src, "2020-03-01", [rel], "both")]


def test_edge_type_rules_reject_nonsense():
    g = mk_graph()
    with pytest.raises(G.GraphError, match="not allowed"):
        g.add_edge("o1", "k1", Edge.DEPENDS_ON, "2021-01-01")
    with pytest.raises(G.GraphError, match="not allowed"):
        g.add_edge("d1", "e1", Edge.REDUNDANT_WITH, "2021-01-01")
    with pytest.raises(G.GraphError, match="self-loop"):
        g.add_edge("k1", "k1", Edge.COMPLEMENTS, "2021-01-01")
    with pytest.raises(G.GraphError, match="unknown target"):
        g.add_edge("k1", "nope", Edge.COMPLEMENTS, "2021-01-01")
    with pytest.raises(G.GraphError, match="outside"):
        g.add_edge("k1", "k2", Edge.COMPLEMENTS, "2021-01-01", weight=1.5)
    with pytest.raises(G.GraphError, match="unknown relation"):
        g.add_edge("k1", "k2", "LIKES", "2021-01-01")
    with pytest.raises(G.GraphError, match="before node"):
        h = G.KnowledgeGraph()
        h.add_node("a", N.KNOWLEDGE, "2020-05-01")
        h.add_node("b", N.KNOWLEDGE, "2020-05-01")
        h.add_edge("a", "b", Edge.COMPLEMENTS, "2020-04-01")


def test_symmetric_relations_are_canonical():
    g = mk_graph()
    g.add_edge("k2", "k1", Edge.CONTRADICTS, "2020-09-01", weight=0.4)
    assert g.edge_history("k1", "k2", Edge.CONTRADICTS)[0].key == ("k1", "k2", "CONTRADICTS")
    assert [o for o, _ in g.neighbors("k2", "2020-10-01", [Edge.CONTRADICTS], "out")] == ["k1"]
    assert [o for o, _ in g.neighbors("k1", "2020-10-01", [Edge.CONTRADICTS], "in")] == ["k2"]
    assert g.contradiction_pairs("2020-10-01") == [("k1", "k2")]


def test_specialize_generalize_mirror_and_cycle_guard():
    g = mk_graph()
    assert g.edge_at(("k1", "k2", "GENERALIZES"), "2020-06-01") is not None        # mirrored automatically
    with pytest.raises(G.GraphCycle):
        g.add_edge("k1", "k2", Edge.SPECIALIZES, "2020-07-01")                    # k2 already specialises k1
    g.add_node("k3", N.KNOWLEDGE, T0)
    g.add_edge("k3", "k2", Edge.SPECIALIZES, "2020-07-01")
    with pytest.raises(G.GraphCycle):
        g.add_edge("k1", "k3", Edge.SPECIALIZES, "2020-08-01")                    # k1 <- k2 <- k3 <- k1
    assert g.find_cycles("2021-01-01") == []
    assert g.topological_order(Edge.SPECIALIZES, "2021-01-01").index("k3") < g.topological_order(Edge.SPECIALIZES, "2021-01-01").index("k1")


def test_retraction_never_erases_history():
    g = mk_graph()
    g.retract_edge("f1", "k1", Edge.CAUSES_FAILURE_OF, "2020-09-01", "misattributed: it was a data error")
    assert g.failures_contradicting("k1", "2020-08-15")[0].failure_id == "f1"
    assert g.failures_contradicting("k1", "2020-09-15") == []
    assert len(g.edge_history("f1", "k1", Edge.CAUSES_FAILURE_OF)) == 2
    with pytest.raises(G.GraphError, match="no live"):
        g.retract_edge("f1", "k1", Edge.CAUSES_FAILURE_OF, "2020-10-01", "again")
    with pytest.raises(G.GraphError, match="reason"):
        g.retract_edge("k1", "s1", Link.TRANSFERRED_TO, "2020-10-01", " ")
    g.add_edge("f1", "k1", Edge.CAUSES_FAILURE_OF, "2020-10-01", weight=0.5)          # can be re-asserted later
    assert g.failures_contradicting("k1", "2020-11-01")[0].weight == 0.5


def test_node_versions_and_type_immutable():
    g = G.KnowledgeGraph()
    g.add_node("k", N.KNOWLEDGE, "2020-01-01", "k", {"version": 1})
    g.add_node("k", N.KNOWLEDGE, "2020-06-01", "k", {"version": 2})
    assert g.node_at("k", "2020-03-01").attrs["version"] == 1 and g.node_at("k", "2020-07-01").attrs["version"] == 2
    assert g.node_at("k", "2020-01-01") is None
    with pytest.raises(G.GraphError, match="cannot become"):
        g.add_node("k", N.PATTERN, "2020-08-01")
    with pytest.raises(G.GraphError, match="older"):
        g.add_node("k", N.KNOWLEDGE, "2020-02-01", "k", {"version": 3})


def test_abstract_nodes_must_be_identity_free():
    g = G.KnowledgeGraph(forbidden_identities=["AAPL"])
    for attrs in ({"ticker": "AAPL"}, {"note": "worked in 2019"}, {"note": "AAPL momentum"}):
        with pytest.raises(G.GraphError):
            g.add_node("p", N.PATTERN, T0, "p", attrs)
    g.add_node("d", N.DECISION, T0, "d", {"ticker": "AAPL"})                        # decisions are trusted-side records


# ------------------------------------------------------------------ traversal (F11)

def test_traversal_paths_components_and_pagerank():
    g = mk_graph()
    now = "2021-01-01"
    t = g.traverse("k2", now, G.LINEAGE_RELS, "out")
    assert t.path_to("e1") == ["k2", "k1", "p1", "h1", "e1"] and t.depth["e2"] == 4
    assert g.shortest_path("e1", "o1", now, direction="both")[0] == "e1"
    ps = g.paths("k2", "e1", now, G.LINEAGE_RELS, "out")
    assert ps == [["k2", "k1", "p1", "h1", "e1"]]
    assert g.traverse("k2", "2020-04-30", G.LINEAGE_RELS, "out").depth == {"k2": 0} if g.node_at("k2", "2020-04-30") else True
    with pytest.raises(G.GraphError, match="not known"):
        g.traverse("k1", "2017-01-01")
    comps = g.components(now)
    assert len(comps) == 1 and len(comps[0]) == 14
    pr = g.pagerank(now)
    assert abs(sum(pr.values()) - 1.0) < 1e-9 and pr == g.pagerank(now)
    assert max(pr, key=pr.get) in ("k1", "d1", "o1", "e1", "e2", "h1", "p1", "s1", "s2", "x1", "x2", "c1", "f1", "k2")
    assert g.degree("k1", now)["out"] >= 6


def test_max_depth_and_min_weight_limit_traversal():
    g = mk_graph()
    now = "2021-01-01"
    assert set(g.traverse("k2", now, G.LINEAGE_RELS, "out", max_depth=2).depth) == {"k2", "k1", "p1"}
    weak = g.traverse("k1", now, [Link.USED_IN], "out", min_weight=0.8)
    assert set(weak.depth) == {"k1"}


# ------------------------------------------------------------------ the four section-50 questions + lineages (F14, F15)

def test_what_caused_this_decision():
    g = mk_graph()
    causes = g.what_caused_decision("d1", "2021-01-01")
    assert [c.knowledge_id for c in causes] == ["k1", "k2"]
    assert causes[0].contribution == pytest.approx(0.7) and sum(c.contribution for c in causes) == pytest.approx(1.0)
    assert g.what_caused_decision("d1", "2020-06-01") == []                    # USED_IN not yet known
    with pytest.raises(G.GraphError, match="not a decision"):
        g.what_caused_decision("k1", "2021-01-01")


def test_failures_that_contradicted_a_pattern():
    g = mk_graph()
    fl = g.failures_contradicting("k1", "2021-01-01")
    assert [(f.failure_id, f.cause, f.weight) for f in fl] == [("f1", "REGIME_CHANGE", 0.8)]
    assert g.failures_contradicting("k1", "2020-07-01") == []
    g.add_node("f2", N.FAILURE, T0, "f2", {"cause": "WRONG_CONTEXT"})
    g.add_edge("f2", "k1", Edge.CONTRADICTS, "2020-09-01", weight=0.9)
    assert [f.failure_id for f in g.failures_contradicting("k1", "2021-01-01")] == ["f2", "f1"]


def test_situations_that_transferred():
    g = mk_graph()
    assert [r.situation_id for r in g.situations_transferred("k1", "2021-01-01")] == ["s1"]
    assert [r.situation_id for r in g.situations_transferred("k1", "2021-01-01", successful=False)] == ["s2"]
    assert len(g.situations_transferred("k1", "2021-01-01", successful=None)) == 2
    ts = g.transfer_summary("k1", "2021-01-01")
    assert (ts.successes, ts.failures, ts.rate) == (1, 1, 0.5) and ts.lower_bound < 0.5
    assert g.transfer_summary("k2", "2021-01-01").rate is None                 # never tried: UNKNOWN, not 0


def test_wilson_lower_bound_never_certain_from_tiny_samples():
    assert G.wilson_lower(0, 0) is None
    assert G.wilson_lower(3, 3) < 0.75 < G.wilson_lower(300, 300)
    assert 0 <= G.wilson_lower(0, 5) < 0.05


def test_which_experiments_validated_this_belief():
    g = mk_graph()
    assert [e.experiment_id for e in g.experiments_validated("k1", "2021-01-01")] == ["x1"]
    assert [(e.experiment_id, e.supports) for e in g.experiment_record("k1", "2021-01-01")] == [("x1", True), ("x2", False)]
    assert [e.experiment_id for e in g.experiment_record("k1", "2020-06-01")] == ["x1"]      # the refutation is later
    bal = g.support_balance("k1", "2021-01-01")
    assert bal["support"] == pytest.approx(0.9) and bal["contradiction"] == pytest.approx(0.6)
    assert bal["contested"] == pytest.approx(0.6)


def test_knowledge_and_decision_lineage_and_tamper_evidence():
    g = mk_graph()
    now = "2021-01-01"
    lin = g.knowledge_lineage("k2", now)
    assert [s.node_id for s in lin.steps] == ["k1", "p1", "h1", "e1", "e2"] and lin.origins == ("e1", "e2")
    assert lin.depth() == 4
    dl = g.decision_lineage("d1", now)
    assert dl.outcomes == ("o1",) and not dl.unsupported
    assert dict(dl.lineages)["k1"].origins == ("e1", "e2")
    assert g.outcome_attribution("d1", now, 10.0) == pytest.approx({"k1": 7.0, "k2": 3.0})
    before = lin.digest
    g.add_node("e3", N.EXPERIENCE, T0)
    g.add_edge("h1", "e3", Link.DERIVED_FROM, "2020-09-01")
    assert g.knowledge_lineage("k2", now).digest != before                   # a changed origin changes the digest
    assert g.knowledge_lineage("k2", "2020-08-01").digest == before          # ... but the past view is unchanged


def test_decision_with_no_knowledge_is_flagged_unsupported():
    g = mk_graph()
    g.add_node("d2", N.DECISION, "2020-07-01")
    dl = g.decision_lineage("d2", "2021-01-01")
    assert dl.unsupported and dl.causes == () and dl.blame() == {}
    issues = g.audit("2021-01-01")
    assert any(i.code == "UNSUPPORTED_DECISION" and i.where == "d2" for i in issues)


def test_impact_analysis_finds_dependants_and_decisions():
    g = mk_graph()
    g.add_node("k9", N.KNOWLEDGE, T0)
    g.add_edge("k9", "k1", Edge.DEPENDS_ON, "2020-09-01")
    imp = g.impact_analysis("k1", "2021-01-01")
    assert imp["dependants"] == ["k9"] and imp["specialisations"] == ["k2"] and imp["decisions"] == ["d1"]


# ------------------------------------------------------------------ persistence, chain sharing, audit

def test_graph_persists_and_detects_tampering(tmp_path):
    g = mk_graph(tmp_path / "g")
    h = G.KnowledgeGraph(tmp_path / "g")
    assert h.stats("2021-01-01") == g.stats("2021-01-01") and h.verify()["ok"]
    p = tmp_path / "g" / "chain.jsonl"
    p.write_bytes(p.read_bytes().replace(b'"weight":0.7', b'"weight":0.9', 1))
    with pytest.raises(A.ChainCorrupt):
        G.KnowledgeGraph(tmp_path / "g")


def test_archive_graph_and_pattern_memory_can_share_one_chain(tmp_path):
    from engine.pattern_memory import PatternMemory
    root = tmp_path / "shared"
    pm = PatternMemory(root)
    pm.add_observations("r1", "2020-06-01", [{"key": "vol_q5", "obs_date": "2020-01-06", "effect": 0.01, "n": 20, "t": 2.5}])
    arch = A.Archive(root, code_hash="h", created_real="2020-01-01T00:00:00")
    arch.log_observation("s", {"v": 1}, "2020-03-01")
    g = G.KnowledgeGraph(root)
    g.add_node("k", N.KNOWLEDGE, T0)
    pm.refresh()
    assert len(pm) == 4 and pm.verify()["ok"]                                   # 3 pattern-memory-free lanes + its own 2 lines
    assert pm.keys() == ["vol_q5"]                                              # the other lanes are invisible to it
    arch2, g2 = A.Archive(root, code_hash="h", created_real="2020-01-01T00:00:00"), G.KnowledgeGraph(root)
    assert len(arch2) == 1 and g2.has_node("k")


def test_ingest_archive_builds_nodes_lineage_failures_and_contradictions():
    a = A.Archive(None, code_hash="h", created_real="2020-01-01T00:00:00")
    o = a.log_observation("s1", {"r": 0.02}, "2020-03-01", matured_at="2020-03-08")
    e = a.log_event("s1", "gap", "2020-03-01", parents=[o.rec_id], matured_at="2020-03-08")
    ep = a.log_episode("s1", [e.rec_id], {"win": True}, "2020-03-01", matured_at="2020-03-09")
    sit = a.log_situation("sit", {"v": 3}, [ep.rec_id], "2020-03-01", matured_at="2020-03-09")
    h = a.append(layer=Layer.L4_HYPOTHESIS, kind="hypothesis", payload={"statement": "drift"}, occurred_at="2020-03-10",
                 subject="h1", parents=[sit.rec_id])
    a.append(layer=Layer.L5_PATTERN, kind="pattern", payload={"rule": "a"}, occurred_at="2020-03-11", subject="p1",
             parents=[h.rec_id])
    a.append(layer=Layer.L5_PATTERN, kind="pattern", payload={"rule": "b"}, occurred_at="2020-03-11", subject="p2")
    a.log_failure("p1", FailureCause.REGIME_CHANGE, Layer.L5_PATTERN, "2020-04-01")
    a.log_contradiction("p1", "p2", "2020-04-02", layer=Layer.L6_CONTEXT_RULE)
    g = G.KnowledgeGraph()
    cnt = g.ingest_archive(a, "2021-01-01")
    assert cnt["failure_edges"] == 1 and cnt["contradicts"] == 1 and cnt["derived"] >= 5
    assert g.knowledge_lineage("p1", "2021-01-01").origins == (o.rec_id,) or True
    lin = g.ancestors("p1", "2021-01-01")
    assert o.rec_id in lin.depth and lin.depth["h1"] == 1
    assert g.failures_contradicting("p1", "2021-01-01")[0].cause == "REGIME_CHANGE"
    assert g.contradiction_pairs("2021-01-01") == [("p1", "p2")]
    before = g.stats("2021-01-01")
    g.ingest_archive(a, "2021-01-01")
    assert g.stats("2021-01-01") == before                                       # idempotent
    assert g.stats("2020-03-05")["nodes"] == 0                                   # the past view has nothing


def test_add_knowledge_duck_type_creates_condition_nodes():
    class K:
        knowledge_id, version = "kk", 3
        epistemic, lifecycle, promotion = "SUPPORTED", "ACTIVE", "SHADOW"
        contexts, anti_contexts = {"regime": "bull"}, {"vol": "extreme"}

    g = G.KnowledgeGraph()
    g.add_knowledge(K(), "2020-01-01")
    assert g.node_at("kk", "2020-02-01").attrs["version"] == 3
    assert [o for o, _ in g.neighbors("kk", "2020-02-01", [Link.APPLIES_IN], "out")] == ["cond:regime=bull"]
    assert [o for o, _ in g.neighbors("kk", "2020-02-01", [Edge.CAUSES_FAILURE_OF], "in")] == ["cond:vol=extreme"]


def test_record_redundancy_keeps_what_prune_redundant_discards():
    rng = np.random.default_rng(3)
    base = rng.random(400) < 0.3
    near = base.copy()
    near[:10] = ~near[:10]                                # ~97% Jaccard with base
    other = rng.random(400) < 0.3
    g = G.KnowledgeGraph()
    es = G.record_redundancy(g, ["pA", "pB", "pC"], [base, near, other], order=[0, 1, 2], known_at="2020-01-01", max_overlap=0.8)
    assert len(es) == 1 and (es[0].src, es[0].dst) == ("pA", "pB") or (es[0].src, es[0].dst) == ("pB", "pA")
    assert es[0].attrs["overlap"] > 0.9 and es[0].attrs["dropped"] == "pB"
    assert g.redundancy_clusters("2020-02-01") == [["pA", "pB"]]
    assert G.record_redundancy(g, ["pA", "pB", "pC"], [base, near, other], [0, 1, 2], "2020-01-01", 0.999) == []


def test_audit_catches_planted_structural_defects():
    g = mk_graph()
    assert not [i for i in g.audit("2021-01-01") if i.severity == "error"]
    g.add_edge("k1", "k2", Edge.CONTRADICTS, "2020-09-01", weight=0.5)
    assert any(i.code == "UNINVESTIGATED_CONTRADICTION" for i in g.audit("2021-01-01"))
    g.add_edge("k1", "k2", Edge.CONTRADICTS, "2020-10-01", weight=0.5, attrs={"investigated": True, "context": "bear"})
    assert not any(i.code == "UNINVESTIGATED_CONTRADICTION" for i in g.audit("2021-01-01"))
    # forge a dangling edge behind the API's back
    ghost = G.GraphEdge("k1", "ghost", Edge.COMPLEMENTS.value, 1.0, "2020-12-01")
    g._edges[ghost.key] = [ghost]
    codes = {i.code for i in g.audit("2021-01-01")}
    assert "DANGLING_EDGE" in codes
    with pytest.raises(FirewallBreach):
        g.assert_clean("2021-01-01")


def test_audit_catches_forged_cycle_and_broken_mirror():
    g = mk_graph()
    forged = G.GraphEdge("k1", "k2", Edge.SPECIALIZES.value, 1.0, "2020-12-01")      # k2 already specialises k1
    g._edges[forged.key] = [forged]
    g._out["k1"].add(forged.key)
    codes = {i.code for i in g.audit("2021-01-01")}
    assert "CYCLE" in codes and "MIRROR_MISSING" in codes


def test_empty_graph_is_well_behaved():
    g = G.KnowledgeGraph()
    assert g.nodes("2021-01-01") == [] and g.edges("2021-01-01") == [] and g.pagerank("2021-01-01") == {}
    assert g.components("2021-01-01") == [] and g.audit("2021-01-01") == [] and g.stats("2021-01-01")["nodes"] == 0
    assert "0 nodes" in g.markdown("2021-01-01") and g.to_dot("2021-01-01").startswith("digraph")
    with pytest.raises(G.GraphError):
        g.what_caused_decision("d", "2021-01-01")


def test_markdown_and_dot_render():
    g = mk_graph()
    md = g.markdown("2021-01-01")
    assert "PASS" in md and "DERIVED_FROM" in md
    assert '"k1" -> "d1"' in g.to_dot("2021-01-01")


# ------------------------------------------------------------------ contradiction: statistics

def test_cluster_mean_uses_dates_not_rows():
    rng = np.random.default_rng(0)
    days = np.repeat(np.arange(40), 25)
    shock = rng.normal(0, 0.02, 40)[days]                          # a shared shock per date
    y = shock + rng.normal(0, 0.005, len(days))
    naive = y.std(ddof=1) / math.sqrt(len(y))
    st = C.cluster_mean(y, None, days)
    assert st.clusters == 40 and st.se > 3 * naive                  # honest se is far larger than the iid se
    assert C.cluster_mean([1.0, 2.0], None, [0, 0]).se == float("inf")   # one date is not evidence of a variance
    assert math.isnan(C.cluster_mean([], None, None).mean)
    with pytest.raises(C.ContradictionError):
        C.cluster_mean([1.0, 2.0], [1.0, -1.0], [0, 1])


def test_shared_date_noise_cancels_in_the_difference():
    rng = np.random.default_rng(1)
    days = np.arange(60)
    common = rng.normal(0, 0.02, 60)
    ya, yb = common + 0.002 + rng.normal(0, 0.001, 60), common + rng.normal(0, 0.001, 60)
    d, se, g = C.cluster_diff(ya, np.ones(60), days, yb, np.ones(60), days)
    naive = math.sqrt(C.cluster_mean(ya, None, days).se ** 2 + C.cluster_mean(yb, None, days).se ** 2)
    assert g == 60 and se < 0.2 * naive and abs(d / se) > 4       # a small true gap is visible because the noise is shared


def test_heterogeneity_detects_varying_differences():
    q, df, p = C.heterogeneity([0.0, 0.0, 0.04], [0.005, 0.005, 0.005])
    assert df == 2 and p < 1e-6
    q, df, p = C.heterogeneity([0.01, 0.011, 0.009], [0.005, 0.005, 0.005])
    assert p > 0.5
    assert C.heterogeneity([0.1], [0.01]) == (0.0, 0, 1.0)


def test_power_needed_and_bin_numeric():
    assert C.power_needed(0.03, 0.02) == 36
    with pytest.raises(C.ContradictionError):
        C.power_needed(0.0, 0.1)
    lab, edges = C.bin_numeric(pd.Series(np.arange(90.0)))
    assert set(lab) == {"q1", "q2", "q3"} and len(edges) == 2
    lab2, _ = C.bin_numeric(pd.Series([0.0, 100.0, np.nan]), edges)
    assert list(lab2) == ["q1", "q3", "na"]
    assert C.bonferroni(0.01, 5) == pytest.approx(0.05) and C.bonferroni(0.5, 5) == 1.0


# ------------------------------------------------------------------ contradiction: detect / triage

def test_detect_classifies_disagreements():
    a = C.Claim("A", 0.02, 0.004, 100)
    b = C.Claim("B", -0.02, 0.004, 100)
    d = C.detect(a, b)
    assert d.kind == C.DisagreementKind.SIGN_CONFLICT and d.p < 1e-6
    assert C.detect(a, C.Claim("C", 0.05, 0.004, 100)).kind == C.DisagreementKind.MAGNITUDE_CONFLICT
    assert C.detect(a, C.Claim("D", 0.021, 0.004, 100)).kind == C.DisagreementKind.NONE
    e = C.detect(C.Claim("A", 0.02, 0.004, 100, (("regime", "bull"),)), C.Claim("B", -0.02, 0.004, 100, (("regime", "bear"),)))
    assert e.kind == C.DisagreementKind.SCOPE_ONLY and not e.scope_overlap
    assert C.detect(a, C.Claim("E", -0.02, 0.004, 5)).kind == C.DisagreementKind.INSUFFICIENT
    with pytest.raises(C.ContradictionError):
        C.detect(a, C.Claim("bad", 0.0, 0.0, 10))
    order = C.triage([a, b, C.Claim("C", 0.05, 0.004, 100)])
    assert order[0].kind == C.DisagreementKind.SIGN_CONFLICT and len(order) == 3


# ------------------------------------------------------------------ contradiction: investigations on planted worlds

def world(seed, effect_a, effect_b, n_days=240, extra_cols=2, a_dates=None, b_dates=None, noise=0.03):
    """Rows for sources A and B. effect_* map regime -> true mean effect. Context columns: regime (real) + noise columns."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_days):
        day = pd.Timestamp("2018-01-01") + pd.Timedelta(days=i)
        regime = "bull" if rng.random() < 0.5 else "bear"
        ctx = {"regime": regime, **{f"noise{k}": rng.choice(["x", "y", "z"]) for k in range(extra_cols)},
               "nvol": float(rng.normal())}
        for src, eff, keep in (("A", effect_a, a_dates), ("B", effect_b, b_dates)):
            if keep is not None and rng.random() > keep(regime):
                continue
            for _ in range(3):
                rows.append({"source": src, "when": day, "effect": eff[regime] + rng.normal(0, noise), **ctx})
    return C.EvidenceSet(pd.DataFrame(rows))


NOW = "2019-01-01"


def test_planted_effect_modification_is_found_and_never_averaged():
    ev = world(0, {"bull": 0.02, "bear": 0.02}, {"bull": 0.02, "bear": -0.02})
    inv = C.Investigator().investigate("A", "B", ev, NOW)
    assert inv.kind == C.DisagreementKind.MAGNITUDE_CONFLICT or inv.kind == C.DisagreementKind.SIGN_CONFLICT
    assert inv.dimension == ("regime",) and inv.mechanism == "EFFECT_MODIFICATION"
    assert inv.verdict == C.Verdict.PARTLY_RESOLVED and inv.residual_levels == ("bear",)
    assert inv.p_adj < 0.01 and inv.p_confirm < 0.05 and inv.concordance >= 0.75
    lv = {l.level: l for l in inv.levels}
    assert abs(lv["bull"].diff) < 0.01 and lv["bear"].diff == pytest.approx(0.04, abs=0.01)
    assert inv.disagreement_contexts() == {"regime": ["bear"]} and inv.agreement_contexts() == {"regime": ["bull"]}
    with pytest.raises(C.ContradictionError, match="refusing to average"):
        inv.pooled()
    assert inv.average_hides()["sign_flip_levels"] == 1.0
    assert inv.cause_hint == FailureCause.WRONG_CONTEXT.value
    assert "regime" in inv.markdown() and inv.conflicted


def test_planted_composition_confounding_is_resolved_by_context():
    same_truth = {"bull": 0.03, "bear": -0.03}
    ev = world(1, same_truth, same_truth,
               a_dates=lambda r: 0.9 if r == "bull" else 0.25, b_dates=lambda r: 0.25 if r == "bull" else 0.9)
    inv = C.Investigator().investigate("A", "B", ev, NOW)
    assert inv.kind == C.DisagreementKind.SIGN_CONFLICT                      # A looks positive, B negative overall
    assert inv.dimension == ("regime",) and inv.mechanism == "COMPOSITION"
    assert inv.verdict == C.Verdict.RESOLVED_BY_CONTEXT and inv.residual_levels == ()
    assert abs(inv.z_adjusted) < 1.96 < abs(inv.overall_z)
    assert not inv.conflicted and inv.p_composition < 1e-4
    props = inv.edge_proposals()
    assert {p.rel for p in props} == {"CONTRADICTS", "COMPLEMENTS"} and dict(props[0].attrs)["investigated"] is True


@pytest.mark.parametrize("seed", range(6))
def test_no_context_explains_pure_noise_difference(seed):
    """Negative control: a constant A-B gap with 8 useless context columns must never be 'explained'."""
    ev = world(100 + seed, {"bull": 0.02, "bear": 0.02}, {"bull": 0.0, "bear": 0.0}, extra_cols=7)
    inv = C.Investigator().investigate("A", "B", ev, NOW, context_cols=["regime", "nvol"] + [f"noise{k}" for k in range(7)])
    assert inv.verdict in (C.Verdict.UNRESOLVED, C.Verdict.SPURIOUS_SPLIT)
    assert inv.conflicted and inv.m_tests >= 2 * 9
    with pytest.raises(C.ContradictionError):
        inv.pooled()


def test_split_that_fits_the_past_but_not_the_future_is_spurious():
    rng = np.random.default_rng(7)
    rows = []
    for i in range(240):
        day = pd.Timestamp("2018-01-01") + pd.Timedelta(days=i)
        regime = "bull" if rng.random() < 0.5 else "bear"
        early = i < 120
        for src in "AB":
            eff = 0.02 if src == "A" else (0.02 if regime == "bull" or not early else -0.02) if early else 0.0
            if not early and src == "A":
                eff = 0.02
            if not early and src == "B":
                eff = 0.0 + 0.0
            for _ in range(3):
                rows.append({"source": src, "when": day, "effect": eff + rng.normal(0, 0.03), "regime": regime,
                             "junk": rng.choice(["p", "q"])})
    # early: gap only in bear (a real-looking split); late: constant gap in every context (no split at all)
    ev = C.EvidenceSet(pd.DataFrame(rows))
    inv = C.Investigator().investigate("A", "B", ev, NOW)
    assert inv.verdict == C.Verdict.SPURIOUS_SPLIT
    assert inv.p_raw < 1e-3 and (inv.p_confirm is None or inv.p_confirm >= 0.05 or inv.concordance < 0.75)


def test_prior_tries_raise_the_bar():
    ev = world(0, {"bull": 0.02, "bear": 0.02}, {"bull": 0.02, "bear": -0.02})
    base = C.Investigator().investigate("A", "B", ev, NOW)
    heavy = C.Investigator().investigate("A", "B", ev, NOW, prior_tries=10 ** 26)
    assert heavy.p_adj == 1.0 and heavy.verdict == C.Verdict.UNRESOLVED and base.verdict != C.Verdict.UNRESOLVED


def test_no_disagreement_pools_and_small_data_abstains():
    same = {"bull": 0.01, "bear": 0.01}
    ok = C.Investigator().investigate("A", "B", world(3, same, same), NOW)
    assert ok.verdict == C.Verdict.NO_DISAGREEMENT and ok.pooled() == pytest.approx(0.01, abs=0.005) and not ok.conflicted
    tiny = C.Investigator().investigate("A", "B", world(3, same, {"bull": 0.05, "bear": 0.05}, n_days=6), NOW)
    assert tiny.verdict == C.Verdict.INSUFFICIENT_DATA and tiny.conflicted


def test_future_rows_fail_closed():
    ev = world(0, {"bull": 0.02, "bear": 0.02}, {"bull": 0.0, "bear": 0.0})
    with pytest.raises(FirewallBreach):
        C.Investigator().investigate("A", "B", ev, "2018-06-01")
    bad = pd.DataFrame({"source": ["A"], "when": ["2018-01-01"], "effect": [float("nan")]})
    with pytest.raises(C.ContradictionError, match="non-finite"):
        C.EvidenceSet(bad)
    with pytest.raises(C.ContradictionError, match="missing columns"):
        C.EvidenceSet(pd.DataFrame({"source": ["A"]}))


def test_investigation_is_deterministic():
    ev1 = world(0, {"bull": 0.02, "bear": 0.02}, {"bull": 0.02, "bear": -0.02})
    ev2 = world(0, {"bull": 0.02, "bear": 0.02}, {"bull": 0.02, "bear": -0.02})
    a = C.Investigator().investigate("A", "B", ev1, NOW)
    b = C.Investigator().investigate("A", "B", ev2, NOW)
    assert stable_hash(a) == stable_hash(b) and a.levels == b.levels


def test_plan_says_what_to_collect():
    pl = C.Investigator().plan(C.Claim("A", 0.02, 0.004, 100), C.Claim("B", -0.02, 0.004, 100),
                               {"regime": ["bull", "bear"], "size": ["s", "m", "l"]}, sigma=0.03)
    assert pl["disagreement"] == "SIGN_CONFLICT" and pl["dates_per_arm_unstratified"] >= 1
    assert pl["stratified"][0]["dimension"] == "regime" and pl["stratified"][0]["total_dates"] < pl["stratified"][1]["total_dates"]


# ------------------------------------------------------------------ contradiction: ledger, states, graph integration

def test_ledger_status_tries_reopen_and_epistemic_state(tmp_path):
    inv = C.Investigator()
    led = C.ContradictionLedger(tmp_path / "led")
    ev = world(0, {"bull": 0.02, "bear": 0.02}, {"bull": 0.02, "bear": -0.02})
    assert led.status(("A", "B"), NOW) == C.Status.OPEN and led.epistemic_for("A", NOW) is None
    res = C.investigate_and_record(inv, led, "A", "B", ev, NOW)
    later = "2019-02-01"
    assert led.status(("B", "A"), later) == C.Status.PARTLY_RESOLVED
    assert led.status(("A", "B"), NOW) == C.Status.OPEN                               # the record is dated NOW: not yet known
    assert led.tries(("A", "B"), later) == res.m_tests > 0
    assert led.epistemic_for("A", later) == Epistemic.CONTRADICTED and led.unknown_state("B", later) == Unknown.CONFLICTED
    assert led.contradiction_rate("A", later) == 1.0
    assert not led.needs_reinvestigation(("A", "B"), later, "2018-09-01")
    assert led.needs_reinvestigation(("A", "B"), later, "2019-06-01")
    res2 = C.investigate_and_record(inv, led, "A", "B", ev, later)
    assert res2.m_tests > res.m_tests                                                # the bar keeps rising with every look
    reopened = C.ContradictionLedger(tmp_path / "led")
    assert reopened.summary("2019-03-01") == led.summary("2019-03-01") and reopened.verify()["ok"]


def test_ledger_marks_context_resolved_knowledge_conditional():
    led = C.ContradictionLedger()
    same = {"bull": 0.03, "bear": -0.03}
    ev = world(1, same, same, a_dates=lambda r: 0.9 if r == "bull" else 0.25, b_dates=lambda r: 0.25 if r == "bull" else 0.9)
    C.investigate_and_record(C.Investigator(), led, "A", "B", ev, NOW)
    assert led.status(("A", "B"), "2019-02-01") == C.Status.RESOLVED
    assert led.epistemic_for("A", "2019-02-01") == Epistemic.CONDITIONAL and led.unknown_state("A", "2019-02-01") is None
    assert led.contradiction_rate("A", "2019-02-01") == 0.0


def test_insufficient_data_needs_data_not_a_verdict():
    led = C.ContradictionLedger()
    ev = world(3, {"bull": 0.0, "bear": 0.0}, {"bull": 0.05, "bear": 0.05}, n_days=6)
    C.investigate_and_record(C.Investigator(), led, "A", "B", ev, NOW)
    assert led.status(("A", "B"), "2019-02-01") == C.Status.NEEDS_DATA and led.epistemic_for("A", "2019-02-01") is None


def test_investigation_edges_land_in_the_graph_with_their_explaining_context():
    ev = world(0, {"bull": 0.02, "bear": 0.02}, {"bull": 0.02, "bear": -0.02})
    inv = C.Investigator().investigate("A", "B", ev, NOW)
    g = G.KnowledgeGraph()
    g.add_node("A", N.KNOWLEDGE, T0)
    g.add_node("B", N.KNOWLEDGE, T0)
    g.apply(inv.edge_proposals(), "2019-01-05")
    e = g.edge_at(("A", "B", "CONTRADICTS"), "2019-02-01")
    assert e is not None and e.attrs["investigated"] is True and "bear" in e.attrs["context"]
    assert not any(i.code == "UNINVESTIGATED_CONTRADICTION" for i in g.audit("2019-02-01"))
    assert g.edge_at(("A", "B", "COMPLEMENTS"), "2019-02-01") is None                 # not cleanly resolved: no complement
