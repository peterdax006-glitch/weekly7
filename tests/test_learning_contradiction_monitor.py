"""J09 contradiction monitor tests: planted contradiction is born, ages, is resolved by a planted context; an uninvestigated one is
escalated; empty graph; future-dated edge refused; contradicting items are never averaged."""
import datetime as dt

import pytest

from engine.learning.contradiction import (ContradictionError, ContradictionLedger, DisagreementKind, Investigation, LevelResult,
                                           Verdict)
from engine.learning.contradiction_monitor import (ContradictionMonitor, Flag, MonitorConfig, Phase)
from engine.learning.core import Edge, FirewallBreach, Health
from engine.learning.knowledge_graph import EdgeProposal, GraphError, KnowledgeGraph, NodeType

D0 = dt.date(2020, 1, 1)


def day(n):
    return (D0 + dt.timedelta(days=n)).isoformat()


def make_graph(*ids):
    g = KnowledgeGraph()
    for i in ids:
        g.add_node(i, NodeType.PATTERN, day(0), label=f"pattern {i}")
    return g


def investigation(a, b, at, verdict, dim=("regime",), rows_through=None, levels=None):
    lv = levels if levels is not None else (LevelResult("calm", 0.02, 0.021, 0.001, 0.01, 0.1, 30, 30),
                                            LevelResult("stress", 0.03, -0.03, 0.06, 0.01, 6.0, 30, 30))
    return Investigation(a=a, b=b, now=at, verdict=verdict, kind=DisagreementKind.SIGN_CONFLICT, overall_a=0.02, overall_b=-0.01,
                         overall_diff=0.03, overall_z=3.0, dimension=dim, levels=lv, rows_through=rows_through or at, m_tests=2)


def test_planted_contradiction_appears_ages_and_is_resolved_by_context():
    g = make_graph("A", "B")
    led = ContradictionLedger()
    mon = ContradictionMonitor(g, led)
    g.add_edge("A", "B", Edge.CONTRADICTS, day(10), weight=0.4, evidence=("e1",))
    assert mon.scan(day(10)).items == ()                       # known AT now is not yet known
    r1 = mon.scan(day(11))
    assert [t.pair for t in r1.items] == [("A", "B")] and r1.items[0].phase == Phase.UNINVESTIGATED
    r2 = mon.scan(day(31))
    assert r2.items[0].age_days == 21 and r2.items[0].age_days > r1.items[0].age_days
    led.record(investigation("A", "B", day(40), Verdict.RESOLVED_BY_CONTEXT))
    r3 = mon.scan(day(41))
    t = r3.items[0]
    assert t.phase == Phase.RESOLVED_BY_CONTEXT and not t.is_open and t.resolved_dimension == "regime"
    assert t.health() == Health.HEALTHY and r3.open_items() == []
    ev = [e["event"] for e in mon.timeline(("B", "A"), day(41))]
    assert ev == ["born", "investigated:RESOLVED_BY_CONTEXT"]
    assert mon.scan(day(39)).items[0].phase == Phase.UNINVESTIGATED     # the investigation is invisible before it happened


def test_uninvestigated_contradiction_is_escalated_and_questioned():
    g = make_graph("A", "B", "C", "D")
    mon = ContradictionMonitor(g, cfg=MonitorConfig(stale_days=30, escalate_days=60))
    g.add_edge("A", "B", Edge.CONTRADICTS, day(1), weight=0.5)
    g.add_edge("C", "D", Edge.CONTRADICTS, day(80), weight=0.5)
    rep = mon.scan(day(100))
    old = next(t for t in rep.items if t.pair == ("A", "B"))
    new = next(t for t in rep.items if t.pair == ("C", "D"))
    assert Flag.STALE_UNINVESTIGATED in old.flags and Flag.ESCALATED in old.flags
    assert new.flags == () and old.priority > new.priority
    qs = mon.research_questions(rep)
    assert qs[0]["subject"] == "A" and qs[0]["escalated"] and qs[0]["kind"] == "CONTRADICTION"
    sigs = mon.signals(rep)
    assert sigs[0].kind.value == "CONTRADICTION" and sigs[0].counterpart == "B"


def test_growth_in_weight_and_evidence_is_flagged_but_noise_is_not():
    g = make_graph("A", "B", "C", "D")
    mon = ContradictionMonitor(g)
    g.add_edge("A", "B", Edge.CONTRADICTS, day(1), weight=0.2, evidence=("e1",))
    g.add_edge("A", "B", Edge.CONTRADICTS, day(20), weight=0.6, evidence=("e1", "e2", "e3"))
    g.add_edge("C", "D", Edge.CONTRADICTS, day(1), weight=0.02)
    g.add_edge("C", "D", Edge.CONTRADICTS, day(20), weight=0.05)        # tripled, but a gain of 0.03 is noise
    rep = mon.scan(day(30))
    assert Flag.GROWING in next(t for t in rep.items if t.pair == ("A", "B")).flags
    assert Flag.GROWING not in next(t for t in rep.items if t.pair == ("C", "D")).flags
    assert mon.scan(day(10)).items[0].weight_now == pytest.approx(0.2)      # growth not yet known on day 10


def test_reopen_when_newer_evidence_arrives_after_investigation():
    g = make_graph("A", "B")
    led = ContradictionLedger()
    mon = ContradictionMonitor(g, led, MonitorConfig(reopen_days=30, stuck_tries=1))
    g.add_edge("A", "B", Edge.CONTRADICTS, day(1), weight=0.3)
    led.record(investigation("A", "B", day(5), Verdict.UNRESOLVED, rows_through=day(5)))
    g.add_edge("A", "B", Edge.CONTRADICTS, day(90), weight=0.7, evidence=("x",))
    t = mon.scan(day(95)).items[0]
    assert t.phase == Phase.UNRESOLVED and {Flag.REOPEN, Flag.STUCK, Flag.GROWING} <= set(t.flags) and Flag.ESCALATED in t.flags


def test_empty_graph_and_no_ledger_are_quiet_not_errors():
    mon = ContradictionMonitor(KnowledgeGraph())
    rep = mon.scan(day(5))
    assert rep.items == () and rep.counts() == {} and mon.research_questions(rep) == []
    assert mon.dashboard_rows(rep)["rows"] == [] and mon.open_series([day(1), day(2)])[0]["open"] == 0
    assert "0 tracked" in rep.markdown() and rep.to_record()["open"] == 0


def test_future_dated_edge_is_refused_and_ignored():
    g = make_graph("A", "B")
    mon = ContradictionMonitor(g)
    with pytest.raises(FirewallBreach):
        mon.ingest([EdgeProposal("A", "B", Edge.CONTRADICTS, 0.5)], day(50), day(50))
    with pytest.raises(FirewallBreach):
        mon.ingest([EdgeProposal("A", "B", Edge.CONTRADICTS, 0.5)], day(60), day(50))
    with pytest.raises(GraphError):
        mon.ingest([EdgeProposal("A", "B", Edge.SUPPORTS, 0.5)], day(10), day(50))
    assert mon.ingest([EdgeProposal("A", "B", Edge.CONTRADICTS, 0.5)], day(10), day(50))
    g.add_edge("A", "B", Edge.CONTRADICTS, day(70), weight=0.9)             # written later, e.g. by a replay
    rep = mon.scan(day(60))
    assert rep.hidden_future == 1 and rep.items[0].weight_now == pytest.approx(0.5)
    with pytest.raises(FirewallBreach):
        mon.scan(day(60), strict=True)


def test_run_period_enforces_order_and_records():
    g = make_graph("A", "B")
    mon = ContradictionMonitor(g)
    g.add_edge("A", "B", Edge.CONTRADICTS, day(1), weight=0.5)
    mon.run_period(day(10))
    with pytest.raises(FirewallBreach):
        mon.run_period(day(10))
    assert mon.run_period(day(20)).open_items()[0].age_days == 19


def test_never_averages_and_answers_per_context():
    g = make_graph("A", "B")
    led = ContradictionLedger()
    mon = ContradictionMonitor(g, led)
    g.add_edge("A", "B", Edge.CONTRADICTS, day(1), weight=0.5)
    with pytest.raises(ContradictionError):
        mon.pooled("A", "B", day(5))
    assert mon.estimate("A", "B", {"regime": "stress"}, day(5))["state"] == "CONFLICTED"
    led.record(investigation("A", "B", day(6), Verdict.RESOLVED_BY_CONTEXT))
    with pytest.raises(ContradictionError):
        mon.pooled("A", "B", day(10))
    est = mon.estimate("A", "B", {"regime": "stress"}, day(10))
    assert est["A"] == 0.03 and est["B"] == -0.03 and est["contested"] is True
    assert mon.estimate("A", "B", {"regime": "other"}, day(10))["state"] == "INSUFFICIENT_DATA"
    assert mon.estimate("A", "Z", {"regime": "calm"}, day(10))["state"] == "UNKNOWN"


def test_dashboard_rows_and_retraction():
    g = make_graph("A", "B", "C", "D")
    led = ContradictionLedger()
    mon = ContradictionMonitor(g, led)
    g.add_edge("A", "B", Edge.CONTRADICTS, day(1), weight=0.6)
    g.add_edge("C", "D", Edge.CONTRADICTS, day(1), weight=0.4)
    g.retract_edge("C", "D", Edge.CONTRADICTS, day(20), "measurement error")
    led.record(investigation("A", "B", day(10), Verdict.UNRESOLVED))
    rep = mon.scan(day(30))
    dash = mon.dashboard_rows(rep)
    rows = {r["knowledge_id"]: r for r in dash["rows"]}
    assert rows["A"]["health"] == "CONTRADICTED" and rows["A"]["epistemic"] == "CONTRADICTED"
    assert rows["C"]["health"] == "UNKNOWN" and dash["counts"] == {"RETRACTED": 1, "UNRESOLVED": 1}
    assert mon.scan(day(10)).counts() == {"UNINVESTIGATED": 2}                   # before retraction it was live
    series = mon.open_series([day(15), day(30)])
    assert [s["open"] for s in series] == [2, 1] and series[1]["newly_closed"] == 1


def test_config_validation_and_report_record_is_stable():
    for bad in (MonitorConfig(stale_days=0), MonitorConfig(stale_days=50, escalate_days=10), MonitorConfig(growth_ratio=1.0),
                MonitorConfig(evidence_gain=0)):
        with pytest.raises(ValueError):
            bad.validate()
    g = make_graph("A", "B")
    g.add_edge("A", "B", Edge.CONTRADICTS, day(1), weight=0.5)
    mon = ContradictionMonitor(g)
    assert mon.scan(day(9)).to_record()["report_id"] == mon.scan(day(9)).to_record()["report_id"]
    assert mon.scan(day(9)).to_record()["report_id"] != mon.scan(day(19)).to_record()["report_id"]
    assert '"job": "J09"' in mon.dumps(mon.scan(day(9)))
