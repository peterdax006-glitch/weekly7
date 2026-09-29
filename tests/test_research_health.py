"""Tests for engine.research.brain_health and engine.research.diversity (C66 sections 37, 38, 43). Synthetic data only."""
import dataclasses
import datetime as dt

import numpy as np
import pytest

from engine.learning.core import FirewallBreach
from engine.research import brain_health as B
from engine.research import diversity as D
from engine.research.brain_health import Area, Level, Outcome

END = "2022-06-01"


def iso(n):
    return dt.date(2020, 1, 1).toordinal() + n


def sim(mode, seed=0, n=480):
    return B.simulate_researcher(n, seed, mode)


def after(rows):
    return B.dt_iso(B.as_date(rows[-1].when).toordinal() + 1)


def mk(i, area="RISK", fam="f", ok=False, day=None, **kw):
    return Outcome(f"X{i:05d}", B.dt_iso(iso(day if day is not None else i)), area, fam, 10.0, ok, 0.3 if ok else 0.0, **kw)


# ------------------------------------------------------------------ brain health: planted defects

@pytest.mark.parametrize("mode,kind", [("monopoly", "FAMILY_MONOPOLY"), ("easy_bias", "EASY_BIAS"),
                                       ("abandon_hard", "HARD_ABANDONMENT"), ("volume", "VOLUME_NOT_KNOWLEDGE")])
def test_planted_defect_is_caught(mode, kind):
    rows = sim(mode)
    rep = B.step(rows, after(rows))
    assert kind in rep.kinds
    assert rep.level in (Level.WATCH, Level.ALARM)


def test_monopoly_alarm_and_directives_cap_the_family():
    rows = [mk(i, "KNOWN_PROMISING", "tune", day=i // 2) for i in range(95)] +            [mk(100 + i, "RISK", f"r{i % 3}", ok=True, day=48 + i, ) for i in range(5)]
    rep = B.step(rows, B.dt_iso(iso(60)))
    f = rep.finding("FAMILY_MONOPOLY")
    assert f.level is Level.ALARM and f.evidence["share"] >= 0.9
    assert rep.directives.family_caps["tune"] <= 0.4
    assert rep.directives.min_explore_share > 0 and not rep.directives.validate()


def test_duplicate_and_memorisation_metrics_fire():
    for mode, name in (("duplicates", "duplicate_rate"), ("memorising", "memorisation_rate")):
        rows = sim(mode)
        m = B.step(rows, after(rows)).metric(name)
        assert m.level in (Level.WATCH, Level.ALARM), (mode, m)


def test_healthy_world_is_quiet():
    alarms = 0
    for s in range(5):
        rows = sim("healthy", s)
        rep = B.step(rows, after(rows))
        alarms += sum(1 for f in rep.findings if f.level is Level.ALARM)
        assert rep.level is not Level.ALARM
    assert alarms == 0


def test_productive_dominant_family_is_watch_not_alarm():
    rows = [mk(i, "KNOWN_PROMISING", "big", ok=True, day=i) for i in range(70)]
    rows[-1] = dataclasses.replace(rows[-1], gain_bits=5.0)
    rows = [dataclasses.replace(o, gain_bits=1.0, claimed_discovery=False) for o in rows]
    f = B.detect_family_monopoly(rows, B.HealthConfig())
    assert f is not None and f.level is Level.WATCH


# ------------------------------------------------------------------ null / empty / degenerate

def test_empty_input_is_unknown_not_ok():
    rep = B.step([], "2021-01-01")
    assert rep.level is Level.UNKNOWN
    assert all(m.level is Level.UNKNOWN and m.value is None for m in rep.metrics)
    assert rep.findings == () and rep.directives.is_neutral()


def test_few_rows_stay_unknown():
    rows = [mk(i) for i in range(5)]
    assert B.step(rows, "2021-01-01").metric("diversity").level is Level.UNKNOWN


def test_null_random_ledger_finds_no_pathology():
    rng = np.random.default_rng(3)
    rows = [Outcome(f"N{i}", B.dt_iso(iso(i // 2)), B.AREAS[int(rng.integers(10))].value, f"fam{int(rng.integers(6))}",
                    float(rng.uniform(5, 10)), bool(rng.random() < 0.2), 0.0, float(rng.uniform(0.2, 0.8))) for i in range(300)]
    rep = B.step(rows, B.dt_iso(iso(160)))
    assert not {"FAMILY_MONOPOLY", "HARD_ABANDONMENT", "EASY_BIAS"} & set(rep.kinds)


# ------------------------------------------------------------------ firewall and validation

def test_future_outcome_is_a_firewall_breach():
    rows = [mk(i) for i in range(20)]
    with pytest.raises(FirewallBreach):
        B.step(rows, B.dt_iso(iso(10)))


def test_outcome_on_now_is_a_breach():
    with pytest.raises(FirewallBreach):
        B.visible([mk(0, day=5)], B.dt_iso(iso(5)))


def test_invalid_records_and_config_raise():
    with pytest.raises(ValueError):
        B.step([mk(0, area="NOPE")], "2021-01-01")
    with pytest.raises(ValueError):
        B.step([mk(0), mk(0, day=1)], "2021-01-01")
    with pytest.raises(ValueError):
        B.step([], "2021-01-01", B.HealthConfig(family_cap=0.95))
    assert Outcome("a", "2020-01-01", "RISK", "f", 1.0, True, 0.0, replicated=True, false_discovery=True).validate()


def test_verified_needs_everything():
    base = dict(claimed_discovery=True, replicated=True, transferred=True)
    assert mk(0, ok=True, **base).verified
    assert not mk(0, ok=True, memorised=True, **base).verified
    assert not mk(0, ok=True, claimed_discovery=True, replicated=True).verified


# ------------------------------------------------------------------ statistics helpers

def test_bh_and_entropy_helpers():
    assert B.benjamini_hochberg([0.001, 0.009, 0.5, 0.9], 0.05) == [0, 1]
    assert B.benjamini_hochberg([], 0.05) == []
    assert B.normalised_entropy([1] * 10, 10) == pytest.approx(1.0)
    assert B.normalised_entropy([5, 0, 0], 3) == 0.0
    assert B.effective_number([1, 1, 1, 1]) == pytest.approx(4.0)
    assert B.hhi([1, 1]) == pytest.approx(0.5)
    assert B.spearman([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)
    assert B.spearman([1, 1, 1], [1, 2, 3]) is None


def test_rate_metric_needs_interval_for_alarm():
    cfg = B.HealthConfig(min_n=5)
    m = B.rate_metric("x", 3, 5, cfg, 0.3, 0.5, "high")
    assert m.level is Level.WATCH                  # point estimate past alarm but interval lower end below warn
    assert B.rate_metric("x", 60, 100, cfg, 0.3, 0.5, "high").level is Level.ALARM
    assert B.rate_metric("x", 1, 2, cfg, 0.3, 0.5, "high").level is Level.UNKNOWN


def test_untagged_repeats_count_as_duplicates():
    rows = [dataclasses.replace(mk(i, day=i), config_hash="same") for i in range(30)]
    m = B.metric_duplicate_rate(rows, B.HealthConfig())
    assert m.value > 0.9 and m.level is Level.ALARM


def test_bh_cross_check_flags_inflated_claims():
    rows = [mk(i, ok=True, claimed_discovery=True, p_value=0.2, replicated=True, transferred=True, false_discovery=False)
            for i in range(20)]
    m = B.metric_fdr(rows, B.HealthConfig())
    assert "BH" in m.note and m.level is Level.WATCH


def test_difficulty_guard_downgrades_when_uninformative():
    rng = np.random.default_rng(1)
    rows = [mk(i, ok=bool(rng.random() < 0.3), difficulty=float(rng.random())) for i in range(200)]
    v = B.difficulty_validity(rows)
    assert v["verdict"] == "UNINFORMATIVE"
    f = B.Finding("EASY_BIAS", Level.ALARM, "x", {})
    assert B.apply_difficulty_guard([f], v)[0].level is Level.WATCH
    assert B.apply_difficulty_guard([f], {"verdict": "INFORMATIVE"})[0].level is Level.ALARM
    assert B.difficulty_validity([])["verdict"] == "INSUFFICIENT"


def test_unjudged_claims_survivor_bias():
    rows = [mk(i, ok=True, claimed_discovery=True, day=i) for i in range(30)]
    f = B.detect_unjudged_claims(rows, B.dt_iso(iso(400)), B.HealthConfig())
    assert f is not None and f.kind == "CLAIMS_UNJUDGED"
    assert B.detect_unjudged_claims([], "2021-01-01", B.HealthConfig()) is None


def test_churn_flipflop_and_future_event():
    ev = []
    for k in range(6):
        ev += [B.KnowledgeEvent(f"k{k}", f"2020-01-0{1 + j}", kind) for j, kind in enumerate(["created", "retired", "revived", "retired"])]
    m = B.metric_churn(ev, "2020-06-01", B.HealthConfig())
    assert m.level in (Level.WATCH, Level.ALARM)
    with pytest.raises(FirewallBreach):
        B.metric_churn(ev, "2020-01-02", B.HealthConfig())
    assert B.metric_churn([], "2020-06-01", B.HealthConfig()).level is Level.UNKNOWN


def test_integrity_gate():
    from engine.research.core import GateVerdict
    rows = sim("memorising")
    v, why = B.integrity_gate(B.step(rows, after(rows)))
    assert v is GateVerdict.QUARANTINED and why
    v, _ = B.integrity_gate(B.step([], "2021-01-01"))
    assert v is GateVerdict.NEEDS_MORE_EVIDENCE


# ------------------------------------------------------------------ ledger, persistence, replay

def test_ledger_monotone_persistence_and_roundtrip(tmp_path):
    rows = sim("monopoly")
    led = B.replay(rows, [B.dt_iso(iso(d)) for d in (150, 190, 230)])
    assert len(led) == 3
    with pytest.raises(FirewallBreach):
        led.append(led.latest())
    p = led.save(tmp_path / "h.json")
    back = B.HealthLedger.load(p)
    assert [r.report_id for r in back.reports] == [r.report_id for r in led.reports]
    assert back.latest().directives.to_dict() == led.latest().directives.to_dict()
    assert len(B.HealthLedger.load(tmp_path / "missing.json")) == 0


def test_replay_never_sees_the_future_and_detects_after_onset():
    rows = sim("monopoly", n=600)
    cps = [B.dt_iso(iso(d)) for d in range(60, 300, 20)]
    led = B.replay(rows, cps)
    onset = B.dt_iso(iso(int(0.4 * 600) // 2))
    lag = B.detection_lag(led, onset, "FAMILY_MONOPOLY")
    assert lag["detected"] and lag["false_alarms_before_onset"] == 0
    assert all(r.n_outcomes <= 2 * (B.as_date(r.now).toordinal() - iso(0)) + 2 for r in led.reports)


def test_persistent_and_resolved():
    led = B.HealthLedger()
    r = B.step(sim("monopoly"), "2022-06-01")
    later = dataclasses.replace(r, now="2022-06-08")
    led.append(r)
    led.append(later)
    assert "FAMILY_MONOPOLY" in led.persistent(2)
    clean = B.step([], "2022-06-15")
    led.append(clean)
    assert "FAMILY_MONOPOLY" in led.resolved() and led.streak("FAMILY_MONOPOLY") == 0
    assert B.stale_ledger(led, "2022-09-01") and not B.stale_ledger(led, "2022-06-16")


def test_report_written_and_deterministic(tmp_path):
    rows = sim("volume")
    a, b = B.step(rows, after(rows)), B.step(rows, after(rows))
    assert a.report_id == b.report_id
    path = B.write_report(a, tmp_path, rows)
    assert path.exists() and path.with_suffix(".txt").exists()
    assert B.BrainHealthReport.from_dict(a.to_dict()).report_id == a.report_id
    assert B.explain_change(None, a) and B.worst_metrics(a) is not None


def test_sprt_and_degradation():
    rows = [mk(i, ok=False, day=i // 2) for i in range(80)]
    assert B.success_collapse_sprt(rows, 0.3, 0.05)["decision"] == "ACCEPT_H1"
    with pytest.raises(ValueError):
        B.success_collapse_sprt(rows, 0.1, 0.3)
    mix = sim("duplicates")
    assert any(d["metric"] == "duplicate_rate" for d in B.degradation_report(mix, B.HealthConfig()))


# ------------------------------------------------------------------ diversity controller

def test_specs_valid_and_infeasible_rejected():
    specs = D.default_specs()
    assert not D.validate_specs(specs)
    bad = dict(specs)
    bad[Area.RISK] = D.AreaSpec(Area.RISK, 0.12, 0.5, 0.6)
    assert D.validate_specs(bad)
    with pytest.raises(ValueError):
        D.DiversityController(specs=bad)
    with pytest.raises(ValueError):
        D.DiversityController(D.DiversityConfig(half_life_days=0))


def test_first_plan_is_prior_and_valid():
    c = D.DiversityController()
    p = D.step(c, "2021-01-04", (), 600.0, 1)
    assert not p.validate() and not D.audit_plan(p, c)
    assert p.shares["RISK"] == pytest.approx(0.12)
    assert D.validate_state(c) == []


def test_learns_strong_area_and_keeps_floors():
    truth = {a: 0.04 for a in D.AREAS}
    truth[Area.VOLATILITY] = 0.5
    r = D.simulate_learning(truth, 40, 600.0, 0)
    fs = r["final_shares"]
    assert fs["VOLATILITY"] > 0.3 and fs["VOLATILITY"] == max(fs.values())
    assert min(fs.values()) >= 0.01 - 1e-9
    assert r["gain_over_uniform"] > 0.05
    assert D.learning_curve(r["controller"])["verdict"] in ("IMPROVING", "FLAT")


def test_null_world_stays_near_prior():
    r = D.simulate_learning({a: 0.15 for a in D.AREAS}, 30, 600.0, 2)
    tv = D.total_variation(r["final_shares"], {a.value: s.prior_share for a, s in D.default_specs().items()})
    assert tv < 0.25
    assert D.balance_report(r["final_shares"])["normalised_entropy"] > 0.8


def test_follows_downward_drift():
    truth = {a: 0.05 for a in D.AREAS}
    truth[Area.VOLATILITY] = 0.5
    r = D.simulate_learning(truth, 70, 600.0, 0, shift=(35, Area.VOLATILITY, 0.02))
    peak = max(t["shares"]["VOLATILITY"] for t in r["trace"])
    assert r["final_shares"]["VOLATILITY"] < peak - 0.1


def test_farming_is_not_paid():
    f = D.farming_probe(0)
    assert f["farmer_rate"] < f["honest_rate"] and f["farmer_share"] < f["farmer_prior"]


def test_usefulness_rules():
    ok = mk(0, ok=True)
    assert D.is_useful(ok)
    assert not D.is_useful(dataclasses.replace(ok, duplicate_of="E1"))
    assert not D.is_useful(dataclasses.replace(ok, memorised=True))
    assert not D.is_useful(dataclasses.replace(ok, train_score=1.0, holdout_score=0.1))
    assert D.usefulness_weight(dataclasses.replace(ok, claimed_discovery=True, replicated=True, transferred=True)) == 2.0


def test_ingest_is_idempotent_firewalled_and_clock_monotone():
    c = D.DiversityController()
    o = mk(0, ok=True, day=0)
    assert not c.observe(o, "2020-02-01").ignored
    assert c.observe(o, "2020-02-01").ignored and c.total_experiments == 1
    with pytest.raises(FirewallBreach):
        c.observe(mk(1, day=100), "2020-02-01")
    c.advance("2020-03-01")
    with pytest.raises(FirewallBreach):
        c.advance("2020-02-01")


def test_stale_repeat_in_failed_area_earns_nothing_and_new_hypothesis_reopens():
    c = D.DiversityController()
    a = dataclasses.replace(mk(0, "FAILED_NEW_HYPOTHESIS", "h", ok=True, day=0), config_hash="c1")
    b = dataclasses.replace(mk(1, "FAILED_NEW_HYPOTHESIS", "h", ok=True, day=1), config_hash="c1")
    assert c.observe(a, "2020-03-01").useful
    ob = c.observe(b, "2020-03-01")
    assert ob.stale_repeat and not ob.useful
    assert not D.propose_hypothesis(c, "FAILED_NEW_HYPOTHESIS", "h", "c1")
    for i in range(35):
        c.observe(dataclasses.replace(mk(10 + i, "FAILED_NEW_HYPOTHESIS", "g", day=2 + i), config_hash=f"z{i}"), "2020-04-01")
    assert c.is_dead(Area.FAILED_NEW_HYPOTHESIS)
    assert D.propose_hypothesis(c, "FAILED_NEW_HYPOTHESIS", "g", "brand-new") and not c.is_dead(Area.FAILED_NEW_HYPOTHESIS)


def test_dead_area_relaxed_but_never_zero():
    c = D.DiversityController()
    for i in range(30):
        c.observe(mk(i, "DIRECTION", "d", day=i), "2020-03-01")
    p = c.allocate("2020-03-01", 600.0, 0)
    assert c.is_dead(Area.DIRECTION)
    assert p.floors["DIRECTION"] == pytest.approx(c.cfg.floor_min) and p.shares["DIRECTION"] > 0


def test_directives_are_honoured():
    c = D.DiversityController()
    d = B.Directives({"RISK": 3.0}, {"famA": 0.3}, ("Q1",), 0.9, ("REGIME",), ("test",))
    for i in range(12):
        c.observe(mk(i, "RISK", "famA", ok=True, day=i), "2020-03-01")
        c.observe(mk(50 + i, "RISK", "famB", ok=True, day=i), "2020-03-01")
    p = c.allocate("2020-03-01", 600.0, 0, d)
    assert p.exploit_share <= 0.11 + 1e-6
    assert p.shares["REGIME"] >= p.floors["REGIME"] >= 2 * 0.02 - 1e-9
    assert D.directive_compliance(p, d) == []
    assert p.family_split["RISK"]["famA"] <= 0.5 + 1e-9
    with pytest.raises(ValueError):
        c.allocate("2020-03-01", 600.0, 0, B.Directives({"RISK": -1.0}))
    with pytest.raises(ValueError):
        c.allocate("2020-03-01", 0.0, 0)


def test_health_to_diversity_end_to_end_redirects_monopoly():
    rows = sim("monopoly", n=400)
    now = after(rows)
    c = D.DiversityController()
    led = B.HealthLedger()
    rep, plan = D.run_cycle(c, led, rows, now, 600.0, 0)
    assert "FAMILY_MONOPOLY" in rep.kinds
    assert plan.shares["KNOWN_PROMISING"] <= 0.2
    assert plan.explore_share >= 0.8 and not D.audit_plan(plan, c)
    assert D.directive_compliance(plan, rep.directives, c.cfg) == []


def test_persistence_roundtrip_and_determinism(tmp_path):
    rows = [dataclasses.replace(o, exp_id=o.exp_id) for o in sim("healthy", n=200)]
    def run():
        c = D.DiversityController()
        p = D.step(c, after(rows), rows, 600.0, 4)
        return c, p
    c1, p1 = run()
    c2, p2 = run()
    assert p1.shares == p2.shares and D.state_hash(c1) == D.state_hash(c2)
    back = D.load_controller(D.save_controller(c1, tmp_path / "c.json"))
    assert D.state_hash(back) == D.state_hash(c1) and D.validate_state(back) == []
    assert back.allocate(after(rows), 600.0, 9).shares == c1.allocate(after(rows), 600.0, 9).shares
    assert D.load_controller(tmp_path / "none.json").total_experiments == 0


def test_empty_step_and_zero_jobs():
    c = D.DiversityController()
    p = D.step(c, "2021-01-04", [], 5.0, 0)
    assert p.n_observed == 0 and D.job_slots(p, 10.0) == []
    slots, carry = D.job_slots_with_carry(p, None, 10.0)
    assert slots == [] and sum(carry.values()) == pytest.approx(0.5)


def test_job_slots_and_largest_remainder():
    assert D.largest_remainder({"a": 1, "b": 1, "c": 1}, 10) == {"a": 4, "b": 3, "c": 3}
    assert sum(D.largest_remainder({"a": 0.2, "b": 0.8}, 7).values()) == 7
    assert D.largest_remainder({"a": 0}, 5) == {"a": 0}
    p = D.step(D.DiversityController(), "2021-01-04", [], 600.0, 0)
    slots = D.job_slots(p, 10.0)
    assert len(slots) == 60 and {s.area for s in slots} == {a.value for a in D.AREAS}


def test_carry_gives_floor_areas_their_share_over_rounds():
    c = D.DiversityController()
    p = D.step(c, "2021-01-04", [], 100.0, 0)
    carry, got = None, {a.value: 0 for a in D.AREAS}
    for _ in range(20):
        slots, carry = D.job_slots_with_carry(p, carry, 10.0)
        for s in slots:
            got[s.area] += 1
    assert min(got.values()) >= 1


def test_question_routing_and_selection():
    Q = type("Q", (), {})
    def q(i, src):
        o = Q()
        o.question_id, o.source, o.problem = f"q{i}", src, None
        return o
    assert D.classify_question("loss_postmortem") is Area.RISK
    assert D.classify_question("???") is Area.UNCERTAIN
    p = D.step(D.DiversityController(), "2021-01-04", [], 600.0, 0)
    qs = [q(i, "loss") for i in range(30)] + [q(100 + i, "regime") for i in range(30)]
    order = D.select_questions(p, qs, 30.0)
    assert len(order) == len(set(order)) <= 20
    assert any(x.startswith("q1") for x in order) and D.select_questions(p, [], 30.0) == []


def test_preview_does_not_mutate_and_compare_splits():
    c = D.DiversityController()
    for i in range(40):
        c.observe(mk(i, "VOLATILITY", "v", ok=i % 2 == 0, day=i), "2020-03-01")
        c.observe(mk(100 + i, "DATA_QUALITY", "q", ok=False, day=i), "2020-03-01")
    before = D.state_hash(c)
    pv = D.preview_distribution(c, "2020-03-01", 600.0, 1, 10)
    assert D.state_hash(c) == before and pv["VOLATILITY"]["hi"] >= pv["VOLATILITY"]["lo"]
    a = {a.value: 0.0 for a in D.AREAS} | {"VOLATILITY": 1.0}
    b = {a.value: 0.0 for a in D.AREAS} | {"DATA_QUALITY": 1.0}
    assert D.compare_splits(c, a, b, 0)["p_a_better"] > 0.95
    with pytest.raises(ValueError):
        D.preview_distribution(c, "2020-03-01", 600.0, 1, 1)


def test_replay_plans_are_walk_forward():
    rows = sim("healthy", n=200)
    cps = [B.dt_iso(iso(d)) for d in (30, 60, 90)]
    plans = D.replay_plans(rows, cps)
    assert [p.now for p in plans] == cps and all(not p.validate() for p in plans)
    late = rows + [mk(9999, "RISK", ok=True, day=95)]
    assert [p.shares for p in D.replay_plans(late, cps)] == [p.shares for p in plans]


def test_stagnant_family_is_starved_inside_area():
    c = D.DiversityController()
    for i in range(12):
        c.observe(mk(i, "RISK", "dead_end", day=i), "2020-03-01")
        c.observe(mk(50 + i, "RISK", "good", ok=True, day=i), "2020-03-01")
    assert "dead_end" in D.stagnant_families(c)["RISK"]
    p = c.allocate("2020-03-01", 600.0, 0)
    assert p.family_split["RISK"]["good"] > p.family_split["RISK"]["dead_end"]


def test_reports_written(tmp_path):
    c = D.DiversityController()
    p = D.step(c, "2021-01-04", [], 600.0, 3)
    path = D.write_plan_report(p, c, tmp_path)
    assert path.exists() and D.area_table(c, "2021-01-04") and D.explain_shift(None, p)
    assert D.hard_quota(B.Directives()) < D.hard_quota(B.Directives(reopen_questions=("q",)))
    assert "Diversity plan" in p.render() and set(D.target_pressure(p)) <= {t.value for t in D.RP.ResearchTarget}
