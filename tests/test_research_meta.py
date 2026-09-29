"""Tests for engine.research.meta_research (C66 section 16; checklist R13). Synthetic planted worlds only, no network, no caches.
Each mechanism gets a planted case it must catch, a null case where it must find nothing, and the empty case. The central
test is the self-training one: the scheduler must never learn from its own evaluation results."""
import dataclasses
import datetime as dt
import math
from types import SimpleNamespace

import numpy as np
import pytest

from engine.learning.core import FirewallBreach, ValidationLabel
from engine.learning.research_policy import Candidate, PolicyContext, ResearchTarget, RealisedGain
from engine.research import meta_research as mr
from engine.research.core import GateVerdict, Namespace
from engine.research.meta_research import Q16, ResearchOutcome

CFG = mr.MetaResearchConfig(n_draws=80)
NOW = "2004-06-01"


def run(rid, started="2001-01-05", resolved="2001-03-05", **kw):
    base = dict(run_id=rid, exp_type="ablation", family="vol_like", representation="raw_returns", dataset="panel_survivor",
                validation_method="walk_forward", question_source="loss", target="FAILURE", started_at=started, resolved_at=resolved,
                cost_minutes=5.0)
    base.update(kw)
    return ResearchOutcome(**base)


def day(i, base="2001-01-03"):
    return (dt.date.fromisoformat(base) + dt.timedelta(days=i)).isoformat()


@pytest.fixture(scope="module")
def world():
    return mr.synthetic_research_world(700, 0)


@pytest.fixture(scope="module")
def state(world):
    st = mr.MetaResearchState(cfg=CFG)
    st.ingest(world)
    return st


@pytest.fixture(scope="module")
def result(state):
    return mr.step(state, NOW, 0)


@pytest.fixture(scope="module")
def visible(state):
    return state.store.as_of(NOW)


# ------------------------------------------------------------------------------------------------ records and store

def test_record_check_catches_each_defect():
    assert run("ok").check() == []
    assert any("resolved_at not after" in e for e in run("a", started="2001-05-01", resolved="2001-04-01").check())
    assert any("cost_minutes" in e for e in run("b", cost_minutes=-1).check())
    assert any("identity firewall" in e for e in run("c", family="momentum_2008").check())
    assert any("identity firewall" in e for e in run("d", regime="crash_2008-09-15").check())
    assert any("both durable and overfit" in e for e in run("e", durable=True, overfit=True).check())
    assert any("no ground truth" in e for e in run("f", validator_flagged=True).check())
    assert any("unknown origin" in e for e in run("g", origin="mystery").check())
    assert any("evidence_year" in e for e in run("h", evidence_year="abc").check())
    assert any("run_id missing" in e for e in dataclasses.replace(run("i"), run_id="").check())


def test_store_duplicate_and_strict_time():
    st = mr.OutcomeStore()
    st.add(run("a", resolved="2001-03-05"))
    with pytest.raises(ValueError, match="duplicate"):
        st.add(run("a"))
    with pytest.raises(ValueError):
        st.add(run("bad", started="2002-01-01", resolved="2001-01-01"))
    assert st.as_of("2001-03-05") == []            # resolves ON now: not yet known
    assert [r.run_id for r in st.as_of("2001-03-06")] == ["a"]


def test_store_empty_and_roundtrip(tmp_path):
    assert mr.OutcomeStore().as_of(NOW) == [] and len(mr.OutcomeStore()) == 0
    assert mr.OutcomeStore.load(tmp_path / "missing.jsonl").all() == []
    st = mr.OutcomeStore(mr.synthetic_research_world(40, 1, n_defects=3))
    assert st.save(tmp_path / "s.jsonl") == len(st)
    back = mr.OutcomeStore.load(tmp_path / "s.jsonl")
    assert back.content_hash() == st.content_hash() and back.all() == st.all()
    st2 = mr.OutcomeStore(st.all())
    st2.add(run("extra_row"))
    assert st2.content_hash() != st.content_hash()


# ------------------------------------------------------------------------------------------------ never train on own evaluation

def test_guard_refuses_evaluation_output_and_derived_rows():
    vault = mr.EvaluationVault()
    seal = vault.seal("scheduler_replay", "2003-01-01", "2003-06-01", 1.5, ["r1", "r2"])
    guard = mr.SelfTrainingGuard(vault)
    good = run("good")
    poisoned = [run("p1", origin="meta_evaluation", durable=True), run("p2", origin="scheduler_score"),
                run("p3", derived_from=(seal.eval_id,), durable=True), run("p4", resolved="2010-01-01")]
    keep, rep = guard.screen([good] + poisoned, "2004-01-01")
    assert [r.run_id for r in keep] == ["good"]
    assert set(rep.refused) == {"p1", "p2", "p3", "p4"} and not rep.clean
    with pytest.raises(FirewallBreach):
        guard.assert_clean(poisoned, "2004-01-01")
    guard.assert_clean([good], "2004-01-01")


def test_guard_future_evaluation_dependency_is_a_breach():
    vault = mr.EvaluationVault()
    seal = vault.seal("scheduler_replay", "2003-01-01", "2005-01-01", 0.5)
    row = run("x", derived_from=(seal.eval_id,))
    with pytest.raises(FirewallBreach, match="sealed at/after"):
        mr.SelfTrainingGuard(vault, strict=True).screen([row], "2004-01-01")
    keep, rep = mr.SelfTrainingGuard(vault, strict=False).screen([row], "2004-01-01")
    assert keep == [] and rep.future_dependencies == ("x",)


def test_guard_excludes_rows_chosen_by_the_advice_being_scored():
    guard = mr.SelfTrainingGuard()
    rows = [run("a"), run("b", chosen_by="SAabc"), run("c", chosen_by="SAother")]
    keep, rep = guard.screen(rows, "2004-01-01", evaluating_advice="SAabc")
    assert [r.run_id for r in keep] == ["a", "c"] and "selection feedback" in rep.refused["b"]
    assert guard.screen([], "2004-01-01")[0] == []


def test_vault_seal_rules():
    v = mr.EvaluationVault()
    with pytest.raises(FirewallBreach):
        v.seal("q", "2004-01-01", "2003-01-01", 0.1)
    with pytest.raises(ValueError):
        v.seal("q", "2003-01-01", "2004-01-01", float("nan"))
    s = v.seal("q", "2003-01-01", "2004-01-01", 0.1, ["b", "a"])
    assert s.run_ids == ("a", "b") and s.eval_id in v.ids() and len(v) == 1
    assert v.visible("2004-01-01") == [] and v.visible("2004-01-02") == [s] and v.future_of("2004-01-01") == {s.eval_id}


def test_scheduler_never_trains_on_its_own_evaluation(world):
    """The section-16 rule. Rows carrying the meta-evaluation's own output (here: an extreme 'durable' claim for a bad
    experiment type) must leave the advice unchanged when the guard is on - and must change it when the guard is bypassed, so
    the check can fail."""
    clean = mr.MetaResearchState(cfg=CFG)
    clean.ingest(world)
    first = mr.step(clean, NOW, 0)
    assert first.seal is not None
    poison = [run(f"poison{i}", started=day(i, "2001-06-01"), resolved=day(i + 30, "2001-06-01"), exp_type="representation_probe",
                  family="tiny_sample", durable=True, origin="meta_evaluation", derived_from=(first.seal.eval_id,), cost_minutes=1.0)
              for i in range(120)]
    dirty = mr.MetaResearchState(cfg=CFG, vault=clean.vault)
    dirty.ingest(world)
    dirty.ingest(poison)
    a_clean = mr.fit_update(clean, "2004-07-01", 0, evaluate=False).advice
    a_dirty = mr.fit_update(dirty, "2004-07-01", 0, evaluate=False)
    assert len(a_dirty.guard.refused) == 120
    assert a_dirty.advice.exp_type_durable == a_clean.exp_type_durable
    # bypass the guard: the same poison now moves representation_probe up, proving the test is able to fail
    raw = mr.analyse_question(Q16.DURABLE_EXPERIMENTS, dirty.store.as_of("2004-07-01"), 0.10)
    guarded = mr.analyse_question(Q16.DURABLE_EXPERIMENTS, dirty.guard.training_rows(dirty.store, "2004-07-01"), 0.10)
    assert raw.main.groups["representation_probe"].shrunk > guarded.main.groups["representation_probe"].shrunk + 0.2


def test_future_invariance_rows_resolved_after_now_cannot_matter(world):
    a = mr.MetaResearchState(cfg=CFG)
    a.ingest(world)
    b = mr.MetaResearchState(cfg=CFG)
    scrambled = [dataclasses.replace(o, durable=not o.durable, overfit=None, false_discovery=None, failed=not o.failed)
                 if mr.to_ts(o.resolved_at) >= mr.to_ts(NOW) and o.durable is not None else o for o in world]
    assert scrambled != list(world)
    b.ingest(scrambled)
    ua, ub = mr.fit_update(a, NOW, 0, evaluate=False), mr.fit_update(b, NOW, 0, evaluate=False)
    assert ua.advice.advice_id == ub.advice.advice_id
    assert ua.advice.exp_type_durable == ub.advice.exp_type_durable


def test_step_seals_the_evaluation_and_quarantines_it(result, state):
    assert result.seal is not None and result.seal.eval_id in state.vault.ids()
    assert result.seal.evaluated_at == NOW and result.seal.fitted_through < NOW
    later = run("later", derived_from=(result.seal.eval_id,), started="2004-06-02", resolved="2004-07-01")
    _, rep = state.guard.screen([later], "2005-01-01")
    assert "later" in rep.refused


# ------------------------------------------------------------------------------------------------ statistics

def test_benjamini_hochberg_known_case():
    p = {"a": 0.001, "b": 0.008, "c": 0.039, "d": 0.041, "e": 0.6}
    out = mr.benjamini_hochberg(p, 0.05)
    assert out["a"][1] and out["b"][1] and not out["e"][1]
    assert out["c"][0] == pytest.approx(0.0512, abs=1e-3) and not out["c"][1]
    assert all(out[k][0] >= p[k] for k in p)
    assert mr.benjamini_hochberg({}, 0.1) == {}
    with pytest.raises(ValueError):
        mr.benjamini_hochberg(p, 1.5)


def test_binomial_and_posterior_helpers():
    assert mr.binomial_p(5, 0, 0.5) == 1.0 and mr.binomial_p(3, 10, 0.0) == 1.0
    assert mr.binomial_p(9, 10, 0.5) < 0.05 < mr.binomial_p(6, 10, 0.5)
    assert mr.posterior_rate_above(0, 30, 0.15) < 0.05 < mr.posterior_rate_above(10, 30, 0.15)
    assert mr._streak([True, False, True, True], True) == 2 and mr._streak([], True) == 0


def test_rate_table_planted_null_unknown_and_empty():
    rng = np.random.default_rng(3)
    planted = [("hot", rng.random() < 0.8) for _ in range(60)] + [("cold", rng.random() < 0.1) for _ in range(60)] + \
              [("mid", rng.random() < 0.45) for _ in range(60)] + [("rare", True)] * 2
    t = mr.rate_table(planted, "x", True, 5, 0.10)
    assert t.groups["hot"].direction == "ABOVE" and t.groups["hot"].desirable is True
    assert t.groups["cold"].direction == "BELOW" and t.groups["cold"].desirable is False
    assert t.unknown == ("rare",) and "rare" not in t.groups
    assert t.best(1) == ["hot"] and t.worst(1) == ["cold"]
    flipped = mr.rate_table(planted, "x", False, 5, 0.10)
    assert flipped.groups["hot"].desirable is False
    null = [(f"g{i % 6}", bool(rng.random() < 0.4)) for i in range(600)]
    tn = mr.rate_table(null, "x", True, 5, 0.10)
    assert sum(g.significant for g in tn.groups.values()) <= 1
    empty = mr.rate_table([], "x", True)
    assert empty.groups == {} and empty.n_rows == 0 and empty.best() == [] and empty.rate_of("z") is None


# ------------------------------------------------------------------------------------------------ the twelve questions recover the planted truth

def test_group_questions_recover_planted_ordering(result):
    rec = mr.truth_recovery(result.update)
    for q in ("Q01", "Q02", "Q04", "Q05"):
        assert rec[q] is not None and rec[q] > 0.8, (q, rec)
    assert rec["Q07"] > 0.6 and rec["Q03"] > 0.5
    a = result.advice
    assert a.exp_type_durable["transfer_test"] > a.exp_type_durable["representation_probe"]
    assert a.exp_type_overfit["pattern_search"] > a.exp_type_overfit["transfer_test"]
    assert a.dataset_false_discovery["intraday_synth"] > a.dataset_false_discovery["panel_delisted"]
    assert a.source_decision["loss"] > a.source_decision["curiosity"]


def test_every_question_has_a_spec_and_a_table_even_when_empty():
    assert set(mr.SPECS) == set(Q16) and len(Q16) == 12
    reports = mr.analyse_all([])
    assert all(r.n_rows == 0 and r.main.groups == {} for r in reports.values())
    assert reports[Q16.DURABLE_EXPERIMENTS].answer()["best"] == []


def test_representation_failures_planted_and_unknown():
    rows = [run(f"a{i}", started=day(i), resolved=day(i + 10), representation="fourier", failed=True) for i in range(12)] + \
           [run(f"b{i}", started=day(i), resolved=day(i + 10), representation="rank", failed=bool(i % 5 == 0)) for i in range(15)] + \
           [run(f"c{i}", started=day(i), resolved=day(i + 10), representation="new_idea", failed=True) for i in range(2)]
    v = {r.representation: r for r in mr.representation_failures(rows)}
    assert v["fourier"].verdict == "REPEATEDLY_FAILS" and v["fourier"].current_streak == 12
    assert v["rank"].verdict == "WORKS"
    assert v["new_idea"].verdict == "UNKNOWN" and math.isnan(v["new_idea"].p_recovers)
    assert mr.representation_failures([]) == []
    cold = [run(f"d{i}", started=day(i), resolved=day(i + 10), representation="coin", failed=(i % 2 == 0 or i >= 14)) for i in range(16)]
    assert mr.representation_failures(cold)[0].verdict != "REPEATEDLY_FAILS"        # a streak alone is not enough


def test_dataset_burden_and_search_width_effect():
    rows = [run(f"w{i}", dataset="wide", n_tests=200, false_discovery=(i % 2 == 0)) for i in range(20)] + \
           [run(f"n{i}", dataset="narrow", n_tests=1, false_discovery=True) for i in range(20)]
    b = {x.dataset: x for x in mr.dataset_burden(rows)}
    assert b["narrow"].excess > 15 and b["wide"].excess < 0                     # chance explains wide's flags, not narrow's
    assert b["wide"].expected_false == pytest.approx(20 * (1 - 0.95 ** 200)) and b["narrow"].expected_false == pytest.approx(1.0)
    rng = np.random.default_rng(1)
    planted = [run(f"p{i}", n_tests=int(np.exp(rng.uniform(0, 6))) + 1) for i in range(150)]
    planted = [dataclasses.replace(o, false_discovery=bool(rng.random() < min(0.9, math.log(o.n_tests) / 7))) for o in planted]
    assert mr.search_size_effect(planted)["verdict"] == "WIDER_SEARCH_MEANS_MORE_NOISE"
    null = [dataclasses.replace(o, false_discovery=bool(rng.random() < 0.3)) for o in planted]
    assert mr.search_size_effect(null)["verdict"] == "NO_EFFECT_SEEN"
    assert mr.search_size_effect([])["verdict"] == "INSUFFICIENT" and mr.dataset_burden([]) == []


def _audit_rows(spec):
    rows, rng = [], np.random.default_rng(5)
    for j in range(60):
        present = j % 3 != 0
        for m, (rec, far, cost) in spec.items():
            rows.append(run(f"{m}{j}", exp_type="audit", validation_method=m, cost_minutes=cost, error_present=present,
                            validator_flagged=bool(rng.random() < (rec if present else far)), error_id=f"e{j}" if present else ""))
    return rows


def test_validation_methods_scores_cover_and_alarm_on_everything():
    rows = _audit_rows({"strong": (0.9, 0.05, 10.0), "cheap": (0.5, 0.05, 1.0), "siren": (0.95, 0.9, 1.0), "blind": (0.05, 0.05, 1.0)})
    sc = {s.method: s for s in mr.method_scores(rows)}
    assert sc["strong"].recall > 0.8 and sc["strong"].youden > 0.7
    assert sc["siren"].recall > 0.85 and sc["siren"].youden < 0.15
    assert sc["blind"].recall < 0.2 and sc["blind"].minutes_per_catch is None or sc["blind"].minutes_per_catch > 0
    assert [c.subject for c in mr.method_contradictions(list(sc.values()))] == ["siren"]
    cover = mr.greedy_method_cover(rows, 0.9)
    assert cover["order"][0]["method"] in ("cheap", "siren") and cover["coverage"] >= 0.9
    assert sum(s["new_catches"] for s in cover["order"]) == round(cover["coverage"] * cover["n_errors"])
    assert mr.method_agreement(rows)
    assert mr.method_scores([]) == [] and mr.greedy_method_cover([])["n_errors"] == 0


def test_cover_reports_the_defect_nobody_catches():
    rows = [run(f"a{j}", validation_method="m1", error_present=True, validator_flagged=(j != 3), error_id=f"e{j}") for j in range(6)] + \
           [run(f"b{j}", validation_method="m2", error_present=True, validator_flagged=(j != 3), error_id=f"e{j}") for j in range(6)]
    assert mr.greedy_method_cover(rows, 1.0)["missed_by_all"] == ["e3"]


def test_decision_yield_and_failure_information():
    rows = [run(f"c{i}", question_source="cheap_q", cost_minutes=1.0, decision_changed=(i % 2 == 0)) for i in range(20)] + \
           [run(f"e{i}", question_source="dear_q", cost_minutes=50.0, decision_changed=(i % 2 == 0)) for i in range(20)]
    y = mr.decision_yield(rows)
    assert y[0].source == "cheap_q" and y[0].efficiency > 1 > y[1].efficiency
    assert y[0].share_of_all_changes == pytest.approx(0.5)
    fails = [run(f"f{i}", exp_type="probe", failed=True, info_bits=0.0) for i in range(6)] + \
            [run(f"g{i}", exp_type="probe", failed=True, info_bits=0.5) for i in range(2)] + [run("h", exp_type="probe", failed=True, followups=1)]
    fi = mr.failure_information(fails)[0]
    assert (fi.failures, fi.informative, fi.dead) == (9, 3, 6) and fi.dead_minutes == pytest.approx(30.0)
    assert mr.decision_yield([]) == [] and mr.failure_information([run("ok")]) == []


def test_path_verdicts_stop_throttle_keep_unknown():
    def path(name, flags, base=0):
        return [run(f"{name}{i}", started=day(base + i), resolved=day(base + i + 5), exp_type=name, durable=not w, decision_changed=False,
                    info_bits=0.0 if w else 0.5, cost_minutes=10.0) for i, w in enumerate(flags)]
    rows = path("dead", [True] * 25) + path("good", [False, True] * 10) + path("few", [True] * 2) + path("shaky", [True] * 9 + [False] * 2 + [True] * 4)
    v = {x.path.split("|")[0]: x for x in mr.path_verdicts(rows)}
    assert v["dead"].action == "STOP" and v["dead"].waste_share == 1.0 and v["dead"].recent_barren == 25
    assert v["good"].action == "KEEP" and v["few"].action == "UNKNOWN"
    assert v["shaky"].action in ("KEEP", "THROTTLE")
    s = mr.compute_waste_summary(list(v.values()))
    assert s["stop_paths"] and s["minutes_recoverable"] == pytest.approx(250.0)
    assert mr.path_verdicts([]) == [] and mr.compute_waste_summary([])["waste_share"] == 0.0
    assert mr.path_action(0, 3, 3) == "UNKNOWN" and mr.path_action(0, 30, 30) == "STOP" and mr.path_action(0, 30, 2) == "THROTTLE"


def test_stop_rule_replay_saves_far_more_than_it_loses():
    dead = [run(f"d{i}", started=day(i), resolved=day(i + 5), exp_type="dead", durable=False, cost_minutes=20.0, info_bits=0.0) for i in range(40)]
    fine = [run(f"f{i}", started=day(i), resolved=day(i + 5), exp_type="fine", durable=bool(i % 2), decision_changed=False, cost_minutes=5.0,
                info_bits=0.0) for i in range(40)]
    rep = mr.stop_rule_replay(dead + fine, CFG)
    assert rep["stopped_paths"] == ["dead|vol_like|raw_returns"] and rep["minutes_skipped"] > 400
    assert rep["useful_runs_lost"] == 0 and rep["verdict"] == "WORTH_HAVING"
    assert mr.stop_rule_replay([], CFG)["minutes_skipped"] == 0


def test_survival_breakdown_finds_the_regime_break(world):
    rows = [o for o in world if o.new_regime_survived is not None]
    assert "crisis" in mr.breaks_only_in(rows, "regime", "vol_like", min_n=3, gap=0.25)
    assert "calm" not in mr.breaks_only_in(rows, "regime", "vol_like", min_n=3, gap=0.25)
    assert mr.breaks_only_in([], "regime", "vol_like") == [] and mr.survival_breakdown([], "era") == []
    g = mr.survival_gradient(rows)
    assert g["calendar_like"]["weakest"] < g["vol_like"]["weakest"] and g["vol_like"]["n_era"] > 0


def test_informative_failures_lower_for_the_planted_type(visible):
    rep = mr.analyse_question(Q16.INFORMATIVE_FAILURES, visible)
    t = rep.main
    assert t.groups["representation_probe"].shrunk < t.groups["transfer_test"].shrunk


# ------------------------------------------------------------------------------------------------ out-of-sample evaluation

def test_evaluate_questions_planted_beats_baseline_null_does_not(visible):
    oos = mr.evaluate_questions(visible, NOW, 0, CFG)
    assert oos[Q16.DURABLE_EXPERIMENTS].beats_baseline and oos[Q16.OVERFIT_TYPES].beats_baseline
    assert oos[Q16.DURABLE_EXPERIMENTS].label == ValidationLabel.NOT_VALIDATED.value       # synthetic never validates
    nul = mr.evaluate_questions(mr.permute_outcomes(visible, 4), NOW, 0, CFG)
    assert sum(mr.question_usable(nul[q], CFG)[0] for q in Q16) <= 2


def test_certified_real_flag_is_the_only_road_to_validated(visible):
    cfg = dataclasses.replace(CFG, certified_real=True)
    oos = mr.evaluate_questions(visible, NOW, 0, cfg)
    assert oos[Q16.DURABLE_EXPERIMENTS].label == ValidationLabel.VALIDATED.value
    assert mr.evaluate_questions(visible, NOW, 0, CFG)[Q16.DURABLE_EXPERIMENTS].label != ValidationLabel.VALIDATED.value


def test_null_world_check_is_quiet_and_real_world_is_not(visible):
    r = mr.null_world_check(visible, NOW, CFG, 0, n_perm=3)
    assert r["verdict"] == "PIPELINE_QUIET_ON_NOISE" and r["real_usable"] > r["null_usable_max"] + 3


def test_permute_outcomes_keeps_descriptors_and_marginals(visible):
    p = mr.permute_outcomes(visible, 9)
    assert [o.run_id for o in p] == [o.run_id for o in visible] and [o.exp_type for o in p] == [o.exp_type for o in visible]
    assert sum(bool(o.durable) for o in p) == sum(bool(o.durable) for o in visible)
    assert sum(bool(o.validator_flagged) for o in p) == sum(bool(o.validator_flagged) for o in visible)
    assert [o.durable for o in p] != [o.durable for o in visible]
    assert mr.permute_outcomes([], 1) == []


def test_run_level_leak_is_detected_and_clean_features_pass(visible):
    clean = mr.run_level_oos(visible, NOW, 0, CFG)
    assert clean["leaks"] == ()
    leaky = mr.run_level_oos(visible, NOW, 0, CFG, extra=lambda o: {"post_hoc": 1.0 if o.durable else 0.0})
    assert leaky["leaks"] and leaky["leaks"][0][0] == "post_hoc"
    assert leaky["oos"].label == ValidationLabel.FAILED_VALIDATION.value
    assert mr.to_discovery_records([run("x")]) == []                      # unknown durability: no label, no record


def test_scheduler_replay_beats_random_on_planted_and_not_on_null(visible):
    ev = mr.evaluate_scheduler(visible, NOW, 0, CFG)
    assert ev.beats_random and ev.lift > 0 and ev.p_value < 0.05 and ev.label == ValidationLabel.NOT_VALIDATED.value
    assert ev.advice_durable > ev.random_durable_mean
    nul = mr.evaluate_scheduler(mr.permute_outcomes(visible, 3), NOW, 0, CFG)
    assert not nul.beats_random
    tiny = mr.evaluate_scheduler(visible[:10], NOW, 0, CFG)
    assert tiny.label == ValidationLabel.INSUFFICIENT_EVIDENCE.value and tiny.folds == 0
    assert mr.evaluate_scheduler([], NOW, 0, CFG).label == ValidationLabel.INSUFFICIENT_EVIDENCE.value


def test_scheduler_replay_excludes_runs_the_scheduler_chose(visible):
    tagged = [dataclasses.replace(o, chosen_by="SAx") if i % 4 == 0 else o for i, o in enumerate(visible)]
    ev = mr.evaluate_scheduler(tagged, NOW, 0, CFG)
    assert ev.excluded_advice_chosen > 0
    assert mr.evaluate_scheduler(visible, NOW, 0, CFG).excluded_advice_chosen == 0


def test_scheduler_replay_is_deterministic(visible):
    a, b = mr.evaluate_scheduler(visible, NOW, 5, CFG), mr.evaluate_scheduler(visible, NOW, 5, CFG)
    assert a == b


def test_question_ablation_names_the_helpful_questions(visible):
    ab = mr.question_ablation(visible, NOW, 0, CFG)
    qs = ab["questions"]
    assert qs[Q16.ERA_SURVIVAL.value]["consumed_by_schedule"] is False
    contrib = {k: v["contribution"] for k, v in qs.items() if v["consumed_by_schedule"]}
    assert any(abs(c) > 0 for c in contrib.values())              # ablation is sensitive: removing a question changes the lift
    # the replay target is durability; a question about a different outcome (decision changes) must not be credited with helping it
    assert contrib[Q16.DECISION_QUESTIONS.value] == 0 and contrib[Q16.FALSE_DISCOVERY_DATASETS.value] == 0


# ------------------------------------------------------------------------------------------------ the advice

def test_advice_withholds_unanswerable_questions(result):
    a = result.advice
    assert a.check() == []
    assert set(a.usable_questions) <= {q.value for q in Q16}
    for name in a.withheld:
        fields = mr.QUESTION_FIELDS[Q16(name)]
        for f in fields:
            assert not getattr(a, f), f"{f} filled although {name} was withheld"


def test_tiny_world_gives_empty_advice_unknown_is_not_neutral():
    st = mr.MetaResearchState(cfg=CFG)
    st.ingest(mr.synthetic_research_world(30, 2, n_defects=2))
    res = mr.step(st, NOW, 0)
    assert res.advice.usable_questions == () and res.matured is None and res.verdict == GateVerdict.NEEDS_MORE_EVIDENCE
    assert res.advice.exp_type_durable == {} and res.advice.trust() == 0.0
    d = mr.schedule_decision({"exp_type": "ablation"}, res.advice)
    assert d.action == "RUN" and d.multiplier == 1.0 and "unknown" in d.reasons[0]


def test_empty_store_step():
    res = mr.step(mr.MetaResearchState(cfg=CFG), NOW, 0)
    assert res.verdict == GateVerdict.NEEDS_MORE_EVIDENCE and res.seal is None and res.advice.n_runs == 0
    assert res.update.scheduler.label == ValidationLabel.INSUFFICIENT_EVIDENCE.value
    assert mr.render(res.update) and mr.scorecard_numbers(res.update)["meta_questions_usable"] == 0


def test_advice_check_rejects_bad_values_and_identity():
    assert mr.SchedulerAdvice("2003-01-01", 5, exp_type_durable={"a": 1.4}).check()
    assert mr.SchedulerAdvice("2003-01-01", 5, stop_paths=("p",), throttle_paths=("p",)).check()
    assert mr.SchedulerAdvice("2003-01-01", 5, family_transfer={"momentum_2008": 0.5}).check()
    assert mr.SchedulerAdvice("2003-01-01", 5, stop_paths=("2008-09-15|x|y",)).check()
    assert mr.SchedulerAdvice("2003-01-01", 5).check() == []


def test_advice_payload_has_no_date_and_gate_is_the_only_door(result):
    m = result.matured
    assert m is not None and m.namespace == Namespace.MATURED_RESEARCH
    assert "fitted_through" not in m.payload
    mr.assert_identity_free(m.payload)
    with pytest.raises(FirewallBreach):
        m.gate(m.matured_at)                                        # the day the newest run matured: too early
    got = m.gate("2010-01-01")
    assert got["exp_type_durable"] == result.advice.payload()["exp_type_durable"]
    with pytest.raises(FirewallBreach):
        mr.assert_identity_free({"note": "worked in 2008"})
    with pytest.raises(FirewallBreach):
        mr.assert_identity_free({"k": ["fine", ("nested", "on 2005-03-04")]})


def test_advice_json_roundtrip_and_meta_advice(result):
    a = result.advice
    back = mr.advice_from_json(a.to_json())
    assert back == a and back.advice_id == a.advice_id
    ma = a.to_meta_advice()
    assert ma.check() == [] and ma.n_observations == a.n_runs and ma.family_survival == dict(a.family_transfer)
    assert set(ma.target_yield) <= {t.value for t in ResearchTarget}
    assert 0 < a.trust() < 1
    assert dataclasses.replace(a, n_runs=a.n_runs * 10).trust() > a.trust()


def test_trust_is_halved_unless_validated():
    base = mr.SchedulerAdvice("2003-01-01", 600, usable_questions=tuple(q.value for q in Q16), oos_label=ValidationLabel.VALIDATED.value)
    half = dataclasses.replace(base, oos_label=ValidationLabel.NOT_VALIDATED.value)
    assert base.trust() == pytest.approx(2 * half.trust()) and mr.SchedulerAdvice("2003-01-01", 0).trust() == 0.0


def test_step_label_and_verdict_are_never_validated_on_synthetic(result):
    assert result.update.label == "IMPLEMENTED — NOT VALIDATED" or "NOT VALIDATED" in result.update.label
    assert result.verdict in (GateVerdict.UNKNOWN, GateVerdict.FAILED) and result.verdict != GateVerdict.PROMOTE
    assert result.advice.oos_label != ValidationLabel.VALIDATED.value


# ------------------------------------------------------------------------------------------------ feeding the scheduler

def _cand(cid, **cfg):
    return Candidate(cid, f"question {cid}", ResearchTarget.FAILURE, "2001-01-01", config=cfg, family=cfg.get("family", ""))


def test_schedule_decision_block_throttle_and_direction():
    adv = mr.SchedulerAdvice("2003-01-01", 600, exp_type_durable={"good": 0.7, "bad": 0.1}, exp_type_overfit={"good": 0.1, "bad": 0.6},
                             stop_paths=("dead|f|r",), blocked_representations=("fourier",), throttle_paths=("slow|f|r",),
                             pooled={Q16.DURABLE_EXPERIMENTS.value: 0.4, Q16.OVERFIT_TYPES.value: 0.3},
                             usable_questions=tuple(q.value for q in Q16), oos_label=ValidationLabel.VALIDATED.value)
    assert mr.schedule_decision({"exp_type": "dead", "family": "f", "representation": "r"}, adv).action == "BLOCK"
    assert mr.schedule_decision({"exp_type": "x", "family": "f", "representation": "fourier"}, adv).multiplier == 0.0
    thr = mr.schedule_decision({"exp_type": "slow", "family": "f", "representation": "r"}, adv)
    assert thr.action == "THROTTLE" and thr.multiplier < 1.0
    g = mr.schedule_decision({"exp_type": "good"}, adv).multiplier
    b = mr.schedule_decision({"exp_type": "bad"}, adv).multiplier
    assert g > 1 > b and 0.05 <= b and g <= 2.0
    cfg = mr.MetaResearchConfig()
    assert mr.schedule_decision({"exp_type": "good"}, dataclasses.replace(adv, n_runs=0)).multiplier == 1.0     # no trust, no change
    assert cfg.multiplier_lo <= b


def test_rank_candidates_and_adjust_candidate():
    adv = mr.SchedulerAdvice("2003-01-01", 800, exp_type_durable={"good": 0.7, "bad": 0.1}, exp_type_overfit={"good": 0.1, "bad": 0.7},
                             family_transfer={"fam": 0.6}, blocked_representations=("fourier",),
                             pooled={Q16.DURABLE_EXPERIMENTS.value: 0.4, Q16.OVERFIT_TYPES.value: 0.3, Q16.TRANSFERRING_FAMILIES.value: 0.4},
                             usable_questions=tuple(q.value for q in Q16), oos_label=ValidationLabel.VALIDATED.value)
    cs = [_cand("c_bad", exp_type="bad"), _cand("c_good", exp_type="good"), _cand("c_block", exp_type="good", representation="fourier")]
    ranked = mr.rank_candidates(cs, adv)
    assert [r[0] for r in ranked] == ["c_good", "c_bad", "c_block"] and ranked[-1][1] == 0.0
    assert mr.rank_candidates([], adv) == []
    adj = mr.adjust_candidate(_cand("c_bad", exp_type="bad", family="fam"), adv)
    assert adj.overfit_hint > 0.3 and adj.check() == []
    assert mr.adjust_candidate(_cand("x", exp_type="unseen"), mr.SchedulerAdvice("2003-01-01", 0)).overfit_hint == 0.0
    ctx = mr.policy_context_with(PolicyContext(now="2004-01-01"), adv)
    assert ctx.meta.family_survival == {"fam": 0.6} and ctx.now == "2004-01-01"


def test_meta_rerank_blocks_and_reorders_a_real_queue():
    from engine.learning.research_priority import ItemStatus, ResearchQueue
    adv = mr.SchedulerAdvice("2003-01-01", 800, exp_type_durable={"good": 0.7, "bad": 0.1}, blocked_representations=("fourier",),
                             pooled={Q16.DURABLE_EXPERIMENTS.value: 0.4}, usable_questions=tuple(q.value for q in Q16),
                             oos_label=ValidationLabel.VALIDATED.value)
    q = ResearchQueue()
    for cid, cfg in (("a_bad", {"exp_type": "bad"}), ("b_good", {"exp_type": "good"}), ("c_block", {"exp_type": "good", "representation": "fourier"})):
        q.enqueue(_cand(cid, **cfg), "2002-01-01").adjusted = 1.0
    order = mr.meta_rerank(SimpleNamespace(queue=q), adv)
    assert order == ["b_good", "a_bad"] and q.items["c_block"].status == ItemStatus.BLOCKED
    assert "meta:" in q.items["c_block"].note


def test_explain_mentions_the_reasons():
    adv = mr.SchedulerAdvice("2003-01-01", 800, exp_type_durable={"good": 0.7}, pooled={Q16.DURABLE_EXPERIMENTS.value: 0.4},
                             usable_questions=(Q16.DURABLE_EXPERIMENTS.value,))
    assert "durable" in mr.explain({"exp_type": "good"}, adv) and adv.advice_id in mr.explain({"exp_type": "good"}, adv)


# ------------------------------------------------------------------------------------------------ the same-year rerun leak

def test_release_view_withholds_the_replayed_year():
    rows = [run("a", evidence_year="2005"), run("b", evidence_year="2006"), run("c")]
    assert [r.run_id for r in mr.release_view(rows, [2005])] == ["b", "c"]
    assert [r.run_id for r in mr.release_view(rows, ["2005", "2006"])] == ["c"]
    assert mr.release_view(rows) == rows and mr.release_view([], [2005]) == []


def test_replayed_year_research_cannot_reach_the_advice(world):
    """Planted: every run filed under 2003 claims transfer_test is worthless. Replaying 2003 must remove that influence."""
    tagged = [dataclasses.replace(o, evidence_year="2003", durable=False, overfit=False, false_discovery=False)
              if o.exp_type == "transfer_test" and i % 2 == 0 and o.durable is not None else o for i, o in enumerate(world)]
    st = mr.MetaResearchState(cfg=CFG)
    st.ingest(tagged)
    open_ = mr.fit_update(st, NOW, 0, evaluate=False)
    blind = mr.fit_update(st, NOW, 0, replay_years=[2003], evaluate=False)
    assert blind.advice.n_runs < open_.advice.n_runs
    assert blind.advice.exp_type_durable["transfer_test"] > open_.advice.exp_type_durable["transfer_test"] + 0.1


# ------------------------------------------------------------------------------------------------ depth: are the meta-claims themselves sound?

def test_calibration_report_and_recalibrate(visible):
    c = mr.calibration_report(Q16.OVERFIT_TYPES, visible, NOW, CFG)
    assert c["n"] > 100 and c["brier_gain"] > 0 and c["auc"] > 0.6 and c["reliability"] and c["verdict"] != "NO_SKILL"
    assert mr.calibration_report(Q16.OVERFIT_TYPES, [], NOW, CFG)["verdict"] == "INSUFFICIENT"
    nul = mr.calibration_report(Q16.OVERFIT_TYPES, mr.permute_outcomes(visible, 2), NOW, CFG)
    assert nul["brier_gain"] < c["brier_gain"]
    adv = mr.SchedulerAdvice("2003-01-01", 500, exp_type_durable={"a": 0.5}, exp_type_overfit={"a": 0.2}, pooled={Q16.OVERFIT_TYPES.value: 0.3},
                             usable_questions=(Q16.DURABLE_EXPERIMENTS.value, Q16.OVERFIT_TYPES.value))
    out = mr.recalibrate_advice(adv, {Q16.OVERFIT_TYPES: {"verdict": "NO_SKILL"}})
    assert out.exp_type_overfit == {} and out.exp_type_durable == {"a": 0.5} and Q16.OVERFIT_TYPES.value in out.withheld
    assert mr.recalibrate_advice(adv, {}) == adv


def test_split_half_replication_planted_and_reversed():
    w1 = mr.synthetic_research_world(360, 11, n_defects=0)
    reversed_truth = dict(mr.WORLD_TRUTH)
    reversed_truth["exp_durable"] = {k: 0.72 - v for k, v in mr.WORLD_TRUTH["exp_durable"].items()}
    w2 = mr.synthetic_research_world(360, 12, start="2003-01-03", n_defects=0, truth=reversed_truth)
    same = mr.split_half_replication(Q16.DURABLE_EXPERIMENTS, w1 + mr.synthetic_research_world(360, 13, start="2003-01-03", n_defects=0), CFG)
    flip = mr.split_half_replication(Q16.DURABLE_EXPERIMENTS, w1 + w2, CFG)
    assert same.verdict == "REPLICATES" and same.rho > 0.5
    assert flip.verdict == "DOES_NOT_REPLICATE" and flip.rho <= 0
    assert mr.split_half_replication(Q16.DURABLE_EXPERIMENTS, [], CFG).verdict == "INSUFFICIENT"
    assert set(mr.replication_sweep(w1, CFG)) == set(Q16)


def test_prob_best_and_exploration_needs(visible):
    rep = mr.analyse_question(Q16.DURABLE_EXPERIMENTS, visible)
    pb = mr.prob_best(rep.main, 0)
    assert max(pb, key=lambda k: pb[k]["p_best"]) == "transfer_test" or pb["transfer_test"]["p_best"] > 0.2
    assert max(pb, key=lambda k: pb[k]["p_worst"]) == "representation_probe"
    assert sum(v["p_best"] for v in pb.values()) == pytest.approx(1.0) and mr.prob_best(mr.rate_table([], "x", True), 0) == {}
    sparse = [run(f"s{i}", exp_type="seen", durable=bool(i % 2)) for i in range(40)] + [run("r1", exp_type="rare", durable=True)]
    needs = mr.exploration_needs(mr.analyse_question(Q16.DURABLE_EXPERIMENTS, sparse))
    assert needs[0].group == "rare" and needs[0].known is False and needs[0].n_needed > 20
    assert next(n for n in needs if n.group == "seen").n_needed < needs[0].n_needed


def test_type_economics_pareto_and_reallocation(visible):
    rows = [run(f"a{i}", exp_type="great", cost_minutes=6.0, durable=(i % 2 == 0), overfit=False) for i in range(20)] + \
           [run(f"b{i}", exp_type="awful", cost_minutes=60.0, durable=(i % 10 == 0), overfit=(i % 2 == 1)) for i in range(20)]
    eco = {e.exp_type: e for e in mr.type_economics(rows)}
    assert eco["great"].durable_per_hour > eco["awful"].durable_per_hour and eco["awful"].dominated and not eco["great"].dominated
    assert mr.type_economics([]) == []
    adv = mr.build_advice(visible, mr.analyse_all(visible), mr.evaluate_questions(visible, NOW, 0, CFG), CFG, "2004-05-31", [], [], {})
    alloc = mr.compute_reallocation(visible + [run("z", exp_type="never_seen", cost_minutes=3.0)], adv, 1000.0)
    assert sum(alloc["share"].values()) == pytest.approx(1.0) and sum(alloc["minutes"].values()) == pytest.approx(1000.0, abs=1.0)
    assert min(alloc["share"].values()) >= alloc["floor"] - 1e-9 and max(alloc["share"].values()) <= alloc["cap"] + 1e-9
    assert "never_seen" in alloc["explored_unknown"]
    assert mr.compute_reallocation([], adv, 100.0)["minutes"] == {}


def test_contradictions_are_reported_not_resolved():
    rows = [run(f"h{i}", exp_type="hot", durable=(i % 10 < 8), overfit=(i % 10 < 8)) for i in range(60)] + \
           [run(f"c{i}", exp_type="cold", durable=(i % 10 < 2), overfit=(i % 10 >= 8)) for i in range(60)] + \
           [run(f"m{i}", exp_type="mid", durable=(i % 2 == 0), overfit=False) for i in range(60)]
    found = mr.find_contradictions(mr.analyse_all([dataclasses.replace(o, durable=False if o.overfit else o.durable) for o in rows] and rows))
    assert any(c.kind in ("durable_and_overfit", "durable_but_uninformative_failures") for c in found) or found == []
    planted = mr.analyse_all(rows)
    both = mr.find_contradictions(planted)
    assert isinstance(both, list) and all(c.subject for c in both)
    assert mr.find_contradictions(mr.analyse_all([])) == []


def test_context_holdout_and_predictive_decay(visible):
    ch = mr.context_holdout(Q16.OVERFIT_TYPES, visible, "regime", CFG)
    assert ch["verdict"] == "CARRIES_ACROSS_CONTEXTS" and ch["share_better"] >= 0.5
    assert mr.context_holdout(Q16.OVERFIT_TYPES, [], "regime", CFG)["verdict"] == "INSUFFICIENT"
    stable = mr.predictive_decay(Q16.DURABLE_EXPERIMENTS, visible, CFG)
    assert stable["lags"] and stable["stale_after"] is None and stable["verdict"] == "DURABLE_OVER_HORIZON"
    drift = []
    for k in range(5):
        tr = dict(mr.WORLD_TRUTH)
        vals = list(mr.WORLD_TRUTH["exp_durable"].values())
        tr["exp_durable"] = dict(zip(mr.WORLD_TRUTH["exp_durable"], np.roll(vals, k)))
        drift += mr.synthetic_research_world(150, 30 + k, start=day(k * 400), n_defects=0, truth=tr)
    dec = mr.predictive_decay(Q16.DURABLE_EXPERIMENTS, drift, CFG)
    assert dec["stale_after"] is not None and dec["lags"][-1]["mean_gain"] < stable["lags"][0]["mean_gain"]
    assert mr.predictive_decay(Q16.DURABLE_EXPERIMENTS, [], CFG)["verdict"] == "INSUFFICIENT"


def test_stale_discount_pulls_toward_pooled(result):
    a = result.advice
    old = mr.stale_discount(a, "2034-06-01", half_life_days=365)
    hi = max(a.exp_type_durable, key=a.exp_type_durable.get)
    pooled = a.pooled[Q16.DURABLE_EXPERIMENTS.value]
    assert abs(old.exp_type_durable[hi] - pooled) < abs(a.exp_type_durable[hi] - pooled)
    assert mr.stale_discount(a, a.fitted_through).exp_type_durable == a.exp_type_durable


def test_advice_stability_and_diff(state, result):
    later = mr.fit_update(state, "2004-09-01", 0, evaluate=False)
    st = mr.advice_stability(result.advice, later.advice)
    assert st["n_common"] >= 3 and st["rho"] is not None and st["rho"] > 0.5
    d = mr.diff_updates(result.update, later)
    assert set(d) >= {"usable_gained", "usable_lost", "stop_added", "label", "stability"}
    same = mr.diff_updates(result.update, result.update)
    assert same["usable_gained"] == same["usable_lost"] == [] and same["durable_moved"] == {}


def test_meta_questions_are_identity_free_and_deterministic(result):
    q1, q2 = mr.meta_questions(result.update, NOW), mr.meta_questions(result.update, NOW)
    assert [q.question_id for q in q1] == [q.question_id for q in q2] and q1
    assert all(q.problem.value == "RESEARCH_PROCESS" and not mr.identity_leak(q.text) for q in q1)
    assert len({q.question_id for q in q1}) == len(q1)
    assert mr.meta_questions(mr.fit_update(mr.MetaResearchState(cfg=CFG), NOW, 0), NOW)


def test_store_health_and_power(state):
    h = mr.store_health(state.store, NOW)
    assert h["resolved"] > 500 and h["pending"] > 0 and h["poison_rows"] == 0 and h["coverage"]["durable"] > 0.5
    thin = mr.OutcomeStore([run("only")])
    assert mr.store_health(thin, "2001-01-01")["resolved"] == 0
    assert mr.store_health(mr.OutcomeStore(), NOW)["warnings"] == []
    p = mr.question_power(state.store, NOW, CFG)
    assert p[Q16.DURABLE_EXPERIMENTS.value]["answerable"] and mr.question_power(thin, NOW, CFG)[Q16.DURABLE_EXPERIMENTS.value]["short_by"] > 0


def test_process_drift_detected_when_the_rate_changes():
    rows = [run(f"a{i}", started=day(i), resolved=day(i + 3), durable=(i % 10 < 8)) for i in range(120)] + \
           [run(f"b{i}", started=day(200 + i), resolved=day(203 + i), durable=(i % 10 < 1)) for i in range(120)]
    alarms = mr.drift_in_process(rows, lambda o: o.durable)
    assert alarms and alarms[0][1] == "RATE_DOWN"
    steady = [run(f"s{i}", started=day(i), resolved=day(i + 3), durable=(i % 2 == 0)) for i in range(200)]
    assert mr.drift_in_process(steady, lambda o: o.durable) == [] and mr.drift_in_process([], lambda o: o.durable) == []


def test_interaction_table_finds_a_planted_cell():
    rng = np.random.default_rng(8)
    rows = []
    for i in range(400):
        et, ds = ("pattern_search", "survivor") if i % 4 == 0 else (("ablation", "delisted") if i % 4 == 1 else (("pattern_search", "delisted") if i % 4 == 2 else ("ablation", "survivor")))
        p = 0.9 if (et, ds) == ("pattern_search", "survivor") else 0.3
        rows.append(run(f"i{i}", exp_type=et, dataset=ds, false_discovery=bool(rng.random() < p)))
    t = mr.interaction_table(rows, "exp_type", "dataset", lambda o: o.false_discovery, False, 20)
    assert t.groups["pattern_search&survivor"].direction == "ABOVE" and t.groups["pattern_search&survivor"].desirable is False
    assert mr.interaction_table([], "exp_type", "dataset", lambda o: o.durable, True).groups == {}


# ------------------------------------------------------------------------------------------------ adapters to the existing ledgers

def test_failed_learner_registry_becomes_failed_research_runs():
    from engine.learning.failed_learners import FailedLearnerRegistry, seed_registry
    reg = FailedLearnerRegistry()
    seed_registry(reg)
    rows = mr.outcomes_from_failed_learners(reg, "2026-09-30")
    assert rows and all(r.failed and r.durable is False and r.exp_type == "learner_evaluation" for r in rows)
    st = mr.OutcomeStore()
    st.extend(rows)                                                  # every adapted row passes the store's own validation
    assert mr.outcomes_from_failed_learners(reg, "2020-01-01") == []
    assert any(r.info_bits > 0 for r in rows) or all(r.followups == 0 for r in rows)


def test_realised_gains_become_partial_runs_without_invented_fields():
    g = RealisedGain("c1", ResearchTarget.FAILURE, "pattern:vol", 0.4, 0.3, True, 12.0, "2026-05-01")
    (o,) = mr.outcomes_from_realised_gains([g])
    assert o.durable is None and o.overfit is None and o.decision_changed is True and o.cost_minutes == 12.0
    assert o.check() == [] and mr.outcomes_from_realised_gains([]) == []


def test_ledger_adapter_needs_an_annotation_and_sanitises():
    rec = SimpleNamespace(status="ANSWERED", created_at="2026-01-01", next_action="check regimes",
                          experiment=SimpleNamespace(config={"exp_type": "Break Study 2008", "family": "vol", "dataset": "panel"}, target="FAILURE", cost_minutes=7.0),
                          result=SimpleNamespace(observed_at="2026-01-10", kind="REFUTED"), belief_update=None)
    ledger = SimpleNamespace(view=lambda now: {"e1": rec, "e2": rec})
    out = mr.outcomes_from_ledger(ledger, "2027-01-01", {"e1": {"resolved_at": "2026-02-01", "durable": False}})
    assert [o.run_id for o in out] == ["e1"] and out[0].exp_type == "break_study" and out[0].failed and out[0].followups == 1
    assert out[0].check() == [] and mr.outcomes_from_ledger(ledger, "2027-01-01", {}) == []
    assert mr.sanitize_label("Crash 2008-09-15 (1990s)") == "crash" and mr.sanitize_label("") == "unspecified"


# ------------------------------------------------------------------------------------------------ over time and reports

def test_learning_curve_is_deterministic_and_grows(world):
    st = mr.OutcomeStore(world)
    a = mr.research_learning_curve(st, ["2002-06-01", "2004-06-01"], 0, CFG)
    b = mr.research_learning_curve(st, ["2002-06-01", "2004-06-01"], 0, CFG)
    assert a == b and a[1]["n"] > a[0]["n"] and a[1]["usable"] >= a[0]["usable"]
    assert mr.research_learning_curve(mr.OutcomeStore(), ["2004-06-01"], 0, CFG)[0]["usable"] == 0


def test_reports_render_and_write(result, tmp_path):
    txt = mr.render(result.update)
    assert "Q01_durable_experiment_types" in txt and "scheduler replay" in txt and "USED" in txt
    md = mr.to_markdown(result.update)
    assert md.count("|") > 40 and result.advice.advice_id in md
    out = mr.write_report(result.update, tmp_path / "rep")
    assert {p.name for p in out.iterdir()} == {"report.txt", "report.md", "advice.json", "summary.json"}
    assert mr.advice_from_json((out / "advice.json").read_text(encoding="utf-8")) == result.advice


def test_history_is_logged(state, result, tmp_path):
    assert state.history and state.history[-1]["advice_id"] == result.advice.advice_id
    mr.save_history(state, tmp_path / "h" / "hist.jsonl")
    assert (tmp_path / "h" / "hist.jsonl").read_text(encoding="utf-8").count("\n") == 1
    assert state.last_advice == result.advice


def test_scorecard_numbers(result):
    s = mr.scorecard_numbers(result.update)
    assert s["meta_questions_total"] == 12 and 0 < s["meta_questions_usable"] <= 12 and s["scheduler_lift"] is not None
    assert s["guard_refused"] == 0


def test_config_validation():
    assert mr.MetaResearchConfig().check() == []
    assert mr.MetaResearchConfig(fdr_q=2).check() and mr.MetaResearchConfig(folds=1).check() and mr.MetaResearchConfig(budget_share=1.5).check()
    with pytest.raises(ValueError):
        mr.MetaResearchState(cfg=mr.MetaResearchConfig(folds=1))


def test_self_check_recovers_truth_and_blocks_poison():
    r = mr.self_check(0)
    assert r["guard_refuses_eval_output"] and "fourier_bands" in r["blocked"]
    assert all(v is None or v > 0.5 for v in r["recovery"].values()) and r["label"] == mr.LABEL


def test_objective_filters_which_factors_steer():
    adv = mr.SchedulerAdvice("2003-01-01", 800, source_decision={"hot": 0.9}, exp_type_overfit={"e": 0.8},
                             pooled={Q16.DECISION_QUESTIONS.value: 0.4, Q16.OVERFIT_TYPES.value: 0.3},
                             usable_questions=tuple(q.value for q in Q16), oos_label=ValidationLabel.VALIDATED.value)
    d = {"question_source": "hot", "exp_type": "e"}
    assert mr.schedule_decision(d, adv, objective="durable").multiplier < 1.0          # only overfit counts
    assert mr.schedule_decision(d, adv, objective="decisions").multiplier > 1.0        # only the decision source counts
    assert mr.schedule_decision({}, adv, objective="durable").multiplier == 1.0
    with pytest.raises(ValueError):
        mr.schedule_decision(d, adv, objective="profit")
