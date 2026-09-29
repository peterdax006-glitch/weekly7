"""Integration of the learning brain into the old engine (S17a; contract C62 sections 55, 61, 85; canon C62-C65).

Every hook in engine/learning/wiring.py is exercised END TO END through the real old call site (lessons.post_mortem,
LessonBook.learn, Memory.export_lessons, MissedLedger.observe, improve.log_experiment / spawn_challengers / test_and_promote,
Registry.audit, ExperimentMemory.record, PatternMiner.fit, LessonBook.factor) and the test asserts that the data ARRIVED in the
learning structure, not merely that nothing crashed. Each mechanism also has a planted case it must catch and an empty case.
Synthetic data only; nothing here validates anything on real data: IMPLEMENTED - NOT VALIDATED."""
import dataclasses
import datetime as dt
import json
import re

import numpy as np
import pandas as pd
import pytest

from engine import improve as IMP
from engine import lessons as LS
from engine import memory as MEM
from engine import missed_winners as MWB
from engine import registry as REG
from engine.learning import archive as AR
from engine.learning import champion as CH
from engine.learning import curator as CU
from engine.learning import experiment_memory as EM
from engine.learning import trader_view as TV
from engine.learning import failure as FA
from engine.learning import wiring as W
from engine.learning.core import (Confidence, DecisionEffect, Edge, Epistemic, FirewallBreach, Lifecycle, Promotion, Provenance,
                                  Subsystem, ValidationLabel, TemporalClass)
from engine.learning.reproducibility import ReproRecord
from engine.patterns import PatternMiner

from tests.test_learning_firewalls import clean_ctx
from tests.test_learning_identity_firewall import make_panel as ident_panel
from tests.test_learning_scorecard import good_card


@pytest.fixture(autouse=True)
def fresh_hub():
    W.HUB.reset()
    yield W.HUB
    W.HUB.reset()


# ================================================================================================================ fixtures
def lesson_panel(seed=0, n_days=120, n_tk=40):
    """f0 is the model's signal; in the trap (m_vix high and f2 > 0.3) the signal reverses (the same world tests/test_lessons.py uses)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    tks = [f"T{i:02d}" for i in range(n_tk)]
    idx = pd.MultiIndex.from_product([dates, tks], names=["date", "ticker"])
    n = len(idx)
    X = pd.DataFrame({"f0": rng.normal(size=n), "f1": rng.normal(size=n), "f2": rng.normal(size=n), "f3": rng.normal(size=n),
                      "m_vix": np.repeat(rng.normal(size=n_days), n_tk), "m_breadth": np.repeat(rng.normal(size=n_days), n_tk)}, index=idx)
    y = 0.02 * X["f0"] + rng.normal(scale=0.03, size=n)
    region = (X["m_vix"] > 0.3) & (X["f2"] > 0.3)
    y = y.where(~region, -0.04 * X["f0"] + rng.normal(scale=0.03, size=n))
    return X, y


@pytest.fixture(scope="module")
def world():
    from engine.antimemo import play_window
    X, y = lesson_panel()
    _, fr = play_window(X, y, lambda Xd: Xd["f0"].copy(), None, k=5, collect_frame=True, horizon_days=7)
    return X, y, fr


def full_fields(**over):
    base = dict(window_ids=["w1"], train_range="a", validation_range="b", test_range="c", metrics={"m": 1.0}, gates={"g": True},
                outcome="reject", reason="did not help")
    base.update(over)
    return base


def soon():
    return dt.datetime.utcnow() + dt.timedelta(seconds=30)


# ================================================================================================================ post_mortem -> failure
def test_post_mortem_losses_reach_the_failure_ledger_with_hypotheses_only(world, fresh_hub):
    X, y, fr = world
    eps = LS.post_mortem(fr, X, fr["resolved"].max(), seed=0)
    assert eps                                                                  # the old return value is untouched
    hub = fresh_hub
    assert hub.calls["post_mortem"] == 1 and not hub.errors
    n = len(hub.failure_ledger)
    assert 0 < n <= W.MAX_CLASSIFY
    assert hub.delivered["failure_classifications"] == n
    assert hub.hypotheses, "losses were classified but no hypothesis proposed"
    for h in hub.hypotheses.values():
        assert h.validate() == [] and h.epistemic == Epistemic.HYPOTHESIS and not h.production_effect
    blob = json.dumps({k: dataclasses.asdict(v) for k, v in hub.hypotheses.items()}, default=str)
    assert not re.search(r"\bT\d\d\b", blob), "a ticker leaked into a hypothesis"
    assert 0.0 <= hub.failure_ledger.unknown_rate() <= 1.0                     # a classifier that never says UNKNOWN would be forcing


def test_post_mortem_never_classifies_an_outcome_that_matures_at_or_after_now(world, fresh_hub):
    X, y, fr = world
    cut = fr["resolved"].sort_values().iloc[len(fr) // 3]
    LS.post_mortem(fr, X, cut, seed=0)
    early = len(fresh_hub.failure_ledger)
    losses_before = W._worst_taken_losses(fr, cut, 0.0005, 10 ** 9)
    assert (losses_before["resolved"] < cut).all()
    assert early <= len(losses_before)
    fresh_hub.reset()
    LS.post_mortem(fr, X, fr["resolved"].min(), seed=0)                        # nothing has resolved strictly before the first resolution
    assert len(fresh_hub.failure_ledger) == 0
    fresh_hub.reset()
    LS.post_mortem(fr, X, fr["resolved"].max() + pd.Timedelta(days=1), seed=0)
    assert len(fresh_hub.failure_ledger) >= early


def test_post_mortem_ranks_come_from_the_whole_cross_section_not_the_loser_subset(world, fresh_hub):
    X, y, fr = world
    now = fr["resolved"].max() + pd.Timedelta(days=1)
    seen = []
    orig = fresh_hub.classifier.classify
    fresh_hub.classifier.classify = lambda t, env, nw: (seen.append(t), orig(t, env, nw))[1]
    W.on_post_mortem(fr, X, now)
    worst = W._worst_taken_losses(fr, now, 0.0005, W.MAX_CLASSIFY)
    full = fr["score"].abs().groupby(level=0).rank(pct=True).loc[worst.index]
    sub = worst["score"].abs().groupby(level=0).rank(pct=True)
    assert sorted(round(t.rank_pct, 9) for t in seen) == sorted(round(float(v), 9) for v in full)
    assert sorted(round(float(v), 9) for v in full) != sorted(round(float(v), 9) for v in sub), "the planted difference vanished"


def test_post_mortem_is_idempotent_per_loss_and_handles_empty_input(world, fresh_hub):
    X, y, fr = world
    now = fr["resolved"].max() + pd.Timedelta(days=1)
    first = W.on_post_mortem(fr, X, now)
    n = len(fresh_hub.failure_ledger)
    again = W.on_post_mortem(fr, X, now)
    assert first["classified"] == n and again["classified"] == 0 and len(fresh_hub.failure_ledger) == n
    assert W.on_post_mortem(fr.iloc[:0], X, now) == {"seen": 0, "classified": 0, "named": 0, "hypotheses": 0}
    assert W.on_post_mortem(None, X, now)["seen"] == 0


def test_a_failing_sink_never_breaks_the_old_path_but_is_visible(world, fresh_hub):
    X, y, fr = world
    with pytest.warns(UserWarning, match="sink post_mortem failed"):
        assert W.on_post_mortem(fr, X, "not a date") is None
    assert fresh_hub.errors and fresh_hub.errors[0].hook == "post_mortem"
    with pytest.raises(FirewallBreach, match="sink error"):
        W.assert_hooks_healthy()
    fresh_hub.enabled = False
    calls = fresh_hub.calls["post_mortem"]
    assert LS.post_mortem(fr, X, fr["resolved"].max(), seed=0)                  # old path with the hub off: same return, no hook call
    assert fresh_hub.calls["post_mortem"] == calls


# ================================================================================================================ lessons -> hypotheses
def test_learned_lessons_become_failure_hypotheses(world, fresh_hub):
    X, y, fr = world
    eps = LS.post_mortem(fr, X, fr["resolved"].max(), seed=0)
    fresh_hub.hypotheses.clear()
    book = LS.LessonBook(seed=0)
    book.record(eps)
    new = book.learn()
    assert new, "the planted trap was not learned (fixture drifted)"
    expect = {h.hid for h in FA.hypotheses_from_lessons(new)}
    assert expect and set(fresh_hub.hypotheses) == expect
    assert all(h.source == "lessons.Lesson" and h.subsystem == Subsystem.SELECTION for h in fresh_hub.hypotheses.values())
    assert fresh_hub.calls["lessons"] == 1


def test_learn_by_kind_dict_is_flattened_and_the_same_lesson_twice_is_one_hypothesis(world, fresh_hub):
    X, y, fr = world
    eps = LS.post_mortem(fr, X, fr["resolved"].max(), seed=0)
    book = LS.LessonBook(seed=0)
    book.record(eps)
    by_kind = book.learn_by_kind()
    flat = [l for v in by_kind.values() for l in v]
    fresh_hub.hypotheses.clear()
    ids = W.on_lessons(by_kind)
    assert len(ids) == len({h.hid for h in FA.hypotheses_from_lessons(flat)})
    assert W.on_lessons(flat) == []                                             # already known: nothing new
    assert W.on_lessons([]) == []                                               # the empty case


def test_memory_lessons_teach_only_when_they_were_wrong(fresh_hub):
    m = MEM.Memory()
    ctx = np.zeros(len(MEM.CTX))
    for wk, (arm, out, exp) in enumerate([("arm_a", 0.03, 0.02), ("arm_a", -0.02, 0.02), ("arm_b", 0.03, -0.02), ("arm_b", 0.0, 0.0)]):
        m.record(arm, float(wk), ctx, out, date=f"2021-0{wk + 1}-05", source_experiment="e1", expected=exp)
    df = m.export_lessons()
    errs = sorted(df["error_type"])
    assert errs.count("false_positive") == 1 and errs.count("false_negative") == 1
    assert {h.source for h in fresh_hub.hypotheses.values()} == {"memory.Lesson"}
    assert len(fresh_hub.hypotheses) == 2                                        # correct / noise episodes taught nothing
    assert all("arm_" in h.target for h in fresh_hub.hypotheses.values())


# ================================================================================================================ missed winners
DET = MWB.DET_FEATS


def missed_week(seed, n=80):
    rng = np.random.default_rng(seed)
    idx = pd.Index([f"S{i:03d}" for i in range(n)], name="ticker")
    p0 = pd.DataFrame({c: rng.normal(size=n) for c in DET}, index=idx)
    z = (p0["e_max20"].rank(pct=True) - 0.5)
    fwd = pd.Series(0.01 + 0.12 * z + rng.normal(0, 0.05, n), index=idx)
    return p0, fwd


def test_missed_ledger_observe_feeds_both_ledgers_and_keeps_the_old_row(fresh_hub):
    led = MWB.MissedLedger()
    total_missed = 0
    for i in range(6):
        p0, fwd = missed_week(i)
        picked = list(fwd.sort_values().index[:10])                               # deliberately the WORST ten: every winner is missed
        decided = pd.Timestamp("2021-01-04") + pd.Timedelta(weeks=i)
        led.observe(decided, decided + pd.Timedelta(days=7), p0, fwd, picked, score=-fwd.rank(), era="e1")
        total_missed += int((fwd >= MWB.WINNER).sum())
    assert len(led.rows) == 6 and led.rows[0]["missed"] == led.rows[0]["winners"]     # old ledger: identical to the by-hand computation
    assert led.rows[0]["date"] == str(pd.Timestamp("2021-01-11"))
    assert len(fresh_hub.missed.rows) == total_missed and total_missed > 0
    assert fresh_hub.delivered["rejections"] == total_missed
    tab = fresh_hub.missed.reason_table()
    assert tab["n"].sum() == total_missed and set(tab["reason"])                   # every miss got a reason, named or UNKNOWN
    assert 0.0 <= fresh_hub.missed.unknown_share() <= 1.0


def test_observe_skips_weeks_without_winners_and_survives_bad_input(fresh_hub):
    led = MWB.MissedLedger()
    p0, fwd = missed_week(1)
    led.observe("2021-01-04", "2021-01-11", p0, fwd * 0.0, list(fwd.index[:5]))     # no winner at all
    assert led.rows == [] and len(fresh_hub.missed.rows) == 0
    with pytest.warns(UserWarning):
        led.observe("2021-01-11", "2021-01-04", p0, fwd, list(fwd.index[:5]))       # resolved BEFORE decided: the week is invalid
    assert len(led.rows) == 1 and fresh_hub.errors and fresh_hub.errors[0].hook == "missed_week"


# ================================================================================================================ experiments
@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setattr(IMP, "REG", tmp_path / "experiments.jsonl")
    monkeypatch.setattr(IMP, "CHAL", tmp_path / "challengers.json")
    return tmp_path


def test_log_experiment_writes_a_repro_record_and_mirrors_into_the_ledger(registry, fresh_hub):
    IMP.log_experiment({"event": "unit", "note": "x"}, cfg={"w_model": 0.3}, seed=7, **full_fields())
    rec = json.loads((registry / "experiments.jsonl").read_text().splitlines()[0])
    r = ReproRecord.from_dict(rec["repro"])
    assert r.experiment_id == rec["experiment_id"] and r.code_hash == rec["code_hash"] and r.seed == 7
    assert r.validate() == [] and r.memory_hash == W.UNRECORDED                      # complete, and honest that no memory snapshot was taken
    assert r.run_key == rec["repro"]["run_key"] and r.config_hash == rec["config_hash"]
    led = fresh_hub.ledger_for(registry / "experiments.jsonl")
    assert len(led) == 1 and fresh_hub.delivered["ledger_rows"] == 1
    on_disk = EM.ExperimentLedger(registry / "experiment_ledger.jsonl")               # persisted, not just in memory
    assert len(on_disk) == 1
    row = list(on_disk.view(soon()).values())[0]
    assert dict(row.experiment.config) == {"w_model": 0.3} and row.experiment.code_hash == rec["code_hash"]


def test_log_experiment_still_writes_when_the_learning_sinks_fail(registry, fresh_hub, monkeypatch):
    monkeypatch.setattr(W.EM, "import_legacy", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ledger disk full")))
    with pytest.warns(UserWarning, match="experiment failed"):
        IMP.log_experiment({"event": "unit"}, cfg={"a": 1}, seed=1, **full_fields())
    assert len((registry / "experiments.jsonl").read_text().splitlines()) == 1
    assert [e.hook for e in fresh_hub.errors] == ["experiment"]


def test_an_already_rejected_configuration_is_not_launched_again_until_the_code_changes(registry, fresh_hub):
    cfg = {"w_model": 0.3}
    assert W.blocks_launch(W.pre_launch("lean toward the evidence prior", cfg, soon(), IMP.REG)) is False    # empty ledger: novel
    IMP.log_experiment({"event": "rejected", "id": "CH001", "desc": "lean toward the evidence prior"}, cfg=cfg, seed=7, **full_fields())
    v = W.pre_launch("lean toward the evidence prior", cfg, soon(), IMP.REG)
    assert v.status == EM.DuplicateStatus.SAME_CONFIG_REPEAT and W.blocks_launch(v)
    v2 = W.pre_launch("lean toward the evidence prior", cfg, soon(), IMP.REG, code_hash="a-different-code-hash")
    assert v2.status == EM.DuplicateStatus.RETEST_JUSTIFIED and not W.blocks_launch(v2)
    far = W.pre_launch("something else entirely", {"stop_atr": 4.0, "bank": 0.1, "brake": 0.5}, soon(), IMP.REG)
    assert far.status == EM.DuplicateStatus.NOVEL and not W.blocks_launch(far)


def test_spawn_challengers_skips_a_rejected_idea_and_relaunches_when_the_hub_is_off(registry, fresh_hub):
    findings = {"f1": {"status": "finding", "tag": "MODEL_UNDERPERFORMS_PRIOR", "what": "model", "last_detail": "d"}}
    meta = {"w_model": 0.5, "evidence_weights": {}}
    C = IMP.spawn_challengers(findings, meta)
    assert len(C) == 1 and C[0]["change"] == {"w_model": 0.3}
    ch = C[0]
    ch["status"] = "rejected"
    IMP.log_experiment({"event": "rejected", "id": ch["id"], "desc": ch["desc"]}, cfg=ch["change"], seed=7, **full_fields())
    IMP._w(IMP.CHAL, C)
    C2 = IMP.spawn_challengers(findings, meta)
    assert len(C2) == 1, "the same rejected configuration was launched again"
    fresh_hub.enabled = False                                                        # control: without the hub the old behaviour returns
    C3 = IMP.spawn_challengers(findings, meta)
    assert len(C3) == 2


def test_experiment_memory_entries_land_in_the_same_facade(registry, fresh_hub):
    mem = REG.ExperimentMemory(registry / "experiment_memory.jsonl")
    answers = {q: "yes" for q in REG.QUESTIONS}
    answers.update(adopted=False, if_rejected_why="no gain", what_changed="more weight on the prior")
    mem.record("E1", {"w_model": 0.3}, answers, "2026-01-01T00:00:00")
    led = fresh_hub.ledger_for(registry / "experiment_memory.jsonl")
    assert len(led) == 1 and next(iter(led.view("2026-06-01").values())).experiment_id == "E1:memory"
    assert mem.already_tried({"w_model": 0.3})["blocked"] is True                    # the old answer is unchanged


def test_registry_audit_adds_provenance_findings_without_changing_ok(registry, fresh_hub):
    good = {f: "x" for f in REG.REQUIRED}
    good.update(seed=7, metrics={"a": 1}, gates={"g": True}, window_ids=["w"], model_params={"k": 1}, outcome="adopt")
    r = REG.Registry(registry / "experiments.jsonl")
    assert r.audit()["learning_provenance"]["n_records"] == 0                              # empty registry: nothing to audit, nothing invented
    r.append({**good, "experiment_id": "E1"})
    r.append({"experiment_id": "E2", "outcome": "reject"})                              # planted: almost every required field missing
    a = r.audit()
    lp = a["learning_provenance"]
    assert lp["by_check"].get("record-incomplete", 0) >= 1 and lp["by_check"].get("registry-incomplete") == 1
    assert lp["n_records"] == 2 and lp["repro_records"] == 0
    assert a["ok"] is False and "missing_fields" in a                                    # the old verdict and fields are all still there


def test_registry_audit_counts_records_that_carry_a_valid_repro(registry, fresh_hub):
    IMP.log_experiment({"event": "unit"}, cfg={"a": 1}, seed=3, **full_fields())
    lp = REG.Registry(IMP.REG).audit()["learning_provenance"]
    assert lp["repro_records"] == 1 and lp["n_records"] == 1
    assert W.repro_from_registry({"repro": {"experiment_id": "x"}}) is None and W.repro_from_registry({}) is None


# ================================================================================================================ miner redundancy -> graph
def dup_panel(weeks=90, stocks=60, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2012-01-02", periods=weeks * 5)[::5]
    idx = pd.MultiIndex.from_product([dates, [f"T{i:03d}" for i in range(stocks)]], names=["date", "ticker"])
    X = pd.DataFrame(rng.standard_normal((len(idx), 5)), index=idx, columns=[f"f{i}" for i in range(5)])
    X["f1"] = X["f0"] * 1.0                                                            # planted duplicate feature
    top = X["f0"].groupby(level=0).rank(pct=True).values >= 0.8
    y = pd.Series(rng.normal(0, 0.04, len(idx)) + 0.04 * top, index=idx)
    return X, y - y.groupby(level=0).transform("mean")


def test_pruned_duplicate_patterns_become_redundant_with_edges(fresh_hub):
    X, y = dup_panel()
    m = PatternMiner({"max_pairs": 60, "max_unless": 10, "null_reps": 1, "min_n": 100, "half_life_years": 50}).fit(X, y, X.index.get_level_values(0).max())
    P = m.patterns
    dups = P[P["status"] == "duplicate"]
    assert len(dups) > 0, "the planted duplicate feature produced no duplicate pattern (fixture drifted)"
    assert fresh_hub.calls["redundancy"] == 1 and not fresh_hub.errors
    now = X.index.get_level_values(0).max() + pd.Timedelta(days=1)
    edges = fresh_hub.graph.edges(now, Edge.REDUNDANT_WITH)
    assert len(edges) == len(dups) == fresh_hub.delivered["redundancy_edges"]
    pairs = {frozenset((str(r["key_named"]), str(r["duplicate_of"]))) for _, r in dups.iterrows()}
    assert {frozenset((e.src, e.dst)) for e in edges} == pairs                           # exactly the miner's own duplicate_of relation
    assert all(0.8 < e.weight <= 1.0 for e in edges)                                     # weight = the overlap that condemned it


def test_redundancy_sink_with_no_duplicates_writes_nothing(fresh_hub):
    assert W.on_redundancy(["a", "b"], {}, [np.array([True]), np.array([False])], "2020-01-01") == 0
    assert len(fresh_hub.graph.edges("2030-01-01")) == 0


# ================================================================================================================ board weight
@dataclasses.dataclass(frozen=True)
class Know:
    knowledge_id: str
    version: int = 1
    epistemic: Epistemic = Epistemic.SUPPORTED
    lifecycle: Lifecycle = Lifecycle.ACTIVE
    promotion: Promotion = Promotion.SHADOW
    confidence: Confidence = Confidence(truth=0.9)
    provenance: Provenance = Provenance("2021-07-01T00:00:00", "2021-06-30", "c", "d", "cfg", "e1", "r1", 7, "2021-06-30")
    contexts: dict = dataclasses.field(default_factory=dict)
    anti_contexts: dict = dataclasses.field(default_factory=dict)
    decision_effect: tuple = (DecisionEffect.RANKING,)


def board_with(tmp_path, shadow="lesson-shadow", champion="lesson-champ"):
    b = CH.KnowledgeBoard(tmp_path / "board.jsonl")
    slot = CH.Slot(DecisionEffect.RANKING, None, "global")
    mid = b.register(Know(shadow), slot, dt.date(2021, 1, 1))
    cslot = CH.Slot(DecisionEffect.RANKING, None, "small_cap")
    cm = b.register(Know(champion), cslot, dt.date(2021, 1, 1))
    b.to_shadow(mid, dt.date(2021, 1, 2))
    for day, ev in (("2021-01-03", "to_shadow"), ("2021-01-20", "to_challenger"), ("2021-02-01", "promoted")):
        b._apply(ev, cm, day, {})                                                           # replay of the ledger events that made it champion
    return b


def test_effective_weight_is_legacy_without_a_board_and_for_unregistered_knowledge(tmp_path, fresh_hub):
    assert W.effective_weight("anything", 0.7) == 0.7
    W.configure(board=board_with(tmp_path))
    assert W.effective_weight("never-registered", 0.7) == 0.7                                # adapter: untested paths do not move
    assert W.effective_weight("lesson-champ", 0.7) == 0.7
    assert W.effective_weight("lesson-shadow", 0.7) == 0.0                                  # a shadow can never influence a decision
    W.configure(strict=True)
    assert W.effective_weight("never-registered", 0.7) == 0.0                               # production end state: unknown = no weight
    assert W.effective_weight("lesson-champ", 0.7) == 0.7


def test_a_shadow_lesson_stops_moving_scores_through_the_real_lessonbook(world, tmp_path, fresh_hub):
    X, y, fr = world
    eps = LS.post_mortem(fr, X, fr["resolved"].max(), seed=0)
    book = LS.LessonBook(seed=0)
    book.record(eps)
    book.learn()
    act = [L for L in book.active() if L.action == "reweight" and L.weight() > 0]
    assert act, "no active reweighting lesson (fixture drifted)"
    base = book.factor(X)
    assert (base != 1.0).any()
    b = CH.KnowledgeBoard(tmp_path / "board.jsonl")
    mids = [b.register(Know(L.lid), CH.Slot(DecisionEffect.RANKING, None, f"s{L.lid}"), dt.date(2021, 1, 1)) for L in book.active()]
    for mid in mids:
        b.to_shadow(mid, dt.date(2021, 1, 2))
    W.configure(board=b)
    gated = book.factor(X)
    assert (gated == 1.0).all()                                                             # every lesson is a shadow: none may act
    assert book.advice(X).empty                                                             # advisories are gated by the same weight
    for mm in b.members.values():
        mm.role = Promotion.CHAMPION                                                            # replay: every lesson is now a champion
    pd.testing.assert_series_equal(book.factor(X), base)                                    # champions: the legacy factor returns exactly


# ================================================================================================================ promotion gate
def test_a_non_learning_challenger_is_judged_by_the_old_test_alone(fresh_hub):
    v = W.promotion_allowed({"id": "CH001", "kind": "meta"}, "2026-09-29")
    assert v.allowed and not v.claims_learning and v.checks[0].name == "no_learning_claim"
    assert v.label == ValidationLabel.NOT_VALIDATED and len(fresh_hub.decisions) == 1


def test_a_learning_claim_without_evidence_is_refused_on_all_three_gates(fresh_hub):
    v = W.promotion_allowed({"id": "CH002", "claims_learning": True}, "2026-09-29")
    assert not v.allowed and [c.name for c in v.checks] == ["scorecard", "firewalls", "identity"] and all(not c.ok for c in v.checks)
    with pytest.raises(FirewallBreach, match="refused"):
        W.require_promotion("CH002", "2026-09-29", True, None)


def test_a_learning_claim_passes_only_when_scorecard_firewalls_and_identity_all_pass(fresh_hub):
    X, y = ident_panel(n_dates=60, beta=0.1)
    half = X.index.get_level_values(0).unique()[30]
    tr = X.index.get_level_values(0) < half
    job = W.IdentityJob(lambda Xt, yt, Xe, seed: Xe["f1"].astype(float), X[tr], y[tr], X[~tr], y[~tr],
                        attack_kwargs={"episode_substitution": {"block": 8, "frac": 1.0}})
    good = W.LearningEvidence(good_card(), clean_ctx(), job)
    v = W.promotion_gate("L1", "2020-06-01", True, good)
    assert v.allowed, v.blockers
    assert v.label == ValidationLabel.NOT_VALIDATED                                          # passing the gate is never a validation claim
    # planted defects, one per gate, each must flip the verdict by its own check
    bad_card = dataclasses.replace(good_card(), future_leak_status=__import__("engine.learning.scorecard", fromlist=["LeakStatus"]).LeakStatus.UNAUDITED)
    v1 = W.promotion_gate("L1", "2020-06-01", True, dataclasses.replace(good, card=bad_card))
    assert not v1.allowed and [c.name for c in v1.checks if not c.ok] == ["scorecard"]
    leaky = clean_ctx(items=[{"knowledge_id": "k1", "version": 1, "contexts": {"vol": "high"}, "anti_contexts": {}, "payload": None,
                              "provenance": Provenance("2026-09-29T00:00:00", "2020-07-01", "c1", "d1", "cfg1", "e1", "r1", 7, "2020-07-01")}])
    v2 = W.promotion_gate("L1", "2020-06-01", True, dataclasses.replace(good, gate_ctx=leaky))
    assert not v2.allowed and [c.name for c in v2.checks if not c.ok] == ["firewalls"] and "MEMORY" in v2.checks[1].detail.upper()
    memoriser = W.IdentityJob(lambda Xt, yt, Xe, seed: yt.reindex(Xe.index).fillna(0.0), X, y, X, y,
                              attack_kwargs={"episode_substitution": {"block": 8, "frac": 1.0}})
    v3 = W.promotion_gate("L1", "2020-06-01", True, dataclasses.replace(good, identity=memoriser))
    assert not v3.allowed and [c.name for c in v3.checks if not c.ok] == ["identity"]
    v4 = W.promotion_gate("L1", "2021-01-01", True, good)                                    # evidence gathered for another day
    assert not v4.allowed and "for 2020-06-01" in v4.checks[1].detail


def test_a_check_that_raises_is_a_failed_check_not_an_exception(fresh_hub):
    ev = W.LearningEvidence(card=object(), gate_ctx=clean_ctx(), identity=object())        # wrong types on purpose
    v = W.promotion_gate("bad", "2020-06-01", True, ev)
    assert not v.allowed and all(not c.ok for c in v.checks if c.name in ("scorecard", "identity"))
    assert any("could not run" in c.detail for c in v.checks)


@pytest.fixture
def promotable(registry, monkeypatch):
    diffs = np.array([0.05, 0.06, 0.04, 0.07, 0.05, 0.06, 0.05, 0.04])                         # a strong live-shadow win
    monkeypatch.setattr(IMP, "evaluate", lambda ch, meta, stocks: diffs)
    monkeypatch.setattr(IMP, "n_trials", lambda: 1)
    return diffs


def challenger(cid, **kw):
    return {"id": cid, "kind": "meta", "change": {"w_model": 0.7}, "desc": "lean toward the ML model", "status": "testing",
            "created": "2026-01-01", **kw}


def test_test_and_promote_still_promotes_an_ordinary_challenger(promotable, fresh_hub):
    meta = {"version": "1.0", "w_model": 0.5}
    C = IMP.test_and_promote([challenger("CH001")], meta, None)
    assert C[0]["status"] == "promoted" and meta["w_model"] == 0.7 and meta["version"] == "1.1"
    assert fresh_hub.decisions and fresh_hub.decisions[0].allowed
    assert fresh_hub.delivered["ledger_rows"] >= 1                                             # the promotion itself reached the facade


def test_test_and_promote_holds_back_a_learning_claim_without_evidence(promotable, fresh_hub):
    meta = {"version": "1.0", "w_model": 0.5}
    C = IMP.test_and_promote([challenger("CH009", claims_learning=True)], meta, None)
    assert C[0]["status"] == "testing" and meta == {"version": "1.0", "w_model": 0.5}          # nothing changed
    assert len(C[0]["gate_blocked"]) == 3 and C[0]["live"]["z"] > C[0]["live"]["z_needed"]     # it WON the z-test and was still refused


def test_test_and_promote_promotes_a_learning_claim_once_its_evidence_passes(promotable, fresh_hub):
    X, y = ident_panel(n_dates=60, beta=0.1)
    half = X.index.get_level_values(0).unique()[30]
    tr = X.index.get_level_values(0) < half
    job = W.IdentityJob(lambda Xt, yt, Xe, seed: Xe["f1"].astype(float), X[tr], y[tr], X[~tr], y[~tr],
                        attack_kwargs={"episode_substitution": {"block": 8, "frac": 1.0}})
    W.register_evidence("CH010", W.LearningEvidence(good_card(), clean_ctx(now=dt.date.today().isoformat()), job))
    meta = {"version": "1.0", "w_model": 0.5}
    C = IMP.test_and_promote([challenger("CH010", claims_learning=True)], meta, None)
    assert C[0]["status"] == "promoted"


# ================================================================================================================ the two queued fixes
@pytest.mark.parametrize("text,has", [("AAPL_2008", True), ("since 2008", True), ("x2008y", True), ("vol_2019_high", True),
                                      ("f20080915ab", False), ("rel_vix_q3", False), ("v1.19", False), ("a" * 12 + "2015" + "b", False)])
def test_archive_year_detector_sees_a_year_glued_to_a_name(text, has):
    assert AR._has_year(text) is has


def test_archive_year_detector_ignores_content_hashes_that_contain_year_like_digits():
    h = "ab2015cd9e8f7a6b"                                                                        # 16 hex chars: a hash, not a date
    assert not AR._has_year(h) and AR._has_year(h + " AAPL_2008")


def test_strength_band_is_coarse_era_free_and_capped_for_one_year_memories():
    cfg = CU.RelevanceConfig()
    assert cfg.check() == []
    bands = [CU.strength_band(r, 1.0, 5, cfg) for r in (0.05, 0.25, 0.45, 0.9)]
    assert bands == [0.0, 1 / 3, 2 / 3, 1.0]
    assert CU.strength_band(0.99, 0.99, 1, cfg) == pytest.approx(1 / 3)                        # one year proves little
    assert CU.strength_band(0.9, 0.0, 5, cfg) == 0.0                                          # no cross-year agreement: weak whatever the reliability
    assert dataclasses.replace(cfg, strength_edges=(0.5, 0.4)).check() != []


def _filed(cur, name, rel, lean, year, ctx):
    d = dt.date(year, 3, 1)
    item = TV.TraderMemoryItem.make(TV.opaque_token(name), "pattern", 1.0, {"rel_vix": 0.4}, lean, 5)
    cur.file(item, d, year, ctx, matured_at=d + dt.timedelta(days=7), reliability=rel, calibration_error=0.0,
             temporal=TemporalClass.PERSISTENT)


def test_curator_release_carries_absolute_strength_so_a_lone_weak_memory_is_not_a_confident_one(tmp_path):
    cur = CU.Curator(None, code_hash="testhash", created_real="2026-09-29T00:00:00")
    cur.day(dt.date(2019, 1, 2), {"m_vix": 0.1})
    ctx = {"m_vix": 0.1}
    _filed(cur, "weakk", 0.10, 0.3, 2010, ctx)
    rel = cur.release(dt.date(2019, 1, 2), {"m_vix": 0.1})
    assert len(rel) == 1 and rel.items[0].weight == pytest.approx(1.0)                          # the old design flaw: weight 1.0 ...
    assert json.loads(rel.items[0].features_json)["strength_band"] == 0.0                       # ... but the band says it is weak
    cur2 = CU.Curator(None, code_hash="testhash", created_real="2026-09-29T00:00:00")
    cur2.day(dt.date(2019, 1, 2), {"m_vix": 0.1})
    for y_ in (2010, 2011, 2012, 2013):
        _filed(cur2, "strong", 0.95, 0.3, y_, ctx)
    rel2 = cur2.release(dt.date(2019, 1, 2), {"m_vix": 0.1})
    assert json.loads(rel2.items[0].features_json)["strength_band"] == 1.0
    assert not CU.release_year_hits(rel2) and not CU.release_year_hits(rel)                 # the band does not smuggle in a date


# ================================================================================================================ health of the wiring itself
def test_hook_report_names_the_silent_hooks_and_the_health_check_can_fail(fresh_hub):
    rep = W.hook_report()
    assert set(rep["silent_hooks"]) == set(W.HOOKS)                                             # nothing has fired yet
    with pytest.raises(FirewallBreach, match="never fired"):
        W.assert_hooks_healthy(required=["lessons"])
    W.on_lessons([])
    W.assert_hooks_healthy(required=["lessons"])
    assert "lessons" not in W.hook_report()["silent_hooks"]


def test_persistence_writes_each_hypothesis_once_across_hub_restarts(world, tmp_path, fresh_hub):
    X, y, fr = world
    W.configure(root=tmp_path / "hub")
    now = fr["resolved"].max() + pd.Timedelta(days=1)
    W.on_post_mortem(fr, X, now)
    f = tmp_path / "hub" / "hypotheses.jsonl"
    rows = [json.loads(l) for l in f.read_text().splitlines()]
    assert rows and len({r["key"] for r in rows}) == len(rows) == len(fresh_hub.hypotheses)
    fresh_hub.reset(root=tmp_path / "hub")                                                       # a new process: same root
    W.on_post_mortem(fr, X, now)
    assert len(f.read_text().splitlines()) == len(rows)                                          # nothing duplicated on disk


# ================================================================================================================ history, audit, determinism
def test_the_ledger_backfills_the_whole_legacy_registry_so_old_experiments_are_remembered(registry, fresh_hub):
    cfg = {"stop_atr": 3.0}
    rows = [{"experiment_id": "E-old-1", "t": "2026-01-01T00:00:00", "code_hash": "old", "data_snapshot": "d", "seed": 7, "model_params": cfg,
             "outcome": "reject", "reason": "stop too wide", "event": "rejected", "desc": "loosen the stop rule", "metrics": {"z": -1.0}},
            {"experiment_id": "E-old-2", "t": "2026-01-02T00:00:00", "code_hash": "old", "data_snapshot": "d", "seed": 7,
             "model_params": {"bank": 0.5}, "outcome": "adopt", "reason": "helped", "event": "promoted", "desc": "raise the bank exposure"}]
    (registry / "experiments.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json at all\n[1, 2]\n", encoding="utf-8")
    assert len(EM.ExperimentLedger(registry / "experiment_ledger.jsonl")) == 0                     # nothing mirrored yet
    v = W.pre_launch("loosen the stop rule", cfg, soon(), IMP.REG, code_hash="old")               # this process never wrote E-old-1
    assert v.status == EM.DuplicateStatus.SAME_CONFIG_REPEAT and W.blocks_launch(v)
    assert v.matches and v.matches[0].experiment_id == "E-old-1"
    assert fresh_hub.delivered["ledger_rows"] == 2 and fresh_hub.delivered["backfill_bad_lines"] == 2      # two bad lines counted, not fatal
    assert len(EM.ExperimentLedger(registry / "experiment_ledger.jsonl")) == 2                    # and persisted
    W.pre_launch("again", cfg, soon(), IMP.REG)
    assert fresh_hub.delivered["ledger_rows"] == 2                                                 # backfill runs once per process


def test_backfill_of_a_missing_registry_is_empty_not_an_error(registry, fresh_hub):
    v = W.pre_launch("anything", {"a": 1}, soon(), IMP.REG)
    assert v is not None and v.status == EM.DuplicateStatus.NOVEL and not fresh_hub.errors
    assert W.read_jsonl(registry / "does-not-exist.jsonl") == ([], 0)


def test_audit_hub_is_clean_after_real_traffic_and_flags_planted_corruption(world, registry, fresh_hub):
    X, y, fr = world
    now = fr["resolved"].max() + pd.Timedelta(days=1)
    LS.post_mortem(fr, X, now, seed=0)
    IMP.log_experiment({"event": "unit"}, cfg={"a": 1}, seed=1, **full_fields())
    assert W.audit_hub(now) == []
    h = next(iter(fresh_hub.hypotheses.values()))
    fresh_hub.hypotheses["planted"] = dataclasses.replace(h, hid="planted", statement="downweight when vix spiked in 2008")
    fresh_hub.hypotheses["planted2"] = dataclasses.replace(h, hid="planted2", production_effect=True)
    fresh_hub._classified.add("phantom")
    issues = W.audit_hub(now)
    assert any("planted:" in i and "date" in i for i in issues)
    assert any("planted2" in i and "production effect" in i for i in issues)
    assert any("failure ledger holds" in i for i in issues)


def test_sink_state_digest_is_deterministic_and_sees_a_change(world, fresh_hub):
    X, y, fr = world
    now = fr["resolved"].max() + pd.Timedelta(days=1)
    W.on_post_mortem(fr, X, now)
    d1 = W.state_digest()
    fresh_hub.reset()
    W.on_post_mortem(fr, X, now)
    assert W.state_digest() == d1                                                                # same inputs, same fingerprint
    fresh_hub.reset()
    W.on_post_mortem(fr, X, fr["resolved"].sort_values().iloc[len(fr) // 4])                     # earlier `now`: fewer losses known
    assert W.state_digest() != d1


def test_lessons_p_value_adapter_matches_the_old_normal_cdf_form_and_keeps_the_nonfinite_rule():
    from statistics import NormalDist
    n = NormalDist()
    for t in np.linspace(-12, 12, 481):
        assert abs(LS._p_two_sided(t) - 2.0 * (1.0 - n.cdf(abs(t)))) < 1e-14
    assert LS._p_two_sided(float("nan")) == 1.0 and LS._p_two_sided(float("inf")) == 1.0



# ================================================================================================================ S20 contradiction period loop
def _plant_contradiction(hub, a="patA", b="patB", known="2021-03-01"):
    from engine.learning.knowledge_graph import NodeType
    for n in (a, b):
        hub.graph.add_node(n, NodeType.PATTERN, "2021-01-01", n)
    hub.graph.add_edge(a, b, Edge.CONTRADICTS, known, weight=0.5, evidence=("e1",))


def test_period_run_tracks_a_planted_contradiction_and_orders_periods(fresh_hub):
    _plant_contradiction(fresh_hub)
    with pytest.warns(UserWarning, match="contradiction_period"):
        assert W.on_period("2021-03-01") is None                                     # the edge is known AT this day: a live run refuses it
    fresh_hub.monitor = None
    fresh_hub.errors.clear()
    fresh_hub._warned.clear()
    assert W.on_period("2021-03-05") is not None and fresh_hub.calls["contradiction_period"] == 2
    rep = W.on_period("2021-03-10")
    assert [t.pair for t in rep.items] == [("patA", "patB")] and fresh_hub.delivered["contradictions_tracked"] == 2   # tracked in both periods
    with pytest.warns(UserWarning, match="contradiction_period"):
        assert W.on_period("2021-03-10") is None                                     # not after the previous period: refused, recorded
    assert fresh_hub.errors[-1].error_type == "FirewallBreach"


def test_period_run_on_an_empty_graph_is_empty_not_an_error(fresh_hub):
    rep = W.on_period("2021-03-10")
    assert rep.items == () and not fresh_hub.errors


def test_contradiction_signals_reach_the_research_priority_engine(fresh_hub, tmp_path):
    from engine.learning import research_priority as RP
    _plant_contradiction(fresh_hub)
    rep = W.on_period("2021-06-01")                                                   # 3 months uninvestigated: stale and escalated
    led = EM.ExperimentLedger(tmp_path / "led.jsonl")
    eng = RP.ResearchPriorityEngine()
    step = W.research_step(rep, eng, led, RP.ComputeBudget(cpu_minutes=120, ram_gb_free=8), "2021-06-01", seed=1)
    assert fresh_hub.delivered["contradiction_signals"] == 1
    blob = json.dumps(dataclasses.asdict(step), default=str)
    assert "patA" in blob and "CONTRADICTION" in blob.upper()                          # the question about the pair is in the plan
    with pytest.raises(FirewallBreach, match="no monitor"):
        fresh_hub.reset()
        W.research_step(rep, eng, led, RP.ComputeBudget(cpu_minutes=120, ram_gb_free=8), "2021-06-01", seed=1)


def test_monitor_rows_reach_the_health_dashboard_as_contradicted(fresh_hub, tmp_path):
    from engine.learning import reports as RPT
    _plant_contradiction(fresh_hub)
    _plant_contradiction(fresh_hub, "patC", "patD", "2021-03-02")
    rep = W.on_period("2021-06-01")
    paths = W.write_dashboard_inputs(rep, tmp_path / "art")
    assert fresh_hub.delivered["dashboard_rows"] == 4 and all(p.exists() for p in paths.values())
    ctx = RPT.ReportContext(tmp_path / "art", "2021-06-02", paths=paths, code_hash="x")
    dash = RPT.health_dashboard(ctx)
    assert dash["status"] in ("MEASURED", "PARTIAL")
    got = {r["knowledge_id"]: r["health"] for k in ("trusted", "losing_trust", "other") for r in dash.get(k, [])}
    assert set(got) == {"patA", "patB", "patC", "patD"} and set(got.values()) == {"CONTRADICTED"}
    empty = W.write_dashboard_inputs(type(rep)(rep.now, (), 0, rep.graph_head), tmp_path / "e")
    assert RPT.health_dashboard(RPT.ReportContext(tmp_path / "e", "2021-06-02", paths=empty, code_hash="x"))["status"] == "UNMEASURED"
