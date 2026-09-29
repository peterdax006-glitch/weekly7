"""Tests for engine/learning/firewalls.py, memory_firewall.py and reproducibility.py (contract C62 sections 28, 55, 56;
section 62 tests 7, 11, 12). Synthetic data only; every defect kind is planted and must be caught, and the clean case must
pass (a check that cannot fail is worthless)."""
import dataclasses
import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from engine.learning import firewalls as F
from engine.learning import memory_firewall as M
from engine.learning import reproducibility as R
from engine.learning.core import FirewallBreach, Provenance, stable_hash

NOW = "2020-06-01"


# ---------------------------------------------------------------- fixtures
def prov(learned="2019-12-31", seen=None, code="c1", data="d1", exp="e1", parents=(), sealed=(), seed=7, config="cfg1"):
    return Provenance(created_real="2026-09-29T00:00:00", learned_at=learned, code_hash=code, data_hash=data, config_hash=config,
                      experiment_id=exp, run_id="r1", seed=seed, outcomes_seen_through=seen or learned, sealed_windows=tuple(sealed),
                      parents=tuple(parents))


def item(kid="k1", version=1, contexts=None, payload=None, **pk):
    return {"knowledge_id": kid, "version": version, "provenance": prov(**pk), "contexts": contexts or {"vol": "high"},
            "anti_contexts": {}, "payload": payload}


def panel(n_dates=40, n_names=25, seed=0, beta=0.15):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-04", periods=n_dates * 5)[::5][:n_dates]
    tickers = [f"S{i:03d}" for i in range(n_names)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    X = pd.DataFrame({"f1": rng.normal(size=len(idx)), "f2": rng.normal(size=len(idx))}, index=idx)
    y = pd.Series(beta * 0.02 * X["f1"].to_numpy() + rng.normal(0, 0.03, len(idx)), index=idx, name="y")
    return X, y


def experiment(**kw):
    base = dict(experiment_id="e1", registered_real="2026-09-01T00:00:00", first_result_real="2026-09-02T00:00:00",
                config_hash="cfg1", seed=7, hypothesis="h", n_variants_tried=1, selected_from=1, baseline_declared=True,
                training_windows=(("2015-01-01", "2016-12-31"),), tuned_windows=(("2015-01-01", "2016-12-31"),))
    base.update(kw)
    return F.ExperimentRecord(**base)


def evaluation(**kw):
    base = dict(evaluation_windows=(("2019-01-01", "2019-12-31"),), training_windows=(("2015-01-01", "2016-12-31"),),
                sealed_real="2026-08-01T00:00:00", training_started_real="2026-08-15T00:00:00", state_hash_before="s1",
                state_hash_after="s1", n_decisions=200, disguised=True, paired_baseline=True)
    base.update(kw)
    return F.EvaluationRecord(**base)


def code_state(**kw):
    rec = {"code_hash": "c1", "code_files": ["engine/x.py"], "code_mixed": []}
    return F.CodeState(recorded=kw.pop("recorded", rec), current_hash=kw.pop("current_hash", "c1"), **kw)


class Ident:
    """Stand-in identity report (the real one is exercised in test_learning_identity_firewall.py)."""
    def __init__(self, status="OK"):
        self.verdicts = [type("V", (), {"kind": "ticker_permutation", "mode": "eval", "status": status, "retention": 0.95})()]


def clean_ctx(**over):
    X, y = panel()
    kw = dict(now=NOW, subject="unit", items=[item()], X=X, y=y, horizon=5, identity_report=Ident(), experiment=experiment(),
              evaluation=evaluation(), code=code_state(), registry_records=[registry_record()])
    kw.update(over)
    return F.GateContext(**kw)


def registry_record(**kw):
    from engine.registry import REQUIRED
    rec = {f: "x" for f in REQUIRED}
    rec.update({"seed": 7, "metrics": {"a": 1}, "gates": {"g": True}, "window_ids": ["w1"], "model_params": {"k": 1}, "outcome": "adopt"})
    rec.update(kw)
    return rec


GATE = F.LearningFirewallGate()


# ---------------------------------------------------------------- the gate
def test_clean_context_passes_every_layer_but_is_never_called_validated():
    v = GATE.evaluate(clean_ctx())
    assert v.passed, v.markdown()
    assert set(v.verdicts) == set(F.LayerName)
    assert str(v.label) == "IMPLEMENTED — NOT VALIDATED"
    v.require()


def test_layers_are_independent_a_failure_in_one_does_not_hide_another():
    X, y = panel()
    ctx = clean_ctx(items=[item(learned="2020-07-01")], code=code_state(current_hash="OTHER"), experiment=experiment(selected_from=5, n_variants_tried=5))
    v = GATE.evaluate(ctx)
    assert not v.passed
    assert {F.LayerName.MEMORY, F.LayerName.CODE_VERSION, F.LayerName.EXPERIMENT} <= set(v.failed_layers)
    with pytest.raises(FirewallBreach):
        v.require()
    assert v.verdicts[F.LayerName.DATA].passed and v.verdicts[F.LayerName.TIME].passed


def test_missing_inputs_fail_closed_but_an_empty_memory_is_a_definite_pass():
    v = GATE.evaluate(F.GateContext(now=NOW))
    assert not v.passed
    assert all(f.check == "input-missing" for lv in v.verdicts.values() for f in lv.failures)
    ok = GATE.evaluate(clean_ctx(items=[]))
    assert ok.verdicts[F.LayerName.MEMORY].passed
    assert any(f.check == "empty-memory" for f in ok.verdicts[F.LayerName.MEMORY].findings)


def test_no_relevant_layer_is_a_failure_not_a_pass():
    v = GATE.evaluate(clean_ctx(relevant=frozenset()))
    assert not v.passed
    with pytest.raises(FirewallBreach):
        v.require()


def test_layer_exception_is_an_error_status_that_rejects():
    class Broken(F.FirewallLayer):
        name = F.LayerName.DATA

        def inspect(self, ctx):
            raise RuntimeError("boom")
    v = F.LearningFirewallGate([Broken()]).evaluate(clean_ctx())
    assert v.verdicts[F.LayerName.DATA].status == F.LayerStatus.ERROR and not v.passed


def test_duplicate_layers_and_missing_now_are_refused():
    with pytest.raises(ValueError):
        F.LearningFirewallGate([F.DataFirewall(), F.DataFirewall()])
    with pytest.raises(FirewallBreach):
        GATE.evaluate(F.GateContext(now=None))


def test_planted_leak_probe_every_mutation_rejected_by_its_layer():
    def future_rows(c):
        X = pd.concat([c.X, c.X.iloc[:3].rename(index={c.X.index[0][0]: pd.Timestamp("2021-01-04")}, level=0)])
        return dataclasses.replace(c, X=X, y=None, horizon=None, label_close=None)

    muts = {
        "future_memory": lambda c: dataclasses.replace(c, items=[item(learned="2020-08-01")]),
        "stale_code": lambda c: dataclasses.replace(c, code=code_state(current_hash="zzz")),
        "tuned_on_eval": lambda c: dataclasses.replace(c, experiment=experiment(tuned_windows=(("2019-03-01", "2019-04-01"),))),
        "identity_collapse": lambda c: dataclasses.replace(c, identity_report=Ident("COLLAPSE")),
        "label_feature": lambda c: dataclasses.replace(c, X=c.X.assign(fwd_ret=c.y)),
        "state_changed": lambda c: dataclasses.replace(c, evaluation=evaluation(state_hash_after="s2")),
        "future_rows": future_rows,
    }
    out = F.planted_leak_probe(GATE, clean_ctx(), muts)
    assert out["clean_passed"]
    for name in muts:
        assert out[name]["rejected"], name
    assert "MEMORY" in out["future_memory"]["layers"] and "CODE_VERSION" in out["stale_code"]["layers"]
    assert "EXPERIMENT" in out["tuned_on_eval"]["layers"] and "EVALUATION" in out["state_changed"]["layers"]
    assert "DATA" in out["label_feature"]["layers"] and "TIME" in out["future_rows"]["layers"]
    F.assert_gate_can_fail(GATE, clean_ctx(), muts["future_memory"](clean_ctx()))
    with pytest.raises(FirewallBreach):
        F.assert_gate_can_fail(GATE, clean_ctx(), clean_ctx())


def test_data_layer_catches_label_copy_duplicates_and_inf():
    X, y = panel()
    leaky = X.assign(copy=y * 3.0)
    v = GATE.evaluate(clean_ctx(X=leaky))
    assert any(f.check == "label-copy" for f in v.findings(F.Severity.FAIL))
    dup = pd.concat([X, X.iloc[:2]])
    assert any(f.check == "duplicate-rows" for f in GATE.evaluate(clean_ctx(X=dup, y=None, horizon=None)).findings(F.Severity.FAIL))
    inf = X.copy()
    inf.iloc[0, 0] = np.inf
    assert any(f.check == "infinite-values" for f in GATE.evaluate(clean_ctx(X=inf)).findings(F.Severity.FAIL))
    assert any(f.check == "empty-panel" for f in GATE.evaluate(clean_ctx(X=X.iloc[:0], y=None)).findings(F.Severity.FAIL))


def test_time_layer_label_not_closed_purge_overlap_and_fill_timing():
    X, y = panel()
    last = X.index.get_level_values(0).max()
    v = GATE.evaluate(clean_ctx(now=str(last.date()), horizon=5))
    assert any(f.check == "label-not-closed" for f in v.findings(F.Severity.FAIL))
    dates = pd.DatetimeIndex(sorted(set(X.index.get_level_values(0))))
    test_start = dates[30]
    v2 = GATE.evaluate(clean_ctx(now="2021-01-04", horizon=10, test_start=test_start))
    assert any(f.check == "unpurged-overlap" for f in v2.findings(F.Severity.FAIL))
    v3 = GATE.evaluate(clean_ctx(now="2021-01-04", horizon=10, test_start=test_start, embargo_days=0,
                                 X=X[X.index.get_level_values(0) < dates[20]], y=None))
    assert not any(f.check == "unpurged-overlap" for f in v3.findings())
    dec = pd.DataFrame({"decision_date": ["2020-01-03", "2020-01-10"], "fill_date": ["2020-01-03", "2020-01-13"]})
    v4 = GATE.evaluate(clean_ctx(decisions=dec))
    assert [f.check for f in v4.findings(F.Severity.FAIL) if f.layer == F.LayerName.TIME] == ["fill-not-next-session"]


def test_experiment_layer_preregistration_multiplicity_and_seed():
    bad = experiment(registered_real="2026-09-05T00:00:00", first_result_real="2026-09-02T00:00:00", thresholds_changed_after_results=True,
                     selected_from=8, n_variants_tried=8, seed=None, config_hash="")
    checks = {f.check for f in GATE.evaluate(clean_ctx(experiment=bad)).findings(F.Severity.FAIL)}
    assert {"registered-after-results", "post-hoc-thresholds", "best-of-n", "seed-missing", "config-unhashed"} <= checks
    corrected = experiment(selected_from=8, n_variants_tried=8, multiplicity_correction="holm")
    assert GATE.evaluate(clean_ctx(experiment=corrected)).verdicts[F.LayerName.EXPERIMENT].passed


def test_evaluation_layer_seal_overlap_labels_rerun_and_decisions():
    bad = evaluation(sealed_real="2026-09-01T00:00:00", training_started_real="2026-08-15T00:00:00",
                     training_windows=(("2018-06-01", "2019-03-01"),), labels_visible_before_decision=True, learner_told_rerun=True,
                     best_of_runs_reported=True, n_decisions=5)
    checks = {f.check for f in GATE.evaluate(clean_ctx(evaluation=bad)).findings(F.Severity.FAIL)}
    assert {"sealed-after-training", "train-eval-overlap", "labels-visible", "rerun-identified", "best-run-reported", "too-few-decisions"} <= checks
    unhashed = evaluation(state_hash_before="", state_hash_after="")
    assert any(f.check == "state-hash-missing" for f in GATE.evaluate(clean_ctx(evaluation=unhashed)).findings(F.Severity.FAIL))


def test_window_use_ledger_burns_a_window_after_repeated_use():
    led = F.WindowUseLedger(max_uses=2)
    w = ("2019-01-01", "2019-12-31")
    assert led.record(w, "a") == 1 and led.record(w, "a") == 1 and led.record(w, "b") == 2
    assert led.burned(("2019-06-01", "2020-06-01"))
    assert not led.burned(("2021-01-01", "2021-12-31"))
    assert [f.check for f in led.findings([w])] == ["window-burned"]
    assert led.findings([("2021-01-01", "2021-12-31")]) == []


def test_identity_layer_statuses():
    for status, ok in (("OK", True), ("COLLAPSE", False), ("NONDETERMINISTIC", False), ("NO_SKILL", False), ("INSUFFICIENT", False), ("??", False)):
        assert GATE.evaluate(clean_ctx(identity_report=Ident(status))).verdicts[F.LayerName.IDENTITY].passed == ok
    lenient = F.LearningFirewallGate([F.IdentityFirewall(strict_inconclusive=False)])
    assert lenient.evaluate(clean_ctx(identity_report=Ident("NO_SKILL"))).passed


def test_promotion_decision_uses_the_layers_the_kind_requires():
    ctx = clean_ctx(identity_report=None)
    assert not F.decide_promotion(GATE, ctx, "policy").admitted
    assert F.decide_promotion(GATE, ctx, "pattern").admitted
    assert F.relevant_layers_for("no-such-kind") == frozenset(F.LayerName)
    d = F.decide_promotion(GATE, clean_ctx(items=[item(learned="2020-08-01")]), "pattern")
    assert not d.admitted and any("MEMORY" in r for r in d.reasons)


def test_ledger_is_hash_chained_and_detects_edits(tmp_path):
    led = F.GateLedger(tmp_path / "gate.jsonl")
    v1, v2 = GATE.evaluate(clean_ctx(subject="a")), GATE.evaluate(clean_ctx(subject="b", items=[item(learned="2020-09-01")]))
    led.append(v1)
    led.append(v2)
    assert led.verify() == [] and [h["subject"] for h in led.history()] == ["a", "b"] and len(led.history("b")) == 1
    p = tmp_path / "gate.jsonl"
    p.write_bytes(p.read_bytes().replace(b'"subject": "a"', b'"subject": "z"', 1))
    assert led.verify()
    back = F.verdict_from_dict(led.history("b")[0])
    assert back.digest() == v2.digest() and not back.passed


def test_diff_explain_summaries_and_catalog():
    a, b = GATE.evaluate(clean_ctx()), GATE.evaluate(clean_ctx(items=[item(learned="2020-09-01")]))
    d = F.diff_verdicts(a, b)
    assert d["newly_failing"] and not d["fixed"] and d["status_changes"]["MEMORY"] == ("PASS", "FAIL")
    assert "not a validation claim" in F.explain(a) and "MEMORY" in F.explain(b)
    tab = F.summarize_layers([a, b])
    assert tab.loc["MEMORY", "rejected"] == 1 and tab.loc["DATA", "rejected"] == 0
    res = F.evaluate_many(GATE, [clean_ctx(), clean_ctx(items=[item(learned="2020-09-01")])])
    assert res["admitted"] == 1 and res["admission_rate"] == 0.5
    cov = F.coverage_matrix([b])
    assert ((cov["layer"] == "MEMORY") & (cov["FAIL"] > 0)).any()
    assert F.uncatalogued_findings([b]) == []
    assert any(c.check == "learned-after-now" for c in F.unexercised_checks([a]))
    assert not any(c.check == "learned-after-now" for c in F.unexercised_checks([b]))
    assert F.summarize_layers([]).empty and F.coverage_matrix([]).empty


def test_gate_filter_items_splits_admitted_and_rejected_without_deleting():
    items = [item("ok"), item("late", learned="2020-09-01"), item("nohash", code="")]
    good, bad = GATE.filter_items(clean_ctx(items=items))
    assert [i["knowledge_id"] for i in good] == ["ok"] and {i["knowledge_id"] for i in bad} == {"late", "nohash"}
    assert len(items) == 3


# ---------------------------------------------------------------- provenance audit (the failing registry audit)
def test_provenance_registry_audit_rejects_incomplete_records_as_in_the_real_audit():
    """state/research/registry_audit.json: 2 of 53 records complete. The provenance layer must reject such a registry."""
    recs = [registry_record(experiment_id=f"e{i}") for i in range(2)]
    recs += [{"experiment_id": f"old{i}", "t": "2026-01-01", "metrics": {"a": 1}} for i in range(51)]
    v = GATE.evaluate(clean_ctx(registry_records=recs))
    fails = {f.check for f in v.verdicts[F.LayerName.PROVENANCE].failures}
    assert {"record-incomplete", "registry-incomplete"} <= fails and not v.passed
    reg = [f for f in F.registry_findings(recs) if f.check == "registry-incomplete"][0]
    assert reg.evidence["complete"] == 2 and reg.evidence["total"] == 53
    assert F.registry_findings([registry_record()]) == []
    assert F.registry_findings([])[0].check == "registry-empty" and F.registry_findings(None)[0].check == "registry-missing"
    dup = [registry_record(experiment_id="same"), registry_record(experiment_id="same")]
    assert any(f.check == "duplicate-experiment-id" for f in F.registry_findings(dup))


def test_provenance_layer_item_holes():
    holes = [item("a", code="", data="", exp=""), item("b", parents=("ghost",)), item("c", seed=None)]
    v = GATE.evaluate(clean_ctx(items=holes))
    checks = {(f.subject, f.check) for f in v.verdicts[F.LayerName.PROVENANCE].findings}
    assert ("a", "hash-missing") in checks and ("b", "parent-unresolved") in checks and ("c", "seed-missing") in checks
    clash = [item("k", version=1, learned="2019-01-01"), item("k", version=1, learned="2019-02-01")]
    assert any(f.check == "version-conflict" for f in GATE.evaluate(clean_ctx(items=clash)).findings(F.Severity.FAIL))


def test_code_version_layer_stale_mixed_legacy_and_unregistered():
    for st, check in ((code_state(current_hash="new"), "code-changed"),
                      (code_state(recorded={"code_hash": "c1", "code_files": ["e.py"], "code_mixed": ["e.py"]}), "code-edited-mid-run"),
                      (code_state(recorded="deadbeef"), "legacy-code-hash"),
                      (code_state(known_hashes={"other": "2026-01-01T00:00:00"}), "unregistered-code"),
                      (code_state(known_hashes={"c1": "2026-12-01T00:00:00"}), "code-postdates-record")):
        v = GATE.evaluate(clean_ctx(code=st))
        assert check in {f.check for f in v.verdicts[F.LayerName.CODE_VERSION].failures}, check
    assert any(f.check == "older-code" for f in GATE.evaluate(clean_ctx(items=[item(code="ancient")])).findings())


# ---------------------------------------------------------------- memory firewall (section 28; tests 7 and 11)
@pytest.mark.parametrize("kind,kw,reason", [
    ("learned after now", dict(learned="2020-06-02"), "learned-after-now"),
    ("learned ON now", dict(learned="2020-06-01"), "learned-after-now"),
    ("saw future outcomes", dict(learned="2020-01-01", seen="2020-07-01"), "saw-future-outcomes"),
    ("no code", dict(code=""), "code_hash-missing"),
    ("no data", dict(data=""), "data_hash-missing"),
    ("no experiment", dict(exp=""), "experiment_id-missing"),
])
def test_memory_firewall_rejects_each_planted_contamination(kind, kw, reason):
    ex = M.could_exist_at(item(**kw), NOW)
    assert not ex.could_exist and reason in ex.reasons, kind
    assert M.could_exist_at(item(), NOW).could_exist


def test_memory_item_without_provenance_or_id_is_rejected():
    ex = M.could_exist_at({"knowledge_id": "x", "version": 1}, NOW)
    assert not ex.could_exist and ex.reasons == ("no-provenance",)
    with pytest.raises(FirewallBreach):
        M.view({"version": 1})
    rep = M.audit_store([{"version": 1}, item()], NOW)
    assert rep.rejected == ("?",) and rep.admissible == ("k1",)


def test_lineage_taint_missing_parent_cycle_and_child_before_parent():
    parent = item("p", learned="2020-05-01", seen="2020-08-01")          # saw outcomes after now
    child = item("c", learned="2020-01-01", parents=("p",))
    rep = M.audit_store([parent, child], NOW)
    assert set(rep.rejected) == {"p", "c"} and "tainted-by-parent" in rep.existences[1].reasons
    assert "parent-missing" in M.could_exist_at(item("o", parents=("nobody",)), NOW).reasons
    a, b = item("a", parents=("b",)), item("b", parents=("a",))
    assert "lineage-cycle" in M.audit_store([a, b], NOW).existences[0].reasons
    early_child = item("kid", learned="2019-01-01", parents=("dad",))
    late_dad = item("dad", learned="2019-06-01")
    assert "child-predates-parent" in M.audit_store([early_child, late_dad], NOW).existences[0].reasons
    ok = M.audit_store([item("p", learned="2019-01-01"), item("c", learned="2019-06-01", parents=("p",))], NOW)
    assert not ok.rejected
    anc = M.ancestry("c", {"c": M.view(item("c", learned="2019-06-01", parents=("p",))), "p": M.view(item("p", learned="2019-01-01"))})
    assert anc.depth == 1 and str(anc.effective_seen) == "2019-06-01"


def test_sealed_window_evaluation_experiment_and_eval_window_rules():
    inside = item("i", learned="2019-06-30", seen="2019-06-30")
    assert "saw-sealed-window" in M.could_exist_at(inside, NOW, sealed_windows=[("2019-06-01", "2019-08-01")]).reasons
    own = item("o", sealed=[("2019-06-01", "2019-08-01")], learned="2019-07-01")
    assert "saw-sealed-window" in M.could_exist_at(own, NOW).reasons
    env = M.MemoryEnvironment(evaluation_experiments=frozenset({"evalrun"}), eval_window=("2019-01-01", "2019-12-31"))
    assert "learned-from-evaluation" in M.could_exist_at(item(exp="evalrun", learned="2018-05-01"), NOW, env=env).reasons
    assert "not-before-eval-window" in M.could_exist_at(item(learned="2019-03-01"), NOW, env=env).reasons
    assert M.could_exist_at(item(learned="2018-05-01"), NOW, env=env).could_exist
    assert "sealed-window-unparseable" in M.could_exist_at(item(), NOW, sealed_windows=["nonsense"]).reasons


def test_data_and_code_registries():
    env = M.MemoryEnvironment(data=M.DataRegistry({"d1": "2020-07-01", "dOK": "2019-12-31", "dBefore": "2020-05-01"}),
                              code=M.CodeRegistry({"c1": "2026-01-01T00:00:00"}))
    assert "data-reaches-now" in M.could_exist_at(item(data="d1"), NOW, env=env).reasons
    assert "data-snapshot-unknown" in M.could_exist_at(item(data="mystery"), NOW, env=env).reasons
    assert M.could_exist_at(item(data="dOK"), NOW, env=env).could_exist
    warn_only = M.could_exist_at(item(data="dBefore", learned="2019-12-31"), NOW, env=env)
    assert warn_only.could_exist and any(f.check == "data-beyond-learned" for f in warn_only.findings)
    assert "code-unregistered" in M.could_exist_at(item(code="nope", data="dOK"), NOW, env=env).reasons
    late = M.MemoryEnvironment(code=M.CodeRegistry({"c1": "2030-01-01T00:00:00"}))
    assert "code-postdates-record" in M.could_exist_at(item(), NOW, env=late).reasons


def test_answer_lookup_tables_identity_contexts_and_hidden_labels_are_rejected():
    table = M.plant_hidden_answer_table(40)
    assert "answer-lookup-table" in M.could_exist_at(item(payload=table), NOW).reasons
    assert "identity-context" in M.could_exist_at(item(contexts={"ticker": "AAPL"}), NOW).reasons
    assert "outcome-labels-stored" in M.could_exist_at(item(payload={"stats": {"forward_return": [0.1, 0.2]}}), NOW).reasons
    honest = {"threshold": 0.7, "weights": [0.2, 0.3], "n": 80}
    assert M.could_exist_at(item(payload=honest, contexts={"vol": "high", "regime": "bull"}), NOW).could_exist
    y = np.random.default_rng(0).normal(0, 0.05, 200)
    disguised = {"blob": y[:80].tolist()}                                   # labels under an innocent name
    assert not M.label_fields(disguised)
    f = M.hidden_label_overlap(disguised, y)
    assert f is not None and f.check == "hidden-label-storage"
    assert M.hidden_label_overlap({"blob": np.random.default_rng(9).normal(size=80).tolist()}, y) is None
    assert M.hidden_label_overlap({"blob": [1.0]}, y) is None


def test_as_of_view_time_travel_and_monotone_admissibility():
    v1 = item("k", version=1, learned="2019-01-01")
    v2 = item("k", version=2, learned="2020-03-01")
    v3 = item("k", version=3, learned="2020-09-01")
    got = M.as_of_view([v1, v2, v3], NOW)
    assert [g["version"] for g in got] == [2]
    assert M.as_of_view([v1, v2, v3], "2019-06-01")[0]["version"] == 1
    assert M.as_of_view([v3], NOW) == []
    assert M.admissibility_monotone([v1, v2, v3], ["2018-01-01", "2019-06-01", NOW, "2021-01-01"]) == []
    tl = M.admissibility_timeline([v1, v3], ["2019-06-01", "2021-01-01"])
    assert tl.loc["2019-06-01"].tolist() == [True, False] and tl.loc["2021-01-01"].tolist() == [True, True]
    assert list(tl.columns) == ["k@1", "k@3"]


def test_retrieval_log_snapshot_and_bank_frames():
    items = [item("a", learned="2019-01-01"), item("b", learned="2020-03-01")]
    log = [{"now": "2019-06-01", "knowledge_id": "a"}, {"now": "2019-06-01", "knowledge_id": "b"}, {"now": NOW, "knowledge_id": "zz"}, {"knowledge_id": "a"}]
    checks = [f.check for f in M.audit_retrieval_log(log, items)]
    assert checks == ["retrieved-before-existence", "retrieved-unknown-item", "retrieval-entry-malformed"]
    snap = M.snapshot_memory(items, "2020-05-01")
    assert M.verify_snapshot(snap, items, NOW) == []
    tampered = [dict(items[0], contexts={"vol": "low"}), items[1]]
    assert [f.check for f in M.verify_snapshot(snap, tampered, NOW)] == ["item-edited-after-snapshot"]
    assert {f.check for f in M.verify_snapshot(snap, items + [item("c")], NOW)} == {"item-added-after-snapshot"}
    assert {f.check for f in M.verify_snapshot(snap, items[:1], NOW)} == {"item-vanished"}
    assert M.verify_snapshot(snap, items, "2020-04-01")[0].check == "snapshot-from-future"
    assert M.snapshot_diff(snap, M.snapshot_memory(tampered + [item("c")], "2020-05-01")) == {
        "added": ["c@1"], "removed": [], "edited": ["a@1"]}
    bank = pd.DataFrame({"learned_at": ["2019-01-01", "2020-07-01"], "real_end": ["2019-01-01", "2020-07-01"]})
    assert {f.check for f in M.audit_bank_frame(bank, NOW)} >= {"bank-future-lessons", "bank-window-end"}
    assert M.audit_bank_frame(bank.iloc[:1], NOW) == [] and M.audit_bank_frame(None, NOW)[0].check == "bank-missing"
    usable, refused = M.causal_bank(bank, NOW)
    assert len(usable) == 1 and len(refused) == 1
    with pytest.raises(FirewallBreach):
        M.causal_bank(bank.drop(columns="real_end"), NOW)


def test_memory_future_invariance_hides_the_future_from_the_builder():
    """Section 62 test 7: hide the pattern from the evaluation period - a builder must not use it."""
    idx = pd.bdate_range("2020-01-01", periods=120)
    s = pd.Series(np.random.default_rng(1).normal(size=120), index=idx, name="r")

    def honest(d):                                  # only sees data up to its own last date
        return [round(float(d.iloc[:60].mean()), 10)]

    def cheat(d):                                   # peeks at the whole series (includes the future)
        return [round(float(d.mean()), 10)]
    now = idx[59]
    assert M.memory_future_invariance(honest, s, now) == []
    bad = M.memory_future_invariance(cheat, s, now)
    assert bad and bad[0].check == "memory-depends-on-future"
    assert M.memory_future_invariance(honest, s.iloc[:60], now)[0].check == "no-future-to-scramble"


def test_firewalled_memory_enforces_on_write_and_read_and_never_deletes():
    mem = M.FirewalledMemory()
    mem.add(item("a", learned="2019-01-01"), NOW)
    with pytest.raises(FirewallBreach):
        mem.add(item("late", learned="2020-09-01"), NOW)
    with pytest.raises(FirewallBreach):
        mem.add(item("a", version=1, learned="2019-02-02"), NOW)                # immutable history
    with pytest.raises(FirewallBreach):
        mem.add(item("orphan", parents=("nope",)), NOW)
    mem.add(item("a", version=2, learned="2019-06-01"), NOW)
    assert [i["version"] for i in mem.retrieve(NOW)] == [2]
    mem.retire("a", "2020-01-01", "stopped working in the 2020 regime")
    assert mem.retrieve(NOW) == [] and len(mem.retrieve(NOW, include_retired=True)) == 1 and len(mem) == 2
    with pytest.raises(FirewallBreach):
        mem.retire("a", NOW, "")
    with pytest.raises(FirewallBreach):
        mem.retire("ghost", NOW, "why")
    assert mem.verify_log() == [] and len(mem.log) >= 2
    assert len(mem.quarantine) == 3
    later = mem.requalify("2021-06-01")                                    # 'late' becomes admissible once now has moved on
    assert [i["knowledge_id"] for i in later] == ["late"] and not mem.audit("2021-06-01").rejected


def test_planted_contamination_suite_and_reporting():
    verdicts = M.planted_contamination_suite(lambda: item(), NOW)
    assert verdicts["clean"].could_exist
    assert all(not verdicts[k].could_exist for k in ("learned_in_future", "saw_future_outcomes", "no_code_hash", "no_experiment"))
    assert not M.could_exist_at(M.plant_future_item(lambda: item(), NOW), NOW).could_exist
    rep = M.audit_store([item("a"), item("b", learned="2020-09-01"), item("c", code="")], NOW)
    assert rep.contamination_rate == pytest.approx(2 / 3) and rep.reasons()["learned-after-now"] == 1
    assert "REJECT b" in M.existence_markdown(rep) and not M.contamination_by_layer(rep).empty
    with pytest.raises(FirewallBreach):
        rep.require_clean()
    assert list(rep.to_frame()["could_exist"]) == [True, False, False]
    prof = M.age_profile([item("a", learned="2020-05-20"), item("b", learned="2015-01-01")], NOW)
    assert prof.sum() == 2
    assert M.provenance_completeness([item("a"), item("b", code="")]).iloc[0]["knowledge_id"] == "b"
    assert not M.sealed_window_exposure([item("a", learned="2019-06-01")], [("2019-01-01", "2019-12-31")]).empty
    assert M.windows_overlapping([item("a", sealed=[("2019-06-01", "2019-08-01")])], ("2019-07-01", "2019-09-01")) == ["a"]
    empty = M.audit_store([], NOW)
    assert empty.contamination_rate == 0.0 and empty.to_frame().empty


# ---------------------------------------------------------------- leak channel 4: learned state from the future
def test_channel4_learned_state_trained_on_future_or_same_windows_is_rejected():
    """Leak channel 4 (state/research/leak_audit): basis cfg/meta trained on later or same windows."""
    W = M.TrainingWindow
    states = [M.LearnedState("v1", "basis_cfg", (W("w1", "1990-01-01", "1990-12-31"), W("w2", "2005-01-01", "2005-12-31"))),
              M.LearnedState("v2", "basis_cfg", (W("w1", "1990-01-01", "1990-12-31"),))]
    played = [{"id": "p1", "real_start": "2000-01-01", "version": "v1"},          # v1 saw 2005: the future
              {"id": "p2", "real_start": "1990-01-01", "version": "v2"},          # v2 saw p2's own window
              {"id": "p3", "real_start": "2000-01-01", "version": "v2"}]          # clean
    found = {(f.subject, f.check) for f in M.audit_state_lineage(states, played)}
    assert ("p1", "state-trained-on-future-window") in found and ("p2", "state-trained-on-same-window") in found
    assert not any(s == "p3" for s, _ in found)
    assert M.audit_state_lineage(states, played[2:]) == []
    assert any(f.check == "state-version-unknown" for f in M.audit_state_lineage(states, [{"id": "q", "real_start": "2000-01-01", "version": "v9"}]))
    tuned = M.LearnedState("d0", "defaults", (), tuned_years=(1999, 2000))
    hit = M.audit_state_lineage([tuned], [{"id": "p", "real_start": "1999-06-01", "version": "d0"}])
    assert any(f.check == "defaults-tuned-on-window" for f in hit)
    assert M.eligible_state(states, "2000-01-01").version == "v2" and M.eligible_state(states, "1980-01-01") is None
    share = M.future_training_share(states, played)
    assert share["share_of_windows_touched"] == pytest.approx(2 / 3)
    ctx = clean_ctx(learned_states=states, played=played)
    v = GATE.evaluate(ctx)
    assert F.LayerName.MEMORY in v.failed_layers
    assert GATE.evaluate(clean_ctx(learned_states=states, played=played[2:])).passed


# ---------------------------------------------------------------- reproducibility (section 56, test 12)
def make_rec(**kw):
    base = dict(cfg={"a": 1}, seed=3, data="d-hash", memory_hash="m1", code_hash="code1", worker=R.WorkerConfig("3.11", "Win", {"numpy": "2"}, 1),
                code_files=("engine/x.py",), created_real="2026-09-29T00:00:00")
    base.update(kw)
    return R.make_record(**base)


def test_repro_identical_records_are_reproducible_and_key_is_deterministic():
    a, b = make_rec(), make_rec()
    assert a.run_key == b.run_key and R.compare(a, b).reproducible and R.compare(a, b).status == "REPRODUCIBLE"
    assert a.experiment_id == b.experiment_id and a.experiment_id.startswith("exp-")
    assert R.ReproRecord.from_dict(json.loads(json.dumps(a.to_dict()))) == a


@pytest.mark.parametrize("change,label", [
    (dict(code_hash="code2"), R.ReproLabel.CODE_CHANGED), (dict(data="other"), R.ReproLabel.DATA_CHANGED),
    (dict(cfg={"a": 2}), R.ReproLabel.CONFIG_CHANGED), (dict(seed=4), R.ReproLabel.SEED_CHANGED),
    (dict(memory_hash="m2"), R.ReproLabel.MEMORY_CHANGED),
    (dict(worker=R.WorkerConfig("3.12", "Win", {"numpy": "2"}, 1)), R.ReproLabel.WORKER_CHANGED),
])
def test_every_component_difference_is_labelled_and_marks_the_result(change, label):
    v = R.compare(make_rec(), make_rec(**change))
    assert label in v.labels and not v.reproducible
    marked = v.mark({"sharpe": 1.2})
    assert marked["repro"]["status"].startswith("NOT_REPRODUCIBLE") and str(label) in marked["repro"]["labels"] and marked["sharpe"] == 1.2
    with pytest.raises(FirewallBreach):
        v.require()


def test_soft_worker_difference_and_experiment_id_mismatch_and_incomplete_record():
    soft = R.compare(make_rec(experiment_id="e"), make_rec(experiment_id="e", worker=R.WorkerConfig("3.11", "Win", {"numpy": "2"}, 4)))
    assert soft.reproducible and R.ReproLabel.WORKER_CHANGED_SOFT in soft.labels and soft.status.startswith("REPRODUCIBLE_WITH_NOTES")
    assert R.ReproLabel.EXPERIMENT_MISMATCH in R.compare(make_rec(experiment_id="a"), make_rec(experiment_id="b")).labels
    inc = R.compare(make_rec(memory_hash=""), make_rec(memory_hash=""))
    assert R.ReproLabel.RECORD_INCOMPLETE in inc.labels and not inc.reproducible
    assert "memory_hash" in make_rec(memory_hash="").missing() and make_rec(seed=None).missing() == ("seed",)
    assert R.diff_records(make_rec(experiment_id="e"), make_rec(experiment_id="e", seed=9, memory_hash="x")) == {"seed": (3, 9), "memory_hash": ("m1", "x")}
    assert "run key" in R.markdown(make_rec())


def test_code_edited_during_a_worker_run_makes_the_result_stale(tmp_path):
    """Section 62 test 12: change code during a worker run; the stale result must not be accepted."""
    src = tmp_path / "learner.py"
    src.write_bytes(b"def f():\r\n    return 1\r\n")
    w = R.CodeWatcher(["learner.py"], root=tmp_path)
    assert w.check() == [] and w.accept({"r": 1})[0]
    src.write_bytes(b"def f():\n    return 1\n")                              # CRLF -> LF is not an edit
    assert w.check() == []
    src.write_bytes(b"def f():\n    return 2\n")
    assert w.check() == ["learner.py"]
    with pytest.raises(FirewallBreach):
        w.assert_fresh("worker result")
    ok, marked = w.accept({"r": 1})
    assert not ok and "CODE_EDITED_MID_RUN" in marked["repro"]["labels"]
    src.unlink()
    assert w.check() == ["learner.py"]
    with pytest.raises(ValueError):
        R.CodeWatcher([])


def test_stale_record_is_refused_and_the_gate_agrees():
    rec = make_rec()
    R.assert_fresh_record(rec, "code1")
    with pytest.raises(FirewallBreach):
        R.assert_fresh_record(rec, "code2")
    with pytest.raises(FirewallBreach):
        R.assert_fresh_record(dataclasses.replace(rec, code_mixed=("engine/x.py",)), "code1")
    with pytest.raises(FirewallBreach):
        R.assert_fresh_record(make_rec(code_files=()))
    assert R.stale_results([rec, make_rec(code_hash="code9")], "code1") == [make_rec(code_hash="code9").experiment_id]
    v = GATE.evaluate(clean_ctx(code=code_state(current_hash="code2", recorded={"code_hash": "code1", "code_files": ["e.py"], "code_mixed": []})))
    assert "code-changed" in {f.check for f in v.verdicts[F.LayerName.CODE_VERSION].failures}


def test_reproduce_reruns_and_names_the_differing_artifact():
    def exp(cfg, seed):
        rng = np.random.default_rng(seed)
        return {"metrics": {"m": float(rng.normal())}, "predictions": rng.normal(size=4)}

    cfg = {"k": 1}
    art = exp(cfg, 5)
    rec = R.make_record(cfg, 5, data="d", memory_hash="m", code_hash="c", worker=R.WorkerConfig("3.11", "W", {}, 1), artifacts=art, code_files=("x.py",))
    rep, _ = R.reproduce(exp, cfg, 5, rec, worker=rec.worker)
    assert rep.reproduced and rep.deterministic
    bad_rec = dataclasses.replace(rec, artifact_hashes={**rec.artifact_hashes, "metrics": "0" * 64})
    rep2, _ = R.reproduce(exp, cfg, 5, bad_rec, check_determinism=False, worker=rec.worker)
    assert not rep2.reproduced and rep2.differing_artifacts == ("metrics",) and R.ReproLabel.RESULT_DIFFERS in rep2.verdict.labels


def test_hidden_input_probe_finds_an_environment_dependence(monkeypatch):
    import os

    def leaky(cfg, seed):
        return {"v": float(os.environ.get("W7_HIDDEN", "0")) + seed}

    def clean(cfg, seed):
        return {"v": float(seed)}
    ps = [R.env_perturbation("W7_HIDDEN", "5"), R.global_rng_perturbation(3)]
    assert R.hidden_input_probe(clean, {}, 1, ps) == {}
    assert list(R.hidden_input_probe(leaky, {}, 1, ps)) == ["env:W7_HIDDEN=5"]
    assert "W7_HIDDEN" not in os.environ
    assert R.config_order_invariant(clean, {"a": 1, "b": 2}, 1)
    assert not R.config_order_invariant(lambda c, s: {"first": list(c)[0]}, {"a": 1, "b": 2}, 1)
    assert R.global_rng_use(lambda c, s: {"x": float(np.random.rand())}, {}, 1)


def test_ledger_manifest_consistency_and_seed_derivation(tmp_path):
    led = R.ReproLedger(tmp_path / "runs.jsonl")
    a = make_rec(artifacts={"m": 1.0})
    led.append(a)
    assert led.find(a.experiment_id) == a and led.find("nope") is None
    with pytest.raises(FirewallBreach):
        led.append(make_rec(memory_hash=""))
    with pytest.raises(FirewallBreach):
        led.append(make_rec(experiment_id=a.experiment_id, seed=99))
    twin = dataclasses.replace(make_rec(experiment_id="twin", artifacts={"m": 2.0}), )
    assert twin.run_key != a.run_key                                          # different experiment id -> different key
    same_key = dataclasses.replace(a, artifact_hashes={"m": "different"}, created_real="later")
    led.append(same_key)
    conf = led.conflicts()
    assert conf and conf[0]["artifacts"] == ["m"]
    seeds = R.derive_worker_seeds(11, 4)
    assert seeds == R.derive_worker_seeds(11, 4) and len(set(seeds)) == 4 and seeds != R.derive_worker_seeds(12, 4)
    ws = [make_rec(seed=s, experiment_id=f"w{i}") for i, s in enumerate(seeds)]
    assert R.check_manifest_consistency(ws, base_seed=11) == []
    probs = R.check_manifest_consistency([ws[0], make_rec(seed=seeds[0], code_hash="other", experiment_id="w9")], base_seed=11)
    assert any("code_hash" in p for p in probs) and any("share a seed" in p for p in probs)
    assert R.check_manifest_consistency([]) and R.label_counts([R.compare(a, a), R.compare(a, make_rec(seed=1))])["SEED_CHANGED"] == 1


def test_memory_snapshot_hash_and_stamp_adapter():
    items = [item("a"), item("b")]
    h1 = R.memory_snapshot_hash(items, "2020-05-01")
    assert h1 == R.memory_snapshot_hash(list(reversed(items)), "2020-05-01")
    assert h1 != R.memory_snapshot_hash(items[:1], "2020-05-01")
    rec = R.record_from_stamp({"code_hash": "c", "config_hash": "cf", "data_snapshot": "ds", "seed": 4, "code_files": ["a.py"]}, "e-1", memory_hash=h1)
    assert rec.missing() == () and rec.seed == 4
    exp = F.experiment_record_from_stamp({"config_hash": "cf", "seed": 4}, "e-1")
    assert exp.config_hash == "cf" and exp.seed == 4


# ---------------------------------------------------------------- second-wave additions: context validation, strict gate, runner, corpus
def test_validate_context_and_strict_gate_and_runner(tmp_path):
    assert F.validate_context(clean_ctx()) == []
    errs = F.validate_context(clean_ctx(now="not-a-date", horizon=0, embargo_days=-1, relevant=frozenset({"DATA"})))
    assert len(errs) == 4
    unaligned = clean_ctx(y=pd.Series([1.0], index=pd.MultiIndex.from_tuples([(pd.Timestamp("2001-01-01"), "ZZZ")], names=["date", "ticker"])))
    assert any("share no index" in e for e in F.validate_context(unaligned))
    strict = F.StrictGate(["no-baseline"])
    ctx = clean_ctx(experiment=experiment(baseline_declared=False))
    assert GATE.evaluate(ctx).passed and not strict.evaluate(ctx).passed
    assert any("[strict]" in f.message for f in strict.evaluate(ctx).findings(F.Severity.FAIL))
    runner = F.GateRunner(ledger=F.GateLedger(tmp_path / "runs.jsonl"), windows=F.WindowUseLedger(max_uses=1))
    first = runner.run(clean_ctx())
    assert first.passed
    second = runner.run(clean_ctx())                                          # same evaluation window judged again
    assert not second.passed and any(f.check == "window-burned" for f in second.findings())
    assert len(runner.ledger.history()) == 2 and runner.ledger.verify() == []
    with pytest.raises(FirewallBreach):
        runner.run(clean_ctx(now="junk"))
    with pytest.raises(FirewallBreach):
        runner.admit(clean_ctx())
    assert F.severity_counts(second)["FAIL"] >= 1


def test_builtin_planted_corpus_is_all_rejected_and_the_reference_passes():
    res = F.run_corpus()
    assert res["clean_passed"] and res["missed"] == [], res
    assert len(res["by_layer"]) == 12 and "MEMORY" in res["by_layer"]["memory_learned_in_future"]
    assert "IDENTITY" in res["by_layer"]["identity_collapse"] and "EVALUATION" in res["by_layer"]["evaluation_state_changed"]
    blind = F.LearningFirewallGate([layer for layer in F.default_layers() if layer.name != F.LayerName.MEMORY])
    assert "memory_learned_in_future" in F.run_corpus(blind)["missed"]                 # a disabled layer is exposed by the corpus
    assert F.gate_from_names(["DATA", "TIME"]).evaluate(F.reference_context()).verdicts.keys() == {F.LayerName.DATA, F.LayerName.TIME}
    with pytest.raises(ValueError):
        F.gate_from_names(["NOPE"])
    assert F.worst_findings(GATE.evaluate(clean_ctx(items=[item(learned="2020-09-01")])), 2)


def test_memory_retirement_lineage_and_merge_second_wave():
    a, b = item("a", learned="2019-01-01"), item("b", learned="2019-02-01", parents=("a",))
    c = item("c", learned="2019-03-01", parents=("b",))
    tomb = [M.Tombstone("a", "2019-12-01", "regime ended"), M.Tombstone("ghost", "2019-12-01", "x"), M.Tombstone("b", "2021-01-01", ""),
            M.Tombstone("c", "2019-12-01", "no longer works")]
    checks = {(f.subject, f.check) for f in M.audit_retirements([a, b, c], tomb, NOW, [{"knowledge_id": "c", "now": "2020-02-01"}])}
    assert {("ghost", "tombstone-without-item"), ("b", "tombstone-without-reason"), ("b", "tombstone-not-in-past"), ("c", "retired-item-retrieved")} <= checks
    assert M.audit_retirements([a], [M.Tombstone("a", "2019-12-01", "why")], NOW) == []
    assert M.descendants_of([a, b, c], ["a"]) == {"b", "c"} and M.descendants_of([a, b, c], ["c"]) == set()
    assert M.lineage_depth([a, b, c]) == {"a": 0, "b": 1, "c": 2}
    bad_root = item("a", code="")
    rep = M.audit_store([bad_root, b, c], NOW)
    assert M.quarantine_closure(rep, [bad_root, b, c]) == {"a", "b", "c"}
    assert M.contamination_score(rep) > 0 and M.contamination_score(M.audit_store([], NOW)) == 0.0
    assert M.contamination_score(M.audit_store([item(learned="2020-09-01")], NOW)) == 1.0
    assert [M.view(i).knowledge_id for i in M.merge_stores([a, b], [b, c])] == ["a", "b", "c"]
    with pytest.raises(FirewallBreach):
        M.merge_stores([a], [item("a", learned="2019-01-01", contexts={"vol": "low"})])
    assert M.validate_policy(M.LENIENT_POLICY) and M.validate_policy(M.MemoryPolicy()) == []
    assert "NO - REJECT" in M.explain_existence(M.could_exist_at(item(learned="2020-09-01"), NOW))
    assert "YES" in M.explain_existence(M.could_exist_at(item(), NOW))
    assert M.could_exist_at(item(data=""), NOW, policy=M.LENIENT_POLICY).could_exist and not M.could_exist_at(item(data=""), NOW).could_exist


def test_repro_manifest_environment_and_sealing(tmp_path):
    recs = [make_rec(seed=s, experiment_id=f"m{i}") for i, s in enumerate(R.derive_worker_seeds(5, 3))]
    man = R.ExperimentManifest("m", tuple(recs), {"metrics": "abc"})
    assert man.problems(base_seed=5) == []
    d = man.save(tmp_path / "man.json")
    assert R.ExperimentManifest.load(tmp_path / "man.json").digest() == d
    p = tmp_path / "man.json"
    p.write_bytes(p.read_bytes().replace(b'"metrics": "abc"', b'"metrics": "abd"'))
    with pytest.raises(FirewallBreach):
        R.ExperimentManifest.load(p)
    other = R.ExperimentManifest("m", (make_rec(experiment_id="zzz"),))
    assert any("different experiment" in x for x in other.problems())
    a = make_rec(experiment_id="a")
    cmp_ = R.compare_many(a, [make_rec(experiment_id="b", seed=9), dataclasses.replace(a, experiment_id="c")])
    assert R.ReproLabel.SEED_CHANGED in cmp_["b"].labels and R.ReproLabel.EXPERIMENT_MISMATCH in cmp_["c"].labels
    assert set(R.by_label(cmp_)["EXPERIMENT_MISMATCH"]) == {"b", "c"}
    older = dataclasses.replace(a, created_real="2026-01-01T00:00:00")
    newer = dataclasses.replace(make_rec(experiment_id="n", worker=R.WorkerConfig("3.12", "Win", {"numpy": "2"}, 1)), created_real="2026-02-01T00:00:00")
    drift = R.environment_drift([newer, older])
    assert len(drift) == 1 and drift[0]["hard"] and "python" in drift[0]["changed"]
    assert "py3.11" in R.worker_summary(older.worker)
    env = tmp_path / "env.json"
    fp = R.save_environment(env, older.worker)
    assert R.load_environment(env) == older.worker and fp == older.worker.fingerprint()
    env.write_bytes(env.read_bytes().replace(b'"python": "3.11"', b'"python": "3.10"'))
    with pytest.raises(FirewallBreach):
        R.load_environment(env)
    assert R.explain_difference("m", {"a": [1, 2]}, {"a": [1, 2]})["same"]
    assert not R.explain_difference("m", np.array([1.0, 2.0]), np.array([1.0, 3.0]))["same"]
    sealed = R.seal_result({"sharpe": 1.5}, a)
    assert R.verify_sealed(sealed) == a
    sealed["sharpe"] = 9.9
    with pytest.raises(FirewallBreach):
        R.verify_sealed(sealed)
    with pytest.raises(FirewallBreach):
        R.verify_sealed({"sharpe": 1.0})


def test_earliest_use_date_and_bitwise_reproducibility():
    a = item("a", learned="2019-01-01", seen="2019-03-01")
    b = item("b", learned="2019-02-01", parents=("a",))
    assert str(M.earliest_use_date(a)) == "2019-03-02"
    assert str(M.earliest_use_date(b, [a, b])) == "2019-03-02"                     # inherits the ancestor's outcomes
    assert M.earliest_use_date(b) is None and M.earliest_use_date({"knowledge_id": "x", "version": 1}) is None
    tab = M.usable_from([a, b, item("c", parents=("nobody",))])
    assert tab["a"] == pd.Timestamp("2019-03-02") and pd.isna(tab["c"])
    assert R.assert_bitwise_reproducible(lambda c, s: {"v": np.random.default_rng(s).normal(size=3)}, {}, 1) is None
    with pytest.raises(FirewallBreach):
        R.assert_bitwise_reproducible(lambda c, s: {"v": np.random.rand(3)}, {}, 1)


def test_require_labelled_results():
    ok = R.compare(make_rec(), make_rec()).mark({"x": 1})
    assert R.require_labelled(ok) == "REPRODUCIBLE"
    assert R.require_labelled(R.compare(make_rec(), make_rec(seed=1)).mark({"x": 1})).startswith("NOT_REPRODUCIBLE")
    with pytest.raises(FirewallBreach):
        R.require_labelled({"x": 1})


def test_hard_labels_and_is_reproduced():
    assert R.is_reproduced(R.compare(make_rec(), make_rec()))
    v = R.compare(make_rec(experiment_id="e"), make_rec(experiment_id="e", seed=2))
    assert R.hard_labels(v) == (R.ReproLabel.SEED_CHANGED,) and not R.is_reproduced(v)
