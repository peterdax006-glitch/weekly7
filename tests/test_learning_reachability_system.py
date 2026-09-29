"""S21b system integration (contract C62 sections 55, 60, 83, 85; canon C62, C63, C65): each hook that the 29 Sep integration audit
found dead or inert now moves data on the path production really runs, and scripts/reachability.py tells a reached hook from one
that only exists as text. Behavioural, not textual: every test drives the real host (adaptive.replay, improve.log_experiment,
LessonBook.adjust, PromotionGate.evaluate) and reads what the sinks persisted. Synthetic data only; planted defects must be caught."""
import dataclasses
import datetime as dt
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine import adaptive as A
from engine import improve as IMP
from engine import lessons as LS
from engine.learning import champion as CH
from engine.learning import promotion as PR
from engine.learning import wiring as W
from engine.learning.core import (Confidence, DecisionEffect, Epistemic, FirewallBreach, Lifecycle, Promotion, Provenance)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import reachability as R  # noqa: E402

CFG = {"k": 3, "exit_q": 0.8, "rebalance_weeks": 1, "brake": None, "max_per_sector": None, "w_model": 1.0, "pick": "top",
       "pool_q": 0.5, "liq_q": 0.0, "vol_filter": False, "stress_thr": None, "stress_k": 2, "trend_filter": None, "trend_gross": 0.0}


@pytest.fixture
def hub(tmp_path):
    W.HUB.reset()
    W.configure_production("unit", root=tmp_path / "hub")
    yield W.HUB
    W.HUB.reset()


def world(n=90, nt=30, seed=1):
    """A volatile synthetic market: weekly moves of +/-7% are common, so every closed week has winners the adapter did not pick."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-06", periods=n)
    tk = [f"T{i:02d}" for i in range(nt)]
    closes = pd.DataFrame(20 * np.exp(np.cumsum(rng.normal(0, 0.035, (n, nt)), axis=0)), index=idx, columns=tk)
    opens = closes.shift(1).fillna(closes.iloc[0])
    snaps = {}
    for d in closes.index:
        c = closes.loc[:d]
        mu = c.iloc[-5:].pct_change().sum().values if len(c) > 5 else np.zeros(nt)
        snaps[str(d.date())] = pd.DataFrame({"mu_raw": mu, "evidence": 0.5, "vol20": 0.03, "max20": 0.06, "log_dv": 18.0,
                                             "ev_red_flag": 0.0, "ev_offering": 0.0, "r5": 0.0, "m_vix": 0.5, "m_vix_term": 0.9,
                                             "m_spy_ma200": 1.05}, index=pd.Index(tk, name="ticker"))
    return closes, opens, snaps


def run(enabled=True):
    closes, opens, snaps = world()
    W.HUB.enabled = enabled
    S = A.replay(CFG, snaps, closes, 5.0, {}, adaptive=True, opens=opens)
    res = S.result()
    return S, res


def rows(hub, name):
    p = hub.root / f"{name}.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


# ============================================================================================ 1 missed winners, 6 lessons, 5 sinks
def test_the_test_loops_closed_weeks_reach_the_learning_ledgers_and_persist(hub):
    S, res = run()
    assert not hub.errors, hub.errors[:2]
    ad = S.adapter
    assert hub.calls["missed_week"] == len(ad.ledger.rows) > 0            # every closed week with a winner went through observe()
    assert len(hub.missed.rows) > 0 and len(hub.missed.base.rows) == len(ad.ledger.rows)
    mw = rows(hub, "missed_winners")
    assert len(mw) == len(hub.missed.rows) and {r["body"]["reason"] for r in mw} <= {r["reason"] for r in hub.missed.rows}
    blob = (hub.root / "missed_winners.jsonl").read_text(encoding="utf-8")
    assert not any(f'"T{i:02d}"' in blob for i in range(30))              # the ticker never leaves the trusted side
    fails = rows(hub, "failures")
    assert hub.calls["post_mortem"] > 0 and len(fails) == len(hub.failure_ledger) > 0
    assert all(f["body"]["resolved_at"] > f["body"]["decided_at"] for f in fails)
    assert hub.calls["session_lessons"] == 1 and rows(hub, "hypotheses")
    assert hub.lane == "unit" and hub.root.name == "unit"


def test_the_sinks_never_change_a_decision_or_the_old_ledger(hub):
    S1, r1 = run(enabled=True)
    S2, r2 = run(enabled=False)
    assert r1["audit_digest"] == r2["audit_digest"] and S1.decisions == S2.decisions
    same = lambda x: json.dumps(x, sort_keys=True, default=str)             # NaN skill (no scored week yet) compares as text
    assert same(r1["missed_winners"]) == same(r2["missed_winners"])         # observe() keeps exactly the row add() used to write


def test_repeated_results_export_lessons_once_and_persistence_is_append_only_across_restarts(hub, tmp_path):
    S, _ = run()
    n_hyp = len(rows(hub, "hypotheses"))
    S.result()
    S.result()
    assert hub.calls["session_lessons"] == 1                                # the memory did not change: nothing re-sent
    W.HUB.reset()                                                           # a new process on the same lane
    W.configure_production("unit", root=tmp_path / "hub")
    run()
    assert len(rows(W.HUB, "hypotheses")) == n_hyp                          # same content -> same keys -> no duplicate lines


def test_a_week_is_post_mortemed_only_strictly_after_its_outcome_matured(hub):
    closes, _, snaps = world(n=30)
    p0 = snaps["2020-01-13"]
    fwd = closes.loc["2020-01-20"] / closes.loc["2020-01-13"] - 1
    ad = A.Adapter(CFG)
    ad._pm_pending.append(W.post_mortem_frame("2020-01-13", "2020-01-20", p0, fwd, ["T00", "T01"], p0["mu_raw"]))
    assert ad.flush_post_mortems("2020-01-20") == 0 and len(ad._pm_pending) == 1    # matured ON now: not yet known
    assert ad.flush_post_mortems("2020-01-21") == 1 and not ad._pm_pending
    frame, X = W.post_mortem_frame("2020-01-13", "2020-01-20", p0.iloc[:0], fwd, [], None)
    assert frame.empty and X.empty                                          # degenerate: an empty snapshot explains nothing


def test_a_broken_sink_is_recorded_and_the_adapter_keeps_trading(hub, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("planted sink failure")
    monkeypatch.setattr(W, "week_of", boom)
    S, res = run()
    assert any(e.hook == "missed_week" and "planted" in e.message for e in hub.errors)
    assert res["audit_len"] > 0 and S.adapter.ledger.rows                  # the old path continued


def test_the_vectorised_week_equals_the_learning_modules_own_builder():
    rng = np.random.default_rng(4)
    tk = [f"N{i}" for i in range(60)]
    from engine.missed_winners import DET_FEATS, winner_type, winner_types
    p0 = pd.DataFrame(rng.normal(size=(60, len(DET_FEATS))), index=tk, columns=DET_FEATS)
    p0.iloc[3, 0] = np.nan
    p0.loc["N5", "vol20"] = np.inf                                          # planted: an infinite input must not become a type
    p0["ear"] = rng.normal(size=60)
    fwd = pd.Series(rng.normal(0, 0.06, 60), index=tk)
    fwd.iloc[7] = np.nan
    score = pd.Series(rng.normal(size=60), index=tk).drop("N9")
    a = W.week_of("2021-03-01", p0, fwd, ["N1", "N2"], score=score, k=4, resolved_at="2021-03-08", era="e")
    b = W.MW.week_from_base("2021-03-01", p0, fwd, ["N1", "N2"], score=score, k=4, resolved_at="2021-03-08", era="e")
    from engine.learning.core import canonical_json
    assert len(a.candidates) == 60 and canonical_json(a) == canonical_json(b)     # NaN features compare as text, not by identity
    assert list(winner_types(p0)) == [winner_type(p0, t) for t in tk]
    assert W.week_of("2021-03-01", p0.iloc[:0], fwd, []).candidates == ()


# ============================================================================================ 2 registry memory in production
@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setattr(IMP, "REG", tmp_path / "experiments.jsonl")
    return tmp_path


def test_a_decided_experiment_becomes_a_phase30_memory_version_of_the_same_record(registry, hub):
    fields = dict(window_ids=["w1"], train_range="a", validation_range="b", test_range="sealed", metrics={"m": 1.0}, gates={"g": True})
    IMP.log_experiment({"event": "basis", "desc": "raise k"}, cfg={"k": 3}, seed=1, outcome="reject", reason="lost to the champion", **fields)
    IMP.log_experiment({"event": "round"}, cfg={"k": 3}, seed=1, outcome="continue_testing", reason="measurement", **fields)
    led = hub.ledger_for(registry / "experiments.jsonl")
    assert len(led) == 2 and hub.calls["memory_entry"] == 1               # a measurement is not a memory entry
    rec = json.loads((registry / "experiments.jsonl").read_text().splitlines()[0])
    hist = led.history(rec["experiment_id"])
    assert [h.version for h in hist] == [1, 2] and "phase30" in hist[-1].tags
    assert any("rejected: raise k - lost to the champion" == x for x in hist[-1].learned)
    assert "data_used=['w1']" in hist[-1].tags and "learned" not in hist[-1].legacy_missing
    assert W.on_memory_entry(rec["experiment_id"], {}, {"adopted": False}, rec["t"], registry / "experiments.jsonl",
                             version_existing=True) == "skipped"             # idempotent: one memory version per experiment


def test_memory_answers_carry_only_what_the_record_says():
    a = W.memory_answers({"outcome": "adopt", "desc": "x", "reason": "beat it", "window_ids": ["w"]})
    assert a == {"adopted": True, "what_changed": "x", "why_changed": "beat it", "data_used": ["w"]}
    with pytest.raises(ValueError):
        W.memory_answers({"outcome": "continue_testing"})


# ============================================================================================ 3 board + 4 decision-source audit
@dataclasses.dataclass(frozen=True)
class Know:
    knowledge_id: str
    version: int = 1
    epistemic: Epistemic = Epistemic.SUPPORTED
    lifecycle: Lifecycle = Lifecycle.ACTIVE
    promotion: Promotion = Promotion.CHALLENGER
    confidence: Confidence = Confidence(truth=0.9)
    provenance: Provenance = Provenance("2021-07-01T00:00:00", "2021-06-30", "c", "d", "cfg", "e1", "r1", 7, "2021-06-30")
    contexts: dict = dataclasses.field(default_factory=dict)
    anti_contexts: dict = dataclasses.field(default_factory=dict)
    decision_effect: tuple = (DecisionEffect.RANKING,)


def board(tmp_path):
    b = CH.KnowledgeBoard(tmp_path / "hub" / "board.jsonl")
    shadow = b.register(dataclasses.replace(Know("L-shadow"), promotion=Promotion.RESEARCH), CH.Slot(DecisionEffect.RANKING, None, "global"),
                        dt.date(2021, 1, 1))
    champ = b.register(dataclasses.replace(Know("L-champ"), promotion=Promotion.RESEARCH), CH.Slot(DecisionEffect.RANKING, None, "small"),
                       dt.date(2021, 1, 1))
    b.to_shadow(shadow, dt.date(2021, 1, 2))
    b.to_shadow(champ, dt.date(2021, 1, 2))
    b._emit("to_challenger", champ, dt.date(2021, 1, 3))
    b._emit("promoted", champ, dt.date(2021, 1, 4), reason="unit")
    return shadow, champ


def test_production_config_makes_the_board_scale_weights_and_stays_neutral_without_one(tmp_path):
    W.HUB.reset()
    try:
        assert W.effective_weight("L-shadow", 0.6) == 0.6                   # no board: legacy weight, untouched
        shadow, champ = board(tmp_path)                                     # the ledger the production board replays
        W.configure_production("unit", root=tmp_path / "hub")
        assert W.effective_weight("L-shadow", 0.6) == 0.0 and W.effective_weight("L-champ", 0.6) == 0.6
        assert W.effective_weight("unregistered", 0.6) == 0.6              # neutral for knowledge the board never saw
        assert W.champion_of(DecisionEffect.RANKING, None, "small") == champ
        assert W.champion_of(DecisionEffect.RANKING, None, "large") is None
        with pytest.raises(ValueError):
            W.configure_production("../escape", root=tmp_path / "hub")
    finally:
        W.HUB.reset()


def test_the_source_audit_catches_a_planted_shadow_in_a_decision_run(tmp_path):
    W.HUB.reset()
    try:
        shadow, champ = board(tmp_path)
        W.configure_production("unit", root=tmp_path / "hub")
        W.effective_weight("L-champ", 0.5, run="r1")
        assert W.end_decision_run("r1") == []                               # a champion may decide
        W.note_use("r2", [shadow])                                          # planted: a shadow leaked into a decision
        bad = W.end_decision_run("r2")
        assert len(bad) == 1 and bad[0]["mid"] == shadow and "SHADOW" in bad[0]["problem"]
        assert rows(W.HUB, "decision_source_violations") and W.HUB.source_violations == bad
        W.configure(strict=True)
        W.note_use("r3", ["never-registered"])
        with pytest.raises(FirewallBreach):
            W.end_decision_run("r3")
    finally:
        W.HUB.reset()
    assert W.end_decision_run("empty") == []                               # no board: nothing could have been used


def test_lessonbook_adjust_is_an_audited_decision_run():
    W.HUB.reset()
    book = LS.LessonBook(seed=0)
    X = pd.DataFrame({"f0": np.arange(5.0)}, index=pd.Index(list("abcde")))
    out = book.adjust(pd.Series(1.0, index=X.index), X)
    assert (out == 1.0).all() and W.HUB.calls["decision_sources"] == 1 and book._runs == 1
    W.HUB.reset()


# ============================================================================================ 4 promotion of learned knowledge
def evidence():
    return PR.PromotionEvidence()


def test_every_learned_promotion_registers_its_evidence_and_runs_the_learning_checks(hub):
    k = Know("K-learned")
    d = PR.PromotionGate(code_hash="c").evaluate(k, evidence(), dt.date(2022, 1, 3))
    assert "learning_claim" not in {r.gate for r in d.results}               # record mode: the ten-gate decision is unchanged
    v = hub.knowledge_verdicts["K-learned"]
    assert v.claims_learning and not v.allowed and [c.name for c in v.checks] == ["scorecard", "firewalls", "identity"]
    assert "K-learned" in hub.evidence and hub.evidence["K-learned"].gate_ctx is not None
    assert any(r["body"]["subject"] == "K-learned" for r in rows(hub, "promotion_decisions"))


def test_enforce_mode_blocks_a_learning_claim_with_no_scorecard(hub):
    d = PR.PromotionGate(code_hash="c", learning_claim="enforce").evaluate(Know("K-enf"), evidence(), dt.date(2022, 1, 3))
    lc = {r.gate: r for r in d.results}["learning_claim"]
    assert lc.status == PR.FAIL and lc.critical and "scorecard" in lc.detail and not d.promote
    assert "learning_claim" in PR.render_rejection_report(d)
    assert PR.failure_statistics([d])["by_gate"]["learning_claim"]["fail"] == 1
    d0 = PR.PromotionGate(code_hash="c", learning_claim="off").evaluate(Know("K-off"), evidence(), dt.date(2022, 1, 3))
    assert "K-off" not in hub.knowledge_verdicts and d0.results[-1].gate == "provenance_completeness"
    with pytest.raises(ValueError):
        PR.PromotionGate(code_hash="c", learning_claim="maybe")


def test_a_crashing_learning_gate_fails_closed_only_when_enforced(hub, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("planted")
    monkeypatch.setattr(W, "on_knowledge_promotion", boom)
    d = PR.PromotionGate(code_hash="c", learning_claim="enforce").evaluate(Know("K-x"), evidence(), dt.date(2022, 1, 3))
    assert {r.gate: r for r in d.results}["learning_claim"].status == PR.FAIL
    d2 = PR.PromotionGate(code_hash="c").evaluate(Know("K-y"), evidence(), dt.date(2022, 1, 3))
    assert "learning_claim" not in {r.gate for r in d2.results} and any(e.hook == "knowledge_promotion" for e in hub.errors)


def test_scorecards_persist_only_under_a_configured_root():
    W.HUB.reset()
    assert W.store_scorecard(object()) is None                              # no root: nothing is written (and nothing is read)
    with pytest.raises(ValueError):
        W.register_scorecard("k", invalid_card())


def invalid_card():
    from engine.learning.scorecard import LearningScorecard
    return LearningScorecard(learner_version="", now=dt.date(2022, 1, 3), code_hash="", seed=None)


# ============================================================================================ 7 reachability checker
def plant(root: Path, files: dict[str, str]) -> Path:
    for rel, src in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(src), encoding="utf-8")
    return root


FAKE = {
    "engine/__init__.py": "",
    "engine/learning/__init__.py": "",
    "engine/learning/wiring.py": """
        HOOKS = {"good": Hook("on_good", ("engine/host.py",), "x"), "dead": Hook("on_dead", ("engine/host.py",), "x"),
                 "live": Hook("on_live", ("engine/live.py",), "x"), "res": Hook("on_res", ("engine/host.py",), "x"),
                 "cb": Hook("on_cb", ("engine/plugin.py",), "x")}
        def Hook(*a):
            return a
        def on_good(): return 1
        def on_dead(): return 2
        def on_live(): return 3
        def on_res(): return 4
        def on_cb(): return 5
        """,
    "engine/learning/used.py": "def f():\n    return 1\n",
    "engine/learning/imported_only.py": "def g():\n    return 2\n",
    "engine/learning/dead.py": "def h():\n    return 3\n",
    "engine/host.py": """
        from .learning import wiring
        from .learning import imported_only
        class Host:
            def run(self):
                wiring.on_good()
                from .learning.used import f
                return f()
            def never(self):
                wiring.on_dead()          # the text is here, but nothing calls never()
        def research():
            wiring.on_res()
        if __name__ == "__main__":
            wiring.on_dead()              # runs only when host.py itself is the entry: not from prod
        """,
    "engine/plugin.py": """
        from .learning import wiring
        class Plugin:
            def on_tick(self):
                wiring.on_cb()
        def factory():
            return Plugin()
        """,
    "engine/runner.py": """
        def drive(make):
            p = make()
            p.on_tick()                   # dynamic dispatch on an object built elsewhere
        """,
    "engine/live.py": "from .learning import wiring\ndef job():\n    wiring.on_live()\nif __name__ == '__main__':\n    job()\n",
    "scripts/prod.py": """
        import engine.host as H
        from engine import runner, plugin
        def main():
            H.Host().run()
            runner.drive(plugin.factory)
        if __name__ == "__main__":
            main()
        """,
    "scripts/res.py": "from engine import host\nhost.research()\n",
}
TABLE = (R.HookSpec("T.used", "S99", "engine.learning.used:f"), R.HookSpec("T.missing", "S99", "engine.learning.used:nope"))


def test_reachability_tells_a_called_hook_from_one_that_only_exists_as_text(tmp_path):
    root = plant(tmp_path, FAKE)
    ck = R.Checker(root, production={"scripts/prod.py": "unit"}, hooks=R.wiring_hooks(root) + TABLE)
    hv = {v.name: v for v in ck.hook_verdicts()}
    assert hv["wiring.good"].status == "REACHED"
    assert hv["wiring.dead"].status == "UNREACHED"                           # planted: text present, never executed
    assert hv["wiring.res"].status == "RESEARCH-ONLY" and "scripts/res.py" in hv["wiring.res"].reason
    assert hv["wiring.live"].status == "RESEARCH-ONLY" and "C65" in hv["wiring.live"].reason
    assert hv["wiring.cb"].status == "REACHED"                               # a callback object handed across modules is followed
    assert hv["T.used"].status == "REACHED" and hv["T.missing"].status == "UNREACHED" and "not defined" in hv["T.missing"].reason
    mv = {v.name: v for v in ck.module_verdicts()}
    assert mv["engine.learning.used"].status == "REACHED" and mv["engine.learning.dead"].status == "UNREACHED"
    assert mv["engine.learning.imported_only"].status == "UNREACHED" and mv["engine.learning.imported_only"].reason == "imported, never called"
    rep = ck.report()
    bad = R.failing(rep, "all")
    assert "hook wiring.dead" in bad and "module engine.learning.dead" in bad and "hook wiring.good" not in bad
    assert R.failing(rep, "none") == [] and R.main(["--root", str(root), "--fail-on", "none"]) == 0


def test_reachability_on_an_empty_tree_fails_closed_on_the_missing_entry(tmp_path):
    ck = R.Checker(tmp_path, production={"scripts/prod.py": "unit"}, hooks=())
    rep = ck.report()
    assert rep["n_files"] == 0 and rep["module_counts"]["REACHED"] == 0
    assert R.failing(rep, "all") == ["production entry missing: scripts/prod.py"]


def test_the_real_tree_reaches_the_hooks_this_wave_connected():
    rep = R.Checker(ROOT).report()
    hv = {v["name"]: v["status"] for v in rep["hooks"]}
    for h in ("wiring.missed_week", "wiring.adapter_post_mortem", "wiring.session_lessons", "wiring.memory_entry",
              "wiring.knowledge_promotion", "wiring.lessons", "S05.missed_week", "S05.post_mortem"):
        assert hv[h] == "REACHED", (h, hv[h])
    assert hv["wiring.production_config"] == "RESEARCH-ONLY"                 # improve.weekly runs only from the parked live path
    assert rep["unmapped_integration_ids"] == []
