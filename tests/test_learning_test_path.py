"""Tests for the integration wave S17b: the API gaps S12 found and the Test path of canon C64 (engine/learning/test_path.py).
Synthetic data only; state/livesim is redirected to tmp_path and never read.  Status: IMPLEMENTED - NOT VALIDATED.

Part A - API gaps, each proved on a planted case:  ancestry has ONE canonical parent form and the firewall resolves either;
a typed ContextSet reaches context.py and archive.put_knowledge; the quadratic transfer test runs once per (item, case count) and
still fails closed on a case from the future; similarity defaults do not mark every pair non-comparable when ~25% of fields are seen.
Part B - the blind half:  DayData refuses a panel of another day and labels that have not matured; the memory advisor votes only from
the release; a strongly opposed pick is vetoed only in veto mode; the trader classes reference no curator symbol (AST) and the import
closure of livesim/learner/adaptive never reaches the curator or this module.
Part C - the trusted half:  labels are served only once the exit open is strictly before today; memory is filed under the REAL year,
once, and a feature name that reads as a date/rank is refused at the wall.
Part D - one synthetic window end to end, twice under two disguises of the same real months: every session is ticked, the learner
learns, memory is filed by real year, nothing the trader is shown names the real year, a memory from the future is never released,
and the filed memory is identical under both disguises."""
import ast
import functools
import importlib.util
import inspect
import json
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine import blind_gates as BG, livesim
from engine.learning import archive as AR
from engine.learning import context as CX
from engine.learning import knowledge as KN
from engine.learning import memory_firewall as MF
from engine.learning import retrieval as RV
from engine.learning import similarity as SM
from engine.learning import situation as ST
from engine.learning import test_path as TP
from engine.learning import trader_view as TV
from engine.learning.core import (Confidence, DecisionEffect, Epistemic, FirewallBreach, Lifecycle, Promotion, Provenance,
                                  TemporalClass)
from engine.learning.curator import Curator

ROOT = Path(__file__).resolve().parent.parent
NOW = "2020-06-01"


# =====================================================================================================================
# Part A - the API gaps
# =====================================================================================================================

def prov(learned="2020-01-01", through=None, parents=()):
    return Provenance(created_real="2026-01-01T00:00:00", learned_at=learned, code_hash="c", outcomes_seen_through=through or learned,
                      parents=tuple(parents))


def item(kid, version=1, parents=(), learned="2020-01-01", through=None, contexts=None, anti=None):
    return type("It", (), {"knowledge_id": kid, "version": version, "provenance": prov(learned, through, parents),
                           "contexts": contexts or {}, "anti_contexts": anti or {}, "epistemic": Epistemic.SUPPORTED,
                           "lifecycle": Lifecycle.ACTIVE, "promotion": Promotion.CHAMPION, "confidence": Confidence(),
                           "decision_effect": (DecisionEffect.RANKING,)})()


@functools.lru_cache(maxsize=None)
def _helpers(name):
    spec = importlib.util.spec_from_file_location(f"{name}_helpers", ROOT / "tests" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _base_knowledge(**over):
    return _helpers("test_learning_knowledge").base(**over)


def test_new_version_writes_bare_parents_and_no_self_reference():
    v1 = _base_knowledge(promotion=Promotion.RESEARCH)
    v2 = v1.new_version("2020-06-01", "more evidence", evidence=KN.Evidence(300, 140.0, 0.9, 0.7, "2020-05-30"))
    assert v2.provenance.parents == () and v2.parent_hash == v1.record_hash()          # the predecessor is the hash link, not an ancestor
    child = _base_knowledge(knowledge_id="K-child", promotion=Promotion.RESEARCH,
                            provenance=prov(parents=("K-a@v1", "K-a", "K-b@v3", "K-child@v1")))
    revised = child.new_version("2020-06-02", "again", evidence=KN.Evidence(310, 141.0, 0.9, 0.7, "2020-05-31"))
    assert revised.provenance.parents == ("K-a", "K-b")                                # bare, deduplicated, self dropped


POLICY = MF.MemoryPolicy(require_data_hash=False, require_experiment_id=False)


def exists(it, pool):
    return MF.could_exist_at(it, NOW, MF.views_as_of(pool, NOW), policy=POLICY)


def test_firewall_resolves_ancestry_by_either_parent_form():
    root = item("K-root", learned="2020-01-01")
    for form in ("K-root", "K-root@v1"):
        child = item("K-child", learned="2020-02-01", parents=(form,))
        assert MF.view(child).parents == ("K-root",)
        ex = exists(child, [root, child])
        assert ex.could_exist, ex.reasons
    revised = item("K-root", version=2, learned="2020-03-01", parents=("K-root@v1",))     # a new version listing its predecessor
    assert MF.view(revised).parents == ()
    assert exists(revised, [root, revised]).could_exist


def test_a_missing_or_tainting_ancestor_is_still_caught_in_either_form():
    orphan = item("K-orphan", parents=("K-ghost@v2",))
    ex = exists(orphan, [orphan])
    assert not ex.could_exist and "parent-missing" in ex.reasons
    future_root = item("K-root", learned="2020-01-01", through="2020-09-01")            # saw outcomes after NOW
    for form in ("K-root", "K-root@v1"):
        child = item("K-child", learned="2020-02-01", through="2020-02-01", parents=(form,))
        ex = exists(child, [future_root, child])
        assert not ex.could_exist and "tainted-by-parent" in ex.reasons


def test_typed_contextset_reaches_context_and_archive(tmp_path):
    cs = KN.ContextSet((KN.Condition("regime", "regime.label", "in", labels=("stress", "correction_calm")),
                        KN.Condition("volatility", "volatility.vol_rank", "ge", nums=(0.5,))))
    anti = KN.ContextSet((KN.Condition("market", "regime.label", "eq", labels=("bull_calm",)),), any_of=True)
    spec = CX.ContextSpec.from_typed(cs)
    assert spec.to_mapping() == {"regime.label": {"in": ["correction_calm", "stress"]}, "volatility.vol_rank": {"in": ["b3", "b4"]}}
    k = type("K", (), {"contexts": cs, "anti_contexts": anti})()
    ctx, a = CX.contexts_from_knowledge(k)
    assert ctx == spec and a.to_mapping() == {"regime.label": {"in": ["bull_calm"]}}
    plain = type("K", (), {"contexts": spec.to_mapping(), "anti_contexts": {}})()
    assert CX.contexts_from_knowledge(plain)[0] == spec                                  # the mapping form still works
    obj = _base_knowledge(contexts=cs, anti_contexts=anti)
    arc = AR.Archive(tmp_path / "arc")
    rec = arc.put_knowledge(obj, occurred_at="2020-01-01")
    assert rec.payload["contexts"] == spec.to_mapping() and rec.payload["contexts_typed"]["conditions"][1]["op"] == "ge"
    assert rec.payload["anti_contexts_typed"]["any_of"] is True


def test_typed_contextset_that_cannot_be_buckets_is_refused_not_widened(tmp_path):
    with pytest.raises(ValueError):
        CX.ContextSpec.from_typed(KN.ContextSet((KN.Condition("regime", "regime.label", "eq", labels=("nonsense",)),)))
    with pytest.raises(ValueError):
        CX.ContextSpec.from_typed(KN.ContextSet((KN.Condition("regime", "regime.label", "lt", nums=(3.0),)))) if False else \
            CX.ContextSpec.from_typed(KN.ContextSet((KN.Condition("market", "regime.label", "ge", nums=(3.0,)),)))
    with pytest.raises(ValueError):                                                       # a disjunction over two features
        CX.ContextSpec.from_typed(KN.ContextSet((KN.Condition("regime", "regime.label", "eq", labels=("stress",)),
                                                 KN.Condition("volatility", "volatility.vol_rank", "ge", nums=(0.8,))), any_of=True))
    with pytest.raises(ValueError):                                                       # contradictory: accepts nothing
        CX.ContextSpec.from_typed(KN.ContextSet((KN.Condition("regime", "regime.label", "eq", labels=("stress",)),
                                                 KN.Condition("regime", "regime.label", "ne", labels=("stress",)))))
    bad = _base_knowledge(contexts=KN.ContextSet((KN.Condition("volatility", "volatility.vol_rank", "between", nums=(0.41, 0.42)),)))
    with pytest.raises(AR.ArchiveError):                                                  # covers no whole bucket: stored empty would be universal
        AR.Archive(tmp_path / "arc2").put_knowledge(bad, occurred_at="2020-01-01")
    assert CX.ContextSpec.from_typed(KN.ContextSet(())).conditions == ()                 # the empty case: matches everything


def _index_with_cases(n_cases, transfer=None):
    idx = RV.KnowledgeIndex()
    it = _base_knowledge(confidence=Confidence(truth=0.8, transfer=transfer))
    idx.add_item(it)
    rng = np.random.default_rng(0)
    for i in range(n_cases):
        sit = ST.Situation(tuple(ST.Block.make(k) for k in ST.BLOCK_ORDER), ())
        idx.add_support(it.knowledge_id, sit, f"2019-{1 + i % 12:02d}-{1 + i % 27:02d}", float(rng.normal(0.01, 0.02)))
    return idx, it


def test_transfer_test_runs_once_per_item_not_once_per_row(monkeypatch):
    idx, it = _index_with_cases(30)
    calls = []
    real = RV.transfer_evidence
    monkeypatch.setattr(RV, "transfer_evidence", lambda *a, **k: calls.append(1) or real(*a, **k))
    r = RV.Retriever(idx)
    for _ in range(25):                                                                   # 25 rows of one decision day
        r._transfer_factor(it, it.knowledge_id, NOW)
    assert len(calls) == 1
    sit = ST.Situation(tuple(ST.Block.make(k) for k in ST.BLOCK_ORDER), ())
    idx.add_support(it.knowledge_id, sit, "2019-12-30", 0.02)                             # a new case invalidates the cache
    r._transfer_factor(it, it.knowledge_id, NOW)
    assert len(calls) == 2
    assert r._transfer_factor(it, it.knowledge_id, NOW).value == real(idx, it.knowledge_id, NOW)["score"]


def test_cached_transfer_still_refuses_a_case_from_the_future():
    idx, it = _index_with_cases(30)
    r = RV.Retriever(idx)
    r._transfer_factor(it, it.knowledge_id, NOW)                                          # cached
    with pytest.raises(FirewallBreach):
        r._transfer_factor(it, it.knowledge_id, "2019-03-01")                             # asked as of a date before its cases matured
    idx2, it2 = _index_with_cases(4)
    assert r.__class__(idx2)._transfer_factor(it2, it2.knowledge_id, NOW).value is None    # too few cases: UNTESTED, not a guess
    idx3, it3 = _index_with_cases(30, transfer=0.6)
    assert RV.Retriever(idx3)._transfer_factor(it3, it3.knowledge_id, NOW).value == 0.6    # a recorded value is used, no test run


def _sparse(blocks):
    return ST.coarsen(_full(), keep=blocks)


def _full():
    return _helpers("test_learning_situation").mk()


def test_similarity_defaults_are_coverage_aware():
    full = _full()
    sparse = _sparse(["regime", "volatility", "market"])                                  # ~18% of the fields observed
    fixed = SM.SimilarityWeights(coverage_aware=False)
    assert SM.compare(full, sparse, fixed).total is None                                   # the old default: not comparable
    assert SM.compare(full, sparse).total is not None                                      # coverage-aware default: comparable
    assert SM.compare(sparse, sparse).total is not None
    lone = _sparse(["regime"])                                                             # one block only: still unknown, never a guess
    assert SM.compare(full, lone).total is None and SM.compare(lone, lone).total is None
    M = SM.SituationMatrix([full, sparse, lone])
    tot, cov, ok = M.totals(full)
    assert bool(ok[0]) and bool(ok[1]) and not bool(ok[2])
    for q in (full, sparse, lone):                                                        # scalar and vectorised paths agree
        t, _, _ = M.totals(q)
        for j, c in enumerate((full, sparse, lone)):
            s = SM.compare(q, c).total
            assert (np.isnan(t[j]) and s is None) or abs(t[j] - s) < 1e-5
    assert SM.DEFAULT.weights_id() != fixed.weights_id()
    assert SM.weights_from_record(SM.weights_to_record(SM.DEFAULT)) == SM.DEFAULT
    assert SM.SimilarityWeights(coverage_floor=0.0).validate() and SM.SimilarityWeights(coverage_factor=1.5).validate()
    lo, hi = SM.DEFAULT.thresholds(0.25), SM.DEFAULT.thresholds(0.95)
    assert lo[1] < hi[1] == SM.DEFAULT.min_total_coverage and lo[1] >= SM.DEFAULT.coverage_floor


# =====================================================================================================================
# Part B - the blind half
# =====================================================================================================================

def make_panel(day="2209-01-07", n=40, seed=0, feats=("f1", "f2")):
    rng = np.random.default_rng(seed)
    ix = pd.MultiIndex.from_product([[pd.Timestamp(day)], [f"S{i:04d}" for i in range(n)]], names=["date", "ticker"])
    return pd.DataFrame({f: rng.normal(size=n) for f in feats}, index=ix)


def release_of(items, step=0):
    tot = sum(w for _, _, w, _ in items) or 1.0
    return TV.TraderRelease(step, tuple(sorted((TV.TraderMemoryItem.make(TV.opaque_token(pid), "pattern", w / tot, f, lean, 5)
                                                for pid, f, w, lean in items), key=lambda i: i.item_id)))


def test_daydata_refuses_another_days_panel_and_unmatured_labels():
    now = pd.Timestamp("2209-01-07")
    TP.DayData(now, make_panel("2209-01-07")).check()
    with pytest.raises(FirewallBreach, match="cross-section"):
        TP.DayData(now, make_panel("2209-01-06")).check()
    ok = pd.DataFrame({"ret": [0.01], "matured": [pd.Timestamp("2209-01-06")]}, index=make_panel().index[:1])
    TP.DayData(now, None, ok).check()
    for late in ("2209-01-07", "2209-01-20"):
        bad = ok.assign(matured=pd.Timestamp(late))
        with pytest.raises(FirewallBreach, match="not matured"):
            TP.DayData(now, None, bad).check()
    with pytest.raises(FirewallBreach):
        TP.DayData(now, None, ok.drop(columns="matured")).check()
    TP.DayData(now).check()                                                               # the empty day is fine


def test_advisor_votes_come_only_from_the_release():
    panel = make_panel(n=50)
    adv = TP.MemoryAdvisor(5)
    top = adv.levels(panel, "f1") == 4
    rel = release_of([("f1:q4", {"f1": 1.0, "level": 4.0, "strength_band": 0.5}, 1.0, 0.8)])
    v = adv.votes(rel, panel)
    assert np.allclose(v[top], 0.4) and (v[~top] == 0.0).all()                # weight 1 x band 0.5 x lean 0.8
    assert (adv.votes(release_of([("f1:q4", {"f1": 1.0, "level": 4.0}, 1.0, -0.9)]), panel)[top] < 0).all()
    weak = adv.votes(release_of([("f1:q4", {"f1": 1.0, "level": 4.0, "strength_band": 0.0}, 1.0, 1.0)]), panel)
    assert (weak == 0.0).all()                                                             # a lone weak memory (weight 1.0) carries no vote
    assert (adv.votes(TV.TraderRelease(0, ()), panel) == 0.0).all()                       # empty release: no vote
    other = release_of([("zz:q4", {"zz": 1.0, "level": 4.0}, 1.0, 1.0)])
    assert (adv.votes(other, panel) == 0.0).all()                                         # a feature the panel does not have
    two = release_of([("f1:q4", {"f1": 1.0, "level": 4.0}, 1.0, 1.0), ("f2:q4", {"f2": 1.0, "level": 4.0}, 1.0, 1.0)])
    assert adv.votes(two, panel).max() <= 1.0
    nanned = panel.copy()
    nanned.iloc[0, 0] = np.nan
    assert adv.levels(nanned, "f1").iloc[0] == -1


class StubLearner:
    """The three things PathTrader asks of a learner: pending episodes, decide_batch, picks."""

    class cfg:
        n_quantiles = 5

    def __init__(self, panel, long_keys):
        self.pending, self.panel, self.long_keys = {}, panel, long_keys

    def ready(self, ep, now):
        return False

    def decide_batch(self, now, panel):
        return object()

    def picks(self, ep):
        return [(k, type("D", (), {"size": 0.25})()) for k in self.long_keys]


def day_of(release, step=0):
    return TV.TraderDay(TV.TraderSituation.make(step, {"rel_vix": 0.1}), release)


def test_veto_mode_drops_only_strongly_opposed_picks():
    panel = make_panel(n=50)
    adv = TP.MemoryAdvisor(5)
    lv = adv.levels(panel, "f1")
    against, neutral = lv[lv == 4].index[0], lv[lv == 2].index[0]
    rel = release_of([("f1:q4", {"f1": 1.0, "level": 4.0, "strength_band": 1.0}, 1.0, -1.0)])
    data = TP.DayData(pd.Timestamp("2209-01-07"), panel)
    res = {}
    for mode in ("off", "record", "veto"):
        tr = TP.PathTrader(StubLearner(panel, [against, neutral]), TP.PathConfig(advice=mode, min_rows=5))
        res[mode] = tr.act(day_of(rel), data)
    assert [k for k, _ in res["off"].picks] == [k for k, _ in res["record"].picks] == [against, neutral]
    assert [k for k, _ in res["veto"].picks] == [neutral] and res["veto"].vetoed == 1 and res["veto"].n_long == 2
    assert res["off"].mean_vote is None and res["record"].mean_vote == pytest.approx(-0.5)


def test_trader_skips_days_without_a_panel_or_enough_rows():
    panel = make_panel(n=10)
    tr = TP.PathTrader(StubLearner(panel, []), TP.PathConfig(min_rows=30))
    assert not tr.act(day_of(TV.TraderRelease(0, ())), TP.DayData(pd.Timestamp("2209-01-07"))).decided
    r = tr.act(day_of(TV.TraderRelease(0, ())), TP.DayData(pd.Timestamp("2209-01-07"), panel))
    assert r.decided and "10 rows" in r.skipped and r.picks == ()


def test_trader_refuses_a_traderday_that_carries_a_date():
    sit = TV.TraderSituation.make(0, {"rel_vix": 0.1})
    tr = TP.PathTrader(StubLearner(make_panel(), []), TP.PathConfig())
    with pytest.raises(FirewallBreach):
        TV.TraderDay(sit, TV.TraderRelease(1, ()))                                        # step mismatch is refused at construction
    with pytest.raises(FirewallBreach):
        TV.TraderSituation.make(0, {"real_date": 3.0})                                    # a date-describing field name cannot be built
    tr.act(day_of(TV.TraderRelease(0, ())), TP.DayData(pd.Timestamp("2209-01-07")))       # and a clean one passes


def test_trader_classes_reference_no_curator_symbol():
    assert TP.trader_side_violations() == []
    src = inspect.getsource(TP)
    leaky = src.replace('        self.advisor = MemoryAdvisor(learner.cfg.n_quantiles)',
                        '        self.advisor = MemoryAdvisor(learner.cfg.n_quantiles)\n        self.peek = Curator')
    assert leaky != src and any("PathTrader: uses Curator" in v for v in TP.trader_side_violations(leaky))
    anchor = "        x = panel[feature].astype(float)\n"
    leaky2 = src.replace(anchor, anchor + '        note = "curator_store"\n')
    assert leaky2 != src and any("MemoryAdvisor" in v and "store" in v for v in TP.trader_side_violations(leaky2))
    leaky3 = src.replace("        self.days = 0\n", "        self.days = RealClock\n", 1)
    assert any("PathTrader: uses RealClock" in v for v in TP.trader_side_violations(leaky3))


def test_import_closure_of_the_trader_path_never_reaches_the_curator_or_this_module():
    assert TV.assert_trader_path_clean() > 20
    closure = TV.trader_closure()
    assert "engine.learning.curator" not in closure and "engine.learning.test_path" not in closure
    assert "engine.learning.learner" in closure and "engine.livesim" in closure
    bad = TV.trader_path_violations(entries=["engine/learning/test_path.py"])            # the trusted file itself DOES reach the curator
    assert bad and any("curator" in v.target.lower() for v in bad)
    live = (ROOT / "engine" / "livesim.py").read_text(encoding="utf-8")
    tree = ast.parse(live)
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    assert "hook_factory" in [a.arg for a in run.args.args] and run.args.defaults[-1].value is None
    imported = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)] +                [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert imported and not [m for m in imported if "test_path" in m or "curator" in m]


# =====================================================================================================================
# Part C - the trusted half
# =====================================================================================================================

def test_outcome_book_serves_a_label_only_once_its_exit_open_is_strictly_past():
    sessions = pd.bdate_range("2209-01-03", periods=20)
    opens = pd.DataFrame({"A": np.arange(100.0, 120.0), "B": np.arange(50.0, 70.0)}, index=sessions)
    keys = pd.MultiIndex.from_tuples([(sessions[2], "A"), (sessions[2], "B")])
    book = TP.OutcomeBook(5)
    book.record(sessions[2], keys)
    exit_pos = 2 + 1 + 5
    assert book.outcomes(sessions, opens.iloc[: exit_pos], sessions[exit_pos - 1]) is None
    assert book.outcomes(sessions, opens.iloc[: exit_pos + 1], sessions[exit_pos]) is None      # exit open is today: not served
    out = book.outcomes(sessions, opens.iloc[: exit_pos + 2], sessions[exit_pos + 1])
    assert len(out) == 2 and (out["matured"] == sessions[exit_pos]).all()
    assert out.loc[(sessions[2], "A"), "ret"] == pytest.approx(opens.iloc[exit_pos]["A"] / opens.iloc[3]["A"] - 1)
    TP.DayData(sessions[exit_pos + 1], None, out).check()
    book.consume(sessions[2])
    assert len(book) == 0 and book.outcomes(sessions, opens, sessions[-1]) is None
    holes = opens.copy()
    holes.iloc[exit_pos, 1] = np.nan                                                          # a name with no exit price is dropped, not zeroed
    book2 = TP.OutcomeBook(5)
    book2.record(sessions[2], keys)
    assert list(book2.outcomes(sessions, holes, sessions[exit_pos + 1]).index.get_level_values(1)) == ["A"]


class FakeLearner:
    def __init__(self, weekly):
        self._weekly = weekly


def summary(decided, matured):
    from engine.learning.learner import EpisodeSummary
    return EpisodeSummary("E1", decided, 40, 3, 37, 0, True, matured, 0.001)


def make_filer(tmp_path=None, **cfg):
    clock = TP.RealClock(pd.Timedelta(days=7 * 9000))
    cur = Curator(store_root=None, code_hash="pinned-test-code")
    return TP.MemoryFiler(cur, clock, TP.PathConfig(**cfg)), cur, clock


def test_memory_is_filed_under_the_real_year_once_and_only_when_strong():
    filer, cur, clock = make_filer()
    dec, mat = clock.disguised("2021-12-29"), clock.disguised("2022-01-07")                  # the week straddles a real new year
    cur.day("2022-01-10", {"m_vix": 20.0})                                                    # today, after the label matured
    weekly = {"r5:q4": [(str(mat.date()), 0.02, 0.005, 12)], "vol20:q0": [(str(mat.date()), 0.0001, 0.02, 12)],
              "log_dv:q4": [(str(mat.date()), -0.03, 0.006, 12)]}
    learner = FakeLearner(weekly)
    n = filer.file(learner, [summary(str(dec.date()), str(mat.date()))], {str(dec.date()): {"m_vix": 19.0, "m_breadth": 0.5}})
    assert n == 2 and filer.filed == 2 and filer.weak == 1 and filer.orphans == 0
    mems = cur.store.memories()
    assert {m.real_year for m in mems} == {2021} and {m.filed_date for m in mems} == {"2021-12-29"}     # filed by the day it operated
    assert {m.matured_at for m in mems} == {"2022-01-07"} and cur.store.counts_by_year() == {2021: 2}
    leans = {m.payload["features"].get("r5", m.payload["features"].get("log_dv")): m.lean for m in mems}
    assert all(m.context == {"m_vix": 19.0, "m_breadth": 0.5} for m in mems) and min(m.lean for m in mems) < 0 < max(m.lean for m in mems)
    assert filer.file(learner, [summary(str(dec.date()), str(mat.date()))], {str(dec.date()): {"m_vix": 19.0}}) == 0     # cursor: nothing new
    filer2, cur2, _ = make_filer()
    cur2.day("2022-01-10", {"m_vix": 20.0})
    assert filer2.file(FakeLearner(weekly), [summary(str(dec.date()), str(mat.date()))],
                        {str(dec.date()): {"m_vix": 19.0, "m_breadth": 0.5}}) == 2
    assert [m.mem_id for m in cur2.store.memories()] == [m.mem_id for m in mems]              # same content, same ids


def test_filer_refuses_feature_names_that_read_as_dates_or_ranks_and_files_nothing_without_state():
    filer, cur, clock = make_filer()
    dec, mat = clock.disguised("2021-03-05"), clock.disguised("2021-03-12")
    cur.day("2021-03-15", {"m_vix": 20.0})
    weekly = {"rank_vol:q4": [(str(mat.date()), 0.02, 0.004, 12)], "date_x:q0": [(str(mat.date()), 0.02, 0.004, 12)]}
    st = {str(dec.date()): {"m_vix": 20.0}}
    assert filer.file(FakeLearner(weekly), [summary(str(dec.date()), str(mat.date()))], st) == 0
    assert filer.refused == 2 and len(cur.store) == 0
    f2, c2, _ = make_filer()
    assert f2.file(FakeLearner({"r5:q4": [(str(mat.date()), 0.02, 0.004, 12)]}), [summary(str(dec.date()), str(mat.date()))], {}) == 0
    assert f2.no_state == 1 and len(c2.store) == 0
    f3, c3, _ = make_filer(file_max_per_week=2)
    many = {f"r{i}:q4": [(str(mat.date()), 0.02 + i / 1000, 0.004, 12)] for i in range(6)}
    c3.day("2021-03-15", {"m_vix": 20.0})
    assert f3.file(FakeLearner(many), [summary(str(dec.date()), str(mat.date()))], st) == 2       # only the strongest two
    empty, ce, _ = make_filer()
    assert empty.file(FakeLearner({}), [], {}) == 0 and empty.file(FakeLearner({}), [summary("2021-03-05", "2021-03-12")], {}) == 0


def test_a_label_that_has_not_matured_cannot_be_filed():
    filer, cur, clock = make_filer()
    dec, mat = clock.disguised("2021-03-05"), clock.disguised("2021-03-12")
    cur.day("2021-03-12", {"m_vix": 20.0})                                                     # the clock is ON the maturity day: not yet known
    with pytest.raises(FirewallBreach):
        filer.file(FakeLearner({"r5:q4": [(str(mat.date()), 0.03, 0.004, 12)]}), [summary(str(dec.date()), str(mat.date()))],
                   {str(dec.date()): {"m_vix": 20.0}})


def test_config_validation_and_the_empty_cases():
    assert TP.PathConfig().validate() == []
    for bad in (dict(horizon_sessions=0), dict(release_k=0), dict(advice="on"), dict(veto_vote=0.5), dict(sample_names=1),
                dict(min_rows=1), dict(features=()), dict(min_feature_coverage=0.0), dict(file_max_per_week=0)):
        assert TP.PathConfig(**bad).validate(), bad
    hist = pd.DataFrame({"a": np.arange(100.0), "b": np.nan, "c": 1.0, "d": np.r_[np.arange(50.0), [np.nan] * 50]})
    cfg = TP.PathConfig(features=("a", "b", "c", "d", "zz"))
    assert TP.choose_features(hist, cfg) == ("a",)                                             # empty, constant, half-missing, absent: all left out
    assert TP.choose_features(hist.iloc[:0], cfg) == ()
    assert TP.PathConfig().digest() == TP.PathConfig().digest() != TP.PathConfig(seed=1).digest()
    clock = TP.RealClock(pd.Timedelta(days=63000))
    assert clock.disguised(clock.real("2209-01-07")) == pd.Timestamp("2209-01-07")


# =====================================================================================================================
# Part D - one synthetic window end to end, under two disguises
# =====================================================================================================================

FEATS = ("r5", "r20", "vol20", "atr_pct", "log_dv")
START = "2022-05-01"


def make_data(start, n=40, seed=3, years=1):
    start = pd.Timestamp(start)
    idx = pd.bdate_range(start - pd.DateOffset(years=years), start + pd.DateOffset(months=12) - pd.Timedelta(days=1))
    rng = np.random.default_rng(seed)
    tick = sorted(f"AB{chr(65 + i % 26)}{i}" for i in range(n))
    ret = rng.normal(0.0004, 0.015, (len(idx), len(tick)))
    close = pd.DataFrame(50 * np.exp(ret.cumsum(0)), index=idx, columns=tick)
    opn = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.002, close.shape))
    vol = pd.DataFrame(rng.uniform(5e5, 2e6, close.shape), index=idx, columns=tick)
    stocks = {"Close": close, "Open": opn, "High": close * 1.01, "Low": close * 0.99, "Volume": vol}
    mk = pd.DataFrame({"SPY": 100 * np.exp(rng.normal(0.0003, 0.01, len(idx)).cumsum()), "^VIX": 18 + rng.normal(0, 1, len(idx))}, index=idx)
    market = {"Close": mk, "Open": mk, "High": mk, "Low": mk, "Volume": mk * 0 + 1e6}
    at = pd.DatetimeIndex(idx[::9], tz="UTC") + pd.Timedelta(hours=13)
    ev = pd.DataFrame({"ticker": [tick[i % n] for i in range(len(at))], "accepted": at, "kind": "8K", "form": "8-K"})
    fd = idx[::7]
    ins = pd.DataFrame({"symbol": [tick[i % n] for i in range(len(fd))], "filed": fd, "tdate": fd - pd.Timedelta(days=2)})
    return stocks, market, ev, ins, pd.DataFrame({"ticker": tick, "sic": 3570})


def future_item():
    return {"item_id": TV.opaque_token("planted-future"), "kind": "pattern", "weight": 1.0, "features": {"vol20": 1.0, "level": 4.0},
            "lean": 0.9, "horizon": 5}


def past_item():
    return {"item_id": TV.opaque_token("planted-past"), "kind": "pattern", "weight": 1.0, "features": {"r5": 1.0, "level": 4.0},
            "lean": 0.9, "horizon": 5}


def play(home, shift_weeks, monkeypatch):
    """One enforced window of the hardened feed with the path runner ticking beside a stand-in trader. Two memories are planted
    before the clock starts: one that matured BEFORE the window (may be released) and one that matures after it ends (must never be)."""
    monkeypatch.setattr(livesim, "DIR", home)
    rec = BG.seal_window([], 7, "2026-01-01", tag="t1")
    rec.update(start=START, shift_days=7 * shift_weeks)
    rec["digest"] = BG.seal_digest(rec)
    (home / "sealed_t1.json").write_text(json.dumps(rec))
    from engine.leak_audit import hardened_feed_class
    feed = hardened_feed_class()(livesim.SealedYear("t1"), warmup_years=1, data=make_data(START), use_insider=False)
    feed.precompute_features()
    cfg = TP.PathConfig(min_rows=20, sample_names=None, preroll_days=40, features=FEATS, seed=5)
    hist = feed.features_until_now()[0]
    state = {c: float(v) for c, v in hist[[c for c in hist.columns if str(c).startswith("m_")]].iloc[-1].items() if np.isfinite(v)}
    pre = Curator(store_root=None, code_hash="pinned-test-code")                 # no clock yet, so a future memory can be planted
    pre.file(past_item(), "2022-03-01", 2022, state, matured_at="2022-03-08", reliability=0.9)
    pre.file(future_item(), "2023-06-05", 2023, state, matured_at="2023-06-12", reliability=0.9)
    with pytest.raises(FirewallBreach):                                           # once the clock runs the curator itself refuses this
        c2 = Curator(store_root=None, code_hash="pinned-test-code")
        c2.day("2022-04-29", state)
        c2.file(future_item(), "2023-06-05", 2023, state, matured_at="2023-06-12")
    cur = Curator(store=pre.store, code_hash="pinned-test-code")
    runner = TP.PathRunner(feed, cfg, curator=cur, code_hash_fn=lambda: "pinned-test-code")
    seen = []
    orig = runner.trader.act

    def spy(day, data):
        seen.append((runner.clock.real(data.now), day, data))
        return orig(day, data)
    runner.trader.act = spy

    def tick():
        runner.on_tick()
        feed.mark_processed()
    livesim.drive(feed, tick)
    return feed, runner, seen


@pytest.fixture(scope="module")
def window_a(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    try:
        yield play(tmp_path_factory.mktemp("a"), 9000, mp)
    finally:
        mp.undo()


@pytest.fixture(scope="module")
def window_b(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    try:
        yield play(tmp_path_factory.mktemp("b"), 9500, mp)
    finally:
        mp.undo()


def test_every_session_ticks_the_clock_and_the_window_stays_clean(window_a):
    feed, runner, seen = window_a
    c = runner.counts
    assert c["days"] == len(seen) == len(feed.clock.sessions) > 240
    assert 45 <= c["decision_days"] <= 56 and c["skipped_days"] == 0
    assert c["releases"] == c["decision_days"]                                            # a release is served on decision days only
    assert not [f for f in feed.audit() if f.severity == "fail"]
    assert not [f for f in runner.findings() if f.severity == "fail"], runner.findings()
    assert runner.curator.step + 1 == c["days"] + runner.preroll_days                     # the curator's clock ran every session
    assert [r.step for r in runner.results] == list(range(runner.preroll_days, runner.preroll_days + c["days"]))


def test_the_learner_learns_and_memory_is_filed_by_real_year(window_a):
    feed, runner, _ = window_a
    rep = runner.report()
    assert rep["counts"]["learned_episodes"] >= 40 and rep["learner"]["learned"] == rep["counts"]["learned_episodes"]
    assert rep["counts"]["unresolved_episodes"] == 0 and rep["filed"] > 0 and rep["orphans"] == 0 and rep["refused_at_the_wall"] == 0
    assert set(rep["curator_years_filed"]) == {"2022", "2023"}                            # a window that crosses New Year files into both
    assert sum(rep["curator_years_filed"].values()) == rep["filed"]
    mems = [m for m in runner.curator.store.memories() if m.key not in (TV.opaque_token("planted-past"), TV.opaque_token("planted-future"))]
    assert mems and all(m.real_year == as_year(m.filed_date) for m in mems)
    start, end = pd.Timestamp("2022-05-01") - pd.Timedelta(days=1), pd.Timestamp("2023-05-01")
    assert all(start <= pd.Timestamp(m.filed_date) < end and pd.Timestamp(m.filed_date) < pd.Timestamp(m.matured_at) < end for m in mems)
    assert runner.curator.store.verify()["ok"] and runner.curator.audit_verify()["ok"]
    assert rep["label"].endswith("NOT VALIDATED") and rep["dropped_features"] == [] and rep["features"] == list(FEATS)
    json.dumps(rep, default=str)                                                          # the report is what result2.json stores


def as_year(s):
    return int(s[:4])


def test_nothing_the_trader_was_shown_names_the_real_year(window_a):
    feed, runner, seen = window_a
    assert len(seen) > 240
    hits = 0
    for real, day, data in seen:
        text = day.json()
        assert TV.find_violations(day.to_dict()) == []
        hits += sum(TV.leak_scan_text(text).values())
        assert data.now.year >= 2100                                                      # the feed's disguised clock, never the real one
        assert not re.search(r"(?<![\d.])(2022|2023)(?!\d)", text)
    assert hits == 0 and runner.release_hits == {}
    sample = seen[-1][1].json()
    assert "curator" not in sample.lower() and "year" not in sample.lower() and "real" not in sample.lower()


def test_a_memory_from_the_future_is_never_released_and_a_past_one_can_be(window_a):
    feed, runner, seen = window_a
    fut, past = TV.opaque_token("planted-future"), TV.opaque_token("planted-past")
    ids_by_day = [(real, {i.item_id for i in day.release.items}) for real, day, _ in seen]
    assert not any(fut in ids for _, ids in ids_by_day)                                   # matures 2023-06-12, after the window ends
    assert any(past in ids for _, ids in ids_by_day)                                      # matured before the window: legitimately released
    first = min(real for real, ids in ids_by_day if past in ids)
    assert first > pd.Timestamp("2022-03-08")
    earliest = {}
    for m in runner.curator.store.memories():
        earliest[m.key] = min(earliest.get(m.key, m.matured_at), m.matured_at)
    for real, day, _ in seen:                                                             # every released key had a record matured before that day
        for it in day.release.items:
            assert pd.Timestamp(earliest[it.item_id]) < real


def test_filed_memory_and_releases_do_not_depend_on_the_disguise(window_a, window_b):
    fa, ra, sa = window_a
    fb, rb, sb = window_b
    assert fa._shift != fb._shift and sorted(fa._map.values()) == sorted(fb._map.values()) and fa._map != fb._map
    ids = lambda r: sorted(m.mem_id for m in r.curator.store.memories())
    assert ids(ra) == ids(rb) and len(ids(ra)) > 10                                        # same real months, same memory, whatever the costume
    assert ra.counts == rb.counts and ra.filer.by_year == rb.filer.by_year
    assert [x.release.digest() for _, x, _ in sa] == [x.release.digest() for _, x, _ in sb]
    assert [r.n_long for r in ra.results] == [r.n_long for r in rb.results]


# =====================================================================================================================
# the loop flag
# =====================================================================================================================

@pytest.fixture(scope="module")
def loop(tmp_path_factory):
    real, livesim.DIR = livesim.DIR, tmp_path_factory.mktemp("loopdir")
    argv, sys.argv = sys.argv, ["livesim_loop2.py"]
    try:
        spec = importlib.util.spec_from_file_location("livesim_loop2_s17b", ROOT / "scripts" / "livesim_loop2.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        yield m
    finally:
        sys.argv = argv
        livesim.DIR = real


def test_learner_flag_defaults_off_and_refuses_anything_else(loop):
    assert loop.LEARNER == "off" and loop.LEARNER_ARGS == []
    assert loop.learner_flag(["x.py"]) == "off" and loop.learner_flag(["x.py", "3", "--learner", "legit"]) == "legit"
    assert loop.learner_flag(["x.py", "--learner", "off"]) == "off"
    for bad in (["x.py", "--learner"], ["x.py", "--learner", "yes"], ["x.py", "--learner", "LEGIT"]):
        with pytest.raises(SystemExit):
            loop.learner_flag(bad)


def test_workers_inherit_the_flag_and_the_gates_are_still_wired(loop):
    src = (ROOT / "scripts" / "livesim_loop2.py").read_text(encoding="utf-8")
    assert src.count("*LEARNER_ARGS") == 2                                                # the supervised worker and the stale rerun
    for gate in ("fill_audit.gate", "scramble_after", "reveal_round", "checkpoint_run", "plan_round", "classify_round"):
        assert gate in src
    assert '**extra' in src and "trader.hook.findings()" in src and 'r["legit"] = trader.hook.report()' in src
    assert "MAXW = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 60" in src     # positional arg unchanged
