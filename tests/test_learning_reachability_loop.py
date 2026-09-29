"""Reachability of the learning loop (contract C62 sections 4, 60, 61, 83, 85; canon C58, C61, C64; checklist L02).
Status: IMPLEMENTED — NOT VALIDATED.  Synthetic data only; no real cache is read.

Text presence is not integration: the 29 Sep audit found hooks whose `wiring.x(` text sat in functions nothing called.  This file
replaces that check with REACHABILITY: an AST call graph built from the one production entry of the learner
(scripts/livesim_loop2.py main / worker with --learner legit -> test_path.PathRunner -> learner.LegitimateLearner) must reach every
engine/learning module (except the research-only ones listed below, each with its reason), and every queued hook in S21a's scope
must have a CALL SITE inside a reached function.  Then behavioural tests on a planted world prove that data actually flows:
a planted contradiction reaches the research queue and the in-loop experiment's result closes it; a credit result changes a
belief; a break in an item's outcomes changes its lifecycle through the retirement gate; the firewalls refuse planted leaks;
and what the loop learned is on disk and reloads."""
from __future__ import annotations

import ast
import dataclasses
import functools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from engine.learning import checkpoints as CK                                                     # noqa: E402
from engine.learning import credit as CR                                                          # noqa: E402
from engine.learning import decision_contract as DC                                               # noqa: E402
from engine.learning import firewalls as FW                                                       # noqa: E402
from engine.learning import health as HE                                                          # noqa: E402
from engine.learning import knowledge as KN                                                       # noqa: E402
from engine.learning import knowledge_graph as KG                                                 # noqa: E402
from engine.learning import learner as LN                                                         # noqa: E402
from engine.learning import loop_hooks as LH                                                      # noqa: E402
from engine.learning import planted_world as PW                                                   # noqa: E402
from engine.learning import research_policy as RP                                                 # noqa: E402
from engine.learning import research_priority as RPR                                              # noqa: E402
from engine.learning import retirement as RT                                                      # noqa: E402
from engine.learning.core import Edge, FirewallBreach, Promotion                                  # noqa: E402

ENTRY = "scripts/livesim_loop2.py"
ENTRY_ROOTS = ("main", "worker")                    # the two things its __main__ block runs (the round loop and one window)
# Modules that no production decision should ever run, each with the reason it is research-only.
RESEARCH_ONLY = {}
# (module, callable) of every INTEGRATION.md hook in S21a's scope; the call must sit in a function the entry reaches.
HOOKS = (
    ("decision_contract", "policy_check"), ("decision_contract", "readiness"),
    ("knowledge", "with_failure"), ("knowledge", "with_relation"), ("knowledge", "audit_future"), ("knowledge", "as_of"),
    ("interpretation", "next_test"), ("unknowns", "rank_unknowns"),
    ("competition", "boundary_field"), ("questions", "ask_many"), ("boundary", "to_knowledge"),
    ("complexity", "spec_from_knowledge"), ("complexity", "within_budget"), ("hierarchy", "estimate"),
    ("credit", "update_proposals"), ("credit", "credit_edges"), ("credit", "masked_pairs"), ("redundancy", "analyze"),
    ("redundancy", "edges"), ("belief", "update"),
    ("break_detection", "run"), ("reliability", "contexts_from_condition"), ("lifecycle", "apply_to_ledger"),
    ("health", "inputs_from_knowledge"), ("health", "epistemic_proposals"), ("temporal", "to_evidence"),
    ("calibration", "combined_influence"),
    ("contradiction_monitor", "run_period"), ("contradiction_monitor", "research_questions"),
    ("contradiction_monitor", "dashboard_rows"), ("reports", "health_dashboard"), ("knowledge_graph", "contradiction_keys"),
    ("research_priority", "step"), ("research_priority", "propose_selected"), ("research_priority", "update_from_result"),
    ("research_priority", "signals_from_health"), ("research_priority", "signals_from_data_audit"),
    ("experiment_memory", "record_result"),
    ("failed_learners", "seed_registry"), ("failed_learners", "check_proposal"),
    ("disagreement", "conflict_score"), ("disagreement", "assess"),
    ("portfolio_value", "decompose_value"), ("scorecard", "scorecard_for_learner"), ("scorecard", "append"),
    ("same_year", "record_of"), ("controls", "standard_controls"),
    ("compute", "job_for"), ("checkpoints", "resume_verified"), ("checkpoints", "write_interruption"),
    ("champion", "audit_decision_sources"),
)
# Written reason for the one hook satisfied by a sibling call: ContradictionMonitor.signals() is research_questions() + make_signal
# on the raw knowledge ids, and a raw hash id can contain a year-like digit run that the research identity firewall refuses, so the
# loop builds the same signals from research_questions() on safe tokens.


# ================================================================================================ the call graph
def _module_name(path: Path) -> str:
    rel = path.relative_to(ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


class Mod:
    """One parsed module: top-level symbols, class methods, an alias table of every import in the file, and module-level code."""

    def __init__(self, name: str, path: Path, tree: ast.Module):
        self.name, self.path, self.tree = name, path, tree
        self.pkg = name if path.name == "__init__.py" else name.rpartition(".")[0]
        self.defs: dict[str, ast.AST] = {}
        self.classes: dict[str, ast.ClassDef] = {}
        self.aliases: dict[str, str] = {}
        self.toplevel: list[ast.AST] = []
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.defs[n.name] = n
            elif isinstance(n, ast.ClassDef):
                self.defs[n.name] = n
                self.classes[n.name] = n
            else:
                self.toplevel.append(n)
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    self.aliases[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
            elif isinstance(n, ast.ImportFrom):
                if n.level:
                    base = self.pkg.split(".")
                    base = base[: len(base) - (n.level - 1)] if n.level > 1 else base
                    mod = ".".join(base + ([n.module] if n.module else []))
                else:
                    mod = n.module or ""
                for a in n.names:
                    self.aliases[a.asname or a.name] = f"{mod}.{a.name}"


@functools.lru_cache(maxsize=1)
def parsed_tree(roots_dirs=("engine", "scripts")) -> dict:
    """Every production module parsed once per test session (tests/ and .venv are not production)."""
    out = {}
    for d in roots_dirs:
        for p in sorted((ROOT / d).rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            try:
                tree = ast.parse(p.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            out[_module_name(p)] = Mod(_module_name(p), p, tree)
    return out


class CallGraph:
    """Symbols are (module, name) or (module, Class.method). A reached class reaches all its methods and bases; a reached function
    reaches everything its body references that resolves through the module's own names and imports; a reached module runs its
    module-level code. Calls are recorded per reached function as (resolved module or None, called name)."""

    def __init__(self, mods: dict | None = None):
        self.mods: dict[str, Mod] = dict(mods if mods is not None else parsed_tree())
        self.reached: set[tuple[str, str]] = set()
        self.mods_run: set[str] = set()
        self.calls: dict[tuple[str, str], list[tuple[str | None, str]]] = {}

    # ---- name resolution
    def _resolve_dotted(self, mod: Mod, parts: list[str]) -> tuple[str | None, str | None]:
        """(module, symbol) a dotted reference resolves to; symbol None means the reference is the module itself."""
        head = parts[0]
        if head in mod.defs:
            return mod.name, head
        target = mod.aliases.get(head)
        if target is None:
            return None, None
        chain = target.split(".") + parts[1:]
        for i in range(len(chain), 0, -1):
            m = ".".join(chain[:i])
            if m in self.mods:
                rest = chain[i:]
                if not rest:
                    return m, None
                sym = rest[0]
                if sym in self.mods[m].defs:
                    return m, sym
                tgt = self.mods[m].aliases.get(sym)             # re-exported name
                if tgt is not None:
                    return self._resolve_dotted(self.mods[m], [sym] + rest[1:])
                return m, None
        return None, None

    @staticmethod
    def _dotted(node) -> list[str] | None:
        out = []
        while isinstance(node, ast.Attribute):
            out.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            return [node.id] + out[::-1]
        return None

    # ---- traversal
    def reach(self, roots: list[tuple[str, str]]) -> "CallGraph":
        todo = list(roots)
        while todo:
            m, sym = todo.pop()
            if (m, sym) in self.reached or m not in self.mods:
                continue
            self.reached.add((m, sym))
            mod = self.mods[m]
            if m not in self.mods_run:
                self.mods_run.add(m)
                todo += self._refs(mod, mod.toplevel, None)
            node = self._node(mod, sym)
            if node is None:
                continue
            if isinstance(node, ast.ClassDef):
                for b in node.bases:
                    d = self._dotted(b)
                    if d:
                        r = self._resolve_dotted(mod, d)
                        if r[0] and r[1]:
                            todo.append(r)
                for s in node.body:
                    if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        todo.append((m, f"{sym}.{s.name}"))
                    else:
                        todo += self._refs(mod, [s], None)
            else:
                todo += self._refs(mod, [node], (m, sym))
        return self

    def _node(self, mod: Mod, sym: str):
        if "." in sym:
            cls, meth = sym.split(".", 1)
            c = mod.classes.get(cls)
            return next((s for s in c.body if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)) and s.name == meth), None) if c else None
        return mod.defs.get(sym)

    def _refs(self, mod: Mod, nodes, owner) -> list[tuple[str, str]]:
        out = []
        calls = self.calls.setdefault(owner, []) if owner is not None else None
        for top in nodes:
            for n in ast.walk(top):
                if isinstance(n, (ast.Name, ast.Attribute)):
                    d = self._dotted(n)
                    if d:
                        r = self._resolve_dotted(mod, d)
                        if r[0] and r[1]:
                            out.append(r)
                        elif r[0]:
                            out.append((r[0], "<module>"))
                if calls is not None and isinstance(n, ast.Call):
                    f = n.func
                    name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None
                    if name:
                        d = self._dotted(f)
                        r = self._resolve_dotted(mod, d) if d else (None, None)
                        calls.append((r[0], name))
        return out

    def modules_reached(self) -> set[str]:
        return {m for m, s in self.reached if s != "<module>"} | self.mods_run

    def call_sites(self, module: str, name: str) -> list[tuple[str, str]]:
        """Reached functions that contain a call named `name` whose base resolves to `module` (or does not resolve: a method on
        an instance, whose class the graph reaches by construction)."""
        return sorted(owner for owner, cs in self.calls.items() if owner in self.reached
                      and any(n == name and (rm is None or rm == module) for rm, n in cs))


@pytest.fixture(scope="module")
def graph():
    g = CallGraph()
    entry = _module_name(ROOT / ENTRY)
    return g.reach([(entry, r) for r in ENTRY_ROOTS])


def test_the_call_graph_resolves_a_planted_chain_and_misses_a_planted_orphan(tmp_path):
    """The instrument must be able to fail: a function nothing calls is not reached, and a call through an alias is."""
    g = CallGraph(mods={})
    src = ("from engine.learning import loop_hooks as LH\n\ndef entry():\n    helper()\n\ndef helper():\n    LH.safe_token('x')\n\n"
           "def orphan():\n    LH.run_same_year_harness(None)\n")
    tree = ast.parse(src)
    g.mods["planted"] = Mod("planted", ROOT / "planted.py", tree)
    lh = ROOT / "engine" / "learning" / "loop_hooks.py"
    g.mods["engine.learning.loop_hooks"] = Mod("engine.learning.loop_hooks", lh, ast.parse(lh.read_text(encoding="utf-8")))
    g.reach([("planted", "entry")])
    assert ("planted", "helper") in g.reached and ("planted", "orphan") not in g.reached
    assert ("engine.learning.loop_hooks", "safe_token") in g.reached
    assert g.call_sites("engine.learning.loop_hooks", "safe_token") == [("planted", "helper")]
    assert g.call_sites("engine.learning.loop_hooks", "run_same_year_harness") == []        # the orphan's call does not count


def test_every_learning_module_is_reached_from_the_production_entry(graph):
    learning = sorted(n for n in graph.mods if n.startswith("engine.learning.") and not n.endswith("__init__"))
    assert len(learning) >= 60                                              # the census saw the package
    missing = [m for m in learning if m not in graph.modules_reached() and m.rsplit(".", 1)[1] not in RESEARCH_ONLY]
    assert missing == [], f"engine/learning modules the production entry never reaches: {missing}"
    for m, why in RESEARCH_ONLY.items():
        assert len(why) > 20, f"research-only module {m} needs a written reason"


def test_every_hook_in_scope_has_a_reachable_call_site(graph):
    lost = [f"{m}.{n}" for m, n in HOOKS if not graph.call_sites(f"engine.learning.{m}", n)]
    assert lost == [], f"hooks without a reachable call site: {lost}"


def test_the_graph_loses_the_audited_dead_modules_without_the_loop_hooks():
    """Mutation check on the real tree: take loop_hooks out and the modules the 29 Sep audit found dead are unreached again."""
    g = CallGraph()
    del g.mods["engine.learning.loop_hooks"]
    g.reach([(_module_name(ROOT / ENTRY), r) for r in ENTRY_ROOTS])
    lost = {m.rsplit(".", 1)[1] for m in g.mods if m.startswith("engine.learning.") and m not in g.modules_reached()}
    assert {"break_detection", "health", "hierarchy", "questions", "unknowns", "interpretation", "lifecycle", "redundancy",
            "competition", "failed_learners", "portfolio_value", "disagreement", "checkpoints", "compute", "same_year"} <= lost
    assert not g.call_sites("engine.learning.research_priority", "propose_selected")


# ================================================================================================ behaviour on a planted world
from engine.learning import belief as BL                                                          # noqa: E402
from test_learning_learner import make_cfg, mini_spec                                              # noqa: E402

PIN = "pinned-test-code"
RIVAL = "hyp:planted-rival"
PLANT_WEEK = 16


def hook_cfg(**kw) -> LH.HookConfig:
    """Cadences short enough that a 50-week miniature exercises every mechanism several times."""
    base = dict(every_health=2, every_break=4, every_competition=4, every_questions=4, every_interpretation=4, every_redundancy=4,
                every_disagreement=4, every_value=8, every_persist=6, every_kcredit=4, kcredit_min=30, exp_wait=2, exp_max_wait=4,
                health_min_n=12, lifecycle_window=8, boot=40)
    base.update(kw)
    return LH.HookConfig(**base)


def new_learner(workdir, claim: str = "record", **kw) -> LN.LegitimateLearner:
    """learning_claim='record' by default HERE: a one-year miniature can never produce a storable scorecard (the controls need
    forward-year folds), so under the production default 'enforce' nothing is promoted and the decision-side hooks would never
    run. The enforce gate itself is tested separately below."""
    cfg = dataclasses.replace(make_cfg(), hooks=hook_cfg(**kw), learning_claim=claim)
    return LN.LegitimateLearner(cfg, workdir=workdir, code_hash_fn=lambda: PIN)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """The loop on a planted world, with one contradiction planted part-way: an item is said to contradict a rival hypothesis."""
    world = PW.make_world(mini_spec(weeks=50), 3)
    feed = LN.WorldFeed(world)
    L = new_learner(tmp_path_factory.mktemp("loop"))
    LN.learn_from(L, feed, range(PLANT_WEEK))
    assert L._pid_of, f"the miniature must have stored knowledge by week {PLANT_WEEK}"
    paired = {k for pair in L._contradicts for k in pair}
    kid = next((k for k in sorted(L._pid_of) if k not in paired), sorted(L._pid_of)[0])   # a question nobody else asks yet
    at = L.last_learned_on
    L.graph.add_node(RIVAL, KG.NodeType.HYPOTHESIS, at, label="planted rival")
    L.graph.add_edge(kid, RIVAL, Edge.CONTRADICTS, at, weight=0.9)
    LN.learn_from(L, feed, range(PLANT_WEEK, len(world.dates)))
    return L, kid, world


def test_data_flows_through_every_hook_on_the_planted_world(run):
    L, _, _ = run
    fired = L.hooks.fired
    must = ("policy_check", "combined_influence", "audit_decision_sources", "conflict_score", "hierarchy_update", "boundary_to_knowledge",
            "boundary_field", "explain_break", "apply_to_ledger", "inputs_from_knowledge", "epistemic_proposals", "run_period",
            "contradiction_signals", "ask_many", "next_test", "rank_unknowns", "complexity_budget", "disagreement_assess",
            "decompose_value", "kcredit_assess", "masked_pairs", "update_proposals", "signals_from_health", "signals_from_data_audit",
            "check_proposal", "propose_selected", "update_from_result", "with_relation", "persist", "health_dashboard", "readiness",
            "evidence_from_pattern_row", "claim_evidence", "identity_harness")
    silent = [h for h in must if fired.get(h, 0) == 0]
    assert silent == [], f"hooks that never fired on the planted world: {silent}"
    rep = L.report()
    assert rep["hooks"]["label"] == LH.LABEL and rep["learned"] == len(run[2].dates) - 2
    assert L.store.verify() == [] and L.decision_log.verify() == [] and L.beliefs.verify_integrity() == []


def test_a_planted_contradiction_reaches_the_research_queue_and_its_result_closes_it(run):
    L, kid, _ = run
    rival, subject = LH.safe_token(RIVAL), LH.safe_token(kid)
    items = [it for it in L.research.queue.items.values() if rival in it.candidate.question]
    assert items, "the planted contradiction never became a research question"
    it = items[0]
    assert subject in it.candidate.config["subjects"] and it.candidate.config["kind"] == "CONTRADICTION"
    assert it.status == RPR.ItemStatus.DONE, f"the experiment on the planted contradiction never closed ({it.status})"
    done = [r for r in L.hooks.exp_results if r["experiment"] == f"exp-{it.candidate.cid}"]
    assert done and done[0]["outcome"] in ("inconclusive", "supports_noise", "supports_leading")
    rec = L.experiments.get(f"exp-{it.candidate.cid}", pd.Timestamp("2100-01-01"))
    assert rec.result is not None and rec.belief_update is not None
    assert pd.Timestamp(rec.result.observed_at) > pd.Timestamp(rec.created_at)          # prospective: only later outcomes count


def test_a_credit_result_changes_a_belief_and_a_null_one_does_not(tmp_path):
    L = new_learner(tmp_path / "a")
    comps = ("pattern", "analog", "memory", "direction", "timing", "risk")
    eng = CR.CreditEngine(CR.WeightedSumCombiner({c: 1.0 for c in comps}), CR.CreditConfig(n_boot=80, n_perm=150, seed=3))

    def assess(truth, seed):
        led = CR.simulate_decisions(320, seed, truth)
        rows = led.mature(pd.Timestamp("2100-01-01"))[0]
        now = pd.Timestamp(max(d.matured for d in rows)) + pd.Timedelta(days=3)
        return eng.assess(led, now), max(d.matured for d in rows), now

    rep, learned, now = assess(lambda rng, df: 0.8 * df["pattern"].to_numpy(), 11)
    assert rep.component("pattern").verdict == CR.CreditVerdict.EARNS_CREDIT
    applied = L.hooks.apply_credit(rep, (), learned, now)
    st = L.beliefs.current("credit:pattern")
    assert any(a["subject"] == "credit:pattern" and a["action"] == "REINFORCE" for a in applied)
    assert st is not None and st.mean > 0 and st.prob_sign_right() > 0.9
    L2 = new_learner(tmp_path / "b")
    rep0, learned0, now0 = assess(lambda rng, df: np.zeros(len(df)), 12)
    applied0 = L2.hooks.apply_credit(rep0, (), learned0, now0)
    assert not any(a["action"] == "REINFORCE" for a in applied0)                            # noise earns no reinforcement


def _plant_item(L, pid: str, good: int, bad: int, start="2010-01-01", up=0.02, down=-0.03):
    dates = [str((pd.Timestamp(start) + pd.Timedelta(days=7 * i)).date()) for i in range(good + bad)]
    noise = np.random.default_rng(len(pid)).normal(0.0, 0.006, good + bad)                # a real series is never constant
    weekly = [(d, (up if i < good else down) + float(noise[i]), 0.004, 30) for i, d in enumerate(dates)]
    L._weekly[pid] = weekly[:good]                                      # born on the good weeks only: nothing from its future
    L.beliefs.register(pid, 0.0, 0.02)
    for d, e, se, n in weekly[:good]:
        st = L.beliefs.update(pid, BL.Evidence(pid, d, e, se, n), pd.Timestamp(d) + pd.Timedelta(days=1))
    born = dates[good - 1]
    L._birth(pid, st, pd.Timestamp(born) + pd.Timedelta(days=1), born)
    L._weekly[pid] = weekly                                             # then the later weeks arrive, as the loop would add them
    return L._kid_of[pid], pd.Timestamp(dates[-1]) + pd.Timedelta(days=1)


def test_a_break_in_the_outcomes_changes_the_lifecycle_through_the_retirement_gate(tmp_path):
    L = new_learner(tmp_path / "brk")
    kid, now = _plant_item(L, "f9:q4", good=30, bad=20)
    ok_kid, _ = _plant_item(L, "f8:q4", good=50, bad=0)
    assert L.retirement.state(kid, now) == RT.State.ACTIVE
    L.hooks.lifecycle_step(now)
    later = now + pd.Timedelta(days=1)                                  # a transition written at `now` governs decisions after it
    assert L.retirement.state(kid, later) != RT.State.ACTIVE, "a 20-week reversal did not reach the retirement gate"
    ch = [v for v in L.hooks.lifecycle_verdicts if v["kid"] == kid]
    assert ch and ch[-1]["from"] == "ACTIVE" and ch[-1]["stage"] in ("FAILURE", "DECAY")
    assert L.retirement.state(ok_kid, later) == RT.State.ACTIVE and not [v for v in L.hooks.lifecycle_verdicts if v["kid"] == ok_kid]


def test_a_validated_break_condition_becomes_an_anti_context_on_the_situation_paths():
    from engine.learning import break_detection as BK
    from engine.learning import reliability as RL
    ctx, anti = RL.contexts_from_condition(BK.Condition((BK.Term("vix", "<=", 20.0),)))
    works, fails = LH._context_set(ctx), LH._context_set(anti)
    assert works.evaluate({"market.vix": 15.0}) is True and works.evaluate({"market.vix": 30.0}) is False
    assert fails.evaluate({"market.vix": 30.0}) is True and fails.evaluate({"market.vix": 15.0}) is False
    assert LH._context_set({"not_a_column": (1.0, None)}) is None                          # an unknown column claims nothing
    assert LH._context_set({}) is None


def test_the_memory_firewall_reads_the_store_through_as_of_and_catches_a_stale_version(run):
    L, _, _ = run
    kid, now = next((k, pd.Timestamp(v.updated_at) + pd.Timedelta(days=1)) for k in sorted(L._pid_of) for v in L.store.history(k)
                    if pd.Timestamp(L.store.latest(k).updated_at) > pd.Timestamp(v.updated_at))
    latest = L.store.latest(kid)
    assert L.store.as_of(kid, now).version < latest.version                # the version in play did not exist yet at `now`
    found, n = FW.store_findings(L.store, now, FW.LayerName.MEMORY, "planted", [latest])
    assert any(f.check == "not-as-of-version" and f.is_fail for f in found)             # planted: a later version in play
    assert n >= 1 and not any(f.check == "store-future-memory" for f in found)
    v = L.store.as_of(kid, now)                                         # planted leak: a visible version that saw the future
    leak = KN.KnowledgeStore()
    leak.add(dataclasses.replace(v, provenance=dataclasses.replace(v.provenance, outcomes_seen_through=str(now.date()))))
    bad, _ = FW.store_findings(leak, now, FW.LayerName.MEMORY, "leak")
    assert any(f.check == "store-future-memory" and f.is_fail for f in bad)
    clean, _ = FW.store_findings(L.store, now, FW.LayerName.MEMORY, "clean", L.store.visible(now))
    assert not [f for f in clean if f.is_fail]
    with pytest.raises(FirewallBreach):
        FW.LearningFirewallGate().admit(FW.GateContext(now=now, items=[latest], knowledge_store=L.store,
                                                       relevant=frozenset({FW.LayerName.MEMORY})))


def test_only_champions_may_influence_a_tracked_decision(run):
    from types import SimpleNamespace as NS
    L, _, _ = run
    shadow = next((k for k in sorted(L._pid_of) if L.board.members[L._mid[k]].role != Promotion.CHAMPION), None)
    assert shadow is not None
    ep = NS(eid="PLANT", track=True, rows=[NS(parts=[[shadow, 0.5, 0.01, "x"]], decision=NS(slot=0))])
    with pytest.raises(FirewallBreach):
        L.hooks.on_decide(ep)
    L.hooks.on_decide(NS(eid="PROBE", track=False, rows=ep.rows))                           # a dry probe is judged by the contract alone


def test_the_contract_policy_is_applied_and_can_only_tighten(run):
    L, _, _ = run
    prod = L.production_ids()
    k = L.store.latest(prod[0]) if prod else L.store.latest(sorted(L._pid_of)[0])
    now = pd.Timestamp(k.updated_at) + pd.Timedelta(days=1)
    strict = LH.LoopHooks(L, hook_cfg(contract_policy=DC.ContractPolicy(min_usefulness=1.0)), root=None)
    assert strict.contract_allows(k, now) is False
    assert L.hooks.contract_allows(k, now) == DC.check(k, now).allowed                     # default policy == the plain contract
    with pytest.raises(ValueError):                                                           # a broken policy is refused, not ignored
        LH.LoopHooks(L, LH.HookConfig(contract_policy=DC.ContractPolicy(min_usefulness=2.0)), root=None)


def test_what_the_loop_learned_is_on_disk_and_reloads(run):
    from engine.learning import experiment_memory as EM
    L, _, _ = run
    now = pd.Timestamp(L.last_learned_on) + pd.Timedelta(days=1)
    L.hooks.persist(now)
    r = L.workdir / "loop"
    st = KN.KnowledgeStore.load(r / "knowledge.jsonl")
    assert st.verify() == [] and sorted(st.ids()) == sorted(L.store.ids())
    assert BL.BeliefLedger.load_jsonl(r / "beliefs.jsonl").verify_integrity() == []
    assert DC.load_log(json.loads((r / "decision_log.json").read_text(encoding="utf-8"))).verify() == []
    q = RPR.queue_restore((r / "research_queue.json").read_text(encoding="utf-8"))
    assert len(q.items) == len(L.research.queue.items)
    hb = json.loads((r / "health.json").read_text(encoding="utf-8"))
    assert len(hb["records"]) == sum(len(L.hooks.health.book.history(k)) for k in L.hooks.health.book.items()) > 0
    assert len(hb["chain"]) == len(hb["records"]) and L.hooks.health.book.verify() == []
    assert len(EM.ExperimentLedger(r / "experiments.jsonl").view(pd.Timestamp("2100-01-01"))) >= 1
    g = KG.KnowledgeGraph.from_records(json.loads((r / "graph.json").read_text(encoding="utf-8")))
    assert g.has_node(RIVAL)
    dash = json.loads((r / "dashboard.json").read_text(encoding="utf-8"))
    assert dash["section"] == 46 and dash["dashboard_id"] and "contradiction_rows" in dash
    assert json.loads((r / "loop_report.json").read_text(encoding="utf-8"))["fired"]["persist"] >= 1
    assert (r / "failed_learners.jsonl").read_text(encoding="utf-8").count("\n") >= L.hooks.failed_seed["added"]


def test_the_empty_learner_has_nothing_to_signal_and_still_persists(tmp_path):
    L = new_learner(tmp_path / "empty")
    now, learned = pd.Timestamp("2010-01-08"), "2010-01-01"
    assert L.hooks.research_signals(learned, now) == []
    step = L.research.step([], L.experiments, RP.ComputeBudget(cpu_minutes=30.0, ram_gb_free=8.0), now, 0)
    out = L.hooks.close_research(step, learned, now)
    assert out["proposed"] == [] and out["answered"] == [] and out["blocked"] == {}
    L.hooks.persist(now)
    assert (L.workdir / "loop" / "knowledge.jsonl").read_text(encoding="utf-8") == ""
    rep = L.hooks.report()
    assert rep["rows"] == 0 and rep["open_experiments"] == 0 and rep["value"] is None


def test_the_hooks_are_deterministic_given_the_seed(tmp_path):
    world = PW.make_world(mini_spec(weeks=16, stocks=30), 5)
    digests = []
    for i in range(2):
        L = new_learner(tmp_path / f"d{i}")
        LN.learn_from(L, LN.WorldFeed(world), range(len(world.dates)))
        digests.append((L.hooks.digest(), L.trace_digest()))
    assert digests[0] == digests[1]


def test_the_run_guard_resumes_only_same_code_and_writes_the_interruption_record(tmp_path):
    from engine import resources as R
    now = pd.Timestamp("2011-03-04")
    g = LH.RunGuard(tmp_path, "w2008a", "codeA", every=2)
    assert not any(ch.isdigit() for ch in g.run_id)                     # an opaque key: no year-like folder on disk
    assert g.start(now) == {"resumed_from": None, "stale": False}
    job = g.job("test_path", {"config": "c"}, 7, now)
    assert isinstance(job, R.Job) and (tmp_path / "specs" / f"{g.spec.key}.json").exists()
    for i in range(4):
        g.tick(now + pd.Timedelta(days=i), {"days": i})
    assert len(g.store.sequences()) == 3
    p = g.interrupted(now + pd.Timedelta(days=5), "RuntimeError: planted")
    rec = CK.read_interruption(p)
    assert rec.next_action.startswith("resume") and rec.code_hash == "codeA" and rec.current_experiment == g.spec.key
    same = LH.RunGuard(tmp_path, "w2008a", "codeA")
    assert same.start(now)["stale"] is False
    other = LH.RunGuard(tmp_path, "w2008a", "codeB")                    # planted: the code changed since the checkpoints
    assert other.start(now)["stale"] is True and other.plan is None and "codeA" in other.stale


def test_the_same_year_harness_is_a_callable_entry():
    rec = LH.run_same_year_harness(PW.make_world(mini_spec(weeks=30, stocks=30), 7), n_runs=2, seed=0, modes=("fresh_plain",))
    assert rec["label"] == LH.LABEL and "fresh_plain" in rec["modes"] and len(rec["modes"]["fresh_plain"]) == 5
    assert all(len(v["gains"]) == 2 for v in rec["modes"]["fresh_plain"].values())


def test_the_trader_path_still_never_reaches_the_curator():
    from engine.learning.trader_view import assert_trader_path_clean, trader_closure
    assert assert_trader_path_clean() > 50
    assert "engine.learning.loop_hooks" in trader_closure()                # the hooks are on the trader path, and clean


def test_the_production_default_enforces_the_learning_claim_gate(tmp_path):
    """Under learning_claim='enforce' an item with no registered scorecard or identity report cannot pass the eleventh gate."""
    from engine.learning import wiring as W
    L = LN.LegitimateLearner(dataclasses.replace(make_cfg(), hooks=hook_cfg()), workdir=tmp_path / "enf", code_hash_fn=lambda: PIN)
    assert L.cfg.learning_claim == "enforce" and L.board.gate.learning_claim == "enforce"
    kid, now = _plant_item(L, "f7:q4", good=30, bad=0)
    out = L.hooks.claim_evidence(kid, now)
    assert out == {"identity": None, "scorecard": False}                  # nothing to register yet: no panels, no valid card
    v = W.on_knowledge_promotion(L.store.latest(kid), now)
    assert not v.allowed and any("scorecard" in b for b in v.blockers) and any("identity" in b for b in v.blockers)
    with pytest.raises(ValueError):
        dataclasses.replace(make_cfg(), learning_claim="sometimes").validate() and LN.LegitimateLearner(
            dataclasses.replace(make_cfg(), learning_claim="sometimes"), workdir=tmp_path / "bad", code_hash_fn=lambda: PIN)


def test_an_identity_free_rule_passes_the_identity_harness(run):
    L, _, _ = run
    kid = sorted(L._pid_of)[0]
    rep = L.hooks.identity_report(kid)
    assert rep is not None and rep.deterministic and rep.verdicts
    assert rep.passed or rep.base_ic < 0.01                                # a rule on a quantile cell never needs the identities
