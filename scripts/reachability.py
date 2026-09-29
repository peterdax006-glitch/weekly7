"""Honest reachability of the learning brain from the entry points that really run (contract C62 sections 55, 60, 83, 85; canon
C62, C63, C65). IMPLEMENTED - NOT VALIDATED.

wiring.unwired_hooks() only checks that the TEXT `wiring.<fn>(` appears in a file. The 29 Sep integration audit found five of its
eleven hooks in functions that production never calls. This checker replaces that text test with a static call graph:

  1. Every .py under engine/ and scripts/ is parsed (tests and .venv never). Each module gets a body node (what runs on import) and
     one node per function, method and class. Edges are the names each node references - calls, but also functions passed as
     values (callbacks, hook factories) - resolved through the module's imports, including lazy imports inside functions.
  2. Resolution is deliberately generous where Python is dynamic: `self.m()` resolves inside the class (and its named bases); a
     method call on an unknown receiver `x.m()` resolves to every method `m` of every class defined in a module the caller imports
     (or its own module). Generous means REACHED can over-state; UNREACHED never under-states for code reached through attribute
     calls, so an UNREACHED verdict is trustworthy and a REACHED one is an upper bound.
  3. Reachability starts from the body of each entry script. Importing a module runs its body; a function is reached only when a
     reached node references it. An `if __name__ == "__main__":` block runs only in the entry script itself.
  4. C65 (Live is parked): engine/live.py, engine/tick.py and engine/broker.py are never followed from production. A target that is
     reachable only through them is RESEARCH-ONLY with the reason "parked live path".

Statuses, for every engine/learning module and every declared hook:
  REACHED        reached from a PRODUCTION entry (the Test system; PRODUCTION_ENTRIES below, each with its reason)
  RESEARCH-ONLY  reached only from research scripts (named) or only through the parked live path
  UNREACHED      reached from nothing (dead), or the target function does not exist
A module whose body is imported but none of whose functions or classes is referenced is UNREACHED, with the reason "imported, never
called": imported is not the same as called.

  python scripts/reachability.py [--json out.json] [--fail-on hooks|modules|all|none] [--root DIR] [--show research|all]

Exit status 1 when anything in the --fail-on scope is UNREACHED (CI runs --fail-on all). Statically invisible run-time switches
(e.g. livesim_loop2 --learner legit is off by default) are reported as REACHED: static reachability says the code CAN run from a
production entry, not that it did - the persisted sinks under state/learning/hub/ are the evidence that it did."""
from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import re
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("engine", "scripts")
SKIP_PARTS = frozenset({"__pycache__", ".venv", "tests", "node_modules"})
PRODUCTION_ENTRIES: dict[str, str] = {
    "scripts/livesim_loop2.py": "the Test loop: blind windows played by the adaptive system, basis training, --learner legit shadow",
    "scripts/livesim_cycle.py": "the Test cycle on the live clock (sealed hidden year -> examine blind -> adjust -> reveal)",
    "scripts/check_retester.py": "Phase 23 gate: every Test run must be reproduced by the re-tester",
    "scripts/learning_report.py": "the learning reports K01-K15 and the section-46 health dashboard data",
    "scripts/dashboard_build.py": "the one-page status board built from the Test system's state",
    "scripts/research_loop.py": "the C66 trusted-side autonomous research loop (engine.research.loop, every section-3 stage)",
}
RESEARCH_PKG = "engine.research"
PARKED = {"engine.live": "C65", "engine.tick": "C65", "engine.broker": "C65"}      # Live is parked: never followed from production
LEARNING_PKG = "engine.learning"
MAIN_GUARD = "__main__"


# ================================================================================================================== hook table
@dataclasses.dataclass(frozen=True)
class HookSpec:
    """One integration seam. `target` is 'module:qualname'. With `callers`, the hook is REACHED only when a reached node inside one of
    those modules references the target (the call site itself must run); without, the target being reached is enough."""
    hook: str
    source: str                       # who declared it: 'wiring.HOOKS' or the INTEGRATION.md builder id
    target: str
    callers: tuple[str, ...] = ()
    note: str = ""


def _w(fn: str) -> str:
    return f"engine.learning.wiring:{fn}"


# INTEGRATION.md, section "C62 contract wave 1": one row per queued hook, pointing at the function whose execution IS the hook.
INTEGRATION_HOOKS: tuple[HookSpec, ...] = (
    HookSpec("S01.decision_contract.check", "S01", "engine.learning.decision_contract:check", note="promotion/health/archive call it"),
    HookSpec("S01.policy_check", "S01", "engine.learning.decision_contract:policy_check"),
    HookSpec("S01.readiness", "S01", "engine.learning.decision_contract:readiness"),
    HookSpec("S01.as_of", "S01", "engine.learning.knowledge:KnowledgeStore.as_of", ("engine.learning.firewalls",), "firewalls use as_of"),
    HookSpec("S01.audit_future", "S01", "engine.learning.knowledge:audit_future", ("engine.learning.firewalls",), "firewalls use audit_future"),
    HookSpec("S01.with_failure", "S01", "engine.learning.knowledge:with_failure"),
    HookSpec("S01.with_relation", "S01", "engine.learning.knowledge:with_relation"),
    HookSpec("S01.DecisionLog", "S01", "engine.learning.decision_contract:DecisionLog.record_all"),
    HookSpec("S01.next_test", "S01", "engine.learning.interpretation:next_test"),
    HookSpec("S01.rank_unknowns", "S01", "engine.learning.unknowns:rank_unknowns"),
    HookSpec("S05.post_mortem", "S05", "engine.learning.failure:records_from_lessons_frame"),
    HookSpec("S05.lessons", "S05", "engine.learning.failure:hypotheses_from_lessons"),
    HookSpec("S05.missed_week", "S05", "engine.learning.missed_winners:MissedLearningLedger.add_week"),
    HookSpec("S08.to_evidence", "S08", "engine.learning.temporal:to_evidence"),
    HookSpec("S08.retirement", "S08", "engine.learning.retirement:RetirementLedger.evaluate"),
    HookSpec("S08.combined_influence", "S08", "engine.learning.calibration:combined_influence"),
    HookSpec("S08.surprise_priority", "S08", "engine.learning.surprise:SurpriseTracker.research_priority"),
    HookSpec("S11.board_weight", "S11", "engine.learning.champion:KnowledgeBoard.weight"),
    HookSpec("S11.effective_champion", "S11", "engine.learning.champion:effective_champion"),
    HookSpec("S11.audit_decision_sources", "S11", "engine.learning.champion:audit_decision_sources"),
    HookSpec("S11.job_for", "S11", "engine.learning.compute:job_for"),
    HookSpec("S07.evidence_from_pattern_row", "S07", "engine.learning.belief:evidence_from_pattern_row"),
    HookSpec("S07.belief_update", "S07", "engine.learning.belief:BeliefLedger.update"),
    HookSpec("S07.ask_many", "S07", "engine.learning.questions:QuestionEngine.ask_many"),
    HookSpec("S07.to_knowledge", "S07", "engine.learning.boundary:to_knowledge"),
    HookSpec("S07.boundary_field", "S07", "engine.learning.competition:boundary_field"),
    HookSpec("S06.update_proposals", "S06", "engine.learning.credit:update_proposals"),
    HookSpec("S06.credit_edges", "S06", "engine.learning.credit:credit_edges"),
    HookSpec("S06.redundancy_edges", "S06", "engine.learning.redundancy:RedundancyReport.edges"),
    HookSpec("S06.masked_pairs", "S06", "engine.learning.credit:masked_pairs"),
    HookSpec("S09.import_legacy", "S09", "engine.learning.experiment_memory:import_legacy"),
    HookSpec("S09.record_result", "S09", "engine.learning.experiment_memory:ExperimentLedger.record_result"),
    HookSpec("S09.already_tested", "S09", "engine.learning.experiment_memory:ExperimentLedger.already_tested"),
    HookSpec("S09.meta_advice", "S09", "engine.learning.meta_learning:MetaLearner.update"),
    HookSpec("S09.research_step", "S09", "engine.learning.research_priority:ResearchPriorityEngine.step"),
    HookSpec("S09.propose_selected", "S09", "engine.learning.research_priority:propose_selected"),
    HookSpec("S09.update_from_result", "S09", "engine.learning.research_priority:ResearchPriorityEngine.update_from_result"),
    HookSpec("S09.signals_from_health", "S09", "engine.learning.research_priority:signals_from_health"),
    HookSpec("S09.signals_from_data_audit", "S09", "engine.learning.research_priority:signals_from_data_audit"),
    HookSpec("S09.seed_registry", "S09", "engine.learning.failed_learners:seed_registry"),
    HookSpec("S09.check_proposal", "S09", "engine.learning.failed_learners:FailedLearnerRegistry.check_proposal"),
    HookSpec("S10.gate_improvement_claim", "S10", "engine.learning.scorecard:gate_improvement_claim"),
    HookSpec("S10.scorecard_for_learner", "S10", "engine.learning.scorecard:scorecard_for_learner"),
    HookSpec("S10.scorecard_store", "S10", "engine.learning.scorecard:ScorecardStore.append"),
    HookSpec("S10.delta_from_records", "S10", "engine.learning.learning_curve:delta_from_records"),
    HookSpec("S10.curve_from_play_records", "S10", "engine.learning.learning_curve:curve_from_play_records"),
    HookSpec("S11.run_experiment_process", "S11", "engine.learning.compute:run_experiment_process"),
    HookSpec("S11.resume_verified", "S11", "engine.learning.checkpoints:resume_verified"),
    HookSpec("S11.write_interruption", "S11", "engine.learning.checkpoints:write_interruption"),
    HookSpec("S15.apply_to_ledger", "S15", "engine.learning.lifecycle:apply_to_ledger"),
    HookSpec("S15.inputs_from_knowledge", "S15", "engine.learning.health:inputs_from_knowledge"),
    HookSpec("S15.epistemic_proposals", "S15", "engine.learning.health:epistemic_proposals"),
    HookSpec("S15.contexts_from_condition", "S15", "engine.learning.reliability:contexts_from_condition"),
    HookSpec("S19.run_day", "S19", "engine.learning.curator:Curator.run_day"),
    HookSpec("S19.assert_trader_path_clean", "S19", "engine.learning.trader_view:assert_trader_path_clean"),
    HookSpec("S20.run_period", "S20", "engine.learning.contradiction_monitor:ContradictionMonitor.run_period"),
    HookSpec("S20.on_period", "S20", _w("on_period")),
    HookSpec("S20.research_step", "S20", _w("research_step")),
    HookSpec("S20.write_dashboard_inputs", "S20", _w("write_dashboard_inputs")),
    HookSpec("S20.contradiction_keys", "S20", "engine.learning.knowledge_graph:KnowledgeGraph.contradiction_keys"),
)


def wiring_hooks(root: Path) -> tuple[HookSpec, ...]:
    """wiring.HOOKS read from the source (no import: this checker must run on a tree whose imports are broken), each site file becomes
    the caller module that must reach the call."""
    path = root / "engine" / "learning" / "wiring.py"
    if not path.exists():
        return ()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: list[HookSpec] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            tgt = node.targets[0] if isinstance(node, ast.Assign) else node.target
            if isinstance(tgt, ast.Name) and tgt.id == "HOOKS" and isinstance(node.value, ast.Dict):
                for k, v in zip(node.value.keys, node.value.values):
                    if not (isinstance(k, ast.Constant) and isinstance(v, ast.Call) and len(v.args) >= 2):
                        continue
                    fn, sites = v.args[0], v.args[1]
                    if not (isinstance(fn, ast.Constant) and isinstance(sites, ast.Tuple)):
                        continue
                    callers = tuple(module_of(str(s.value)) for s in sites.elts if isinstance(s, ast.Constant))
                    out.append(HookSpec(f"wiring.{k.value}", "wiring.HOOKS", _w(str(fn.value)), callers))
    return tuple(out)


# ================================================================================================================== AST index
def module_of(rel: str) -> str:
    """'engine/learning/wiring.py' -> 'engine.learning.wiring'; a package __init__ is the package itself."""
    p = rel.replace("\\", "/")
    p = p[:-3] if p.endswith(".py") else p
    parts = [x for x in p.split("/") if x]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


@dataclasses.dataclass
class ModuleInfo:
    name: str
    rel: str
    is_pkg: bool
    defs: dict[str, str]                                   # qualname -> kind (function | class | method)
    lines: dict[str, int]
    classes: dict[str, list[str]]                          # class -> base names (as written)
    methods: dict[str, set[str]]                           # method name -> classes defining it
    aliases: dict[str, tuple[str, str]]                    # local name -> (module, symbol or '')
    imported: set[str]                                     # every module this file imports anywhere
    edges: dict[str, set[str]]                             # node qualname ('<module>' = body) -> referenced 'module:qual' or 'mod:<module>'


def _resolve_relative(mod: str, is_pkg: bool, level: int, target: str | None) -> str:
    if level == 0:
        return target or ""
    base = mod.split(".") if is_pkg else mod.split(".")[:-1]
    base = base[: len(base) - (level - 1)] if level > 1 else base
    return ".".join([*base, target] if target else base)


class _Collector(ast.NodeVisitor):
    """Walks one module: records definitions, import aliases, and for each node the names it references."""

    def __init__(self, info: ModuleInfo, known: set[str]):
        self.info, self.known = info, known
        self.stack: list[str] = []                         # current node qualname ('' = module body)
        self.cls: list[str] = []
        self.raw: dict[str, list[tuple[str, ...]]] = defaultdict(list)     # node -> dotted reference chains
        self.lazy: dict[str, set[str]] = defaultdict(set)  # node -> modules imported inside it

    @property
    def node(self) -> str:
        return self.stack[-1] if self.stack else "<module>"

    # ---- imports
    def _add_import(self, local: str, module: str, symbol: str = "") -> None:
        self.info.aliases.setdefault(local, (module, symbol))
        mod = module if not symbol or f"{module}.{symbol}" not in self.known else f"{module}.{symbol}"
        self.info.imported.add(mod)
        self.lazy[self.node].add(mod)

    def visit_Import(self, n: ast.Import) -> None:
        for a in n.names:
            if a.asname:
                self._add_import(a.asname, a.name)
            else:
                self._add_import(a.name.split(".")[0], a.name.split(".")[0])
                self.info.imported.add(a.name)
                self.lazy[self.node].add(a.name)

    def visit_ImportFrom(self, n: ast.ImportFrom) -> None:
        base = _resolve_relative(self.info.name, self.info.is_pkg, n.level or 0, n.module)
        for a in n.names:
            if a.name == "*":
                continue
            self._add_import(a.asname or a.name, base, a.name)

    # ---- definitions
    def _enter(self, name: str, kind: str, lineno: int) -> str:
        q = ".".join([*self.cls, name])                    # 'f', 'Cls', 'Cls.m', 'Outer.Inner.m'
        self.info.defs.setdefault(q, kind)
        self.info.lines.setdefault(q, lineno)
        return q

    def _visit_func(self, n: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for d in n.decorator_list:                         # decorators run where the def is
            self.visit(d)
        for d in n.args.defaults + [x for x in n.args.kw_defaults if x is not None]:
            self.visit(d)
        in_class_body = bool(self.cls) and bool(self.stack) and self.stack[-1] == ".".join(self.cls)
        if self.stack and not in_class_body:
            for s in n.body:                               # a nested function is part of its enclosing node
                self.visit(s)
            return
        kind = "method" if in_class_body else "function"
        q = self._enter(n.name, kind, n.lineno)
        if kind == "method":
            self.info.methods.setdefault(n.name, set()).add(".".join(self.cls))
        self.stack.append(q)
        for s in n.body:
            self.visit(s)
        self.stack.pop()

    visit_FunctionDef = _visit_func
    visit_AsyncFunctionDef = _visit_func

    def visit_ClassDef(self, n: ast.ClassDef) -> None:
        for d in n.decorator_list:
            self.visit(d)
        q = self._enter(n.name, "class", n.lineno)
        self.info.classes[q] = [_dotted(b) or "" for b in n.bases]
        self.cls.append(n.name)
        self.stack.append(q)                               # class-body statements run on import; attributed to the class node
        for d in n.bases + [k.value for k in n.keywords]:
            self.visit(d)                                  # reaching a class reaches its bases (their methods dispatch too)
        for s in n.body:
            self.visit(s)
        self.stack.pop()
        self.cls.pop()

    def visit_If(self, n: ast.If) -> None:
        if not self.stack and _is_main_guard(n.test):      # runs only when this file is the entry script
            self.stack.append(MAIN_GUARD)
            for s in n.body:
                self.visit(s)
            self.stack.pop()
            for s in n.orelse:
                self.visit(s)
            return
        self.generic_visit(n)

    # ---- references
    def visit_Call(self, n: ast.Call) -> None:
        f = n.func
        dyn = (isinstance(f, ast.Name) and f.id == "__import__") or (isinstance(f, ast.Attribute) and f.attr == "import_module")
        if dyn and n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
            mod = n.args[0].value                          # __import__("engine.improve").weekly(): a dynamic import is still an import
            self.info.imported.add(mod)
            self.lazy[self.node].add(mod)
        self.generic_visit(n)

    def visit_Name(self, n: ast.Name) -> None:
        self.raw[self.node].append((n.id,))

    def visit_Attribute(self, n: ast.Attribute) -> None:
        chain = _chain(n)
        if chain is not None:
            self.raw[self.node].append(chain)
        else:
            self.raw[self.node].append(("?", n.attr))      # receiver is an expression: resolved by method name
            self.visit(n.value)


def _is_main_guard(t: ast.expr) -> bool:
    return (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name) and t.left.id == "__name__"
            and any(isinstance(c, ast.Constant) and c.value == MAIN_GUARD for c in t.comparators))


def _chain(n: ast.expr) -> tuple[str, ...] | None:
    parts: list[str] = []
    while isinstance(n, ast.Attribute):
        parts.append(n.attr)
        n = n.value
    if isinstance(n, ast.Name):
        return tuple([n.id, *reversed(parts)])
    return None


def _dotted(n: ast.expr) -> str | None:
    c = _chain(n)
    return ".".join(c) if c else None


def scan(root: Path, dirs: Sequence[str] = SCAN_DIRS) -> dict[str, ModuleInfo]:
    files: list[Path] = []
    for d in dirs:
        base = root / d
        if base.exists():
            files += [p for p in base.rglob("*.py") if not (SKIP_PARTS & set(p.relative_to(root).parts))]
    known = {module_of(p.relative_to(root).as_posix()) for p in files}
    mods: dict[str, ModuleInfo] = {}
    for p in sorted(files):
        rel = p.relative_to(root).as_posix()
        name = module_of(rel)
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue                                       # a file that does not parse runs nothing
        info = ModuleInfo(name, rel, p.name == "__init__.py", {}, {}, {}, {}, {}, set(), {})
        col = _Collector(info, known)
        col.visit(tree)
        info.imported = {m for m in info.imported if m}
        _resolve(info, col, known)
        mods[name] = info
    return mods


def _resolve(info: ModuleInfo, col: _Collector, known: set[str]) -> None:
    """Turn raw reference chains into edges 'module:qual' (a def) or 'module:<module>' (an import that runs a module body)."""
    for node, chains in col.raw.items():
        out = info.edges.setdefault(node, set())
        kind = info.defs.get(node)
        cls = node.rsplit(".", 1)[0] if kind == "method" else node if kind == "class" else None
        for ch in chains:
            out.update(_resolve_chain(info, ch, cls, known))
    for node, mods_ in col.lazy.items():
        info.edges.setdefault(node, set()).update(f"{m}:<module>" for m in mods_ if m in known)
    for node in list(info.defs):
        info.edges.setdefault(node, set())


def _resolve_chain(info: ModuleInfo, ch: tuple[str, ...], cls: str | None, known: set[str]) -> set[str]:
    head, rest = ch[0], ch[1:]
    if head == "?":
        return {f"?{rest[0]}"}                             # method by name, resolved after every module is indexed
    if head in ("self", "cls") and cls and rest:
        return {f"{info.name}:{cls}.{rest[0]}", f"?{rest[0]}", f"?{rest[-1]}"}
    if head in info.defs and info.defs[head] in ("function", "class"):
        tgt = {f"{info.name}:{head}"}
        if rest and info.defs.get(head) == "class":
            tgt.add(f"{info.name}:{head}.{rest[0]}")
        elif rest:
            tgt.add(f"?{rest[-1]}")
        return tgt
    if head in info.aliases:
        mod, sym = info.aliases[head]
        if sym:
            full = f"{mod}.{sym}"
            if full in known:                              # 'from pkg import module'
                return _in_module(full, rest, known)
            tgt = {f"{mod}:{sym}"}
            if rest:
                tgt |= {f"{mod}:{sym}.{rest[0]}", f"?{rest[-1]}"}
            return tgt
        dotted = ".".join([mod, *rest])                   # 'import a.b.c' then a.b.c.f
        for cut in range(len(rest), -1, -1):
            m = ".".join([mod, *rest[:cut]])
            if m in known:
                return _in_module(m, rest[cut:], known)
        return {f"?{rest[-1]}"} if rest and dotted else set()
    return {f"?{rest[-1]}"} if rest else set()


def _in_module(mod: str, rest: tuple[str, ...], known: set[str]) -> set[str]:
    if not rest:
        return {f"{mod}:<module>"}
    out = {f"{mod}:{rest[0]}"}
    if len(rest) > 1:
        out |= {f"{mod}:{rest[0]}.{rest[1]}", f"?{rest[-1]}"}
    return out


# ================================================================================================================== graph walk
@dataclasses.dataclass
class Walk:
    reached: set[str]                                      # 'module:qual' and 'module:<module>'
    via: dict[str, str]                                    # node -> the node that first reached it (for explanations)
    parked_hits: set[str]


def walk(mods: Mapping[str, ModuleInfo], entries: Iterable[str], follow_parked: bool = False,
         cache: dict[str, set[str]] | None = None) -> Walk:
    """BFS over the call graph from each entry module's body (its __main__ block included). `cache` memoises each node's statically
    resolved edges across walks. Dynamic dispatch is rapid-type-analysis style: a method call on an unknown receiver (`x.m()`) also
    reaches method `m` of every class this walk has already reached (constructed or referenced) - and of any class reached later -
    so a callback object handed across modules (a hook factory, a trader) is followed."""
    memo = cache if cache is not None else {}
    reached: set[str] = set()
    via: dict[str, str] = {}
    parked: set[str] = set()
    names_seen: set[str] = set()
    by_name: dict[str, set[str]] = defaultdict(set)       # method name -> method nodes of classes reached so far
    q: deque[str] = deque()

    def reach(tgt: str, src: str) -> None:
        tmod, tq = tgt.split(":", 1)
        if tmod in PARKED and not follow_parked:
            parked.add(tgt)
            return
        tinfo = mods.get(tmod)
        if tgt in reached or tinfo is None or (tq not in ("<module>", MAIN_GUARD) and tq not in tinfo.defs):
            return
        reached.add(tgt)
        via[tgt] = src
        q.append(tgt)
        if tq in ("<module>", MAIN_GUARD):
            return
        reach(f"{tmod}:<module>", src)                    # using a def means its module was imported
        if tinfo.defs.get(tq) == "class":
            for mq, kind in tinfo.defs.items():
                if kind != "method" or mq.rsplit(".", 1)[0] != tq:
                    continue
                name = mq.rsplit(".", 1)[1]
                node = f"{tmod}:{mq}"
                by_name[name].add(node)
                if name in names_seen or (name.startswith("__") and name.endswith("__")):
                    reach(node, tgt)                       # dunders run implicitly; a name already called dispatches here

    for e in entries:
        if e in mods:
            for n in (f"{e}:<module>", f"{e}:{MAIN_GUARD}"):
                reach(n, "<entry>")
    while q:
        node = q.popleft()
        mod, qual = node.split(":", 1)
        info = mods.get(mod)
        if info is None:
            continue
        if node not in memo:
            memo[node] = _expand(mods, info, qual)
        for tgt in memo[node]:
            reach(tgt, node)
        for name in _dyn_names(info, qual):
            if name not in names_seen:
                names_seen.add(name)
                for m in sorted(by_name.get(name, ())):
                    reach(m, node)
            else:
                for m in sorted(by_name.get(name, ())):
                    if m not in reached:
                        reach(m, node)
    return Walk(reached, via, parked)


def _dyn_names(info: ModuleInfo, qual: str) -> set[str]:
    return {r[1:] for r in info.edges.get(qual, ()) if r.startswith("?")}


def _expand(mods: Mapping[str, ModuleInfo], info: ModuleInfo, qual: str) -> set[str]:
    raw = info.edges.get(qual, set())
    out: set[str] = set()
    scope = {info.name} | info.imported
    for r in raw:
        if r.startswith("?"):
            name = r[1:]
            for m in scope:
                mi = mods.get(m)
                if mi is not None:
                    out |= {f"{m}:{c}.{name}" for c in mi.methods.get(name, ())}
                    if mi.defs.get(name) in ("class", "function"):     # a module held in an attribute: self.A.Session(...)
                        out.add(f"{m}:{name}")
        else:
            out.add(r)
    return out


# ================================================================================================================== verdicts
@dataclasses.dataclass(frozen=True)
class Verdict:
    kind: str                                              # module | hook
    name: str
    status: str                                            # REACHED | RESEARCH-ONLY | UNREACHED
    reason: str
    source: str = ""


def research_entries(mods: Mapping[str, ModuleInfo], production: Iterable[str]) -> list[str]:
    prod = set(production)
    return sorted(m for m, i in mods.items() if i.rel.startswith("scripts/") and m not in prod
                  and (i.edges.get(MAIN_GUARD) or i.edges.get("<module>")))


def _path(w: Walk, node: str, limit: int = 6) -> str:
    chain = [node]
    while chain[-1] in w.via and w.via[chain[-1]] != "<entry>" and len(chain) < 40:
        chain.append(w.via[chain[-1]])
    chain.reverse()
    short = chain if len(chain) <= limit else chain[:2] + ["..."] + chain[-(limit - 3):]
    return " -> ".join(short)


class Checker:
    def __init__(self, root: Path = ROOT, production: Mapping[str, str] | None = None,
                 hooks: Sequence[HookSpec] | None = None):
        self.root = Path(root)
        self.mods = scan(self.root)
        prod = production if production is not None else PRODUCTION_ENTRIES
        self.production = {module_of(k): v for k, v in prod.items() if module_of(k) in self.mods}
        self.missing_entries = sorted(k for k in prod if module_of(k) not in self.mods)
        self.hooks = tuple(hooks) if hooks is not None else wiring_hooks(self.root) + INTEGRATION_HOOKS
        self.cache: dict[str, set[str]] = {}
        self.prod = walk(self.mods, self.production, cache=self.cache)
        self.parked_walk = walk(self.mods, [*self.production, *PARKED], follow_parked=True, cache=self.cache)
        self.research = {e: walk(self.mods, [e], cache=self.cache) for e in research_entries(self.mods, self.production)}

    def _status(self, test) -> tuple[str, str]:
        if test(self.prod):
            return "REACHED", ""
        who = sorted(e for e, w in self.research.items() if test(w))
        if who:
            return "RESEARCH-ONLY", "only from research scripts: " + ", ".join(self.mods[e].rel for e in who[:4]) + (
                f" (+{len(who) - 4} more)" if len(who) > 4 else "")
        if test(self.parked_walk):
            return "RESEARCH-ONLY", "only through the parked live path (C65: engine.live / engine.tick)"
        return "UNREACHED", ""

    def module_verdicts(self, package: str = LEARNING_PKG) -> list[Verdict]:
        out = []
        for m in sorted(x for x in self.mods if x.startswith(package + ".")):
            info = self.mods[m]
            defs = [f"{m}:{q}" for q in info.defs]

            def used(w: Walk, defs=defs) -> bool:
                return any(d in w.reached for d in defs)
            st, why = self._status(used)
            if st == "REACHED":
                n = sum(d in self.prod.reached for d in defs)
                why = f"{n}/{len(defs)} definitions reached"
            elif st == "UNREACHED":
                why = "imported, never called" if f"{m}:<module>" in self.prod.reached else "never imported by any entry"
            out.append(Verdict("module", m, st, why))
        return out

    def hook_verdicts(self) -> list[Verdict]:
        out = []
        for h in self.hooks:
            tmod, tq = h.target.split(":", 1)
            tinfo = self.mods.get(tmod)
            if tinfo is None or tq not in tinfo.defs:
                out.append(Verdict("hook", h.hook, "UNREACHED", f"target {h.target} is not defined anywhere", h.source))
                continue
            if h.callers:
                def test(w: Walk, h=h) -> bool:
                    return any(n in w.reached and h.target in self._edges(n) for c in h.callers if c in self.mods for n in self._nodes(c))
            else:
                def test(w: Walk, h=h) -> bool:
                    return h.target in w.reached
            st, why = self._status(test)
            if st == "REACHED":
                why = _path(self.prod, h.target)
            elif st == "UNREACHED" and h.callers:
                why = f"no reached function in {', '.join(h.callers)} calls {h.target.split(':')[1]}"
            out.append(Verdict("hook", h.hook, st, why, h.source))
        return out

    def _edges(self, node: str) -> set[str]:
        if node not in self.cache:
            mod, qual = node.split(":", 1)
            self.cache[node] = _expand(self.mods, self.mods[mod], qual)
        return self.cache[node]

    def _nodes(self, mod: str) -> list[str]:
        info = self.mods[mod]
        return [f"{mod}:<module>", f"{mod}:{MAIN_GUARD}"] + [f"{mod}:{q}" for q in info.defs]

    def integration_coverage(self, path: Path | None = None) -> list[str]:
        """Builder ids named in INTEGRATION.md's C62 section that have no row in the hook table: an unmapped hook cannot be checked."""
        p = path or self.root / "state" / "build" / "INTEGRATION.md"
        if not p.exists():
            return []
        text = p.read_text(encoding="utf-8")
        sec = text.split("## C62", 1)[1] if "## C62" in text else ""
        ids = sorted(set(re.findall(r"^- (S\d\d)\b", sec, flags=re.M)))
        have = {h.source for h in self.hooks}
        return [i for i in ids if i not in have and i != "S14"]         # S14 declared "no hook needed"

    def report(self, packages: Sequence[str] = (LEARNING_PKG,)) -> dict:
        mv = [v for pkg in packages for v in self.module_verdicts(pkg)]
        hv = self.hook_verdicts()
        count = lambda vs: {s: sum(v.status == s for v in vs) for s in ("REACHED", "RESEARCH-ONLY", "UNREACHED")}
        return {"production_entries": {self.mods[m].rel: why for m, why in sorted(self.production.items())},
                "missing_entries": self.missing_entries, "parked": sorted(PARKED),
                "modules": [dataclasses.asdict(v) for v in mv], "hooks": [dataclasses.asdict(v) for v in hv],
                "module_counts": count(mv), "hook_counts": count(hv), "unmapped_integration_ids": self.integration_coverage(),
                "n_files": len(self.mods), "n_research_entries": len(self.research)}


def render(rep: Mapping, show: str = "all") -> str:
    L = [f"reachability over {rep['n_files']} files; production entries: {', '.join(rep['production_entries'])}; "
         f"{rep['n_research_entries']} research entries; parked (C65): {', '.join(rep['parked'])}"]
    if rep["missing_entries"]:
        L.append(f"MISSING production entries: {rep['missing_entries']}")
    for kind in ("hooks", "modules"):
        c = rep[kind[:-1] + "_counts"]
        L.append(f"\n{kind}: {c['REACHED']} REACHED, {c['RESEARCH-ONLY']} RESEARCH-ONLY, {c['UNREACHED']} UNREACHED")
        for v in rep[kind]:
            if show == "all" or v["status"] != "REACHED":
                L.append(f"  {v['status']:<13} {v['name']:<44} {v['reason'][:150]}")
    if rep["unmapped_integration_ids"]:
        L.append(f"\nINTEGRATION.md builder ids with no hook row: {rep['unmapped_integration_ids']}")
    return "\n".join(L)


def failing(rep: Mapping, scope: str) -> list[str]:
    kinds = {"hooks": ("hooks",), "modules": ("modules",), "all": ("hooks", "modules"), "none": ()}[scope]
    bad = [f"{k[:-1]} {v['name']}" for k in kinds for v in rep[k] if v["status"] == "UNREACHED"]
    if scope != "none":
        bad += [f"production entry missing: {e}" for e in rep["missing_entries"]]
        bad += [f"INTEGRATION.md id with no hook row: {i}" for i in rep["unmapped_integration_ids"]]
    return bad


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--json", default="")
    ap.add_argument("--fail-on", default="all", choices=("hooks", "modules", "all", "none"))
    ap.add_argument("--show", default="research", choices=("research", "all"))
    ap.add_argument("--package", action="append", default=None,
                    help=f"module package(s) to judge (default {LEARNING_PKG}; add {RESEARCH_PKG} for the C66 research brain)")
    a = ap.parse_args(argv)
    rep = Checker(Path(a.root)).report(tuple(a.package or (LEARNING_PKG,)))
    print(render(rep, "all" if a.show == "all" else "not-reached"))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_bytes(json.dumps(rep, indent=1, sort_keys=True).encode("utf-8"))
    bad = failing(rep, a.fail_on)
    if bad:
        print(f"\nFAIL ({a.fail_on}): {len(bad)} unreached: " + "; ".join(bad[:12]) + (" ..." if len(bad) > 12 else ""))
        return 1
    print("\nOK: nothing in scope is UNREACHED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
