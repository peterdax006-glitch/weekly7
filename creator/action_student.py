"""Nupen's model student v2: the small local model CHOOSES among exact program transformations instead of writing edits.

v1 (model_student.py) asked the 1.5B model to write search/replace edits; it echoed the file or made no-op edits. A small model is
far better at choosing than writing, so here the system enumerates every VALID instantiation of a fixed action space deterministically
(lazy_import, remove_unused, inline_temp, drop_unused_import, add_empty_guard - the appliers are creator.student's rules, one source
of truth), shows the model the objective, a few adopted lessons as (objective, chosen actions) and the numbered candidates, and the
model replies with numbers. Names are never invented by the model; an applied action is exact, parse-checked and reverted on failure.
`export_choices` maps adopted lessons onto the action space: the supervised signal a later fine-tune / classifier can learn from.
The kernel's measurement decides adoption; nothing here judges itself."""
from __future__ import annotations

import ast
import dataclasses
import difflib
import json
import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from creator import generator as G
from creator import model_student as MS
from creator import student as ST
from creator.curriculum import Lesson, LessonLog

if TYPE_CHECKING:
    from creator.kernel import WorkResult

SYSTEM = ("You choose code transformations. Read the task, then pick the numbered actions that do exactly what the task asks. "
          "Reply with exactly two lines. Line 1: 'CHOICE: <numbers separated by commas>' (only numbers from the list; "
          "'CHOICE: none' only if no listed action fits). Line 2: 'WHY: <one short sentence>'.\n"
          "Example: for the task 'inline the temp x in f' with actions '1. inline_temp x in f' and '2. remove_unused g', reply:\n"
          "CHOICE: 1\nWHY: inlining x in f is what was asked.")
KINDS = ("lazy_import", "remove_unused", "inline_temp", "drop_unused_import", "add_empty_guard")
Elsewhere = Optional[Callable[[str], bool]]
CHOOSER_DEFAULT = False                  # flipped only when the held-out measurement shows the chooser beats model+lexical (see chooser_bench)


def _state_dir() -> Path:
    from creator import shadow
    return shadow.STATE


def _policy(state: Path) -> bool:
    """state/creator/policy.json (written by creator.shadow's pre-registered rule) overrides CHOOSER_DEFAULT."""
    try:
        from creator import shadow
        v = shadow.read_policy(state)
    except Exception:                                       # noqa: BLE001
        v = None
    return CHOOSER_DEFAULT if v is None else v


@dataclasses.dataclass(frozen=True)
class Action:
    kind: str
    params: tuple[tuple[str, str], ...]              # sorted (key, value) pairs: hashable, JSON-friendly
    path: str = ""

    @property
    def p(self) -> dict[str, str]:
        return dict(self.params)

    def label(self, rich: bool = False) -> str:
        p = self.p
        if rich:
            if self.kind == "lazy_import":
                return (f"lazy_import: move the import of '{p['alias']}' from the top of the file into the function(s) that use it "
                        f"({p.get('into', '')}); '{p['alias']}' stays in use, it is only loaded later")
            if self.kind == "remove_unused":
                return f"remove_unused: delete the whole function '{p['name']}' (dead code: no code calls or references it)"
            if self.kind == "inline_temp":
                return (f"inline_temp: in function '{p['function']}' replace the temporary variable '{p['variable']}' "
                        f"by its expression and delete the assignment")
            if self.kind == "drop_unused_import":
                return f"drop_unused_import: delete the import of '{p['name']}' (the file never uses '{p['name']}')"
        if self.kind == "lazy_import":
            return f"lazy_import: move top-level import of '{p['alias']}' into the function(s) that use it ({p.get('into', '')})"
        if self.kind == "remove_unused":
            return f"remove_unused: delete function '{p['name']}' (nothing references it)"
        if self.kind == "inline_temp":
            return f"inline_temp: in '{p['function']}' inline the temporary variable '{p['variable']}'"
        if self.kind == "drop_unused_import":
            return f"drop_unused_import: delete the unused import '{p['name']}'"
        return f"add_empty_guard: in '{p['function']}' return {p['return_value']} early when '{p['param']}' is empty"

    def short(self) -> str:
        p = self.p
        key = {"lazy_import": "alias", "remove_unused": "name", "drop_unused_import": "name"}.get(self.kind)
        return f"{self.kind}({p[key]})" if key else f"{self.kind}({p.get('function')},{p.get('variable') or p.get('param')})"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "params": self.p, "path": self.path}


def _act(kind: str, path: str = "", **kw: str) -> Action:
    return Action(kind, tuple(sorted(kw.items())), path)


# ------------------------------------------------------------------------------------------------ appliers (shared with student.py)

def rule_drop_unused_import(src: str, params: dict[str, Any]) -> Optional[str]:
    """Remove one imported name that nothing in the file references (no string mention, not a re-export module)."""
    tree = ST._parse(src)
    if tree is None:
        return None
    want = params.get("name")
    strings = {c.value for c in ast.walk(tree) if isinstance(c, ast.Constant) and isinstance(c.value, str)}
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    lines = src.split("\n")
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)) or (isinstance(node, ast.ImportFrom) and (node.module == "__future__" or node.level)):
            continue
        if any(a.name == "*" for a in node.names) or "noqa" in lines[node.lineno - 1]:
            continue
        for a in node.names:
            b = ST._bound(a)
            if (want and b != want) or b in used or b in strings or any(b in s.split() for s in strings if len(s) < 40):
                continue
            keep = [x for x in node.names if x is not a]
            start, end = node.lineno, node.end_lineno or node.lineno
            new = [] if not keep else [" " * node.col_offset + ast.unparse(
                ast.Import(names=keep) if isinstance(node, ast.Import) else ast.ImportFrom(module=node.module, names=keep, level=0))]
            out = "\n".join(lines[:start - 1] + new + lines[end:])
            return out if ST._parse(out) is not None else None
    return None


def _run_params(a: Action) -> Action:
    """The action with only the keys the shared rule understands (alias/name/function+variable/function)."""
    keys = {"lazy_import": ("alias",), "remove_unused": ("name",), "inline_temp": ("function", "variable"),
            "drop_unused_import": ("name",), "add_empty_guard": ("function",)}[a.kind]
    return Action(a.kind, tuple((k, v) for k, v in a.params if k in keys), a.path)


def apply_action(src: str, act: Action, elsewhere: Elsewhere = None) -> Optional[str]:
    """The source after `act`, or None when it no longer applies / would not parse / changes nothing."""
    act = _run_params(act)
    p: dict[str, Any] = act.p
    if act.kind == "lazy_import":
        new = ST.rule_lazy_import(src, p)
    elif act.kind == "remove_unused":
        new = ST.rule_remove_unused(src, p, elsewhere)
    elif act.kind == "inline_temp":
        new = ST.rule_inline_temp(src, p)
    elif act.kind == "drop_unused_import":
        new = rule_drop_unused_import(src, p)
    elif act.kind == "add_empty_guard":
        new = ST.rule_empty_guard(src, p)
    else:
        return None
    return new if new is not None and new != src and ST._parse(new) is not None else None


# ------------------------------------------------------------------------------------------------ candidate enumeration

def enumerate_actions(src: str, path: str = "", elsewhere: Elsewhere = None) -> list[Action]:
    """Every VALID instantiation for `src` (each is applied to the source to prove it: names come from the AST, never the model)."""
    tree = ST._parse(src)
    if tree is None:
        return []
    out: list[Action] = []
    seen: set[Action] = set()

    def add(a: Action) -> None:
        if a not in seen and apply_action(src, a, elsewhere) is not None:
            seen.add(a)
            out.append(a)

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for al in node.names:
                b = ST._bound(al)
                _, users = ST._scopes(tree, {b})
                into = ",".join(sorted(getattr(f, "name", "?") for f in users.values()))
                cand = _act("lazy_import", path, alias=b, into=into)
                if apply_action(src, _act("lazy_import", path, alias=b), elsewhere) is not None and cand not in seen:
                    seen.add(cand)
                    out.append(cand)
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for al in node.names:
                add(_act("drop_unused_import", path, name=ST._bound(al)))
        if isinstance(node, ST.FuncT):
            add(_act("remove_unused", path, name=node.name))
    for fn in ast.walk(tree):
        if not isinstance(fn, ST.FuncT):
            continue
        for st in fn.body:
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                add(_act("inline_temp", path, function=fn.name, variable=st.targets[0].id))
        if isinstance(fn, ast.FunctionDef):
            body = fn.body[1:] if fn.body and ST._is_doc(fn.body[0]) else fn.body
            if len(body) == 2 and isinstance(body[0], ast.For) and isinstance(body[0].iter, ast.Name) and isinstance(body[1], ast.Return) \
                    and isinstance(body[1].value, ast.Constant):
                a = _act("add_empty_guard", path, function=fn.name, param=body[0].iter.id, return_value=repr(body[1].value.value))
                if apply_action(src, a, elsewhere) is not None:
                    seen.add(a)
                    out.append(a)
    return out


# ------------------------------------------------------------------------------------------------ lessons -> actions

def _norm(src: str) -> Optional[str]:
    t = ST._parse(src)
    return ast.dump(t) if t is not None else None


def _distance(a: str, b: str) -> int:
    return sum(1 for x in difflib.ndiff(a.split("\n"), b.split("\n")) if x[0] in "+-")


def explain(before: str, after: str, path: str = "", elsewhere: Elsewhere = None, max_steps: int = 4) -> tuple[list[Action], bool]:
    """Which candidate actions (greedily, each reducing the distance to `after`) reproduce the teacher's change; exact = AST-equal."""
    target, cur, chosen = _norm(after), before, []
    if target is None or _norm(before) == target:
        return [], False
    for _ in range(max_steps):
        best: Optional[tuple[int, Action, str]] = None
        base = _distance(cur, after)
        for a in enumerate_actions(cur, path, elsewhere):
            new = apply_action(cur, a, elsewhere)
            if new is None:
                continue
            d = _distance(new, after)
            if d < base and (best is None or d < best[0]):
                best = (d, a, new)
        if best is None:
            break
        chosen.append(best[1])
        cur = best[2]
        if _norm(cur) == target:
            return chosen, True
    return chosen, False


def lesson_actions(les: Lesson) -> list[Action]:
    out: list[Action] = []
    for p, after in les.files_after.items():
        if p.endswith(".py") and les.files_before.get(p):
            out += explain(les.files_before[p], after, p)[0]
    return out


def export_choices(lessons_path: Path, out_jsonl: Path, adopted_only: bool = True) -> dict[str, int]:
    """Map every adopted lesson's change onto the action space; one JSON line per (lesson, file) with the numbered candidates and the
    indices that reproduce the teacher's change. Returns counts: lessons, with_python, mapped (>= 1 action), exact (AST-equal)."""
    rows: list[dict[str, Any]] = []
    stats = {"lessons": 0, "with_python": 0, "mapped": 0, "exact": 0, "rows": 0}
    for les in LessonLog(Path(lessons_path)).lessons():
        if les.adopted is not True and (adopted_only or les.adopted is False):
            continue
        stats["lessons"] += 1
        files = [p for p, a in les.files_after.items() if p.endswith(".py") and les.files_before.get(p)]
        if not files:
            continue
        stats["with_python"] += 1
        any_mapped, all_exact = False, True
        for p in files:
            cands = enumerate_actions(les.files_before[p], p)
            chosen, exact = explain(les.files_before[p], les.files_after[p], p)
            keyed = {a: i for i, a in enumerate(cands, 1)}
            idx = [keyed[a] for a in chosen if a in keyed]
            any_mapped |= bool(chosen)
            all_exact &= exact
            rows.append({"lesson_id": les.lesson_id, "task_kind": les.task_kind, "objective": les.objective, "path": p,
                         "candidates": [a.to_dict() | {"label": a.label()} for a in cands], "chosen": idx, "exact": exact})
        stats["mapped"] += int(any_mapped)
        stats["exact"] += int(any_mapped and all_exact)
    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats["rows"] = len(rows)
    return stats


# ------------------------------------------------------------------------------------------------ the student

_KIND_WORDS = {"lazy_import": {"defer", "lazy", "lazily", "later", "startup", "start", "load"},
               "remove_unused": {"remove", "dead", "calls", "nothing", "helper", "function"},
               "inline_temp": {"inline", "temporary", "temp", "single", "variable"},
               "drop_unused_import": {"unused", "import", "delete", "never"},
               "add_empty_guard": {"empty", "guard", "early"}}


def _words(text: str) -> set[str]:
    return {w for w in re.split(r"[^a-z0-9]+", text.lower()) if w}


def lexical_scores(objective: str, cands: list[Action]) -> list[float]:
    """Non-LLM baseline: 3 points per candidate name (alias/function/variable/param) found whole in the objective,
    plus 0.5 per generic kind word present. Names come from the candidate; kind words are task-agnostic synonyms."""
    ow = _words(objective.replace("_", " ")) | _words(objective)
    out = []
    for a in cands:
        sc = 0.0
        names = {v for k, v in a.params if k in ("alias", "name", "function", "variable", "param")}
        for n in names:
            nw = _words(n.replace("_", " "))
            sc_n = 3.0 if nw and nw <= ow else 0.0
            sc += sc_n or 1.0 * len(nw & ow)
        sc += 0.5 * len(_KIND_WORDS.get(a.kind, set()) & ow)
        out.append(sc)
    return out


def lexical_pick(objective: str, cands: list[Action]) -> Optional[int]:
    """1-based index of the unique best lexical candidate, else None (ties are not guessed)."""
    sc = lexical_scores(objective, cands)
    if not sc or max(sc) <= 0 or sc.count(max(sc)) != 1:
        return None
    return sc.index(max(sc)) + 1


def combine_choice(model_pick: Optional[int], objective: str, cands: list[Action], margin: float = 3.0) -> Optional[int]:
    """Lexical prior + model: the model's pick stands unless a unique lexical candidate beats it by >= `margin` points
    (or the model gave none)."""
    sc = lexical_scores(objective, cands)
    lex = lexical_pick(objective, cands)
    if model_pick is None:
        return lex
    if lex is not None and lex != model_pick and sc[lex - 1] - sc[model_pick - 1] >= margin:
        return lex
    return model_pick


def majority_vote(picks: list[Optional[int]]) -> Optional[int]:
    """Most common non-None pick; ties go to the earliest seen. None when nobody picked."""
    seen = [x for x in picks if x is not None]
    return max(dict.fromkeys(seen), key=seen.count) if seen else None


def _elsewhere(others: list[str]) -> Callable[[str], bool]:
    return lambda name: any(ST._refs(t, name) for t in others)


def parse_choice(reply: str, n: int) -> Optional[list[int]]:
    """Integers from the CHOICE line only; None when absent, out of range, or 'none'. Never guesses."""
    m = re.search(r"CHOICE:\s*(.*)", reply, re.I)
    if not m:
        return None
    line = m.group(1).strip()
    if re.fullmatch(r"none\.?", line, re.I):
        return None
    if not re.fullmatch(r"\d+(?:(?:\s*[,;]\s*|\s+and\s+|\s+)\d+)*\s*\.?", line, re.I):
        return None
    nums = [int(x) for x in re.findall(r"\d+", line)]
    if any(not 1 <= x <= n for x in nums):
        return None
    return list(dict.fromkeys(nums))


def _py_texts(workdir: Path) -> dict[str, str]:
    """Every .py file of the work tree (posix path -> text), pruning .git, virtualenvs, caches and the root state/ (run artefacts:
    tens of thousands of directories nothing imports)."""
    out: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(workdir):
        at_root = Path(dirpath) == workdir
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".git", ".venv", "node_modules") and not (at_root and d == "state")]
        for fn in filenames:
            if fn.endswith(".py"):
                q = Path(dirpath) / fn
                out[q.relative_to(workdir).as_posix()] = q.read_text(encoding="utf-8", errors="replace")
    return out


WHY_CHARS = 200                                      # the teacher's reasoning shown with each few-shot example


PREDICT_TAG = "PREDICT"


def parse_prediction(text: str) -> Optional[dict[str, Any]]:
    """{'metric', 'size_delta', 'act_delta'|None} from a WorkResult note/reasoning written by ActionStudent; None when absent."""
    m = re.search(PREDICT_TAG + r" metric=(\S*) size_delta=(-?\d+) act_delta=(-?\d+|na)", text or "")
    if not m:
        return None
    return {"metric": m.group(1), "size_delta": int(m.group(2)), "act_delta": None if m.group(3) == "na" else int(m.group(3))}


class ActionStudent:
    name = "nupen-model-v2"

    def __init__(self, lessons_path: Path, llm: Any = None, k: int = 3, max_tokens: int = 80, max_candidates: int = 30,
                 max_files: int = 4, timeout_s: float = 300.0, rich_labels: bool = True, lexical_prior: bool = True,
                 use_chooser: Optional[bool] = None, chooser: Any = None, state_dir: Optional[Path] = None,
                 prescreen: Optional[bool] = None, prescreen_tests: bool = True, max_screened: int = 3) -> None:
        # prescreen (creator.prescreen): None = on exactly when the work tree is a Creator tree (has creator/kernel.py)
        self.prescreen, self.prescreen_tests, self.max_screened = prescreen, prescreen_tests, max_screened
        self.last_prescreen: list[dict[str, Any]] = []
        self._metric = "size"
        self.rich_labels, self.lexical_prior = rich_labels, lexical_prior
        # learned chooser (creator.chooser): None = the module default CHOOSER_DEFAULT; when on and trained it replaces the lexical stage
        self.state_dir = Path(state_dir) if state_dir is not None else _state_dir()
        self.use_chooser = use_chooser if use_chooser is not None else _policy(self.state_dir)
        self.chooser = chooser
        self._texts: dict[str, str] = {}
        self.lessons_path, self.llm, self.k = Path(lessons_path), llm, k
        self.max_tokens, self.max_candidates, self.max_files, self.timeout_s = max_tokens, max_candidates, max_files, timeout_s
        self.last_prompt, self.last_reply, self.last_seconds, self.last_calls = "", "", 0.0, 0

    def runtime_ok(self) -> bool:
        return self.llm is not None or (G.SERVER_EXE.is_file() and G.DEFAULT_MODEL.is_file())

    def can_attempt(self, task: Any) -> bool:
        return self.runtime_ok()

    def targets(self, package: Any, workdir: Path) -> list[str]:
        outs = [str(x) for x in getattr(package, "outputs", ()) or () if str(x).endswith(".py")
                and not MS._is_test(str(x)) and (workdir / str(x)).is_file()]
        return outs[: self.max_files]

    def candidates(self, package: Any, workdir: Path) -> list[Action]:
        rels = self.targets(package, workdir)
        texts = self._texts = _py_texts(workdir)
        out: list[Action] = []
        for rel in rels:
            others = [t for r, t in texts.items() if r != rel]
            out += enumerate_actions(texts[rel], rel, _elsewhere(others))
        return out[: self.max_candidates]

    def build_prompt(self, plan: Any, package: Any, cands: list[Action], workdir: Path) -> str:
        from creator.curriculum import task_kind
        kind, objective = task_kind(plan), str(getattr(package, "objective", ""))
        parts: list[str] = []
        for les in MS.retrieve(LessonLog(self.lessons_path).lessons(), kind, objective, str(getattr(plan, "component", "")), 6):
            acts = lesson_actions(les)
            if acts and len(parts) < self.k:
                parts.append(f"Example task: {les.objective[:300]}\nChosen actions: " + "; ".join(a.short() for a in acts[:4])
                             + (f"\nWhy: {' '.join(les.reasoning.split())[:WHY_CHARS]}" if les.reasoning.strip() else ""))
        task_file = workdir / ".creator_task.md"
        text = task_file.read_text(encoding="utf-8", errors="replace") if task_file.is_file() else objective
        parts.append(f"Task ({kind}):\n{text[:1500]}")
        parts.append("Candidate actions:\n" + "\n".join(f"{i}. [{a.path}] {a.label(self.rich_labels)}" for i, a in enumerate(cands, 1)))
        return "\n\n".join(parts)

    def _chooser(self) -> Any:
        if self.chooser is None:
            from creator import chooser as CH
            self.chooser = CH.Chooser.load() or False
        return self.chooser if self.chooser and self.chooser.trained else None

    def _ask(self, messages: list[dict[str, str]]) -> str:
        self.last_calls += 1
        if self.llm is not None:
            return str(self.llm.chat(messages, max_tokens=self.max_tokens, temperature=0.0))
        with MS._SERVER_LOCK:
            with G.LocalModel(startup_s=min(120.0, self.timeout_s)) as llm:
                return str(llm.chat(messages, max_tokens=self.max_tokens, temperature=0.0))

    def __call__(self, plan: Any, package: Any, workdir: Path) -> "WorkResult":
        from creator.kernel import WorkResult
        workdir = Path(workdir)
        if not self.runtime_ok():
            return WorkResult(False, "local model runtime missing")
        try:
            self.last_calls = 0
            cands = self.candidates(package, workdir)
            if not cands:
                return WorkResult(False, "no valid action candidates for the target files")
            self.last_prescreen, screen = [], self._screening(workdir)
            if screen:
                cands = self._viable(cands, plan, package, workdir)
                if not cands:
                    return WorkResult(False, "prescreen: no candidate can improve the targeted metric: "
                                      + "; ".join(r["reason"] for r in self.last_prescreen[:3]), calls=0)
            self.last_prompt = self.build_prompt(plan, package, cands, workdir)
            t0 = time.monotonic()
            reply = self._ask([{"role": "system", "content": SYSTEM}, {"role": "user", "content": self.last_prompt}])
            self.last_seconds, self.last_reply = time.monotonic() - t0, reply
            why = (re.search(r"WHY:\s*(.*)", reply) or re.search(r"(.*)", reply))
            reasoning = (why.group(1).strip() if why else "")[:500]
            picks = parse_choice(reply, len(cands))
            model_pick = picks[0] if picks and len(picks) == 1 else None
            ch = self._chooser() if self.use_chooser else None
            if ch is not None and (picks is None or len(picks) == 1):
                alt = ch.pick(str(getattr(package, "objective", "")), cands, self._texts, picks[0] if picks else None)
                picks = [alt] if alt else picks
            elif self.lexical_prior and (picks is None or len(picks) == 1):
                alt = combine_choice(picks[0] if picks else None, str(getattr(package, "objective", "")), cands)
                picks = [alt] if alt else picks
            if not picks:
                return WorkResult(False, f"model reply held no valid choice: {reply[:120]!r}", calls=self.last_calls, reasoning=reasoning)
            chosen = [cands[i - 1] for i in picks]
            if screen:
                chosen = self._first_passing(chosen, cands, workdir)
                if not chosen:
                    return WorkResult(False, "prescreen rejected every attempted candidate: "
                                      + "; ".join(r["reason"] for r in self.last_prescreen[-3:]), calls=self.last_calls, reasoning=reasoning)
                picks = [cands.index(a) + 1 for a in chosen]
            res = self._apply(chosen, package, workdir, reasoning)
            if res.claimed_done:
                self._shadow(plan, package, cands, model_pick, picks)
            return res
        except Exception as e:                              # noqa: BLE001 - a student never crashes the swarm
            return WorkResult(False, f"model call failed: {type(e).__name__}: {str(e)[:200]}", calls=max(self.last_calls, 1))

    # -- pre-screen (creator.prescreen): may only REJECT early; the kernel's measurement stays the only adoption authority
    def _screening(self, workdir: Path) -> bool:
        return self.prescreen if self.prescreen is not None else (Path(workdir) / "creator" / "kernel.py").is_file()

    def _log_prescreen(self, rec: dict[str, Any]) -> None:
        self.last_prescreen.append(rec)
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            with (self.state_dir / "prescreen_log.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
        except OSError:
            pass

    def _viable(self, cands: list[Action], plan: Any, package: Any, workdir: Path) -> list[Action]:
        """Candidates whose in-process metric delta can improve the targeted metric, best predicted gain first; the rest are
        rejected here with the reason recorded."""
        from creator import prescreen as PS
        metric = PS.metric_for(plan)
        self._foot = foot = PS.Footing(workdir)
        keep: list[tuple[int, int, Action]] = []
        for n, a in enumerate(cands):
            texts = self._texts
            others = [t for r, t in texts.items() if r != a.path]
            new = apply_action(texts[a.path], a, _elsewhere(others)) if a.path in texts else None
            v = PS.metric_verdict(foot, a.path, new, metric) if new is not None else PS.Verdict(False, "does not apply", "metric")
            self._log_prescreen({"package_id": str(getattr(plan, "package_id", "") or getattr(package, "id", "")), "action": a.short(),
                                 "path": a.path, "metric": metric, "ok": v.ok, "reason": v.reason, "size_delta": v.size_delta,
                                 "act_delta": v.act_delta, "stage": "metric"})
            if v.ok:
                keep.append((-PS.predicted_gain(v.size_delta or 0, v.act_delta, metric), n, a))
        self._metric = metric
        return [a for _, _, a in sorted(keep)]

    def _first_passing(self, chosen: list[Action], cands: list[Action], workdir: Path) -> list[Action]:
        """The first attempt (the student's own choice, then the rest best-predicted-gain-first, at most `max_screened`) whose
        import and direct tests pass in the work tree. [] when none does."""
        from creator import prescreen as PS
        attempts = [chosen] + [[a] for a in cands if [a] != chosen]
        graph = None
        for atts in attempts[: self.max_screened]:
            by_path: dict[str, str] = {}
            for a in atts:
                src = by_path.get(a.path) or self._texts.get(a.path, "")
                others = [t for r, t in self._texts.items() if r != a.path]
                new = apply_action(src, a, _elsewhere(others))
                if new is not None:
                    by_path[a.path] = new
            rec = {"action": "+".join(a.short() for a in atts), "metric": self._metric, "stage": "runtime", "ok": bool(by_path), "reason": "" if by_path else "does not apply"}
            for rel, new in by_path.items():
                if graph is None and self.prescreen_tests:
                    from creator import testrun as TR
                    graph = TR.ImportGraph.build(workdir)
                v = PS.check_in_tree(workdir, rel, new, graph, self.prescreen_tests)
                if not v.ok:
                    rec.update(ok=False, reason=v.reason, stage=v.stage)
                    break
            self._log_prescreen(rec)
            if rec["ok"]:
                return atts
        return []

    def _shadow(self, plan: Any, package: Any, cands: list[Action], model_pick: Optional[int], picks: list[int]) -> None:
        """Record what the default policy and the chooser each pick for a decision that was applied. Never alters the decision."""
        try:
            from creator import shadow
            obj = str(getattr(package, "objective", ""))
            default = combine_choice(model_pick, obj, cands)
            ch = self._chooser()
            alt = ch.pick(obj, cands, self._texts, model_pick) if ch is not None else None
            shadow.record(self.state_dir, str(getattr(plan, "package_id", "") or getattr(package, "id", "")), cands, default, alt,
                          model_pick, "chooser" if self.use_chooser else "default")
        except Exception as e:                              # noqa: BLE001 - shadowing never affects the student
            self.last_shadow_error = f"{type(e).__name__}: {e}"

    def _apply(self, chosen: list[Action], package: Any, workdir: Path, reasoning: str) -> "WorkResult":
        from creator.kernel import WorkResult
        texts = _py_texts(workdir)
        saved = {a.path: texts[a.path] for a in chosen if a.path in texts}
        cur, done = dict(saved), []
        for a in chosen:
            if a.path not in cur:
                continue
            others = [t for r, t in texts.items() if r != a.path]
            new = apply_action(cur[a.path], a, _elsewhere(others))
            if new is not None:
                cur[a.path] = new
                done.append(a.short())
        changed = [p for p in cur if cur[p] != saved[p]]
        if not changed or any(ST._parse(cur[p]) is None for p in changed):
            return WorkResult(False, "chosen actions did not apply", calls=self.last_calls, reasoning=reasoning)
        for p in changed:
            eol = "\r\n" if b"\r\n" in (workdir / p).read_bytes() else "\n"          # a CRLF file stays CRLF: no whole-file diff
            (workdir / p).write_text(cur[p], encoding="utf-8", newline=eol)
        pred = self.prediction(chosen)
        note = f"{len(done)} actions applied: {', '.join(done)}" + (f" | {PREDICT_TAG} {pred}" if pred else "")
        why = (reasoning + f" [{PREDICT_TAG} {pred}]") if pred else reasoning
        return WorkResult(True, note, calls=self.last_calls, by=self.name, reasoning=why)

    def prediction(self, chosen: list[Action]) -> str:
        """The pre-screen's own predicted effect of the claimed change, in a parseable form (see `parse_prediction`), so that
        calibration can score it against the kernel's measured verdict. '' when the pre-screen did not run."""
        rows = {r["action"]: r for r in self.last_prescreen if r.get("stage") == "metric" and "action" in r}
        got = [rows[a.short()] for a in chosen if a.short() in rows]
        if not got:
            return ""
        sd = sum(int(r.get("size_delta") or 0) for r in got)
        ads = [r["act_delta"] for r in got if r.get("act_delta") is not None]
        return f"metric={got[0].get('metric', '')} size_delta={sd} act_delta={sum(ads) if ads else 'na'}"
