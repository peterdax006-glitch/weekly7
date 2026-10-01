"""Nupen's model-backed student: the small local model (qwen2.5-coder 1.5B) behind the Student protocol.

Owner, 1 Oct 2026: the model on disk MAY be used, but only as a trainable student - the kernel's sandbox tests and improvement
verdict decide every change; this class never judges itself. Learning from lessons is in-context for now: the k most similar
ADOPTED lessons (same task_kind first, then word overlap) go into the prompt as worked examples (objective, reasoning, the change
as search/replace edits). `export_sft` writes the same lessons as chat-format examples for a later LoRA fine-tune (no training here).

Resources: the llama server starts lazily on the first call, is stopped when the call ends (one server at a time, process-wide
lock), tokens and wall time are bounded, and a missing runtime makes `can_attempt` False - it never crashes the swarm."""
from __future__ import annotations

import ast
import dataclasses
import difflib
import json
import re
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from creator import generator as G
from creator.curriculum import Lesson, LessonLog

if TYPE_CHECKING:
    from creator.kernel import WorkResult

SYSTEM = ("You are Nupen, a careful Python developer learning from a teacher's solved work packages. "
          "Reply with a short line starting 'REASONING:' saying what you change and why, then ONLY search/replace edits, each as:\n"
          "FILE: <path>\n<<<<<<< SEARCH\n<exact existing lines>\n=======\n<replacement lines>\n>>>>>>> REPLACE\n"
          "The SEARCH text must match the file exactly and be unique. Change as little as possible. Never edit tests.")
_SERVER_LOCK = threading.Lock()                           # one llama server at a time, process-wide
_WORD = re.compile(r"[a-z_]{3,}")


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


# ------------------------------------------------------------------------------------------------ diffs as edits

def _unique(lines: list[str], i: int, j: int) -> bool:
    block = lines[i:j]
    return sum(1 for k in range(len(lines) - len(block) + 1) if lines[k:k + len(block)] == block) == 1


def edits_between(path: str, before: str, after: str, context: int = 2, max_chars: int = 4000) -> str:
    """The change before -> after as search/replace edits in the EDIT_RE format (compact; context grown until SEARCH is unique)."""
    if not before.strip():
        return f"FILE: {path}\n<<<<<<< SEARCH\n\n=======\n{after.rstrip()}\n>>>>>>> REPLACE\n"[:max_chars]
    a, b = before.split("\n"), after.split("\n")
    out: list[str] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        lo, hi = max(0, i1 - context), min(len(a), i2 + context)
        while not _unique(a, lo, hi) and (lo > 0 or hi < len(a)) and hi - lo < 40:
            lo, hi = max(0, lo - 1), min(len(a), hi + 1)
        old = a[lo:i1] + a[i1:i2] + a[i2:hi]
        new = a[lo:i1] + b[j1:j2] + a[i2:hi]
        out.append(f"FILE: {path}\n<<<<<<< SEARCH\n" + "\n".join(old) + "\n=======\n" + "\n".join(new) + "\n>>>>>>> REPLACE\n")
    return "".join(out)[:max_chars]


def lesson_edits(les: Lesson, max_chars: int = 3500) -> str:
    parts = [edits_between(p, les.files_before.get(p, ""), after) for p, after in les.files_after.items() if p.endswith(".py")]
    return "".join(parts)[:max_chars]


def _example_text(les: Lesson) -> str:
    return (f"Objective ({les.task_kind}): {les.objective}\nTeacher's reasoning: {les.reasoning or '(none)'}\n"
            f"Teacher's change:\n{lesson_edits(les)}")


# ------------------------------------------------------------------------------------------------ retrieval

def retrieve(lessons: Sequence[Lesson], kind: str, objective: str, component: str = "", k: int = 2) -> list[Lesson]:
    """The k most similar ADOPTED lessons that carry a Python change: same task_kind first, then word overlap with objective/component."""
    want = _words(objective + " " + component)
    pool = [les for les in lessons if les.adopted is True and any(p.endswith(".py") for p in les.files_after)]
    pool.sort(key=lambda les: (les.task_kind != kind, -len(want & _words(les.objective + " " + les.component))))
    return pool[:k]


# ------------------------------------------------------------------------------------------------ SFT export

def sft_example(les: Lesson) -> dict[str, Any]:
    user = f"Task ({les.task_kind}, component {les.component}):\n{les.objective}\n\n{les.context}\n\nCurrent files:\n" + "\n".join(
        f"FILE: {p}\n```python\n{les.files_before.get(p, '')}```" for p in les.files_after if p.endswith(".py"))
    reply = f"REASONING: {les.reasoning.strip() or 'apply the change'}\n{lesson_edits(les, 20000)}"
    return {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user},
                         {"role": "assistant", "content": reply}],
            "lesson_id": les.lesson_id, "task_kind": les.task_kind, "solver": les.solver}


def export_sft(lessons_path: Path, out_jsonl: Path) -> int:
    """Write adopted lessons as chat-format training examples (one JSON object per line); returns how many. No training is run."""
    rows = [sft_example(les) for les in LessonLog(Path(lessons_path)).lessons()
            if les.adopted is True and any(p.endswith(".py") for p in les.files_after)]
    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


# ------------------------------------------------------------------------------------------------ the student

class ModelStudent:
    name = "nupen-model-v1"

    def __init__(self, lessons_path: Path, llm: Any = None, k: int = 2, max_tokens: int = 700, max_file_chars: int = 9000,
                 timeout_s: float = 600.0, max_files: int = 2, attempts: int = 2) -> None:
        self.lessons_path, self.llm, self.k = Path(lessons_path), llm, k
        self.max_tokens, self.max_file_chars, self.timeout_s, self.max_files = max_tokens, max_file_chars, timeout_s, max_files
        self.attempts, self.last_calls = attempts, 0
        self.last_prompt = ""
        self.last_seconds = 0.0

    # -- protocol
    def runtime_ok(self) -> bool:
        return self.llm is not None or (G.SERVER_EXE.is_file() and G.DEFAULT_MODEL.is_file())

    def can_attempt(self, task: Any) -> bool:
        return self.runtime_ok()

    # -- prompt
    def targets(self, package: Any, workdir: Path) -> list[str]:
        outs = [str(x) for x in getattr(package, "outputs", ()) or () if str(x).endswith(".py")
                and not Path(str(x)).name.startswith("test_") and (workdir / str(x)).is_file()]
        return outs[: self.max_files]

    def build_prompt(self, plan: Any, package: Any, workdir: Path) -> str:
        from creator.curriculum import task_kind
        kind, objective = task_kind(plan), str(getattr(package, "objective", ""))
        parts = []
        for les in retrieve(LessonLog(self.lessons_path).lessons(), kind, objective, str(getattr(plan, "component", "")), self.k):
            parts.append("Worked example (adopted by the kernel):\n" + _example_text(les))
        task_file = workdir / ".creator_task.md"
        text = task_file.read_text(encoding="utf-8", errors="replace") if task_file.is_file() else objective
        parts.append(f"Your task ({kind}):\n{text[:3000]}")
        for rel in self.targets(package, workdir):
            body = (workdir / rel).read_text(encoding="utf-8", errors="replace")[: self.max_file_chars]
            parts.append(f"FILE: {rel}\n```python\n{body}\n```")
        return "\n\n".join(parts)

    # -- the attempt
    def __call__(self, plan: Any, package: Any, workdir: Path) -> "WorkResult":
        from creator.kernel import WorkResult
        workdir = Path(workdir)
        if not self.runtime_ok():
            return WorkResult(False, "local model runtime missing")
        if not self.targets(package, workdir):
            return WorkResult(False, "no target python file to edit")
        try:
            self.last_calls = 0
            prompt = self.build_prompt(plan, package, workdir)
            self.last_prompt = prompt
            t0 = time.monotonic()
            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
            res = WorkResult(False, "no attempt")
            for attempt in range(self.attempts):            # one retry, told exactly why the first reply was unusable
                reply = self._ask(messages, 0.2 + 0.3 * attempt)
                res = self._apply(reply, workdir)
                if res.claimed_done or time.monotonic() - t0 > self.timeout_s:
                    break
                messages += [{"role": "assistant", "content": reply},
                             {"role": "user", "content": f"That did not work: {res.notes}\nCopy the SEARCH lines EXACTLY from the "
                              "file (only the few lines you change, not the whole file) and reply again in the same format."}]
            self.last_seconds = time.monotonic() - t0
        except Exception as e:                              # noqa: BLE001 - a student never crashes the swarm
            return WorkResult(False, f"model call failed: {type(e).__name__}: {str(e)[:200]}", calls=1)
        return dataclasses.replace(res, calls=self.last_calls)

    def _ask(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        self.last_calls += 1
        if self.llm is not None:
            return str(self.llm.chat(messages, max_tokens=self.max_tokens, temperature=temperature))
        with _SERVER_LOCK:                                  # lazy start; always stopped when the call is done
            with G.LocalModel(startup_s=min(120.0, self.timeout_s)) as llm:
                return str(llm.chat(messages, max_tokens=self.max_tokens, temperature=temperature))

    def close(self) -> None:
        """Nothing stays running between calls (the server lives only inside _ask); kept for the context-manager style."""

    def __enter__(self) -> "ModelStudent":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _apply(self, reply: str, workdir: Path) -> "WorkResult":
        from creator.kernel import WorkResult
        m = re.search(r"REASONING:\s*(.*?)(?=\nFILE:|\Z)", reply, re.S)
        reasoning = (m.group(1).strip() if m else reply.split("FILE:")[0].strip())[:1500]
        edits = [e for e in G.parse_edits(reply) if not _is_test(e[0])]
        if not edits:
            return WorkResult(False, "model reply held no usable search/replace edit", calls=1, reasoning=reasoning)
        touched = sorted({e[0] for e in edits})
        saved = {p: (workdir / p).read_text(encoding="utf-8") if (workdir / p).is_file() else None for p in touched}
        applied, refused = G.apply_edits(workdir, edits)
        bad = [p for p in set(applied) if not _parses((workdir / p).read_text(encoding="utf-8"))]
        changed = [p for p in set(applied) if (workdir / p).read_text(encoding="utf-8") != saved.get(p)]
        if not changed and not bad:
            refused.append("the edits changed nothing")
        if not changed or bad:
            for p, text in saved.items():                    # put everything back: a half-applied change is never left behind
                if text is None:
                    (workdir / p).unlink(missing_ok=True)
                else:
                    (workdir / p).write_text(text, encoding="utf-8", newline="\n")
            why = f"files no longer parse: {bad}" if bad else f"no edit applied: {'; '.join(refused)[:300]}"
            return WorkResult(False, why, calls=1, reasoning=reasoning)
        note = f"{len(applied)} edits applied to {', '.join(sorted(set(changed)))}" + (f"; refused: {len(refused)}" if refused else "")
        return WorkResult(True, note, calls=1, by=self.name, reasoning=reasoning)


def _is_test(path: str) -> bool:
    return Path(path).name.startswith("test_") or "tests" in Path(path).parts


def _parses(src: str) -> bool:
    try:
        ast.parse(src)
        return True
    except (SyntaxError, ValueError):
        return False
