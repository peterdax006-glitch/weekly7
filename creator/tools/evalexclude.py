"""The eval-suite exclusion list (R13): what the ~200-task held-out suite was built from and so must never reach training data.

The list lives OUTSIDE the repo (<runtime>/gpuday/eval_suite200/EXCLUDE.json, or the file named by env NUPEN_EVAL_EXCLUDE) because the held-out material
must not be committed. Three guards read it: creator.tools.phase2data.Guard, creator.tools.reuse.LeakGuard (default_guard) and creator.trainmix.eval_sets
(so trainmix.screen drops matching rows). A missing file means no extra exclusions (the older guards still apply).

EXCLUDE.json: {"ids": [task ids and source ids], "names": [function names], "source_paths": [[repo, path]], "body_hashes": [normalized function-body hashes
(reuse.body_hash)], "sigdoc_hashes": [...], "text_hashes": [sha of whitespace-normalized lines >= 30 chars of the requests / hidden tests],
"texts": [[id, text]] (requests, for near-duplicate screening)}."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

ENV = "NUPEN_EVAL_EXCLUDE"
MIN_SHARED_LINES = 3             # distinct normalized lines (>= 30 chars) shared with the suite that make a text a copy
MIN_NODES = 30                   # smallest function (AST nodes) whose normalized body is hashed


def default_path(runtime: Optional[Path] = None) -> Path:
    e = os.environ.get(ENV)
    if e:
        return Path(e)
    return (runtime or Path.home() / "creator_runtime") / "gpuday" / "eval_suite200" / "EXCLUDE.json"


def norm_line(line: str) -> str:
    return " ".join(line.split()).lower()


def line_hash(line: str) -> str:
    return hashlib.sha256(norm_line(line).encode("utf-8")).hexdigest()[:16]


def code_blocks(text: str) -> list[str]:
    """Python code in a training row: fenced blocks, else the whole text."""
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", text, re.S)
    return blocks or [text]


def function_hashes(code: str) -> set[str]:
    from creator.tools import reuse as RU
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return set()
    return {RU.body_hash(n) for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


@dataclass
class Exclusions:
    ids: set[str] = field(default_factory=set)
    names: set[str] = field(default_factory=set)
    source_paths: set[tuple[str, str]] = field(default_factory=set)
    body_hashes: set[str] = field(default_factory=set)
    sigdoc_hashes: set[str] = field(default_factory=set)
    text_hashes: set[str] = field(default_factory=set)
    texts: list[tuple[str, str]] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.ids or self.names or self.body_hashes or self.text_hashes)

    def violation(self, text: str, item_id: str = "") -> Optional[str]:
        """Why a training row / text must be dropped, or None."""
        if item_id and item_id in self.ids:
            return "an eval-suite task / source id"
        shared = {line_hash(ln) for ln in text.splitlines() if len(norm_line(ln)) >= 30} & self.text_hashes
        if len(shared) >= MIN_SHARED_LINES:                          # one common idiom line is not a leak; three distinct lines are a copy
            return f"contains {len(shared)} lines of the eval suite's requests/tests/references"
        if self.body_hashes:
            for blk in code_blocks(text):
                if function_hashes(blk) & self.body_hashes:
                    return "contains an eval-suite function body"
        if self.names:
            m = re.search(r"\bdef (" + "|".join(re.escape(n) for n in sorted(self.names, key=len, reverse=True)) + r")\s*\(", text)
            if m:
                return f"defines an eval-suite function name ({m.group(1)})"
        return None

    def eval_set(self) -> dict[str, Any]:
        """The trainmix.eval_sets entry: texts for the near-duplicate screen, ids for the id screen, fn_names for the def screen."""
        return {"texts": [(i, t) for i, t in self.texts if len(t) >= 20], "ids": set(self.ids), "hash": None, "fn_names": sorted(self.names)}


def load(path: Optional[Path] = None) -> Exclusions:
    p = Path(path) if path else default_path()
    if not p.is_file():
        return Exclusions()
    d = json.loads(p.read_text(encoding="utf-8"))
    return Exclusions(ids=set(d.get("ids") or []), names=set(d.get("names") or []), source_paths={(a, b) for a, b in d.get("source_paths") or []},
                      body_hashes=set(d.get("body_hashes") or []), sigdoc_hashes=set(d.get("sigdoc_hashes") or []),
                      text_hashes=set(d.get("text_hashes") or []), texts=[(str(a), str(b)) for a, b in d.get("texts") or []])


def _source_path(source_id: str) -> Optional[tuple[str, str]]:
    if source_id.startswith("pub:"):
        _, repo, rest = source_id.split(":", 2)
        return repo, rest.rsplit(":", 1)[0]
    return None


def write_exclude(path: Path, fn_tasks: Iterable[Mapping[str, Any]], app_tasks: Iterable[Mapping[str, Any]], repo_root: Any = None, public_root: Any = None) -> dict[str, Any]:
    """fn_tasks: suite task dicts carrying `_src` (source id) and `_ref` (reference code); app_tasks: {id, request, accept}."""
    from creator.tools import reuse as RU
    ex: dict[str, Any] = {"ids": [], "names": [], "source_paths": [], "body_hashes": [], "sigdoc_hashes": [], "text_hashes": [], "texts": []}
    ids: set[str] = set()
    names: set[str] = set()
    paths: set[tuple[str, str]] = set()
    bh: set[str] = set()
    sd: set[str] = set()
    th: set[str] = set()
    texts: list[tuple[str, str]] = []

    def lines_of(*ts: str) -> None:
        for t in ts:
            for ln in t.splitlines():
                if len(norm_line(ln)) >= 30:
                    th.add(line_hash(ln))
    for t in fn_tasks:
        ids |= {t["id"], t["_src"]}
        names.add(t["name"])
        sp = _source_path(t["_src"])
        if sp:
            paths.add(sp)
        tree = ast.parse(t["_ref"])
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if sum(1 for _ in ast.walk(n)) >= MIN_NODES:                 # `return x` style bodies would block half the corpus
                    bh.add(RU.body_hash(n))
                if n.name == t["name"]:
                    sd.add(RU.sigdoc_hash(n.name, RU._args(n), ast.get_docstring(n) or ""))
        lines_of(t["request"], t["stub"], t["_ref"])
        lines_of(*[json.dumps(x) for x in t["tests"]])
        texts.append((t["id"], t["request"]))
    for t in app_tasks:
        ids.add(t["id"])
        lines_of(t["request"], t["accept"])
        texts.append((t["id"], t["request"].strip()))
        texts.append((t["id"] + ":accept", t["accept"]))
    ex.update(ids=sorted(ids), names=sorted(names), source_paths=sorted(paths), body_hashes=sorted(bh), sigdoc_hashes=sorted(sd), text_hashes=sorted(th),
              texts=[list(x) for x in texts])
    Path(path).write_text(json.dumps(ex, indent=1), encoding="utf-8")
    return {k: len(v) for k, v in ex.items()}
