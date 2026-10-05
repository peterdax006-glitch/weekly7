"""R5 task-class router: the cheapest coder that is likely to solve this kind of task, escalating on a visible-example failure.

Owner 5 Oct 2026: pass rate >= Claude's 72% AND >= 5000 workable lines/h. lines/h = tok/s x useful ratio / tokens per line, so a task
that a 0.6B coder (58 tok/s) can solve must not cost a 1.7B (22 tok/s) or a MoE (~3B active) run, and a hard task must not burn three
cheap attempts first.

1. FEATURES   code only, from the task: function vs app, signature/docstring length, visible examples, imports needed, estimated output
              tokens (learned per class from past tasks, heuristic until seen), optional reuse score (creator/tools/reuse.py).
2. CLASS      fn_s / fn_m / fn_l / app_s / app_l from a difficulty score; a strong reuse hit moves a task one class down.
3. POLICY     per (class, coder) a Beta posterior of "passes the visible examples when attempted" (prior from the first measurements,
              strength PRIOR_N). The ladder is the available coders cheapest-first; the start rung k minimises expected seconds per
              solved task of ladder[k:] (E_k / P_k), so a rung that almost never solves a class is skipped, not paid for.
4. LEARNING   observe() adds one success/failure count (no training); state is a small JSON file written atomically.
Lazy imports only (json, re); model speeds/availability come from creator.resources.MODEL_CATALOG."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Optional

LADDER = ("C06", "C17", "MOE")                      # cheapest first = fastest first
PRIOR_N = 3.0                                       # pseudo-observations behind a prior
PROMPT_X = 1.8                                      # prompt reading speed = this x generation speed (measured ~40 vs 22 tok/s)
OVERHEAD_S = 3.0                                    # tests + tool calls per attempt
CHARS_PER_TOKEN = 3.5
FIXED_PROMPT_TOK = 150                              # the fast-path system prompt + prefill
# prior P(pass | attempted), from P0.9/P0.10 (1.7B: 12.5% overall, ~35% on easy functions); 0.6B and MoE are estimates until measured
PRIORS: dict[str, dict[str, float]] = {
    "fn_s": {"C06": 0.45, "C17": 0.60, "MOE": 0.85},
    "fn_m": {"C06": 0.20, "C17": 0.35, "MOE": 0.70},
    "fn_l": {"C06": 0.08, "C17": 0.20, "MOE": 0.60},
    "app_s": {"C06": 0.03, "C17": 0.12, "MOE": 0.50},
    "app_l": {"C06": 0.01, "C17": 0.05, "MOE": 0.40},
}
_STDLIB = ("re", "math", "itertools", "collections", "datetime", "json", "heapq", "bisect", "functools", "string", "random", "hashlib",
           "fractions", "decimal", "statistics", "operator", "typing", "dataclasses", "textwrap", "struct", "base64", "copy")
_IMPORT_RE = re.compile(r"\b(" + "|".join(_STDLIB) + r")\.\w+|\bimport (\w+)|\bfrom (\w+) import")


def features(task: Mapping[str, Any]) -> dict[str, Any]:
    """Feature dict of a suite task ({id, family, request, stub?, examples?, n_visible?, reuse_score?})."""
    fam = "app" if task.get("family") == "app" else "fn"
    req = str(task.get("request", ""))
    stub = str(task.get("stub", ""))
    m = re.search(r"def \w+\((.*?)\)", stub or req, re.S)
    sig_len = len(m.group(0)) if m else 0
    doc_len = len(stub.split('"""', 2)[1]) if stub.count('"""') >= 2 else len(req)
    n_ex = int(task.get("n_visible") or 0) or (str(task.get("examples", "")).count("\n") + 1 if task.get("examples") else 0)
    mods = {a or b or c for a, b, c in _IMPORT_RE.findall(req + stub)} - {"annotations", "__future__", ""}
    n_imp = len(mods - {"typing"})
    score = (doc_len / 250.0 + sig_len / 150.0 + 0.4 * n_imp - 0.15 * min(n_ex, 3)) if fam == "fn" else len(req) / 300.0 + 0.4 * n_imp
    f = {"id": str(task.get("id", "")), "family": fam, "sig_len": sig_len, "doc_len": doc_len, "n_examples": n_ex, "n_imports": n_imp,
         "reuse": float(task.get("reuse_score") or 0.0), "score": round(score, 2), "prompt_chars": len(req) + len(stub)}
    f["est_tokens"] = (int(40 + 0.35 * doc_len + 4 * sig_len / 10) if fam == "fn" else int(250 + 0.5 * len(req)))
    return f


def task_class(f: Mapping[str, Any]) -> str:
    """Difficulty class; a strong reuse hit (>= 0.8) moves a function task one class down (adapting proven code is easier than writing)."""
    s = float(f["score"])
    if f["family"] == "app":
        return "app_s" if s < 1.6 else "app_l"
    i = 0 if s < 1.3 else 1 if s < 2.6 else 2
    if float(f.get("reuse", 0.0)) >= 0.8:
        i = max(0, i - 1)
    return ("fn_s", "fn_m", "fn_l")[i]


class Router:
    """Counts + the ladder policy. `speeds` = {coder: tok/s}; coders without a speed are not routable (e.g. MoE until measured)."""

    def __init__(self, speeds: Mapping[str, Optional[float]], state: Optional[Path] = None) -> None:
        self.speeds = {k: float(v) for k, v in speeds.items() if v}
        self.state = Path(state) if state else None
        self.counts: dict[str, dict[str, list[int]]] = {}       # class -> coder -> [passes, fails]
        self.tokens: dict[str, float] = {}                      # class -> EMA of generated tokens per attempt
        if self.state and self.state.exists():
            try:
                d = json.loads(self.state.read_text(encoding="utf-8"))
                self.counts, self.tokens = d.get("counts", {}), d.get("tokens", {})
            except (OSError, ValueError):
                pass

    # ---- policy
    def p(self, cls: str, coder: str) -> float:
        a, b = self.counts.get(cls, {}).get(coder, [0, 0])
        p0 = PRIORS[cls][coder]
        return (a + PRIOR_N * p0) / (a + b + PRIOR_N)

    def seconds(self, f: Mapping[str, Any], cls: str, coder: str) -> float:
        tps = self.speeds[coder]
        out = self.tokens.get(cls) or f["est_tokens"]
        return OVERHEAD_S + (FIXED_PROMPT_TOK + f["prompt_chars"] / CHARS_PER_TOKEN) / (tps * PROMPT_X) + out / tps

    def ladder(self, cls: str, available: Optional[set[str]] = None) -> list[str]:
        return [c for c in LADDER if c in self.speeds and (available is None or c in available)]

    def route(self, task: Mapping[str, Any], available: Optional[set[str]] = None) -> dict[str, Any]:
        f = features(task)
        cls = task_class(f)
        lad = self.ladder(cls, available)
        if not lad:
            raise ValueError("no coder available")
        best = None
        for k in range(len(lad)):
            reach, e = 1.0, 0.0
            for c in lad[k:]:
                e += reach * self.seconds(f, cls, c)
                reach *= 1.0 - self.p(cls, c)
            ps = 1.0 - reach
            cost = e / max(ps, 1e-6)
            if best is None or cost < best[0]:
                best = (cost, k, e, ps)
        assert best is not None
        _, k, e, ps = best
        return {"id": f["id"], "class": cls, "start": lad[k], "ladder": lad[k:], "expected_s": round(e, 1), "p_solve": round(ps, 3),
                "s_per_solved": round(e / max(ps, 1e-6), 1), "features": f}

    def next_coder(self, route: Mapping[str, Any], failed: str) -> Optional[str]:
        """Escalation on a visible-example failure: the next rung of the route's ladder, None when the ladder is spent."""
        lad = list(route["ladder"])
        return lad[lad.index(failed) + 1] if failed in lad and lad.index(failed) + 1 < len(lad) else None

    # ---- learning
    def observe(self, cls: str, coder: str, passed: bool, tokens: Optional[int] = None) -> None:
        c = self.counts.setdefault(cls, {}).setdefault(coder, [0, 0])
        c[0 if passed else 1] += 1
        if tokens:
            self.tokens[cls] = tokens if cls not in self.tokens else 0.7 * self.tokens[cls] + 0.3 * tokens
        self.save()

    def save(self) -> None:
        if not self.state:
            return
        self.state.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state.with_suffix(".tmp")
        tmp.write_text(json.dumps({"counts": self.counts, "tokens": self.tokens}), encoding="utf-8")
        os.replace(tmp, self.state)


def default_speeds(profile: str = "pc", assume_moe: bool = False) -> dict[str, Optional[float]]:
    from creator import resources as R
    out: dict[str, Optional[float]] = {}
    for n, s in R.MODEL_CATALOG.items():
        out[n] = s["tok_s"].get(profile) or (s.get("tok_s_assumed", {}).get(profile) if assume_moe else None)
    return out


def dry_run(suite: Path, profile: str = "pc", assume_moe: bool = False) -> str:
    """Routing table of the suite from its metadata alone (no model runs)."""
    tasks = json.loads(Path(suite).read_text(encoding="utf-8"))["tasks"]
    r = Router(default_speeds(profile, assume_moe))
    rows = [r.route(t) for t in tasks]
    lines = [f"{'task':<28}{'class':<7}{'start':<6}{'ladder':<14}{'E[s]':>7}{'P(solve)':>9}{'s/solved':>9}"]
    for x in rows:
        lines.append(f"{x['id']:<28}{x['class']:<7}{x['start']:<6}{'>'.join(x['ladder']):<14}{x['expected_s']:>7}{x['p_solve']:>9}{x['s_per_solved']:>9}")
    n = len(rows)
    ps = sum(x["p_solve"] for x in rows)
    lines.append(f"TOTAL expected {sum(x['expected_s'] for x in rows):.0f} s for {n} tasks, expected pass {ps / n:.1%}, "
                 f"{sum(x['expected_s'] for x in rows) / max(ps, 1e-6):.1f} s per solved task (priors, not measurements)")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    print(dry_run(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "pc", "--assume-moe" in sys.argv))
