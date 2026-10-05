"""Phase 2 training data generated on the PC (R7): PINPOINT + DEBUG_FIX rows from planted bugs, verified CODE rows, and the leakage guards.

Loaded on demand. Output stays OUTSIDE the repo (<runtime>/gpuday/phase2_data/). Every row is a chat row
{"id", "source", "kind", "split", "messages": [system, user, assistant], "meta": {...}}.

  pinpoint   the top-K suspect lines (+-1 line of context, most suspicious first) of a failing planted bug -> the number of the true line
  debug_fix  the function holding the bug + failing test + traceback -> a minimal SEARCH/REPLACE edit; the ORIGINAL code is the label
  code       'write this function' rows from public tasks whose reference solution was RE-RUN here against the tests (verified)

Leakage rules (asserted, not hoped for - see Guard and audit()):
  * trust gate: planted bugs come from a `git archive` snapshot of the last commit BEFORE codetrust_heldout.json's cut (so no file content
    from a tree at/after the cut) and that commit must not be a held-out commit; public rows carry no Nupen state.
  * the 32-task baseline suite (baseline_suite/) and anything derived from it: ids, requests, accept tests, stubs, the app_base source
    lines, the fn.* function names. A row that mentions any of them is dropped, and audit() raises when one slipped through.
"""
from __future__ import annotations

import ast
import collections
import hashlib
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

CUT_UTC = "2026-10-01T21:58:30+00:00"
SNAPSHOT_PATHS = ("creator", "engine", "tests", "scripts", "pyproject.toml")        # never state/ (live data) or docs/
LINE_MIN = 60                    # only long exact suite lines count: short common one-liners (def f(x):, return x) are not leaks
K_CAND = 10                      # candidates shown to the PINPOINT model
CTX = 1                          # lines of context above / below each candidate
SYS_PIN = ("[ROLE: PINPOINT] You are Nupen's bug locator. A test fails after a change. The numbered candidate lines are the most suspicious "
           "lines (each with one line of context above and below; >>> marks the candidate). Reply with the number of the candidate that "
           "holds the bug and nothing else.")
SYS_FIX = ("[ROLE: DEBUG] You are Nupen's debugger. A test fails because of one wrong line. Reply with one short line starting 'REASONING:' "
           "saying what is wrong, then ONLY one search/replace edit:\nFILE: <path>\n<<<<<<< SEARCH\n<exact existing lines>\n=======\n"
           "<replacement lines>\n>>>>>>> REPLACE\nSEARCH text must match the file exactly and be unique. Change as little as possible; "
           "never touch the tests.")
CODE_SYSTEM = "You are a careful Python programmer. Reply with the complete function(s) in one ```python code block and nothing else."
DESC = {"cmp_flip": "a comparison operator is wrong", "off_by_one": "a constant is off by one", "swap_var": "the wrong variable is used",
        "drop_return": "the returned value was replaced by None", "negate_cond": "the condition is inverted",
        "arith_flip": "an arithmetic operator is wrong", "bool_flip": "'and' and 'or' are swapped", "in_flip": "'in' and 'not in' are swapped",
        "aug_flip": "an augmented assignment has the wrong sign", "const_flip": "a True/False constant is flipped",
        "drop_not": "a 'not' was dropped"}


# ----------------------------------------------------------------------------------------------------------------- snapshot
def _heldout(repo: Path) -> dict[str, Any]:
    p = repo / "creator" / "codetrust_heldout.json"
    return dict(json.loads(p.read_text(encoding="utf-8")))


def precut_commit(repo: str | Path) -> tuple[str, float]:
    """(sha, committer unix time) of the newest commit strictly before the trust-gate cut; asserts it is not a held-out commit."""
    repo = Path(repo)
    h = _heldout(repo)
    cut = float(h["cut_ts"])
    sha = subprocess.run(["git", "rev-list", "-1", f"--before={int(cut) - 1}", "HEAD"], cwd=str(repo), capture_output=True, text=True,
                         check=True).stdout.strip()
    ts = float(subprocess.run(["git", "log", "-1", "--format=%ct", sha], cwd=str(repo), capture_output=True, text=True, check=True).stdout)
    assert ts < cut, f"snapshot commit {sha[:9]} is not before the cut"
    assert not any(s.startswith(sha) or sha.startswith(s) for s in (h.get("held_out_commits") or [])), "snapshot commit is held out"
    return sha, ts


def make_snapshot(repo: str | Path, dest: str | Path, *, extract: bool = True) -> dict[str, Any]:
    """`git archive` of the last pre-cut commit (only SNAPSHOT_PATHS) into dest + dest/snapshot_meta.json."""
    repo, dest = Path(repo), Path(dest)
    sha, ts = precut_commit(repo)
    if extract:
        dest.mkdir(parents=True, exist_ok=True)
        with subprocess.Popen(["git", "archive", "--format=tar", sha, "--", *SNAPSHOT_PATHS], cwd=str(repo), stdout=subprocess.PIPE) as pr:
            assert pr.stdout is not None
            with tarfile.open(fileobj=pr.stdout, mode="r|") as tf:
                tf.extractall(dest)
    meta = {"commit": sha, "committed_unix": ts, "cut_utc": CUT_UTC, "paths": list(SNAPSHOT_PATHS)}
    (dest / "snapshot_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


# ------------------------------------------------------------------------------------------------------------------- guard
def _nl(line: str) -> str:
    return " ".join(line.split()).lower()


class Guard:
    """What must never reach a training row: the 32-task baseline suite and anything derived from it."""

    def __init__(self, suite_dir: str | Path, exclude_json: Sequence[str | Path] = ()) -> None:
        sd = Path(suite_dir)
        tasks = json.loads((sd / "tasks.json").read_text(encoding="utf-8"))["tasks"]
        self.ids = {str(t["id"]) for t in tasks}
        self.names = {str(t["name"]) for t in tasks if t.get("name")}
        self.texts: list[str] = []
        for t in tasks:
            for k in ("request", "query", "accept", "stub", "examples", "tests"):
                if t.get(k):
                    self.texts.append(str(t[k]))
        self.lines: set[str] = set()
        self.files: list[str] = []
        for p in sorted(sd.rglob("*")):
            if p.is_file() and p.suffix in (".py", ".md", ".txt", ".json") and p.name != "tasks.json":
                self.files.append(p.name)
                self.texts.append(p.read_text(encoding="utf-8", errors="replace"))
        self.src_paths: set[str] = set()
        for ej in exclude_json:                      # further eval suites (e.g. eval_suite200/EXCLUDE.json): ids, names, texts, source paths
            ex = json.loads(Path(ej).read_text(encoding="utf-8"))
            self.ids |= {str(i) for i in ex.get("ids") or []}
            self.names |= {str(n) for n in ex.get("names") or []}
            self.src_paths |= {str(p[1]) for p in ex.get("source_paths") or [] if len(p) > 1}
            self.texts += [str(t[1]) for t in ex.get("texts") or [] if len(t) > 1]
        for t in self.texts:
            for ln in t.splitlines():
                n = _nl(ln)
                if len(n) >= LINE_MIN:
                    self.lines.add(n)
        self.name_re = re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(re.escape(n) for n in sorted(self.names, key=len, reverse=True)) + r")(?![A-Za-z0-9])") if self.names else None
        self.id_re = re.compile("|".join(re.escape(i) for i in sorted(self.ids, key=len, reverse=True))) if self.ids else None
        from creator import trainmix as TM
        self.TM = TM
        self.near = TM.NearIndex(0.6)
        for i, t in enumerate(self.texts):
            if len(t) >= 80:
                self.near.add(f"suite:{i}", t)

    def id_violation(self, cid: str) -> str | None:
        if cid in self.ids:
            return "task id is in an eval suite"
        if any(p in cid for p in self.src_paths):
            return "taken from a source file of an eval suite"
        return None

    def violation(self, text: str) -> str | None:
        low = text.lower()
        if "baseline_suite" in low or "minishop" in low:
            return "mentions the baseline suite / minishop"
        if self.id_re and self.id_re.search(text):
            return "contains a suite task id"
        if self.name_re and self.name_re.search(text):
            return "mentions a suite function name"
        for ln in text.splitlines():
            if _nl(ln) in self.lines:
                return "contains a line of the suite's sources/tests"
        if len(text) >= 80 and self.near.match(text):
            return "near-duplicate of a suite text"
        return None


def audit(rows: Iterable[Mapping[str, Any]], guard: Guard, snapshot_meta: Mapping[str, Any] | None = None, heldout: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Raise AssertionError when any row leaks; returns the counts when clean."""
    n = 0
    for r in rows:
        n += 1
        text = "\n".join(str(m.get("content") or "") for m in r.get("messages") or [])
        v = guard.violation(text)
        assert v is None, f"row {r.get('id')}: {v}"
        meta = r.get("meta") or {}
        if meta.get("tree") == "precut":
            assert snapshot_meta is not None and float(snapshot_meta["committed_unix"]) < float((heldout or {"cut_ts": 1790891910.0})["cut_ts"]), \
                f"row {r.get('id')}: tree row without a pre-cut snapshot"
            assert meta.get("commit") == snapshot_meta["commit"], f"row {r.get('id')}: wrong snapshot commit"
        else:
            assert meta.get("public") is True, f"row {r.get('id')}: neither a pre-cut tree row nor a public row"
    return {"rows": n, "clean": True}


# --------------------------------------------------------------------------------------------------------- planted-bug rows
def _line_at(src: str, ln: int) -> str:
    lines = src.splitlines()
    return lines[ln - 1] if 1 <= ln <= len(lines) else ""


def _enclosing(src: str, ln: int) -> tuple[int, int] | None:
    """(start, end) lines of the innermost def containing line ln."""
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return None
    best: tuple[int, int] | None = None
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.lineno <= ln <= (n.end_lineno or n.lineno):
            if best is None or n.lineno >= best[0]:
                best = (n.lineno, n.end_lineno or n.lineno)
    return best


def _short(s: str, n: int) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n - 1] + "~"


def _failure_header(row: Mapping[str, Any]) -> tuple[str, str]:
    from creator.tools import pinpoint as P
    tb = P.parse_traceback(str(row.get("traceback") or ""), None)
    err = f"{tb.exc_type}: {tb.message}".strip(": ") if tb.exc_type else ""
    if tb.assert_line:
        err = (err + "  | " if err else "") + tb.assert_line
    return str(row["failing_tests"][0]) if row.get("failing_tests") else "", _short(err, 220)


def pinpoint_row(row: Mapping[str, Any], mutated: str, k: int = K_CAND, ctx: int = CTX) -> dict[str, Any] | None:
    """One PINPOINT chat row, or None when the true line is not among the first k candidates (by the cheap pre-score order)."""
    cand = row.get("cand") or {}
    keys = cand.get("keys") or []
    xs = cand.get("x") or []
    f, true_line = row["file"], int(row["true_line"])
    top = [(kk[0], int(kk[1]), x) for kk, x in zip(keys[:k], xs[:k])]
    idx = next((i for i, (ff, ll, _x) in enumerate(top) if ff == f and ll == true_line), None)
    if idx is None:
        return None
    srcs: dict[str, list[str]] = {f: mutated.splitlines()}
    blocks = []
    for i, (ff, ll, x) in enumerate(top, 1):
        lines = srcs.get(ff)
        if lines is None:
            return None                                              # a candidate in another file: its text is not in this row's snapshot
        lo, hi = max(1, ll - ctx), min(len(lines), ll + ctx)
        body = "\n".join(f"{'>>>' if j == ll else '   '} {j}| {lines[j - 1].rstrip()[:160]}" for j in range(lo, hi + 1))
        blocks.append(f"[{i}] {ff}:{ll} (in {int(x[0])}/{int(x[2])} failing tests)\n{body}")
    test, err = _failure_header(row)
    user = f"Failing test: {test}\nError: {err}\nCandidates:\n" + "\n".join(blocks) + "\nWhich candidate holds the bug?"
    return {"messages": [{"role": "system", "content": SYS_PIN}, {"role": "user", "content": user},
                         {"role": "assistant", "content": str(idx + 1)}],
            "label": idx + 1, "k": len(top)}


def _unique_edit(orig: str, mutated: str, ln: int) -> tuple[str, str] | None:
    """(SEARCH, REPLACE) texts for the single planted line: grow the window of whole lines until the SEARCH text is unique in the mutated file."""
    mut = mutated.splitlines()
    org = orig.splitlines()
    if len(mut) != len(org) or not (1 <= ln <= len(mut)):
        return None
    for w in range(0, 6):
        lo, hi = max(1, ln - w), min(len(mut), ln + w)
        s = "\n".join(mut[lo - 1:hi])
        if s.strip() and mutated.count(s) == 1:
            return s, "\n".join(org[lo - 1:hi])
    return None


def debug_fix_row(row: Mapping[str, Any], orig: str, mutated: str, max_fn_lines: int = 70) -> dict[str, Any] | None:
    ln = int(row["true_line"])
    edit = _unique_edit(orig, mutated, ln)
    if edit is None:
        return None
    span = _enclosing(mutated, ln) or (max(1, ln - 20), ln + 20)
    lo, hi = span
    if hi - lo + 1 > max_fn_lines:
        lo, hi = max(lo, ln - max_fn_lines // 2), min(hi, ln + max_fn_lines // 2)
    mut = mutated.splitlines()
    code = "\n".join(f"{j}| {mut[j - 1]}" for j in range(lo, min(hi, len(mut)) + 1))
    test, err = _failure_header(row)
    tb = str(row.get("traceback") or "")
    tail = "\n".join(tb.strip().splitlines()[-12:])[-900:]
    user = f"FILE: {row['file']}\n```python\n{code}\n```\nFailing test: {test}\nError: {err}\nTraceback (end):\n{tail}"
    kind = str(row.get("mutation") or "")
    answer = (f"REASONING: {DESC.get(kind, 'one line is wrong')} (line {ln}); restore the original.\nFILE: {row['file']}\n<<<<<<< SEARCH\n"
              f"{edit[0]}\n=======\n{edit[1]}\n>>>>>>> REPLACE")
    return {"messages": [{"role": "system", "content": SYS_FIX}, {"role": "user", "content": user}, {"role": "assistant", "content": answer}]}


def split_of(file: str, files: Sequence[str], test_frac: float = 0.2) -> str:
    from creator.tools import pinpoint_feat as PF
    _train, test = PF.split_modules(list(files), test_frac=test_frac)
    return "eval" if file in test else "train"


def build_planted(ds_paths: Sequence[str | Path], snapshot: str | Path, out_dir: str | Path, guard: Guard, *, max_rows: int | None = None,
                  progress: Any = None) -> dict[str, Any]:
    """PINPOINT + DEBUG_FIX rows from generator output (rows with src_sha) over the pre-cut snapshot."""
    snap = Path(snapshot)
    meta = json.loads((snap / "snapshot_meta.json").read_text(encoding="utf-8"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: set[str] = set()
    rows: list[dict[str, Any]] = []
    for p in ds_paths:
        with open(p, encoding="utf-8") as fh:
            for ln in fh:
                if ln.strip():
                    r = json.loads(ln)
                    if r.get("src_sha"):
                        rows.append(r)
                        files.add(r["file"])
    srcs: dict[str, str] = {}
    drops: collections.Counter[str] = collections.Counter()
    seen: set[str] = set()
    stats: collections.Counter[str] = collections.Counter()
    by_file: collections.Counter[str] = collections.Counter()
    fp, fd = open(out / "pinpoint.jsonl", "w", encoding="utf-8"), open(out / "debug_fix.jsonl", "w", encoding="utf-8")
    try:
        for i, r in enumerate(rows):
            if max_rows and stats["bugs"] >= max_rows:
                break
            f = r["file"]
            if f not in srcs:
                srcs[f] = (snap / f).read_text(encoding="utf-8")
            orig = srcs[f]
            if hashlib.sha1(orig.encode("utf-8")).hexdigest()[:12] != r["src_sha"]:
                drops["source differs from the snapshot"] += 1
                continue
            from creator.tools import pinpoint as P
            m = r["mutant"]
            mutated = P.apply_mutant(orig, P.Mutant(r["mutation"], r["true_line"], m["start"], m["end"], m["new"]))
            stats["bugs"] += 1
            split = split_of(f, sorted(files))
            base_meta = {"tree": "precut", "commit": meta["commit"], "file": f, "line": r["true_line"], "mutation": r["mutation"]}
            for kind, build, fh in (("pinpoint", lambda: pinpoint_row(r, mutated), fp), ("debug_fix", lambda: debug_fix_row(r, orig, mutated), fd)):
                row = build()
                if row is None:
                    drops[f"{kind}: no row (true line outside the top-{K_CAND} / no unique edit)"] += 1
                    continue
                text = "\n".join(x["content"] for x in row["messages"])
                v = guard.violation(text)
                if v:
                    drops[f"{kind}: guard: {v}"] += 1
                    continue
                key = hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]
                if key in seen:
                    drops[f"{kind}: exact duplicate"] += 1
                    continue
                seen.add(key)
                rec = {"id": f"{kind}:{f}:{r['true_line']}:{key[:8]}", "source": f"planted_{kind}", "kind": kind, "split": split,
                       "messages": row["messages"], "meta": {**base_meta, **{k: v for k, v in row.items() if k in ("label", "k")}}}
                fh.write(json.dumps(rec) + "\n")
                stats[kind] += 1
                stats[f"{kind}/{split}"] += 1
                by_file[f] += 1
            if progress and i % 5000 == 0:
                progress(i, dict(stats))
    finally:
        fp.close()
        fd.close()
    return {"bugs": stats["bugs"], "counts": dict(stats), "drops": dict(drops), "modules": len(files), "rows_by_module_top": by_file.most_common(8)}


# ------------------------------------------------------------------------------------------------------------- CODE rows
RUNNER = r"""
import json, sys
prog = sys.stdin.buffer.read().decode("utf-8")
ns = {"__name__": "__verify__"}
try:
    exec(compile(prog, "<row>", "exec"), ns)
except BaseException as e:
    print("FAIL", type(e).__name__); raise SystemExit(0)
print("PASS")
"""
IDLE = 0x00000040


def run_program(prog: str, timeout: float = 10.0) -> bool:
    flags = IDLE if sys.platform == "win32" else 0
    with tempfile.TemporaryDirectory() as td:
        try:
            p = subprocess.run([sys.executable, "-I", "-c", RUNNER], input=prog.encode("utf-8"), capture_output=True, timeout=timeout, cwd=td,
                               creationflags=flags)
        except subprocess.TimeoutExpired:
            return False
    return p.stdout.decode("utf-8", "replace").strip().splitlines()[-1:] == ["PASS"]      # the candidate may print: only the runner's last line counts


def _cases_program(name: str, tests: Any, code: str) -> str:
    cases = json.dumps(tests)
    return (f"{code}\n\nimport json as _j\n_f = {name}\nfor _t in _j.loads({cases!r}):\n"
            f"    assert _j.loads(_j.dumps(_f(*_t['args']))) == _t['expect']\n")


def _code_of(text: str) -> str:
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def code_candidates(gpuday: Path) -> Iterable[dict[str, Any]]:
    """Unverified CODE candidates from the sources the training mixes already use. Each: {source, id, system, prompt, code, program, licence}."""
    ex = gpuday / "export"

    def jl(p: Path) -> list[dict[str, Any]]:
        out = []
        if p.is_file():
            for ln in p.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    out.append(json.loads(ln))
        return out
    for name in ("rl_tasks_hf", "rl_tasks_more"):
        rows, ids = jl(ex / f"{name}.jsonl"), jl(ex / f"{name}.ids.jsonl")
        byid = {m.get("id"): m for m in ids}
        for r in rows:
            m = byid.get(r["id"]) or {}
            ref = str(m.get("reference") or "").strip()
            if not ref:
                continue
            yield {"source": name, "id": r["id"], "split": r.get("split"), "prompt": str(r["prompt"]), "code": ref,
                   "program": _cases_program(str(r["name"]), r["tests"], ref), "licence": str(m.get("licence") or ""),
                   "group": str(r["id"]), "fname": str(r["name"])}
    for r in jl(gpuday / "distill" / "ladder_distill.jsonl"):
        mt = r.get("meta") or {}
        if mt.get("suite") != "coding" or mt.get("kind") != "mbpp":
            continue
        msgs = r["messages"]
        user = next((x["content"] for x in msgs if x["role"] == "user"), "")
        ans = next((x["content"] for x in reversed(msgs) if x["role"] == "assistant"), "")
        asserts = [ln for ln in user.splitlines() if ln.strip().startswith("assert ")]
        code = _code_of(ans)
        if not asserts or not code:
            continue
        yield {"source": "ladder_mbpp", "id": str(mt.get("id")), "split": "train", "prompt": user, "code": code,
               "program": code + "\n\n" + "\n".join(asserts) + "\n", "licence": "MBPP CC-BY-4.0 prompt; answer by a larger open model, re-verified here",
               "group": str(mt.get("id")), "fname": ""}


def build_code(gpuday: Path, out_dir: Path, guard: Guard, heldout: Mapping[str, Any], workers: int = 3, progress: Any = None) -> dict[str, Any]:
    """Verified CODE rows (train side only): the reference is re-run against the tests; near-duplicates (trainmix.NearIndex) and anything the
    guard flags are dropped; the trust-gate eval sets (trainmix.eval_sets-style) are checked by prompt near-dup."""
    from concurrent.futures import ThreadPoolExecutor
    from creator import trainmix as TM
    out_dir.mkdir(parents=True, exist_ok=True)
    cands = [c for c in code_candidates(gpuday) if c.get("split") != "eval"]
    drops: collections.Counter[str] = collections.Counter()
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        verdicts = list(ex.map(lambda c: run_program(c["program"]), cands))
    near = TM.NearIndex()
    held = TM.NearIndex()
    for t in heldout.get("tasks") or []:
        if str(t.get("task") or "") and len(str(t["task"])) >= 20:
            held.add(f"held:{t.get('id')}", str(t["task"]))
    counts: collections.Counter[str] = collections.Counter()
    verified: collections.Counter[str] = collections.Counter()
    with open(out_dir / "code.jsonl", "w", encoding="utf-8") as fh:
        for c, ok in zip(cands, verdicts):
            if not ok:
                drops[f"{c['source']}: reference fails its tests"] += 1
                continue
            verified[c["source"]] += 1
            if not TM._code_ok(c["code"]):
                drops[f"{c['source']}: does not parse"] += 1
                continue
            text = c["prompt"] + "\n" + c["code"]
            v = guard.id_violation(str(c["id"])) or guard.violation(text)
            if v:
                drops[f"{c['source']}: guard: {v}"] += 1
                continue
            if held.match(c["prompt"]):
                drops[f"{c['source']}: near-duplicate of a trust-gate held-out task"] += 1
                continue
            k = near.match(text)
            if k:
                drops[f"{c['source']}: near-duplicate of {k.split(':', 1)[0]}"] += 1
                continue
            near.add(f"{c['source']}:{c['id']}", text)
            msgs = [{"role": "system", "content": CODE_SYSTEM}, {"role": "user", "content": c["prompt"]},
                    {"role": "assistant", "content": f"```python\n{c['code']}\n```"}]
            fh.write(json.dumps({"id": f"code:{c['id']}", "source": c["source"], "kind": "code", "split": "train", "messages": msgs,
                                 "meta": {"public": True, "licence": c["licence"], "verified_by": "re-executed reference against the task tests",
                                          "group": c["group"]}}) + "\n")
            counts[c["source"]] += 1
    return {"candidates": len(cands), "verified_pass": dict(verified), "kept": dict(counts), "kept_total": sum(counts.values()),
            "drops": dict(drops), "verify_s": round(time.monotonic() - t0, 1)}


def read_rows(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for ln in fh:
                if ln.strip():
                    yield json.loads(ln)


# --------------------------------------------------------------------------------------------------------------------- CLI
def _main(argv: Sequence[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="phase2data")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("--repo", default=".")
    s.add_argument("--dest", required=True)
    s.add_argument("--no-extract", action="store_true")
    b = sub.add_parser("planted")
    b.add_argument("--ds", action="append", required=True)
    b.add_argument("--snapshot", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--suite", required=True)
    b.add_argument("--exclude", action="append", default=[])
    b.add_argument("--max-bugs", type=int, default=None)
    c = sub.add_parser("code")
    c.add_argument("--gpuday", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--suite", required=True)
    c.add_argument("--exclude", action="append", default=[])
    c.add_argument("--repo", default=".")
    c.add_argument("--workers", type=int, default=3)
    a = sub.add_parser("audit")
    a.add_argument("--out", required=True)
    a.add_argument("--suite", required=True)
    a.add_argument("--exclude", action="append", default=[])
    a.add_argument("--snapshot", required=True)
    a.add_argument("--repo", default=".")
    ns = ap.parse_args(list(argv))
    if ns.cmd == "snapshot":
        print(json.dumps(make_snapshot(ns.repo, ns.dest, extract=not ns.no_extract)))
    elif ns.cmd == "planted":
        g = Guard(ns.suite, ns.exclude)
        res = build_planted(ns.ds, ns.snapshot, ns.out, g, max_rows=ns.max_bugs, progress=lambda i, st: print(i, st, flush=True))
        print(json.dumps(res, indent=1))
    elif ns.cmd == "code":
        g = Guard(ns.suite, ns.exclude)
        res = build_code(Path(ns.gpuday), Path(ns.out), g, _heldout(Path(ns.repo)), ns.workers)
        print(json.dumps(res, indent=1))
    else:
        g = Guard(ns.suite, ns.exclude)
        out = Path(ns.out)
        meta = json.loads((Path(ns.snapshot) / "snapshot_meta.json").read_text(encoding="utf-8"))
        res = audit(read_rows(sorted(out.glob("*.jsonl"))), g, meta, _heldout(Path(ns.repo)))
        print(json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
