"""R13: build the ~200-task held-out suite (same format as baseline_suite/ so scripts/pc_baseline.py runs it unchanged).

  python scripts/lowprio.py --idle python scripts/build_suite200.py build  [--out <dir>] [--src <app task sources dir>]
  python scripts/lowprio.py --idle python scripts/build_suite200.py verify [--out <dir>]     (reference passes, stub fails; no model runs)

Outside the repo (held-out material never enters it): <runtime>/gpuday/eval_suite200/{tasks.json, app_base/, refs.json, EXCLUDE.json, SUMMARY.json, src/}.
Function tasks = the eval split of the rl-task exports that the 32-task suite was NOT drawn from (public sympy/requests/pydantic functions + the
public OpenCodeInstruct eval functions; their reference solutions are in the export ids files and are re-run here against the hidden tests).
App tasks = small features on the same minishop app_base, written in <out>/src/app_*.py (TASKS lists of {id, request, accept, ref}); the reference
edits are applied to a copy of app_base and the hidden acceptance tests must pass, and must fail on the untouched base.
EXCLUDE.json lists everything the training side must never see (creator.tools.evalexclude reads it)."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from creator.tools import evalexclude as EX  # noqa: E402
from creator.tools import reuse as RU  # noqa: E402
import pc_baseline as PB  # noqa: E402

PER_BAND = 50
MIN_TESTS = 5


def gpuday() -> Path:
    return Path.home() / "creator_runtime" / "gpuday"


def jl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------------------ function tasks
def ref_fn(ref: str, name: str) -> ast.FunctionDef | None:
    for n in ast.walk(ast.parse(ref)):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    return None


def code_lines(ref: str, name: str) -> int:
    """Reference length: non-blank, non-comment lines of the function, docstring excluded."""
    fn = ref_fn(ref, name)
    assert fn is not None
    body = fn.body[1:] if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(getattr(fn.body[0], "value", None), ast.Constant) \
        and isinstance(fn.body[0].value.value, str) else fn.body
    start = fn.lineno
    end = max(getattr(n, "end_lineno", start) for n in body) if body else start
    src = ref.splitlines()[start - 1:end]
    # drop the docstring lines when it is first
    if len(body) != len(fn.body):
        d = fn.body[0]
        src = [ln for i, ln in enumerate(ref.splitlines()[start - 1:end], start) if not (d.lineno <= i <= d.end_lineno)]
    return len([x for x in src if x.strip() and not x.strip().startswith("#")])


def signature(ref: str, name: str) -> str:
    fn = ref_fn(ref, name)
    assert fn is not None
    lines = ref.splitlines()
    first = fn.lineno
    last = fn.body[0].lineno - 1 if fn.body else first
    if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(getattr(fn.body[0], "value", None), ast.Constant):
        last = fn.body[0].lineno - 1
    seg = lines[first - 1:max(last, first)]
    return "\n".join(seg).rstrip()


def fn_task(r: dict[str, Any], m: dict[str, Any]) -> dict[str, Any] | None:
    name = r["name"]
    ref = m["reference"]
    if r["id"].startswith("pub:"):
        mm = re.search(r"\n\n(def .*?)(\n\nExamples:\n.*)$", r["prompt"], re.S)
        if not mm or not re.search(r'"""[^"]{20,}', r["prompt"]):
            return None
        sig, ex = mm.group(1), mm.group(2).strip().split("\n", 1)[1]
    else:
        mm = re.search(r"^(.*?)\n\nExamples:\n(.*)$", r["prompt"].split("\n\n", 1)[1], re.S)
        if not mm:
            return None
        desc, ex = mm.group(1).strip(), mm.group(2).strip()
        head = signature(ref, name)
        doc = desc.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
        sig = f'{head}\n    """{doc}"""'
    return {"id": "fn." + name, "family": "fn", "name": name, "query": f"{name} {sig[:300]}",
            "request": f"Write the Python function `{name}` described below (standard library only; return the function definition in one ```python block).\n\n{sig}",
            "examples": ex + "\n(`null` means None; lists may stand for tuples.)",
            "stub": PB.FN_PRE + (PB.Z_LINE if re.search(r"=\s*Z[,)]", sig) else "") + sig.rstrip() + "\n    raise NotImplementedError\n",
            "tests": r["tests"], "n_visible": ex.count("\n") + 1 - 1}


def run_check(py: str, code: str, name: str, tests: list[dict[str, Any]]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        (dp / "solution.py").write_text(code, encoding="utf-8")
        (dp / "t.json").write_text(json.dumps(tests), encoding="utf-8")
        (dp / "chk.py").write_text(PB.CHECK, encoding="utf-8")
        try:
            p = subprocess.run([py, "chk.py", "t.json", name], cwd=d, capture_output=True, text=True, timeout=60)
            return json.loads(p.stdout.strip().splitlines()[-1])
        except Exception as e:                                                       # noqa: BLE001
            return {"passed": 0, "total": len(tests), "fails": [type(e).__name__]}


def build_fns(py: str) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    gd = gpuday()
    suite_names = {t["name"] for t in json.loads((gd / "baseline_suite" / "tasks.json").read_text(encoding="utf-8"))["tasks"] if t["family"] == "fn"}
    rows: dict[str, dict[str, Any]] = {}
    meta: dict[str, dict[str, Any]] = {}
    for n in ("rl_tasks_hf", "rl_tasks_more"):
        for r in jl(gd / "export_plus" / f"{n}.jsonl"):
            rows[r["id"]] = r
        for m in jl(gd / "export_plus" / f"{n}.ids.jsonl"):
            meta[m["id"]] = m
    pool = []
    seen_names: set[str] = set(suite_names)
    rej: dict[str, int] = {}
    for rid in sorted(rows, key=lambda x: sha(x)):
        r = rows[rid]
        if r["split"] != "eval" or r["name"] in seen_names or rid not in meta or not meta[rid].get("reference"):
            continue
        if len(r["tests"]) < MIN_TESTS:
            rej["tests<5"] = rej.get("tests<5", 0) + 1
            continue
        t = fn_task(r, meta[rid])
        if t is None:
            rej["no signature/doc"] = rej.get("no signature/doc", 0) + 1
            continue
        ref = meta[rid]["reference"]
        good = run_check(py, ref, t["name"], t["tests"])
        bad = run_check(py, t["stub"], t["name"], t["tests"])
        if good["passed"] != good["total"]:
            rej["reference fails its tests"] = rej.get("reference fails its tests", 0) + 1
            continue
        if bad["passed"] != 0:
            rej["stub passes a test"] = rej.get("stub passes a test", 0) + 1
            continue
        seen_names.add(r["name"])
        t["_lines"], t["_src"], t["_ref"] = code_lines(ref, t["name"]), rid, ref
        t["_kind"] = "pub" if rid.startswith("pub:") else "hf"
        pool.append(t)
    lens = sorted(t["_lines"] for t in pool)
    c1, c2 = lens[len(lens) // 3], lens[2 * len(lens) // 3]
    band = lambda n: "easy" if n <= c1 else ("medium" if n <= c2 else "hard")        # noqa: E731
    by: dict[str, list[dict[str, Any]]] = {"easy": [], "medium": [], "hard": []}
    for t in pool:
        t["_band"] = band(t["_lines"])
        by[t["_band"]].append(t)
    chosen: list[dict[str, Any]] = []
    for b, ts in by.items():
        ts.sort(key=lambda t: (t["_kind"] != "pub", sha(t["id"])))                  # public-repo functions first, then the others by hash
        chosen += ts[:PER_BAND]
    chosen.sort(key=lambda t: t["id"])
    stats = {"pool": len(pool), "rejected": rej, "cuts": [c1, c2], "bands": {b: len([t for t in chosen if t["_band"] == b]) for b in by}}
    return chosen, stats, {"rows": rows, "meta": meta}


# ------------------------------------------------------------------------------------------------------------ app tasks
def apply_ref(base: Path, edits: list[tuple]) -> None:
    for e in edits:
        p = base / e[1]
        if e[0] == "edit":
            s = p.read_text(encoding="utf-8")
            assert s.count(e[2]) == 1, f"anchor not unique/found in {e[1]}: {e[2][:50]!r}"
            p.write_text(s.replace(e[2], e[3]), encoding="utf-8")
        elif e[0] == "append":
            with p.open("a", encoding="utf-8") as fh:
                fh.write(e[2])
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(e[2], encoding="utf-8")


def count_checks(accept: str) -> int:
    return len(re.findall(r"^\s*assert\b", accept, re.M)) + len(re.findall(r"pytest\.raises\(", accept))


def pytest_rc(py: str, ws: Path, accept: str) -> tuple[int, str]:
    (ws / "tests" / "test_zz_acceptance.py").write_text(accept, encoding="utf-8")
    p = subprocess.run([py, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "tests"], cwd=ws, capture_output=True, text=True, timeout=180)
    return p.returncode, "\n".join(p.stdout.splitlines()[-6:])


def load_app_sources(src: Path) -> list[dict[str, Any]]:
    sys.path.insert(0, str(src))
    out: list[dict[str, Any]] = []
    for f in sorted(src.glob("app_*.py")):
        mod = importlib.import_module(f.stem)
        out += mod.TASKS
    ids = [t["id"] for t in out]
    assert len(ids) == len(set(ids)), "duplicate app task ids"
    return out


def verify_apps(py: str, base: Path, apps: list[dict[str, Any]]) -> dict[str, Any]:
    res: dict[str, Any] = {"ok": [], "bad": {}}
    for t in apps:
        with tempfile.TemporaryDirectory() as d:
            ws = Path(d) / "ws"
            shutil.copytree(base, ws, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
            rc0, tail0 = pytest_rc(py, ws, t["accept"])                              # untouched base must FAIL the acceptance test
            shutil.rmtree(ws)
            shutil.copytree(base, ws, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
            apply_ref(ws, t["ref"])
            rc1, tail1 = pytest_rc(py, ws, t["accept"])
        n = count_checks(t["accept"])
        why = []
        if rc0 == 0:
            why.append("stub passes the acceptance test")
        if rc1 != 0:
            why.append("reference fails: " + tail1.replace("\n", " | ")[-300:])
        if n < MIN_TESTS:
            why.append(f"only {n} checks")
        if why:
            res["bad"][t["id"]] = why
        else:
            res["ok"].append(t["id"])
    return res


# ------------------------------------------------------------------------------------------------------------ build
def build(a: argparse.Namespace) -> int:
    out = Path(a.out)
    src = Path(a.src or out / "src")
    py = a.python
    base = gpuday() / "baseline_suite" / "app_base"
    out.mkdir(parents=True, exist_ok=True)
    apps = load_app_sources(src)
    vr = verify_apps(py, base, apps)
    print(f"apps: {len(vr['ok'])} ok, {len(vr['bad'])} bad")
    for k, v in vr["bad"].items():
        print("  BAD", k, v)
    if vr["bad"]:
        return 1
    fns, st, _ = build_fns(py)
    print("fn pool", st)
    if (out / "app_base").exists():
        shutil.rmtree(out / "app_base")
    shutil.copytree(base, out / "app_base", ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    tasks: list[dict[str, Any]] = []
    refs: dict[str, Any] = {}
    for t in apps:
        tasks.append({"id": t["id"], "family": "app", "request": t["request"].strip(), "query": t["request"].strip(), "accept": t["accept"].lstrip("\n")})
        refs[t["id"]] = {"edits": [list(e) for e in t["ref"]], "band": t.get("band", "app")}
    ex_rows: list[dict[str, Any]] = []
    diff: dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}
    for t in fns:
        tasks.append({k: v for k, v in t.items() if not k.startswith("_")})
        refs[t["id"]] = {"reference": t["_ref"], "source": t["_src"], "ref_lines": t["_lines"], "band": t["_band"], "kind": t["_kind"]}
        diff[t["_band"]] += 1
        ex_rows.append(t)
    (out / "tasks.json").write_text(json.dumps({"tasks": tasks}, indent=1), encoding="utf-8")
    (out / "refs.json").write_text(json.dumps(refs, indent=1), encoding="utf-8")
    EX.write_exclude(out / "EXCLUDE.json", fn_tasks=ex_rows, app_tasks=apps, repo_root=ROOT, public_root=Path.home() / "creator_runtime" / "public_repos")
    n_app = len(apps)
    app_lines = sorted(sum(len(e[-1].splitlines()) for e in t["ref"]) for t in apps)
    summ = {"tasks": len(tasks), "app": n_app, "fn": len(fns), "fn_bands": diff, "fn_ref_line_cuts": st["cuts"], "fn_pool": st["pool"], "fn_rejected": st["rejected"],
            "fn_kinds": {k: len([t for t in fns if t["_kind"] == k]) for k in ("pub", "hf")},
            "fn_ref_lines": {"min": min(t["_lines"] for t in fns), "max": max(t["_lines"] for t in fns)},
            "app_ref_added_lines": {"min": app_lines[0], "median": app_lines[len(app_lines) // 2], "max": app_lines[-1]},
            "min_hidden_tests_fn": min(len(t["tests"]) for t in fns), "min_checks_app": min(count_checks(t["accept"]) for t in apps)}
    (out / "SUMMARY.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))
    return 0


def verify(a: argparse.Namespace) -> int:
    """Claude-free grader check with pc_baseline's own make_ws/grade: reference solutions must pass, untouched stubs must fail."""
    out = Path(a.out)
    py = a.python
    tasks = json.loads((out / "tasks.json").read_text(encoding="utf-8"))["tasks"]
    refs = json.loads((out / "refs.json").read_text(encoding="utf-8"))
    ref_pass = stub_pass = 0
    bad: list[str] = []
    for t in tasks:
        with tempfile.TemporaryDirectory() as d:
            ws = Path(d) / "ws"
            PB.make_ws(t, out, ws)
            g0 = PB.grade(t, out, ws, py)                                            # untouched workspace = the stub / the unmodified app
            PB.make_ws(t, out, ws)
            if t["family"] == "fn":
                (ws / "solution.py").write_text(refs[t["id"]]["reference"], encoding="utf-8")
            else:
                apply_ref(ws, [tuple(e) for e in refs[t["id"]]["edits"]])
            g1 = PB.grade(t, out, ws, py)
        stub_pass += bool(g0["passed"])
        ref_pass += bool(g1["passed"])
        if g0["passed"] or not g1["passed"]:
            bad.append(f"{t['id']}: stub={g0['passed']} ref={g1['passed']} {g1.get('detail')}")
    res = {"tasks": len(tasks), "reference_pass": ref_pass, "stub_pass": stub_pass, "bad": bad}
    (out / "VERIFY.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    return 0 if not bad else 1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "verify"])
    ap.add_argument("--out", default=str(gpuday() / "eval_suite200"))
    ap.add_argument("--src", default="")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args(argv)
    return build(a) if a.cmd == "build" else verify(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
