"""C77 sec 6 - Phase 0 reconnaissance, COMPUTED from a repository tree (checklist CR001-CR031).

`compute(root)` reads the tree, git, the checklists and the evidence files and returns one section per checklist box. Nothing is
typed from memory. The ledger carries `tree_hash`, a content hash of everything the sections are derived from, so a consumer can
tell a reconnaissance of THIS tree from a stale one (`is_current`). scripts/phase0_recon.py is the thin CLI around this module."""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

from creator import selfmodel as SM
from creator import testrun as T

SCOPE = ("engine", "scripts", "creator", "tests", "canon")
AUTHORITATIVE = ("CREATOR_MASTER_PROMPT.md", "BLUEPRINT.md", "MASTER_BLUEPRINT.md", "ALGORITHM_BLUEPRINT.md", "BIBLE.md",
                 "MASTER_EXECUTION_PROMPT.md", "ULTIMATE_MASTER_PROMPT.md", "RESEARCH_BRAIN_CONTRACT.md",
                 "SELF_LEARNING_CONTRACT.md", "PREDICTION_ERROR_ADDITION.md", "TEN_HOUR_EXECUTION_CHECKLIST.md")
TRADING_PATH = ("engine/train.py", "engine/backtest.py", "engine/live.py", "engine/livesim.py", "engine/broker.py",
                "engine/portfolio.py", "engine/tick.py", "engine/policy.py", "engine/stops.py", "engine/exits.py")
INFRA = {
    "self_learning": r"^engine/learning/",
    "research": r"^engine/research/",
    "experiments": r"(experiment|ablation|retester|repro)",
    "health_monitoring": r"(health|monitor|brain_health|run_report)",
    "compute_resources": r"(compute|resources)",
    "replay_determinism": r"(replay|repro|parity)",
    "rollback_checkpoints": r"(checkpoint|rollback|sandbox)",
    "firewalls": r"(firewall|isolation|leak_audit|scramble_audit|separation|boundary)",
    "future_leak_controls": r"(future_firewall|leak_audit|pit\.py|timeline|blind_gates)",
    "provenance": r"(provenance|claims|registry|ledger)",
}


def sh(root: Path, *args: str) -> str:
    try:
        return subprocess.run(list(args), cwd=root, capture_output=True, text=True, timeout=120, encoding="utf-8",
                              errors="replace").stdout
    except (OSError, subprocess.SubprocessError) as e:
        return f"<error {e}>"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hash(root: Path, scope: Sequence[str] = SCOPE) -> str:
    """Content hash of every input of the reconnaissance: python files in scope, root documents, CI workflows."""
    root = Path(root)
    rels = set(SM.python_files(root, scope))
    rels |= {p.name for p in root.glob("*.md")}
    rels |= {p.relative_to(root).as_posix() for p in (root / ".github" / "workflows").glob("*.yml")}
    h = hashlib.sha256()
    for rel in sorted(rels):
        h.update(f"{rel}\0{_sha(root / rel)}\n".encode())
    return h.hexdigest()


def is_current(ledger: dict[str, Any], root: Path, scope: Sequence[str] = SCOPE) -> bool:
    """True only when the ledger was computed on exactly the tree now on disk."""
    recorded = ledger.get("tree_hash")
    return bool(recorded) and recorded == tree_hash(root, scope)


def body_hash(node: ast.AST) -> str:
    body = list(getattr(node, "body", []))
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
        body = body[1:]
    return hashlib.sha256("\n".join(ast.dump(b, include_attributes=False) for b in body).encode()).hexdigest()[:16]


def duplicates(root: Path, files: list[str], min_body: int = 4) -> list[dict[str, Any]]:
    seen: dict[str, list[str]] = defaultdict(list)
    for rel in files:
        if rel.startswith("tests/"):
            continue
        try:
            tree = ast.parse((root / rel).read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and len(n.body) >= min_body:
                seen[body_hash(n)].append(f"{rel}:{n.name}")
    return [{"body": h, "sites": v} for h, v in sorted(seen.items(), key=lambda kv: -len(kv[1])) if len(set(v)) > 1]


def checklist_states(root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in sorted((root / "state" / "build").glob("*CHECKLIST*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            out[p.name] = {"error": str(e)}
            continue
        items = d.get("items", d if isinstance(d, list) else [])
        st = Counter(str(i.get("status")) for i in items if isinstance(i, dict))
        unjustified = [i.get("id") for i in items if isinstance(i, dict) and str(i.get("status")) in ("VALIDATED", "TESTED", "TESTING")
                       and not i.get("evidence")]
        out[p.name] = {"items": len(items), "status": dict(st), "done_without_evidence": len(unjustified),
                       "examples": unjustified[:8]}
    return out


def boundary_check(graph: T.ImportGraph) -> dict[str, Any]:
    """Trading-path modules must not reach engine.research (statically, incl. function-level imports)."""
    mod_of = {p: m for m, p in graph.modules.items()}
    bad = []
    for path in TRADING_PATH:
        start = mod_of.get(path)
        if not start:
            continue
        seen: set[str] = set()
        frontier = [start]
        while frontier:
            m = frontier.pop()
            if m in seen or m not in graph.modules:
                continue
            seen.add(m)
            for imp in graph.imports.get(m, set()):
                for cand in (imp, imp.rsplit(".", 1)[0]):
                    if cand in graph.modules and cand not in seen:
                        frontier.append(cand)
        hits = sorted(m for m in seen if m.startswith("engine.research"))
        if hits:
            bad.append({"from": path, "reaches": hits[:10]})
    return {"rule": "trading path never imports engine.research (statically, incl. function-level imports)", "violations": bad}


def ci(root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for wf in sorted((root / ".github" / "workflows").glob("*.yml")):
        text = wf.read_text(encoding="utf-8")
        out[wf.name] = {"runs": [ln.split("run:", 1)[1].strip()[:160] for ln in text.splitlines() if "run:" in ln],
                        "secrets": sorted(set(re.findall(r"secrets\.([A-Z0-9_]+)", text)))}
    return out


def documents(root: Path, masterstock: Optional[Path]) -> dict[str, Any]:
    """Every authoritative document with its sha256; `missing` names the ones absent (a reading list that fails when incomplete)."""
    present = {n: _sha(root / n) for n in AUTHORITATIVE if (root / n).is_file()}
    return {
        "authoritative": present, "missing": [n for n in AUTHORITATIVE if n not in present],
        "blueprints": sorted(p.name for p in root.glob("*BLUEPRINT*.md")),
        "contracts": sorted(p.name for p in root.glob("*.md") if p.name != "README.md"),
        "checklists": sorted(p.name for p in (root / "state" / "build").glob("*CHECKLIST*.json")),
        "masterstock": str(masterstock) if masterstock else None,
        "masterstock_sha256": _sha(masterstock) if masterstock and masterstock.is_file() else None,
        "masterstock_bytes": masterstock.stat().st_size if masterstock and masterstock.is_file() else None}


def known_failures(root: Path, collection_errors: list[str]) -> dict[str, Any]:
    """Test files whose latest recorded outcome is not PASS (state/creator/test_evidence.json) plus collection errors."""
    ev_path = root / "state" / "creator" / "test_evidence.json"
    failing: dict[str, str] = {}
    note = "no test_evidence.json recorded"
    if ev_path.is_file():
        try:
            ev = json.loads(ev_path.read_text(encoding="utf-8"))
            failing = {k: str(v.get("outcome")) for k, v in sorted(ev.items()) if str(v.get("outcome")) != "PASS"}
            note = f"{len(ev)} test file(s) with recorded evidence"
        except (OSError, json.JSONDecodeError, AttributeError) as e:
            note = f"unreadable test_evidence.json: {e}"
    return {"note": note, "failing_test_files": failing, "collection_errors": collection_errors[:20]}


def processes() -> list[str]:
    ps = sh(Path.cwd(), "powershell", "-NoProfile", "-Command",
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object -ExpandProperty CommandLine")
    return [ln.strip()[:200] for ln in ps.splitlines() if "weekly7" in ln.lower()]


def compute(root: Path, scope: Sequence[str] = SCOPE, *, masterstock: Optional[Path] = None, collect: bool = False,
            external: bool = True) -> dict[str, Any]:
    """The reconnaissance ledger of `root`. `external=False` skips machine-level probes (processes, canon/provenance verify)."""
    root = Path(root).resolve()
    t0 = time.time()
    files = [f for f in SM.python_files(root, scope) if not f.startswith("creator/devbench/")]   # benchmark material, not code
    model = SM.build(root, scope=scope, capabilities=[], include_versions=False)
    model = dataclasses.replace(model, components={k: v for k, v in model.components.items() if k in set(files)},
                                limitations=tuple(x for x in model.limitations if not x.subject.startswith("creator/devbench/")))
    graph = T.ImportGraph.build(root)
    pkg_lines: Counter[str] = Counter()
    pkg_files: Counter[str] = Counter()
    for rel, c in model.components.items():
        top = "/".join(rel.split("/")[:2]) if rel.startswith("engine/") and rel.count("/") >= 2 else rel.split("/")[0]
        pkg_lines[top] += c.meaningful
        pkg_files[top] += 1
    lims = Counter(x.kind for x in model.limitations)
    tests = [p for p in files if T.is_test_file(p)]
    test_funcs = sum(1 for p in tests for i in model.components[p].interfaces if i.kind == "function" and i.name.startswith("test"))
    untested = sorted(x.subject for x in model.limitations if x.kind == "UNTESTED_MODULE")

    def entry_point(rel: str) -> bool:
        return "__name__ ==" in (root / rel).read_text(encoding="utf-8", errors="replace")
    unreached = sorted(x.subject for x in model.limitations if x.kind == "UNREACHED" and not entry_point(x.subject))
    stubs = [f"{x.subject}:{x.detail}" for x in model.limitations if x.kind == "STUB"]
    shallow = sorted(p for p, c in model.components.items() if not T.is_test_file(p) and c.interfaces and c.meaningful < 15
                     and not p.endswith("__init__.py"))
    infra = {k: sorted(p for p in files if re.search(rx, p) and not T.is_test_file(p)) for k, rx in INFRA.items()}
    integrity = "SKIPPED"
    canon = "SKIPPED"
    if external:
        sys.path.insert(0, str(root))
        try:
            from engine import provenance as PV
            PV.verify_integrity()
            integrity = "OK"
        except Exception as e:                                          # noqa: BLE001 - reported
            integrity = f"FAILED: {type(e).__name__}: {e}"
        canon = sh(root, sys.executable, "canon/build_canon.py", "--verify").strip()[-300:]
    collect_line = ""
    errors: list[str] = []
    if collect:
        out = sh(root, sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "tests")
        collect_line = out.strip().splitlines()[-1] if out.strip() else ""
        errors = [ln for ln in out.splitlines() if ln.startswith("ERROR")]
    stale = checklist_states(root)
    return {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "scope": list(scope),
        "tree_hash": tree_hash(root, scope), "selfmodel_digest": model.digest(),
        "CR001-004_documents": documents(root, masterstock),
        "CR005_structure": {"files": len(files), "file_list": sorted(files), "by_package_files": dict(pkg_files.most_common()),
                            "by_package_meaningful_lines": dict(pkg_lines.most_common())},
        "CR006_branch_state": {"head": sh(root, "git", "rev-parse", "HEAD").strip(),
                               "branch": sh(root, "git", "branch", "--show-current").strip(),
                               "dirty_files": len(sh(root, "git", "status", "--porcelain").splitlines()),
                               "sandbox_branches": [b.strip() for b in sh(root, "git", "branch", "--list", "creator/sbx-*").splitlines()],
                               "ahead_behind_origin": sh(root, "git", "rev-list", "--left-right", "--count", "HEAD...origin/main").strip()},
        "CR007_builders": {"running_python_processes": processes() if external else [],
                           "kernel_lock": (root / "state" / "creator" / "kernel.lock").exists()},
        "CR008_integrations": {"ci_workflows": ci(root)},
        "CR009_tests": {"test_files": len(tests), "test_functions": test_funcs, "collect": collect_line,
                        "collection_errors": errors[:20], "modules_without_any_test": len(untested), "examples": untested[:25]},
        "CR010_validation_infra": infra["experiments"] + infra["provenance"],
        "CR011_provenance": {"verify_integrity": integrity, "canon_verify_tail": canon, "infra": infra["provenance"]},
        "CR012_rollback": infra["rollback_checkpoints"],
        "CR013_deterministic_replay": infra["replay_determinism"],
        "CR014_reachability": {"unreached_modules": len(unreached), "examples": unreached[:40]},
        "CR015_data_flow": {"package_edges": sorted({(a.split(".")[0] + "." + a.split(".")[1] if a.count(".") > 1 else a,
                                                       b.split(".")[0]) for a, imps in graph.imports.items()
                                                      for b in imps if b.split(".")[0] in ("engine", "creator", "scripts")
                                                      and b.split(".")[0] != a.split(".")[0]})[:80]},
        "CR016_firewalls": infra["firewalls"],
        "CR017_research_boundary": boundary_check(graph),
        "CR018_learner_truth_boundary": infra["future_leak_controls"],
        "CR019_known_failures": known_failures(root, errors),
        "CR020_known_limitations": {"by_kind": dict(lims), "stubs": stubs[:40]},
        "CR021_stale_checklists": stale,
        "CR022_duplicates": duplicates(root, files)[:40],
        "CR023_disconnected": unreached[:60],
        "CR024_implemented_not_validated": {k: v.get("status") for k, v in stale.items()},
        "CR025_shallow": shallow[:60],
        "CR026_future_leak_risks": infra["future_leak_controls"],
        "CR027_self_learning_infra": infra["self_learning"],
        "CR028_research_infra": infra["research"],
        "CR029_experiment_infra": infra["experiments"],
        "CR030_health_infra": infra["health_monitoring"],
        "CR031_compute_controls": infra["compute_resources"],
        "seconds": round(time.time() - t0, 1),
    }
