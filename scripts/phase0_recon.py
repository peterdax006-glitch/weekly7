"""C77 sec 6 - Phase 0 reconnaissance of the WHOLE repository, computed (checklist CR001-CR031).

Writes state/build/CREATOR_PHASE0_LEDGER.json (+ .md). Every section is produced by code reading the tree, git, the checklists and
the provenance locks - nothing is typed from memory. Reading-only; runs no tests except `pytest --collect-only` when --collect."""
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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from creator import selfmodel as SM  # noqa: E402
from creator import testrun as T  # noqa: E402

SCOPE = ("engine", "scripts", "creator", "tests", "canon")
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


def sh(*args: str) -> str:
    try:
        return subprocess.run(list(args), cwd=ROOT, capture_output=True, text=True, timeout=120, encoding="utf-8",
                              errors="replace").stdout
    except (OSError, subprocess.SubprocessError) as e:
        return f"<error {e}>"


def body_hash(node: ast.AST) -> str:
    """Normalized function body (names kept, docstring dropped) for duplicate detection."""
    body = list(getattr(node, "body", []))
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
        body = body[1:]
    return hashlib.sha256("\n".join(ast.dump(b, include_attributes=False) for b in body).encode()).hexdigest()[:16]


def duplicates(files: list[str]) -> list[dict[str, Any]]:
    seen: dict[str, list[str]] = defaultdict(list)
    for rel in files:
        if rel.startswith("tests/"):
            continue
        try:
            tree = ast.parse((ROOT / rel).read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and len(n.body) >= 4:
                seen[body_hash(n)].append(f"{rel}:{n.name}")
    return [{"body": h, "sites": v} for h, v in sorted(seen.items(), key=lambda kv: -len(kv[1])) if len(set(v)) > 1]


def checklist_states() -> dict[str, Any]:
    out = {}
    for p in sorted((ROOT / "state" / "build").glob("*CHECKLIST*.json")):
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
    """Trading-path modules must not reach engine.research (memory: 'lazy imports reach the trader')."""
    mod_of = {p: m for m, p in graph.modules.items()}
    bad = []
    for path in TRADING_PATH:
        start = mod_of.get(path)
        if not start:
            continue
        seen, frontier = set(), [start]
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


def ci() -> dict[str, Any]:
    out = {}
    for wf in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        text = wf.read_text(encoding="utf-8")
        out[wf.name] = {"runs": [ln.split("run:", 1)[1].strip()[:160] for ln in text.splitlines() if "run:" in ln],
                        "secrets": sorted(set(re.findall(r"secrets\.([A-Z0-9_]+)", text)))}
    return out


def processes() -> list[str]:
    ps = sh("powershell", "-NoProfile", "-Command",
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object -ExpandProperty CommandLine")
    return [ln.strip()[:200] for ln in ps.splitlines() if "weekly7" in ln.lower()]


def main(argv: list[str]) -> int:
    t0 = time.time()
    files = [f for f in SM.python_files(ROOT, SCOPE) if not f.startswith("creator/devbench/")]   # benchmark material, not code
    model = SM.build(ROOT, scope=SCOPE, capabilities=SM.load_capabilities(), include_versions=True)
    model = dataclasses.replace(model, components={k: v for k, v in model.components.items() if k in set(files)},
                                limitations=tuple(x for x in model.limitations if not x.subject.startswith("creator/devbench/")))
    graph = T.ImportGraph.build(ROOT)
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
        return "__name__ ==" in (ROOT / rel).read_text(encoding="utf-8", errors="replace")
    unreached = sorted(x.subject for x in model.limitations if x.kind == "UNREACHED" and not entry_point(x.subject))
    stubs = [f"{x.subject}:{x.detail}" for x in model.limitations if x.kind == "STUB"]
    shallow = sorted(p for p, c in model.components.items() if not T.is_test_file(p) and c.interfaces and c.meaningful < 15
                     and not p.endswith("__init__.py"))
    infra = {k: sorted(p for p in files if re.search(rx, p) and not T.is_test_file(p)) for k, rx in INFRA.items()}
    try:
        from engine import provenance as PV
        PV.verify_integrity()
        integrity = "OK"
    except Exception as e:                                              # noqa: BLE001 - reported
        integrity = f"FAILED: {type(e).__name__}: {e}"
    canon = sh(sys.executable, "canon/build_canon.py", "--verify").strip()[-300:]
    collect = ""
    if "--collect" in argv:
        out = sh(sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "tests")
        collect = out.strip().splitlines()[-1] if out.strip() else ""
        errors = [ln for ln in out.splitlines() if ln.startswith("ERROR")]
    else:
        errors = []
    ledger: dict[str, Any] = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "scope": SCOPE, "selfmodel_digest": model.digest(),
        "CR001-004_documents": {
            "blueprints": sorted(p.name for p in ROOT.glob("*BLUEPRINT*.md")),
            "contracts": sorted(p.name for p in ROOT.glob("*.md") if p.name not in ("README.md",)),
            "checklists": sorted(p.name for p in (ROOT / "state" / "build").glob("*CHECKLIST*.json")),
            "masterstock": str(Path.home() / "Masterstock" / "MASTERSTOCK.md"),
            "masterstock_bytes": (Path.home() / "Masterstock" / "MASTERSTOCK.md").stat().st_size
            if (Path.home() / "Masterstock" / "MASTERSTOCK.md").is_file() else None},
        "CR005_structure": {"files": len(files), "by_package_files": dict(pkg_files.most_common()),
                            "by_package_meaningful_lines": dict(pkg_lines.most_common())},
        "CR006_branch_state": {"head": sh("git", "rev-parse", "HEAD").strip(), "branch": sh("git", "branch", "--show-current").strip(),
                               "dirty_files": len(sh("git", "status", "--porcelain").splitlines()),
                               "sandbox_branches": [b.strip() for b in sh("git", "branch", "--list", "creator/sbx-*").splitlines()],
                               "ahead_behind_origin": sh("git", "rev-list", "--left-right", "--count", "HEAD...origin/main").strip()},
        "CR007_builders": {"running_python_processes": processes(),
                           "kernel_lock": (ROOT / "state" / "creator" / "kernel.lock").exists()},
        "CR008_integrations": {"ci_workflows": ci()},
        "CR009_tests": {"test_files": len(tests), "test_functions": test_funcs, "collect": collect, "collection_errors": errors[:20],
                        "modules_without_any_test": len(untested), "examples": untested[:25]},
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
        "CR019_known_failures": {"note": "creator tests: see state/creator/test_evidence.json; engine/full-suite failures need a "
                                         "full run (not executed by recon)", "collection_errors": errors[:20]},
        "CR020_known_limitations": {"by_kind": dict(lims), "stubs": stubs[:40]},
        "CR021_stale_checklists": checklist_states(),
        "CR022_duplicates": duplicates(files)[:40],
        "CR023_disconnected": unreached[:60],
        "CR024_implemented_not_validated": {k: v.get("status") for k, v in checklist_states().items()},
        "CR025_shallow": shallow[:60],
        "CR026_future_leak_risks": infra["future_leak_controls"],
        "CR027_self_learning_infra": infra["self_learning"],
        "CR028_research_infra": infra["research"],
        "CR029_experiment_infra": infra["experiments"],
        "CR030_health_infra": infra["health_monitoring"],
        "CR031_compute_controls": infra["compute_resources"],
        "seconds": round(time.time() - t0, 1),
    }
    out = ROOT / "state" / "build" / "CREATOR_PHASE0_LEDGER.json"
    out.write_text(json.dumps(ledger, indent=1, default=str), encoding="utf-8")
    md = ["# C77 Phase 0 - computed reconnaissance ledger", "",
          f"Generated {ledger['generated']} by scripts/phase0_recon.py (reading only). Self-model digest {model.digest()}.", ""]
    for k, v in ledger.items():
        if k.startswith("CR"):
            body = json.dumps(v, indent=1, default=str)
            md += [f"## {k}", "", "```json", body[:4000] + ("\n... (truncated; full in the .json)" if len(body) > 4000 else ""),
                   "```", ""]
    (ROOT / "state" / "build" / "CREATOR_PHASE0_LEDGER.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, (dict, list)) else (len(v) if isinstance(v, list) else list(v)[:6]))
                      for k, v in ledger.items()}, default=str)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
