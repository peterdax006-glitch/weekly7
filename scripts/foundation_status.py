"""Foundation tracker (canon C46, C50, C51): one row per WORK UNIT (a builder task, or a group of pre-existing
modules), its line count against the SUM of the Bible ranges of the phases it covers (no file counted twice), and
whether its tests pass. "Foundation done" = every unit in range with passing tests and every estimated Bible phase
covered. Writes state/build/FOUNDATION.md (included in Masterstock).

Usage: python scripts/foundation_status.py [--run]      (--run executes each unit's tests; otherwise counts only)"""
import re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
Q = ROOT / "state" / "build" / "queue"
BIBLE = (ROOT / "BIBLE.md").read_text(encoding="utf-8")
NL = chr(10)

# phases implemented by pre-existing modules (not from a builder task file); a file a builder owns is counted there
PREEXISTING = {
    "0": ["engine/provenance.py", "tests/test_provenance.py", "engine/improve.py", "scripts/foundation_status.py"],
    "1": ["engine/fill_audit.py", "tests/test_fill_audit.py"],
    "3": ["engine/patterns.py", "tests/test_planted_patterns.py"],
    "25": ["scripts/planted_calibration.py", "engine/planted.py", "tests/test_planted_library.py"],
    "A7": ["engine/miner_tuning.py", "tests/test_miner_tuning.py", "scripts/miner_tune_real.py"],
    "8": ["engine/analogs.py", "scripts/analog_test.py"],
    "A1": ["engine/candles.py", "tests/test_candles.py"],
}


def bible_ranges():
    """{phase: (lo, hi, title)} from '# PHASE n — TITLE' followed by 'Estimated code: **lo–hi lines**'."""
    out = {}
    pat = r"# PHASE (\d+) — ([^\n]+)\n+Estimated code: \*\*([\d,]+)[–-]([\d,]+) lines\*\*"
    for m in re.finditer(pat, BIBLE):
        out[m.group(1)] = (int(m.group(3).replace(",", "")), int(m.group(4).replace(",", "")), m.group(2).strip())
    return out


def task_files():
    """{task id: (phases, [file patterns])} parsed from the builder task files."""
    out = {}
    for f in sorted(Q.glob("B*.md")):
        s = f.read_text(encoding="utf-8")
        head = s.split("Bible:")[1].split("(")[0] if "Bible:" in s else ""
        phases = re.findall(r"\b(\d{1,2})\b", head)
        own = re.search(r"You own: ([^\n]+)", s)
        files = re.findall(r"((?:engine|scripts|tests|docs)/[\w./*-]+)", own.group(1)) if own else []
        out[f.stem] = (phases, files)
    return out


def expand(pattern):
    if "*" in pattern:
        return sorted(p.relative_to(ROOT).as_posix() for p in ROOT.glob(pattern) if p.is_file())
    return [pattern.rstrip("/")]


def lines(rel):
    p = ROOT / rel
    if p.is_dir():
        return sum(lines(q.relative_to(ROOT).as_posix()) for q in p.rglob("*.py"))
    return len(p.read_text(encoding="utf-8", errors="replace").splitlines()) if p.exists() else 0


def run_tests(files):
    tests = [f for f in files if f.startswith("tests/") and (ROOT / f).exists()]
    if not tests:
        return "no tests"
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests], cwd=ROOT,
                       capture_output=True, text=True, timeout=1800)
    last = [l for l in r.stdout.splitlines() if l.strip()][-1:] or ["?"]
    return ("PASS " if r.returncode == 0 else "FAIL ") + last[0]


# files builders created or took over beyond their task file's "You own:" list (from their reports)
EXTRA = {
    "B01_pit_firewall": ["scripts/pit_audit_real.py"],
    "B02_parity": ["engine/parity_suite.py", "scripts/run_parity.py", "tests/test_parity_suite.py"],
    "B03_lifecycle_bank": ["scripts/run_pattern_bank.py"],
    "B04_heavy_algo": ["scripts/heavy_algo_real.py"],
    "B05_analogs_ext": ["scripts/run_analogs_ext.py"],
    "B06_lessons": ["scripts/lessons_real.py"],
    "B07_trust_direction": ["engine/direction_calib.py", "engine/direction_ablate.py", "engine/trust_store.py",
                            "scripts/direction_study.py", "tests/test_direction_calib.py", "tests/test_direction_ablate.py",
                            "tests/test_direction_study.py", "tests/test_trust_store.py"],
    "B08_exits_stops": ["engine/gaprisk.py", "tests/test_gaprisk.py", "scripts/run_exits_stops.py", "scripts/run_gaprisk.py",
                        "scripts/run_pattern_fail.py"],
    "B09_timeline_basis": ["scripts/timeline_basis_report.py", "scripts/basis_offline.py", "scripts/dial_offline.py",
                           "scripts/livesim_loop2.py", "tests/test_livesim_loop2.py"],
    "B10_blind_gates": ["scripts/blind_gates_real.py", "engine/livesim.py", "scripts/check_retester.py",
                        "scripts/livesim_cycle.py", "tests/test_livesim_gates.py"],
    "B11_antioverfit": ["scripts/antioverfit_real.py"],
    "B12_control_reports": ["engine/baseline.py", "engine/experiment_memory.py", "scripts/audit_registry.py",
                            "tests/test_baseline.py", "tests/test_experiment_memory.py", "tests/test_audit_registry.py"],
    "B13_data_live_safety": ["scripts/data_live_audit.py", "scripts/fetch_delisted.py"],
    "B14_miner_hardening": ["engine/patterns.py", "tests/test_patterns_integration.py"],
    "B15_memory_adapter": ["scripts/run_b15_memory_adapter.py", "tests/test_memory.py", "tests/test_session.py"],
    "B17_quality_pyramid": ["tests/integration/"],
}


def units():
    out = [(tid, phases, sorted(set(sum((expand(f) for f in files + EXTRA.get(tid, [])), []))))
           for tid, (phases, files) in task_files().items()]
    claimed = {f for _, _, fs in out for f in fs}
    for ph, files in PREEXISTING.items():
        fs = [f for f in files if f not in claimed]
        if fs:
            out.append(("pre:" + ph, [ph] if ph.isdigit() else [], fs))
    return out


def main(run=False):
    R = bible_ranges()
    rows, covered = [], set()
    U = units()
    by_builder = {ph for uid, phs, _ in U if not uid.startswith("pre:") for ph in phs}
    for uid, phases, files in U:
        covered.update(phases)
        # a pre-existing group only carries a range for a phase no builder covers (else the range would count twice)
        ranged = [ph for ph in phases if not (uid.startswith("pre:") and ph in by_builder)]
        lo = sum(R.get(ph, (0, 0, ""))[0] for ph in ranged)
        hi = sum(R.get(ph, (0, 0, ""))[1] for ph in ranged)
        n = sum(lines(f) for f in files)
        status = "NO CODE" if n == 0 else ("BELOW RANGE" if lo and n < lo else "IN RANGE")
        rows.append((uid, ",".join(phases), lo, hi, n, status, run_tests(files) if run else "-", files))
    missing = [f"{ph} {R[ph][2]}" for ph in sorted(R, key=int) if ph not in covered]
    out = ["# Foundation status (scripts/foundation_status.py)", "",
           "Rule (C46/C50/C51): every unit in or above its Bible range with passing tests, before any rabbit hole.",
           "One row per work unit; range = sum of the Bible ranges of the phases it covers; no file counted twice.", "",
           "| unit | phases | range | lines | status | tests |", "|---|---|---|---|---|---|"]
    for uid, phs, lo, hi, n, st, tres, _ in rows:
        out.append(f"| {uid} | {phs} | {lo:,}-{hi:,} | {n:,} | {st} | {tres} |")
    code = {d: sum(lines(q.relative_to(ROOT).as_posix()) for q in (ROOT / d).rglob("*.py"))
            for d in ("engine", "scripts", "tests")}
    out += ["", "Repository Python lines: " + ", ".join(f"{k} {v:,}" for k, v in code.items())
            + f" = {sum(code.values()):,} (floor 25,000).", "",
            "Units below range or without code: " + (", ".join(r[0] for r in rows if r[5] != "IN RANGE") or "none"),
            "Bible phases with an estimate but no unit: " + (", ".join(missing) or "none"), "", "## Files per unit", ""]
    for uid, *_, files in rows:
        out.append(f"- **{uid}**: " + (", ".join(f"`{f}` ({lines(f)})" for f in files) or "_none_"))
    (ROOT / "state" / "build" / "FOUNDATION.md").write_text(NL.join(out) + NL, encoding="utf-8", newline=NL)
    print(NL.join(out[:len(rows) + 12]))


if __name__ == "__main__":
    main(run="--run" in sys.argv)
