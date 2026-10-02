"""CR207/CR208 (C77 sec 78): independent reproduction of the Creator's critical results - IMPLEMENTED, NOT VALIDATED.

Given a repo and a ledger file, in a fresh process and WITHOUT the Ledger class (no write path, no lock, no shared state):

  1. verify the hash chain line by line (stops at the first bad line; later lines are untrustworthy),
  2. re-hash every cited evidence file,
  3. recompute every ImprovementClaim verdict with the CURRENT creator.model.improvement_verdict,
  4. for every ADOPT Decision, check the cited merge commit exists in git and that the files it changed parse at that commit,
  5. optionally re-run the tests the merge commit touched at that commit in a temporary detached worktree (removed afterwards)
     and compare the pass rate to the recorded `affected_tests_pass` measurement.

Per claim: REPRODUCED (everything re-derived and agreed), NOT_REPRODUCED (something re-derived and DISAGREED: changed evidence,
different verdict, missing/unparseable commit, test rerun differs) or UNVERIFIABLE (could not be checked: broken chain, missing
evidence file, unreadable measurements, no git, rerun could not run). NOT_REPRODUCED outranks UNVERIFIABLE outranks REPRODUCED."""
from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from creator import model as M
from creator.ledger import GENESIS, _line_hash, record_id

REPRODUCED, NOT_REPRODUCED, UNVERIFIABLE = "REPRODUCED", "NOT_REPRODUCED", "UNVERIFIABLE"
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
RERUN_METRIC = "affected_tests_pass"
Rule = Callable[..., tuple[M.Verdict, dict[str, Any]]]


def _worst(statuses: Sequence[str]) -> str:
    for s in (NOT_REPRODUCED, UNVERIFIABLE):
        if s in statuses:
            return s
    return REPRODUCED


# ------------------------------------------------------------------------------------------------ 1. the chain

def read_chain(ledger_path: Path) -> tuple[list[dict[str, Any]], Optional[str], Optional[int]]:
    """Envelopes (record decoded under '_rec') up to the first bad line; returns (envelopes, error, bad_seq)."""
    if not ledger_path.is_file():
        return [], f"ledger not found: {ledger_path}", 0
    raw = ledger_path.read_bytes().decode("utf-8")
    lines = raw.split("\n")
    torn = lines[-1] != ""
    lines = lines[:-1]
    envs: list[dict[str, Any]] = []
    head = GENESIS
    for n, line in enumerate(lines):
        err: Optional[str] = None
        env: dict[str, Any] = {}
        try:
            env = json.loads(line)
            stored = env.pop("hash", None)
            if stored != _line_hash(env):
                err = "hash mismatch - the line was edited"
            elif env.get("seq") != n:
                err = f"seq {env.get('seq')} at position {n}"
            elif env.get("prev") != head:
                err = "chain broken (prev != previous hash)"
            else:
                rec = M.record_from_dict(env["rtype"], env["version"], env["data"])
                if record_id(rec, env["prev"], n) != env["id"]:
                    err = "id does not match its content"
                else:
                    env["_rec"], env["_hash"] = rec, stored
                    head = stored
        except (json.JSONDecodeError, M.ModelError, KeyError, TypeError, ValueError, AttributeError) as e:
            err = f"unreadable: {e}"
        if err:
            return envs, f"line {n + 1}: {err}", n
        envs.append(env)
    if torn:
        return envs, "torn final line", len(envs)
    return envs, None, None


# ------------------------------------------------------------------------------------------------ git helpers

def _git(repo: Path, *args: str, timeout: int = 60) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, timeout=timeout)


def commit_of(ev: M.EvidenceRef) -> Optional[str]:
    last = ev.path.replace("\\", "/").rsplit("/", 1)[-1]
    return last if _SHA40.match(last) else None


def check_commit(repo: Path, commit: str) -> tuple[str, list[str], list[str]]:
    """(status, reasons, changed_files). Exists in git, and every added/modified .py/.json/.jsonl file parses at that commit."""
    try:
        t = _git(repo, "cat-file", "-t", commit)
    except (OSError, subprocess.SubprocessError) as e:
        return UNVERIFIABLE, [f"git unavailable: {e}"], []
    if t.returncode != 0:
        if _git(repo, "rev-parse", "--git-dir").returncode != 0:
            return UNVERIFIABLE, [f"{repo} is not a git repository"], []
        return NOT_REPRODUCED, [f"merge commit {commit[:12]} does not exist in git"], []
    if t.stdout.strip() != b"commit":
        return NOT_REPRODUCED, [f"{commit[:12]} is a {t.stdout.decode().strip()}, not a commit"], []
    has_parent = _git(repo, "rev-parse", "--verify", "-q", commit + "^1").returncode == 0
    d = _git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", "--diff-filter=AMR", "-z",
             *((commit + "^1", commit) if has_parent else ("--root", commit)))
    files = [f for f in d.stdout.decode("utf-8", "replace").split("\0") if f]
    bad: list[str] = []
    for f in files:
        suffix = Path(f).suffix
        if suffix not in (".py", ".json", ".jsonl"):
            continue
        blob = _git(repo, "show", f"{commit}:{f}")
        if blob.returncode != 0:
            bad.append(f"{f}: cannot be read at the commit")
            continue
        text = blob.stdout.decode("utf-8", "replace")
        try:
            if suffix == ".py":
                ast.parse(text, filename=f)
            elif suffix == ".json":
                json.loads(text)
            else:
                for ln in text.splitlines():
                    if ln.strip():
                        json.loads(ln)
        except (SyntaxError, ValueError) as e:
            bad.append(f"{f}: does not parse at {commit[:12]} ({type(e).__name__}: {e})")
    if bad:
        return NOT_REPRODUCED, bad, files
    return REPRODUCED, [f"commit exists; {len(files)} changed file(s), all parse"], files


def rerun_tests(repo: Path, commit: str, files: Sequence[str], timeout: int = 600) -> dict[str, Any]:
    """Run the test files the commit touched in a temporary detached worktree; the worktree is always removed."""
    tests = [f for f in files if re.match(r"^tests/test_[^/]*\.py$", f)]
    if not tests:
        return {"status": "NO_TESTS", "tests": []}
    tmp = Path(tempfile.mkdtemp(prefix="repro_wt_"))
    wt = tmp / "wt"
    added = False
    try:
        r = _git(repo, "worktree", "add", "--detach", str(wt), commit, timeout=300)
        if r.returncode != 0:
            return {"status": "CANNOT_RUN", "tests": tests, "why": r.stderr.decode("utf-8", "replace")[-300:]}
        added = True
        try:
            from creator import testslots
            p = testslots.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--no-header", *tests], cwd=wt,
                              text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"status": "CANNOT_RUN", "tests": tests, "why": f"timeout after {timeout}s"}
        out = p.stdout + p.stderr
        last = out.strip().splitlines()[-1] if out.strip() else ""
        counts = {k: sum(int(n) for n in re.findall(rf"(\d+) {k}", last)) for k in ("passed", "failed", "error", "skipped")}
        total = counts["passed"] + counts["failed"] + counts["error"]
        return {"status": "RAN", "tests": tests, "returncode": p.returncode, **counts,
                "pass_rate": (counts["passed"] / total) if total else None, "tail": out[-400:]}
    finally:
        if added:
            _git(repo, "worktree", "remove", "--force", str(wt), timeout=120)
        shutil.rmtree(tmp, ignore_errors=True)
        _git(repo, "worktree", "prune")


# ------------------------------------------------------------------------------------------------ the whole check

def _meas(by_id: Mapping[str, Any], ids: Sequence[str]) -> list[M.Measurement]:
    out = []
    for i in ids:
        e = by_id.get(i)
        if e is None or not isinstance(e["_rec"], M.Measurement):
            raise KeyError(i)
        out.append(e["_rec"])
    return out


def _meas_safe(by_id: Mapping[str, Any], ids: Sequence[str]) -> list[M.Measurement]:
    try:
        return _meas(by_id, ids)
    except KeyError:
        return []


def reproduce(repo: str | Path, ledger_path: str | Path, rerun: bool = False, rule: Rule = M.improvement_verdict,
              rerun_timeout: int = 600) -> dict[str, Any]:
    repo, ledger_path = Path(repo), Path(ledger_path)
    envs, chain_err, bad_seq = read_chain(ledger_path)
    by_id = {e["id"]: e for e in envs}
    out: dict[str, Any] = {"repo": str(repo), "ledger": str(ledger_path),
                           "chain": {"ok": chain_err is None, "records": len(envs), "error": chain_err}}

    # 2. every evidence hash, once per record
    ev_status: dict[str, tuple[str, list[str]]] = {}
    ev_summary = {"checked": 0, "changed": 0, "missing": 0}
    for e in envs:
        res, why = REPRODUCED, []
        for ev in e["_rec"].evidence:
            ev_summary["checked"] += 1
            p = ev.problem(repo)
            if p:
                if p.startswith("evidence missing"):
                    ev_summary["missing"] += 1
                    res = _worst([res, UNVERIFIABLE])
                else:
                    ev_summary["changed"] += 1
                    res = NOT_REPRODUCED
                why.append(p)
        ev_status[e["id"]] = (res, why)
    out["evidence"] = ev_summary

    adopts: dict[str, list[dict[str, Any]]] = {}
    for e in envs:
        r = e["_rec"]
        if isinstance(r, M.Decision) and r.verdict is M.DecisionVerdict.ADOPT and r.claim_id:
            adopts.setdefault(r.claim_id, []).append(e)
    commit_cache: dict[str, tuple[str, list[str], list[str]]] = {}
    rerun_cache: dict[str, dict[str, Any]] = {}

    claims = []
    for e in envs:
        c = e["_rec"]
        if not isinstance(c, M.ImprovementClaim):
            continue
        parts: list[tuple[str, str]] = []                          # (status, reason)
        recomputed: Optional[str] = None
        # 3. the verdict
        try:
            base, cand = _meas(by_id, c.baseline_ids), _meas(by_id, c.candidate_ids)
            regs = list(zip(_meas(by_id, c.regression_baseline_ids), _meas(by_id, c.regression_candidate_ids)))
            hold = None
            if c.holdout_baseline_id and c.holdout_candidate_id:
                hb, hc = _meas(by_id, [c.holdout_baseline_id, c.holdout_candidate_id])
                hold = (hb, hc)
            v, detail = rule(base, cand, regs, hold, c.min_effect, c.z)
            recomputed = v.value
            if v is c.verdict:
                parts.append((REPRODUCED, f"verdict {c.verdict.value} recomputed identically"))
            else:
                parts.append((NOT_REPRODUCED,
                              f"recorded {c.verdict.value}, current rule computes {v.value} ({detail.get('why', '')})"))
        except KeyError as k:
            parts.append((UNVERIFIABLE, f"measurement {k} not readable in the verified part of the ledger"))
        # 2. evidence on the claim and everything it rests on
        ids = [e["id"], *c.baseline_ids, *c.candidate_ids, *c.regression_baseline_ids, *c.regression_candidate_ids,
               *[i for i in (c.holdout_baseline_id, c.holdout_candidate_id) if i]]
        for i in ids:
            if i in ev_status and ev_status[i][1]:
                parts.append((ev_status[i][0], f"{i}: " + "; ".join(ev_status[i][1])))
        # 4/5. adoption
        adoption: list[dict[str, Any]] = []
        for d in adopts.get(e["id"], []):
            dr = d["_rec"]
            if ev_status[d["id"]][1]:
                parts.append((ev_status[d["id"]][0], f"{d['id']}: " + "; ".join(ev_status[d["id"]][1])))
            commits = [x for x in (commit_of(ev) for ev in dr.evidence if ev.kind == "merge_commit") if x]
            if not commits:
                parts.append((UNVERIFIABLE, f"{d['id']}: ADOPT cites no merge_commit evidence"))
                adoption.append({"decision": d["id"], "status": UNVERIFIABLE, "reasons": ["no merge_commit evidence"]})
                continue
            for cm in commits:
                if cm not in commit_cache:
                    commit_cache[cm] = check_commit(repo, cm)
                st, why, files = commit_cache[cm]
                item: dict[str, Any] = {"decision": d["id"], "commit": cm, "status": st, "reasons": why}
                parts.append((st, f"{d['id']} commit {cm[:12]}: " + "; ".join(why)))
                if rerun and st == REPRODUCED:
                    if cm not in rerun_cache:
                        rerun_cache[cm] = rerun_tests(repo, cm, files, rerun_timeout)
                    rr = dict(rerun_cache[cm])
                    rec_vals = [m.value for m in _meas_safe(by_id, c.candidate_ids) if m.metric == RERUN_METRIC]
                    if rr["status"] != "RAN" or rr.get("pass_rate") is None:
                        rr["compare"] = UNVERIFIABLE
                        parts.append((UNVERIFIABLE, f"rerun at {cm[:12]}: {rr['status']} {rr.get('why', '')}".strip()))
                    elif not rec_vals:
                        rr["compare"] = UNVERIFIABLE
                        parts.append((UNVERIFIABLE, f"rerun pass rate {rr['pass_rate']:.3f} but no recorded {RERUN_METRIC}"))
                    else:
                        rec = sum(rec_vals) / len(rec_vals)
                        rr["compare"] = REPRODUCED if abs(rec - rr["pass_rate"]) <= 0.01 else NOT_REPRODUCED
                        parts.append((rr["compare"], f"rerun pass rate {rr['pass_rate']:.3f} vs recorded {rec:.3f}"))
                    item["rerun"] = rr
                adoption.append(item)
        status = _worst([s for s, _ in parts])
        claims.append({"claim": e["id"], "seq": e["seq"], "subject": c.subject_id, "recorded_verdict": c.verdict.value,
                       "recomputed_verdict": recomputed, "status": status, "reasons": [w for _, w in parts],
                       "adoptions": adoption})
    if chain_err:
        out["after_break"] = f"records at and after the break are not examined and not counted: {chain_err}"
    out["claims"] = claims
    counts = {s: sum(1 for c in claims if c["status"] == s) for s in (REPRODUCED, NOT_REPRODUCED, UNVERIFIABLE)}
    out["counts"] = {"claims": len(claims), **counts, "adopts": sum(len(v) for v in adopts.values()),
                     "commits_checked": len(commit_cache)}
    out["overall"] = _worst([c["status"] for c in claims] + ([] if chain_err is None else [UNVERIFIABLE]))
    return out
