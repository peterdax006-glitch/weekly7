"""Creator K26b - the claims register (C77 CR208: the evidence supports ALL major claims) - IMPLEMENTED, NOT VALIDATED.

`build_register()` lists every major claim the system makes about itself, each with the evidence it cites and a RECOMPUTATION done
now, from the sources, not from the claim:

    improvement_claim  every ImprovementClaim in the ledger: its measurements and their evidence files, the verdict recomputed by
                       creator.reproduce (hash chain, evidence hashes, the CURRENT rule)
    adoption           every ADOPT decision: the claim it rests on is reproduced and the merge commit exists in git
    headline           the numbers in state/creator/STATUS.json and the checklist (capabilities, audit, adversary, ledger size,
                       checklist counts), each recomputed from a different source than the one that states it
    extra              claims added by hand in a JSON file ({id, statement, evidence:[{path, sha256}], recompute:{...}}); a
                       claim without evidence, with changed evidence, or without a recomputation that agrees is UNSUPPORTED

A claim is SUPPORTED only when all of its evidence checks out AND its recomputation agrees. Everything else is UNSUPPORTED, with the
reason; nothing is ever downgraded silently and nothing is marked supported by default. The register is written by
scripts/claims_register.py to state/build/CLAIMS.json."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from creator import model as M
from creator import reproduce as RP

SUPPORTED, UNSUPPORTED = "SUPPORTED", "UNSUPPORTED"


def _claim(cid: str, kind: str, statement: str, evidence: list[dict[str, Any]], recomputation: dict[str, Any], ok: bool,
           why: str) -> dict[str, Any]:
    return {"id": cid, "kind": kind, "statement": statement, "evidence": evidence, "recomputation": recomputation,
            "status": SUPPORTED if ok else UNSUPPORTED, "why": why}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_ref(root: Path, rel: str, sha: Optional[str] = None) -> dict[str, Any]:
    p = root / rel
    if not p.is_file():
        return {"path": rel, "sha256": sha, "ok": False, "problem": "file does not exist"}
    now = _sha(p)
    return {"path": rel, "sha256": now, "ok": sha is None or sha == now, **({} if sha is None or sha == now else {"problem": "sha256 changed"})}


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _pointer(doc: Any, pointer: str) -> Any:
    for part in [p for p in pointer.split("/") if p]:
        if isinstance(doc, Mapping) and part in doc:
            doc = doc[part]
        elif isinstance(doc, list) and part.isdigit() and int(part) < len(doc):
            doc = doc[int(part)]
        else:
            raise KeyError(pointer)
    return doc


# ------------------------------------------------------------------------------------------------ ledger claims

def ledger_claims(repo: Path, ledger_path: Path, rerun: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rep = RP.reproduce(repo, ledger_path, rerun=rerun)
    envs, _err, _bad = RP.read_chain(ledger_path)
    by = {e["id"]: e for e in envs}
    out: list[dict[str, Any]] = []
    status = {c["claim"]: c for c in rep["claims"]}
    for e in envs:
        rec = e["_rec"]
        if not isinstance(rec, M.ImprovementClaim):
            continue
        ids = [*rec.baseline_ids, *rec.candidate_ids, *rec.regression_baseline_ids, *rec.regression_candidate_ids,
               *[i for i in (rec.holdout_baseline_id, rec.holdout_candidate_id) if i]]
        refs: dict[str, dict[str, Any]] = {}
        for mid in ids:
            m = by.get(mid)
            for ev in (m["_rec"].evidence if m is not None else ()):
                refs[ev.path] = {**_file_ref(repo, ev.path, ev.sha256), "record": mid}
        rr = status.get(e["id"], {})
        evidence = [{"record": e["id"], "type": "ImprovementClaim"}] + [{"record": i, "type": "Measurement"} for i in ids] + \
            list(refs.values())
        ok = rr.get("status") == RP.REPRODUCED and bool(refs) and all(r["ok"] for r in refs.values())
        why = ("recomputed verdict matches and all evidence is unchanged" if ok else
               "no evidence files are cited" if not refs else
               "; ".join(rr.get("reasons", [])) or f"reproduction status {rr.get('status')}")
        out.append(_claim(f"claim:{e['id']}", "improvement_claim",
                          f"{rec.subject_id}: verdict {rec.verdict.value} (n={len(ids)} measurements)", evidence,
                          {"recorded_verdict": rec.verdict.value, "recomputed_verdict": rr.get("recomputed_verdict"),
                           "status": rr.get("status")}, ok, why))
    for e in envs:
        rec = e["_rec"]
        if not (isinstance(rec, M.Decision) and rec.verdict is M.DecisionVerdict.ADOPT):
            continue
        cl = status.get(str(rec.claim_id))
        merges = [ev for ev in rec.evidence if ev.kind == "merge_commit"]
        item = next((a for c in rep["claims"] for a in c.get("adoptions", []) if a.get("decision") == e["id"]), None)
        commit_ok = item is not None and item.get("status") == RP.REPRODUCED
        ok = cl is not None and cl["status"] == RP.REPRODUCED and bool(merges) and commit_ok
        why = ("claim reproduced and the merge commit exists in git" if ok else
               "cites no merge commit" if not merges else
               "cites no reproducible ImprovementClaim" if cl is None or cl["status"] != RP.REPRODUCED else
               "merge commit could not be confirmed in git")
        out.append(_claim(f"adoption:{e['id']}", "adoption", f"{rec.subject_id} was adopted: {rec.reason[:120]}",
                          [{"record": e["id"], "type": "Decision"}, {"record": str(rec.claim_id), "type": "ImprovementClaim"},
                           *[{"path": m.path, "kind": "merge_commit"} for m in merges]],
                          {"claim_status": cl and cl["status"], "merge_commit_checked": commit_ok}, ok, why))
    return out, rep


# ------------------------------------------------------------------------------------------------ headline numbers

def headline_claims(repo: Path, rep: Mapping[str, Any], live_adversary: bool = False) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    status_p, audit_p = repo / "state" / "creator" / "STATUS.json", repo / "state" / "creator" / "AUDIT.json"
    caps_p = repo / "creator" / "capabilities.json"
    check_p = repo / "state" / "build" / "CREATOR_MASTER_CHECKLIST.json"
    remaining_p = repo / "state" / "build" / "CHECKLIST_REMAINING.md"
    st, au, caps, chk = _json(status_p), _json(audit_p), _json(caps_p), _json(check_p)

    def ev(*paths: Path) -> list[dict[str, Any]]:
        return [_file_ref(repo, str(p.relative_to(repo)).replace("\\", "/")) for p in paths]

    if isinstance(st, dict) and isinstance(caps, dict):
        declared = sorted(c["id"] for c in caps.get("capabilities", []))
        listed = sorted(c["id"] for c in st.get("capabilities", []))
        out.append(_claim("headline:capabilities", "headline", f"STATUS lists {len(listed)} capabilities, all of those declared",
                          ev(status_p, caps_p), {"declared": len(declared), "listed": len(listed)}, declared == listed,
                          "STATUS lists exactly the declared capabilities" if declared == listed else
                          f"differ: missing {sorted(set(declared) - set(listed))}, extra {sorted(set(listed) - set(declared))}"))
    else:
        out.append(_claim("headline:capabilities", "headline", "STATUS capabilities", ev(status_p, caps_p), {}, False,
                          "STATUS.json or creator/capabilities.json is missing or unreadable"))
    if isinstance(st, dict) and isinstance(au, dict):
        adv = au.get("adversary", [])
        claimed = st.get("adversary", {})
        got = {"attacks": len(adv), "caught": sum(1 for a in adv if a.get("caught"))}
        ok = (claimed.get("attacks"), claimed.get("caught")) == (got["attacks"], got["caught"])
        out.append(_claim("headline:adversary", "headline", f"adversary caught {claimed.get('caught')} of {claimed.get('attacks')} attacks",
                          ev(status_p, audit_p), {"recomputed_from_AUDIT.json": got}, ok,
                          "STATUS agrees with the attack list in AUDIT.json" if ok else f"STATUS says {claimed}, AUDIT.json lists {got}"))
        counts = Counter(f.get("severity") for f in au.get("audit", {}).get("findings", []))
        claimed_a = st.get("audit", {})
        ok2 = all(int(claimed_a.get(s, 0)) == counts.get(s, 0) for s in ("CRITICAL", "HIGH", "MEDIUM", "LOW"))
        out.append(_claim("headline:audit", "headline", f"audit counts {claimed_a}", ev(status_p, audit_p),
                          {"recomputed_from_AUDIT.json": dict(counts)}, ok2,
                          "STATUS counts equal the findings listed in AUDIT.json" if ok2 else f"STATUS {claimed_a} vs findings {dict(counts)}"))
        if live_adversary:
            from creator.audit import checks as AUD
            live = AUD.adversary()
            now = {"attacks": len(live), "caught": sum(a.caught for a in live)}
            out.append(_claim("headline:adversary_live", "headline",
                              f"the adversary catches {claimed.get('caught')} of {claimed.get('attacks')} attacks", ev(status_p),
                              {"recomputed_now": now}, now["attacks"] == now["caught"] == claimed.get("caught") == claimed.get("attacks"),
                              "re-ran every attack now: all caught and the STATUS count is current" if
                              now["attacks"] == now["caught"] == claimed.get("attacks") == claimed.get("caught") else
                              f"uncaught now: {[a.name for a in live if not a.caught]}" if now["attacks"] != now["caught"] else
                              f"STATUS is stale: it says {claimed.get('caught')}/{claimed.get('attacks')}, re-running now gives "
                              f"{now['caught']}/{now['attacks']} (regenerate with scripts/creator_status.py)"))
    else:
        out.append(_claim("headline:adversary", "headline", "adversary/audit counts", ev(status_p, audit_p), {}, False,
                          "STATUS.json or AUDIT.json is missing or unreadable"))
    if isinstance(st, dict):
        n = st.get("ledger_records")
        chain = rep.get("chain", {})
        ok3 = isinstance(n, int) and chain.get("ok") is True and chain.get("records", 0) >= n
        out.append(_claim("headline:ledger_records", "headline", f"the ledger holds {n} verified records (at STATUS time)",
                          ev(status_p), {"chain_ok": chain.get("ok"), "records_now": chain.get("records")}, ok3,
                          "the hash chain verifies and contains at least that many records" if ok3 else
                          f"chain ok={chain.get('ok')} records={chain.get('records')} vs claimed {n}"))
    if isinstance(chk, dict) and remaining_p.is_file():
        m = re.search(r"Counts: (\{[^}]*\}) \((\d+) items\)", remaining_p.read_text(encoding="utf-8"))
        items = chk.get("items", [])
        now = dict(Counter(i.get("status") for i in items))
        try:
            claimed_c = json.loads(m.group(1).replace("'", '"')) if m else None
        except ValueError:
            claimed_c = None
        ok4 = claimed_c == now and m is not None and int(m.group(2)) == len(items)
        out.append(_claim("headline:checklist_counts", "headline", f"checklist counts {claimed_c} ({m.group(2) if m else '?'} items)",
                          ev(check_p, remaining_p), {"recomputed_from_checklist": now, "items": len(items)}, ok4,
                          "CHECKLIST_REMAINING.md equals the counts of CREATOR_MASTER_CHECKLIST.json" if ok4 else
                          f"stated {claimed_c}, recomputed {now} - regenerate with scripts/checklist_remaining.py"))
        validated = [i["id"] for i in items if i.get("status") == "VALIDATED"]
        out.append(_claim("headline:nothing_validated_by_hand", "headline", "no checklist item is VALIDATED without the validator report",
                          ev(check_p), {"validated_items": validated[:20], "count": len(validated)},
                          not validated or (repo / "state" / "validation" / "VALIDATION_REPORT.md").is_file(),
                          "no item is VALIDATED" if not validated else "VALIDATED items exist and the validation report is present"
                          if (repo / "state" / "validation" / "VALIDATION_REPORT.md").is_file() else
                          f"{len(validated)} VALIDATED items but no state/validation/VALIDATION_REPORT.md"))
    else:
        out.append(_claim("headline:checklist_counts", "headline", "checklist counts", ev(check_p, remaining_p), {}, False,
                          "the checklist or CHECKLIST_REMAINING.md is missing"))
    return out


# ------------------------------------------------------------------------------------------------ hand-added claims

def extra_claims(repo: Path, claims: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """A claim is {id, statement, evidence:[{path, sha256?}], recompute:{kind:'json', path, pointer, expected}} - 'json' re-reads the
    file now and compares the value at `pointer`. No evidence, changed evidence, no recomputation or a different value: UNSUPPORTED."""
    out = []
    for c in claims:
        cid, stmt = str(c.get("id", "?")), str(c.get("statement", ""))
        refs = [_file_ref(repo, str(e.get("path", "")), e.get("sha256")) for e in c.get("evidence", []) or []]
        rc = c.get("recompute") or {}
        got: dict[str, Any] = {"kind": rc.get("kind")}
        if not refs:
            ok, why = False, "no evidence is cited"
        elif not all(r["ok"] for r in refs):
            ok, why = False, "; ".join(f"{r['path']}: {r.get('problem')}" for r in refs if not r["ok"])
        elif rc.get("kind") != "json" or not rc.get("path") or "expected" not in rc:
            ok, why = False, "no recomputation is given (kind 'json' with path, pointer and expected)"
        else:
            doc = _json(repo / str(rc["path"]))
            try:
                value = _pointer(doc, str(rc.get("pointer", "")))
                got["recomputed"] = value
                ok = value == rc["expected"]
                why = "recomputed value equals the claimed value" if ok else f"recomputed {value!r}, claimed {rc['expected']!r}"
            except KeyError:
                ok, why = False, f"{rc.get('pointer')!r} not found in {rc['path']}"
        out.append(_claim(f"extra:{cid}", "extra", stmt, refs, got, ok, why))
    return out


def build_register(repo: Path, ledger_path: Path, extra: Sequence[Mapping[str, Any]] = (), rerun: bool = False,
                   live_adversary: bool = False) -> dict[str, Any]:
    repo, ledger_path = Path(repo), Path(ledger_path)
    claims, rep = ledger_claims(repo, ledger_path, rerun)
    claims += headline_claims(repo, rep, live_adversary) + extra_claims(repo, extra)
    counts = Counter(c["status"] for c in claims)
    return {"ledger": str(ledger_path.relative_to(repo)).replace("\\", "/") if ledger_path.is_relative_to(repo) else str(ledger_path),
            "chain": rep.get("chain"), "counts": {"claims": len(claims), SUPPORTED: counts[SUPPORTED], UNSUPPORTED: counts[UNSUPPORTED]},
            "unsupported": [c["id"] for c in claims if c["status"] == UNSUPPORTED], "claims": claims}
