"""The owner's daily change digest (owner, 4 Oct 2026: the checker / safety loop). One readable page per day:
state/creator/digest/YYYY-MM-DD.md - every adoption, rollback and refusal in plain English (what changed, why, evidence, tests run,
gate state), then what Nupen is waiting on the owner for. Source: state/creator/safety/events.jsonl (written by the kernel through
creator.safety), the safety policy and the coding trust gate. A reader: it changes nothing but the digest file.

    python scripts/nupen_digest.py [--date YYYY-MM-DD] [--state state/creator] [--stdout]
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WHAT = {"ADOPTED": "ADOPTED (merged into main)", "ROLLED_BACK": "ROLLED BACK (merged, then reverted automatically)",
        "REJECTED": "REJECTED", "CANCELLED": "DEFERRED / PULLED BACK", "DEFERRED": "DEFERRED", "ERROR": "ERROR", "BUDGET": "BUDGET STOP"}
SHOWN = ("ADOPTED", "ROLLED_BACK")                    # always shown; other outcomes only when the safety loop refused them


def _day(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def _clock(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%H:%M")


def _who(e: Mapping[str, Any]) -> str:
    by = e.get("by") or "unknown worker"
    return f"{by} ({'the teacher, supervised' if e.get('supervised') else 'Nupen, unsupervised'})"


def _tests(e: Mapping[str, Any]) -> str:
    s = e.get("suite") or {}
    parts = [f"sandbox regression report {e.get('regression') or 'n/a'}"]
    if s.get("skipped"):
        parts.append(f"protected suite skipped ({s['skipped']})")
    elif s:
        res = "green" if not s.get("failed") else f"{len(s['failed'])} failing: {', '.join(s['failed'][:4])}"
        parts.append(f"full protected suite after the merge: {res}" + (f" ({s.get('cases')} cases, {s.get('seconds')} s, "
                     f"{s.get('workers')} workers{', cached by tree hash' if s.get('cached') else ''})" if s.get("cases") is not None else ""))
        if s.get("pre_existing"):
            parts.append(f"already failing before the merge (not caused by it): {', '.join(s['pre_existing'][:4])}")
    return "; ".join(parts)


def event_text(e: Mapping[str, Any]) -> str:
    """One event in plain English."""
    out = e.get("outcome", "?")
    head = f"### {_clock(float(e.get('at', 0)))} {WHAT.get(out, out)}: {e.get('package', '?')}"
    lines = [head, "",
             f"- **What:** {e.get('objective') or e.get('requirement') or '(no objective recorded)'}",
             f"- **Who:** {_who(e)}; change class **{e.get('cls', '?')}**; files: "
             + (", ".join(f"`{p}`" for p in (e.get("paths") or [])[:8]) or "none recorded")
             + (f" (+{len(e['paths']) - 8} more)" if len(e.get("paths") or []) > 8 else "")]
    if out == "ADOPTED":
        lines.append(f"- **Why adopted:** the measured claim was {e.get('verdict') or '?'} with a clean regression report; "
                     f"merge `{str(e.get('merge', ''))[:12]}` (undo: `git revert -m 1 {str(e.get('merge', ''))[:12]}`)")
    elif out == "ROLLED_BACK":
        lines.append(f"- **Why reverted:** {e.get('reason')}; merge `{str(e.get('merge', ''))[:12]}` was undone by a revert commit; "
                     "unsupervised adoptions now cool down")
    else:
        lines.append(f"- **Why not adopted:** {e.get('reason')}")
    if e.get("refused"):
        lines.append("- **Safety loop:** " + ("trust gate closed for this class - it needs the owner/teacher; the diff is saved in "
                                               "state/creator/pending/" if e["refused"] == "trust" else
                                               "rate limit / cool-down - retried later, nothing was lost"))
    lines += [f"- **Gate:** {e.get('gate') or e.get('trust') or 'n/a'}", f"- **Tests:** {_tests(e)}",
              f"- **Evidence:** `{e.get('evidence', '')}` (diff.patch, evaluation.json, cycle.json) and the ledger", ""]
    return "\n".join(lines)


def waiting(state: Path, evs: Sequence[Mapping[str, Any]], st: Mapping[str, Any]) -> list[str]:
    """What Nupen is waiting on the owner (or teacher) for, right now."""
    out = []
    refused = {e.get("package") for e in evs if e.get("refused") == "trust"}
    pend = sorted(p.name for p in (state / "pending").glob("*.patch")) if (state / "pending").is_dir() else []
    for pkg in sorted(x for x in refused if x):
        files = [p for p in pend if p.startswith(f"{pkg}_")]
        out.append(f"- Review **{pkg}**: Nupen's own change was held back by the trust gate"
                   + (f" (diff: state/creator/pending/{files[-1]})" if files else ""))
    hand = sorted((state / "handoffs").glob("*")) if (state / "handoffs").is_dir() else []
    if hand:
        out.append(f"- {len(hand)} package(s) handed to the teacher are waiting in state/creator/handoffs/ "
                   f"({', '.join(h.name for h in hand[:5])}{' ...' if len(hand) > 5 else ''})")
    if st.get("rate") not in (None, "open"):
        out.append(f"- Unsupervised adoption is paused: {st['rate']}")
    trust = st.get("trust") or {}
    if not trust:
        out.append("- No measured coding setup is named in state/creator/safety.json (trust_records): every change Nupen makes on its "
                   "own stays supervised until the owner names one that passed the coding trust gate")
    for by, classes in trust.items():
        opened = [c for c, v in classes.items() if v == "open"]
        out.append(f"- Worker `{by}` may change on its own: {', '.join(opened) if opened else 'nothing yet (no class has passed the gate)'}")
    return out or ["- Nothing."]


def digest(state: Path, day: str, now: Optional[float] = None) -> str:
    from creator import safety as SF
    evs = [e for e in SF.events(state) if _day(float(e.get("at", 0))) == day]
    shown = [e for e in evs if e.get("outcome") in SHOWN or e.get("refused")]
    st = SF.status(state, now)
    pol = st["policy"]
    count = {k: sum(1 for e in shown if e.get("outcome") == k) for k in ("ADOPTED", "ROLLED_BACK")}
    refusals = sum(1 for e in shown if e.get("refused"))
    lines = [f"# Nupen change digest - {day}", "",
             f"{count['ADOPTED']} adopted, {count['ROLLED_BACK']} rolled back, {refusals} refused by the safety loop "
             f"({len(evs)} cycles recorded).", "",
             "## Gate state", "",
             f"- Unsupervised adoptions: at most {pol['max_adoptions_per_hour']} per hour, none for {pol['cooldown_hours']:g} h after a "
             f"rollback; now: **{st['rate']}**",
             f"- Supervised workers (trust gate does not apply): {', '.join(pol['supervised']) or 'none'}",
             f"- Full protected suite after every merge: {'on' if pol['full_suite'] else 'OFF'}"]
    for by, classes in (st.get("trust") or {}).items():
        lines.append(f"- Coding trust gate for `{by}`: " + "; ".join(f"{c} {'OPEN' if v == 'open' else 'closed'}" for c, v in classes.items()))
    lines += ["", "## Changes", ""] + ([event_text(e) for e in shown] or ["Nothing was adopted, rolled back or refused today.", ""])
    lines += ["## Waiting on the owner", ""] + waiting(state, evs, st)
    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--date", default=dt.date.today().isoformat())
    ap.add_argument("--state", default=str(ROOT / "state" / "creator"))
    ap.add_argument("--stdout", action="store_true")
    a = ap.parse_args(argv)
    state = Path(a.state)
    text = digest(state, a.date)
    if a.stdout:
        print(text)
        return 0
    out = state / "digest" / f"{a.date}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
