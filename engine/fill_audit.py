"""Bible Phase 1.4 / canon C33, as a Test-loop gate: prove from a finished Session that every decision was taken at a
close and filled at the NEXT session's open - never at the same close, never on a weekend - by feeding the Session's
own records into engine.pit.audit_fills. Works on a live Session or on a replay.

The Session must expose `fills` (dicts with order_id, ticker, decision_date, fill_date, fill_price). Older Sessions
without it cannot be audited; the gate then reports UNAUDITABLE instead of passing (fail closed)."""
import pandas as pd

from . import pit


STRICT = {"late_fill"}


def session_calendar(closes: pd.DataFrame) -> pit.Calendar:
    """The window's own sessions are the calendar: a fill on a date that is not a session of the window is an error."""
    return pit.Calendar(pd.DatetimeIndex(closes.index))


def frames_from_session(S):
    """(decisions, fills) frames in the shape pit.audit_fills expects, or (None, None) if the Session keeps no fills."""
    fills = getattr(S, "fills", None)
    if fills is None:
        return None, None
    if not fills:
        return (pd.DataFrame(columns=["order_id", "ticker", "decision_date"]),
                pd.DataFrame(columns=["order_id", "ticker", "fill_date", "fill_price"]))
    F = pd.DataFrame(fills)
    D = F[["order_id", "ticker", "decision_date"]].drop_duplicates("order_id")
    return D, F[["order_id", "ticker", "fill_date", "fill_price"]]


def audit_session(S, opens: pd.DataFrame, closes: pd.DataFrame) -> dict:
    """Run the fill audit. Returns {"status": PASS|FAIL|UNAUDITABLE, "errors": [...], "n_fills": int, "summary": str}.
    opens may be None for archives built before opening prices were saved: then fills are at the close by
    construction and the audit correctly FAILS the same-close rule rather than being skipped."""
    D, F = frames_from_session(S)
    if D is None:
        return {"status": "UNAUDITABLE", "errors": ["session records no fills"], "n_fills": 0,
                "summary": "Session has no .fills - cannot prove next-open execution"}
    if opens is None:                                    # substituting closes would make close fills look legal
        return {"status": "UNAUDITABLE", "errors": ["no opening prices archived"], "n_fills": int(len(F)),
                "summary": "window has no opens - next-open execution cannot be proven"}
    rep = pit.audit_fills(D, F, opens, closes, session_calendar(closes))
    # the Session always fills at the very next open (no halts in a replay), so pit's "late_fill" warning is an error here
    errs = [f"{f.code}:{f.subject}" for f in rep.errors] +            [f"{f.code}:{f.subject}" for f in rep.findings if f.code in STRICT and f.severity != "error"]
    return {"status": "PASS" if not errs else "FAIL", "errors": errs[:50], "n_errors": len(errs),
            "n_fills": int(len(F)), "summary": rep.summary()}


def gate(S, opens, closes) -> tuple[bool, str]:
    """(ok, one-line message) for the Test loop's gate printout. UNAUDITABLE is not ok."""
    r = audit_session(S, opens, closes)
    if r["status"] == "PASS":
        return True, f"fill audit OK ({r['n_fills']} fills at next open)"
    return False, f"fill audit {r['status']}: {r.get('n_errors', 0)} errors, e.g. {r['errors'][:3]}"
