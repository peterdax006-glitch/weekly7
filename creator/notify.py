"""Owner push notifications through ntfy (stdlib only, lazy, optional).

Topic: <runtime>/phone/ntfy_topic.txt or env NUPEN_NTFY_TOPIC (never in the repo). Server: env NUPEN_NTFY_SERVER, default https://ntfy.sh.
Phone-server base URL for decision buttons: <runtime>/phone/base_url.txt or env NUPEN_PHONE_URL (absent -> no decision buttons).
Not configured -> every call is a silent no-op. A notification NEVER raises into the caller.
Guards: at most MAX_PER_HOUR per hour and BURST per minute; an identical message is dropped within DEDUPE_S; text is scrubbed of paths,
hosts, addresses and secret-looking strings and truncated. The sent-log lives outside the repo: <runtime>/notify/sent.jsonl.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.parse
from pathlib import Path
from typing import Any, Optional, Sequence

DEFAULT_SERVER = "https://ntfy.sh"
MAX_PER_HOUR = 12
BURST = 3                      # per 60 s
DEDUPE_S = 600.0
MAX_TITLE = 80
MAX_MSG = 300
TIMEOUT_S = 5.0
PRIORITIES = ("min", "low", "default", "high", "max", "urgent")
SHORTCUT_DEFAULT = "Nupen"

_PATH = re.compile(r"(?:[A-Za-z]:)?[\\/](?:[\w.\- ]+[\\/])+([\w.\-]+)")
_SECRET = re.compile(r"\b(?:sk|pk|rk|ghp|gho|ghs|hf|xox[abprs])[-_][A-Za-z0-9_\-]{16,}|\b[A-Za-z0-9+/_\-]{32,}\b")
_IP = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_HOST = re.compile(r"\b[\w-]+@[\w.-]+\b|\b(?:ssh|proxy|gpu)\d*\.vast\.ai\b", re.I)


def _runtime() -> Path:
    from creator import device as DEV
    return Path(DEV.runtime_dir())


def scrub(text: str, limit: int = MAX_MSG) -> str:
    """Strip what must never leave the machine: paths (keep the file name), addresses, user@host, long key-like strings."""
    t = _PATH.sub(lambda m: m.group(1), str(text))
    t = _HOST.sub("<host>", t)
    t = _IP.sub("<ip>", t)
    t = _SECRET.sub("<secret>", t)
    t = " ".join(t.split())
    return t[:limit]


def _ascii(t: str) -> str:
    return t.encode("ascii", "replace").decode("ascii").replace("\r", " ").replace("\n", " ")


def topic() -> Optional[str]:
    if "PYTEST_CURRENT_TEST" in os.environ and not os.environ.get("NUPEN_NTFY_SERVER"):
        return None                                     # a test run never pushes to the real phone (tests point NUPEN_NTFY_SERVER at a fake)
    v = os.environ.get("NUPEN_NTFY_TOPIC", "").strip()
    if not v:
        try:
            v = (_runtime() / "phone" / "ntfy_topic.txt").read_text(encoding="utf-8").strip()
        except OSError:
            return None
    return v if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", v) else None


def server() -> str:
    return (os.environ.get("NUPEN_NTFY_SERVER", "").strip() or DEFAULT_SERVER).rstrip("/")


def phone_url() -> Optional[str]:
    v = os.environ.get("NUPEN_PHONE_URL", "").strip()
    if not v:
        try:
            v = (_runtime() / "phone" / "base_url.txt").read_text(encoding="utf-8").strip()
        except OSError:
            return None
    return v.rstrip("/") or None


def _log_path() -> Path:
    return _runtime() / "notify" / "sent.jsonl"


def _recent(now: float, horizon: float = 3600.0) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        for ln in _log_path().read_text(encoding="utf-8").splitlines()[-400:]:
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if isinstance(r, dict) and now - float(r.get("at", 0)) <= horizon:
                out.append(r)
    except OSError:
        pass
    return out


def _record(now: float, key: str, title: str, message: str, status: str) -> None:
    try:
        p = _log_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as h:
            h.write(json.dumps({"at": now, "key": key, "title": title, "message": message, "status": status}) + "\n")
    except OSError:
        pass


def view_action(label: str, url: str) -> str:
    return f"view, {_ascii(label)}, {url}"


def http_action(label: str, url: str, headers: dict[str, str], body: str) -> str:
    hs = ", ".join(f"headers.{k}={v}" for k, v in headers.items())
    return f"http, {_ascii(label)}, {url}, method=POST, {hs}, body='{body}', clear=true"


def shortcut_name() -> str:
    """iPhone shortcut to open: env NUPEN_SHORTCUT, else <runtime>/phone/shortcut_name.txt, else 'Nupen'."""
    v = os.environ.get("NUPEN_SHORTCUT", "").strip()
    if not v:
        try:
            v = (_runtime() / "phone" / "shortcut_name.txt").read_text(encoding="utf-8").strip()
        except OSError:
            v = ""
    return v[:64] or SHORTCUT_DEFAULT


def talk_action() -> str:
    return view_action("Talk to Nupen", "shortcuts://run-shortcut?name=" + urllib.parse.quote(shortcut_name()))


def decision_actions(decision_id: str, token: Optional[str] = None) -> list[str]:
    """Approve / Later buttons: http POST /decision {id, choice} with a key scoped to this one id (never the master token)."""
    base = phone_url()
    if token is None:
        try:
            from creator import decisions as D
            token = (D.default_dir() / "token.txt").read_text(encoding="utf-8").strip()
        except OSError:
            token = ""
    if not base or not token:
        return []
    from creator import decisions as D
    hdr = {"X-Decision-Key": D.decision_key(token, decision_id), "Content-Type": "application/json"}
    return [http_action(lbl, base + "/decision", hdr, json.dumps({"id": decision_id, "choice": ch}, separators=(",", ":")))
            for lbl, ch in (("Approve", "approve"), ("Later", "later"))]


def notify(title: str, message: str, priority: str = "default", tags: Sequence[str] = (), actions: Sequence[str] = (),
           click: Optional[str] = None, now: Optional[float] = None) -> bool:
    """Send one push. True only if it was sent. Silent no-op (False) when not configured, rate-limited, duplicate, or on any error."""
    try:
        tp = topic()
        if not tp:
            return False
        now = time.time() if now is None else now
        title, message = scrub(title, MAX_TITLE), scrub(message)
        if not message and not title:
            return False
        key = hashlib.sha256((title + "\n" + message).encode()).hexdigest()[:16]
        recent = _recent(now)
        sent = [r for r in recent if r.get("status") == "sent"]
        if any(r.get("key") == key and now - float(r["at"]) <= DEDUPE_S for r in sent):
            return False
        if len(sent) >= MAX_PER_HOUR or sum(1 for r in sent if now - float(r["at"]) <= 60.0) >= BURST:
            _record(now, key, title, message, "rate_limited")
            return False
        import urllib.request
        acts = list(actions) or [talk_action()]
        h = {"Title": _ascii(title) or "Nupen", "Priority": priority if priority in PRIORITIES else "default",
             "Actions": "; ".join(acts[:3])}
        if tags:
            h["Tags"] = ",".join(re.sub(r"[^\w-]", "", t) for t in tags)
        if click:
            h["Click"] = click
        req = urllib.request.Request(f"{server()}/{urllib.parse.quote(tp)}", data=message.encode("utf-8"), headers=h, method="POST")
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:                  # noqa: S310 - fixed https server, topic quoted
            ok = 200 <= r.status < 300
        _record(now, key, title, message, "sent" if ok else "failed")
        return ok
    except Exception:                                                              # a notification must never break the caller
        return False


def notify_decision(decision_id: str, title: str, message: str, priority: str = "high") -> bool:
    """Ask the owner: registers a pending decision (read by the engine) and sends Approve / Later / Talk buttons."""
    try:
        from creator import decisions as D
        if topic() is None or not D.ask(decision_id, title):
            return False
        return notify(title, message, priority, ("question",), decision_actions(decision_id) + [talk_action()])
    except Exception:
        return False


# ---------------------------------------------------------------------------------------------- event hooks (one-line calls elsewhere)
def adopt_outcome(res: dict[str, Any]) -> bool:
    """Engine adopt cycle: ADOPTED (with the measured saving) or PROPOSED (needs the owner's decision: Approve / Later)."""
    try:
        out, cid, v = str(res.get("outcome")), str(res.get("id", "?")), res.get("verdict") or {}
        if out == "ADOPTED":
            sv = v.get("saving")
            return notify("Nupen adopted a change", f"{cid}: adopted" + (f", measured saving {float(sv) * 100:.0f}%" if isinstance(sv, (int, float)) else ""),
                          tags=("white_check_mark",))
        if out == "PROPOSED":
            return notify_decision(re.sub(r"[^A-Za-z0-9_.-]", "_", f"adopt-{cid}")[:64], "Nupen needs your decision", f"{cid}: {str(res.get('reason', ''))[:160]}")
    except Exception:
        pass
    return False


def stuck_parked(goal: str, reason: str) -> bool:
    """A goal exhausted its ladder and is parked to the owner rung."""
    return notify("Nupen is stuck", f"{goal}: parked for you. {reason[:160]}", "high", ("warning",))


def gpu_event(kind: str, detail: str = "") -> bool:
    """GPU operator: rental_idle | budget_near_cap | job_done."""
    t = {"rental_idle": ("GPU rental is idle", "default", "zzz"), "budget_near_cap": ("GPU budget near its cap", "high", "moneybag"),
         "job_done": ("GPU job done", "default", "checkered_flag")}.get(kind)
    return bool(t) and notify(t[0], detail or t[0], t[1], (t[2],))


def queue_finished(name: str, done: int, failed: int = 0) -> bool:
    return notify("Nupen finished a queue", f"{name}: {done} done" + (f", {failed} failed" if failed else ""), tags=("tada",))


def reminder(text: str) -> bool:
    return notify("Reminder", text, "high", ("alarm_clock",))
