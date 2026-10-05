"""Conversation learning (R12): Nupen's phone log -> failure detectors -> engine candidates, a capability backlog, and voice training rows.

The log (phone/conversations.jsonl: t, device, in, reply, intent, action, ms) is the owner's private data: it lives outside the repo, is
read here, and is never copied into tracked files. Candidates carry only anonymized, truncated excerpts. Voice rows are owner-only
training data for the Phase 2 voice module: never committed, never uploaded except to the owner's rented GPU for training.

Locations (all derived from the state dir so tests on a temp state touch nothing real; env overrides): NUPEN_PHONE_DIR, NUPEN_VOICE_DIR.
Lazy: nothing is read until a function is called; failures never raise into the engine.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import time as _time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional

SYNTHETIC_DEVICES = ("warmup", "probe")      # health probes, not the owner
SESSION_GAP_S = 1800.0
REPEAT_TURNS = 2
REPEAT_SIM = 0.75
LONG_REPLY_WORDS = 120
SHORT_Q_WORDS = 8
EXCERPT = 60
LOG_NAME = "conversations.jsonl"
BACKLOG_NAME = "capability_backlog.json"
VOICE_FILE = "conv_rows.jsonl"
FIX_KINDS = ("new_action", "style", "bug", "knowledge")
SUBTYPE_FIX = {"repeat": "knowledge", "correction": "style", "unsupported": "new_action", "action_failed": "bug", "no_url": "bug",
               "leak": "bug", "long": "style", "abandon": "knowledge"}
# Value model (engine currency: CPU-seconds): a conversation failure costs the owner about a minute of attention.
C0_FAIL, C1_FIXED, P_FIX, BUILD_S = 60.0, 0.0, 0.5, 1800.0

_CORRECT = re.compile(r"(that'?s not what i (asked|meant|said)|not what i asked|\bstupid\b|\bwrong\b|you didn'?t|you did not|\bidiot\b|\buseless\b|"
                      r"^\s*no[\s,.!]*($|that|not|i said|i meant|wrong)|\bno,? (that|i)\b|try again|that'?s (wrong|incorrect))", re.I)
_DIDNT_WORK = re.compile(r"(it didn'?t work|that didn'?t work|did not work|doesn'?t work|didn'?t (open|play|start|set|show)|nothing happened|still not working)", re.I)
_UNSUPPORTED = re.compile(r"(cannot do that|can'?t do that|not able to|unable to|don'?t (yet )?have the (ability|capability)|beyond my (current )?capabilit|"
                          r"not (yet )?supported|i'?m afraid i can(not|'?t))", re.I)
_LEAK = re.compile(r"(\[doc:|[A-Za-z]:\\|/Users/|/home/|\.py\b|\.jsonl?\b|^\s*[\[{]|\"\w+\"\s*:)", re.I)
_URL_ACTIONS = ("open", "play", "search", "navigate", "link", "browse", "video", "music")


# ------------------------------------------------------------------------------------------------ paths and reading
def phone_dir(state: Optional[Path] = None) -> Path:
    env = os.environ.get("NUPEN_PHONE_DIR")
    if env:
        return Path(env)
    return Path(state).parent / "phone" if state is not None else Path("phone")


def voice_dir(state: Optional[Path] = None) -> Path:
    env = os.environ.get("NUPEN_VOICE_DIR")
    if env:
        return Path(env)
    return (Path(state).parent / "gpuday" / "phase2_data" / "voice") if state is not None else Path("voice")


def _ts(s: Any) -> float:
    try:
        return dt.datetime.strptime(str(s), "%Y-%m-%d %H:%M:%S").timestamp()
    except Exception:                                                   # noqa: BLE001
        try:
            return float(s)
        except Exception:                                               # noqa: BLE001
            return 0.0


def load_log(path: Path, days: Optional[float] = None, now: Optional[float] = None, exclude: tuple[str, ...] = SYNTHETIC_DEVICES) -> list[dict[str, Any]]:
    """Rows sorted by time, each with a float 'ts'; bad lines skipped; synthetic devices dropped; optional day window."""
    now = _time.time() if now is None else now
    out: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:                                       # noqa: BLE001
                    continue
                if not isinstance(r, dict) or str(r.get("device")) in exclude:
                    continue
                r["ts"] = _ts(r.get("t"))
                r["in"], r["reply"] = str(r.get("in") or ""), str(r.get("reply") or "")
                if days is None or now - r["ts"] <= days * 86400.0:
                    out.append(r)
    except OSError:
        return []
    return sorted(out, key=lambda r: r["ts"])


def sessions(rows: list[dict[str, Any]], gap_s: float = SESSION_GAP_S) -> list[list[dict[str, Any]]]:
    by: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(str(r.get("device")), []).append(r)
    out: list[list[dict[str, Any]]] = []
    for rs in by.values():
        cur: list[dict[str, Any]] = []
        for r in rs:
            if cur and r["ts"] - cur[-1]["ts"] > gap_s:
                out.append(cur)
                cur = []
            cur.append(r)
        if cur:
            out.append(cur)
    return out


# ------------------------------------------------------------------------------------------------ detectors (one per failure class)
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", s.lower()).strip()


def _similar(a: str, b: str) -> bool:
    a, b = _norm(a), _norm(b)
    return bool(a and b) and (a == b or SequenceMatcher(None, a, b).ratio() >= REPEAT_SIM)


def _fail(subtype: str, r: dict[str, Any], i: int) -> dict[str, Any]:
    return {"subtype": subtype, "fix": SUBTYPE_FIX[subtype], "ts": r["ts"], "turn": i, "text": r["in"], "reply": r["reply"]}


def detect_repeat(sess: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for i, r in enumerate(sess):
        if any(_similar(r["in"], sess[j]["in"]) for j in range(i + 1, min(len(sess), i + 1 + REPEAT_TURNS))):
            out.append(_fail("repeat", r, i))
    return out


def detect_correction(sess: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The owner's frustration/correction phrase blames the previous answer."""
    return [_fail("correction", sess[i - 1], i - 1) for i in range(1, len(sess)) if _CORRECT.search(sess[i]["in"])]


def detect_unsupported(sess: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for i, r in enumerate(sess):
        if _UNSUPPORTED.search(r["reply"]):
            out.append(_fail("unsupported", r, i))
        if i and _DIDNT_WORK.search(r["in"]):
            out.append(_fail("action_failed", sess[i - 1], i - 1))
        a = r.get("action")
        if r.get("intent") == "action" and (not isinstance(a, dict) or (str(a.get("type", "")).lower() in _URL_ACTIONS and not a.get("url"))):
            out.append(_fail("no_url", r, i))
    return out


def detect_leak(sess: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_fail("leak", r, i) for i, r in enumerate(sess) if _LEAK.search(r["reply"])]


def detect_long(sess: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_fail("long", r, i) for i, r in enumerate(sess)
            if len(r["reply"].split()) > LONG_REPLY_WORDS and len(r["in"].split()) <= SHORT_Q_WORDS]


def detect_abandon(sess: list[dict[str, Any]], now: float, bad_turns: set[int]) -> list[dict[str, Any]]:
    """The session is over (quiet for a gap) and its last answer was bad."""
    if sess and len(sess) - 1 in bad_turns and now - sess[-1]["ts"] > SESSION_GAP_S:
        return [_fail("abandon", sess[-1], len(sess) - 1)]
    return []


def failures(sess: list[dict[str, Any]], now: Optional[float] = None) -> list[dict[str, Any]]:
    now = _time.time() if now is None else now
    out: list[dict[str, Any]] = []
    for fn in (detect_repeat, detect_correction, detect_unsupported, detect_leak, detect_long):
        out.extend(fn(sess))
    bad = {f["turn"] for f in out if f["subtype"] in ("unsupported", "no_url", "leak", "long", "correction", "action_failed", "repeat")}
    out.extend(detect_abandon(sess, now, bad))
    return out


def detect_failures(rows: list[dict[str, Any]], now: Optional[float] = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in sessions(rows):
        out.extend(failures(s, now))
    return out


# ------------------------------------------------------------------------------------------------ anonymising, candidates, backlog
def anonymize(text: str, n: int = EXCERPT) -> str:
    t = re.sub(r"https?://\S+|www\.\S+", "<url>", text)
    t = re.sub(r"\S+@\S+", "<email>", t)
    t = re.sub(r"[A-Za-z]:\\\S*|/(?:Users|home)/\S*", "<path>", t)
    t = re.sub(r"\d", "#", t)
    return t.strip()[:n]


def candidates(fails: list[dict[str, Any]], days: float) -> list[dict[str, Any]]:
    """One engine candidate per (subtype): {kind: conv_failure, subtype, evidence, f_per_day, proposed fix kind} in the engine's shape."""
    g: dict[str, list[dict[str, Any]]] = {}
    for f in fails:
        g.setdefault(f["subtype"], []).append(f)
    out = []
    for sub, fs in sorted(g.items()):
        fix = SUBTYPE_FIX[sub]
        out.append({"kind": "conv_failure", "key": f"conv_failure:{sub}", "subtype": sub, "fix_kind": fix,
                    "evidence": {"count": len(fs), "excerpt": anonymize(fs[-1]["text"]), "fix_kind": fix},
                    "f_per_day": round(len(fs) / max(days, 1e-9), 4), "c0": C0_FAIL, "c1": C1_FIXED, "p": P_FIX,
                    "fix": f"conversation failure '{sub}': {fix}", "risk": "low", "build_s": BUILD_S})
    return out


def backlog(fails: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Unsupported / failed-action requests aggregated by normalised request, ranked by how often they were asked."""
    g: dict[str, dict[str, Any]] = {}
    for f in fails:
        if f["subtype"] not in ("unsupported", "no_url", "action_failed"):
            continue
        k = _norm(f["text"])[:80]
        if not k:
            continue
        e = g.setdefault(k, {"request": k, "count": 0, "subtypes": set(), "last": 0.0})
        e["count"] += 1
        e["subtypes"].add(f["subtype"])
        e["last"] = max(e["last"], f["ts"])
    rk = sorted(g.values(), key=lambda e: (-e["count"], -e["last"], e["request"]))
    return [{"request": e["request"], "count": e["count"], "subtypes": sorted(e["subtypes"]), "last": e["last"]} for e in rk]


def write_backlog(path: Path, items: list[dict[str, Any]]) -> None:
    """Merge into the persisted backlog (max count seen per request, so re-running the same window does not double count)."""
    cur: dict[str, dict[str, Any]] = {}
    try:
        for e in json.loads(Path(path).read_text(encoding="utf-8")).get("items", []):
            cur[e["request"]] = e
    except Exception:                                                   # noqa: BLE001
        pass
    for e in items:
        old = cur.get(e["request"])
        cur[e["request"]] = {**e, "count": max(e["count"], old["count"]) if old else e["count"]}
    ranked = sorted(cur.values(), key=lambda e: (-e["count"], -e.get("last", 0.0), e["request"]))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"items": ranked}, indent=1), encoding="utf-8")


# ------------------------------------------------------------------------------------------------ learning data for the voice module
VOICE_SYSTEM = "You are Nupen, the owner's private assistant. Answer briefly, directly and politely; never expose internals."


def _row(kind: str, user: str, reply: str, extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    h = hashlib.sha1((kind + "\0" + user + "\0" + reply).encode("utf-8")).hexdigest()[:20]
    r = {"id": f"voice:conv:{h}", "source": "phone_conversation", "kind": kind, "split": "train", "owner_only": True,
         "messages": [{"role": "system", "content": VOICE_SYSTEM}, {"role": "user", "content": user}, {"role": "assistant", "content": reply}]}
    if extra:
        r.update(extra)
    return r


def voice_rows(rows: list[dict[str, Any]], now: Optional[float] = None) -> list[dict[str, Any]]:
    """good: no failure signal on the turn and the owner continued without correcting it. corrected: a bad answer, the owner's correction,
    then the next good answer (the training target; the bad one is kept as 'rejected')."""
    out: list[dict[str, Any]] = []
    for s in sessions(rows):
        bad = {f["turn"] for f in failures(s, now) if f["subtype"] != "abandon"}
        correction_at = {f["turn"]: f["turn"] + 1 for f in detect_correction(s)}
        for i, r in enumerate(s):
            if not r["in"].strip() or not r["reply"].strip():
                continue
            if i in bad:
                j = correction_at.get(i)
                if j is None or j >= len(s) - 1:
                    continue
                k = next((x for x in range(j + 1, len(s)) if x not in bad and s[x]["reply"].strip()), None)
                if k is not None and not _LEAK.search(s[k]["reply"]):
                    out.append(_row("corrected", s[j]["in"] if len(s[j]["in"].split()) > 4 else r["in"], s[k]["reply"],
                                    {"rejected": r["reply"], "correction": s[j]["in"]}))
            elif i + 1 < len(s) and not _CORRECT.search(s[i + 1]["in"]):
                out.append(_row("good", r["in"], r["reply"]))
    return out


def write_voice_rows(path: Path, rows: list[dict[str, Any]]) -> int:
    """Append only rows whose id is new; returns how many were added."""
    path = Path(path)
    have: set[str] = set()
    try:
        for line in open(path, "r", encoding="utf-8"):
            try:
                have.add(json.loads(line)["id"])
            except Exception:                                           # noqa: BLE001
                continue
    except OSError:
        pass
    new = [r for r in rows if r["id"] not in have]
    if new:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "ab") as f:
            for r in new:
                f.write((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8"))
    return len(new)


# ------------------------------------------------------------------------------------------------ engine hooks
def detect(state: Path, days: float = 7.0, now: Optional[float] = None) -> list[dict[str, Any]]:
    """Engine candidates from the conversation log (empty when there is none)."""
    now = _time.time() if now is None else now
    rows = load_log(phone_dir(state) / LOG_NAME, days, now)
    return candidates(detect_failures(rows, now), days) if rows else []


def side_effects(state: Path, days: float = 7.0, now: Optional[float] = None) -> dict[str, int]:
    """Refresh the capability backlog and the voice rows from the log; no-op without a log. Never raises."""
    try:
        now = _time.time() if now is None else now
        rows = load_log(phone_dir(state) / LOG_NAME, days, now)
        if not rows:
            return {"backlog": 0, "voice_rows": 0}
        items = backlog(detect_failures(rows, now))
        if items:
            write_backlog(phone_dir(state) / BACKLOG_NAME, items)
        return {"backlog": len(items), "voice_rows": write_voice_rows(voice_dir(state) / VOICE_FILE, voice_rows(rows, now))}
    except Exception:                                                   # noqa: BLE001
        return {"backlog": 0, "voice_rows": 0}
