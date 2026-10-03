"""ANTICIPATION RATE - the top-level trust metric (owner, 2 Oct 2026: "I want it to get smart enough where me talking is a waste because it already
knows when to think outside of the box, or change direction, or use xyz ... no new idea is gonna mean anything to it").

    anticipation rate = share of the owner's directives that Nupen had ALREADY proposed, in writing, BEFORE the owner said them.

Directives (time-stamped): Masterstock/OWNER_MESSAGES.md (verbatim, UTC) and the dated '- HH:MM OWNER' lines of Masterstock/JOURNAL.md (local time
MDT = UTC-6, the date comes from the nearest preceding date heading / '- D Oct HH:MM' line, rolling over when the clock jumps back by more than 6 h).
Nupen's own ideas (time-stamped): goal_proposals.jsonl, constraints.jsonl (limiting factors named and the remedies it chose), plan_explanations.jsonl
and every dated BLUEPRINT snapshot in state/creator/blueprints/ (scripts/nupen_blueprint.py keeps one per run).

HONEST MATCHING (reviewed on the real data, see tests): a directive is ANTICIPATED only if an artefact dated EARLIER shares at least MIN_SHARED
distinctive content words with it AND the idf-weighted share of the smaller text reaches MATCH_SCORE. Echo is removed first: any word the owner
had already said (in any earlier directive) before the artefact's date is deleted from that artefact, so a blueprint that repeats the owner's
last sentence anticipates nothing. Every matched pair is stored for audit (state/creator/thinking/anticipation.json). Directives before Nupen's
first artefact are reported separately (nothing could have anticipated them); the headline rate is over ALL directives, `rate_eligible` over those
that came after Nupen began writing.

BACKLOG: the unanticipated directives are training targets; each carries the earlier Nupen signals closest to it (below threshold), i.e. what Nupen
could have read. DRILL: for each directive, predict its content words from information available just before it (recent directives + recent Nupen
artefacts) and score precision@10 against a baseline (the ten most frequent earlier owner words), paired, with a 95% CI."""
from __future__ import annotations

import datetime as dt
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from creator import thinking as T

DEFAULT_OWNER_DIR = Path.home() / "Masterstock"            # no hard-coded user path (device portability)
MIN_SHARED = 3
MATCH_SCORE = 0.30
TOP_K = 10
RECENT_DIRECTIVES = 5
RECENT_ARTEFACT_S = 86400.0
STOP = frozenset("""about above after again all also and any are because been before being below between both but can could did does doing down during each
few for from further had has have having her here hers him his how into its just more most not now off once only other our out over own same she should
some such than that the their them then there these they this those through too under until very was were what when where which while who whom why will
with would you your yours need needs want wants make made get gets use used using one two new thing things like way ways also still keep must may might
then than let lets there's i'm it's dont don't cant can't wont won't""".split())


@dataclass
class Doc:
    t: float                  # UTC epoch
    text: str
    source: str
    ref: str = ""


_TOKENS: dict[str, tuple[str, ...]] = {}


def tokens(text: str) -> set[str]:
    """Content words of `text` (memoised per text: h38, 3 Oct - anticipate() re-tokenised the same directives ~64,000 times per trust
    report, 12.7 of its 18 s). Always a fresh set, built by adding the words in their first-seen order exactly like the uncached loop, so
    even its iteration order (tie-breaks downstream) is unchanged."""
    hit = _TOKENS.get(text)
    if hit is None:
        if len(_TOKENS) >= 20000:
            _TOKENS.clear()
        hit = _TOKENS[text] = tuple(dict.fromkeys(_words(text)))
    return set(hit)


def _words(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z][a-z0-9_]{3,}", text.lower()):
        if w in STOP:
            continue
        for suf in ("ing", "ed", "es", "s"):
            if w.endswith(suf) and len(w) - len(suf) >= 4:
                w = w[: -len(suf)]
                break
        out.append(w)
    return out


# ------------------------------------------------------------------------------------------------ sources
def _utc(y: int, mo: int, d: int, hh: int, mm: int, local: bool) -> float:
    return dt.datetime(y, mo, d, hh, mm, tzinfo=dt.timezone.utc).timestamp() + (6 * 3600 if local else 0)


_MON = {"sep": 9, "oct": 10, "nov": 11}


def owner_messages(owner_dir: Path) -> list[Doc]:
    p = owner_dir / "OWNER_MESSAGES.md"
    if not p.is_file():
        return []
    out: list[Doc] = []
    cur: Optional[tuple[float, str]] = None
    body: list[str] = []

    def flush() -> None:
        if cur and body:
            out.append(Doc(cur[0], " ".join(body), "owner_messages", cur[1]))
    for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"### (\d{4})-(\d\d)-(\d\d) (\d\d):(\d\d) UTC", ln)
        if m:
            flush()
            body = []
            y, mo, d, hh, mm = (int(x) for x in m.groups())
            cur = (_utc(y, mo, d, hh, mm, False), f"{y}-{mo:02d}-{d:02d} {hh:02d}:{mm:02d}Z")
        elif ln.startswith(">") and cur:
            body.append(ln.lstrip("> ").strip())
    flush()
    return out


def journal_directives(owner_dir: Path) -> list[Doc]:
    p = owner_dir / "JOURNAL.md"
    if not p.is_file():
        return []
    out: list[Doc] = []
    date: Optional[tuple[int, int, int]] = None
    last_min = -1
    for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
        s = ln.strip()
        m = re.match(r"^(?:#+\s*|-\s*)(\d{4})-(\d\d)-(\d\d)\b", s)
        if m:
            date = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            last_min = -1
        else:
            m = re.search(r"\b(\d{1,2}) (Sep|Oct|Nov)[a-z]* (?:(\d{4}),? )?(?:\d\d:\d\d)?", s) if s.startswith(("#", "- ")) else None
            if m and not re.search(r"OWNER", s[:40]) or (m and re.match(r"^- \d{1,2} (Sep|Oct)", s)):
                date = (2026, _MON[m.group(2).lower()], int(m.group(1)))      # type: ignore[union-attr]
                last_min = -1
        m = re.match(r"^-\s+(?:\d{1,2} (?:Sep|Oct) )?(\d\d):(\d\d)\s+OWNER\b[^:]*:?\s*(.*)", s)
        if m and date:
            hh, mm = int(m.group(1)), int(m.group(2))
            y, mo, d = date
            if last_min - (hh * 60 + mm) > 360:                                       # the clock jumped back by more than 6 h: it is the next day
                dd = dt.date(y, mo, d) + dt.timedelta(days=1)
                date, (y, mo, d), last_min = (dd.year, dd.month, dd.day), (dd.year, dd.month, dd.day), -1
            last_min = max(last_min, hh * 60 + mm)
            out.append(Doc(_utc(y, mo, d, hh, mm, True), m.group(3), "journal", f"{y}-{mo:02d}-{d:02d} {hh:02d}:{mm:02d}L"))
    return out


def directives(owner_dir: Path) -> list[Doc]:
    ds = [d for d in owner_messages(owner_dir) + journal_directives(owner_dir) if len(tokens(d.text)) >= 3]
    return sorted(ds, key=lambda d: d.t)


def blueprints_dir(state: Path) -> Path:
    return state / "blueprints"


def snapshot_blueprint(state: Path, text: str, now: Optional[float] = None) -> Path:
    """Keep a dated copy of every blueprint (later directives are matched against the ones that existed before them)."""
    t = dt.datetime.fromtimestamp(now if now is not None else dt.datetime.now(dt.timezone.utc).timestamp(), dt.timezone.utc)
    p = blueprints_dir(state) / f"BLUEPRINT_{t.strftime('%Y%m%dT%H%M%SZ')}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def nupen_artefacts(state: Path) -> list[Doc]:
    out: list[Doc] = []
    for r in T._jsonl(state / "goal_proposals.jsonl"):
        if r.get("event") == "proposal":
            out.append(Doc(T._local_ts(r.get("created", "1970-01-01T00:00:00")),
                           " ".join(str(r.get(k, "")) for k in ("key", "title", "rationale", "text")) + " " + json.dumps(r.get("spec", {})),
                           "goal_proposal", r.get("id", "")))
    for r in T._jsonl(state / "constraints.jsonl"):
        t = T._local_ts(r["at"]) if r.get("at") else 0.0
        if r.get("event") == "act":
            out.append(Doc(t, f"{r.get('constraint', '')} {r.get('text', '')} {r.get('note', '')}", "constraint_act", r.get("constraint", "")))
        elif r.get("event") == "snapshot":
            out.append(Doc(t, " ".join(f"{x.get('name', '')} {x.get('unit', '')}" for x in (r.get("ranked") or [])[:3]) + " " + " ".join(r.get("messages") or []),
                           "constraint_snapshot", r["at"]))
    for r in T._jsonl(state / "plan_explanations.jsonl"):
        if r.get("chosen") and r.get("at"):
            out.append(Doc(T._ts(r["at"].replace("Z", "+00:00")), f"{r.get('component', '')} {r.get('step', '')} {r.get('why', '')}", "plan", r.get("node", "")))
    bd = blueprints_dir(state)
    if bd.is_dir():
        for f in sorted(bd.glob("BLUEPRINT_*.md")):
            m = re.match(r"BLUEPRINT_(\d{8}T\d{6})Z", f.name)
            if m:
                t = dt.datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=dt.timezone.utc).timestamp()
                out.append(Doc(t, f.read_text(encoding="utf-8", errors="replace"), "blueprint", f.name))
    return sorted(out, key=lambda d: d.t)


# ------------------------------------------------------------------------------------------------ matching
def _idf(docs: Sequence[Doc]) -> dict[str, float]:
    df: dict[str, int] = {}
    for d in docs:
        for w in tokens(d.text):
            df[w] = df.get(w, 0) + 1
    n = len(docs) + 1
    return {w: math.log(n / (1 + c)) + 0.1 for w, c in df.items()}


def match(directive: Doc, art: Doc, said_before: set[str], idf: dict[str, float]) -> tuple[float, list[str]]:
    """(score, shared novel words): echo of earlier owner words is removed from the artefact first."""
    a = tokens(art.text) - said_before
    dtoks = tokens(directive.text)
    shared = sorted(a & dtoks, key=lambda w: -idf.get(w, 0.0))
    if not a or not dtoks:
        return 0.0, []
    si = sum(idf.get(w, 0.0) for w in shared)
    denom = min(sum(idf.get(w, 0.0) for w in a), sum(idf.get(w, 0.0) for w in dtoks)) or 1.0
    return si / denom, shared


def anticipate(dirs: Sequence[Doc], arts: Sequence[Doc]) -> dict[str, Any]:
    idf = _idf(list(dirs) + list(arts))
    first_art = arts[0].t if arts else float("inf")
    anticipated: list[dict[str, Any]] = []
    backlog: list[dict[str, Any]] = []
    for i, d in enumerate(dirs):
        best: Optional[tuple[float, list[str], Doc]] = None
        weak: list[tuple[float, list[str], Doc]] = []
        for a in arts:
            if a.t >= d.t:
                break
            said = set().union(*(tokens(x.text) for x in dirs if x.t < a.t)) if any(x.t < a.t for x in dirs) else set()
            sc, sh = match(d, a, said, idf)
            weak.append((sc, sh, a))
            if len(sh) >= MIN_SHARED and sc >= MATCH_SCORE and (best is None or sc > best[0]):
                best = (sc, sh, a)
        row = {"directive": d.ref, "source": d.source, "text": d.text[:300], "at": d.t}
        if best:
            anticipated.append({**row, "matched": {"artefact": best[2].source + ":" + best[2].ref, "artefact_at": best[2].t, "score": round(best[0], 3),
                                                   "shared_words": best[1][:12], "lead_s": round(d.t - best[2].t)}})
        else:
            weak.sort(key=lambda x: -x[0])
            backlog.append({**row, "eligible": d.t > first_art, "closest_earlier_signals": [
                {"artefact": a.source + ":" + a.ref, "score": round(s, 3), "shared_words": sh[:6]} for s, sh, a in weak[:3] if s > 0]})
    elig = [d for d in dirs if d.t > first_art]
    ant_e = [a for a in anticipated if float(a["at"]) > first_art]
    return {"directives": len(dirs), "anticipated": len(anticipated), "rate": round(len(anticipated) / len(dirs), 4) if dirs else None,
            "eligible_directives": len(elig), "rate_eligible": round(len(ant_e) / len(elig), 4) if elig else None,
            "first_nupen_artefact": first_art if arts else None, "thresholds": {"min_shared": MIN_SHARED, "match_score": MATCH_SCORE},
            "pairs": anticipated, "backlog": backlog}


def drill(dirs: Sequence[Doc], arts: Sequence[Doc]) -> dict[str, Any]:
    """Predict each directive's content words from what existed just before it; precision@TOP_K vs the most frequent earlier owner words."""
    model, base = [], []
    for i in range(3, len(dirs)):
        d = dirs[i]
        prev = dirs[max(0, i - RECENT_DIRECTIVES):i]
        recent_art = [a for a in arts if d.t - RECENT_ARTEFACT_S <= a.t < d.t]
        idf = _idf(list(dirs[:i]) + [a for a in arts if a.t < d.t])
        score: dict[str, float] = {}
        for rank, x in enumerate(reversed(prev)):
            for w in tokens(x.text):
                score[w] = score.get(w, 0.0) + idf.get(w, 0.1) * (0.8 ** rank)
        for a in recent_art:
            for w in tokens(a.text):
                score[w] = score.get(w, 0.0) + 0.5 * idf.get(w, 0.1)
        freq: dict[str, int] = {}
        for x in dirs[:i]:
            for w in tokens(x.text):
                freq[w] = freq.get(w, 0) + 1
        top = [w for w, _ in sorted(score.items(), key=lambda kv: -kv[1])[:TOP_K]]
        topb = [w for w, _ in sorted(freq.items(), key=lambda kv: -kv[1])[:TOP_K]]
        dt_ = tokens(d.text)
        model.append(sum(1 for w in top if w in dt_) / TOP_K)
        base.append(sum(1 for w in topb if w in dt_) / TOP_K)
    n = len(model)
    if not n:
        return {"n": 0}
    diff = [m - b for m, b in zip(model, base)]
    dm = sum(diff) / n
    se = math.sqrt(sum((x - dm) ** 2 for x in diff) / (n - 1) / n) if n > 1 else float("inf")
    return {"n": n, "precision_at_10_model": round(sum(model) / n, 4), "precision_at_10_baseline": round(sum(base) / n, 4),
            "gain": round(dm, 4), "gain_ci95": [round(dm - 1.96 * se, 4), round(dm + 1.96 * se, 4)] if n > 1 else [None, None]}


def report(state: Path, owner_dir: Optional[Path] = None, write: bool = True) -> dict[str, Any]:
    """The anticipation report; with write, also state/creator/thinking/anticipation.json (pairs and backlog, for audit)."""
    od = Path(owner_dir) if owner_dir is not None else DEFAULT_OWNER_DIR
    dirs, arts = directives(od), nupen_artefacts(state)
    rep = anticipate(dirs, arts)
    rep["drill"] = drill(dirs, arts)
    rep["artefacts"] = len(arts)
    if write:
        p = state / "thinking" / "anticipation.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rep, indent=1, sort_keys=True), encoding="utf-8")
    return rep


def summary(rep: dict[str, Any]) -> dict[str, Any]:
    """The headline for trust.json (no pair/backlog bodies)."""
    return {k: rep.get(k) for k in ("rate", "anticipated", "directives", "rate_eligible", "eligible_directives", "thresholds", "drill", "artefacts")} | \
        {"backlog": len(rep.get("backlog", [])), "target": "owner talking is a waste: rate -> 1.0"}
