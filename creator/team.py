"""Team plumbing (MASTER_BLUEPRINT P0.8): who does which step, what they may be sent, and what they are sent.

    team = Team(state_dir, actors, goal_id="g1")        # one active goal; board + cache live in state_dir/team.sqlite
    r = team.run_task(task, ix)                           # SPEC -> LOCATE -> PLAN -> CODE -> REVIEW -> VALIDATE

Pieces (all deterministic, no model needed to test them):
  ROUTES        step type -> actor (a code tool, or a slot CHECKER / CODER / THINKER)
  Actor         declares the step types it accepts (jurisdiction); a mismatched dispatch is refused BEFORE it is sent
  Envelope      {goal_id, step, inputs[board ids | path:a-b ranges], constraints, success_test, prior (<=2 lines), budget}
                token-checked (chars / 3.6, target <= 50, hard 120); big content goes on the Board and travels as an id
  Board         SQLite facts/artifacts by id, outside the repo
  Team          goal gate (off-goal envelopes refused), jurisdiction gate, 'HANDOFF: <ROLE>' re-routing, result cache
                (exact key + minhash near-duplicate) in front of slot calls, optional event hook
Logging: pass log=callable(actor, **fields); `slowpath_log()` returns an adapter for creator.slowpath.event once that exists.
Light imports only (stdlib); the locate index is passed in by the caller.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

CHARS_PER_TOKEN = 3.6
ENVELOPE_TARGET_TOKENS = 50          # ratchet 4 Oct: measured floor ~30
ENVELOPE_MAX_TOKENS = 120
NEAR_THRESHOLD = 0.9
MAX_HOPS = 3
SLOTS = ("CHECKER", "CODER", "THINKER")

# step type -> actor. Code tools answer from files/indexes; slots are model calls (cheapest able slot first).
ROUTES: dict[str, str] = {
    "SPEC": "CHECKER", "LOCATE": "locate", "PACK": "pack", "PLAN": "THINKER", "CODE": "CODER",
    "DEBUG_PINPOINT": "locate", "DEBUG_FIX": "CODER", "REVIEW": "CHECKER", "VALIDATE": "tests",
    "SAFETY": "safety", "CONFIDENCE": "CHECKER", "SUMMARIZE": "CHECKER", "DESIGN": "THINKER",
    "BENCH": "CODER",                       # P1.3: the builder's required benchmark spec for the claimed saving (VERIFY runs it, never the builder)
}


class TeamError(ValueError):
    pass


class Refused(TeamError):
    """A dispatch refused before sending (off-goal, wrong jurisdiction, oversized, unknown step)."""


def route(step: str) -> str:
    try:
        return ROUTES[step]
    except KeyError:
        raise Refused(f"unknown step type {step!r}") from None


def est_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN + 0.999)


# ---------------------------------------------------------------------------------------------------------------------- envelope
DEFAULT_BUDGET = {"max_tok": 400, "max_s": 60}


@dataclass
class Envelope:
    goal_id: str
    step: str
    inputs: list[str] = field(default_factory=list)        # board ids ("b:ab12cd34") and file:line ranges ("creator/x.py:10-40")
    constraints: list[str] = field(default_factory=list)
    success_test: str = ""
    prior: str = ""                                       # what came before, <= 2 lines
    budget: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_BUDGET))

    def validate(self) -> None:
        if not self.goal_id:
            raise Refused("envelope has no goal_id")
        if self.step not in ROUTES:
            raise Refused(f"unknown step type {self.step!r}")
        if len([ln for ln in self.prior.splitlines() if ln.strip()]) > 2:
            raise Refused("prior longer than 2 lines")
        if not {"max_tok", "max_s"} <= set(self.budget):
            raise Refused("budget needs max_tok and max_s")

    def text(self) -> str:
        """Compact wire form: short field codes, empty fields and the default budget omitted."""
        d = {"g": self.goal_id, "s": self.step, "i": self.inputs, "c": self.constraints, "t": self.success_test, "p": self.prior}
        if self.budget != DEFAULT_BUDGET:
            d["b"] = [self.budget.get("max_tok"), self.budget.get("max_s")]
        return json.dumps({k: v for k, v in d.items() if v not in ("", [], None)}, sort_keys=True, separators=(",", ":"))

    def tokens(self) -> int:
        return est_tokens(self.text())

    def check_size(self, limit: int = ENVELOPE_TARGET_TOKENS) -> bool:
        return self.tokens() <= limit


_RANGE = re.compile(r"^[^\s:]+:\d+(-\d+)?$")
_BOARD = re.compile(r"^b:[0-9a-f]{8,}$")


def input_kind(ref: str) -> str:
    if _BOARD.match(ref):
        return "board"
    if _RANGE.match(ref):
        return "range"
    raise Refused(f"bad input reference {ref!r}")


# ---------------------------------------------------------------------------------------------------------------------- board
class Board:
    """Facts and artifacts by id, in SQLite. put() is content-addressed, so the same content is stored once."""

    def __init__(self, db: str | Path) -> None:
        Path(db).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(db))
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS board(id TEXT PRIMARY KEY, goal_id TEXT, kind TEXT, body TEXT, t REAL);
            CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
            CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY, step TEXT, actor TEXT, ins TEXT, sig TEXT, result TEXT, t REAL);
        """)

    def put(self, goal_id: str, kind: str, body: str) -> str:
        bid = "b:" + hashlib.sha256((goal_id + "\0" + kind + "\0" + body).encode()).hexdigest()[:10]
        self.db.execute("INSERT OR IGNORE INTO board VALUES(?,?,?,?,?)", (bid, goal_id, kind, body, time.time()))
        self.db.commit()
        return bid

    def get(self, bid: str) -> Optional[dict[str, Any]]:
        r = self.db.execute("SELECT id, goal_id, kind, body FROM board WHERE id=?", (bid,)).fetchone()
        return None if r is None else {"id": r[0], "goal_id": r[1], "kind": r[2], "body": r[3]}

    def by_goal(self, goal_id: str, kind: Optional[str] = None) -> list[dict[str, Any]]:
        q, a = "SELECT id, kind, body FROM board WHERE goal_id=?", [goal_id]
        if kind:
            q, a = q + " AND kind=?", a + [kind]
        return [{"id": i, "kind": k, "body": b} for i, k, b in self.db.execute(q + " ORDER BY t, id", a)]

    def meta_get(self, k: str) -> Optional[str]:
        r = self.db.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return None if r is None else r[0]

    def meta_set(self, k: str, v: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (k, v))
        self.db.commit()


# ---------------------------------------------------------------------------------------------------------------------- cache
_PERM = 32
_SEEDS = [(zlib.crc32(b"a%d" % i) | 1, zlib.crc32(b"b%d" % i)) for i in range(_PERM)]


def minhash(text: str) -> list[int]:
    """32-permutation minhash over word 2-shingles (stdlib only)."""
    w = re.findall(r"\w+", text.lower())
    sh = {" ".join(w[i:i + 2]) for i in range(max(1, len(w) - 1))} or {""}
    hs = [zlib.crc32(s.encode()) for s in sh]
    return [min((a * h + b) & 0xFFFFFFFF for h in hs) for a, b in _SEEDS]


def similarity(a: Sequence[int], b: Sequence[int]) -> float:
    return sum(x == y for x, y in zip(a, b)) / len(a)


def _resolved(board: Board, env: Envelope, actor_name: str = "") -> str:
    """Envelope with board ids replaced by their content hash, so the key follows content, not ids' goal prefix."""
    parts = []
    for ref in env.inputs:
        if input_kind(ref) == "board":
            row = board.get(ref)
            parts.append(hashlib.sha256((row["body"] if row else "?").encode()).hexdigest()[:12])
        else:
            parts.append(ref)
    return "|".join([actor_name, env.step, ",".join(sorted(parts)), ";".join(env.constraints), env.success_test, env.prior])


def _free(env: Envelope) -> str:
    return " ".join(env.constraints + [env.success_test, env.prior])


def _ins(board: Board, env: Envelope, actor_name: str) -> str:
    return _resolved(board, Envelope(env.goal_id, env.step, env.inputs), actor_name)


class ResultCache:
    def __init__(self, board: Board, near: float = NEAR_THRESHOLD) -> None:
        self.board, self.near = board, near
        self.exact_hits = self.near_hits = self.misses = 0

    def key(self, env: Envelope, actor_name: str = "") -> tuple[str, str]:
        norm = _resolved(self.board, env, actor_name)
        return hashlib.sha256(norm.encode()).hexdigest()[:24], norm

    def get(self, env: Envelope, actor_name: str = "") -> Optional[tuple[str, str]]:
        k, norm = self.key(env, actor_name)
        r = self.board.db.execute("SELECT result FROM cache WHERE key=?", (k,)).fetchone()
        if r:
            self.exact_hits += 1
            return r[0], "exact"
        free = _free(env)
        if not free.strip():
            self.misses += 1
            return None
        sig = minhash(free)
        best, hit = 0.0, None
        # near hits only among entries with the SAME step, actor and resolved inputs: only the free text may differ
        for res, s in self.board.db.execute("SELECT result, sig FROM cache WHERE step=? AND actor=? AND ins=?",
                                            (env.step, actor_name, _ins(self.board, env, actor_name))):
            sim = similarity(sig, json.loads(s))
            if sim > best:
                best, hit = sim, res
        if hit is not None and best >= self.near:
            self.near_hits += 1
            return hit, "near"
        self.misses += 1
        return None

    def put(self, env: Envelope, result: str, actor_name: str = "") -> None:
        k, norm = self.key(env, actor_name)
        self.board.db.execute("INSERT OR REPLACE INTO cache VALUES(?,?,?,?,?,?,?)",
                              (k, env.step, actor_name, _ins(self.board, env, actor_name), json.dumps(minhash(_free(env))), result, time.time()))
        self.board.db.commit()

    def stats(self) -> dict[str, float]:
        n = self.exact_hits + self.near_hits + self.misses
        return {"exact": self.exact_hits, "near": self.near_hits, "miss": self.misses,
                "hit_rate": round((self.exact_hits + self.near_hits) / n, 4) if n else 0.0}


# ---------------------------------------------------------------------------------------------------------------------- actors
@dataclass
class Actor:
    name: str
    accepts: frozenset[str]
    fn: Callable[[Envelope, "Team"], str]
    is_slot: bool = False


def actor(name: str, fn: Callable[[Envelope, "Team"], str], accepts: Optional[Sequence[str]] = None) -> Actor:
    """Default jurisdiction = every step the routing table sends to this actor."""
    acc = frozenset(accepts) if accepts is not None else frozenset(s for s, a in ROUTES.items() if a == name)
    return Actor(name, acc, fn, is_slot=name in SLOTS)


_HANDOFF = re.compile(r"^\s*HANDOFF:\s*([A-Za-z_]+)\s*$", re.M)


def slowpath_log() -> Optional[Callable[..., Any]]:
    """Adapter onto creator.slowpath.event when that module has it; None otherwise (no hard dependency)."""
    try:
        from creator import slowpath
        ev = getattr(slowpath, "event")
    except Exception:                                    # noqa: BLE001
        return None
    return lambda actor_name, **kw: ev(actor_name, **kw)


# ---------------------------------------------------------------------------------------------------------------------- team
class Team:
    def __init__(self, state: str | Path, actors: Sequence[Actor], goal_id: Optional[str] = None,
                 log: Optional[Callable[..., Any]] = None, near: float = NEAR_THRESHOLD) -> None:
        self.state = Path(state)
        self.board = Board(self.state / "team.sqlite")
        self.cache = ResultCache(self.board, near)
        self.actors = {a.name: a for a in actors}
        self.log = log
        self.stats = {"dispatched": 0, "refused": 0, "misroutes": 0, "out_of_jurisdiction": 0, "off_goal": 0, "handoffs": 0,
                      "slot_calls": 0, "env_tokens": []}
        if goal_id:
            self.set_goal(goal_id)

    # goal record: exactly one active goal
    def set_goal(self, goal_id: str, title: str = "") -> None:
        self.board.meta_set("active_goal", json.dumps({"id": goal_id, "title": title, "t": time.time()}))

    def goal(self) -> Optional[str]:
        v = self.board.meta_get("active_goal")
        return json.loads(v)["id"] if v else None

    def _refuse(self, kind: Optional[str], msg: str) -> Refused:
        self.stats["refused"] += 1
        if kind:
            self.stats[kind] += 1
        if self.log:
            self.log("team", step="refused", outcome=msg[:80])
        return Refused(msg)

    def check(self, env: Envelope, actor_name: str) -> None:
        """All gates, before anything is sent."""
        try:
            env.validate()
        except Refused as e:
            raise self._refuse(None, str(e)) from None
        if env.goal_id != self.goal():
            raise self._refuse("off_goal", f"off-goal envelope {env.goal_id!r} (active {self.goal()!r})")
        for ref in env.inputs:
            try:
                input_kind(ref)
            except Refused as e:
                raise self._refuse(None, str(e)) from None
        a = self.actors.get(actor_name)
        if a is None:
            raise self._refuse(None, f"no actor {actor_name!r}")
        if env.step not in a.accepts:
            raise self._refuse("out_of_jurisdiction", f"{actor_name} does not accept {env.step}")
        if not env.check_size(ENVELOPE_MAX_TOKENS):
            raise self._refuse(None, f"envelope {env.tokens()} tokens, hard limit {ENVELOPE_MAX_TOKENS}")

    def dispatch(self, env: Envelope, actor_name: Optional[str] = None) -> str:
        """Send an envelope to its routed actor (or an explicit one, which must still be in jurisdiction)."""
        name = actor_name or route(env.step)
        hops = 0
        while True:
            self.check(env, name)
            if name != route(env.step) and hops == 0:
                self.stats["misroutes"] += 1                # explicit override that disagrees with the table
            a = self.actors[name]
            cached = self.cache.get(env, name) if a.is_slot else None
            t0 = time.perf_counter()
            if cached:
                out, how = cached
            else:
                out, how = a.fn(env, self), "run"
                if a.is_slot:
                    self.stats["slot_calls"] += 1
                    self.cache.put(env, out, name)
            self.stats["dispatched"] += 1
            self.stats["env_tokens"].append(env.tokens())
            if self.log:
                self.log(name, goal_id=env.goal_id, step=env.step, in_tok=env.tokens(), out_tok=est_tokens(out),
                         wall_s=time.perf_counter() - t0, cache_hit=how != "run", model=a.is_slot)
            m = _HANDOFF.search(out)
            if not m:
                return out
            hops += 1
            self.stats["handoffs"] += 1
            if hops > MAX_HOPS:
                raise self._refuse(None, "handoff loop")
            target = m.group(1).upper()
            if target not in self.actors:
                raise self._refuse(None, f"handoff to unknown role {target}")
            name = target                                   # re-route; jurisdiction is checked again at the top of the loop

    # ------------------------------------------------------------------------------------------------------------------ driver
    def run_task(self, task: dict[str, Any], ix: Any = None, steps: Sequence[str] = ("SPEC", "LOCATE", "PLAN", "CODE", "REVIEW",
                                                                                      "VALIDATE")) -> dict[str, Any]:
        """Run one task through the steps. Each step's output goes on the board; the next envelope carries only ids."""
        gid = self.goal() or "g0"
        refs: list[str] = []
        prior = ""
        out: dict[str, str] = {}
        task_id = self.board.put(gid, "task", json.dumps(task, sort_keys=True))
        refs.append(task_id)
        self._ix = ix
        for step in steps:
            lean = step in ("SPEC", "LOCATE", "PLAN")           # the task itself is on the board: ids, not text
            env = Envelope(gid, step, inputs=list(refs[-3:]),
                           constraints=task.get("constraints", [])[:3] if step == "CODE" else [],
                           success_test=task.get("success_test", "") if step in ("REVIEW", "VALIDATE") else "",
                           prior="" if lean else prior)
            res = self.dispatch(env)
            bid = self.board.put(gid, step.lower(), res)
            out[step] = res
            refs.append(bid)
            if step == "LOCATE":                             # ranges (not code) travel with later envelopes
                refs.extend(r for r in res.splitlines() if _RANGE.match(r))
                refs = refs[:1] + [bid] + refs[-2:] if len(refs) > 4 else refs
            first = res.strip().splitlines()[0][:100] if res.strip() else ""
            prior = f"{step.lower()}: {first}"
        return out


def locate_actor(ix: Any, k: int = 3) -> Actor:
    """Real code tool: the locate index. Query = the task text on the board; returns file:line ranges, one per line."""
    def fn(env: Envelope, team: "Team") -> str:
        q = ""
        for ref in env.inputs:
            if input_kind(ref) == "board":
                row = team.board.get(ref)
                if row and row["kind"] == "task":
                    q = json.loads(row["body"]).get("query", "")
        hits = ix.locate(q, k=k)
        return "\n".join(f"{h.path}:{h.line}-{h.end}" for h in hits) or "none"
    return actor("locate", fn)
