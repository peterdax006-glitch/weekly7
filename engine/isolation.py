"""Live/research separation and live safety audits (Bible PHASE 28; checklist L2-L5).

Canons served: C6 (regular hours only), paper-only broker, C11 (sealed windows stay sealed), and the rule that no
research experiment may change Live on its own.

Four layers, each independently testable:
  1. Static import audit  - AST scan of every module: research code may not import the broker order paths,
                            live code may not touch sealed windows. Catches lazy (in-function) imports too.
  2. Runtime guards       - context managers that make a violation raise instead of silently working.
  3. Firewall audits      - paper-only and trading-hours checks that drive the REAL broker classes with fakes,
                            plus an independent NYSE calendar (holidays, half days) to measure what
                            engine.broker.regular_hours does not screen.
  4. Activation / drift   - Live needs an explicit activation record; any change to live config after
                            activation (e.g. a research run rewriting config_overrides.json) is detected.
Nothing here uses the network or writes outside paths the caller passes in."""
from __future__ import annotations

import ast, builtins, contextlib, hashlib, json
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from . import config as K

ET = ZoneInfo("America/New_York")

# ------------------------------------------------------------------ 1. static import audit
LIVE_MODULES = {"live", "broker", "tick"}          # own broker order paths; tick schedules the live jobs
BROKER_ORDER_MODULES = {"broker"}                  # anything that can place an order
GUARD_MODULES = {"isolation"}                      # the guard itself must touch broker to disarm it
# Documented, reviewed couplings: research modules that lazily import engine.live for path/constant reads only.
# Any NEW entry here must be argued for; the audit fails on everything outside this map.
KNOWN_CONSTANT_COUPLINGS = {
    "improve": {"live"}, "scoring": {"live"}, "shadows": {"live"}, "site_data": {"live"}, "tick": {"live"},
}


def _imports(tree: ast.AST, pkg: str = "engine") -> list[tuple[str, int]]:
    """All imported engine-module short names with line numbers (top-level and nested)."""
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                p = a.name.split(".")
                if p[0] == pkg and len(p) > 1:
                    out.append((p[1], n.lineno))
        elif isinstance(n, ast.ImportFrom):
            if n.level == 1 and n.module:                       # from .broker import x
                out.append((n.module.split(".")[0], n.lineno))
            elif n.level == 1:                                  # from . import broker, config
                out.extend((a.name, n.lineno) for a in n.names)
            elif n.level == 0 and n.module and n.module.split(".")[0] == pkg:
                p = n.module.split(".")
                if len(p) > 1:
                    out.append((p[1], n.lineno))
                else:
                    out.extend((a.name, n.lineno) for a in n.names)
    return out


def _classify(name: str, source: str, live_modules, known) -> tuple[list[dict], list[dict]]:
    viol, known_hits = [], []
    imps = _imports(ast.parse(source))
    if name in live_modules:
        for m, ln in imps:
            if m == "livesim":
                viol.append({"kind": "live_reads_sealed", "module": name, "line": ln, "detail": "imports livesim"})
        for i, line in enumerate(source.splitlines(), 1):
            if "state/livesim" in line or "sealed_" in line or ("livesim" in line and "K.STATE" in line):
                viol.append({"kind": "live_reads_sealed", "module": name, "line": i, "detail": line.strip()[:80]})
        return viol, known_hits
    for m, ln in imps:
        if m in BROKER_ORDER_MODULES:
            viol.append({"kind": "research_imports_broker", "module": name, "line": ln, "detail": m})
        elif m in live_modules:
            if m in known.get(name, ()):
                known_hits.append({"module": name, "line": ln, "imports": m})
            else:
                viol.append({"kind": "research_imports_live", "module": name, "line": ln, "detail": m})
    return viol, known_hits


def audit_source_text(name: str, source: str, live_modules=LIVE_MODULES, known=None) -> list[dict]:
    """Rules applied to one source string (used to prove the audit catches a planted violation)."""
    return _classify(name, source, live_modules, known or {})[0]


def audit_imports(engine_dir: Path | None = None, live_modules=LIVE_MODULES, known=KNOWN_CONSTANT_COUPLINGS) -> dict:
    """Return {'violations', 'known_couplings', 'modules'}. Violation kinds:
      research_imports_broker  a non-live module imports a broker order module (never allowed, no allow-list)
      research_imports_live    a non-live module imports engine.live and is not in the reviewed `known` map
      live_reads_sealed        a live module imports livesim or mentions a sealed-window path."""
    d = Path(engine_dir or K.ROOT / "engine")
    viol, hits, n = [], [], 0
    for f in sorted(d.glob("*.py")):
        if f.stem == "__init__" or f.stem in GUARD_MODULES:
            continue
        n += 1
        v, h = _classify(f.stem, f.read_text(encoding="utf-8", errors="replace"), live_modules, known)
        viol += v
        hits += h
    return {"violations": viol, "known_couplings": hits, "modules": n}


# ------------------------------------------------------------------ 2. runtime guards
class IsolationError(RuntimeError):
    pass


@contextlib.contextmanager
def research_mode(broker_module=None, scrub_env: bool = True):
    """Inside this block any broker order (Local or Alpaca) or get_broker() call raises IsolationError, and
    (scrub_env) the ALPACA_* credentials are absent from the environment. Research code may simulate; it may
    never submit. Restores the originals on exit, even on error."""
    if broker_module is None:
        from . import broker as broker_module

    def _deny(*a, **k):
        raise IsolationError("research mode: broker access is forbidden")
    targets = [(broker_module, "get_broker")]
    for cls in ("LocalBroker", "AlpacaBroker"):
        c = getattr(broker_module, cls, None)
        if c is not None:
            targets += [(c, a) for a in ("order", "close_all") if hasattr(c, a)]
    saved = {}
    with (scrubbed_env() if scrub_env else contextlib.nullcontext()):
        try:
            for obj, attr in targets:
                saved[(obj, attr)] = getattr(obj, attr)
                setattr(obj, attr, _deny)
            yield
        finally:
            for (obj, attr), v in saved.items():
                setattr(obj, attr, v)


def is_sealed_path(p, sealed_dir: Path | None = None) -> bool:
    sd = Path(sealed_dir or K.STATE / "livesim").resolve()
    try:
        q = Path(p).resolve()
    except (OSError, TypeError, ValueError):
        return False
    return q == sd or sd in q.parents or (q.name.startswith("sealed_") and q.suffix == ".json")


@contextlib.contextmanager
def live_mode(sealed_dir: Path | None = None):
    """Inside this block opening any sealed-window file raises IsolationError (live can never read them)."""
    real_open, path_open = builtins.open, Path.open

    def guarded(file, *a, **k):
        if isinstance(file, (str, Path)) and is_sealed_path(file, sealed_dir):
            raise IsolationError(f"live mode: sealed window is unreadable: {Path(file).name}")
        return real_open(file, *a, **k)

    def p_open(self, *a, **k):
        if is_sealed_path(self, sealed_dir):
            raise IsolationError(f"live mode: sealed window is unreadable: {self.name}")
        return path_open(self, *a, **k)
    builtins.open, Path.open = guarded, p_open
    try:
        yield
    finally:
        builtins.open, Path.open = real_open, path_open


# ------------------------------------------------------------------ 3a. NYSE calendar (independent of broker.py)
def _easter(y: int) -> date:
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    mo = (h + l - 7 * m + 114) // 31
    return date(y, mo, ((h + l - 7 * m + 114) % 31) + 1)


def _nth_weekday(y, m, wd, n):
    d = date(y, m, 1)
    return d + timedelta(days=(wd - d.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(y, m, wd):
    d = date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - wd) % 7)


def _observed(d: date) -> date:
    if d.weekday() == 5:
        return d - timedelta(days=1)
    return d + timedelta(days=1) if d.weekday() == 6 else d


def nyse_holidays(y: int) -> set[date]:
    """Full-day NYSE closures for year y (Juneteenth from 2022). A Saturday New Year is not observed Dec 31."""
    h = {_nth_weekday(y, 1, 0, 3), _nth_weekday(y, 2, 0, 3), _easter(y) - timedelta(days=2),
         _last_weekday(y, 5, 0), _nth_weekday(y, 9, 0, 1), _nth_weekday(y, 11, 3, 4)}
    for d in [date(y, 7, 4), date(y, 12, 25)] + ([date(y, 6, 19)] if y >= 2022 else []):
        h.add(_observed(d))
    if date(y, 1, 1).weekday() != 5:
        h.add(_observed(date(y, 1, 1)))
    return {d for d in h if d.year == y}


def half_days(y: int) -> set[date]:
    """13:00 ET closes: day after Thanksgiving, Dec 24 and Jul 3 when they are ordinary weekdays."""
    out = {_nth_weekday(y, 11, 3, 4) + timedelta(days=1)}
    hol = nyse_holidays(y)
    for d in (date(y, 12, 24), date(y, 7, 3)):
        if d.weekday() < 5 and d not in hol and (d.month == 12 or date(y, 7, 4).weekday() < 5):
            out.add(d)
    return out


def strict_session(ts: datetime) -> bool:
    """Regular session with holidays and half days applied. ts must be tz-aware."""
    if ts.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    t = ts.astimezone(ET)
    d = t.date()
    if t.weekday() >= 5 or d in nyse_holidays(d.year):
        return False
    close = dtime(13, 0) if d in half_days(d.year) else dtime(16, 0)
    return dtime(9, 30) <= t.time() < close


# ------------------------------------------------------------------ 3b. firewall audits
def audit_hours_firewall(broker_module, timestamps: list[datetime]) -> dict:
    """Compare broker.regular_hours to the strict calendar on every timestamp.
    'unsafe' = broker allows while the strict calendar says closed (e.g. holidays, half-day afternoons);
    'over_blocked' = broker refuses an open moment (harmless, but measured)."""
    unsafe, over = [], []
    for ts in timestamps:
        b, s = bool(broker_module.regular_hours(ts)), strict_session(ts)
        if b and not s:
            unsafe.append(ts.isoformat())
        elif s and not b:
            over.append(ts.isoformat())
    return {"n": len(timestamps), "unsafe": unsafe, "over_blocked": over}


class RecordingClient:
    """Fake alpaca TradingClient: records submissions, never touches a network."""
    def __init__(self):
        self.submitted = []

    def submit_order(self, req):
        self.submitted.append(req)
        return type("O", (), {"id": "fake-1"})()

    def close_position(self, t):
        self.submitted.append(("close", t))


def probe_order(broker_obj, when_open: bool, broker_module, ticker="AAPL", qty=1.0, price=100.0) -> dict:
    """Call broker_obj.order() with regular_hours forced open/closed. Returns whether it raised OutsideHours and
    whether a fill/submission record came back. The caller supplies a throwaway broker."""
    orig = broker_module.regular_hours
    broker_module.regular_hours = lambda ts=None: when_open
    raised, rec = False, None
    try:
        try:
            rec = broker_obj.order(ticker, qty, price, "probe")
        except broker_module.OutsideHours:
            raised = True
    finally:
        broker_module.regular_hours = orig
    return {"raised": raised, "filled": rec is not None}


def audit_paper_only(source: str) -> dict:
    """Static paper-only check of broker source: every TradingClient(...) must pass paper=True literally, no live
    endpoint URL, and no `paper=` keyword anywhere that is not literal True."""
    problems, clients = [], 0
    for n in ast.walk(ast.parse(source)):
        if isinstance(n, ast.Call):
            fn = n.func
            nm = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if nm == "TradingClient":
                clients += 1
                v = {k.arg: k.value for k in n.keywords}.get("paper")
                if not (isinstance(v, ast.Constant) and v.value is True):
                    problems.append(f"line {n.lineno}: TradingClient without literal paper=True")
            for k in n.keywords:
                if k.arg == "paper" and not (isinstance(k.value, ast.Constant) and k.value.value is True):
                    problems.append(f"line {k.value.lineno}: paper flag is not literal True")
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            s = n.value.lower()
            if "api.alpaca.markets" in s and "paper-api" not in s:
                problems.append(f"line {n.lineno}: live endpoint string {n.value!r}")
    if clients == 0:
        problems.append("no TradingClient construction found (audit cannot vouch for anything)")
    return {"ok": not problems, "problems": list(dict.fromkeys(problems)), "clients": clients}


# ------------------------------------------------------------------ L4 broker safety (pre-trade checks)
def check_order(ticker: str, qty: float, price: float, equity: float, held_qty: float = 0.0,
                max_position_frac: float = 0.30, min_notional: float = 1.0) -> list[str]:
    """Reasons this order must be refused ([] = acceptable): no shorting, no crypto, sane price and quantity,
    minimum notional, and no position above max_position_frac of equity."""
    r = []
    t = (ticker or "").upper()
    if not t or not t.replace(".", "").replace("-", "").replace("/", "").isalnum():
        r.append("bad ticker")
    if "/" in t or t.endswith("-USD") or t.endswith("USDT"):
        r.append("crypto pair not allowed")
    if not (price and 0 < price < 1e6):
        r.append("bad price")
    if qty != qty or qty == 0:
        r.append("bad quantity")
    if not equity or equity <= 0:
        r.append("no equity")
    if r:
        return r
    if qty < 0 and -qty > held_qty + 1e-6:
        r.append("would open a short")
    if abs(qty) * price < min_notional:
        r.append("below minimum notional")
    if qty > 0 and (held_qty + qty) * price > max_position_frac * equity + 1e-6:
        r.append("position exceeds concentration cap")
    return r


class SafeBroker:
    """Wraps any broker; enforces check_order and a per-run buy budget before delegating. Used to prove the real
    planner (live.execute) never emits an unsafe order set."""
    def __init__(self, inner, equity: float, max_position_frac=0.30, gross_cap=1.05):
        self.inner, self.equity = inner, equity
        self.max_position_frac, self.gross_cap = max_position_frac, gross_cap
        self.name, self.refused, self.bought = inner.name, [], 0.0

    def __getattr__(self, k):
        return getattr(self.inner, k)

    def order(self, ticker, qty, price, reason, **kw):
        held = self.inner.positions().get(ticker, {}).get("qty", 0.0)
        why = check_order(ticker, qty, price, self.equity, held, self.max_position_frac)
        if qty > 0 and self.bought + qty * price > self.gross_cap * self.equity:
            why.append("run notional budget exceeded")
        if why:
            self.refused.append((ticker, qty, why))
            raise IsolationError(f"unsafe order refused {ticker} {qty:+.4f}: {'; '.join(why)}")
        if qty > 0:
            self.bought += qty * price
        return self.inner.order(ticker, qty, price, reason, **kw)


# ------------------------------------------------------------------ 4. activation and drift
LIVE_CONFIG_FILES = ("config_overrides.json",)


def _hash_files(root: Path, names) -> dict:
    return {n: (hashlib.sha256((Path(root) / n).read_bytes()).hexdigest() if (Path(root) / n).exists() else None)
            for n in names}


def write_activation(path: Path, state_dir: Path, activated_by: str, now: datetime, paper: bool = True,
                     files=LIVE_CONFIG_FILES) -> dict:
    """Explicit activation record: who, when, paper flag, and the hash of every live-config file at that moment."""
    if not paper:
        raise IsolationError("only paper activation is permitted")
    if not activated_by.strip():
        raise ValueError("activation needs a named activator")
    rec = {"activated_by": activated_by, "at": now.astimezone(timezone.utc).isoformat(timespec="seconds"),
           "paper": True, "config_hashes": _hash_files(state_dir, files)}
    Path(path).write_text(json.dumps(rec, indent=1))
    return rec


def verify_activation(path: Path, state_dir: Path) -> dict:
    """Live may start only if the record exists, is paper, names an activator, and live-config files still match
    their hashes. A missing or unreadable record is a failure, never a pass."""
    p = Path(path)
    if not p.exists():
        return {"ok": False, "problems": ["no activation record"]}
    try:
        rec = json.loads(p.read_text())
    except (ValueError, OSError):
        return {"ok": False, "problems": ["activation record unreadable"]}
    probs = []
    if rec.get("paper") is not True:
        probs.append("record is not paper")
    if not str(rec.get("activated_by", "")).strip():
        probs.append("no activator")
    hashes = rec.get("config_hashes")
    if not isinstance(hashes, dict) or not hashes:
        probs.append("no config hashes recorded")
    else:
        now_h = _hash_files(state_dir, hashes.keys())
        probs += [f"live config changed after activation: {k}" for k in hashes if now_h[k] != hashes[k]]
    return {"ok": not probs, "problems": probs}


# ------------------------------------------------------------------ 5. state-write, credential and preflight audits
LIVE_STATE_NAMES = ("config_overrides.json", "ledger.json", "positions_meta.json", "orders.jsonl",
                    "decisions.jsonl", "equity.json")
WRITE_VERBS = ("write_text", "write_bytes", "_w(", "jsave(", "json.dump", "to_parquet", "to_csv", "unlink(",
               ", 'w'", ', "w"', ", 'a'", ', "a"', "open(")
# Reviewed exception: engine.improve promotes a challenger into config_overrides.json only after the Bonferroni-
# corrected live z-test (Part M4). It is the single automatic path from research to Live and is listed so it cannot
# be joined by a second one unnoticed.
KNOWN_LIVE_WRITERS = {"improve": {"config_overrides.json"}}
CREDENTIAL_OK = {"broker", "tick", "live"}
_COMPOUND = (ast.FunctionDef, ast.ClassDef, ast.If, ast.For, ast.While, ast.Try, ast.With, ast.AsyncFunctionDef)


def audit_state_writes(engine_dir: Path | None = None, live_modules=LIVE_MODULES, known=KNOWN_LIVE_WRITERS) -> dict:
    """Find statements in NON-live modules that write to a live state file (statement mentions the file name and
    a write verb). Returns {'violations', 'known', 'reads'}; reads are listed for information only."""
    d = Path(engine_dir or K.ROOT / "engine")
    viol, kn, reads = [], [], []
    for f in sorted(d.glob("*.py")):
        if f.stem in live_modules or f.stem in GUARD_MODULES or f.stem == "__init__":
            continue
        tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
        for st in ast.walk(tree):
            if not isinstance(st, ast.stmt) or isinstance(st, _COMPOUND):
                continue
            txt = ast.unparse(st)
            for name in LIVE_STATE_NAMES:
                if name not in txt:
                    continue
                rec = {"module": f.stem, "line": st.lineno, "file": name}
                if any(v in txt for v in WRITE_VERBS):
                    (kn if name in known.get(f.stem, ()) else viol).append(rec)
                else:
                    reads.append(rec)
    return {"violations": viol, "known": kn, "reads": reads}


def audit_credentials(engine_dir: Path | None = None, ok=CREDENTIAL_OK) -> list[dict]:
    """Modules outside `ok` that read ALPACA_* environment variables (research must never hold broker keys)."""
    d = Path(engine_dir or K.ROOT / "engine")
    out = []
    for f in sorted(d.glob("*.py")):
        if f.stem in ok or f.stem in GUARD_MODULES:
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if "ALPACA_" in line and "environ" in line:
                out.append({"module": f.stem, "line": i, "detail": line.strip()[:80]})
    return out


def audit_paper_only_tree(root: Path | None = None) -> dict:
    """Run audit_paper_only over EVERY source file (engine and scripts) that constructs a TradingClient."""
    root = Path(root or K.ROOT)
    res = {}
    for pat in ("engine/*.py", "scripts/*.py"):
        for f in sorted(root.glob(pat)):
            src = f.read_text(encoding="utf-8", errors="replace")
            if "TradingClient(" in src and f.stem != "isolation":
                key = f.relative_to(root).as_posix()
                try:
                    res[key] = audit_paper_only(src)
                except SyntaxError as e:
                    res[key] = {"ok": False, "problems": [f"unparseable: {e}"], "clients": 0}
    return res


@contextlib.contextmanager
def scrubbed_env(keys=("ALPACA_KEY", "ALPACA_SECRET")):
    """Temporarily remove broker credentials from the environment, restoring them afterwards."""
    import os
    saved = {k: os.environ.pop(k) for k in keys if k in os.environ}
    try:
        yield
    finally:
        os.environ.update(saved)


def previous_session(d: date) -> date:
    d = d - timedelta(days=1)
    while d.weekday() >= 5 or d in nyse_holidays(d.year):
        d -= timedelta(days=1)
    return d


def data_is_current(last_bar: date, now: datetime, max_lag_sessions: int = 1) -> bool:
    """Live uses current data only: the latest bar must be from the current session (or, on a closed day, the last
    one) or at most `max_lag_sessions` sessions earlier, and never dated in the future. now must be tz-aware."""
    n = now.astimezone(ET).date()
    d = previous_session(n + timedelta(days=1)) if (n.weekday() >= 5 or n in nyse_holidays(n.year)) else n
    allowed, e = {d}, d
    for _ in range(max_lag_sessions):
        e = previous_session(e)
        allowed.add(e)
    return last_bar in allowed and last_bar <= n


def live_preflight(now: datetime, activation_path: Path, state_dir: Path, last_bar: date, orders: list[tuple],
                   equity: float, held: dict[str, float], max_position_frac: float = 0.30) -> dict:
    """One decision for 'may Live trade right now?' composing every firewall: strict session (holidays, half days),
    verified paper-only activation with unchanged config, current data, and check_order on each proposed order
    (ticker, qty, price). Returns {'go', 'reasons', 'refused_orders'}; anything unknown is a no-go."""
    why, refused = [], []
    if not strict_session(now):
        why.append("outside regular session (calendar-aware)")
    why += verify_activation(activation_path, state_dir)["problems"]
    if not data_is_current(last_bar, now):
        why.append(f"stale data: last bar {last_bar}")
    for t, q, p in orders:
        r = check_order(t, q, p, equity, held.get(t, 0.0), max_position_frac)
        if r:
            refused.append({"ticker": t, "qty": q, "reasons": r})
    if refused:
        why.append(f"{len(refused)} order(s) fail broker safety")
    return {"go": not why, "reasons": why, "refused_orders": refused}


def full_audit(root: Path | None = None) -> dict:
    """Every static audit in one dict (used by scripts/data_live_audit.py)."""
    root = Path(root or K.ROOT)
    imp = audit_imports(root / "engine")
    sw = audit_state_writes(root / "engine")
    cred = audit_credentials(root / "engine")
    paper = audit_paper_only_tree(root)
    ok = not imp["violations"] and not sw["violations"] and not cred and all(v["ok"] for v in paper.values())
    return {"ok": ok, "imports": imp, "state_writes": sw, "credentials": cred, "paper_only": paper}
