"""Future-leak audit of the blind Test path (Bible PHASES 1, 2, 21, 26; canon C56 "only information available live at
that moment" and C55 "the rerun of a year must be disguised").

Every function answers one question about one CHANNEL through which a blind run could know the future or recognise the
year, and returns numbers, never a verdict typed by hand. `Audit` collects one `Channel` record per channel (status
LEAK / CLEAN / FIXED / QUARANTINED, evidence, the test that fails when the leak exists, the fix or the exact hook).

Channels (numbering follows queue/B23_future_leak_audit.md):
  1  survivorship      attrition_profile, delisting_hazard, survivorship_haircut, inject_dead_names, pit_universe
  2  adjusted prices   price_level_drift, cum_future_split_factor, reconstruct_as_traded, level_rule_disagreement,
                       split_invariant_tradable
  3  today's metadata  crypto_exposure, coarsen_sic
  4  learned state     BasisLineage, simulate_window_draws, basis_future_share (loop2 trains its basis on every archived
                       window, including windows whose REAL dates lie after the window being played)
  5  macro vintages    macro_revision_risk, pit_macro
  6  year fingerprints daily_fingerprint_series, window_features, FingerprintProbe, regular_grid_index
  7  network           NetworkGuard, poison, network_markers
  8  other             import_closure, data_access, truncation_invariance, feed_exposure, HardenedFeed (opt-in fixed feed)

Nothing here reads state/livesim (sealed windows), writes to data/cache, uses a clock, or draws without a seed.
Python cannot stop deliberate reflection; the guards stop honest mistakes and the tests prove they catch planted ones."""
from __future__ import annotations

import ast
import json
import socket
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as K

LEAK, CLEAN, FIXED, QUARANTINED = "LEAK", "CLEAN", "FIXED", "QUARANTINED"
STATUSES = (LEAK, CLEAN, FIXED, QUARANTINED)


# =====================================================================================================================
# Report objects
# =====================================================================================================================
@dataclass
class Channel:
    key: str
    name: str
    status: str
    evidence: dict = field(default_factory=dict)
    test: str = ""
    fix: str = ""
    hook: str = ""
    measured_on: str = "real"

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, not {self.status!r}")


class Audit:
    """Ordered collection of channel records with a markdown + json writer that stamps provenance."""

    def __init__(self):
        self.channels: dict[str, Channel] = {}

    def add(self, ch: Channel):
        self.channels[ch.key] = ch
        return ch

    def counts(self) -> dict:
        out = {s: 0 for s in STATUSES}
        for c in self.channels.values():
            out[c.status] += 1
        return out

    def open_leaks(self) -> list[str]:
        """Channels still leaking with neither a fix nor a quarantine (the number that must reach zero)."""
        return [k for k, c in self.channels.items() if c.status == LEAK]

    def summary(self, stamp=None) -> dict:
        return {"counts": self.counts(), "open_leaks": self.open_leaks(), "provenance": stamp,
                "channels": {k: _jsonable(asdict(c)) for k, c in self.channels.items()}}

    def markdown(self, stamp=None) -> str:
        lines = ["# Future-leak audit (canon C56 / C55)", ""]
        if stamp:
            lines += [f"code {stamp.get('code_hash')} | git {stamp.get('git_commit')} | seed {stamp.get('seed')}", ""]
        c = self.counts()
        lines += [f"**{len(self.channels)} channels: " + ", ".join(f"{c[s]} {s}" for s in STATUSES) + "**", ""]
        lines += ["LEAK = present in the current default blind path (a tested fix or hook exists where stated); FIXED = closed in the default path; "
                  "QUARANTINED = cannot be removed from the data, measured, results must carry the stated rule; CLEAN = nothing found, tested.", ""]
        lines += ["| # | channel | status | test |", "|---|---|---|---|"]
        for k, ch in self.channels.items():
            lines.append(f"| {k} | {ch.name} | {ch.status} | {ch.test or '-'} |")
        for k, ch in self.channels.items():
            lines += ["", f"## {k}. {ch.name} - {ch.status} (measured on {ch.measured_on})", ""]
            for ek, ev in ch.evidence.items():
                lines.append(f"- {ek}: {_short(ev)}")
            if ch.fix:
                lines.append(f"- fix: {ch.fix}")
            if ch.hook:
                lines.append(f"- HOOK: {ch.hook}")
        return "\n".join(lines) + "\n"

    def save(self, out_dir, stamp=None):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(self.summary(stamp), indent=1, default=str))
        (out / "report.md").write_text(self.markdown(stamp), encoding="utf-8")
        return out


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(o).date())
    if isinstance(o, pd.Series):
        return _jsonable(o.to_dict())
    if isinstance(o, pd.DataFrame):
        return _jsonable(o.to_dict("index"))
    return o


def _short(v, n=1500):
    s = json.dumps(_jsonable(v), default=str) if isinstance(v, (dict, list, tuple, pd.Series, pd.DataFrame)) else str(v)
    return s if len(s) <= n else s[:n] + " ..."


# =====================================================================================================================
# 7. Network: a blind worker has none
# =====================================================================================================================
class NetworkBlocked(RuntimeError):
    """A blind worker tried to reach the network (data refresh, EDGAR, yfinance, a socket)."""


_LOOPBACK = {"127.0.0.1", "::1", "localhost", "0.0.0.0"}


class NetworkGuard:
    """Makes every socket connection and DNS lookup raise NetworkBlocked. It patches `socket` itself, so requests, urllib,
    yfinance and anything else built on sockets is covered without naming them. `allow_loopback` exists for tests that
    need a local server. Use as a context manager (tests) or `install()` once at the top of a worker (never undone)."""

    def __init__(self, allow_loopback: bool = False):
        self.allow_loopback = allow_loopback
        self.blocked: list[tuple[str, str]] = []
        self._orig = None

    def _host_ok(self, host) -> bool:
        return self.allow_loopback and str(host) in _LOOPBACK

    def _refuse(self, what, target):
        self.blocked.append((what, str(target)))
        raise NetworkBlocked(f"blind worker attempted network access: {what} {target}")

    def install(self):
        if self._orig is not None:
            return self
        g = self
        o_connect, o_ex, o_gai = socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo
        self._orig = (o_connect, o_ex, o_gai)

        def connect(sock, address):
            if isinstance(address, tuple) and not g._host_ok(address[0]):
                g._refuse("connect", address)
            return o_connect(sock, address)

        def connect_ex(sock, address):
            if isinstance(address, tuple) and not g._host_ok(address[0]):
                g._refuse("connect_ex", address)
            return o_ex(sock, address)

        def getaddrinfo(host, *a, **k):
            if not g._host_ok(host):
                g._refuse("getaddrinfo", host)
            return o_gai(host, *a, **k)

        socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo = connect, connect_ex, getaddrinfo
        return self

    def uninstall(self):
        if self._orig is not None:
            socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo = self._orig
            self._orig = None

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.uninstall()
        return False


def poison(obj, names, reason="network is closed to blind workers"):
    """Replace attributes of `obj` (usually a module such as engine.data) with functions that raise NetworkBlocked, so a
    refresh call fails at the call site with a clear message even before it reaches a socket. Returns a restore()."""
    saved = {}
    for n in names:
        if hasattr(obj, n):
            saved[n] = getattr(obj, n)

            def _raise(*a, _n=n, **k):
                raise NetworkBlocked(f"{getattr(obj, '__name__', obj)}.{_n}: {reason}")
            setattr(obj, n, _raise)

    def restore():
        for n, f in saved.items():
            setattr(obj, n, f)
    return restore


NETWORK_MODULES = {"requests", "yfinance", "urllib", "urllib3", "http", "socket", "ftplib", "smtplib", "aiohttp",
                   "websockets", "alpaca", "httpx"}
NETWORK_CALLS = {"download", "update", "refresh", "refresh_sources", "fetch_yfinance", "urlopen"}


def network_markers(path) -> list[dict]:
    """Static: imports of network libraries and calls to refresh-style functions in one source file."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] in NETWORK_MODULES:
                    out.append({"line": n.lineno, "kind": "import", "what": a.name})
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0 and n.module.split(".")[0] in NETWORK_MODULES:
            out.append({"line": n.lineno, "kind": "import", "what": n.module})
        elif isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None
            if name in NETWORK_CALLS:
                out.append({"line": n.lineno, "kind": "call", "what": name})
    return sorted(out, key=lambda d: d["line"])


# =====================================================================================================================
# 8. What the blind path can reach: import closure and file reads
# =====================================================================================================================
def _engine_imports(tree: ast.AST) -> set[str]:
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            if n.level == 1:
                out.update([n.module.split(".")[0]] if n.module else [a.name for a in n.names])
            elif n.level == 0 and n.module:
                p = n.module.split(".")
                if p[0] == "engine":
                    out.update([p[1]] if len(p) > 1 else [a.name for a in n.names])
        elif isinstance(n, ast.Import):
            for a in n.names:
                p = a.name.split(".")
                if p[0] == "engine" and len(p) > 1:
                    out.add(p[1])
    return out


def import_closure(entries, engine_dir=None) -> dict[str, set[str]]:
    """Engine modules reachable from the entry files (paths), following top-level AND lazy in-function imports.
    Returns {module: set(direct engine imports)}. Entry files outside engine/ are keyed by their stem."""
    engine_dir = Path(engine_dir) if engine_dir else K.ROOT / "engine"
    have = {p.stem for p in engine_dir.glob("*.py")}
    todo = [Path(e) for e in entries]
    seen: dict[str, set[str]] = {}
    while todo:
        p = todo.pop()
        if p.stem in seen or not p.exists():
            continue
        deps = {d for d in _engine_imports(ast.parse(p.read_text(encoding="utf-8"))) if d in have}
        seen[p.stem] = deps
        todo += [engine_dir / f"{d}.py" for d in deps if d not in seen]
    return seen


READERS = {"read_parquet", "read_csv", "read_json", "read_pickle", "read_text", "read_bytes", "open", "load", "loads",
           "read_feather", "read_excel"}


def data_access(path) -> list[dict]:
    """Static: every file-read call in a module with its enclosing function and source text (the argument shows which
    cache or state file). This is the inventory of files a blind run can pull information from."""
    src = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    out = []

    def walk(node, fn):
        for ch in ast.iter_child_nodes(node):
            f = ch.name if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)) else fn
            if isinstance(ch, ast.Call):
                c = ch.func
                nm = c.attr if isinstance(c, ast.Attribute) else c.id if isinstance(c, ast.Name) else None
                if nm in READERS:
                    txt = ast.unparse(ch)
                    if nm in ("load", "loads") and not any(t in txt for t in ("json", "np.", "pickle", "pd.")):
                        pass
                    else:
                        out.append({"line": ch.lineno, "function": fn or "<module>", "call": txt[:140]})
            walk(ch, f)
    walk(tree, None)
    return out


class FileAccessRecorder:
    """Runtime inventory of every file a piece of code OPENS (Python audit hook, event `open`). Static reachability over-
    approximates (a lazily imported module may never run); this records what actually happened during a run. The hook stays
    installed for the life of the process but records only inside a `with` block."""
    _installed = False
    _active: list["FileAccessRecorder"] = []

    def __init__(self):
        self.paths: set[str] = set()

    @classmethod
    def _hook(cls, event, args):
        if event == "open" and cls._active:
            p = args[0]
            if isinstance(p, bytes):
                p = p.decode(errors="ignore")
            if isinstance(p, (str, Path)):
                for r in cls._active:
                    r.paths.add(str(p))

    def __enter__(self):
        import sys
        if not FileAccessRecorder._installed:
            sys.addaudithook(FileAccessRecorder._hook)
            FileAccessRecorder._installed = True
        FileAccessRecorder._active.append(self)
        return self

    def __exit__(self, *exc):
        FileAccessRecorder._active.remove(self)
        return False

    def project_files(self, root=None) -> dict[str, list[str]]:
        """Opened files under the repo, grouped: code (.py), data/cache, state, other. Interpreter and site-packages files,
        __pycache__ and the virtualenv are dropped."""
        root = Path(root or K.ROOT).resolve()
        out = {"code": [], "data_cache": [], "state": [], "other": []}
        for p in sorted(self.paths):
            try:
                rp = Path(p).resolve()
                rel = rp.relative_to(root)
            except (ValueError, OSError):
                continue
            parts = rel.parts
            if parts[0] in (".venv", "venv") or "__pycache__" in parts or "site-packages" in parts:
                continue
            key = "code" if rp.suffix == ".py" else "data_cache" if parts[:2] == ("data", "cache") else "state" if parts[0] == "state" else "other"
            out[key].append(rel.as_posix())
        return out


# modules that own the real data on the FEED side: reading caches there is their job
FEED_SIDE = {"livesim", "replay", "data", "blind_gates", "health", "provenance", "improve", "config", "registry", "resources",
             "checkpoint", "basis_search", "objective", "isolation", "leak_audit"}


def trader_side_reads(closure: dict[str, set[str]], engine_dir=None) -> dict[str, list[dict]]:
    """File reads inside modules the TRADER runs (everything reachable that is not feed-side). Each is a channel to
    review: the trader must get its information only through what the feed serves."""
    engine_dir = Path(engine_dir) if engine_dir else K.ROOT / "engine"
    out = {}
    for m in sorted(closure):
        if m in FEED_SIDE or not (engine_dir / f"{m}.py").exists():
            continue
        r = data_access(engine_dir / f"{m}.py")
        if r:
            out[m] = r
    return out


# =====================================================================================================================
# 1. Survivorship
# =====================================================================================================================
def attrition_profile(close: pd.DataFrame, early_sessions: int = 20) -> pd.DataFrame:
    """Per calendar year of a wide close panel: names with a price that year, first listings, and TERMINAL exits (last
    price more than `early_sessions` sessions before the panel end). A survivor panel shows ~0 exits in every year."""
    if close.empty:
        return pd.DataFrame(columns=["n_alive", "n_first", "n_exit", "exit_rate"])
    first, last = close.apply(lambda s: s.first_valid_index()), close.apply(lambda s: s.last_valid_index())
    end = close.index[max(0, len(close) - 1 - early_sessions)]
    yrs = sorted(set(close.index.year))
    alive = close.notna().groupby(close.index.year).any().sum(axis=1)
    fy = first.dropna().dt.year.value_counts()
    ex = last[last < end].dropna().dt.year.value_counts()
    df = pd.DataFrame({"n_alive": alive.reindex(yrs).fillna(0).astype(int), "n_first": fy.reindex(yrs).fillna(0).astype(int),
                       "n_exit": ex.reindex(yrs).fillna(0).astype(int)})
    df["exit_rate"] = df["n_exit"] / df["n_alive"].replace(0, np.nan)
    return df


def delisting_hazard(events: pd.DataFrame, alive_by_year: pd.Series, date_col="delist_date", terminal_col="terminal") -> pd.DataFrame:
    """Terminal delistings per year from a registry / SEC event table over names alive that year. The event table holds
    every terminal filing (M&A, going private, non-exchange filers), so the rate is an UPPER bound on exchange
    delistings for cause; `literature_low/high` bracket it (CRSP-type 3-6% a year)."""
    if events is None or not len(events):
        return pd.DataFrame({"events": 0, "alive": alive_by_year, "rate": 0.0})
    e = events[events[terminal_col].astype(bool)] if terminal_col in events else events
    yr = pd.to_datetime(e[date_col]).dt.year.value_counts().sort_index()
    idx = alive_by_year.index
    df = pd.DataFrame({"events": yr.reindex(idx).fillna(0).astype(int), "alive": alive_by_year})
    df["rate"] = df["events"] / df["alive"].replace(0, np.nan)
    return df


LITERATURE_HAZARD = (0.03, 0.06)         # annual share of listed names that leave for cause / by acquisition (CRSP-style)
TERMINAL_LOSS = {"bankruptcy": -0.85, "performance": -0.45, "other": -0.30, "acquisition": 0.20}   # Shumway (1997) style


def survivorship_haircut(annual_hazard: float, terminal_loss: float = -0.45, concentration: float = 1.0, weeks: int = 52) -> dict:
    """Expected weekly return NOT earned by a survivor-only panel: a held name leaves with probability hazard/weeks per
    week and loses `terminal_loss` on exit. `concentration` = how much likelier the names a high-volatility rule holds
    are to leave than the average name (delisting concentrates in cheap, volatile stocks)."""
    if not (0 <= annual_hazard <= 1):
        raise ValueError("annual_hazard must be a share between 0 and 1")
    p_week = 1 - (1 - min(1.0, annual_hazard * concentration)) ** (1.0 / weeks)
    drag = p_week * terminal_loss
    return {"p_exit_week": p_week, "weekly_drag": drag, "annual_drag": (1 + drag) ** weeks - 1}


def concentration_from_registry(registry: pd.DataFrame, alive_last_close: pd.Series, floor: float = 3.0) -> dict:
    """How much likelier delisted names are to be cheap: share of dead names with last close below `floor` versus the share
    of live names below it. Only registry rows that carry a last_close count."""
    lc = pd.to_numeric(registry.get("last_close", pd.Series(dtype=float)), errors="coerce").dropna()
    lc = lc[lc > 0]
    live = alive_last_close.dropna()
    if not len(lc) or not len(live):
        return {"n_dead_priced": int(len(lc)), "ratio": float("nan")}
    dead_share, live_share = float((lc < floor).mean()), float((live < floor).mean())
    return {"n_dead_priced": int(len(lc)), "dead_below_floor": dead_share, "live_below_floor": live_share,
            "ratio": dead_share / live_share if live_share > 0 else float("inf")}


def inject_dead_names(close: pd.DataFrame, delist_dates, seed: int, terminal_mean=-0.45, terminal_sd=0.30,
                      high_vol_bias: float = 0.7, prefix="DEAD") -> pd.DataFrame:
    """Add synthetic names that vanish on the given dates - the survivorship correction a panel of survivors lacks.
    Each dead name copies the price path of a donor alive on that date (drawn from the top volatility third with
    probability `high_vol_bias`), scaled by a seeded level, up to the delisting date; the day after it prints one terminal
    return drawn from N(terminal_mean, terminal_sd) truncated at -100%, then NaN. An ASSUMPTION model, used to size the
    bias - it does not recover the real dead names' true paths."""
    rng = np.random.default_rng(seed)
    r = np.log(close / close.shift(1))
    vol = r.rolling(60, min_periods=30).std()
    cols, made = {}, 0
    idx = close.index
    for i, d in enumerate(pd.to_datetime(list(delist_dates))):
        k = idx.searchsorted(d)
        if k <= 60 or k >= len(idx) - 1:
            continue
        alive = close.iloc[k].dropna().index
        if not len(alive):
            continue
        v = vol.iloc[k].reindex(alive).dropna()
        pool = v.index[v >= v.quantile(2 / 3)] if len(v) >= 6 and rng.random() < high_vol_bias else alive
        donor = pool[int(rng.integers(len(pool)))]
        path = close[donor].copy() * float(np.exp(rng.normal(0, 0.5)))
        term = float(np.clip(rng.normal(terminal_mean, terminal_sd), -1.0, 5.0))
        path.iloc[k + 1:] = np.nan
        if term > -1.0:
            path.iloc[k + 1] = path.iloc[k] * (1 + term)
        cols[f"{prefix}{i:04d}"] = path
        made += 1
    if not cols:
        return close.copy()
    return pd.concat([close, pd.DataFrame(cols, index=idx)], axis=1)


def vol_basket_weekly(close: pd.DataFrame, top_frac: float = 0.1, min_price: float = 0.0, lookback: int = 60,
                      step: int = 5, entry_lag: int = 1, clip=(-0.9, 2.0)) -> pd.Series:
    """Equal-weight weekly return of the top-volatility slice (the kind of basket a 7%-a-week rule holds): rank by trailing
    volatility at the close of t, buy at the close of t+entry_lag (a stand-in for the next open), sell `step` sessions
    later; a name with no price at the exit earns its last valid return (terminal print if any). Each name-week return is
    clipped to `clip`: penny-stock data glitches otherwise swamp the mean of a volatile slice (raw weekly sd was 140%). Used to
    compare a survivor panel with the same panel plus dead names."""
    r = np.log(close / close.shift(1))
    vol = r.rolling(lookback, min_periods=lookback // 2).std()
    out = {}
    for t in range(lookback, len(close) - step - entry_lag, step):
        v = vol.iloc[t]
        ok = v.notna() & (close.iloc[t] >= min_price) & close.iloc[t + entry_lag].notna()
        v = v[ok]
        if len(v) < 10:
            continue
        pick = v.nlargest(max(3, int(len(v) * top_frac))).index
        p0 = close.iloc[t + entry_lag][pick]
        seg = close.iloc[t + entry_lag: t + entry_lag + step + 1][pick]
        p1 = seg.ffill().iloc[-1]
        out[close.index[t]] = float((p1 / p0 - 1).clip(*clip).mean())
    return pd.Series(out, dtype=float)


def pit_universe(close: pd.DataFrame, as_of, registry: pd.DataFrame | None = None, recent: int = 5) -> list[str]:
    """Names investable at `as_of`: a price within the last `recent` sessions up to as_of, first price on or before as_of, and
    no delisting ANNOUNCED on or before as_of that has already taken effect. Later listings and later delistings never
    enter or leave early."""
    as_of = pd.Timestamp(as_of)
    c = close.loc[:as_of]
    if c.empty:
        return []
    live = c.iloc[-recent:].notna().any()
    names = set(live[live].index)
    if registry is not None and len(registry):
        gone = registry[pd.to_datetime(registry["announced"]) <= as_of]
        gone = gone[pd.to_datetime(gone["delist_date"]) <= as_of]
        names -= set(gone["ticker"].astype(str))
    return sorted(names)


# =====================================================================================================================
# 2. Split / dividend adjusted prices
# =====================================================================================================================
def price_level_drift(close: pd.DataFrame, floors=(3.0, 10.0)) -> pd.DataFrame:
    """Per year: median close over ticker-days and the share of ticker-days below each floor. On as-traded prices these are
    roughly flat over decades; on prices back-adjusted for later splits and dividends the OLD years sit far lower, which
    is the direct trace of future corporate actions in every absolute price level."""
    g = close.groupby(close.index.year)
    med = g.apply(lambda d: float(np.nanmedian(d.to_numpy(dtype="float64"))) if d.notna().any().any() else np.nan)
    out = pd.DataFrame({"median_close": med})
    for f in floors:
        out[f"share_below_{f:g}"] = g.apply(lambda d, f=f: float((d < f).sum().sum() / max(1, d.notna().sum().sum())))
    return out


def cum_future_split_factor(splits: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """as_traded = adjusted * factor. factor(d) = product of the ratios of every split dated AFTER d (a 4:1 split is 4.0).
    `splits` is indexed by split date. Splits on or before the first date leave factor 1 there."""
    f = pd.Series(1.0, index=index)
    for d, ratio in splits.items():
        if ratio and np.isfinite(ratio) and ratio > 0:
            f[f.index < pd.Timestamp(d)] *= float(ratio)
    return f


def reconstruct_as_traded(adj_close: pd.Series, splits: pd.Series, div_factor: pd.Series | None = None) -> pd.Series:
    """Prices as printed on the day: undo the later splits and (if given) the cumulative dividend adjustment
    (`div_factor` = adjusted / split-only-adjusted, from a vendor's Adj Close and Close)."""
    p = adj_close * cum_future_split_factor(splits, adj_close.index)
    return p / div_factor.reindex(p.index).ffill().bfill() if div_factor is not None else p


def level_rule_disagreement(adj: pd.DataFrame, traded: pd.DataFrame, floor: float = 3.0, rank_q: float = 0.2) -> dict:
    """Where an absolute-level rule decides differently on adjusted vs as-traded prices, over the shared cells: the price
    floor (`floor`) and the cross-sectional price-rank filter (bottom `rank_q` excluded, as features.build does)."""
    c = adj.columns.intersection(traded.columns)
    a, t = adj[c], traded[c].reindex(adj.index)
    both = a.notna() & t.notna()
    n = int(both.sum().sum())
    if n == 0:
        return {"n_cells": 0}
    fl = ((a >= floor) != (t >= floor)) & both
    ra, rt = a.rank(axis=1, pct=True), t.rank(axis=1, pct=True)
    rk = ((ra >= rank_q) != (rt >= rank_q)) & both
    return {"n_cells": n, "floor_flip_share": float(fl.sum().sum() / n), "rank_filter_flip_share": float(rk.sum().sum() / n),
            "mean_abs_rank_shift": float(((ra - rt).abs()).where(both).stack().mean()),
            "adj_below_floor_but_traded_above": float((((a < floor) & (t >= floor)) & both).sum().sum() / n),
            "adj_above_floor_but_traded_below": float((((a >= floor) & (t < floor)) & both).sum().sum() / n)}


def split_invariant_tradable(C: pd.DataFrame, V: pd.DataFrame, dv_q: float = 0.4, window: int = 20) -> pd.DataFrame:
    """Relative tradability without any price level: cross-sectional rank of the 20-day median DOLLAR volume >= dv_q. Split
    adjustment scales price and volume in opposite directions, so price x volume - and this rank - does not move when a
    later split is added to the history (features.build's price-rank clause does)."""
    dv = (C * V).rolling(window, min_periods=max(1, int(window * 0.75))).median()
    return (dv.rank(axis=1, pct=True) >= dv_q) & C.notna()


# =====================================================================================================================
# 3. Today's metadata
# =====================================================================================================================
def crypto_exposure(panel_first_valid: pd.Series, panel_last_valid: pd.Series, universe_names: pd.Series | None = None) -> pd.DataFrame:
    """Tickers the current crypto rules single out (name regex or hard list) with the dates they traded. A name filtered
    by TODAY's business (Strategy/MSTR was a software company 1998-2019) is removed for every year, including years
    when it was not crypto: the filter knows the future. `universe_names` is ticker -> security name."""
    from .universe import CRYPTO_NAME, CRYPTO_TICKERS
    rows = []
    for t in panel_first_valid.index:
        nm = "" if universe_names is None else str(universe_names.get(t, ""))
        by_list, by_name = t in CRYPTO_TICKERS, bool(CRYPTO_NAME.search(nm))
        if by_list or by_name:
            rows.append({"ticker": t, "first": panel_first_valid[t], "last": panel_last_valid.get(t), "by_list": by_list, "by_name": by_name})
    return pd.DataFrame(rows, columns=["ticker", "first", "last", "by_list", "by_name"])


def coarsen_sic(sic: pd.DataFrame, digits: int = 1) -> pd.DataFrame:
    """Neutral industry map: keep only the first `digits` of the SIC code (division level for 1). A company that changed
    business after the window keeps the same broad bucket far more often than the same 4-digit code."""
    out = sic.copy()
    out["sic"] = out["sic"].astype(str).str[:digits].str.ljust(4, "0")
    return out


# =====================================================================================================================
# 4. Learned state: basis lineage
# =====================================================================================================================
@dataclass
class TrainedOn:
    window_id: str
    real_start: pd.Timestamp
    real_end: pd.Timestamp


def simulate_window_draws(n_rounds: int, par: int = 3, seed: int = 0) -> list[list[dict]]:
    """Play the loop's draw process (blind_gates.seal_window: random start month 1965-2025, 12 months, overlap allowed up
    to half a year) for `n_rounds` rounds of `par` windows. Pure simulation with seeds - it never reads state/livesim.
    Returns rounds of {id, start, end}."""
    from . import blind_gates as BG
    rng = np.random.default_rng(seed)
    used, rounds = [], []
    for r in range(n_rounds):
        rnd = []
        for j in range(par):
            rec = BG.seal_window(used, int(rng.integers(2 ** 62)), "2026-01-01", tag=f"sim{r}{j}")
            st = pd.Timestamp(rec["start"])
            used.append(st)
            rnd.append({"id": f"w{r:02d}{'abc'[j % 3]}", "start": st, "end": BG.window_end(st)})
        rounds.append(rnd)
    return rounds


def basis_future_share(rounds: list[list[dict]], allow_same_window: bool = False) -> dict:
    """The measure of channel 4 as loop2 runs today: window W of round r is played with the basis trained on windows of
    rounds < r. A trained-on window is 'not past' only if it ENDED before W's real start. Returns, over every W with at
    least one trained-on window: the mean share of its basis training set that lies in W's real future or overlaps it, and
    the share of W whose basis touched at least one such window."""
    shares, touched, n = [], 0, 0
    for r, rnd in enumerate(rounds):
        prior = [w for q in rounds[:r] for w in q]
        if not prior:
            continue
        for w in rnd:
            bad = [p for p in prior if p["end"] >= w["start"]]
            shares.append(len(bad) / len(prior))
            touched += bool(bad)
            n += 1
    if not n:
        return {"n_played": 0, "mean_future_share": float("nan"), "share_of_windows_touched": float("nan")}
    return {"n_played": n, "mean_future_share": float(np.mean(shares)), "share_of_windows_touched": touched / n,
            "max_future_share": float(np.max(shares))}


class BasisLineage:
    """Versions of the training basis (cfg + meta) and the windows each was trained on. `basis_for(real_start)` returns the
    newest version whose training windows ALL ended before `real_start` (and, by default, none is the same real window),
    else None - the caller then uses the untrained defaults. The referee (the loop, which owns the seals) calls it; the
    trader never sees a real date."""

    def __init__(self):
        self.versions: list[dict] = []

    def register(self, version, cfg, meta, trained_on: list[TrainedOn]):
        self.versions.append({"version": version, "cfg": cfg, "meta": meta, "trained_on": list(trained_on)})

    def eligible(self, rec, real_start, allow_same_window=False) -> bool:
        rs = pd.Timestamp(real_start)
        for w in rec["trained_on"]:
            if pd.Timestamp(w.real_end) >= rs:
                if allow_same_window and pd.Timestamp(w.real_start) == rs:
                    continue
                return False
        return True

    def basis_for(self, real_start, allow_same_window=False):
        for rec in reversed(self.versions):
            if self.eligible(rec, real_start, allow_same_window):
                return rec
        return None

    def violations(self, played: list[dict], allow_same_window=False) -> list[dict]:
        """Audit of a finished run: `played` = [{id, real_start, version}] - list every window whose basis was trained on a
        window that had not ended by its start."""
        by_v = {r["version"]: r for r in self.versions}
        out = []
        for p in played:
            rec = by_v.get(p["version"])
            if rec is None:
                continue
            late = [w.window_id for w in rec["trained_on"] if pd.Timestamp(w.real_end) >= pd.Timestamp(p["real_start"])
                    and not (allow_same_window and pd.Timestamp(w.real_start) == pd.Timestamp(p["real_start"]))]
            if late:
                out.append({"window": p["id"], "version": p["version"], "trained_on_late": late})
        return out


def strict_training_windows(windows: list[dict], target_start, key="end") -> list[dict]:
    """Windows a basis search may use for a target that starts at `target_start`: those that ended strictly before it."""
    ts = pd.Timestamp(target_start)
    return [w for w in windows if pd.Timestamp(w[key]) < ts]


def defaults_contamination(tuned_years, starts, months: int = 12) -> dict:
    """The loop's starting defaults were chosen by a sensitivity study on the outcomes of `tuned_years` (real calendar
    years of earlier blind cycles). A window whose 12 months touch one of those years is in-sample for its own defaults.
    Returns the share of possible window starts (`starts`) that are contaminated and per-window flags."""
    ty = {int(y) for y in tuned_years}
    st = pd.DatetimeIndex(starts)
    if not len(st):
        return {"n_windows": 0, "contaminated_share": float("nan"), "n_tuned_years": len(ty)}
    ends = st + pd.DateOffset(months=months) - pd.Timedelta(days=1)
    hit = np.array([any(y in ty for y in range(a.year, b.year + 1)) for a, b in zip(st, ends)])
    return {"n_windows": int(len(st)), "contaminated_share": float(hit.mean()), "n_tuned_years": len(ty),
            "clean_starts": [str(x.date()) for x in st[~hit]][:10]}


def neutral_default_cfg(space: dict) -> dict:
    """A data-free starting configuration: the middle grid value of every knob in `space` (lists as in CFG_SPACE; duplicates
    collapse first so a weighted list does not tilt it). Chosen by position, never by any outcome."""
    out = {}
    for k, vals in space.items():
        uniq = []
        for v in vals:
            if v not in uniq:
                uniq.append(v)
        out[k] = uniq[len(uniq) // 2]
    return out


def free_text_columns(df: pd.DataFrame, max_len: int = 80) -> list[str]:
    """Columns of a frame the trader could read as prose (news, headlines, speeches): string columns whose 99th-percentile
    length is at least `max_len`. Codes, form types and ticker-like fields are far shorter."""
    out = []
    for c in df.columns:
        s = df[c]
        if s.dtype == object or str(s.dtype) in ("str", "string") or str(s.dtype).startswith("string"):
            ss = s.dropna().astype(str)
            if len(ss) and float(ss.str.len().quantile(0.99)) >= max_len:
                out.append(c)
    return out


# =====================================================================================================================
# 5. Macro vintages
# =====================================================================================================================
# series whose history is rewritten after first release (or set retroactively) on FRED's current vintage
MACRO_REVISED = {"UNRATE": "monthly, revised (seasonal factors, benchmark)", "CPIAUCSL": "seasonally adjusted, revised each January",
                 "INDPRO": "monthly, revised for up to several years", "USREC": "NBER dates a turning point 6-21 months later",
                 "NFCI": "whole history re-estimated every week", "STLFSI4": "whole history re-estimated every week",
                 "UMCSENT": "small revisions to the final reading", "DTWEXBGS": "weights revised annually"}
MACRO_UNREVISED = {"DGS10", "DGS2", "DGS3MO", "T10Y2Y", "T10Y3M", "DFF", "BAMLH0A0HYM2", "BAMLC0A0CM", "T10YIE", "VIXCLS", "DCOILWTICO"}
MACRO_LAG_DAYS = {"UNRATE": 35, "CPIAUCSL": 45, "INDPRO": 50, "UMCSENT": 30, "DTWEXBGS": 7, "NFCI": 7, "STLFSI4": 7, "USREC": 700}


def macro_revision_risk(columns) -> pd.DataFrame:
    """Classify macro columns: revised (needs first-release vintages or a drop) vs market-priced (never revised)."""
    rows = [{"series": c, "revised": c in MACRO_REVISED, "reason": MACRO_REVISED.get(c, "market-priced daily"
            if c in MACRO_UNREVISED else "unknown - treated as revised"), "known": c in MACRO_REVISED or c in MACRO_UNREVISED}
            for c in columns]
    df = pd.DataFrame(rows, columns=["series", "revised", "reason", "known"])
    df.loc[~df["known"], "revised"] = True
    return df


def pit_macro(M: pd.DataFrame, as_of, vintages: dict | None = None, drop_revised: bool = True) -> pd.DataFrame:
    """Macro as known at `as_of`: rows after as_of removed; every revised series either taken from its own first-release
    `vintages[col]` (a Series dated by observation, as first published) or dropped; unrevised daily series kept but
    shifted by one day. Monthly series that are kept are pushed by their publication lag in calendar days."""
    as_of = pd.Timestamp(as_of)
    out = {}
    risk = macro_revision_risk(M.columns).set_index("series")
    for c in M.columns:
        if vintages and c in vintages:
            s = vintages[c].dropna()
        elif risk.loc[c, "revised"] and drop_revised:
            continue
        else:
            s = M[c].dropna()
        lag = MACRO_LAG_DAYS.get(c, 1)
        s = s.copy()
        s.index = s.index + pd.Timedelta(days=lag)
        out[c] = s[s.index <= as_of]
    return pd.DataFrame(out).sort_index()


# =====================================================================================================================
# 6. Year fingerprints
# =====================================================================================================================
def daily_fingerprint_series(close: pd.DataFrame, open_: pd.DataFrame | None, high: pd.DataFrame | None, low: pd.DataFrame | None,
                             volume: pd.DataFrame | None, market_close: pd.DataFrame | None = None, chunk: int = 400) -> pd.DataFrame:
    """One row per session of the per-day quantities a blind feed exposes about the world, in chunks of rows so a 16,000 x
    6,800 panel never sits twice in memory: names with a price, median log price, share of prices under $3, median log
    dollar volume, share of zero-volume names, share of bars with high == low (a data-vendor artefact of old years), share of
    missing opens, plus SPY log level and VIX."""
    n = len(close)
    cols = {k: np.full(n, np.nan) for k in ("n_names", "med_logp", "share_lt3", "med_logdv", "zero_vol", "hl_equal", "open_nan")}
    for a in range(0, n, chunk):
        sl = slice(a, min(n, a + chunk))
        c = close.iloc[sl].to_numpy(dtype="float64")
        ok = np.isfinite(c)
        cnt = ok.sum(1)
        cols["n_names"][sl] = cnt
        with np.errstate(all="ignore"):
            lp = np.where(ok, np.log(np.where(c > 0, c, np.nan)), np.nan)
            cols["med_logp"][sl] = np.nanmedian(lp, axis=1) if c.shape[1] else np.nan
            cols["share_lt3"][sl] = np.where(cnt > 0, np.where(ok, c < 3, False).sum(1) / np.maximum(cnt, 1), np.nan)
            if volume is not None:
                v = volume.iloc[sl].reindex(columns=close.columns).to_numpy(dtype="float64")
                cols["med_logdv"][sl] = np.nanmedian(np.log1p(np.where(ok, c * v, np.nan)), axis=1)
                cols["zero_vol"][sl] = np.where(cnt > 0, (ok & (v == 0)).sum(1) / np.maximum(cnt, 1), np.nan)
            if high is not None and low is not None:
                h = high.iloc[sl].reindex(columns=close.columns).to_numpy(dtype="float64")
                l = low.iloc[sl].reindex(columns=close.columns).to_numpy(dtype="float64")
                cols["hl_equal"][sl] = np.where(cnt > 0, (ok & (h == l)).sum(1) / np.maximum(cnt, 1), np.nan)
            if open_ is not None:
                o = open_.iloc[sl].reindex(columns=close.columns).to_numpy(dtype="float64")
                cols["open_nan"][sl] = np.where(cnt > 0, (ok & ~np.isfinite(o)).sum(1) / np.maximum(cnt, 1), np.nan)
    df = pd.DataFrame(cols, index=close.index)
    if market_close is not None:
        for name, col in (("spy_log", "SPY"), ("vix", "^VIX")):
            if col in market_close:
                s = market_close[col].reindex(df.index)
                df[name] = np.log(s) if name == "spy_log" else s
    return df


CAL_FEATS = ["gap2_rate", "gap4_rate", "gap5plus", "sessions_per_year", "mon_share", "wed_share", "fri_share", "max_gap",
             "short_week_rate", "very_short_weeks"]
LEVEL_RAW = ["log_n_names", "med_logp", "share_lt3", "med_logdv", "zero_vol", "hl_equal", "open_nan", "spy_log_start", "spy_log_end"]
LEVEL_SCRUB = ["n_names_change", "spy_ret", "logp_change", "logdv_change"]
STATE_FEATS = ["vix_mean", "vix_max", "vix_sd", "spy_vol", "spy_dd", "spy_up_share"]
TRADER_COLS = ["m_spy_ma50", "m_spy_ma200", "m_spy_r5", "m_vix", "m_vix_chg5", "m_breadth", "m_dispersion"]
TRADER_INPUT_FEATS = [f"{c}_{a}" for c in TRADER_COLS for a in ("mean", "sd")]        # what the model and the memory context consume
LEVEL_HARDENED = [f for f in LEVEL_RAW if not f.startswith("spy_log")]                 # levels left after HardenedFeed rebases SPY
GROUPS = {"calendar": CAL_FEATS, "levels_raw": LEVEL_RAW, "levels_after_hardening": LEVEL_HARDENED, "levels_scrubbed": LEVEL_SCRUB,
          "market_state": STATE_FEATS, "trader_inputs": TRADER_INPUT_FEATS}


def calendar_features(sessions: pd.DatetimeIndex) -> dict:
    """Holiday / closure signature of a session calendar: 1-day midweek closures, long weekends, multi-day closures, session
    counts, weekday shares. A whole-week shift keeps these exactly (livesim's check_calendar demands it)."""
    d = pd.DatetimeIndex(sessions)
    if len(d) < 30:
        return {k: float("nan") for k in CAL_FEATS}
    gap = np.diff(d.values).astype("timedelta64[D]").astype(int)
    wd = d.dayofweek.to_numpy()[1:]
    years = max((d[-1] - d[0]).days / 365.25, 1e-9)
    mid = (gap == 2) & (wd != 0)                       # single-day midweek close
    wk = pd.Series(1, index=d).groupby([d.isocalendar().year.to_numpy(), d.isocalendar().week.to_numpy()]).sum()
    wk = wk.iloc[1:-1]                                 # the first and last weeks are cut by the window, not by a holiday
    return {"gap2_rate": float(mid.sum() / years), "gap4_rate": float((gap == 4).sum() / years), "gap5plus": float((gap >= 5).sum()),
            "sessions_per_year": float(len(d) / years), "mon_share": float((d.dayofweek == 0).mean()),
            "wed_share": float((d.dayofweek == 2).mean()), "fri_share": float((d.dayofweek == 4).mean()), "max_gap": float(gap.max()),
            "short_week_rate": float((wk < 5).sum() / years), "very_short_weeks": float((wk <= 3).sum())}


def window_features(daily: pd.DataFrame, start, warm_years: int = 6, months: int = 12, first_data="1962-01-01") -> dict:
    """The fingerprint of one blind window as the feed would expose it (warm-up + the 12 hidden months): four feature
    groups - calendar, raw levels, scrubbed (relative) levels, market state. `daily` is daily_fingerprint_series output."""
    start = pd.Timestamp(start)
    w = min(warm_years, (start - pd.Timestamp(first_data)).days // 365)
    lo, hi = start - pd.DateOffset(years=w), start + pd.DateOffset(months=months) - pd.Timedelta(days=1)
    d = daily.loc[lo:hi]
    d = d[d["n_names"] > 0]
    out = calendar_features(d.index)
    if len(d) < 30:
        return {**out, **{k: float("nan") for k in LEVEL_RAW + LEVEL_SCRUB + STATE_FEATS}}
    live = d.loc[start:]
    out.update(log_n_names=float(np.log(d["n_names"].mean())), med_logp=float(d["med_logp"].mean()), share_lt3=float(d["share_lt3"].mean()),
               med_logdv=float(d["med_logdv"].mean()) if "med_logdv" in d else np.nan, zero_vol=float(d["zero_vol"].mean()),
               hl_equal=float(d["hl_equal"].mean()), open_nan=float(d["open_nan"].mean()),
               spy_log_start=float(d["spy_log"].iloc[0]) if "spy_log" in d else np.nan,
               spy_log_end=float(d["spy_log"].iloc[-1]) if "spy_log" in d else np.nan)
    n0, n1 = d["n_names"].iloc[: max(1, len(d) // 4)].mean(), d["n_names"].iloc[-max(1, len(d) // 4):].mean()
    out.update(n_names_change=float(np.log(n1 / n0)), spy_ret=float(d["spy_log"].iloc[-1] - d["spy_log"].iloc[0]) if "spy_log" in d else np.nan,
               logp_change=float(d["med_logp"].iloc[-1] - d["med_logp"].iloc[0]),
               logdv_change=float(d["med_logdv"].iloc[-1] - d["med_logdv"].iloc[0]) if "med_logdv" in d else np.nan)
    for c in TRADER_COLS:
        if c in d:
            out[f"{c}_mean"], out[f"{c}_sd"] = float(d[c].mean()), float(d[c].std())
        else:
            out[f"{c}_mean"] = out[f"{c}_sd"] = float("nan")
    if "vix" in d and "spy_log" in d and len(live) > 20:
        sr = live["spy_log"].diff().dropna()
        eq = np.exp(live["spy_log"] - live["spy_log"].iloc[0])
        out.update(vix_mean=float(live["vix"].mean()), vix_max=float(live["vix"].max()), vix_sd=float(live["vix"].std()),
                   spy_vol=float(sr.std()), spy_dd=float((eq / eq.cummax() - 1).min()), spy_up_share=float((sr > 0).mean()))
    else:
        out.update({k: float("nan") for k in STATE_FEATS})
    return out


def regular_grid_index(n_sessions: int, start="2100-01-03") -> pd.DatetimeIndex:
    """A holiday-free calendar: the n-th session lands on the n-th weekday from `start` (Monday). Relabelling a window's
    sessions onto it removes every closure signature - at the cost that the trader's 'week' becomes exactly 5 sessions
    instead of the real (sometimes 4-session) week. An opt-in scrub, not wired into the Feed (its check_calendar gate
    demands the real pattern)."""
    return pd.bdate_range(pd.Timestamp(start), periods=n_sessions)


def regime_series(close: pd.DataFrame, market_close: pd.DataFrame, col_chunk: int = 800) -> pd.DataFrame:
    """The market-context inputs the trader's model and memory consume (features.regime_frame's m_* columns) as a daily
    series over the WHOLE panel, computed in column blocks so the 16,000 x 6,800 frame is never copied whole: breadth is the
    share of names above their 50-day mean, dispersion the 5-day mean of the cross-sectional std of daily log returns
    (accumulated from n, sum, sum of squares). Uses every name, where the trader's copy uses the tradable ones."""
    idx = close.index
    n_ma, above, n_r, s1, s2 = (np.zeros(len(idx)) for _ in range(5))
    for a in range(0, close.shape[1], col_chunk):
        blk = close.iloc[:, a:a + col_chunk].astype("float64")
        ma = blk.rolling(50).mean()
        above += (blk > ma).sum(axis=1).to_numpy()
        n_ma += blk.notna().sum(axis=1).to_numpy()
        r = np.log(blk / blk.shift(1)).replace([np.inf, -np.inf], np.nan)
        ok = r.notna()
        n_r += ok.sum(axis=1).to_numpy()
        r0 = r.fillna(0.0)
        s1 += r0.sum(axis=1).to_numpy()
        s2 += (r0 ** 2).sum(axis=1).to_numpy()
    with np.errstate(all="ignore"):
        var = (s2 - s1 ** 2 / np.where(n_r > 0, n_r, np.nan)) / (n_r - 1)
        disp = pd.Series(np.sqrt(np.clip(var, 0, None)), index=idx).rolling(5).mean()
        breadth = pd.Series(above / np.where(n_ma > 0, n_ma, np.nan), index=idx)
    spy = market_close["SPY"].reindex(idx)
    vix = market_close["^VIX"].reindex(idx).ffill()
    return pd.DataFrame({"m_spy_ma50": spy / spy.rolling(50).mean() - 1, "m_spy_ma200": spy / spy.rolling(200).mean() - 1,
                         "m_spy_r5": np.log(spy / spy.shift(5)), "m_vix": vix, "m_vix_chg5": np.log(vix / vix.shift(5)),
                         "m_breadth": breadth, "m_dispersion": disp}, index=idx)


def thin_universe_share(alive_by_year: pd.Series, starts, min_names: int = 500, months: int = 12) -> dict:
    """How many possible blind windows run on a universe too thin to mean anything. A window's names = the SMALLEST yearly
    count over the calendar years it touches (a survivor panel is thinnest at the start of history). Returns the share of
    `starts` below `min_names`, the first start that clears it, and the count per decade of starts."""
    st = pd.DatetimeIndex(starts)
    if not len(st) or alive_by_year is None or not len(alive_by_year):
        return {"n_windows": int(len(st)), "share_below": float("nan"), "first_clean_start": None}
    ends = st + pd.DateOffset(months=months) - pd.Timedelta(days=1)
    n = np.array([min(alive_by_year.get(y, 0) for y in range(a.year, b.year + 1)) for a, b in zip(st, ends)])
    below = n < min_names
    clean = st[~below]
    dec = pd.Series(below, index=st).groupby((st.year // 10) * 10).mean()
    return {"n_windows": int(len(st)), "min_names": int(min_names), "share_below": float(below.mean()),
            "first_clean_start": str(clean[0].date()) if len(clean) else None,
            "share_below_by_decade": {int(k): float(v) for k, v in dec.items()},
            "median_names_in_window": float(np.median(n)), "min_names_in_any_window": int(n.min())}


class FingerprintProbe:
    """Can a classifier tell WHEN a window was from what the feed exposes? Trains ExtraTrees on windows from blocked years and
    tests on years it never saw (train: start year mod 4 in {0,1}; test: mod 4 == 3; year mod 4 == 2 is a gap so no training
    window overlaps a test window). Reports MAE in years versus predicting the training mean, and decade accuracy versus the
    majority-decade rate. Chance-level MAE and accuracy mean the channel is closed."""

    def __init__(self, seed: int = 0, n_trees: int = 200):
        self.seed, self.n_trees = seed, n_trees

    @staticmethod
    def split(starts: pd.DatetimeIndex):
        """(train, test) positions. Test = start year mod 4 == 3; train = mod 4 in {0,1}, minus any train window whose 12 months
        overlap a test window's (a Dec-of-year-3 test start would otherwise overlap a Jan-of-year-4 train start). The 6-year
        warm-up still shares data between neighbours, so the estimate is an upper bound on identifiability."""
        y = np.asarray(starts.year)
        te = np.flatnonzero(y % 4 == 3)
        tr = np.flatnonzero(np.isin(y % 4, [0, 1]))
        if len(te) and len(tr):
            ts = starts[te].values.astype("datetime64[D]").astype(np.int64)
            ss = starts[tr].values.astype("datetime64[D]").astype(np.int64)
            near = np.abs(ss[:, None] - ts[None, :]).min(axis=1) < 365
            tr = tr[~near]
        return tr, te

    def score(self, F: pd.DataFrame, starts: pd.DatetimeIndex, cols: list[str]) -> dict:
        from sklearn.ensemble import ExtraTreesRegressor
        tr, te = self.split(starts)
        X = F[cols].to_numpy(dtype="float64")
        keep = np.isfinite(X).all(axis=1)
        tr, te = tr[keep[tr]], te[keep[te]]
        if len(tr) < 10 or len(te) < 5:
            return {"n_train": int(len(tr)), "n_test": int(len(te)), "mae_years": float("nan"), "baseline_mae_years": float("nan"),
                    "decade_acc": float("nan"), "majority_decade_acc": float("nan"), "skill": float("nan")}
        yr = starts.year.to_numpy().astype(float) + (starts.month.to_numpy() - 1) / 12.0
        m = ExtraTreesRegressor(n_estimators=self.n_trees, min_samples_leaf=3, random_state=self.seed, n_jobs=1).fit(X[tr], yr[tr])
        pred = m.predict(X[te])
        mae = float(np.abs(pred - yr[te]).mean())
        base = float(np.abs(yr[tr].mean() - yr[te]).mean())
        dec = lambda v: (np.floor(v / 10)).astype(int)
        maj = pd.Series(dec(yr[tr])).value_counts().idxmax()
        return {"n_train": int(len(tr)), "n_test": int(len(te)), "mae_years": mae, "baseline_mae_years": base,
                "decade_acc": float((dec(pred) == dec(yr[te])).mean()), "majority_decade_acc": float((dec(yr[te]) == maj).mean()),
                "skill": float(1 - mae / base) if base > 0 else float("nan")}

    def run(self, F: pd.DataFrame, starts: pd.DatetimeIndex, groups: dict | None = None) -> dict:
        groups = groups or GROUPS
        res = {name: self.score(F, starts, cols) for name, cols in groups.items()}
        res["scrubbed_all"] = self.score(F, starts, GROUPS["calendar"] + GROUPS["levels_scrubbed"] + GROUPS["market_state"]) \
            if groups is GROUPS else {}
        return res


def fingerprint_verdict(score: dict, skill_open: float = 0.25) -> str:
    """'identifiable' when the probe beats the mean-year guess by at least `skill_open` of the error (a quarter by default)."""
    s = score.get("skill", float("nan"))
    return "identifiable" if np.isfinite(s) and s >= skill_open else "not identifiable"


# =====================================================================================================================
# 8b. Feature causality
# =====================================================================================================================
def truncation_invariance(build, stocks: dict, market: dict, ev, ins, sic, cuts, window: int = 25, tol: float = 1e-5, start=None) -> dict:
    """Future-invariance of a feature builder on real-shaped frames. For each cut date T: build from data cut at T (prices,
    market, filings accepted by the close of T, insider forms filed by T) and from the FULL data; the last `window` sessions
    up to T must match to `tol`. Returns per feature the worst absolute difference over all cuts and the offending cut."""
    full, _ = build(stocks, market, ev, ins, sic, start=start, relative=True)
    worst: dict[str, tuple[float, str]] = {}
    for T in pd.to_datetime(list(cuts)):
        s = {f: v.loc[:T] for f, v in stocks.items()}
        m = {f: v.loc[:T] for f, v in market.items()}
        e = ev[ev["accepted"] <= T.tz_localize("UTC") + pd.Timedelta(hours=20, minutes=30)] if len(ev) else ev
        i = ins[ins["filed"] <= T] if ins is not None and len(ins) else ins
        cut, _ = build(s, m, e, i, sic, start=start, relative=True)
        dates = cut.index.get_level_values(0)
        recent = dates[dates <= T].unique().sort_values()[-window:]
        a = cut[cut.index.get_level_values(0).isin(recent)]
        b = full.reindex(a.index)
        diff = (a - b).abs().where(~(a.isna() & b.isna()), 0.0)
        mismatch = a.isna() != b.isna()
        diff = diff.mask(mismatch, np.inf)
        for c in diff.columns:
            v = float(diff[c].max()) if len(diff) else 0.0
            if v > worst.get(c, (0.0, ""))[0]:
                worst[c] = (v, str(T.date()))
    bad = {c: {"max_diff": v, "cut": t} for c, (v, t) in worst.items() if v > tol}
    return {"n_features": int(full.shape[1]), "n_cuts": len(list(cuts)), "leaky_features": bad, "clean": not bad}


# =====================================================================================================================
# 8c. What the feed shows the trader
# =====================================================================================================================
def _spearman(a, b) -> float:
    a, b = pd.Series(a).rank(), pd.Series(b).rank()
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def feed_exposure(feed) -> dict:
    """Trader-visible facts of a built Feed that carry real-world information: (a) SPY's absolute level (year), (b) whether
    the column ORDER of the disguised frames follows the real alphabetical ticker order (identity), (c) names shown before
    they list (IPOs inside the window), (d) the era-coded trading cost, (e) the warm-up length (short only for starts before
    1968)."""
    close = feed._stocks["Close"]
    inv = {code: real for real, code in feed._map.items()}
    reals = [inv[c] for c in close.columns]
    seen = feed.history()[0]["Close"]
    empty_now = int((~seen.notna().any()).sum())
    spy = feed._market["Close"]["SPY"].dropna() if "SPY" in feed._market["Close"] else pd.Series(dtype=float)
    return {"spy_first_level": float(spy.iloc[0]) if len(spy) else float("nan"),
            "column_order_vs_real_alpha_rho": _spearman(range(len(reals)), pd.Series(reals).rank(method="first")),
            "columns_shown_before_listing": empty_now, "n_columns": int(close.shape[1]),
            "cost_bps": getattr(feed, "cost_bps", None), "warmup_sessions_before_window": int(feed.i + 1)}


def rerun_linkability(feed_a, feed_b, max_names: int = 400, seed: int = 0) -> dict:
    """C55 at the feed level: two disguised runs of the SAME real window (different seals). How many names can an observer
    re-identify across them (a) by column POSITION (plain Feed keeps the real alphabetical order in both runs) and (b) by the
    correlation of their daily returns over the hidden window (always possible: the data is the same, only labels and dates
    differ - so any stored numeric memory of the first run recognises the second, which is why learned state must exclude
    it). Ground truth from the two code maps, which only the audit reads."""
    ca, cb = feed_a._stocks["Close"], feed_b._stocks["Close"]
    inv_a = {c: r for r, c in feed_a._map.items()}
    common = [c for c in ca.columns if inv_a[c] in feed_b._map and feed_b._map[inv_a[c]] in cb.columns]
    if not common:
        return {"n_names": 0}
    rng = np.random.default_rng(seed)
    if len(common) > max_names:
        common = list(rng.choice(common, size=max_names, replace=False))
    truth = {a: feed_b._map[inv_a[a]] for a in common}
    k = min(len(ca), len(cb))
    ra = np.log(ca[common].iloc[-k:] / ca[common].iloc[-k:].shift(1)).iloc[1:]
    rb = np.log(cb.iloc[-k:] / cb.iloc[-k:].shift(1)).iloc[1:]
    A = ra.fillna(0.0).to_numpy(dtype="float64")
    B = rb.fillna(0.0).to_numpy(dtype="float64")
    A = (A - A.mean(0)) / np.where(A.std(0) > 0, A.std(0), 1.0)
    B = (B - B.mean(0)) / np.where(B.std(0) > 0, B.std(0), 1.0)
    best = (A.T @ B).argmax(axis=1)
    cols_b = list(rb.columns)
    by_ret = float(np.mean([cols_b[j] == truth[a] for a, j in zip(common, best)]))
    pos_a = {c: i for i, c in enumerate(ca.columns)}
    pos_b = {c: i for i, c in enumerate(cb.columns)}
    n_a, n_b = len(ca.columns), len(cb.columns)
    by_pos = float(np.mean([abs(pos_a[a] / max(1, n_a - 1) - pos_b[truth[a]] / max(1, n_b - 1)) < 1.0 / max(1, min(n_a, n_b)) * 0.5 + 1e-12 for a in common]))
    same_dates = bool((feed_a.sessions[:5] == feed_b.sessions[:5]).all())
    return {"n_names": len(common), "share_reidentified_by_column_position": by_pos, "share_reidentified_by_return_correlation": by_ret,
            "same_shown_dates": same_dates, "same_code_names": sum(feed_a._map[r] == feed_b._map.get(r) for r in feed_a._map) / max(1, len(feed_a._map))}


def hardened_feed_class():
    """`Feed` with the trader-facing leaks of `feed_exposure` closed (imported lazily: livesim pulls in the whole engine):
    columns sorted by code name (no real alphabetical order), market prices rebased to 100 and market volume to its early mean
    (no absolute SPY level), and history() hides a name until its first price (no columns for future IPOs). Decisions on data
    without IPOs are bit-identical to the plain Feed (tests/test_leak_audit.py)."""
    from . import livesim

    class HardenedFeed(livesim.Feed):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            close = self._stocks["Close"]
            order = sorted(close.columns)
            self._stocks = {f: v[order] for f, v in self._stocks.items()}
            ok = self._stocks["Close"].notna().to_numpy()
            self._first_i = np.where(ok.any(axis=0), ok.argmax(axis=0), np.iinfo(np.int64).max // 2)
            self._market = {f: self._rebase(f, v) for f, v in self._market.items()}

        @staticmethod
        def _rebase(field_, v):
            v = v.copy()
            for c in v.columns:
                if field_ == "Volume":
                    base = v[c].iloc[:250].mean()
                    if np.isfinite(base) and base > 0:
                        v[c] = v[c] / base
                elif not str(c).startswith("^"):                    # index levels (VIX) are market state, not a price level
                    seen = v[c].dropna()
                    if len(seen) and seen.iloc[0] > 0:
                        v[c] = v[c] / seen.iloc[0] * 100.0
            return v

        def history(self, lookback=None):
            stocks, market = super().history(lookback)
            vis = self._first_i <= self.i
            return {f: v.loc[:, vis] for f, v in stocks.items()}, market

        def features_until_now(self):
            """The ATR frame the trainer pairs with history() must show the same names (labels() aligns them)."""
            X, atr = super().features_until_now()
            return X, atr.loc[:, self._first_i <= self.i]

    return HardenedFeed


def hardened_run(cfg, run_id, log=print, check_parity=True, adaptive=False, meta=None, network_guard=True, data=None, warmup_years=6):
    """`livesim.run` on the hardened feed, with the network closed for the whole run (channel 7). `data` replaces the real caches
    (synthetic windows in tests)."""
    import time
    from . import livesim
    guard = NetworkGuard().install() if network_guard else None
    try:
        sealed = livesim.SealedYear(run_id)
        feed = hardened_feed_class()(sealed, warmup_years=warmup_years, data=data)
        t = time.perf_counter()
        feed.precompute_features()
        log(f"  feature service ready in {time.perf_counter() - t:.0f}s")
        if check_parity:
            w = livesim.parity_test(feed)
            log(f"  parity test passed (max diff {w:.1e})")
        trader = livesim.BlindTrader(feed, cfg, adaptive=adaptive, meta=meta)
        trader.train()
        wall = livesim.drive(feed, trader.on_tick)
        return feed, trader, sealed, wall
    finally:
        if guard is not None:
            guard.uninstall()
