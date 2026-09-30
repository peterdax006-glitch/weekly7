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

F06 (C69 W-10/W-11) adds, at the end of the file: the training-call gate (split_training_windows / refuse_late_training:
a basis search never sees a window that had not ended before the first real day the basis may be played), the data-free
adaptation meta (neutral_default_meta, tuned_meta_keys), and the computed channel-6 lookup part (year_lookup_audit /
verdict_year_lookup: can anything that is LOOKED UP by year - curator releases, research records, the memory bank - reach the
trader in a disguised replay?).

Nothing here reads state/livesim (sealed windows), writes to data/cache, uses a clock, or draws without a seed.
Python cannot stop deliberate reflection; the guards stop honest mistakes and the tests prove they catch planted ones."""
from __future__ import annotations

import ast
import json
import re
import socket
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import config as K
from .learning.core import FirewallBreach

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
        out = {s: 0 for s in STATUSES if s != UNMEASURED}         # UNMEASURED appears only when something was not measured
        for c in self.channels.values():
            out[c.status] = out.get(c.status, 0) + 1
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
        lines += [f"**{len(self.channels)} channels: " + ", ".join(f"{c[s]} {s}" for s in c) + "**", ""]
        lines += ["LEAK = present in the current default blind path (a tested fix or hook exists where stated); FIXED = closed in the default path; "
                  "QUARANTINED = cannot be removed from the data, measured, results must carry the stated rule; CLEAN = nothing found, tested; "
                  "UNMEASURED = a part or source the verdict needs is missing (never read as CLEAN). Statuses are computed by compute_verdicts, not typed.", ""]
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


# =====================================================================================================================
# Computed verdicts (S18; contract sections 30, 55, 85; canon C56). A channel's status is DERIVED here from (a) the source of
# the default blind path (AST, never a grep for a comment), (b) the measurements the parts wrote, and (c) proofs that run a
# planted leak through the mechanism that is supposed to stop it. Nothing below returns a status typed by hand: an unsafe
# default flips the answer, and a missing measurement gives UNMEASURED, never CLEAN.
# =====================================================================================================================
UNMEASURED = "UNMEASURED"
STATUSES = (LEAK, CLEAN, FIXED, QUARANTINED, UNMEASURED)
_SEVERITY = {CLEAN: 0, FIXED: 1, QUARANTINED: 2, UNMEASURED: 3, LEAK: 4}
_MISSING = object()
_UNDETERMINED = "<undetermined>"          # a default that is not a literal (or a function that is gone): unknown, never assumed safe
_SOURCE_FILES = {"livesim": "engine/livesim.py", "loop2": "scripts/livesim_loop2.py", "features": "engine/features.py",
                 "adaptive": "engine/adaptive.py", "leak_audit": "engine/leak_audit.py"}
_PLAY_CALLEES = {"run_workers", "classify_round", "worker", "_worker", "run_window", "replay", "run"}


def worst_status(*statuses) -> str:
    """The most severe of several sub-verdicts (LEAK > UNMEASURED > QUARANTINED > FIXED > CLEAN); no input is UNMEASURED."""
    return max(statuses, key=_SEVERITY.__getitem__) if statuses else UNMEASURED


@dataclass(frozen=True)
class Verdict:
    """A computed status with the named checks behind it. A check is True (holds), False (fails) or None (could not be
    evaluated); `reasons` lists, in words, why the status is not better than it is."""
    status: str
    checks: dict
    reasons: tuple = ()

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, not {self.status!r}")

    def as_evidence(self) -> dict:
        return {"computed_status": self.status, "checks": self.checks, "why_not_better": list(self.reasons)}


# ---- AST helpers ----------------------------------------------------------------------------------------------------
def _find_def(tree, name, cls=None):
    body = tree.body
    if cls is not None:
        holder = next((n for n in body if isinstance(n, ast.ClassDef) and n.name == cls), None)
        if holder is None:
            return None
        body = holder.body
    return next((n for n in body if isinstance(n, ast.FunctionDef) and n.name == name), None)


def _default_of(fn, arg):
    """Literal default of `arg` in function `fn`, or _MISSING when the function/argument/literal is absent."""
    if fn is None:
        return _MISSING
    a = fn.args
    pos = a.posonlyargs + a.args
    defaults = [None] * (len(pos) - len(a.defaults)) + list(a.defaults)
    for p, d in list(zip(pos, defaults)) + list(zip(a.kwonlyargs, a.kw_defaults)):
        if p.arg == arg:
            try:
                return ast.literal_eval(d) if d is not None else _MISSING
            except ValueError:
                return _MISSING
    return _MISSING


def _callee(call) -> str | None:
    f = getattr(call, "func", None)
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None


def _calls(node, name):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and _callee(n) == name] if node is not None else []


def _kw(call, name):
    return next((k.value for k in call.keywords if k.arg == name), None)


def _has_kw(call, name) -> bool:
    return any(k.arg == name for k in call.keywords)


def _state_ref(node) -> bool:
    """Does the subtree read st['cfg'] or st['meta'] (the loop's latest global basis)?"""
    return any(isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == "st"
               and isinstance(n.slice, ast.Constant) and n.slice.value in ("cfg", "meta") for n in ast.walk(node))


def _module_assign(tree, name):
    return next((n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)), None)


def _module_literal(tree, name):
    n = _module_assign(tree, name)
    if n is None:
        return None
    try:
        return ast.literal_eval(n.value)
    except ValueError:
        return None


def _read_sources(root=None, overrides=None) -> dict:
    root = Path(root) if root else K.ROOT
    out = {}
    for key, rel in _SOURCE_FILES.items():
        if overrides and key in overrides:
            out[key] = overrides[key]
            continue
        p = root / rel
        out[key] = p.read_text(encoding="utf-8") if p.exists() else None
    return out


def _livesim_facts(tree, text) -> dict:
    feed_init, pre = _find_def(tree, "__init__", "Feed"), _find_def(tree, "precompute_features", "Feed")
    parity, bfc, run = _find_def(tree, "parity_test"), _find_def(tree, "blind_feed_class"), _find_def(tree, "run")
    lit = lambda v: _UNDETERMINED if v is _MISSING else v
    passes = lambda fn, expect: None if fn is None else any(
        _kw(c, "tradable_rule") is not None and ast.unparse(_kw(c, "tradable_rule")) == expect for c in _calls(fn, "build"))
    via = _calls(run, "blind_feed_class")
    return {"feed_tradable_rule_default": lit(_default_of(feed_init, "tradable_rule")),
            "precompute_passes_feed_rule": passes(pre, "self.tradable_rule"),
            "parity_passes_feed_rule": passes(parity, "feed.tradable_rule"),
            "blind_feed_hardened_default": lit(_default_of(bfc, "hardened")),
            "blind_feed_returns_hardened_class": None if bfc is None else bool(_calls(bfc, "hardened_feed_class")),
            "run_hardened_default": lit(_default_of(run, "hardened")),
            "run_builds_feed_via_blind_feed_class": bool(via) and all(c.args and ast.unparse(c.args[0]) == "hardened" for c in via),
            "run_builds_plain_feed_directly": None if run is None else bool(_calls(run, "Feed")),
            "livesim_injects_dead_names": "inject_dead_names" in text}


def _loop2_facts(tree) -> dict:
    main, fresh, plan = _find_def(tree, "main"), _find_def(tree, "fresh_state"), _find_def(tree, "plan_round")
    worker_fn, inner = _find_def(tree, "worker"), _find_def(tree, "_worker")
    neutral = _module_assign(tree, "NEUTRAL_CFG")
    named = lambda node, ident: node is not None and any(isinstance(n, ast.Name) and n.id == ident for n in ast.walk(node))
    f = {"loop2_cfg_space": _module_literal(tree, "CFG_SPACE"),
         "loop2_neutral_cfg_from_neutral_default_cfg": bool(neutral is not None and _callee(neutral.value) == "neutral_default_cfg" and "CFG_SPACE" in ast.unparse(neutral.value)),
         "loop2_fresh_state_starts_neutral": named(fresh, "NEUTRAL_CFG"),
         "loop2_plan_round_uses_lineage_basis_for": None if plan is None else bool(_calls(plan, "basis_for")),
         "loop2_plan_round_falls_back_to_neutral": named(plan, "NEUTRAL_CFG"),
         "loop2_main_plans_each_round": None if main is None else bool(_calls(main, "plan_round")),
         "loop2_worker_installs_network_guard": bool(worker_fn is not None and _calls(worker_fn, "NetworkGuard") and _calls(worker_fn, "install"))}
    if main is not None:
        loops = {name: _calls(main, name) for name in ("run_workers", "classify_round")}
        f["loop2_play_calls_without_per_id"] = sorted(f"{n}@{c.lineno}" for n, cs in loops.items() for c in cs if not _has_kw(c, "per_id"))
        f["loop2_n_run_workers_calls"] = len(loops["run_workers"])
        f["loop2_global_basis_reaches_a_play_call"] = sorted(
            f"{_callee(c)}@{c.lineno}" for c in ast.walk(main) if isinstance(c, ast.Call) and _callee(c) in _PLAY_CALLEES and not _has_kw(c, "per_id")
            and any(_state_ref(a) for a in list(c.args) + [k.value for k in c.keywords]))
        tb, rb = _calls(main, "train_basis"), _calls(main, "register_basis")
        f["loop2_train_basis_calls"] = [{"line": c.lineno, "kwargs": [k.arg for k in c.keywords], "passes_as_of": _has_kw(c, "as_of"),
                                         "training_set": ast.unparse(c.args[0]) if c.args else None} for c in tb]
        f["loop2_register_basis_calls"] = [{"line": c.lineno, "trained_on": ast.unparse(c.args[-1]) if c.args else None} for c in rb]
        f["loop2_registers_exactly_the_training_set"] = bool(tb and rb and all(r["trained_on"] == f["loop2_train_basis_calls"][0]["training_set"] for r in f["loop2_register_basis_calls"]))
        head = next((n for n in ast.walk(main) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "headline" for t in n.targets)), None)
        f["loop2_headline_excludes_legacy"] = None if head is None else "legacy" in ast.unparse(head.value)
    runs = [c for c in _calls(inner, "run") if isinstance(c.func, ast.Attribute) and ast.unparse(c.func.value) == "livesim"]
    f["loop2_worker_runs_livesim_run"] = bool(runs)
    f["loop2_worker_passes_hardened"] = None if not runs else all(_kw(c, "hardened") is None or ast.unparse(_kw(c, "hardened")) == "True" for c in runs)
    f.update(_loop2_gate_facts(tree, main, plan, fresh))
    return f


def default_path_facts(root=None, sources: dict | None = None) -> dict:
    """Facts read from the SOURCE of the default blind path: what the defaults are and whether every call site honours them.
    `sources` replaces file texts by key (livesim / loop2 / features / adaptive / leak_audit) so a test can flip one default
    back to the unsafe setting and watch the verdict move. A value of None means "could not be determined"."""
    src = _read_sources(root, sources)
    trees = {}
    for k, text in src.items():
        try:
            trees[k] = ast.parse(text) if text is not None else None
        except SyntaxError:
            trees[k] = None
    f: dict = {"sources_parsed": {k: t is not None for k, t in trees.items()}}
    if trees["livesim"] is not None:
        f.update(_livesim_facts(trees["livesim"], src["livesim"]))
    if trees["features"] is not None:
        build = _find_def(trees["features"], "build")
        f["features_split_branch_uses_split_invariant_tradable"] = bool(build is not None and any(
            isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "tradable" for t in n.targets)
            and isinstance(n.value, ast.Call) and _callee(n.value) == "split_invariant_tradable" for n in ast.walk(build)))
    if trees["leak_audit"] is not None:
        hf = _find_def(trees["leak_audit"], "hardened_feed_class")
        f["hardened_class_overrides_tradable_rule"] = None if hf is None else "tradable_rule" in (ast.get_source_segment(src["leak_audit"], hf) or "")
    if trees["adaptive"] is not None:
        node = _module_assign(trees["adaptive"], "META_DEFAULT")
        f["meta_default_literal"] = _module_literal(trees["adaptive"], "META_DEFAULT")
        f["meta_default_data_tuned"] = None if node is None else "sensitivity study" in (ast.get_source_segment(src["adaptive"], node) or "").lower()
    if trees["loop2"] is not None:
        f.update(_loop2_facts(trees["loop2"]))
    return f


# ---- proofs: planted leaks pushed through the real mechanism -------------------------------------------------------------
def prove_split_invariance(seed: int = 3) -> dict:
    """Plant a later 20:1 split on a third of the names (price down, volume up, as back-adjustment does). The split-invariant
    rule features.build uses on the Test path must not change a single tradable cell; the price-rank clause of the Live rule must."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=120)
    cols = [f"S{i:03d}" for i in range(45)]
    C = pd.DataFrame(rng.uniform(5, 80, (len(idx), len(cols))), index=idx, columns=cols)
    V = pd.DataFrame(rng.uniform(2e5, 4e6, C.shape), index=idx, columns=cols)
    C2, V2 = C.copy(), V.copy()
    C2.iloc[:, :15] /= 20.0
    V2.iloc[:, :15] *= 20.0
    price_rank = lambda X: X.rank(axis=1, pct=True) >= 0.2
    inv_moved = int((split_invariant_tradable(C, V) != split_invariant_tradable(C2, V2)).to_numpy().sum())
    live_moved = int((price_rank(C) != price_rank(C2)).to_numpy().sum())
    return {"split_invariant_cells_changed_by_planted_split": inv_moved, "price_rank_cells_changed_by_planted_split": live_moved,
            "mechanism_neutralises_planted_leak": inv_moved == 0, "planted_leak_is_real": live_moved > 0}


def prove_lineage_gate(n_rounds: int = 14, par: int = 3, seeds=(0, 1, 2, 3)) -> dict:
    """Replay the loop's draw process (seeded, no sealed file read) twice. NAIVE: every window plays the newest basis, trained
    on every archived window. GATED: each window plays BasisLineage.basis_for(its real start), then the round's windows join
    the training set (exactly plan_round + register_basis). A basis leaks when any window it was trained on had not ended
    when the played window began."""
    naive_touched, gated_violations, gated_untrained, gated_played = [], 0, [], 0
    for seed in seeds:
        rounds = simulate_window_draws(n_rounds, par=par, seed=20260929 + seed)
        naive_touched.append(basis_future_share(rounds)["share_of_windows_touched"])
        lin, seen, played = BasisLineage(), [], []
        for r, rnd in enumerate(rounds):
            for w in rnd:
                rec = lin.basis_for(w["start"])
                played.append({"id": w["id"], "real_start": w["start"], "version": rec["version"] if rec else 0})
                gated_untrained.append(rec is None)
            seen += rnd
            lin.register(r + 1, {}, {}, [TrainedOn(w["id"], w["start"], w["end"]) for w in seen])
        gated_violations += len(lin.violations(played))
        gated_played += len(played)
    return {"naive_share_of_plays_touched_by_future_training": float(np.mean(naive_touched)),
            "gated_plays_checked": gated_played, "gated_violations": gated_violations,
            "gated_share_of_plays_on_untrained_neutral_basis": float(np.mean(gated_untrained)),
            "mechanism_holds": gated_violations == 0, "planted_leak_is_real": float(np.mean(naive_touched)) > 0.0}


def prove_network_guard() -> dict:
    """A DNS lookup under the guard must raise NetworkBlocked before any packet leaves (no network is used by the proof)."""
    with NetworkGuard() as g:
        try:
            socket.getaddrinfo("audit.invalid", 443)
            blocked = False
        except NetworkBlocked:
            blocked = True
    return {"planted_lookup_blocked": blocked, "guard_recorded_attempt": bool(g.blocked)}


def run_proofs() -> dict:
    return {"split_invariance": prove_split_invariance(), "lineage_gate": prove_lineage_gate(), "network_guard": prove_network_guard(),
            "training_gate": prove_training_gate()}


def lineage_state_check(state: dict | None, cfg_space: dict | None, meta_default: dict | None,
                        neutral_meta: dict | None = None) -> dict | None:
    """Audit the loop's own state file (trusted side; it holds real dates only for windows a basis was trained on or that were
    revealed, and this function never opens a sealed window). For every non-legacy played window: an untrained play must have
    used exactly the data-free neutral cfg and either the data-free NEUTRAL meta or (older plays) the default meta - a play on
    the outcome-tuned META_DEFAULT is listed in `tuned_meta_plays` (F06: quarantined, never a headline number); a trained play
    must use a basis whose training windows all ended before the play began. A trained play whose real start cannot be
    recovered (lineage or its revealed `real_start`) is 'unresolved' (never assumed clean). A version registered with a
    `used_from` date (the F06 training gate) must have trained only on windows that ended before it."""
    if state is None:
        return None
    norm = lambda x: json.loads(json.dumps(x, default=str))
    windows, lineage = state.get("windows", []), state.get("lineage", [])
    dates = {i: (pd.Timestamp(a), pd.Timestamp(b)) for v in lineage for i, a, b in v["trained_on"]}
    by_v = {v["version"]: v for v in lineage}
    neutral = norm(neutral_default_cfg(cfg_space)) if cfg_space else None
    if neutral_meta is None and meta_default is not None:
        neutral_meta = default_neutral_meta(meta_default)
    nm = norm(neutral_meta) if neutral_meta is not None else None
    out = {"n_windows": len(windows), "legacy": [], "untrained": [], "trained": [], "unresolved": [], "violations": [],
           "untrained_cfg_mismatch": [], "untrained_meta_mismatch": [], "no_basis_record": [], "versions": [v["version"] for v in lineage],
           "untrained_on_neutral_meta": [], "tuned_meta_plays": [], "tuned_meta_quarantined": [], "used_from_violations": [],
           "versions_without_used_from": []}
    for w in windows:
        wid = w.get("run_id") or w.get("window")
        if w.get("legacy"):
            out["legacy"].append(wid)
            if w.get("legacy") == TUNED_META_LEGACY:                   # a quarantined tuned-meta play: still counted as tuned
                out["tuned_meta_quarantined"].append(wid)
                out["tuned_meta_plays"].append(wid)
            continue
        ver = w.get("basis_version")
        m = norm(w.get("meta") or {})
        if ver is None:
            out["no_basis_record"].append(wid)
        elif ver == 0 or w.get("untrained_basis"):
            out["untrained"].append(wid)
            if neutral is None or norm(w.get("prior_cfg")) != neutral:
                out["untrained_cfg_mismatch"].append(wid)
            on_default = meta_default is not None and not any(k in m and m[k] != norm(meta_default[k]) for k in meta_default)
            on_neutral = nm is not None and not any(k in m and m[k] != nm[k] for k in nm)
            if on_neutral:
                out["untrained_on_neutral_meta"].append(wid)
            elif on_default:
                if nm is not None and tuned_meta_keys(m, meta_default, nm):
                    out["tuned_meta_plays"].append(wid)
            else:
                out["untrained_meta_mismatch"].append(wid)
        else:
            out["trained"].append(wid)
            if meta_default is not None and nm is not None and tuned_meta_keys(m, meta_default, nm, searched=tuple(_meta_space())):
                out["tuned_meta_plays"].append(wid)
            rec = by_v.get(ver)
            own = (pd.Timestamp(w["real_start"]),) if w.get("real_start") else dates.get(wid)
            if rec is None or own is None:
                out["unresolved"].append(wid)
            elif any(pd.Timestamp(b) >= own[0] for _, _, b in rec["trained_on"]):
                out["violations"].append({"window": wid, "version": ver})
    for v in lineage:
        if not v.get("used_from"):
            out["versions_without_used_from"].append(v["version"])
            continue
        late = [i for i, _, b in v["trained_on"] if pd.Timestamp(b) >= pd.Timestamp(v["used_from"])]
        if late:
            out["used_from_violations"].append({"version": v["version"], "used_from": v["used_from"], "late": late})
    # before F06 every version trained on the whole archive, so each contained the previous; a gated version trains only on
    # windows that ended before ITS first use, which may be earlier than the previous version's, so only ungated ones must nest
    legacy_ids = [{i for i, _, _ in by_v[v]["trained_on"]} for v in sorted(by_v) if not by_v[v].get("used_from")]
    out["lineage_is_monotone"] = all(a <= b for a, b in zip(legacy_ids, legacy_ids[1:]))
    return out


# ---- verdicts -----------------------------------------------------------------------------------------------------------
def _all(*checks):
    """True if every check holds, False if any fails, None if none fails but some could not be evaluated."""
    if any(c is False for c in checks):
        return False
    return None if any(c is None for c in checks) else True


def _not(v):
    return None if v is None else not v


def _unparsed(facts: dict, *keys) -> list[str]:
    """Source files a verdict needs that could not be read/parsed (then the verdict is UNMEASURED, never a guess)."""
    ok = (facts or {}).get("sources_parsed") or {}
    return [k for k in keys if not ok.get(k)]


def verdict_adjusted_prices(facts: dict, proofs: dict, measured: dict | None) -> Verdict:
    """Channel 2. FIXED only if the default blind path provably reaches the split-invariant rule at every call site AND the
    planted split is neutralised by it AND the live rule really is moved by that split (else the proof could not fail)."""
    miss = _unparsed(facts, "livesim", "features", "leak_audit", "loop2")
    if miss:
        return Verdict(UNMEASURED, {f"source_{k}_parsed": False for k in miss}, tuple(f"source {k} not readable" for k in miss))
    sp = (proofs or {}).get("split_invariance", {})
    rule = facts.get("feed_tradable_rule_default")
    gate = {"feed_default_is_split_invariant": None if rule in (_UNDETERMINED, "<absent>") or "feed_tradable_rule_default" not in facts else rule == "split_invariant",
            "precompute_passes_feed_rule": facts.get("precompute_passes_feed_rule"),
            "parity_passes_feed_rule": facts.get("parity_passes_feed_rule"),
            "features_branch_calls_split_invariant_tradable": facts.get("features_split_branch_uses_split_invariant_tradable"),
            "hardened_feed_does_not_override_rule": _not(facts.get("hardened_class_overrides_tradable_rule")),
            "worker_reaches_feed_through_livesim_run": facts.get("loop2_worker_runs_livesim_run")}
    proof = {"planted_split_neutralised": sp.get("mechanism_neutralises_planted_leak"), "planted_split_moves_the_live_rule": sp.get("planted_leak_is_real")}
    checks = {**gate, **proof, "swap_effect_measured_on_real_caches": bool((measured or {}).get("tradable_rule_swap_effect"))}
    failed = [k for k, v in gate.items() if v is False]
    if failed:
        return Verdict(LEAK, checks, tuple(f"default path: {k} is False" for k in failed))
    if proof["planted_split_neutralised"] is False:
        return Verdict(LEAK, checks, ("the split-invariant rule changed a tradable cell under a planted split",))
    if _all(*gate.values(), *proof.values()) is not True:
        return Verdict(UNMEASURED, checks, tuple(k for k, v in {**gate, **proof}.items() if v is None or v is False))
    return Verdict(FIXED, checks)


def verdict_feed_shape(facts: dict, causality: dict | None) -> Verdict:
    """Channel 8c. FIXED iff the default path builds the hardened feed AND on real windows the hardened feed shows none of
    the three exposures AND the plain feed shows at least one (a control: the instrument can see what it claims to close)."""
    miss = _unparsed(facts, "livesim", "loop2")
    if miss:
        return Verdict(UNMEASURED, {f"source_{k}_parsed": False for k in miss}, tuple(f"source {k} not readable" for k in miss))
    gate = {"blind_feed_hardened_by_default": _all(*(None if facts.get(k) in (_UNDETERMINED, "<absent>") or k not in facts else facts[k] is True
                                                 for k in ("blind_feed_hardened_default", "run_hardened_default"))),
            "blind_feed_class_returns_hardened": facts.get("blind_feed_returns_hardened_class"),
            "run_uses_blind_feed_class": facts.get("run_builds_feed_via_blind_feed_class"),
            "run_never_builds_plain_feed": _not(facts.get("run_builds_plain_feed_directly")),
            "worker_does_not_opt_out": facts.get("loop2_worker_passes_hardened")}
    fe = (causality or {}).get("feed_exposure_real_windows") or {}
    rows = [v for v in fe.values() if v.get("hardened") and v.get("plain")]
    hard_clean = (all(abs(r["hardened"]["spy_first_level"] - 100.0) < 1e-6 and abs(r["hardened"]["column_order_vs_real_alpha_rho"]) < 0.3
                      and r["hardened"]["columns_shown_before_listing"] == 0
                      and (r.get("hardened_rerun_linkability") or {}).get("share_reidentified_by_column_position", 1.0) < 0.05 for r in rows) if rows else None)
    plain_sees = (any(r["plain"]["column_order_vs_real_alpha_rho"] > 0.9 or r["plain"]["columns_shown_before_listing"] > 0
                      or abs(r["plain"]["spy_first_level"] - 100.0) > 1e-6 for r in rows) if rows else None)
    checks = {**gate, "hardened_feed_shows_none_of_the_three_exposures_on_real_windows": hard_clean,
              "plain_feed_shows_them_control": plain_sees, "n_real_windows": len(rows)}
    failed = [k for k, v in gate.items() if v is False]
    if failed or hard_clean is False:
        return Verdict(LEAK, checks, tuple(failed) or ("the hardened feed still shows an exposure on a real window",))
    if _all(*gate.values(), hard_clean, plain_sees) is not True:
        return Verdict(UNMEASURED, checks, tuple(k for k, v in checks.items() if v is None or v is False))
    return Verdict(FIXED, checks)


def verdict_learned_state(facts: dict, proofs: dict, state_check: dict | None) -> Verdict:
    """Channel 4. Two gates are proven. (1) Between training and play: every play call carries a per-window basis, plan_round
    takes the lineage's past-only basis or the neutral start, the registered training set equals the trained set, and the real
    state file shows no violation. (2) F06, the training call itself: every train_basis call in main names the first real day
    the basis may be played (`used_from`), train_basis refuses (LateTrainingWindow) any window that had not ended by then, main
    splits the archive at that date before the call, and the search starts from a past-only incumbent (never the loop's latest
    global basis); a planted late window must be refused. LEAK when any of that fails.
    The meta residual: META_DEFAULT was tuned on real outcomes (sensitivity study). If untrained plays still use it the verdict is
    capped at QUARANTINED; if the loop plays the data-free NEUTRAL_META, the cap remains only while the state file still holds
    plays that ran on the tuned meta (they are quarantined by the loop, but their lessons stay in the memory bank)."""
    miss = _unparsed(facts, "loop2", "adaptive")
    if miss:
        return Verdict(UNMEASURED, {f"source_{k}_parsed": False for k in miss}, tuple(f"source {k} not readable" for k in miss))
    lg = (proofs or {}).get("lineage_gate", {})
    tg = (proofs or {}).get("training_gate", {})
    calls = facts.get("loop2_train_basis_calls") or []
    gate = {"main_plans_each_round_from_lineage": _all(facts.get("loop2_main_plans_each_round"), facts.get("loop2_plan_round_uses_lineage_basis_for")),
            "unseen_start_falls_back_to_neutral_cfg": facts.get("loop2_plan_round_falls_back_to_neutral"),
            "every_play_call_carries_a_per_window_basis": None if facts.get("loop2_play_calls_without_per_id") is None else (not facts["loop2_play_calls_without_per_id"] and facts.get("loop2_n_run_workers_calls", 0) > 0),
            "global_basis_reaches_no_play_call": None if facts.get("loop2_global_basis_reaches_a_play_call") is None else not facts["loop2_global_basis_reaches_a_play_call"],
            "registered_training_set_equals_trained_set": facts.get("loop2_registers_exactly_the_training_set"),
            "start_cfg_is_data_free": _all(facts.get("loop2_neutral_cfg_from_neutral_default_cfg"), facts.get("loop2_fresh_state_starts_neutral")),
            "legacy_windows_excluded_from_headline": facts.get("loop2_headline_excludes_legacy"),
            "every_training_call_names_its_first_use": facts.get("loop2_every_train_call_passes_used_from"),
            "train_basis_refuses_windows_not_ended_by_first_use": facts.get("loop2_train_basis_refuses_late_windows"),
            "main_splits_the_archive_at_first_use_before_training": facts.get("loop2_main_splits_training_set"),
            "training_incumbent_is_past_only": _not(facts.get("loop2_train_call_uses_global_basis"))}
    proof = {"gated_basis_never_trained_on_unended_window": lg.get("mechanism_holds"), "naive_latest_basis_is_touched_control": lg.get("planted_leak_is_real"),
             "gated_training_call_refuses_a_planted_late_window": tg.get("planted_late_window_refused"),
             "gated_training_call_keeps_the_window_that_ended_the_day_before": tg.get("boundary_day_before_kept"),
             "gated_before_use_design_has_no_violation": tg.get("mechanism_holds")}
    sc = state_check
    real = {"no_trained_play_uses_a_basis_with_unended_training_window": None if sc is None else not sc["violations"],
            "untrained_plays_used_exactly_the_neutral_cfg": None if sc is None else not sc["untrained_cfg_mismatch"],
            "untrained_plays_used_the_neutral_or_default_meta": None if sc is None else not sc["untrained_meta_mismatch"],
            "every_played_window_has_a_basis_record": None if sc is None else not sc["no_basis_record"],
            "trained_plays_all_resolvable": None if sc is None else not sc["unresolved"],
            "lineage_monotone_each_ungated_version_contains_the_previous": None if sc is None else sc["lineage_is_monotone"],
            "gated_versions_trained_only_before_their_first_use": None if sc is None else not sc.get("used_from_violations", [])}
    meta_free = _all(facts.get("loop2_neutral_meta_from_neutral_default_meta"), facts.get("loop2_plan_round_untrained_meta_is_neutral"))
    tuned_plays = None if sc is None else list(sc.get("tuned_meta_plays", []))
    checks = {**gate, **proof, **real, "state_file_read": sc is not None, "train_basis_calls": calls,
              "train_basis_passes_as_of": any(c["passes_as_of"] for c in calls), "meta_default_tuned_on_real_outcomes": facts.get("meta_default_data_tuned"),
              "untrained_meta_is_data_free": meta_free, "loop_quarantines_tuned_meta_plays": facts.get("loop2_quarantines_tuned_meta_plays"),
              "plays_on_tuned_meta": tuned_plays, "plays_on_tuned_meta_quarantined": None if sc is None else sc.get("tuned_meta_quarantined")}
    reasons = [f"gate: {k}" for k, v in gate.items() if v is False]
    reasons += [f"proof: {k}" for k, v in proof.items() if k.startswith("gated") and v is False]
    reasons += [f"state: {k}" for k, v in real.items() if v is False]
    if sc is not None and sc["violations"]:
        reasons.append(f"state: plays whose basis saw a later window: {[v['window'] for v in sc['violations']]}")
    if reasons:
        return Verdict(LEAK, checks, tuple(reasons))
    if _all(*gate.values(), *proof.values(), *real.values()) is not True:
        return Verdict(UNMEASURED, checks, tuple(k for k, v in {**gate, **proof, **real}.items() if v is None or v is False))
    if facts.get("meta_default_data_tuned"):
        if meta_free is not True:
            return Verdict(QUARANTINED, checks, ("META_DEFAULT (used by every untrained play) was set from the sensitivity study on real outcomes: "
                                                 "a data-tuned starting point with no data-free replacement; results must carry this label",))
        if tuned_plays:
            q = "quarantined by the loop (legacy='tuned_meta', out of the headline)" if facts.get("loop2_quarantines_tuned_meta_plays") else "NOT marked by the loop"
            return Verdict(QUARANTINED, checks, (f"{len(tuned_plays)} play(s) in the state file ran on the sensitivity-tuned META_DEFAULT before the loop switched to the "
                                                 f"data-free NEUTRAL_META: {q}; their lessons remain in the memory bank, so results must carry this label",))
    return Verdict(FIXED, checks)


def compute_verdicts(parts: dict, facts: dict, proofs: dict, state_check: dict | None) -> dict[str, Verdict]:
    """All channel verdicts. `parts` = the measurement parts (static, runtime, survivorship, adjusted, metadata, fingerprint,
    causality; missing/empty = not measured). Every channel whose measurement is absent is UNMEASURED."""
    out: dict[str, Verdict] = {}
    S, R, V, J, M, F, Cz = (parts.get(k) or {} for k in ("static", "runtime", "survivorship", "adjusted", "metadata", "fingerprint", "causality"))
    unmeasured = lambda *miss: Verdict(UNMEASURED, {f"part_{m}_present": False for m in miss}, tuple(f"part {m} missing" for m in miss))

    if not V:                       # 1: absent (delisted) names cannot be recovered offline; measured => QUARANTINED with the size reported
        out["1"] = unmeasured("survivorship")
    else:
        alive, exits = sum((V.get("alive_by_year") or {}).values()), sum((V.get("exits_by_year") or {}).values())
        floor = (V.get("hazard_bands") or {}).get("literature_low", 0.03)
        rate = exits / alive if alive else None                    # observed share of name-years that end in an exit
        checks = {"observed_annual_exit_rate_in_panel": rate, "literature_low_annual_delisting_hazard": floor,
                  "dead_names_injected_in_default_path": facts.get("livesim_injects_dead_names")}
        if rate is None:
            out["1"] = Verdict(UNMEASURED, checks, ("no name-years in the survivorship part",))
        else:
            out["1"] = Verdict(CLEAN if rate >= floor else QUARANTINED, checks, () if rate >= floor else (
                "the panel is survivors: its observed exit rate is far below the literature delisting hazard, so delisted names are absent (size measured, not removed)",))
    out["2"] = verdict_adjusted_prices(facts, proofs, J or None)
    if not M:
        out["3"] = unmeasured("metadata")
    else:
        carried = bool(M.get("hard_listed_crypto_tickers_traded_before_2018") or M.get("crypto_matches_in_panel"))
        out["3"] = Verdict(QUARANTINED if carried else CLEAN, {"crypto_tickers_in_panel": len(M.get("crypto_matches_in_panel") or []),
                                                               "filter_inert_on_blind_codes": M.get("not_crypto_on_blind_codes_filters_nothing"), "sic_is_todays_classification": True},
                           ("the panel holds today's listings, crypto-named names and today's SIC codes; the trader never sees a name",))
    out["4"] = verdict_learned_state(facts, proofs, state_check)
    if not S:
        out["5"] = unmeasured("static")
    else:
        reach = sorted({"analogs", "parity"} & set(S.get("closure_modules", [])))
        out["5"] = Verdict(LEAK if reach else CLEAN, {"revision_prone_consumers_reachable_from_blind_path": reach}, tuple(f"{m} reachable" for m in reach))
    if not F:
        out["6"] = unmeasured("fingerprint")
    else:
        exp, hid = F.get("exposed_6y_warmup_plus_window", {}), F.get("hidden_12_months_only", {})
        ident = lambda d: sorted(g for g, r in d.items() if isinstance(r, dict) and r.get("verdict") == "identifiable")
        ctrl = [r.get("skill") for r in (F.get("exposed_6y_warmup_plus_window__shuffled_control") or {}).values() if isinstance(r, dict) and r.get("skill") is not None]
        ctrl_ok = bool(ctrl) and max(ctrl) < 0.15
        trader_open = [v for v, d in (("exposed", exp), ("hidden", hid)) if (d.get("trader_inputs") or {}).get("verdict") == "identifiable"]
        checks = {"identifiable_groups_exposed_window": ident(exp), "identifiable_groups_hidden_only": ident(hid), "trader_inputs_identifiable_in": trader_open,
                  "shuffled_label_control_near_zero": ctrl_ok}
        if not ctrl_ok:
            out["6"] = Verdict(UNMEASURED, checks, ("the shuffled-label control is not near zero: the probe cannot be trusted",))
        elif trader_open:
            out["6"] = Verdict(LEAK, checks, (f"the trader's own inputs identify the year in the {trader_open} view",))
        elif ident(exp) or ident(hid):
            out["6"] = Verdict(QUARANTINED, checks, ("levels/calendar identify the year in the exposed frames but the trader consumes only ranks, ratios and m_* context",))
        else:
            out["6"] = Verdict(CLEAN, checks)
        if "lookup" in parts:                     # F06: can anything keyed by the year be LOOKED UP by the trader?
            out["6"] = combine_fingerprint_and_lookup(out["6"], verdict_year_lookup(parts.get("lookup")))
        else:
            out["6"] = Verdict(out["6"].status, {**out["6"].checks, "year_lookup_part_present": False}, out["6"].reasons)
    if not R or _unparsed(facts, "loop2"):
        out["7"] = unmeasured(*(["runtime"] if not R else []), *(["loop2 source"] if _unparsed(facts, "loop2") else []))
    else:
        blocked = R.get("blocked_network_attempts") or []
        checks = {"blocked_attempts_in_a_full_window": len(blocked), "worker_installs_guard": facts.get("loop2_worker_installs_network_guard"),
                  "guard_blocks_planted_lookup": (proofs or {}).get("network_guard", {}).get("planted_lookup_blocked")}
        if blocked or checks["worker_installs_guard"] is False or checks["guard_blocks_planted_lookup"] is False:
            out["7"] = Verdict(LEAK, checks, ("network attempted, or the worker does not install a working guard",))
        else:
            out["7"] = Verdict(CLEAN if _all(checks["worker_installs_guard"], checks["guard_blocks_planted_lookup"]) is True else UNMEASURED, checks)
    if not S:
        out["8a"] = unmeasured("static")
    else:
        ft = {k: v for k, v in (S.get("free_text_columns") or {}).items() if v}
        out["8a"] = Verdict(LEAK if ft else CLEAN, {"free_text_columns": ft or None}, tuple(f"prose columns in {k}" for k in ft))
    ti = Cz.get("features_truncation_invariance_2012_sample")
    if not ti:
        out["8b"] = unmeasured("causality")
    else:
        leaky = sorted(ti.get("leaky_features") or {})
        out["8b"] = Verdict(CLEAN if ti.get("clean") and not leaky else LEAK,
                            {"leaky_features": leaky, "n_features": ti.get("n_features"),
                             "embargo_covers_label_reach": (S.get("model_embargo_sessions") or 0) >= (S.get("label_reach_sessions") or 99)},
                            tuple(f"look-ahead in {k}" for k in leaky))
    out["8c"] = verdict_feed_shape(facts, Cz or None)
    if not S or not R:
        out["8d"] = unmeasured(*[n for n, p in (("static", S), ("runtime", R)) if not p])
    else:
        research, sealed_open, cache = S.get("research_only_modules_reachable") or [], R.get("livesim_state_files_opened") or [], R.get("data_cache_files_opened") or []
        checks = {"research_only_modules_reachable": research, "sealed_state_files_opened_in_a_window": sealed_open, "cache_files_opened_in_a_window": cache}
        out["8d"] = Verdict(LEAK if (research or sealed_open) else QUARANTINED if cache else CLEAN, checks,
                            tuple(research) + tuple(sealed_open) if (research or sealed_open)
                            else ("the trader side reads today's universe.csv (inert on code names)",) if cache else ())
    e = out["8b"]                    # 8e (universe filters / labels) rests on the same truncation proof as 8b
    out["8e"] = Verdict(e.status, {"derived_from": "8b", **e.checks}, e.reasons)
    return out


# =====================================================================================================================
# F06 leak closure (C69 ledger W-10 / W-11; canon C55, C56, C58, C64; C66 sections 30-31)
#   4a  the training CALL is gated: a basis search never sees a window that had not ended before the first real day the
#       basis may be played on (the loop seals the next round first, so that day is known on the referee side)
#   4b  the untrained start is data-free in the adaptation meta too (neutral_default_meta), and plays that ran on the
#       outcome-tuned META_DEFAULT are named, so the loop can quarantine them
#   6   a computed part asks whether anything that can be LOOKED UP by the year (curator releases, research records filed
#       by year, the memory bank) reaches the trader in a disguised replay
# =====================================================================================================================
TUNED_META_LEGACY = "tuned_meta"          # the loop marks a play that ran on the sensitivity-tuned META_DEFAULT with legacy=this


class LateTrainingWindow(FirewallBreach):
    """A basis search was handed a window whose REAL end is on/after the first real day the resulting basis may be played
    (canon C56): training on it would put the played window's own period, or its future, inside the basis."""


def _span_or_none(span: Callable, wid) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    try:
        a, b = span(wid)
        return pd.Timestamp(a), pd.Timestamp(b)
    except Exception:                      # noqa: BLE001 - an unresolvable real span is refused, never assumed past
        return None


def split_training_windows(windows: Sequence, used_from, span: Callable, key: str = "id") -> tuple[list, list[dict]]:
    """(kept, refused). `used_from` is the first real day the basis being trained may be played; `span(id)` -> (real start, real
    end) is the referee's lookup (the loop passes livesim_loop2.real_span). A window is kept only if its real end is STRICTLY
    before `used_from`; a window ending on that day, after it, or whose span cannot be resolved is refused with the reason.
    `windows` may be loaded windows (mappings with `key`) or plain ids. No first-use date is itself a refusal (fail closed)."""
    if used_from is None:
        raise LateTrainingWindow("no first-use date: a basis must know the earliest real day it may be played before it is trained")
    uf = pd.Timestamp(used_from)
    kept, refused = [], []
    for w in windows:
        wid = w[key] if isinstance(w, Mapping) else w
        sp = _span_or_none(span, wid)
        if sp is None:
            refused.append({"id": wid, "real_end": None, "why": "real span cannot be resolved"})
        elif sp[1] >= uf:
            refused.append({"id": wid, "real_end": str(sp[1].date()), "why": f"ends on/after the first use {uf.date()}"})
        else:
            kept.append(w)
    return kept, refused


def refuse_late_training(windows: Sequence, used_from, span: Callable, key: str = "id") -> dict:
    """The check train_basis runs on its own input: raise LateTrainingWindow if ANY window had not ended before `used_from`
    (the caller was supposed to split first; a refusal here means the split was skipped or wrong). Returns a small receipt."""
    kept, refused = split_training_windows(windows, used_from, span, key)
    if refused:
        raise LateTrainingWindow(f"{len(refused)} of {len(refused) + len(kept)} training windows had not ended before the first use "
                                 f"{pd.Timestamp(used_from).date()}: {refused[:3]}")
    ends = [_span_or_none(span, w[key] if isinstance(w, Mapping) else w)[1] for w in kept]
    return {"n_windows": len(kept), "used_from": str(pd.Timestamp(used_from).date()),
            "max_real_end": str(max(ends).date()) if ends else None}


def neutral_default_meta(space: Mapping, base: Mapping, knobs: Sequence) -> dict:
    """A data-free adaptation meta (the meta twin of neutral_default_cfg). Choice, documented:
      - every knob the basis search can move (`space`, basis_search.META_SPACE) sits at its middle grid value - chosen by
        position, never by an outcome (the sensitivity-tuned values, e.g. ic_beta 0.5, min_weeks 4, are discarded);
      - `adaptive_knobs` (which cfg knobs the adapter may move; tuned by the sensitivity study to 6 of them) becomes every knob
        the adapter has a step grid for (`knobs`, adaptive.STEPS) - a statement about the machinery, not about any result;
      - everything else is copied from `base`: the Phase-18 rails (off), the dial (off), mem_use_dates (off in blind windows),
        mem_score_errors and ic_clip are structural switches and bounds whose values encode no outcome.
    The result is a fresh dict; `base` is not modified."""
    out = {**{k: v for k, v in dict(base).items()}, **neutral_default_cfg(dict(space))}
    out["adaptive_knobs"] = [k for k in knobs]
    return out


def _meta_space() -> dict:
    from .basis_search import META_SPACE              # lazy: basis_search is not on the blind path
    return dict(META_SPACE)


def default_neutral_meta(meta_default: Mapping) -> dict:
    """neutral_default_meta over the repository's own search space and step grid (what livesim_loop2.NEUTRAL_META is)."""
    from .adaptive import STEPS                        # lazy: adaptive imports policy/memory
    return neutral_default_meta(_meta_space(), meta_default, list(STEPS))


def tuned_meta_keys(meta: Mapping | None, meta_default: Mapping, neutral: Mapping, searched: Sequence = ()) -> list[str]:
    """Keys on which `meta` still carries the outcome-tuned default: equal to META_DEFAULT's value where that differs from the
    data-free neutral value. Keys a past-only basis search may have chosen (`searched`) are excluded - for a trained basis the
    search, not the study, set them. An empty list means the play did not run on tuned defaults."""
    norm = lambda x: json.loads(json.dumps(x, default=str))
    m, d, n = norm(dict(meta or {})), norm(dict(meta_default)), norm(dict(neutral))
    skip = set(searched)
    return sorted(k for k in d if k in m and k in n and k not in skip and m[k] == d[k] and d[k] != n[k])


def _loop2_gate_facts(tree, main, plan, fresh) -> dict:
    """F06 facts read from the loop's source: is the training call gated at the first use, and is the untrained meta data-free?"""
    f: dict = {}
    tb_def = _find_def(tree, "train_basis")
    tb = _calls(main, "train_basis") if main is not None else []
    used = lambda c: _has_kw(c, "used_from") and not (isinstance(_kw(c, "used_from"), ast.Constant) and _kw(c, "used_from").value is None)
    f["loop2_every_train_call_passes_used_from"] = None if main is None else (bool(tb) and all(used(c) for c in tb))
    f["loop2_train_call_uses_global_basis"] = None if main is None else any(
        _state_ref(a) for c in tb for a in list(c.args[1:3]) + [k.value for k in c.keywords if k.arg in ("cfg", "meta")])
    if tb_def is None:
        f["loop2_train_basis_refuses_late_windows"] = None
    else:
        params = {a.arg for a in tb_def.args.args + tb_def.args.kwonlyargs}
        refs = _calls(tb_def, "refuse_late_training")
        f["loop2_train_basis_refuses_late_windows"] = "used_from" in params and any(
            any(isinstance(n, ast.Name) and n.id == "used_from" for a in list(r.args) + [k.value for k in r.keywords] for n in ast.walk(a)) for r in refs)
    if main is None or not tb or not tb[0].args:
        f["loop2_main_splits_training_set"] = None if main is None else False
    else:
        name = ast.unparse(tb[0].args[0])
        splits = [n for n in ast.walk(main) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                  and _callee(n.value) == "split_training_windows"
                  and any(isinstance(t, ast.Name) and t.id == name for tgt in n.targets for t in ast.walk(tgt))]
        f["loop2_main_splits_training_set"] = bool(splits) and all(min(s.lineno for s in splits) < c.lineno for c in tb)
    nm = _module_assign(tree, "NEUTRAL_META")
    f["loop2_neutral_meta_from_neutral_default_meta"] = bool(nm is not None and isinstance(nm.value, ast.Call) and _callee(nm.value) == "neutral_default_meta")
    names = lambda node: {n.id if isinstance(n, ast.Name) else n.attr for n in ast.walk(node) if isinstance(n, (ast.Name, ast.Attribute))} if node is not None else set()
    f["loop2_plan_round_untrained_meta_is_neutral"] = None if plan is None else ("NEUTRAL_META" in names(plan) and "META_DEFAULT" not in names(plan))
    st_assign = _module_assign(tree, "st")
    f["loop2_quarantines_tuned_meta_plays"] = bool(st_assign is not None and _calls(st_assign, "quarantine_tuned_meta"))
    f["loop2_learner_default"] = _learner_default(tree)
    return f


def _learner_default(tree) -> str | None:
    """What `--learner` is when the flag is absent: the first constant `learner_flag` returns (None if it cannot be read)."""
    fn = _find_def(tree, "learner_flag")
    if fn is None:
        return None
    for n in ast.walk(fn):
        if isinstance(n, ast.Return) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
            return n.value.value
    return None


def prove_training_gate(n_rounds: int = 14, par: int = 3, seeds=(0, 1, 2, 3)) -> dict:
    """Two proofs of the F06 gate. (1) Planted: a window ending after the first-use day, one ending ON it and one whose span is
    unknown must all be refused by refuse_late_training; one ending the day before must be kept (the boundary, so a gate that
    refuses everything fails). (2) The design, on the loop's seeded draw process: after round r, the NEXT round is sealed, its
    earliest real start is the first-use day, the archive is split there, the basis is registered with exactly that set, and
    round r+1 plays BasisLineage.basis_for. No play may use a basis whose training touched its window; each newly trained basis
    must be eligible for every window of the round it was trained for."""
    T = pd.Timestamp
    table = {"old": (T("1990-01-01"), T("1990-12-31")), "edge": (T("2000-01-01"), T("2000-12-31")),
             "late": (T("2000-06-01"), T("2001-05-31")), "on": (T("2000-01-02"), T("2001-01-01"))}
    span = lambda i: table[i]
    uf = T("2001-01-01")
    try:
        refuse_late_training(["old", "late"], uf, span)
        planted_refused = False
    except LateTrainingWindow:
        planted_refused = True
    kept, refused = split_training_windows(["old", "edge", "on", "ghost"], uf, span)
    boundary_kept = kept == ["old", "edge"]
    on_and_unknown_refused = sorted(r["id"] for r in refused) == ["ghost", "on"]
    try:
        split_training_windows(["old"], None, span)
        no_date_refused = False
    except LateTrainingWindow:
        no_date_refused = True
    violations, untrained, ineligible, played_n, trained_versions = 0, [], 0, 0, 0
    for seed in seeds:
        rounds = simulate_window_draws(n_rounds, par=par, seed=20260929 + seed)
        lin, seen, played, ver = BasisLineage(), [], [], 0
        spans = {w["id"]: (w["start"], w["end"]) for rnd in rounds for w in rnd}
        for r, rnd in enumerate(rounds):
            for w in rnd:
                rec = lin.basis_for(w["start"])
                played.append({"id": w["id"], "real_start": w["start"], "version": rec["version"] if rec else 0})
                untrained.append(rec is None)
            seen += rnd
            if r + 1 >= len(rounds):
                break
            first_use = min(w["start"] for w in rounds[r + 1])
            k, _ = split_training_windows([w["id"] for w in seen], first_use, spans.__getitem__)
            if not k:
                continue
            refuse_late_training(k, first_use, spans.__getitem__)
            ver += 1
            trained_versions += 1
            lin.register(ver, {}, {}, [TrainedOn(i, *spans[i]) for i in k])
            ineligible += sum(lin.basis_for(w["start"]) is None or lin.basis_for(w["start"])["version"] != ver for w in rounds[r + 1])
        violations += len(lin.violations(played))
        played_n += len(played)
    return {"planted_late_window_refused": planted_refused, "boundary_day_before_kept": boundary_kept,
            "window_ending_on_first_use_and_unknown_span_refused": on_and_unknown_refused, "missing_first_use_refused": no_date_refused,
            "gated_plays_checked": played_n, "gated_violations": violations, "trained_versions": trained_versions,
            "new_basis_not_eligible_for_its_round": ineligible,
            "share_of_plays_on_untrained_neutral_basis": float(np.mean(untrained)) if untrained else float("nan"),
            "mechanism_holds": violations == 0 and ineligible == 0 and planted_refused and boundary_kept and on_and_unknown_refused and no_date_refused}


# ---- channel 6: what can be LOOKED UP by year --------------------------------------------------------------------------
HINDSIGHT_KEY_TOKENS = frozenset({"knowability", "hindsight", "unpredictable", "unknown_cause", "external_cause", "could_have_known",
                                  "known_only_after", "post_event", "after_event", "counterfactual", "what_changed", "autopsy",
                                  "error_class", "episode_path", "path_class", "regret"})


def hindsight_feature_keys(features: Mapping) -> list[str]:
    """Feature names that carry a hindsight label (knowability / counterfactual / what_changed / error-record vocabulary). A
    trader item's features must be observable at the decision; a name built from these words is a conclusion reached AFTER the
    outcome. Tokens are split on non-letters so `r5` or `vol20` never match."""
    out = []
    for k in features:
        low = str(k).lower()
        toks = set(t for t in re.split(r"[^a-z]+", low) if t)
        if toks & HINDSIGHT_KEY_TOKENS or any(t in low for t in HINDSIGHT_KEY_TOKENS if "_" in t):
            out.append(str(k))
    return out


def _research_loop_release_passes_replay(root=None) -> bool | None:
    """Does the research loop's release stage hand the firewall a ReplayContext? (engine/research/loop.py st_release)."""
    p = (Path(root) if root else K.ROOT) / "engine" / "research" / "loop.py"
    if not p.exists():
        return None
    try:
        tree = ast.parse(p.read_text(encoding="utf-8"))
    except SyntaxError:
        return None
    fn = _find_def(tree, "st_release")
    if fn is None:
        return None
    calls = [c for c in ast.walk(fn) if isinstance(c, ast.Call) and _callee(c) in ("run_day", "release")]
    return bool(calls) and all(_has_kw(c, "replay") or len(c.args) >= 4 for c in calls)


def _curator_writers(root=None) -> list[str]:
    """Engine modules (outside the curator itself) that file into a curator: the only doors into the year-keyed store."""
    base = (Path(root) if root else K.ROOT) / "engine"
    pat = re.compile(r"curator\.file\(|\bfile_batch\(")
    out = []
    for p in sorted(base.rglob("*.py")):
        if p.name == "curator.py" and p.parent.name == "learning":
            continue
        try:
            if pat.search(p.read_text(encoding="utf-8")):
                out.append(str(p.relative_to(base.parent)).replace("\\", "/"))
        except (OSError, UnicodeDecodeError):
            continue
    return out


def memory_bank_lookup_probe() -> dict:
    """The memory bank recalls lessons by market context (the fingerprint): a planted row from a window that ended after the
    replayed window began must be caught, a clean bank must pass."""
    from . import blind_gates as BG
    late = pd.DataFrame({"real_end": ["2009-12-31", "2011-06-30"], "arm": 1, "ctx": 0, "outcome": 0.0})
    clean = late.iloc[:1]
    return {"planted_late_row_caught": bool(BG.check_memory_bank_causality(late, "2011-03-01")),
            "clean_bank_passes": not BG.check_memory_bank_causality(clean, "2011-03-01")}


def verdict_year_lookup(part: dict | None) -> Verdict:
    """LEAK if anything keyed by the year reaches the trader in a disguised replay: an unmatured or after-window curator memory
    released, a rerun whose release on some day depends on a memory that had not matured by that day, planted date content accepted, a research record filed under the
    replayed year (or a hindsight label) released, the memory bank's late row missed, or the trader closure reaching the curator
    or engine.research. UNMEASURED if a part or a control is missing or the control fails (an instrument that cannot see). CLEAN
    otherwise. Latent items (the curator accepts a hindsight-named feature; the research loop passes no ReplayContext) are
    reported in the checks and reasons but do not move the status: no writer or replay reaches them on the blind path today."""
    if not part:
        return Verdict(UNMEASURED, {"part_lookup_present": False}, ("part lookup missing",))
    c, r, m, s = (part.get(k) or {} for k in ("curator", "research", "memory_bank", "static"))
    tri = lambda v: None if v is None else bool(v)
    fails = {"curator_never_released_an_unmatured_memory": None if c.get("unmatured_releases") is None else c["unmatured_releases"] == 0,
             "curator_never_released_a_memory_maturing_after_the_window": _not(c.get("after_window_released")),
             "curator_late_same_year_memory_released_only_after_maturity": tri(c.get("replayed_year_late_first_release_after_maturity")),
             "curator_rerun_release_uses_only_memories_matured_before_each_day": None if c.get("rerun_prefix_mismatches") is None else c["rerun_prefix_mismatches"] == 0,
             "planted_year_or_date_content_refused": tri(c.get("planted_year_feature_refused")),
             "research_record_filed_under_replayed_year_refused": tri(r.get("same_year_refused_during_replay")),
             "research_hindsight_label_refused": tri(r.get("hindsight_caught")),
             "research_planted_suite_passed": tri(r.get("suite_passed")),
             "memory_bank_planted_late_row_caught": tri(m.get("planted_late_row_caught")),
             "trader_closure_reaches_no_curator_or_research": tri(s.get("trader_closure_clean"))}
    controls = {"control_curator_releases_a_past_memory": tri(c.get("past_control_released")),
                "control_prefix_check_fails_a_curator_that_ignores_maturity": None if c.get("control_curator_ignoring_maturity_mismatches") is None
                else c["control_curator_ignoring_maturity_mismatches"] > 0,
                "control_rerun_instrument_sees_a_planted_extra_memory": tri(c.get("planted_extra_changes_releases")),
                "control_same_record_admitted_outside_the_replayed_year": tri(r.get("admitted_outside_replay")),
                "control_research_suite_clean_record_released": None if r.get("suite_void") is None else not r["suite_void"],
                "control_memory_bank_clean_bank_passes": tri(m.get("clean_bank_passes"))}
    latent = {"latent_curator_refuses_hindsight_feature_keys": c.get("hindsight_feature_refused"),
              "latent_research_loop_release_passes_replay_context": s.get("research_loop_release_passes_replay"),
              "curator_writers": s.get("curator_writers"), "curator_on_default_blind_path": None if s.get("loop2_learner_default") is None
              else s["loop2_learner_default"] != "off"}
    checks = {**fails, **controls, **latent}
    notes = []
    if latent["latent_curator_refuses_hindsight_feature_keys"] is False:
        notes.append(f"latent: the curator accepts a hindsight-named feature key (writers today: {latent['curator_writers']}; they file learner features only)")
    if latent["latent_research_loop_release_passes_replay_context"] is False:
        notes.append("latent: engine/research/loop.py st_release calls firewall.run_day without a ReplayContext, so the same-year rule is inert there "
                     "(the research loop is not on the blind path)")
    failed = [k for k, v in fails.items() if v is False]
    if failed:
        return Verdict(LEAK, checks, tuple(f"year lookup reaches the trader: {k}" for k in failed) + tuple(notes))
    unknown = [k for k, v in {**fails, **controls}.items() if v is None or v is False]
    if unknown:
        return Verdict(UNMEASURED, checks, tuple(f"lookup instrument: {k}" for k in unknown) + tuple(notes))
    return Verdict(CLEAN, checks, tuple(notes))


def combine_fingerprint_and_lookup(fp: Verdict, lk: Verdict) -> Verdict:
    """Channel 6 = the fingerprint probe AND the lookup part. A lookup LEAK is a leak whatever the probe says; an unmeasured
    lookup cannot improve the probe's status; a CLEAN lookup leaves the probe's status as it is (identifiable-but-unusable market
    context is the owner's ruling, never this function's) and says so in the reasons."""
    checks = {**fp.checks, "year_lookup_part_present": "part_lookup_present" not in lk.checks, "year_lookup_status": lk.status,
              **{f"lookup.{k}": v for k, v in lk.checks.items()}}
    if lk.status == LEAK:
        return Verdict(LEAK, checks, lk.reasons + fp.reasons)
    if lk.status == UNMEASURED:
        return Verdict(worst_status(fp.status, UNMEASURED) if fp.status != LEAK else LEAK, checks, fp.reasons + lk.reasons)
    extra = ("no year-keyed lookup reaches the trader (curator releases, research records filed by year, memory bank: all past-only and "
             "same-year-refused, planted cases caught); the fingerprint is identifiable but nothing on the trader side can use it - "
             "keeping LEAK vs QUARANTINED is the owner's ruling",) if fp.status == LEAK else ()
    return Verdict(fp.status, checks, fp.reasons + extra + lk.reasons)
