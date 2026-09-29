"""Bible Phase 23 (canons C11, C19): the re-tester. Replays an archived blind window and demands it reproduces the
original run within 0.5%.

Order of business, and it matters:
  1. Compare provenance code_hash FIRST. If the archived run and the replay were produced by different code, the
     verdict is STALE_CODE - a statement about the code, not a parity failure (see engine/provenance.py; T9, 2026-09-28).
     Nothing else is compared, because a divergence would be uninterpretable.
  2. Compare seven components: holdings, trades, scores, weekly returns, adaptation events, pattern activation,
     memory state. Numbers must agree to <= 0.5% (relative, with an absolute floor for values near zero);
     categorical content (who was held, which events fired, which patterns were active) must match exactly.
  3. Any divergence not covered by an explicit, itemised explanation is a FAIL. An explanation waives one named
     divergence; it never widens the tolerance.

An archive is a plain dict (JSON-safe) so it can live in state/; DataFrames are stored as records."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

TOL = 0.005
ABS_FLOOR = 1e-9
COMPONENTS = ("holdings", "trades", "scores", "weekly_returns", "adaptation_events", "pattern_activation",
              "memory_state")
PASS, FAIL, STALE = "PASS", "FAIL", "STALE_CODE"


@dataclass
class Divergence:
    component: str
    key: str
    detail: str
    magnitude: float = float("nan")
    explained: str | None = None


@dataclass
class Verdict:
    status: str
    tol: float = TOL
    divergences: list = field(default_factory=list)
    compared: dict = field(default_factory=dict)      # component -> number of items compared
    note: str = ""

    @property
    def ok(self):
        return self.status == PASS

    def unexplained(self):
        return [d for d in self.divergences if d.explained is None]

    def summary(self):
        u = self.unexplained()
        head = f"{self.status}: {sum(self.compared.values())} items across {len(self.compared)} components"
        if self.note:
            head += f" ({self.note})"
        return head + (f"; {len(u)} unexplained divergence(s), first: {u[0].component}/{u[0].key} {u[0].detail}" if u else "")


# ------------------------------------------------------------------ provenance
def compare_provenance(orig, replay, current_hash=None, disk=False):
    """Returns (status, note). STALE when the two stamps disagree on code_hash, either is missing it, either records
    that files were edited after its process started (code_mixed), or - with `disk=True` - the archive's code files no
    longer hash to what it recorded (engine.provenance.stale). A missing hash is never treated as a match.
    `current_hash` (legacy) additionally demands the archived hash equal a given one."""
    a, b = (orig or {}).get("code_hash"), (replay or {}).get("code_hash")
    if a is None or b is None:
        return STALE, "code_hash missing from " + ("archive" if a is None else "replay") + "; cannot show same code"
    for who, rec in (("archive", orig), ("replay", replay)):
        if rec.get("code_mixed"):
            return STALE, f"{who} ran while {len(rec['code_mixed'])} of its code files were edited: {rec['code_mixed'][:3]}"
    if a != b:
        return STALE, f"archive code_hash {a} != replay {b}: stale code, not a parity failure"
    if current_hash is not None and a != current_hash:
        return STALE, f"both runs used {a} but the code on disk is now {current_hash}"
    if disk:
        from . import provenance
        if provenance.stale(orig):
            return STALE, "the code files recorded in the archive have changed on disk since the run"
    return PASS, ""


# ------------------------------------------------------------------ numeric primitives
def rel_diff(a, b):
    """|a-b| relative to the larger magnitude, with an absolute floor so 0 vs 1e-12 is not 100%."""
    a, b = float(a), float(b)
    if np.isnan(a) and np.isnan(b):
        return 0.0
    if np.isnan(a) or np.isnan(b):
        return float("inf")
    return abs(a - b) / max(abs(a), abs(b), ABS_FLOOR) if max(abs(a), abs(b)) > ABS_FLOOR else 0.0


def _frame(x):
    if isinstance(x, pd.DataFrame):
        return x
    if x is None or len(x) == 0:
        return pd.DataFrame()
    return pd.DataFrame(x)


def _cmp_numeric_map(component, a, b, tol, out):
    """a, b: {key: number}. Missing keys on either side are divergences."""
    n = 0
    for k in sorted(set(a) | set(b), key=str):
        if k not in a or k not in b:
            out.append(Divergence(component, str(k), "present in only " + ("replay" if k in b else "archive")))
            continue
        n += 1
        r = rel_diff(a[k], b[k])
        if r > tol:
            out.append(Divergence(component, str(k), f"{a[k]:.6g} vs {b[k]:.6g}", r))
    return n


# ------------------------------------------------------------------ components
def cmp_holdings(a, b, tol=TOL):
    """a, b: DataFrame [date, ticker, weight] (or shares). Same names each date; weights within tol."""
    out, fa, fb = [], _frame(a), _frame(b)
    if fa.empty and fb.empty:
        return 0, out
    col = "weight" if "weight" in fa else "shares"
    ka = {(str(r.date), r.ticker): getattr(r, col) for r in fa.itertuples()}
    kb = {(str(r.date), r.ticker): getattr(r, col) for r in fb.itertuples()}
    return _cmp_numeric_map("holdings", ka, kb, tol, out), out


def cmp_trades(a, b, tol=TOL):
    """DataFrame [date, ticker, side, qty, price]. Same number, same (date, ticker, side); qty and price within tol."""
    out, fa, fb = [], _frame(a), _frame(b)
    if fa.empty and fb.empty:
        return 0, out
    if len(fa) != len(fb):
        out.append(Divergence("trades", "count", f"{len(fa)} vs {len(fb)}", rel_diff(len(fa), len(fb))))
    key = ["date", "ticker", "side"]
    for f in (fa, fb):
        f["_n"] = f.groupby(key).cumcount() if len(f) else 0
    ka, kb = key + ["_n"], key + ["_n"]
    m = fa.merge(fb, on=ka, how="outer", suffixes=("_a", "_b"), indicator=True)
    n = 0
    for _, r in m.iterrows():
        tag = f"{r['date']}/{r['ticker']}/{r['side']}"
        if r["_merge"] != "both":
            out.append(Divergence("trades", tag, "trade in only " + ("archive" if r["_merge"] == "left_only" else "replay")))
            continue
        for c in ("qty", "price"):
            n += 1
            d = rel_diff(r[f"{c}_a"], r[f"{c}_b"])
            if d > tol:
                out.append(Divergence("trades", f"{tag}.{c}", f"{r[c + '_a']:.6g} vs {r[c + '_b']:.6g}", d))
    return n, out


def cmp_scores(a, b, tol=TOL):
    """DataFrame [date, ticker, score]. Values within tol, and the top-ranked name per date must be unchanged."""
    out, fa, fb = [], _frame(a), _frame(b)
    if fa.empty and fb.empty:
        return 0, out
    ka = {(str(r.date), r.ticker): r.score for r in fa.itertuples()}
    kb = {(str(r.date), r.ticker): r.score for r in fb.itertuples()}
    n = _cmp_numeric_map("scores", ka, kb, tol, out)
    for d in sorted(set(fa["date"].astype(str)) & set(fb["date"].astype(str))):
        ta = fa[fa["date"].astype(str) == d].nlargest(1, "score")["ticker"].tolist()
        tb = fb[fb["date"].astype(str) == d].nlargest(1, "score")["ticker"].tolist()
        if ta != tb:
            out.append(Divergence("scores", f"{d}.top", f"{ta} vs {tb}"))
    return n, out


def cmp_weekly_returns(a, b, tol=TOL):
    """Series or dict {week: return}. Relative tol with the absolute floor of 1e-4 (one basis point) - a week that
    returned 0.0001 vs 0.0002 is noise, not a divergence, but 0.07 vs 0.0705 is right at the line."""
    out = []
    sa, sb = dict(pd.Series(a).items()), dict(pd.Series(b).items())
    n = 0
    for k in sorted(set(sa) | set(sb), key=str):
        if k not in sa or k not in sb:
            out.append(Divergence("weekly_returns", str(k), "week in only " + ("replay" if k in sb else "archive")))
            continue
        n += 1
        x, y = float(sa[k]), float(sb[k])
        if abs(x - y) > 1e-4 and rel_diff(x, y) > tol:
            out.append(Divergence("weekly_returns", str(k), f"{x:.6f} vs {y:.6f}", rel_diff(x, y)))
    return n, out


def cmp_events(component, a, b, tol=TOL):
    """List of dicts (adaptation events / pattern activations). Compared as ordered sequences of their categorical
    fields; numeric fields (any float) within tol."""
    out, la, lb = [], list(a or []), list(b or [])
    if len(la) != len(lb):
        out.append(Divergence(component, "count", f"{len(la)} vs {len(lb)}", rel_diff(len(la), len(lb))))
    n = 0
    for i, (x, y) in enumerate(zip(la, lb)):
        for k in sorted(set(x) | set(y)):
            n += 1
            va, vb = x.get(k), y.get(k)
            if isinstance(va, float) and isinstance(vb, float):
                if rel_diff(va, vb) > tol:
                    out.append(Divergence(component, f"#{i}.{k}", f"{va:.6g} vs {vb:.6g}", rel_diff(va, vb)))
            elif va != vb:
                out.append(Divergence(component, f"#{i}.{k}", f"{va!r} vs {vb!r}"))
    return n, out


def cmp_pattern_activation(a, b, tol=TOL):
    """{pattern_key: {"status": str, "effect": float}} (final state) - status exact, effect within tol."""
    out = []
    a, b = a or {}, b or {}
    n = 0
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            out.append(Divergence("pattern_activation", k, "pattern in only " + ("replay" if k in b else "archive")))
            continue
        n += 1
        if a[k].get("status") != b[k].get("status"):
            out.append(Divergence("pattern_activation", k, f"status {a[k].get('status')} vs {b[k].get('status')}"))
        if "effect" in a[k] and "effect" in b[k] and rel_diff(a[k]["effect"], b[k]["effect"]) > tol:
            out.append(Divergence("pattern_activation", k + ".effect", f"{a[k]['effect']:.6g} vs {b[k]['effect']:.6g}",
                                  rel_diff(a[k]["effect"], b[k]["effect"])))
    return n, out


def _flatten(x, prefix=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _flatten(v, f"{prefix}{k}.")
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from _flatten(v, f"{prefix}{i}.")
    else:
        yield prefix.rstrip("."), x


def cmp_memory(a, b, tol=TOL, component="memory_state"):
    """Nested dict/list memory state, leaf by leaf: floats within tol, everything else exact."""
    out = []
    fa, fb = dict(_flatten(a or {})), dict(_flatten(b or {}))
    n = 0
    for k in sorted(set(fa) | set(fb)):
        if k not in fa or k not in fb:
            out.append(Divergence(component, k, "leaf in only " + ("replay" if k in fb else "archive")))
            continue
        n += 1
        va, vb = fa[k], fb[k]
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and not isinstance(va, bool):
            if rel_diff(va, vb) > tol:
                out.append(Divergence(component, k, f"{va:.6g} vs {vb:.6g}", rel_diff(va, vb)))
        elif va != vb:
            out.append(Divergence(component, k, f"{va!r} vs {vb!r}"))
    return n, out


# ------------------------------------------------------------------ top level
def compare_runs(archive, replay, tol=TOL, explanations=(), current_hash=None, components=COMPONENTS, disk=False):
    """archive/replay: dicts with 'provenance' and the component keys. `components` names what to compare (default
    all seven; an archive that only kept some may compare the subset - the verdict lists exactly what was compared,
    and "summary" (a flat dict of headline numbers, leaf-by-leaf) is also accepted). `explanations` is a list of
    {"component": ..., "key": ..., "reason": ...}; each waives exactly the divergence with that component and key."""
    status, note = compare_provenance(archive.get("provenance"), replay.get("provenance"), current_hash, disk)
    if status == STALE:
        return Verdict(STALE, tol, note=note)
    v = Verdict(PASS, tol)
    fns = {"holdings": cmp_holdings, "trades": cmp_trades, "scores": cmp_scores,
           "weekly_returns": cmp_weekly_returns, "pattern_activation": cmp_pattern_activation,
           "memory_state": cmp_memory, "summary": lambda a, b, t: cmp_memory(a, b, t, "summary"),
           "adaptation_events": lambda a, b, t: cmp_events("adaptation_events", a, b, t)}
    for c in components:
        if c not in archive or c not in replay:
            v.divergences.append(Divergence(c, "*", "component missing from " + ("replay" if c in archive else "archive")))
            v.compared[c] = 0
            continue
        n, d = fns[c](archive[c], replay[c], tol)
        v.compared[c] = n
        v.divergences += d
    waive = {(e["component"], e["key"]): e.get("reason") for e in explanations if e.get("reason")}
    for d in v.divergences:
        d.explained = waive.get((d.component, d.key))
    if v.unexplained():
        v.status = FAIL
    elif sum(v.compared.values()) == 0:
        v.status, v.note = FAIL, "nothing was compared (both archives empty): a vacuous pass is not a pass"
    return v


def _default(o):
    if isinstance(o, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(o).date())
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, pd.DataFrame):
        return o.to_dict("records")
    if isinstance(o, pd.Series):
        return {str(k): v for k, v in o.items()}
    raise TypeError(type(o))


def save_archive(path, archive):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(archive, default=_default, sort_keys=True))


def load_archive(path):
    a = json.loads(Path(path).read_text())
    for k in ("holdings", "trades", "scores"):
        if k in a:
            a[k] = pd.DataFrame(a[k])
    return a


def retest(archive_path, runner, tol=TOL, explanations=(), current_hash=None):
    """Replay one archived window. `runner(archive) -> replay dict` must re-run from the SEALED window and seed only
    (it receives the archive so it can read window/seed/config, and must not read the archived results). The archive
    is loaded before the run and its results are hidden from the runner by construction."""
    arch = load_archive(archive_path)
    blind = {k: arch[k] for k in ("window", "seed", "config") if k in arch}
    rep = runner(blind)
    for k in ("holdings", "trades", "scores"):
        if k in rep and not isinstance(rep[k], pd.DataFrame):
            rep[k] = pd.DataFrame(rep[k])
    return compare_runs(arch, rep, tol, explanations, current_hash)


def retest_many(items, runner, tol=TOL):
    """items: {name: archive_path}. Returns ({name: Verdict}, summary). A crashed runner is a FAIL with the exception
    recorded - never a skip."""
    res = {}
    for name, p in items.items():
        try:
            res[name] = retest(p, runner, tol)
        except Exception as e:   # noqa: BLE001 - the whole point is to record any failure of the replay itself
            res[name] = Verdict(FAIL, tol, note=f"runner raised {type(e).__name__}: {e}")
    tally = {s: sum(v.status == s for v in res.values()) for s in (PASS, FAIL, STALE)}
    return res, tally


# ------------------------------------------------------------------ diagnostics
def divergence_table(verdict):
    """One row per divergence: component, key, detail, magnitude, explained. Sorted worst first (inf, then size)."""
    rows = [{"component": d.component, "key": d.key, "detail": d.detail,
             "magnitude": d.magnitude, "explained": d.explained} for d in verdict.divergences]
    df = pd.DataFrame(rows, columns=["component", "key", "detail", "magnitude", "explained"])
    if len(df):
        df = df.assign(_m=df["magnitude"].fillna(np.inf)).sort_values("_m", ascending=False).drop(columns="_m")
    return df.reset_index(drop=True)


def worst_by_component(verdict):
    """{component: (n_divergences, worst_magnitude)}; components with none report (0, 0.0)."""
    out = {c: (0, 0.0) for c in verdict.compared}
    for d in verdict.divergences:
        n, w = out.get(d.component, (0, 0.0))
        m = np.inf if d.magnitude != d.magnitude else d.magnitude
        out[d.component] = (n + 1, max(w, m))
    return out


def tolerance_sweep(archive, replay, tols=(0.0, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05)):
    """How the verdict depends on the tolerance. A replay that passes at 0.5% but fails at 0.05% is a near-miss the
    owner should know about; one that fails only below 1e-6 is float noise. Returns [(tol, status, n_unexplained)].
    Provenance is compared once up front (a STALE pair is STALE at every tolerance)."""
    rows = []
    for t in tols:
        v = compare_runs(archive, replay, tol=t)
        rows.append((t, v.status, len(v.unexplained())))
    return rows


def noise_floor(archive, replay):
    """Smallest tolerance in a fixed ladder at which the pair passes, or None when it never does. 0.0 means the
    replay was bit-exact on every compared number."""
    for t, st, _ in tolerance_sweep(archive, replay, (0.0, 1e-9, 1e-6, 1e-4, 1e-3, 0.005, 0.01, 0.05, 0.2)):
        if st == PASS:
            return t
    return None


def per_week_divergence(a_weekly, b_weekly):
    """Weekly-return gap per week, in return points: the series the owner reads to see WHEN a replay drifted."""
    sa, sb = pd.Series(a_weekly, dtype=float), pd.Series(b_weekly, dtype=float)
    idx = sa.index.union(sb.index)
    d = (sb.reindex(idx) - sa.reindex(idx))
    return d


def first_divergence_week(a_weekly, b_weekly, eps=1e-6):
    """First week (in order) at which the two return series differ by more than eps, or None. Everything after a
    first divergence is contaminated (a different portfolio feeds every later week), so this is the root cause."""
    d = per_week_divergence(a_weekly, b_weekly)
    bad = d[(d.abs() > eps) | d.isna()]
    return None if bad.empty else bad.index[0]


def cumulative_gap(a_weekly, b_weekly):
    """Compounded return of each series and the gap between them (replay - archive), in return points."""
    ca = float((1 + pd.Series(a_weekly, dtype=float)).prod() - 1)
    cb = float((1 + pd.Series(b_weekly, dtype=float)).prod() - 1)
    return {"archive": ca, "replay": cb, "gap": cb - ca}


def turnover(holdings):
    """Mean one-way turnover between consecutive dates of a [date, ticker, weight] frame; used to tell a replay that
    holds the right names but trades a different amount apart from one that matches."""
    h = _frame(holdings)
    if h.empty:
        return 0.0
    w = h.pivot_table(index="date", columns="ticker", values="weight", aggfunc="sum").fillna(0.0).sort_index()
    return float(w.diff().abs().sum(axis=1).iloc[1:].mean() / 2) if len(w) > 1 else 0.0


def type_breakdown(archive, replay, tol=TOL):
    """Per-component pass/fail with counts, for the report table."""
    v = compare_runs(archive, replay, tol)
    rows = []
    for c in COMPONENTS:
        n_div = sum(1 for d in v.divergences if d.component == c and d.explained is None)
        rows.append({"component": c, "compared": v.compared.get(c, 0), "unexplained": n_div,
                     "status": "n/a" if v.status == STALE else ("FAIL" if n_div else "PASS")})
    return pd.DataFrame(rows)


def era_breakdown(verdicts_by_window, start_of):
    """{window: Verdict} -> per-era counts of PASS/FAIL/STALE. `start_of[window]` is the window's real start date.
    A re-tester that only fails in one era points at era-specific code (the 40 bps cost tier, missing history)."""
    from .blind_gates import era_of
    rows = {}
    for w, v in verdicts_by_window.items():
        e = era_of(start_of[w])
        r = rows.setdefault(e, {PASS: 0, FAIL: 0, STALE: 0})
        r[v.status] += 1
    return rows


def markdown_report(name, verdict, archive=None, replay=None):
    """Human-readable report for one replay: verdict, per-component table, worst divergences, drift week."""
    lines = [f"# Re-test {name}: {verdict.status}", "", verdict.summary(), ""]
    if verdict.status != STALE:
        lines += ["| component | compared | worst divergence |", "|---|---|---|"]
        wc = worst_by_component(verdict)
        for c in COMPONENTS:
            n, w = wc.get(c, (0, 0.0))
            lines.append(f"| {c} | {verdict.compared.get(c, 0)} | {n} divergence(s), worst {w:.4g} |")
        t = divergence_table(verdict).head(10)
        if len(t):
            lines += ["", "## Worst divergences", ""] + [f"- {r.component}/{r.key}: {r.detail}" + (f" (explained: {r.explained})" if r.explained else "")
                                                        for r in t.itertuples()]
        if archive is not None and replay is not None and "weekly_returns" in archive and "weekly_returns" in replay:
            fw = first_divergence_week(archive["weekly_returns"], replay["weekly_returns"])
            g = cumulative_gap(archive["weekly_returns"], replay["weekly_returns"])
            lines += ["", f"First divergent week: {fw}", f"Compounded return archive {g['archive']:+.4%} replay {g['replay']:+.4%} gap {g['gap']:+.4%}"]
    else:
        lines += ["Nothing was compared: the two runs came from different code. Re-run the replay on the archived code."]
    return "\n".join(lines) + "\n"


def write_reports(results, out_dir, stamp=None):
    """results: {name: Verdict}. Writes summary.json (with provenance stamp) and one markdown file per window."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summ = {"stamp": stamp, "tally": {s: sum(v.status == s for v in results.values()) for s in (PASS, FAIL, STALE)},
            "windows": {n: {"status": v.status, "summary": v.summary(), "compared": v.compared} for n, v in results.items()}}
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    for n, v in results.items():
        (out / f"{n}.md").write_text(markdown_report(n, v))
    return summ


# ------------------------------------------------------------------ archived live runs (state/livesim/r*/result.json)
HEADLINE = ("year_return", "mean_week", "weeks_ge_7", "max_dd")


def judge_archived(res, replay):
    """Verdict for one archived live run against its replay (a dict with the HEADLINE keys and optionally
    weekly_returns). Provenance first: the replay always runs the code on disk now, so it is given the hash of the
    archive's own code_files as they are on disk now; a difference (or files edited during the run) is STALE_CODE.
    A run archived without a code stamp returns stamped=False - it is compared, but its code cannot be shown to be the
    code on disk, so callers must not count it as verified. Returns (stamped, Verdict)."""
    from . import provenance
    prov = res.get("provenance")
    stamped = isinstance(prov, dict) and "code_files" in prov and "code_hash" in prov
    arch = {"provenance": prov if stamped else {"code_hash": "unstamped"}, "summary": {k: res["diagnosis"][k] for k in HEADLINE}}
    rep = {"provenance": ({"code_hash": provenance.code_hash(prov["code_files"]), "code_files": prov["code_files"], "code_mixed": []}
                          if stamped else {"code_hash": "unstamped"}),
           "summary": {k: replay[k] for k in HEADLINE}}
    comps = ["summary"]
    if res.get("weekly_returns") and replay.get("weekly_returns"):
        arch["weekly_returns"], rep["weekly_returns"] = res["weekly_returns"], replay["weekly_returns"]
        comps.append("weekly_returns")
    return stamped, compare_runs(arch, rep, components=tuple(comps))
