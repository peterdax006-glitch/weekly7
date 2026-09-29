"""Bible Phase 4 - pattern lifecycle state machine (canon C43: never hold a failed pattern).

States: candidate -> rejected | duplicate | no_gain | active;  active <-> watch;  active|watch|rescoped -> failed ->
cause_search -> rescoped | discarded;  rescoped -> active;  discarded -> candidate | cause_search (new evidence only).
Every move goes through one guarded function and is written to a transition log with the reason and the numbers behind it.
Nothing may silently remain active after failure (`failed`/`cause_search` are transient: `invariants()` reports any record
left in them) and nothing silently disappears (a pattern the miner no longer finds goes to `watch`, never vanishes).

Operates on PatternMiner.patterns frames (`ingest`) and on raw panels (`review`): failure detector, context terciles,
rescoping search, long-run / discovery / confirmation / recent / sign-consistency tests, and an explicit discard reason.
No look-ahead: a panel only contains rows whose forward-return window closed on or before `as_of`."""
import copy
import hashlib

import numpy as np
import pandas as pd
from scipy import stats as sps

from .patterns import CTX

STATES = ("candidate", "rejected", "duplicate", "no_gain", "active", "watch", "failed", "cause_search", "rescoped",
          "discarded")
ALLOWED = {
    None: {"candidate"},
    "candidate": {"rejected", "duplicate", "no_gain", "active"},
    "rejected": {"candidate"}, "duplicate": {"candidate"}, "no_gain": {"candidate"},
    "active": {"failed", "watch"},
    "watch": {"active", "failed"},
    "failed": {"cause_search"},
    "cause_search": {"rescoped", "discarded"},
    "rescoped": {"active", "failed", "watch"},
    "discarded": {"candidate", "cause_search"},
}
TRADEABLE = {"active": 1.0, "rescoped": 1.0, "watch": 0.0}     # watch is flagged, not traded (C43)
PANEL_MIN_DATES = 40

LIFE_DEFAULT = {"disc_frac": 0.7, "recent_frac": 0.2, "min_obs": 3, "min_recent_weeks": 12, "min_recent_scope": 8,
                "fail_t": 1.5, "watch_ratio": 0.25, "max_watch": 3, "t_long": 2.0, "t_half": 0.5, "t_recent": 1.0,
                "min_scope_weeks": 30, "min_year_hit": 0.6, "min_years": 3, "min_year_weeks": 4,
                "horizon_days": 7, "revive": True}
TESTS = ("long_run", "discovery", "confirmation", "recent", "sign_consistency")


class LifecycleError(ValueError):
    pass


# ---------------------------------------------------------------- identity
def pattern_id(names):
    return hashlib.sha1("|".join(map(str, names)).encode()).hexdigest()[:12]


def parse_key_named(s):
    """'f0 q4 & f1 q2 unless f3 q0' -> ('u', 'f0', 4, 'f1', 2, 'f3', 0); the inverse of PatternMiner._name."""
    base, _, un = str(s).partition(" unless ")
    parts = base.split(" & ")
    lits = []
    for p in parts + ([un] if un else []):
        f, sep, q = p.strip().rpartition(" q")
        if not sep or not f or not q.isdigit() or int(q) > 4:
            raise ValueError(f"unparseable pattern name {s!r}")
        lits += [f, int(q)]
    if un:
        return ("u", *lits) if len(parts) == 2 else _bad(s)
    return ("s", *lits) if len(parts) == 1 else ("p", *lits) if len(parts) == 2 else _bad(s)


def _bad(s):
    raise ValueError(f"unparseable pattern name {s!r}")


def _iso(t):
    return pd.Timestamp(t).isoformat()


def _num(x):
    x = float(x)
    return x if np.isfinite(x) else None


# ---------------------------------------------------------------- panel
class Panel:
    """Quintile codes, date-demeaned forward returns and per-date market context, cut so that nothing after `as_of`
    (or whose outcome window closes after it) is present."""

    def __init__(self, dates, dcode, Q, feats, y, ctx, as_of, params):
        self.dates, self.dcode, self.Q, self.feats, self.y, self.ctx, self.as_of = dates, dcode, Q, feats, y, ctx, as_of
        nd = len(dates)
        idx = np.arange(nd)
        self.disc = idx < int(nd * params["disc_frac"])
        self.conf = ~self.disc
        self.recent = idx >= int(nd * (1 - params["recent_frac"]))
        self.years = dates.year.values if nd else np.array([], int)
        self.p = params

    @classmethod
    def build(cls, X, y, as_of, params=None, demean=True):
        p = {**LIFE_DEFAULT, **(params or {})}
        if not X.index.equals(y.index):
            raise LifecycleError("X and y must share the same (date, ticker) index")
        feats = [c for c in X.columns if not c.startswith("m_")]
        if not len(X):
            e = pd.DatetimeIndex([])
            return cls(e, np.array([], int), np.empty((0, len(feats)), np.int8), feats, np.array([]),
                       pd.DataFrame(index=e), as_of, p)
        d = X.index.get_level_values(0)
        cutoff = pd.Timestamp(as_of) - pd.Timedelta(days=p["horizon_days"])
        keep = (d <= cutoff) & y.notna().values
        X, y = X[keep], y[keep].astype(float)
        feats = [c for c in X.columns if not c.startswith("m_")]
        if not len(X):
            e = pd.DatetimeIndex([])
            return cls(e, np.array([], int), np.empty((0, len(feats)), np.int8), feats, np.array([]),
                       pd.DataFrame(index=e), as_of, p)
        codes, dates = pd.factorize(X.index.get_level_values(0), sort=True)
        dates = pd.DatetimeIndex(dates)
        Q = np.minimum((X[feats].groupby(level=0).rank(pct=True).fillna(0.5).values * 5).astype(np.int8), 4)
        yv = (y - y.groupby(level=0).transform("mean")).values if demean else y.values
        mcols = [c for c in X.columns if c.startswith("m_")]
        ctx = X[mcols].groupby(level=0).first().reindex(dates) if mcols else pd.DataFrame(index=dates)
        return cls(dates, codes, Q, feats, yv, ctx, as_of, p)

    @property
    def enough(self):
        return len(self.dates) >= PANEL_MIN_DATES

    def mask(self, names):
        ix = {c: i for i, c in enumerate(self.feats)}
        try:
            n = tuple(names)
            if n[0] == "s":
                return self.Q[:, ix[n[1]]] == int(n[2])
            if n[0] == "p":
                return (self.Q[:, ix[n[1]]] == int(n[2])) & (self.Q[:, ix[n[3]]] == int(n[4]))
            if n[0] == "u":
                return ((self.Q[:, ix[n[1]]] == int(n[2])) & (self.Q[:, ix[n[3]]] == int(n[4]))
                        & (self.Q[:, ix[n[5]]] != int(n[6])))
        except KeyError:
            return None
        return None

    def means(self, names):
        """Per-date mean outcome of the pattern's stocks (NaN when fewer than min_obs matched) - the unit of evidence is
        the week, because stocks in a week move together."""
        m = self.mask(names)
        if m is None:
            return None
        nd = len(self.dates)
        cnt = np.bincount(self.dcode[m], minlength=nd)
        sm = np.bincount(self.dcode[m], weights=self.y[m], minlength=nd)
        out = np.full(nd, np.nan)
        ok = cnt >= self.p["min_obs"]
        out[ok] = sm[ok] / cnt[ok]
        return out

    def scope_dates(self, scope):
        """Boolean over dates: is the market context inside the scope's tercile band."""
        if scope is None:
            return np.ones(len(self.dates), bool)
        if scope["col"] not in self.ctx:
            return None
        v = self.ctx[scope["col"]].values.astype(float)
        with np.errstate(invalid="ignore"):
            if scope["label"] == "low":
                return v <= scope["lo"]
            if scope["label"] == "high":
                return v > scope["hi"]
            return (v > scope["lo"]) & (v <= scope["hi"])


def wstat(means, sel):
    """Week-clustered mean, t-statistic and n of the selected dates."""
    v = means[sel & np.isfinite(means)]
    n = len(v)
    if n < 2:
        return {"m": float(v.mean()) if n else 0.0, "t": 0.0, "n": n}
    sd = v.std(ddof=1)
    m = float(v.mean())
    t = m / (sd / np.sqrt(n)) if sd > 1e-12 else (0.0 if abs(m) < 1e-12 else float(np.sign(m)) * 99.0)
    return {"m": m, "t": float(np.clip(t, -99, 99)), "n": n}


def _p_two_sided(t, n):
    return float(2 * sps.t.sf(abs(t), max(n - 1, 1)))


# ---------------------------------------------------------------- tests and detectors
def run_tests(means, panel, sign, sel=None, params=None):
    """The five tests an improved form must survive (C43). `sign` is the pattern's identity: +1 or -1."""
    p = {**LIFE_DEFAULT, **(params or {})}
    sel = np.ones(len(means), bool) if sel is None else sel
    st = {"long_run": wstat(means, sel), "discovery": wstat(means, sel & panel.disc),
          "confirmation": wstat(means, sel & panel.conf), "recent": wstat(means, sel & panel.recent)}
    res = {"long_run": sign * st["long_run"]["t"] >= p["t_long"],
           "discovery": sign * st["discovery"]["t"] >= p["t_half"],
           "confirmation": sign * st["confirmation"]["t"] >= p["t_half"],
           "recent": st["recent"]["n"] >= p["min_recent_scope"] and sign * st["recent"]["t"] >= p["t_recent"]}
    hits, blocks = 0, 0
    for yr in np.unique(panel.years[sel & np.isfinite(means)]) if len(means) else []:
        v = means[sel & (panel.years == yr) & np.isfinite(means)]
        if len(v) >= p["min_year_weeks"]:
            blocks += 1
            hits += int(sign * v.mean() > 0)
    st["years"] = {"blocks": blocks, "hits": hits}
    res["sign_consistency"] = blocks >= p["min_years"] and hits / blocks >= p["min_year_hit"]
    return {"pass": all(res.values()), "tests": res, "stats": st, "failed": [k for k in TESTS if not res[k]]}


def detect_failure(means, panel, sign, sel=None, params=None):
    """Failure detector. failed: recent stretch contradicts (|t| >= fail_t, opposite sign) or the whole history has
    reversed. watch: recent evidence weak or far below the long-run effect. Too little recent data is 'insufficient',
    never a silent pass."""
    p = {**LIFE_DEFAULT, **(params or {})}
    sel = np.ones(len(means), bool) if sel is None else sel
    lng, rec = wstat(means, sel), wstat(means, sel & panel.recent)
    out = {"long": lng, "recent": rec, "verdict": "ok", "reason": ""}
    if rec["n"] < p["min_recent_weeks"]:
        return {**out, "verdict": "insufficient", "reason": f"only {rec['n']} recent weeks"}
    if sign * lng["t"] <= -p["t_long"]:
        return {**out, "verdict": "failed", "reason": f"long-run effect reversed (t={lng['t']:.2f})"}
    if sign * rec["t"] <= -p["fail_t"]:
        return {**out, "verdict": "failed", "reason": f"recent stretch contradicts (t={rec['t']:.2f}, n={rec['n']})"}
    if sign * rec["t"] < 0 or (sign * lng["m"] > 0 and sign * rec["m"] < p["watch_ratio"] * sign * lng["m"]):
        return {**out, "verdict": "watch", "reason": "recent effect faded"}
    return out


# ---------------------------------------------------------------- lifecycle
def new_record(names, name, as_of, effect=0.0):
    return {"id": pattern_id(names), "names": [str(x) for x in names], "name": str(name), "state": None, "scope": None,
            "effect": float(effect), "first_seen": _iso(as_of), "changed": _iso(as_of), "version": 0,
            "discovery": {}, "failures": [], "rescopes": [], "last_validation": None, "discard_reason": None,
            "watch_count": 0}


class Lifecycle:
    """Registry of pattern records plus the transition log. Records are plain dicts so the bank can persist them."""

    def __init__(self, records=None, log=None, params=None):
        self.p = {**LIFE_DEFAULT, **(params or {})}
        self.records = {r["id"]: copy.deepcopy(r) for r in (records or [])}
        self.log = [dict(e) for e in (log or [])]

    # ----- the one guarded door
    def _move(self, rec, to, as_of, reason, evidence=None):
        frm = rec["state"]
        if to not in ALLOWED.get(frm, ()):
            raise LifecycleError(f"illegal transition {frm} -> {to} for {rec['name']!r}")
        rec["state"], rec["changed"] = to, _iso(as_of)
        rec["version"] += 1
        self.log.append({"seq": len(self.log), "kind": "transition", "as_of": _iso(as_of), "id": rec["id"],
                         "name": rec["name"], "frm": frm, "to": to, "reason": reason, "evidence": evidence or {}})

    def _note(self, rec, as_of, reason, evidence=None):
        self.log.append({"seq": len(self.log), "kind": "note", "as_of": _iso(as_of), "id": rec["id"] if rec else "*",
                         "name": rec["name"] if rec else "*", "frm": rec["state"] if rec else None,
                         "to": rec["state"] if rec else None, "reason": reason, "evidence": evidence or {}})

    def _walk(self, rec, path, as_of, reason, evidence=None):
        for s in path:
            self._move(rec, s, as_of, reason, evidence)

    # ----- ingest a PatternMiner.patterns frame
    def ingest(self, frame, as_of, run_id=None):
        """Fold one miner run into the registry. Returns counts. Records absent from the frame are moved to watch."""
        seen, skipped = set(), 0
        if frame is not None and len(frame):
            for row in frame.itertuples(index=False):
                try:
                    names = parse_key_named(row.key_named)
                except ValueError as e:
                    skipped += 1
                    self._note(None, as_of, f"skipped row: {e}")
                    continue
                rec = self.records.get(pattern_id(names)) or self._create(names, row.key_named, as_of)
                seen.add(rec["id"])
                self._apply_miner_row(rec, row, as_of)
        for rid, rec in self.records.items():
            if rid not in seen and rec["state"] in ("active", "rescoped"):
                self._move(rec, "watch", as_of, "not re-found by the miner in this run")
                rec["watch_count"] += 1
        return {"seen": len(seen), "skipped": skipped, **self.counts()}

    def _create(self, names, name, as_of):
        rec = new_record(names, name, as_of)
        self.records[rec["id"]] = rec
        self._move(rec, "candidate", as_of, "first seen")
        return rec

    @staticmethod
    def _miner_scope(row):
        sc = getattr(row, "scope", None)
        if sc is None or (isinstance(sc, float) and np.isnan(sc)):
            return None
        ci, lab, lo, hi = sc
        return {"col": CTX[int(ci)], "label": lab, "lo": float(lo), "hi": float(hi)}

    def _apply_miner_row(self, rec, row, as_of):
        ev = {k: _num(getattr(row, k)) for k in ("effect", "t_disc", "t_conf", "p_real", "p_hallucinated", "p_coincidence")
              if hasattr(row, k)}
        rec["effect"] = float(ev.get("effect") or rec["effect"])
        rec["discovery"] = {**ev, "as_of": _iso(as_of)}
        status, cur = row.status, rec["state"]
        if status in ("rejected", "duplicate", "no_gain"):
            if cur == "candidate" or (cur in ("rejected", "duplicate", "no_gain") and cur != status):
                self._walk(rec, (["candidate"] if cur != "candidate" else []) + [status], as_of, f"miner: {status}", ev)
            elif cur in ("active", "watch", "rescoped"):
                if cur != "watch":
                    self._move(rec, "watch", as_of, f"miner no longer confirms it ({status})", ev)
                rec["watch_count"] += 1
            return
        if status == "active":
            if cur in ("candidate", "rejected", "duplicate", "no_gain", "discarded"):
                self._walk(rec, (["candidate"] if cur != "candidate" else []) + ["active"], as_of,
                           "miner: passed every gate", ev)
            elif cur in ("watch", "rescoped"):
                self._move(rec, "active", as_of, "miner reconfirmed the unscoped form", ev)
            rec["scope"], rec["watch_count"] = None, 0
            return
        scope = self._miner_scope(row) if status == "rescoped" else None
        if status not in ("rescoped", "discarded"):
            self._note(rec, as_of, f"unknown miner status {status!r}; left untouched")
            return
        if cur == "discarded" and status == "discarded":
            return
        if cur == "rescoped" and status == "rescoped":
            if scope != rec["scope"]:
                rec["scope"] = scope
                rec["rescopes"].append({"as_of": _iso(as_of), "scope": scope, "source": "miner"})
                self._note(rec, as_of, "scope changed by miner", {"scope": scope})
            return
        if cur in ("rejected", "duplicate", "no_gain"):
            path = ["candidate", "active", "failed", "cause_search"]
        elif cur == "candidate":
            path = ["active", "failed", "cause_search"]
        elif cur == "discarded":
            path = ["cause_search"]
        else:                                           # active, watch, rescoped
            path = ["failed", "cause_search"]
        self._walk(rec, path, as_of, "miner: confirmed, then contradicted by the recent stretch", ev)
        if status == "rescoped" and scope is not None:
            rec["scope"], rec["discard_reason"] = scope, None
            rec["rescopes"].append({"as_of": _iso(as_of), "scope": scope, "source": "miner"})
            self._move(rec, "rescoped", as_of, "miner cause search found a context split", {"scope": scope})
        else:
            rec["scope"], rec["discard_reason"] = None, "miner: no context split held through the whole history"
            self._move(rec, "discarded", as_of, rec["discard_reason"], ev)

    # ----- data-driven review
    def cause_search(self, panel, rec, as_of):
        """Search context terciles for a form of the pattern that survives all five tests. Returns
        {'scope': dict|None, 'tried': [...], 'reason': str}. The adjusted p multiplies by the scopes actually tried."""
        means = panel.means(rec["names"])
        if means is None:
            return {"scope": None, "tried": [], "reason": "pattern features unavailable in this panel"}
        sign = self._sign(rec, means)
        if not len(panel.ctx.columns):
            return {"scope": None, "tried": [], "reason": "no market-context columns to split on"}
        tried, passing = [], []
        cur = rec["scope"] or {}
        for col in panel.ctx.columns:
            v = panel.ctx[col].values.astype(float)
            fin = np.isfinite(v)
            if fin.sum() < 3 * self.p["min_scope_weeks"]:
                continue
            lo, hi = (float(x) for x in np.quantile(v[fin], [1 / 3, 2 / 3]))
            for lab in ("low", "mid", "high"):
                sc = {"col": col, "label": lab, "lo": lo, "hi": hi}
                sel = panel.scope_dates(sc) & fin
                if sel.sum() < self.p["min_scope_weeks"] or (col, lab) == (cur.get("col"), cur.get("label")):
                    continue
                r = run_tests(means, panel, sign, sel, self.p)
                row = {"scope": sc, "weeks": int(sel.sum()), "t_long": r["stats"]["long_run"]["t"],
                       "failed": r["failed"]}
                tried.append(row)
                if r["pass"]:
                    passing.append((sign * r["stats"]["long_run"]["t"], row, r))
        if not tried:
            return {"scope": None, "tried": [], "reason": "no context tercile had enough weeks to test"}
        if not passing:
            near = min(tried, key=lambda r: (len(r["failed"]), -abs(r["t_long"])))
            return {"scope": None, "tried": tried,
                    "reason": f"no context split held ({len(tried)} tried; nearest {near['scope']['col']}:"
                              f"{near['scope']['label']} failed {','.join(near['failed'])})"}
        passing.sort(key=lambda x: -x[0])
        _, row, r = passing[0]
        st = r["stats"]["long_run"]
        return {"scope": row["scope"], "tried": tried, "reason": "",
                "evidence": {"t_long": st["t"], "m_long": st["m"], "weeks": row["weeks"], "n_tried": len(tried),
                             "p_adj": min(1.0, _p_two_sided(st["t"], st["n"]) * len(tried))}}

    @staticmethod
    def _sign(rec, means):
        s = np.sign(rec.get("effect") or 0.0)
        if s == 0:
            s = np.sign(np.nansum(means)) or 1.0
        return float(s)

    def _fail(self, rec, panel, as_of, reason, evidence, allow=("active", "watch", "rescoped")):
        """failed -> cause_search -> rescoped | discarded, all in one call so nothing lingers in a transient state."""
        rec["failures"].append({"as_of": _iso(as_of), "reason": reason, "evidence": evidence,
                                "scope_at_failure": rec["scope"]})
        if rec["state"] in allow:
            self._move(rec, "failed", as_of, reason, evidence)
            self._move(rec, "cause_search", as_of, "searching context terciles for the cause")
        self._resolve_search(rec, panel, as_of)

    def _resolve_search(self, rec, panel, as_of):
        found = self.cause_search(panel, rec, as_of)
        if found["scope"] is not None:
            rec["scope"] = found["scope"]
            rec["discard_reason"], rec["watch_count"] = None, 0
            rec["rescopes"].append({"as_of": _iso(as_of), "scope": found["scope"], "source": "cause_search",
                                    "evidence": found["evidence"]})
            self._move(rec, "rescoped", as_of, "improved form survives long-run, discovery, confirmation, recent and sign "
                       "tests", found["evidence"])
        else:
            rec["scope"], rec["discard_reason"] = None, found["reason"]
            self._move(rec, "discarded", as_of, found["reason"], {"tried": len(found["tried"])})

    def review(self, panel, as_of=None):
        """Retest every live pattern (active, watch, rescoped) on `panel`; retry discarded ones when new evidence might
        have arrived. Returns a summary. Never leaves a record in failed / cause_search."""
        as_of = panel.as_of if as_of is None else as_of
        summ = {"reviewed": 0, "failed": 0, "rescoped": 0, "discarded": 0, "recovered": 0, "revived": 0, "unavailable": 0,
                "insufficient": 0}
        if not panel.enough:
            self._note(None, as_of, f"panel too short to review ({len(panel.dates)} dates < {PANEL_MIN_DATES})")
            summ["insufficient"] = sum(r["state"] in ("active", "watch", "rescoped") for r in self.records.values())
            return summ
        dormant = [r for r in self.records.values() if r["state"] == "discarded"]
        for rec in [r for r in self.records.values() if r["state"] in ("active", "watch", "rescoped")]:
            means = panel.means(rec["names"])
            sel = panel.scope_dates(rec["scope"])
            if means is None or sel is None:
                summ["unavailable"] += 1
                self._note(rec, as_of, "pattern or scope columns unavailable in this panel; not judged")
                continue
            summ["reviewed"] += 1
            sign = self._sign(rec, means)
            det = detect_failure(means, panel, sign, sel, self.p)
            rec["last_validation"] = {"as_of": _iso(as_of), "verdict": det["verdict"], "m_long": _num(det["long"]["m"]),
                                      "t_long": _num(det["long"]["t"]), "n_long": det["long"]["n"],
                                      "m_recent": _num(det["recent"]["m"]), "t_recent": _num(det["recent"]["t"]),
                                      "n_recent": det["recent"]["n"], "scope": rec["scope"]}
            ev = {"long": det["long"], "recent": det["recent"]}
            if det["verdict"] == "insufficient":
                summ["insufficient"] += 1
                self._note(rec, as_of, det["reason"])
                continue
            state = rec["state"]
            if det["verdict"] == "failed":
                summ["failed"] += 1
                self._fail(rec, panel, as_of, det["reason"], ev)
                summ["rescoped" if rec["state"] == "rescoped" else "discarded"] += 1
            elif state == "rescoped":
                t = run_tests(means, panel, sign, sel, self.p)
                if t["pass"] and det["verdict"] == "ok":
                    self._move(rec, "active", as_of, "scoped form passed the retest", ev)
                else:
                    summ["failed"] += 1
                    self._fail(rec, panel, as_of, "scoped form failed retest: " + ",".join(t["failed"] or ["watch"]), ev)
                    summ["rescoped" if rec["state"] == "rescoped" else "discarded"] += 1
            elif det["verdict"] == "watch":
                rec["watch_count"] += 1
                if state == "active":
                    self._move(rec, "watch", as_of, det["reason"], ev)
                elif rec["watch_count"] > self.p["max_watch"]:
                    summ["failed"] += 1
                    self._fail(rec, panel, as_of, f"watch expired after {rec['watch_count']} reviews", ev)
                    summ["rescoped" if rec["state"] == "rescoped" else "discarded"] += 1
            elif state == "watch":
                rec["watch_count"] = 0
                summ["recovered"] += 1
                self._move(rec, "active", as_of, "recent evidence recovered", ev)
        if self.p["revive"]:
            for rec in dormant:
                self._try_revive(rec, panel, as_of, summ)
        return summ

    def _try_revive(self, rec, panel, as_of, summ):
        """A discarded pattern is never deleted: retest it, unscoped first, then by cause search."""
        means = panel.means(rec["names"])
        if means is None:
            return
        sign = self._sign(rec, means)
        t = run_tests(means, panel, sign, None, self.p)
        if t["pass"]:
            self._walk(rec, ["candidate", "active"], as_of, "discarded pattern now passes all five tests unscoped",
                       {"stats": t["stats"]})
            rec["scope"], rec["discard_reason"], rec["watch_count"] = None, None, 0
            summ["revived"] += 1
            return
        found = self.cause_search(panel, rec, as_of)
        if found["scope"] is not None:
            self._move(rec, "cause_search", as_of, "new evidence: re-searching a discarded pattern")
            self._resolve_search(rec, panel, as_of)
            summ["revived"] += 1

    # ----- reporting
    def counts(self):
        out = {s: 0 for s in STATES}
        for r in self.records.values():
            out[r["state"]] += 1
        return out

    def invariants(self):
        """Violations of the lifecycle promises; empty list means healthy."""
        bad = []
        last = {}
        for e in self.log:
            if e["kind"] == "transition":
                if e["to"] not in ALLOWED.get(e["frm"], ()):
                    bad.append(f"illegal logged move {e['frm']}->{e['to']} ({e['name']})")
                last[e["id"]] = e["to"]
        for rid, r in self.records.items():
            if r["state"] in ("failed", "cause_search"):
                bad.append(f"{r['name']} left in transient state {r['state']}")
            if last.get(rid) != r["state"]:
                bad.append(f"{r['name']} state {r['state']} not backed by the log (last logged {last.get(rid)})")
            if r["state"] == "discarded" and not r["discard_reason"]:
                bad.append(f"{r['name']} discarded without a reason")
            if r["state"] == "rescoped" and not r["scope"]:
                bad.append(f"{r['name']} rescoped without a scope")
        return bad

    def tradeable(self):
        rows = [{"id": r["id"], "name": r["name"], "scope": r["scope"], "effect": r["effect"], "state": r["state"],
                 "weight": TRADEABLE[r["state"]]} for r in self.records.values() if r["state"] in TRADEABLE]
        return pd.DataFrame(rows, columns=["id", "name", "scope", "effect", "state", "weight"])

    def log_frame(self):
        return pd.DataFrame(self.log)
