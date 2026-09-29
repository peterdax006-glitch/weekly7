"""Factor-weighted episodic memory for the self-learning system (Bible Phase 9, canon C34).

Every learned outcome ("arm" X did this well in week t, under market context c) is stored as an episode.
When the system asks "how good is arm X right now?", each episode is weighted by six factors:

  recency      0.5 ** (age / half_life)                       - old weeks fade (adapter 8 weeks, miner 4 years)
  similarity   exp(-||c - c_now||^2 / (2 * bandwidth^2))       - past weeks in similar markets count more
  shock        episodes before a detected break in the arm's results are cut by shock_cut (two-sided CUSUM)
  source       long-term episodes from EARLIER windows count prior_scale as much as this window's own
  era          episodes from another market era than today's are scaled by era_other (neutral 1.0)
  modernity    0.5 ** (years_old / modern_half_life): calendar age, independent of the week counter (neutral off)

and the estimate is shrunk toward zero by a pseudo-count (reliability: thin or noisy evidence counts little).

Beyond the estimate the module keeps what Phase 9.1/9.2 ask for:
  * lessons()/export_lessons(): one record per episode with fingerprint, context, error type, source experiment,
    coarse date, era, relevance, reliability and shock state - with every answer-identifying field stripped
    (ticker, exact outcome, exact date, test-window ids). scan_lessons() proves a bank is clean.
  * capacity/eviction: a bounded memory forgets the episodes it would weight least, deterministically.
  * state_dict()/fingerprint(): the whole memory as plain data and a hash, so a replay can be compared byte for byte.
Deterministic; uses only episodes already recorded (the caller records a week only after it has closed).
With every new parameter at its default the numbers are identical to the pre-Phase-9 memory (regression-tested)."""
import ast
import hashlib
import json
import math
import re
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

CTX = ["m_vix", "m_vix_term", "m_spy_ma200", "m_spy_ma50", "m_breadth", "m_dispersion", "m_spy_r5"]
MEM_DEFAULT = {"mem_half_life": 8.0, "mem_bandwidth": 1.5, "mem_prior_scale": 0.3, "mem_shrink": 6.0,
               "mem_shock_k": 2.5, "mem_shock_cut": 0.25}
# Phase 9 extensions. Every default is NEUTRAL so existing adapter/livesim results do not move until a study turns one on.
MEM_EXT_DEFAULT = {"mem_era_other": 1.0,          # multiplier on episodes from a different era than the current one
                   "mem_era_decay": None,         # if set (0-1): multiplier is decay ** era_gap, so neighbouring eras count more
                   "mem_modern_half_life": None,  # years; None = modernity factor off
                   "mem_kind_half_life": None,    # {arm kind: half-life}: 'knob' and 'ic' arms may fade at different speeds
                   "mem_capacity": None,          # max episodes kept (None = unbounded)
                   "mem_arm_capacity": None,      # max episodes kept per arm
                   "mem_lt_age_weeks": 52.0,      # eviction only: assumed age of a long-term episode with no date
                   "mem_cusum_slack": 0.5,        # CUSUM allowance (in standard deviations)
                   "mem_min_var": 1e-8,           # variance floor, keeps the standard error finite
                   "mem_date_grain": "quarter"}   # coarseness of the date a lesson keeps: quarter | year
MINER_HALF_LIFE_YEARS = 4.0                       # Phase 9.3: the pattern miner's memory horizon
ADAPTER_HALF_LIFE_WEEKS = 8.0                     # Phase 9.3: the weekly adapter's memory horizon

# Market eras: coarse, hand-set, public knowledge. A lesson keeps the era, never the date it came from.
ERAS = [("pre_2000", "1900-01-01", "1999-12-31"), ("dotcom_bust", "2000-01-01", "2002-12-31"),
        ("expansion_2003_07", "2003-01-01", "2007-12-31"), ("financial_crisis", "2008-01-01", "2009-12-31"),
        ("zero_rates_2010_19", "2010-01-01", "2019-12-31"), ("pandemic_2020_21", "2020-01-01", "2021-12-31"),
        ("rate_shock_2022_on", "2022-01-01", "2100-01-01")]
ERA_ORDER = [e[0] for e in ERAS]


def era_gap(a, b):
    """How many eras apart two labelled eras are (0 = same). None if either is unknown: an unknown era never discounts."""
    if a in ERA_ORDER and b in ERA_ORDER:
        return abs(ERA_ORDER.index(a) - ERA_ORDER.index(b))
    return None


ERROR_TYPES = ("correct", "false_positive", "false_negative", "noise", "unscored")
# fields a lesson must never carry (Phase 9.2)
FORBIDDEN_KEYS = re.compile(r"(tick|symbol|cusip|isin|permno|window|run_id|test_id|hidden|sealed|seed_year|exact_)", re.I)
OUTCOME_EDGES = (-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10)


def context_of(snap):
    """Market context of a decision day, from the snapshot the system already sees."""
    return np.array([float(snap[c].iloc[0]) if c in snap and snap[c].notna().any() else 0.0 for c in CTX])


def era_of(date):
    """Era label of a calendar date; 'unknown' when there is no date (a lesson never fabricates one)."""
    if date is None:
        return "unknown"
    try:
        t = pd.Timestamp(date)
    except (ValueError, TypeError):
        return "unknown"
    if pd.isna(t):
        return "unknown"
    for name, a, b in ERAS:
        if pd.Timestamp(a) <= t <= pd.Timestamp(b):
            return name
    return "unknown"


def classify_error(expected, outcome, noise=0.0):
    """What kind of miss was this? expected = what memory predicted for the arm before the outcome arrived.
    'noise' when the realised value is within `noise` of zero, so nobody learns a rule from a coin flip."""
    if expected is None or not np.isfinite(expected) or not np.isfinite(outcome):
        return "unscored"
    if abs(outcome) <= noise:
        return "noise"
    if expected > 0 and outcome > 0 or expected < 0 and outcome < 0:
        return "correct"
    if expected > 0 >= outcome:
        return "false_positive"
    if expected <= 0 < outcome:
        return "false_negative"
    return "noise"


def outcome_bin(x):
    """Quantise an outcome to a signed bucket: lessons keep the direction and rough size, not the exact answer."""
    if not np.isfinite(x):
        return "nan"
    i = int(np.searchsorted(OUTCOME_EDGES, x, side="right"))
    lo = "-inf" if i == 0 else f"{OUTCOME_EDGES[i - 1]:+.2f}"
    hi = "+inf" if i == len(OUTCOME_EDGES) else f"{OUTCOME_EDGES[i]:+.2f}"
    return f"[{lo},{hi})"


def coarse_date(date, grain="quarter"):
    """A date coarse enough not to identify a test window: '2019Q3' or '2019'. None stays None."""
    if date is None:
        return None
    t = pd.Timestamp(date)
    if pd.isna(t):
        return None
    return f"{t.year}" if grain == "year" else f"{t.year}Q{(t.month - 1) // 3 + 1}"


def arm_kind(arm):
    """The family an arm belongs to: the first element of a tuple arm ('knob', 'ic', ...), else 'other'."""
    return arm[0] if isinstance(arm, tuple) and arm and isinstance(arm[0], str) else "other"


def _arm_text(arm):
    return arm if isinstance(arm, str) else repr(arm)


def _arm_parse(a):
    """Arms are stored as text in the bank; read them back as the tuple they were. Non-literals stay strings."""
    if not isinstance(a, str):
        return a
    try:
        return ast.literal_eval(a)
    except (ValueError, SyntaxError):
        return a


@dataclass
class Lesson:
    """Phase 9.1: what an episode teaches, with nothing that identifies the answer."""
    arm: str
    fingerprint: str        # hash of the rounded market context: 'have we seen this kind of market' without the date
    context: dict           # CTX name -> rounded reading
    features: dict          # relevant scalar features (rounded), caller supplied and sanitised
    outcome_bin: str        # quantised realised outcome
    error_type: str
    source_experiment: str
    date: str               # coarse: quarter or year
    era: str
    relevance: float        # weight the episode carries for a market like its own, today
    reliability: float      # n_eff / (n_eff + shrink) of its arm
    shock_state: str        # none | pre_break | post_break


def sanitize_features(features, tickers=None):
    """Drop forbidden keys and any string value that is a known ticker; round floats so a value cannot be an exact answer."""
    out = {}
    tk = {t.upper() for t in (tickers or ())}
    for k, v in (features or {}).items():
        if FORBIDDEN_KEYS.search(str(k)):
            continue
        if isinstance(v, str) and v.upper() in tk:
            continue
        if isinstance(v, (float, np.floating)):
            v = round(float(v), 3)
        elif isinstance(v, (np.integer,)):
            v = int(v)
        out[str(k)] = v
    return out


def scan_lessons(lessons, tickers=(), window_ids=(), exact_outcomes=()):
    """Leak scan of a lesson table/list (Phase 9.2). Returns findings: [(row, field, why)]. Empty = clean.
    tickers / window_ids: identifiers that must appear nowhere; exact_outcomes: realised values that must not survive."""
    rows = lessons.to_dict("records") if isinstance(lessons, pd.DataFrame) else [asdict(l) if isinstance(l, Lesson) else dict(l) for l in lessons]
    tk = [t for t in tickers if t]
    tk_re = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(re.escape(t) for t in tk) + r")(?![A-Za-z0-9])") if tk else None
    wid = [str(w) for w in window_ids]
    ex = np.array(sorted({round(float(x), 8) for x in exact_outcomes}), dtype=float)
    found = []
    for i, r in enumerate(rows):
        for k, v in r.items():
            if FORBIDDEN_KEYS.search(str(k)) and k != "source_experiment":
                found.append((i, k, "forbidden field name"))
            txt = json.dumps(v, default=str) if not isinstance(v, str) else v
            if tk_re is not None and tk_re.search(txt):
                found.append((i, k, "contains a ticker"))
            if any(w in txt for w in wid):
                found.append((i, k, "contains a test-window id"))
            if k == "date" and isinstance(v, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}.*", v):
                found.append((i, k, "exact date"))
            if isinstance(v, (float, np.floating)) and len(ex) and k in ("outcome", "future_outcome", "realised"):
                if np.isclose(ex, float(v), rtol=0, atol=1e-8).any():
                    found.append((i, k, "exact future outcome"))
    return found


class Memory:
    def __init__(self, params=None, long_term=None):
        self.p = {**MEM_DEFAULT, **MEM_EXT_DEFAULT, **(params or {})}
        self.ep = []                         # (arm, week, ctx, outcome, source) ; source 0 = this window, 1 = long-term
        self.info = []                       # parallel to ep: {"date","era","exp","err","seq"}
        self.breaks = {}                     # arm -> week of the last detected break
        self.break_log = []                  # (arm, week, direction) every detected break, in order
        self.cusum = {}                      # arm -> (pos, neg, mean, var, n)
        self.scale = None
        self.era_now = None                  # era of today's market, for the era factor
        self.date_now = None                 # calendar 'now', for the modernity factor
        self.seq = 0                         # insertion counter: the deterministic tie-break for eviction
        self.rejected = 0                    # records refused (non-finite outcome)
        self.evicted = 0
        self.eviction_log = []               # last evictions: (arm, week, source, why)
        self._by_arm = {}
        if long_term is not None and len(long_term):
            cols = set(long_term.columns)
            for r in long_term.itertuples(index=False):
                arm = _arm_parse(r.arm)                                             # stored as text
                d = _as_ts(getattr(r, "date", None)) if "date" in cols else None
                er = getattr(r, "era", None) if "era" in cols else None
                self._add((arm, -1e6, np.asarray(r.ctx, dtype=float), float(r.outcome), 1),
                          {"date": d, "era": er or (era_of(d) if d is not None else None), "exp": None, "err": None})
            self._fit_scale()

    # ---------------------------------------------------------------- bookkeeping
    def _add(self, ep, info):
        info = {"date": info.get("date"), "era": info.get("era"), "exp": info.get("exp"), "err": info.get("err"), "seq": self.seq}
        self.seq += 1
        self._by_arm.setdefault(ep[0], []).append(len(self.ep))
        self.ep.append(ep)
        self.info.append(info)

    def _reindex(self):
        self._by_arm = {}
        for i, e in enumerate(self.ep):
            self._by_arm.setdefault(e[0], []).append(i)

    def _fit_scale(self):
        C = np.array([e[2] for e in self.ep]) if self.ep else np.zeros((1, len(CTX)))
        sd = C.std(axis=0)
        self.scale = np.where(sd > 1e-9, sd, 1.0)

    def set_clock(self, date=None, era=None):
        """Tell the memory what 'today' is (for the era and modernity factors). Never reads the future: it is set
        by the caller from the decision day, and record() advances it only forward."""
        if date is not None:
            self.date_now = _as_ts(date)
            if era is None:
                era = era_of(date)
        if era is not None:
            self.era_now = era

    def __len__(self):
        return len(self.ep)

    # ---------------------------------------------------------------- recording
    def record(self, arm, week, ctx, outcome, date=None, era=None, source_experiment=None, expected=None):
        """Store one closed week's lesson for `arm`. `expected` (what memory predicted before this outcome) turns
        the episode's error type from 'unscored' into correct/false_positive/false_negative/noise."""
        ctx = np.asarray(ctx, dtype=float)
        if ctx.shape != (len(CTX),):
            raise ValueError(f"context must have {len(CTX)} entries, got shape {ctx.shape}")
        if not np.isfinite(outcome) or not np.isfinite(ctx).all() or not np.isfinite(week):
            self.rejected += 1                       # a NaN would silently poison every mean built from this arm
            return False
        ts = _as_ts(date)
        er = era or (era_of(ts) if ts is not None else None)
        self._add((arm, float(week), ctx, float(outcome), 0),
                  {"date": ts, "era": er, "exp": source_experiment,
                   "err": classify_error(expected, float(outcome)) if expected is not None else "unscored"})
        if ts is not None and (self.date_now is None or ts > self.date_now):
            self.date_now = ts
        if self.scale is None or len(self.ep) % 25 == 0:
            self._fit_scale()
        self._shock_update(arm, float(week), float(outcome))
        self._enforce_capacity(arm, float(week))
        return True

    def _shock_update(self, arm, week, outcome):
        """Two-sided CUSUM on this arm's own outcomes (Phase 9.6)."""
        pos, neg, m, v, n = self.cusum.get(arm, (0.0, 0.0, 0.0, 0.0, 0))
        n += 1
        d = outcome - m
        m += d / n
        v += d * (outcome - m)
        sd = np.sqrt(v / max(n - 1, 1)) if n > 2 else None
        if sd and sd > 0:
            z = (outcome - m) / sd
            slack = self.p["mem_cusum_slack"]
            pos, neg = max(0.0, pos + z - slack), min(0.0, neg + z + slack)
            limit = self.p["mem_shock_k"] * 2
            if pos > limit or neg < -limit:
                self.breaks[arm] = week
                self.break_log.append((arm, week, "up" if pos > limit else "down"))
                pos, neg = 0.0, 0.0
        self.cusum[arm] = (pos, neg, m, v, n)

    # ---------------------------------------------------------------- capacity
    def retention_score(self, i, week_ref):
        """How much episode i is worth keeping: its recency x source x shock factors as of week_ref, era ignored
        (eviction must not depend on today's market). Lowest score is evicted first; ties evict the older insertion."""
        arm, wk, _, _, src = self.ep[i]
        age = self._retention_age(i, week_ref)
        s = 0.5 ** (age / self.half_life(arm))
        if src:
            s *= self.p["mem_prior_scale"]
        b = self.breaks.get(arm)
        if b is not None and (src or wk < b):
            s *= self.p["mem_shock_cut"]
        return s

    def half_life(self, arm):
        """Recency half-life in weeks for this arm: its kind's override if one is set, else the global one."""
        kinds = self.p["mem_kind_half_life"]
        return kinds.get(arm_kind(arm), self.p["mem_half_life"]) if kinds else self.p["mem_half_life"]

    def _retention_age(self, i, week_ref):
        arm, wk, _, _, src = self.ep[i]
        if not src:
            return max(0.0, week_ref - wk)
        d, now = self.info[i]["date"], self.date_now
        if d is not None and now is not None:
            return max(0.0, (now - d).days / 7.0)
        return float(self.p["mem_lt_age_weeks"])

    def _enforce_capacity(self, arm, week_ref):
        cap_arm, cap = self.p["mem_arm_capacity"], self.p["mem_capacity"]
        if cap_arm is not None:
            for a in ([arm] if arm is not None else list(self._by_arm)):      # arm=None: sweep every arm (after a bulk absorb)
                idx = self._by_arm.get(a, [])
                while len(idx) > cap_arm:
                    self._evict(self._weakest(idx, week_ref), "arm capacity")
                    idx = self._by_arm.get(a, [])
        if cap is not None:
            while len(self.ep) > cap:
                self._evict(self._weakest(range(len(self.ep)), week_ref), "total capacity")

    def _weakest(self, idx, week_ref):
        return min(idx, key=lambda i: (round(self.retention_score(i, week_ref), 15), self.info[i]["seq"]))

    def _evict(self, i, why):
        arm, wk, _, _, src = self.ep[i]
        del self.ep[i]
        del self.info[i]
        self.evicted += 1
        self.eviction_log = (self.eviction_log + [(arm, wk, src, why)])[-1000:]
        self._reindex()

    # ---------------------------------------------------------------- weighting
    def factors(self, e, week_now, ctx_now, info=None):
        """The six weighting factors of one episode, as a dict. weight() is their product."""
        arm, wk, ctx, _, src = e
        p = self.p
        age = 0.0 if src else max(0.0, week_now - wk)
        f = {"recency": 0.5 ** (age / self.half_life(arm)), "similarity": 1.0, "source": p["mem_prior_scale"] if src else 1.0,
             "shock": 1.0, "era": 1.0, "modernity": 1.0}
        if self.scale is not None:
            d = (ctx - ctx_now) / self.scale
            f["similarity"] = float(np.exp(-float(d @ d) / (2 * p["mem_bandwidth"] ** 2)))
        b = self.breaks.get(arm)
        if b is not None and (src or wk < b):
            f["shock"] = p["mem_shock_cut"]
        info = info or {}
        if self.era_now and info.get("era") and info["era"] != self.era_now:
            if p["mem_era_decay"] is not None:
                gap = era_gap(info["era"], self.era_now)
                f["era"] = p["mem_era_decay"] ** gap if gap is not None else 1.0
            elif p["mem_era_other"] != 1.0:
                f["era"] = p["mem_era_other"]
        hl = p["mem_modern_half_life"]
        if hl and self.date_now is not None and info.get("date") is not None:
            f["modernity"] = 0.5 ** (max(0.0, (self.date_now - info["date"]).days / 365.25) / hl)
        return f

    def weight(self, e, week_now, ctx_now):
        i = self._position(e)
        f = self.factors(e, week_now, ctx_now, self.info[i] if i is not None else None)
        # same multiplication order as the pre-Phase-9 memory so default weights are bit-identical
        w = f["recency"] * f["similarity"]
        if e[4]:
            w *= f["source"]
        w *= f["shock"]
        return w * f["era"] * f["modernity"]

    def _position(self, e):
        for i in self._by_arm.get(e[0], []):
            if self.ep[i] is e:
                return i
        return None

    def _arm_weights(self, arm, week_now, ctx_now):
        idx = self._by_arm.get(arm, [])
        ws = np.empty(len(idx))
        for j, i in enumerate(idx):
            e = self.ep[i]
            f = self.factors(e, week_now, ctx_now, self.info[i])
            w = f["recency"] * f["similarity"]
            if e[4]:
                w *= f["source"]
            ws[j] = w * f["shock"] * f["era"] * f["modernity"]
        return idx, ws

    def estimate_detail(self, arm, week_now, ctx_now):
        """Everything behind an estimate: raw and shrunk mean, standard error, effective sample, reliability."""
        idx, w = self._arm_weights(arm, week_now, ctx_now)
        empty = {"mean": 0.0, "raw_mean": 0.0, "se": np.inf, "n_eff": 0.0, "n": len(idx), "sum_w": 0.0,
                 "reliability": 0.0, "var": 0.0}
        if not len(idx):
            return empty
        x = np.array([self.ep[i][3] for i in idx])
        sw = w.sum()
        if sw <= 0:
            return empty
        if w.max() < 1e-100:                 # every episode is ancient: w**2 would underflow to 0 and n_eff to 0/0.
            w = w / w.max()                  # The estimate is invariant to the scale of w, so rescale (sum_w keeps the true value)
            sw_true, sw = sw, w.sum()
        else:
            sw_true = sw
        n_eff = sw ** 2 / (w ** 2).sum()
        raw = float((w * x).sum() / sw)
        mean = float((w * x).sum() / (sw + self.p["mem_shrink"] * w.mean()))       # shrink toward 0
        var = float((w * (x - (w * x).sum() / sw) ** 2).sum() / sw)
        se = np.sqrt(max(var, self.p["mem_min_var"]) / max(n_eff, 1.0))
        return {"mean": mean, "raw_mean": raw, "se": float(se), "n_eff": float(n_eff), "n": len(idx), "sum_w": float(sw_true),
                "reliability": float(sw / (sw + self.p["mem_shrink"] * w.mean())), "var": var}

    def estimate(self, arm, week_now, ctx_now):
        """(shrunk mean, standard error, effective sample size) for an arm, weighted as described above."""
        d = self.estimate_detail(arm, week_now, ctx_now)
        if d["n_eff"] <= 0:
            return 0.0, np.inf, 0.0
        return d["mean"], d["se"], d["n_eff"]

    def reliability(self, arm, week_now=0.0, ctx_now=None):
        """Pseudo-count reliability of an arm in [0, 1): sum_w / (sum_w + shrink * mean_w)."""
        ctx_now = np.zeros(len(CTX)) if ctx_now is None else ctx_now
        return self.estimate_detail(arm, week_now, ctx_now)["reliability"]

    def arms(self):
        return sorted(self._by_arm, key=repr)

    def rank_arms(self, arms, week_now, ctx_now, min_n_eff=0.0):
        """Score several arms at once: shrunk mean, se, z = mean/se, effective sample, reliability. Sorted by z, then by
        the arm's text, so equal scores come out in a fixed order. Arms with less than min_n_eff evidence are listed last
        with z = 0 rather than dropped (an auditor wants to see what was ignored)."""
        rows = []
        for a in arms:
            d = self.estimate_detail(a, week_now, ctx_now)
            ok = d["n_eff"] >= max(min_n_eff, 1e-12) and np.isfinite(d["se"])
            rows.append({"arm": a, "text": repr(a), "mean": d["mean"], "se": d["se"], "z": d["mean"] / d["se"] if ok else 0.0,
                         "n_eff": d["n_eff"], "reliability": d["reliability"], "usable": bool(ok)})
        t = pd.DataFrame(rows, columns=["arm", "text", "mean", "se", "z", "n_eff", "reliability", "usable"])
        return t.sort_values(["usable", "z", "text"], ascending=[False, False, True]).reset_index(drop=True)

    def absorb(self, bank):
        """Fold a long-term bank (a DataFrame like export()'s) into this memory as source-1 episodes, skipping rows
        already present (same arm, context and outcome) so absorbing twice changes nothing. Returns rows added."""
        have = {(repr(e[0]), tuple(np.round(e[2], 9)), round(e[3], 12)) for e in self.ep if e[4]}
        added = 0
        cols = set(bank.columns)
        for r in bank.itertuples(index=False):
            arm = _arm_parse(r.arm)
            c = np.asarray(r.ctx, dtype=float)
            key = (repr(arm), tuple(np.round(c, 9)), round(float(r.outcome), 12))
            if key in have or not np.isfinite(r.outcome) or c.shape != (len(CTX),):
                continue
            d = _as_ts(getattr(r, "date", None)) if "date" in cols else None
            er = getattr(r, "era", None) if "era" in cols else None
            self._add((arm, -1e6, c, float(r.outcome), 1), {"date": d, "era": er or (era_of(d) if d is not None else None),
                                                            "exp": None, "err": None})
            have.add(key)
            added += 1
        if added:
            self._fit_scale()
            self._enforce_capacity(None, max((e[1] for e in self.ep if not e[4]), default=0.0))
        return added

    def counts(self):
        """Episodes per arm (this window / long-term)."""
        out = {}
        for a, ix in self._by_arm.items():
            out[a] = (sum(1 for i in ix if not self.ep[i][4]), sum(1 for i in ix if self.ep[i][4]))
        return out

    # ---------------------------------------------------------------- lessons (Phase 9.1 / 9.2)
    def shock_state(self, i):
        arm, wk, _, _, src = self.ep[i]
        b = self.breaks.get(arm)
        if b is None:
            return "none"
        return "pre_break" if (src or wk < b) else "post_break"

    def lessons(self, features=None, tickers=None, week_now=None):
        """One Lesson per episode of THIS window. `features`: optional {episode_index: {name: value}} of relevant
        scalar features, sanitised here. Nothing that identifies a ticker, an exact outcome, or a test window leaves."""
        grain = self.p["mem_date_grain"]
        wk = max((e[1] for e in self.ep if not e[4]), default=0.0) if week_now is None else week_now
        out = []
        rel = {}
        for i, e in enumerate(self.ep):
            if e[4]:
                continue
            arm, w0, ctx, y, _ = e
            if arm not in rel:
                rel[arm] = self.estimate_detail(arm, wk, ctx)["reliability"]
            rounded = np.round(ctx, 2)
            f = self.factors(e, wk, ctx, self.info[i])
            info = self.info[i]
            out.append(Lesson(arm=_scrub_arm(arm, tickers), fingerprint=hashlib.sha1(rounded.tobytes()).hexdigest()[:12],
                              context={c: float(v) for c, v in zip(CTX, rounded)},
                              features=sanitize_features((features or {}).get(i), tickers),
                              outcome_bin=outcome_bin(y), error_type=info["err"] or "unscored",
                              source_experiment=str(info["exp"] or "unspecified"), date=coarse_date(info["date"], grain) or "undated",
                              era=info["era"] or "unknown", relevance=float(f["recency"] * f["shock"]),
                              reliability=float(rel[arm]), shock_state=self.shock_state(i)))
        return out

    def export_lessons(self, features=None, tickers=None, week_now=None):
        """The sanitised bank (a DataFrame). This, not export(), is what may be shared across experiments."""
        ls = self.lessons(features, tickers, week_now)
        from .learning import wiring
        wiring.on_lessons(ls)                  # S17a: shared lesson bank -> failure hypotheses (sink)
        cols = list(Lesson.__dataclass_fields__)
        return pd.DataFrame([asdict(l) for l in ls], columns=cols)

    def export(self):
        """This window's own episodes, for the long-term bank."""
        rows = []
        for (a, _, c, o, s), inf in zip(self.ep, self.info):
            if s == 0:
                r = {"arm": repr(a), "ctx": [float(x) for x in c], "outcome": float(o)}
                if inf["date"] is not None:
                    r["date"] = str(inf["date"].date())
                if inf["era"]:
                    r["era"] = inf["era"]
                rows.append(r)
        return pd.DataFrame(rows)

    # ---------------------------------------------------------------- self-check
    def validate(self):
        """Invariants the memory must always hold. Returns a list of problems (empty = healthy): parallel lists in step,
        the arm index pointing at the right episodes, finite outcomes and contexts, unique insertion numbers, capacity
        respected, breaks only for arms that have episodes or CUSUM state, a finite context scale. Cheap enough to call
        every week; the adapter's tests do, and so should any long-running process."""
        bad = []
        if len(self.ep) != len(self.info):
            bad.append(f"episodes ({len(self.ep)}) and info ({len(self.info)}) out of step")
            return bad
        seen = 0
        for arm, ix in self._by_arm.items():
            seen += len(ix)
            for i in ix:
                if i >= len(self.ep) or self.ep[i][0] != arm:
                    bad.append(f"index entry {i} does not hold arm {arm!r}")
        if seen != len(self.ep):
            bad.append(f"arm index covers {seen} of {len(self.ep)} episodes")
        for i, (arm, wk, c, y, src) in enumerate(self.ep):
            if not (np.isfinite(y) and np.isfinite(c).all() and (np.isfinite(wk) or src)):
                bad.append(f"episode {i} has a non-finite field")
            if c.shape != (len(CTX),):
                bad.append(f"episode {i} context has shape {c.shape}")
            if src not in (0, 1):
                bad.append(f"episode {i} has source {src}")
        seqs = [i["seq"] for i in self.info]
        if len(set(seqs)) != len(seqs):
            bad.append("insertion numbers are not unique")
        if seqs and max(seqs) >= self.seq:
            bad.append("insertion counter behind the newest episode")
        cap, cap_arm = self.p["mem_capacity"], self.p["mem_arm_capacity"]
        if cap is not None and len(self.ep) > cap:
            bad.append(f"{len(self.ep)} episodes exceed capacity {cap}")
        if cap_arm is not None:
            for arm, ix in self._by_arm.items():
                if len(ix) > cap_arm:
                    bad.append(f"arm {arm!r} holds {len(ix)} > {cap_arm}")
        for arm in self.breaks:
            if arm not in self.cusum:
                bad.append(f"break for {arm!r} with no CUSUM state")
        if self.scale is not None and not (np.isfinite(self.scale).all() and (self.scale > 0).all()):
            bad.append("context scale is not finite and positive")
        return bad

    # ---------------------------------------------------------------- state and determinism
    def state_dict(self):
        """Plain-data image of the memory: episodes, breaks, CUSUM state, counters. Round-trips through from_state()."""
        return {"p": dict(self.p),
                "ep": [[_arm_text(a), w, [float(x) for x in c], o, s] for a, w, c, o, s in self.ep],
                "info": [{"date": None if i["date"] is None else str(i["date"]), "era": i["era"], "exp": i["exp"],
                          "err": i["err"], "seq": i["seq"]} for i in self.info],
                "breaks": [[_arm_text(a), w] for a, w in sorted(self.breaks.items(), key=lambda kv: repr(kv[0]))],
                "cusum": [[_arm_text(a)] + [float(x) for x in v] for a, v in sorted(self.cusum.items(), key=lambda kv: repr(kv[0]))],
                "seq": self.seq, "rejected": self.rejected, "evicted": self.evicted,
                "era_now": self.era_now, "date_now": None if self.date_now is None else str(self.date_now),
                # the context scale is refit only every 25 records, so it is state, not a function of the episodes
                "scale": None if self.scale is None else [float(x) for x in self.scale]}

    @classmethod
    def from_state(cls, st):
        m = cls(st["p"])
        for (a, w, c, o, s), i in zip(st["ep"], st["info"]):
            m.ep.append((_arm_parse(a), w, np.asarray(c, dtype=float), o, s))
            m.info.append({"date": _as_ts(i["date"]), "era": i["era"], "exp": i["exp"], "err": i["err"], "seq": i["seq"]})
        m._reindex()
        m.breaks = {_arm_parse(a): w for a, w in st["breaks"]}
        m.cusum = {_arm_parse(r[0]): (r[1], r[2], r[3], r[4], int(r[5])) for r in st["cusum"]}
        m.seq, m.rejected, m.evicted = st["seq"], st["rejected"], st["evicted"]
        m.era_now, m.date_now = st["era_now"], _as_ts(st["date_now"])
        m.scale = None if st.get("scale") is None else np.asarray(st["scale"], dtype=float)
        return m

    def fingerprint(self):
        """sha256 of the whole memory. Two runs on the same inputs must give the same string."""
        blob = json.dumps(self.state_dict(), sort_keys=True, default=str, allow_nan=True)
        return hashlib.sha256(blob.encode()).hexdigest()


def _as_ts(d):
    if d is None or (isinstance(d, float) and math.isnan(d)):
        return None
    t = pd.Timestamp(d)
    return None if pd.isna(t) else t


def _scrub_arm(arm, tickers):
    """An arm is a knob/indicator tuple; if a string element is a known ticker, replace the whole arm by a hash of it."""
    text = _arm_text(arm)
    tk = {t.upper() for t in (tickers or ())}
    if tk and any(isinstance(x, str) and x.upper() in tk for x in (arm if isinstance(arm, tuple) else (arm,))):
        return "arm:" + hashlib.sha1(text.encode()).hexdigest()[:10]
    return text
