"""Bible Phase 30 (experiment memory), the search-space half: a normalised index of WHAT WAS TRIED.

engine.registry.ExperimentMemory records the twelve answers per experiment. This module answers the question asked
before launching one: has this configuration, or something very close to it, already been run, and how did it go?

* configs live in a declared `Space` (numeric ranges and categorical choices); distance is the mean normalised
  difference over the union of parameters (a parameter one side lacks counts as a full difference);
* `check(cfg)` flags exact repeats and near-duplicates and says whether a near neighbour already FAILED;
* `untried_neighbours(cfg)` proposes one-parameter moves not already covered by a prior attempt, best-first;
* `negative_results()` is the table of rejected / failed attempts with the reason, worst first.
The index is an append-only jsonl so it survives restarts and is never edited."""
import json
import math
from pathlib import Path

from .registry import fingerprint


class Space:
    """spec: {param: (lo, hi, step)} for numeric, {param: [choices]} for categorical. `step` is also the grid on which
    two values are treated as the same (so 0.9 and 0.9004 do not count as different experiments)."""

    def __init__(self, spec):
        self.spec = {}
        for k, v in spec.items():
            if isinstance(v, tuple) and len(v) == 3 and v[0] < v[1] and v[2] > 0:
                self.spec[k] = ("num", float(v[0]), float(v[1]), float(v[2]))
            elif isinstance(v, (list, tuple)) and len(v) >= 2:
                self.spec[k] = ("cat", list(v))
            else:
                raise ValueError(f"bad space spec for {k!r}: {v!r}")

    def snap(self, cfg):
        """Canonical config: numerics snapped to the step grid, unknown params kept as-is (they still count)."""
        out = {}
        for k, v in cfg.items():
            s = self.spec.get(k)
            if s and s[0] == "num" and isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                out[k] = round(round((v - s[1]) / s[3]) * s[3] + s[1], 10)
            else:
                out[k] = v
        return out

    def key(self, cfg):
        return fingerprint(self.snap(cfg))

    def distance(self, a, b):
        keys = set(a) | set(b)
        if not keys:
            return 0.0
        tot = 0.0
        for k in keys:
            if k not in a or k not in b:
                tot += 1.0
                continue
            s = self.spec.get(k)
            if s and s[0] == "num" and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (a[k], b[k])):
                tot += min(1.0, abs(a[k] - b[k]) / (s[2] - s[1]))
            else:
                tot += 0.0 if a[k] == b[k] else 1.0
        return tot / len(keys)


class TriedIndex:
    def __init__(self, path, space):
        self.path, self.space = Path(path), space
        self.rows = []
        if self.path.exists():
            self.rows = [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def add(self, experiment_id, cfg, outcome, score=None, reason=None, now=None):
        """outcome: adopt / reject / continue_testing. score: higher is better (None when unmeasured).
        A reject without a reason is refused: an unexplained negative result cannot stop anyone repeating it."""
        if outcome not in ("adopt", "reject", "continue_testing"):
            raise ValueError(f"unknown outcome {outcome!r}")
        if outcome == "reject" and not reason:
            raise ValueError("a rejected attempt needs a reason")
        if any(r["experiment_id"] == experiment_id for r in self.rows):
            raise ValueError(f"{experiment_id} already indexed")
        if score is not None and not math.isfinite(score):
            raise ValueError("score must be finite or None")
        row = {"experiment_id": experiment_id, "key": self.space.key(cfg), "cfg": self.space.snap(cfg), "outcome": outcome,
               "score": score, "reason": reason, "t": None if now is None else str(now)}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        self.rows.append(row)
        return row

    def import_registry(self, registry, cfg_field="config", score_metric="metrics.score"):
        """Index registry records that carry a config dict, an outcome, and (for rejects) a reason. Returns counts."""
        from .registry import _dig, _get
        added = skipped = 0
        for r in registry.records:
            cfg, oc = r.get(cfg_field), str(_get(r, "outcome") or "").lower()
            eid = r.get("experiment_id")
            if not isinstance(cfg, dict) or not eid or oc not in ("adopt", "reject", "continue_testing") \
                    or any(x["experiment_id"] == eid for x in self.rows) or (oc == "reject" and not _get(r, "reason")):
                skipped += 1
                continue
            sc = _dig(r, score_metric)
            self.add(eid, cfg, oc, sc if isinstance(sc, (int, float)) and math.isfinite(sc) else None, _get(r, "reason"))
            added += 1
        return {"added": added, "skipped": skipped}

    def nearest(self, cfg, k=5):
        snapped = self.space.snap(cfg)
        ds = sorted(((self.space.distance(snapped, r["cfg"]), r) for r in self.rows), key=lambda x: x[0])
        return ds[:k]

    def check(self, cfg, near=0.08):
        """Pre-launch check. verdict: 'repeat' (identical after snapping), 'near_duplicate' (within `near` of a prior
        attempt), 'known_failure_nearby' (a near attempt was rejected and none adopted), or 'novel'."""
        key = self.space.key(cfg)
        exact = [r for r in self.rows if r["key"] == key]
        close = [(d, r) for d, r in self.nearest(cfg, 50) if d <= near and r["key"] != key]
        bad = [r for r in exact + [r for _, r in close] if r["outcome"] == "reject"]
        good = [r for r in exact + [r for _, r in close] if r["outcome"] == "adopt"]
        if exact:
            verdict = "repeat"
        elif close:
            verdict = "near_duplicate"
        else:
            verdict = "novel"
        if bad and not good and verdict != "novel":
            verdict = "repeat" if exact else "known_failure_nearby"
        return {"verdict": verdict, "block": verdict in ("repeat", "known_failure_nearby") and bool(bad) and not good,
                "exact": [r["experiment_id"] for r in exact], "close": [(round(d, 4), r["experiment_id"]) for d, r in close],
                "reasons": [r["reason"] for r in bad]}

    def untried_neighbours(self, cfg, n=10, near=0.08):
        """One-parameter moves from cfg (numeric +-1 and +-2 steps; every other categorical choice) that neither repeat nor
        sit near a prior attempt. Ranked by the best score among tried attempts within 3x `near` of the candidate
        (unknown = middle), so the search leans toward promising regions rather than uniformly."""
        base = self.space.snap(cfg)
        cands = {}
        for k, s in self.space.spec.items():
            if k not in base:
                continue
            opts = []
            if s[0] == "num":
                for m in (-2, -1, 1, 2):
                    v = round(base[k] + m * s[3], 10)
                    if s[1] <= v <= s[2]:
                        opts.append(v)
            else:
                opts = [c for c in s[1] if c != base[k]]
            for v in opts:
                c = {**base, k: v}
                cands[self.space.key(c)] = (k, c)
        out = []
        for key, (k, c) in cands.items():
            if self.check(c, near)["verdict"] != "novel":
                continue
            scores = [r["score"] for d, r in self.nearest(c, 50) if d <= 3 * near and r["score"] is not None]
            out.append({"param": k, "cfg": c, "prior": max(scores) if scores else None})
        known = [o["prior"] for o in out if o["prior"] is not None]
        mid = sum(known) / len(known) if known else 0.0
        out.sort(key=lambda o: -(o["prior"] if o["prior"] is not None else mid))
        return out[:n]

    def negative_results(self):
        """Rejected attempts, lowest score first (unscored last), with the reason each was rejected."""
        rows = [{"experiment_id": r["experiment_id"], "cfg": r["cfg"], "score": r["score"], "reason": r["reason"]}
                for r in self.rows if r["outcome"] == "reject"]
        return sorted(rows, key=lambda r: (r["score"] is None, r["score"] if r["score"] is not None else 0))

    def coverage(self):
        """Share of each declared parameter's range/choices that has been touched: where nobody has looked yet."""
        out = {}
        for k, s in self.space.spec.items():
            vals = [r["cfg"][k] for r in self.rows if k in r["cfg"]]
            if s[0] == "num":
                grid = int(round((s[2] - s[1]) / s[3])) + 1
                out[k] = min(1.0, len({round(v, 8) for v in vals}) / grid) if grid else 0.0
            else:
                out[k] = len({v for v in vals if v in s[1]}) / len(s[1])
        return out


# ==================================================================================================================
# Phase 30, the answers half: four of the twelve questions computed from state/experiments.jsonl and the weekly series
# behind a record, instead of typed in. Each answer is a dict with a `known` flag: "cannot tell" is never rendered as
# "no" (a missing range is not "nothing was unseen", a missing series is not "not significant").
# ==================================================================================================================
import re as _re

import numpy as _np

# metric name -> True when a HIGHER value is better. Anything not listed is assumed higher-is-better and flagged as such.
POLARITY = {"mean_week": True, "median_week": True, "worst_week": True, "p05_week": True, "win_weeks": True, "pct_ge_7": True,
            "share_in_band": True, "positive_week_pct": True, "rate_8_of_10": True, "ic": True, "auc": True, "cagr": True,
            "final": True, "sharpe": True, "direction_accuracy": True, "hit_rate": True,
            "max_dd": True, "max_drawdown": True,               # stored negative: less negative is better
            "pct_le_m7": False, "turnover": False, "turnover_per_week": False, "costs": False, "cost": False, "brier": False,
            "ece": False, "log_loss": False, "fdr": False, "catastrophic_losses": False, "mae": False, "rmse": False}
RISK_METRICS = ("max_dd", "max_drawdown", "worst_week", "p05_week", "pct_le_m7", "catastrophic_losses")
_DATE = _re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def load_experiments(path=None):
    """(records, n_unparseable) from the append-only log. A torn line is counted, never silently dropped."""
    from . import config as _K
    p = Path(path) if path else _K.STATE / "experiments.jsonl"
    rows, bad = [], 0
    if p.exists():
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if isinstance(r, dict):
                rows.append(r)
            else:
                bad += 1
    return rows, bad


def parse_range(x):
    """(start, end) as ISO date strings from '2008-01-01..2015-12-31', 'a to b', {'start','end'}, [a, b]; None when the
    range holds no date at all (open-ended text such as 'holdout from 2023-12-22' yields (start, None))."""
    if x is None:
        return None
    if isinstance(x, dict):
        a, b = x.get("start"), x.get("end")
    elif isinstance(x, (list, tuple)) and len(x) == 2:
        a, b = x
    else:
        found = _DATE.findall(str(x))
        a = "-".join(found[0]) if found else None
        b = "-".join(found[1]) if len(found) > 1 else None
    a = str(a)[:10] if a and _DATE.match(str(a)) else None
    b = str(b)[:10] if b and _DATE.match(str(b)) else None
    return (a, b) if (a or b) else None


def unseen(rec):
    """Phase 30 'what was unseen?': the ranges the record says were held out from training, whether they really are
    disjoint from it, and which ones cannot be checked. known=False when the record carries no ranges at all."""
    ranges = {k: rec.get(f"{k}_range") for k in ("train", "validation", "test")}
    if ranges["validation"] is None:
        ranges["validation"] = rec.get("val_range")
    parsed = {k: parse_range(v) for k, v in ranges.items()}
    held = {k: v for k, v in ranges.items() if v is not None and k != "train"}
    if not held and ranges["train"] is None:
        return {"known": False, "held_out": [], "overlap": [], "unverifiable": [], "clean": False,
                "why": "record carries no train/validation/test range"}
    tr = parsed["train"]
    overlap, unverifiable = [], []
    for k in ("validation", "test"):
        w = parsed[k]
        if ranges[k] is None:
            continue
        if not (tr and tr[0] and tr[1] and w and w[0] and w[1]):
            unverifiable.append(k)
        elif not (w[1] < tr[0] or w[0] > tr[1]):
            overlap.append(k)
    return {"known": True, "held_out": sorted(held), "overlap": overlap, "unverifiable": unverifiable,
            "clean": bool(held) and not overlap and not unverifiable,
            "ranges": {k: (str(v) if v is not None else None) for k, v in ranges.items()}}


def _metrics(rec_or_dict):
    m = rec_or_dict.get("metrics") if isinstance(rec_or_dict.get("metrics"), dict) else rec_or_dict
    return {k: float(v) for k, v in m.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)}


def worsened(cand, base, rel_tol=0.02, abs_tol=1e-12):
    """Phase 30 'what worsened?': every shared metric where the candidate is worse than the baseline by more than
    rel_tol of the baseline's magnitude. Direction comes from POLARITY; a metric absent from it is compared as
    higher-is-better and listed under `assumed_direction` so the assumption is visible. Returns improved/worsened/flat,
    the risk metrics among the worsened, and the metrics only one side has (`unshared`)."""
    c, b = _metrics(cand), _metrics(base)
    shared = sorted(set(c) & set(b))
    imp, wor, flat, assumed = [], [], [], []
    for k in shared:
        hib = POLARITY.get(k)
        if hib is None:
            hib = True
            assumed.append(k)
        delta = (c[k] - b[k]) * (1 if hib else -1)
        scale = max(abs(b[k]), abs_tol)
        row = {"metric": k, "base": b[k], "cand": c[k], "delta": c[k] - b[k], "rel": (c[k] - b[k]) / scale}
        (imp if delta > rel_tol * scale + abs_tol else wor if delta < -(rel_tol * scale + abs_tol) else flat).append(row)
    wor.sort(key=lambda r: -abs(r["rel"]))
    imp.sort(key=lambda r: -abs(r["rel"]))
    return {"known": bool(shared), "improved": imp, "worsened": wor, "flat": flat, "assumed_direction": assumed,
            "risk_worsened": [r["metric"] for r in wor if r["metric"] in RISK_METRICS],
            "unshared": sorted(set(c) ^ set(b))}


def bootstrap_diff(cand, base=None, n_boot=4000, seed=0, block=1, alpha=0.05):
    """Mean of (cand - base) with a percentile bootstrap interval. Paired when `base` is given (equal length required);
    with base=None `cand` is already a difference series. `block` > 1 resamples circular blocks, which keeps the
    week-to-week autocorrelation that an i.i.d. bootstrap would throw away (and so understates the uncertainty).
    p is the two-sided share of resampled means on the other side of zero. Deterministic in `seed`."""
    c = _np.asarray(cand, float)
    if base is not None:
        b = _np.asarray(base, float)
        if len(b) != len(c):
            raise ValueError("paired series differ in length")
        d = c - b
    else:
        d = c
    d = d[_np.isfinite(d)]
    n = len(d)
    if n < 2:
        return {"known": False, "n": n, "why": "fewer than 2 usable observations"}
    rng = _np.random.default_rng(seed)
    block = max(1, min(int(block), n))
    nb = int(_np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, nb))
    idx = (starts[:, :, None] + _np.arange(block)[None, None, :]) % n
    means = d[idx.reshape(n_boot, -1)[:, :n]].mean(axis=1)
    lo, hi = _np.quantile(means, [alpha / 2, 1 - alpha / 2])
    p = 2 * min(float((means <= 0).mean()), float((means >= 0).mean()))
    return {"known": True, "n": n, "diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p": min(1.0, max(p, 1.0 / n_boot)), "block": block, "n_boot": n_boot}


def meaningful(cand, base=None, higher_is_better=True, min_n=20, **kw):
    """Phase 30 'was the improvement statistically meaningful?': the interval must exclude zero on the good side AND
    there must be at least min_n observations. Too few observations is 'unknown', not 'no'."""
    r = bootstrap_diff(cand, base, **kw)
    if not r["known"]:
        return {**r, "meaningful": None}
    if r["n"] < min_n:
        return {**r, "meaningful": None, "why": f"only {r['n']} observations (< {min_n})"}
    good = r["ci_low"] > 0 if higher_is_better else r["ci_high"] < 0
    bad = r["ci_high"] < 0 if higher_is_better else r["ci_low"] > 0
    return {**r, "meaningful": bool(good), "significantly_worse": bool(bad)}


def survived_another_window(windows, min_n=20, discovery=None, seed=0, **kw):
    """Phase 30 'did the improvement survive another window?'. windows: list of {'id', 'cand', 'base'} (or 'diff').
    The discovery window (default: the first) is where the change was found; every OTHER window is out of sample for
    that choice. Verdicts: 'confirmed' (pooled other windows: interval excludes zero, better), 'direction_only'
    (other windows are better on average but not significant), 'failed' (other windows are not better), and
    'unknown' when there is no other window with enough weeks."""
    if not windows:
        return {"known": False, "verdict": "unknown", "why": "no windows supplied"}
    disc = discovery if discovery is not None else windows[0].get("id")
    rows, pooled = [], []
    for w in windows:
        series = _np.asarray(w["diff"], float) if "diff" in w else _np.asarray(w["cand"], float) - _np.asarray(w["base"], float)
        series = series[_np.isfinite(series)]
        r = bootstrap_diff(series, None, seed=seed, **kw) if len(series) >= 2 else {"known": False, "n": len(series)}
        rows.append({"id": w.get("id"), "discovery": w.get("id") == disc, "n": len(series), "diff": r.get("diff"),
                     "ci_low": r.get("ci_low"), "ci_high": r.get("ci_high")})
        if w.get("id") != disc and len(series) >= min_n:
            pooled.append(series)
    if not pooled:
        return {"known": False, "verdict": "unknown", "windows": rows, "why": f"no window besides {disc!r} has {min_n}+ weeks"}
    r = bootstrap_diff(_np.concatenate(pooled), None, seed=seed, **kw)
    better = sum(1 for x in rows if not x["discovery"] and x["diff"] is not None and x["diff"] > 0)
    n_other = sum(1 for x in rows if not x["discovery"])
    verdict = "confirmed" if r["ci_low"] > 0 else "direction_only" if r["diff"] > 0 else "failed"
    return {"known": True, "verdict": verdict, "survived": verdict == "confirmed", "windows": rows, "pooled": r,
            "other_windows_better": f"{better}/{n_other}"}


def answer_phase30(rec, base_rec=None, series=None, windows=None, seed=0):
    """The four computed answers for one experiment, keyed like registry.QUESTIONS where they overlap:
    data_unseen, worsened, statistically_meaningful, survived_another_window. `series` = {'cand','base','higher_is_better'}
    of paired weekly outcomes; `windows` as in survived_another_window. Missing inputs give known=False answers."""
    out = {"data_unseen": unseen(rec)}
    out["worsened"] = worsened(rec, base_rec) if base_rec is not None else {"known": False, "why": "no baseline record supplied"}
    if series is not None:
        out["statistically_meaningful"] = meaningful(series["cand"], series.get("base"), series.get("higher_is_better", True), seed=seed)
    else:
        out["statistically_meaningful"] = {"known": False, "meaningful": None, "why": "no paired series supplied"}
    out["survived_another_window"] = survived_another_window(windows, seed=seed) if windows else \
        {"known": False, "verdict": "unknown", "why": "no other windows supplied"}
    return out


def answer_coverage(rows):
    """How many logged experiments can even answer the Phase 30 questions? Counts, over all records, those that carry
    ranges, metrics and a reason - the honest measure of how much of the memory is real."""
    return {"n": len(rows),
            "with_ranges": sum(1 for r in rows if unseen(r)["known"]),
            "with_clean_unseen": sum(1 for r in rows if unseen(r).get("clean")),
            "with_metrics": sum(1 for r in rows if isinstance(r.get("metrics"), dict) and r["metrics"]),
            "with_outcome": sum(1 for r in rows if r.get("outcome") in ("adopt", "reject", "continue_testing")),
            "with_reason": sum(1 for r in rows if r.get("reason")),
            "without_experiment_id": sum(1 for r in rows if not r.get("experiment_id"))}


# ==================================================================================================================
# B12 (INTEGRATION): the pre-launch gate every grid / loop candidate batch goes through. A candidate is TRIED once it has
# finished (registered by `record_launch` after a clean exit), so a crashed run can be retried but a finished one is never
# silently repeated. Indexes live in state/research/tried/<runner>.jsonl (append-only, one per runner).
# ==================================================================================================================
def launch_index(runner, root=None, space=None):
    """The TriedIndex for one runner. Without a declared Space every parameter counts as categorical, so only an
    identical configuration is an exact repeat."""
    from . import config as _K
    if not runner or Path(str(runner)).name != str(runner):
        raise ValueError(f"unusable runner name {runner!r}")
    d = Path(root) if root else _K.STATE / "research" / "tried"
    d.mkdir(parents=True, exist_ok=True)
    return TriedIndex(d / f"{runner}.jsonl", space or Space({}))


def prelaunch(index, cfg, near=0.08):
    """check() before a launch. Returns the check dict plus 'launch': False for an exact repeat (skip it) or a blocked
    known failure; near-duplicates still launch but are reported."""
    r = index.check(cfg, near)
    r["launch"] = not (r["verdict"] == "repeat" or r["block"])
    return r


def filter_batch(index, candidates, near=0.08, log=print):
    """Split [(experiment_id, cfg), ...] into (to_launch, skipped). Repeats inside the batch itself are skipped too."""
    go, skip, seen = [], [], set()
    for eid, cfg in candidates:
        r = prelaunch(index, cfg, near)
        key = index.space.key(cfg)
        if not r["launch"] or key in seen:
            why = "repeat within this batch" if r["launch"] else f"{r['verdict']} of {r['exact'] or r['close']}"
            skip.append({"experiment_id": eid, "cfg": cfg, "why": why})
            log(f"  experiment memory: skip {eid} ({why})")
            continue
        seen.add(key)
        go.append((eid, cfg))
    return go, skip


def record_launch(index, experiment_id, cfg, outcome="continue_testing", score=None, reason=None, now=None):
    """Register a finished candidate. A second registration of the same id is a no-op (resume after a kill)."""
    if any(r["experiment_id"] == experiment_id for r in index.rows):
        return None
    return index.add(experiment_id, cfg, outcome, score, reason, now)
