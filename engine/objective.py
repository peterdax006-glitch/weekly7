"""Tiered objective firewall (Bible PHASE 20; canon C38 band, C39 lexicographic tiers).

Tier 1  share of weeks that move ~7% (|week| inside 5-10%), penalised by overshoot; satisfied once most weeks are in band.
Tier 2  risk: worst-5% week, max drawdown, catastrophic weeks, overshoot. Only judged once tier 1 is satisfied (C39).
Tier 3  direction: share of in-band weeks that are positive, +10% outcomes, precision on real moves.
A lower-tier gain can never buy back damage to a higher tier: every tier is quantised onto an integer grid and the
grids are packed into one integer, so ordering is exactly lexicographic and transitive (a tolerance comparison is not).
`tiered_legacy` reproduces scripts/livesim_loop2.tiered bit-for-bit so old and new callers can be cross-checked.
"""
from dataclasses import dataclass, field
import numpy as np

BAND = (0.05, 0.10)            # "about 7%": below 5% too low, above 10% too risky (C38)
TARGET = 0.07
REACH = 0.5                    # tier 1 is satisfied once this share of weeks is in band (C39)
CAT_LOSS = -0.20               # a single week worse than this is catastrophic
WIPEOUT = -0.99                # max drawdown beyond this is a total loss (C22): floor score
OVER_RISK_W = 0.25             # overshoot weeks are also risk (Bible: "weeks outside band")
T1_STEP, RISK_STEP, T3_STEP = 0.02, 0.005, 0.02
RISK_FLOOR = 3.0               # risk below -3.0 is clipped: already worst possible
_RN = int(RISK_FLOOR / RISK_STEP) + 1
_TN = int(round(1 / T3_STEP)) + 1
_ROW_KEYS = ("in_band", "over_band", "worst5", "max_dd", "pos_in_band")


def week_row(weeks, band=BAND):
    """One window's tier statistics from its weekly returns (oldest first). Empty input gives an all-zero row."""
    w = np.asarray(weeks, float).ravel()
    if not np.isfinite(w).all():
        raise ValueError("weekly returns contain NaN/inf")
    if len(w) == 0:
        return {"n_weeks": 0, "mean_week": 0.0, "sd_week": 0.0, "in_band": 0.0, "over_band": 0.0, "under_band": 0.0,
                "worst5": 0.0, "max_dd": 0.0, "cat_rate": 0.0, "pos_in_band": 0.0, "plus10": 0.0, "dir_precision": 0.0}
    a = np.abs(w)
    inb = (a >= band[0]) & (a <= band[1])
    eq = np.cumprod(1 + np.maximum(w, -1.0))
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    real = a >= band[0]                                     # weeks that moved at all: the ones with a direction
    return {"n_weeks": int(len(w)), "mean_week": float(w.mean()), "sd_week": float(w.std()) if len(w) > 1 else 0.0,
            "in_band": float(inb.mean()), "over_band": float((a > band[1]).mean()), "under_band": float((a < band[0]).mean()),
            "worst5": float(np.quantile(w, 0.05)) if len(w) > 5 else float(w.min()),
            "max_dd": float((eq / peak - 1).min()), "cat_rate": float((w <= CAT_LOSS).mean()),
            "pos_in_band": float((w[inb] > 0).mean()) if inb.any() else 0.0,
            "plus10": float((w >= band[1]).mean()), "dir_precision": float((w[real] > 0).mean()) if real.any() else 0.0}


def _check(rows):
    rows = list(rows)
    for r in rows:
        for k in _ROW_KEYS:
            if k not in r:
                raise KeyError(f"objective row missing '{k}'")
            if not np.isfinite(r[k]):
                raise ValueError(f"objective row has non-finite '{k}'")
    return rows


def tiered_legacy(rows):
    """Exact port of scripts/livesim_loop2.tiered: (score, tier1, risk, tier3). Kept for parity tests and old callers."""
    rows = _check(rows)
    t1 = float(np.mean([r["in_band"] for r in rows]))
    over = float(np.mean([r["over_band"] for r in rows]))
    risk = float(np.mean([r["worst5"] for r in rows])) + float(min(r["max_dd"] for r in rows)) / 4
    t3 = float(np.mean([r["pos_in_band"] for r in rows]))
    reached = t1 >= 0.5
    return (100 * min(t1, 0.5) - 50 * over + (10 * (risk + 0.3) + t3 if reached else 0.0)), t1, risk, t3


@dataclass(frozen=True)
class TierScore:
    t1: float                  # share of in-band weeks
    over: float
    risk: float                # higher is better (0 = no tail risk)
    t3: float
    reached: bool              # tier 1 satisfied, so tiers 2-3 count
    wiped: bool
    key: tuple = field(compare=False, default=())   # lexicographic integer key: compare THIS, never the floats
    soft: float = field(compare=False, default=0.0)  # smooth companion (legacy-shaped) for bootstraps and tie-breaks
    n_rows: int = 0

    @property
    def scalar(self):
        """One float that orders exactly like `key`; the floor -9 marks a wipe-out or no data."""
        if self.wiped or not self.key:
            return -9.0
        return float((self.key[0] * _RN + self.key[1]) * _TN + self.key[2])


def evaluate(rows, reach=REACH):
    """Score a set of window rows (dicts as from `week_row`, extra keys ignored). Empty set -> worst possible, never a crash."""
    rows = _check(rows)
    if not rows:
        return TierScore(0.0, 0.0, -RISK_FLOOR, 0.0, False, True, (), -9.0, 0)
    t1 = float(np.mean([r["in_band"] for r in rows]))
    over = float(np.mean([r["over_band"] for r in rows]))
    cat = float(np.mean([r.get("cat_rate", 0.0) for r in rows]))
    dd = float(min(r["max_dd"] for r in rows))
    risk = float(np.mean([r["worst5"] for r in rows])) + dd / 4 - cat - OVER_RISK_W * over
    t3 = float(np.mean([r["pos_in_band"] for r in rows]))
    reached = t1 >= reach
    wiped = dd < WIPEOUT
    i1 = int(np.floor(min(t1, reach) / T1_STEP + 1e-9))
    # tier 1 is also pushed down by overshoot (as in the legacy score), on the same grid
    i1 = max(0, i1 - int(np.floor(over / T1_STEP * 0.5 + 1e-9)))
    i2 = int(np.clip(np.floor((risk + RISK_FLOOR) / RISK_STEP + 1e-9), 0, _RN - 1)) if reached else 0
    i3 = int(np.clip(np.floor(t3 / T3_STEP + 1e-9), 0, _TN - 1)) if reached else 0
    soft = tiered_legacy(rows)[0]
    return TierScore(t1, over, risk, t3, reached, wiped, (i1, i2, i3), -9.0 if wiped else soft, len(rows))


def firewall(incumbent, candidate):
    """Does `candidate` replace `incumbent`? Returns (accept, reason). Decided by the highest tier where they differ."""
    if candidate.wiped or not candidate.key:
        return False, "candidate wiped out or has no data"
    if incumbent.wiped or not incumbent.key:
        return True, "incumbent wiped out or has no data"
    names = ("tier1 (weeks in band)", "tier2 (risk)", "tier3 (direction)")
    for n, a, b in zip(names, incumbent.key, candidate.key):
        if b > a:
            return True, f"better on {n}"
        if b < a:
            return False, f"worse on {n}"
    return False, "no measurable gain on any tier"


def compare(rows_a, rows_b):
    """+1 if set b beats set a lexicographically, -1 if worse, 0 if tied on the grid."""
    a, b = evaluate(rows_a), evaluate(rows_b)
    ka = a.key if a.key and not a.wiped else (-1, 0, 0)
    kb = b.key if b.key and not b.wiped else (-1, 0, 0)
    return (kb > ka) - (kb < ka)


def per_window_soft(rows):
    """Smooth legacy-shaped per-window score (diagnostic; jumps at the tier-1 threshold, so not used for inference)."""
    return np.array([tiered_legacy([r])[0] for r in _check(rows)])


def _tier_metric(r, tier):
    if tier == 0:
        return r["in_band"] - 0.5 * r["over_band"]
    if tier == 1:
        return r["worst5"] + r["max_dd"] / 4 - r.get("cat_rate", 0.0) - OVER_RISK_W * r["over_band"]
    return r["pos_in_band"]


def decisive_diffs(rows_a, rows_b):
    """Paired per-window gains of b over a on the tier that the firewall would decide on (the highest tier where the
    aggregate keys differ). Returns (tier index or None, diffs). Bootstrapping THIS, not a blended score, keeps the
    significance test lexicographic too. Rows must be paired window by window."""
    rows_a, rows_b = _check(rows_a), _check(rows_b)
    if len(rows_a) != len(rows_b):
        raise ValueError("decisive_diffs needs paired windows")
    ka, kb = evaluate(rows_a).key, evaluate(rows_b).key
    for tier, (x, y) in enumerate(zip(ka, kb)):
        if x != y:
            return tier, np.array([_tier_metric(b, tier) - _tier_metric(a, tier) for a, b in zip(rows_a, rows_b)])
    return None, np.zeros(len(rows_a))


def paired_bootstrap(diffs, n_boot=400, q=0.10, seed=0):
    """Lower q-quantile of the bootstrap mean of paired per-window differences. > 0 means the gain is not one lucky window."""
    d = np.asarray(diffs, float)
    if len(d) < 2:
        return float("-inf") if len(d) == 0 else float(d[0])
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return float(np.quantile(means, q))


def summary(score):
    """Plain-dict view of a TierScore for reports and experiment logs."""
    return {"key": list(score.key), "in_band": round(score.t1, 4), "over_band": round(score.over, 4),
            "risk": round(score.risk, 4), "pos_in_band": round(score.t3, 4), "tier1_reached": score.reached,
            "wiped": score.wiped, "n_windows": score.n_rows}


def by_group(rows, labels, reach=REACH):
    """Score each label's rows separately (per era, per strategy type). Groups are reported in sorted label order; a
    group with no rows is reported as no-data instead of being silently dropped."""
    rows = list(rows)
    if len(rows) != len(labels):
        raise ValueError("one label per row")
    out = {}
    for g in sorted(set(labels), key=str):
        out[g] = summary(evaluate([r for r, l in zip(rows, labels) if l == g], reach))
    return out


def rank_correlation_of_tiers(scores):
    """Spearman rho between the tier-1 and tier-2 components across a set of TierScores: a value near +1 means the two
    tiers are not really separate objectives in this sample (report it; a firewall over correlated tiers is idle)."""
    from scipy.stats import spearmanr
    a = [s.t1 for s in scores if not s.wiped]
    b = [s.risk for s in scores if not s.wiped]
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b)[0])
