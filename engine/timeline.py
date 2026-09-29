"""Timeline dial and yearly pacing (Bible PHASE 17).

Inputs: regime, analog forecast, progress so far, weekly performance trajectory. Outputs: exposure, k, pool_q, brake.
The dial NEVER sees a date: it is fed only the completed weekly returns (whose count is the run's own age), a stress
reading and an analog forecast. Nothing here can be seasonal or calendar-driven; `step` rejects date-like inputs.
One scalar, aggressiveness a in [-1, 1], is moved by a rate-limited, dead-banded controller and mapped onto the knobs
inside hard bounds. Risk outranks pace: a drawdown brake caps aggressiveness whatever the pace signal says.
The dial is trained on earlier windows and judged on later ones; `promotion_gate` is the champion admission test
(improvement, no risk deterioration, no overfitting, stability across eras) and fails closed.
`simulate` uses a leverage PROXY on recorded weekly returns; only a full adaptive replay can prove the dial.
"""
from dataclasses import dataclass, asdict, replace
import datetime as _dt
import math
import numpy as np
import pandas as pd
from engine import objective as O

_DATES = (_dt.date, _dt.datetime, pd.Timestamp, np.datetime64)


@dataclass(frozen=True)
class DialParams:
    target: float = O.TARGET
    trail: int = 8                   # weeks of trajectory that matter
    ewma_half: float = 4.0           # weeks: half-life of the trajectory weights
    w_vol: float = 0.45              # weights of the four pace inputs
    w_prog: float = 0.20
    w_fc: float = 0.20
    w_reg: float = 0.15
    gain: float = 1.5
    deadband: float = 0.10           # ignore aggressiveness changes smaller than this
    max_step: float = 0.35           # largest aggressiveness move in one week
    over_limit: float = 0.40         # if this share of trailing weeks overshot 10%, force de-risking
    k_bounds: tuple = (1, 6)
    exposure_bounds: tuple = (0.5, 1.0)
    pool_bounds: tuple = (0.3, 0.95)
    brake_dd: float = 0.15           # drawdown from trailing peak that trips the brake
    brake_release: int = 3           # clean weeks before it lifts
    brake_exposure: float = 0.5
    brake_level: float = 0.15        # weekly-loss trigger handed to the trader while defensive
    horizon: int = 52                # weeks in a pacing year (the ~7%/week path is measured against this)

    def validate(self):
        if not (0 < self.target < 0.5 and self.trail >= 2 and self.ewma_half > 0):
            raise ValueError("bad pacing target/trail/half-life")
        if not (0 < self.max_step <= 2 and 0 <= self.deadband < 1):
            raise ValueError("bad controller rate limits")
        for lo, hi in (self.k_bounds, self.exposure_bounds, self.pool_bounds):
            if not lo <= hi:
                raise ValueError("bounds must be ordered")
        if self.horizon < 4:
            raise ValueError("pacing horizon too short")
        if self.k_bounds[0] < 1 or not 0 < self.exposure_bounds[0] or self.exposure_bounds[1] > 1.5:
            raise ValueError("bounds outside the safe envelope")
        return self


@dataclass(frozen=True)
class DialState:
    aggr: float = 0.0
    brake_on: bool = False
    clean: int = 0


@dataclass(frozen=True)
class DialOutput:
    exposure: float
    k: int
    pool_q: float
    brake: object                    # None, or the weekly-loss trigger
    aggr: float
    brake_on: bool
    terms: dict


def _reject_dates(x, name):
    if isinstance(x, _DATES) or isinstance(x, pd.DataFrame) or (isinstance(x, pd.Series) and isinstance(x.index, pd.DatetimeIndex)):
        raise TypeError(f"{name}: the dial may not use calendar information (pass plain values)")
    if isinstance(x, dict) and any(isinstance(v, _DATES) for v in x.values()):
        raise TypeError(f"{name}: the dial may not use calendar information")


def expected_abs_move(mu, sd):
    """E|X| for X ~ Normal(mu, sd): the size of move an analog forecast implies, direction ignored."""
    if sd <= 1e-12:
        return abs(mu)
    return sd * math.sqrt(2 / math.pi) * math.exp(-mu * mu / (2 * sd * sd)) + mu * math.erf(mu / (sd * math.sqrt(2)))


def _weights(n, half):
    w = 0.5 ** (np.arange(n)[::-1] / half)
    return w / w.sum()


def pacing_state(weeks, target=O.TARGET, horizon=52):
    """Where the run stands against the ~7%/week path (the designed yearly pacing). Uses only completed weeks; the week
    count is the run's own age. Returns: n, cum (compounded return so far), path (what the target path would be),
    gap (log cum - log path: negative = behind), weeks_behind (how many path-weeks that gap is), pace (mean week /
    target), required (mean weekly return over the remaining weeks that lands ON the path at the horizon; may be huge,
    and is reported, never chased), projected (year-end return if the mean week persists), remaining."""
    w = np.asarray(weeks, float).ravel()
    n = len(w)
    cum = float(np.prod(1 + np.maximum(w, -1.0))) if n else 1.0
    path = (1 + target) ** n
    lg = math.log(max(cum, 1e-12))
    gap = lg - n * math.log1p(target)
    left = max(horizon - n, 0)
    goal = (1 + target) ** horizon
    req = (goal / max(cum, 1e-12)) ** (1 / left) - 1 if left else float("nan")
    mean = float(w.mean()) if n else 0.0
    return {"n": n, "cum": cum - 1, "path": path - 1, "gap": gap, "weeks_behind": gap / math.log1p(target),
            "pace": mean / target, "required": req, "remaining": left, "projected": cum * (1 + mean) ** left - 1}


def regime_from_market(m, cap=5.0):
    """Turn the m_* market-context columns (a dict or one-row Series) into the dial's regime reading. stress is the
    short-vs-long fear ratio (m_vix_term; above 1 = fear rising), clipped to [0, cap]; missing or non-finite means
    neutral 1.0 so an absent feed can neither raise nor lower aggressiveness."""
    def num(k):
        try:
            v = float(m[k])
        except (KeyError, TypeError, ValueError):
            return float("nan")
        return v
    vt = num("m_vix_term")
    stress = float(np.clip(vt, 0.0, cap)) if np.isfinite(vt) else 1.0
    return {"stress": stress, "vix": num("m_vix"), "breadth": num("m_breadth")}


def forecast_from_analogs(res, beta=3.0, floor_sd=0.005):
    """Weekly forecast {"mean", "sd"} for the PORTFOLIO from an `Analogs.find()` result (market-level 1-month outcomes):
    mean = fwd_ret_1m / 4.33; sd = fwd_vol_1m / sqrt(52); both scaled by `beta`, the portfolio-to-market swing ratio
    (an assumption, measure it). The less the day resembles any past day (uniqueness above 1), the wider the sd: an
    unfamiliar market earns less trust, not more. None when there is no usable analog."""
    if not res or "prediction" not in res:
        return None
    pr = res["prediction"]
    if "fwd_ret_1m" not in pr or "fwd_vol_1m" not in pr:
        return None
    r, v = pr["fwd_ret_1m"], pr["fwd_vol_1m"]
    if not (np.isfinite(r) and np.isfinite(v)):
        return None
    widen = 1.0 + 0.25 * max(0.0, float(res.get("uniqueness", 0.0)) - 1.0)
    return {"mean": beta * r / 4.33, "sd": max(floor_sd, beta * v / math.sqrt(52) * widen)}


def pace_terms(weeks, regime, forecast, p):
    """Pace inputs in [-1, 1]; positive means 'too calm / behind, be bolder'. Also over-band share and trailing drawdown."""
    w = np.asarray(weeks, float)
    tail = w[-p.trail:]
    terms = {"vol": 0.0, "prog": 0.0, "fc": 0.0, "reg": 0.0, "over": 0.0, "dd": 0.0, "edge": 0.0}
    if len(tail):
        mag = float((_weights(len(tail), p.ewma_half) * np.abs(tail)).sum())
        terms["vol"] = float(np.clip((p.target - mag) / p.target, -1, 1))
        terms["over"] = float((np.abs(tail) > O.BAND[1]).mean())
        eq = np.concatenate([[1.0], np.cumprod(1 + np.maximum(tail, -1))])
        terms["dd"] = float(1 - (eq / np.maximum.accumulate(eq)).min())
    if len(w):
        ps = pacing_state(w, p.target, p.horizon)
        r = ps["pace"]                                     # progress: year-to-date mean week vs the ~7% path
        terms["edge"] = float(w.mean())
        terms["path_gap"] = ps["gap"]
        terms["prog"] = float(np.clip(1 - r, -1, 1)) if r >= 0 else -0.5   # losing: defend, do not gamble
    if forecast:
        fm = expected_abs_move(float(forecast.get("mean", 0.0)), float(forecast.get("sd", 0.0)))
        terms["fc"] = float(np.clip((p.target - fm) / p.target, -1, 1))
    if regime is not None:
        stress = float(regime.get("stress", 1.0)) if isinstance(regime, dict) else float(regime)
        terms["reg"] = float(-np.clip(stress - 1.0, -0.5, 1.0))
    return terms


def step(weeks, regime=None, forecast=None, state=None, params=None):
    """One weekly decision. `weeks`: completed weekly returns, oldest first. Returns (DialOutput, next DialState)."""
    p = (params or DialParams()).validate()
    state = state or DialState()
    for x, n in ((weeks, "weeks"), (regime, "regime"), (forecast, "forecast")):
        _reject_dates(x, n)
    w = np.asarray(weeks, float).ravel()
    if not np.isfinite(w).all():
        raise ValueError("weekly returns contain NaN/inf")
    t = pace_terms(w, regime, forecast, p)
    goal = math.tanh(p.gain * (p.w_vol * t["vol"] + p.w_prog * t["prog"] + p.w_fc * t["fc"] + p.w_reg * t["reg"]))
    if t["over"] >= p.over_limit:
        goal = min(goal, -0.25 - 0.5 * t["over"])            # overshooting the band is a risk, not a success
    a = state.aggr
    if abs(goal - a) >= p.deadband:
        a = a + float(np.clip(goal - a, -p.max_step, p.max_step))
    brake_on, clean = state.brake_on, state.clean
    if t["dd"] >= p.brake_dd:
        brake_on, clean = True, 0
    elif brake_on:
        clean += 1
        if clean >= p.brake_release:
            brake_on, clean = False, 0
    if brake_on:
        a = min(a, -0.5)                                     # risk outranks pace
    a = float(np.clip(a, -1, 1))
    u = (a + 1) / 2
    (k_lo, k_hi), (e_lo, e_hi), (q_lo, q_hi) = p.k_bounds, p.exposure_bounds, p.pool_bounds
    k = int(np.clip(round(k_hi - u * (k_hi - k_lo)), k_lo, k_hi))   # fewer names = bigger swings (C21)
    exposure = e_lo + u * (e_hi - e_lo)
    if brake_on:
        exposure = max(e_lo * p.brake_exposure, exposure * p.brake_exposure)
    pool_q = q_lo + u * (q_hi - q_lo)
    brake = p.brake_level if (brake_on or a < 0) else None
    return DialOutput(float(exposure), k, float(pool_q), brake, a, brake_on, {**t, "goal": goal}), DialState(a, brake_on, clean)


def cfg_overrides(out):
    """The cfg keys the trader already understands (k, pool_q, brake); exposure is applied as a gross-weight scale."""
    return {"k": out.k, "pool_q": out.pool_q, "brake": out.brake}


def simulate(base_weeks, params=None, regimes=None, forecasts=None, k_ref=3, elasticity=0.5):
    """Replay a recorded weekly series under the dial. Leverage proxy: exposure * (k_ref/k)**elasticity applied to the
    recorded week. Each decision uses only weeks BEFORE the one it scales. Returns (dial weeks, list of outputs)."""
    p = (params or DialParams()).validate()
    base = np.asarray(base_weeks, float)
    st, hist, outs, res = DialState(), [], [], []
    for i, b in enumerate(base):
        out, st = step(hist, None if regimes is None else regimes[i], None if forecasts is None else forecasts[i], st, p)
        res.append(float(max(-1.0, b * out.exposure * (k_ref / out.k) ** elasticity)))
        outs.append(out)
        hist.append(res[-1])
    return np.array(res), outs


DIAL_SPACE = {"gain": [0.75, 1.0, 1.5, 2.0, 3.0], "deadband": [0.05, 0.10, 0.20], "max_step": [0.2, 0.35, 0.5],
              "ewma_half": [2.0, 4.0, 8.0], "w_vol": [0.3, 0.45, 0.6], "w_prog": [0.1, 0.2, 0.3], "over_limit": [0.3, 0.4, 0.5],
              "brake_dd": [0.10, 0.15, 0.20], "trail": [6, 8, 12]}


def _rows(windows, params, **kw):
    return [O.week_row(simulate(w["weeks"], params, w.get("regime"), w.get("forecast"), **kw)[0]) for w in windows]


def train_dial(windows, n_draws=40, seed=0, **kw):
    """Random search over DIAL_SPACE on the given (earlier) windows by the tiered objective. Default params are always a
    contender, so training only moves away from them for a measured reason."""
    if not windows:
        raise ValueError("no training windows")
    rng = np.random.default_rng(seed)
    best_p = DialParams()
    best = O.evaluate(_rows(windows, best_p, **kw))
    for _ in range(n_draws):
        p = replace(DialParams(), **{k: v[rng.integers(len(v))] for k, v in DIAL_SPACE.items()})
        s = O.evaluate(_rows(windows, p, **kw))
        if (s.key, s.soft) > (best.key, best.soft):
            best_p, best = p, s
    return best_p, best


@dataclass
class GateReport:
    passed: bool
    checks: dict
    reasons: list
    numbers: dict

    def summary(self):
        return "PASS" if self.passed else "FAIL: " + "; ".join(self.reasons)


def promotion_gate(train_windows, test_windows, params, n_eras=3, risk_tol=0.01, dd_tol=0.05, keep_ratio=0.3, seed=0, **kw):
    """Champion admission (Bible Phase 17). Windows carry 'weeks' and optionally 'end'; train must precede test (checked
    when 'end' is present). Compares the dial with the undialled series on the unseen later windows. Fails closed."""
    train_windows, test_windows = list(train_windows), list(test_windows)
    if train_windows and test_windows and all(w.get("end") is not None for w in train_windows + test_windows):
        if max(w["end"] for w in train_windows) > min(w["end"] for w in test_windows):
            raise ValueError("training windows overlap or follow the test windows: look-ahead")
    if len(test_windows) < 3 or not train_windows:
        return GateReport(False, {}, ["insufficient windows to judge (need >=1 train, >=3 test)"], {})
    if all(w.get("end") is not None for w in test_windows):
        test_windows.sort(key=lambda w: w["end"])
    base_r = [O.week_row(w["weeks"]) for w in test_windows]
    dial_r = _rows(test_windows, params, **kw)
    tr_base = [O.week_row(w["weeks"]) for w in train_windows]
    tr_dial = _rows(train_windows, params, **kw)
    return gate_rows(tr_base, tr_dial, base_r, dial_r, asdict(params), n_eras, risk_tol, dd_tol, keep_ratio, seed)


def gate_rows(tr_base, tr_dial, base_r, dial_r, params=None, n_eras=3, risk_tol=0.01, dd_tol=0.05, keep_ratio=0.3, seed=0):
    """The admission checks on already-computed window rows (test rows in chronological order), so the same gate judges
    both the proxy `simulate` and a real adaptive replay. Rows must be paired window by window."""
    if len(base_r) != len(dial_r) or len(tr_base) != len(tr_dial):
        raise ValueError("base and dial rows must be paired")
    if len(base_r) < 3 or not tr_base:
        return GateReport(False, {}, ["insufficient windows to judge (need >=1 train, >=3 test)"], {})
    checks, reasons = {}, []
    sb, sd_ = O.evaluate(base_r), O.evaluate(dial_r)
    tier, diffs = O.decisive_diffs(base_r, dial_r)
    lo = O.paired_bootstrap(diffs, seed=seed)
    ok, why = O.firewall(sb, sd_)
    checks["improvement"] = bool(ok and lo > 0)
    if not checks["improvement"]:
        reasons.append(f"no reliable improvement ({why}; bootstrap lower bound {lo:+.3f})")
    dd_b, dd_d = min(r["max_dd"] for r in base_r), min(r["max_dd"] for r in dial_r)
    new_flags = [f for f in O.gaming_flags(dial_r) if f not in O.gaming_flags(base_r)]
    checks["not_gaming"] = not new_flags
    if new_flags:
        reasons.append(f"dial looks better by gaming the objective: {new_flags}")
    checks["risk"] = bool(sd_.risk >= sb.risk - risk_tol and dd_d >= dd_b - dd_tol and
                          np.mean([r["cat_rate"] for r in dial_r]) <= np.mean([r["cat_rate"] for r in base_r]) + 1e-9)
    if not checks["risk"]:
        reasons.append(f"risk deteriorated (risk {sb.risk:+.3f} -> {sd_.risk:+.3f}, max DD {dd_b:.0%} -> {dd_d:.0%})")
    gain_tr = float(np.mean(O.decisive_diffs(tr_base, tr_dial)[1]))
    gain_te = float(diffs.mean())
    checks["no_overfit"] = bool(gain_te > 0 and (gain_tr <= 0 or gain_te >= keep_ratio * gain_tr))
    if not checks["no_overfit"]:
        reasons.append(f"train gain {gain_tr:+.3f} did not carry to test {gain_te:+.3f}")
    em = [float(e.mean()) for e in np.array_split(diffs, n_eras)] if len(diffs) >= n_eras else []
    checks["stable_eras"] = bool(em and sum(m > 0 for m in em) >= n_eras - 1 and min(em) > -abs(gain_te))
    if not checks["stable_eras"]:
        reasons.append(f"not stable across eras (per-era gain {[round(m, 3) for m in em]})")
    nums = dict(gain_train=gain_tr, gain_test=gain_te, boot_lo=lo, era_gain=em, base_key=sb.key, dial_key=sd_.key,
                params=params)
    return GateReport(all(checks.values()), checks, reasons, nums)


class DialAdmission:
    """The champion door for the dial (Phase 17: 'NOT allowed into the champion until it proves ...'). `admit` needs a
    GateReport that passed every check; anything else is refused and the refusal is recorded. Entries are append-only and
    hash-chained, so the record of what was admitted (and why not) cannot be quietly rewritten. State is a JSON file."""

    def __init__(self, path=None):
        self.path = None if path is None else __import__("pathlib").Path(path)
        self.log = []
        if self.path is not None and self.path.exists():
            self.log = __import__("json").loads(self.path.read_text())

    @staticmethod
    def _h(prev, body):
        import hashlib, json
        return hashlib.sha256((prev + json.dumps(body, sort_keys=True, default=str)).encode()).hexdigest()[:16]

    def _append(self, body):
        prev = self.log[-1]["hash"] if self.log else ""
        self.log.append({**body, "n": len(self.log), "hash": self._h(prev, body)})
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(__import__("json").dumps(self.log, indent=1, default=str))

    def admit(self, params, report, label=""):
        need = ("improvement", "risk", "no_overfit", "stable_eras", "not_gaming")
        missing = [c for c in need if not report.checks.get(c)]
        if not report.passed or missing:
            self._append({"action": "refused", "label": label, "why": report.reasons or [f"unproven: {missing}"], "params": asdict(params)})
            return False
        self._append({"action": "admitted", "label": label, "params": asdict(params), "numbers": report.numbers})
        return True

    def champion(self):
        """Params of the latest admission that has not been revoked (None if there is none)."""
        cur = None
        for e in self.log:
            if e["action"] == "admitted":
                cur = e["params"]
            elif e["action"] == "revoked":
                cur = None
        return None if cur is None else DialParams(**{k: tuple(v) if isinstance(v, list) else v for k, v in cur.items()})

    def revoke(self, why):
        self._append({"action": "revoked", "why": why})

    def verify(self):
        prev = ""
        for e in self.log:
            body = {k: v for k, v in e.items() if k not in ("n", "hash")}
            if e.get("hash") != self._h(prev, body):
                return False
            prev = e["hash"]
        return True
