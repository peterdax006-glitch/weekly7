"""Job-cost model and the PLACEMENT decision (MASTER_BLUEPRINT 7.10; owner 4 Oct 2026: "first it should see how much time it would take on
a PC ... if its gonna take like 5 hours then its probably best to add that to the list of things we need for a GPU, but if its capable of
bringing that time down by making it more efficient then maybe thats better").

1. COST MODEL  measured history -> seconds. Per job kind: seconds = fixed + rate x units (units = tokens, rows, steps ...), fitted on the
               records of that kind (least squares, both terms >= 0; a constant median when the units do not vary or are unknown). A kind
               with no history is priced from a 2% PILOT (fixed + (pilot - fixed) / fraction) and carries a wide band. predict() also
               applies the kind's recalibration factor from the predicted-vs-actual log.
2. PLACEMENT   place(): T_pc <= 30 min -> run on the PC at its priority. Longer -> value two alternatives in one currency (dollars):
               (a) make it faster on the PC (an improvement candidate: T_pc', effort), (b) a GPU WISHLIST entry {value, T_gpu, speed-up S,
               $ = T_gpu x rate + setup, deadline, deps}. Owner's rule of thumb: S >= 5 and T_pc >= ~5 h -> GPU unless (a) brings T_pc under
               ~30 min for less effort.
3. WISHLIST    a JSON file outside the repo; proposal() turns it into ONE digest entry when total value per $ passes a threshold (or a
               deadline needs it). Nothing here rents or contacts a GPU: the entry carries rents=False and waits for the owner's budget.
4. RECALIBRATE record_outcome() logs predicted vs actual per decision; calibration() reports the error per kind and the factor predict()
               multiplies by once a kind is persistently off.
Pure python, no model calls, no network."""
from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

PC_BACKGROUND_S = 30 * 60.0          # at or below this a job simply runs on the PC (blueprint 7.10 step 2)
GPU_MIN_SPEEDUP = 5.0                # owner's rule of thumb: S >= 5 ...
GPU_MIN_PC_S = 5 * 3600.0            # ... and T_pc >= ~5 h
PILOT_FRACTION = 0.02
PILOT_BAND = 2.0                     # a pilot-priced estimate is only good to a factor 2 either way
GPU_MIN_DOLLAR_SPEEDUP = 2.0         # below the owner's S >= 5 rule the dollar comparison may still pick the GPU, but only from S >= 2
PC_HOUR_USD = 0.30                   # what an hour of the (never idle) PC is worth; tunable, recorded in every decision
DEFAULT_GPU_USD_H = 0.47             # measured 3-4 Oct: ~$0.054 for the 412 s judge fine-tune (usd_spent_total deltas in the module logs)
DEFAULT_GPU_SETUP_S = 600.0          # boot + tunnel + prestage of the first base, paid once per rental
PROPOSE_VALUE_PER_USD = 3.0          # rental proposal when total wishlist value / total $ passes this
PROPOSE_MIN_USD = 0.25              # and the plan costs at least this (below it the PC/idle time is cheaper than a digest line)
RECAL_MIN_N = 3                      # outcomes before a kind gets a correction factor
RECAL_TRIGGER = 0.15                 # ... and only when its median error is above this


# ------------------------------------------------------------------------------------------------ records and fits

def record(kind: str, seconds: float, units: Optional[float] = None, where: str = "pc", **extra: Any) -> dict[str, Any]:
    r: dict[str, Any] = {"kind": kind, "seconds": float(seconds), "where": where}
    if units is not None:
        r["units"] = float(units)
    r.update(extra)
    return r


def _median(xs: Sequence[float]) -> float:
    return float(statistics.median(xs))


def _ols(pts: Sequence[tuple[float, float]]) -> Optional[tuple[float, float]]:
    """(fixed, rate) of seconds = fixed + rate x units, both >= 0; None when the points cannot support a slope."""
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    if sxx <= 1e-12 or sxx / n < (0.1 * max(mx, 1e-9)) ** 2:          # units must spread at least ~10% of their mean
        return None
    b = sum((x - mx) * (y - my) for x, y in pts) / sxx
    a = my - b * mx
    if b <= 0:
        return None
    if a < 0:                                                      # a negative overhead is noise: refit through the origin
        b = sum(x * y for x, y in pts) / max(1e-12, sum(x * x for x, _ in pts))
        a = 0.0
    return a, b


class Model:
    """One kind's fitted cost. mode: 'linear' (fixed + rate x units) | 'rate' (median seconds-per-unit x units) | 'const' (median seconds)."""

    def __init__(self, kind: str, recs: Sequence[Mapping[str, Any]], factor: float = 1.0):
        self.kind = kind
        self.n = len(recs)
        self.factor = factor
        secs = [float(r["seconds"]) for r in recs if float(r.get("seconds") or 0) > 0]
        pts = [(float(r["units"]), float(r["seconds"])) for r in recs if r.get("units") and float(r.get("seconds") or 0) > 0]
        self.fixed = 0.0
        self.rate = 0.0
        self.const = _median(secs) if secs else 0.0
        self.mode = "const"
        if len(pts) >= 3 and (fit := _ols(pts)):
            self.mode, (self.fixed, self.rate) = "linear", fit
        elif len(pts) >= 2 and len(pts) == len(secs):
            self.mode, self.rate = "rate", _median([y / x for x, y in pts])
        elif len(pts) == 1 and len(secs) == 1:
            self.mode, self.rate = "rate", pts[0][1] / pts[0][0]
        res = [abs(self._raw(r.get("units")) - float(r["seconds"])) / float(r["seconds"]) for r in recs if float(r.get("seconds") or 0) > 0]
        self.spread = _median(res) if res else 1.0                  # in-sample median relative error: the band of a prediction

    def _raw(self, units: Optional[float]) -> float:
        if self.mode == "const" or units is None:
            return self.const
        return self.fixed + self.rate * float(units)

    def predict(self, units: Optional[float] = None) -> dict[str, Any]:
        s = self._raw(units) * self.factor
        w = max(0.1, min(1.0, 1.5 * self.spread))
        return {"seconds": round(s, 2), "low": round(s * (1 - w), 2), "high": round(s * (1 + w), 2), "basis": f"history:{self.mode}",
                "n": self.n, "kind": self.kind}


def pilot_estimate(pilot_seconds: float, fraction: float = PILOT_FRACTION, fixed_s: float = 0.0, kind: str = "") -> dict[str, Any]:
    """No history: run `fraction` of the job (default 2%) and scale. `fixed_s` is the part that does not scale (load, setup)."""
    s = fixed_s + max(0.0, pilot_seconds - fixed_s) / max(1e-9, fraction)
    return {"seconds": round(s, 2), "low": round(s / PILOT_BAND, 2), "high": round(s * PILOT_BAND, 2), "basis": "pilot", "n": 0, "kind": kind}


class CostModel:
    """All kinds. predict(kind, units, pilot_seconds=None): history when present, else the pilot, else None."""

    def __init__(self, recs: Iterable[Mapping[str, Any]] = (), factors: Optional[Mapping[str, float]] = None):
        by: dict[str, list[Mapping[str, Any]]] = {}
        for r in recs:
            if float(r.get("seconds") or 0) > 0 and r.get("kind"):
                by.setdefault(str(r["kind"]), []).append(r)
        self.records = by
        self.models = {k: Model(k, v, (factors or {}).get(k, 1.0)) for k, v in by.items()}

    def kinds(self) -> list[str]:
        return sorted(self.models)

    def predict(self, kind: str, units: Optional[float] = None, pilot_seconds: Optional[float] = None, pilot_fixed_s: float = 0.0
                ) -> Optional[dict[str, Any]]:
        m = self.models.get(kind)
        if m is not None:
            return m.predict(units)
        if pilot_seconds is not None:
            return pilot_estimate(pilot_seconds, fixed_s=pilot_fixed_s, kind=kind)
        return None


def leave_one_out(recs: Iterable[Mapping[str, Any]], tol: float = 0.30, min_n: int = 2) -> dict[str, Any]:
    """Predict every record from the OTHER records of its kind. A kind is judged when it has >= min_n records; it passes when at least
    80% of its leave-one-out predictions are within `tol` (relative to the actual)."""
    by: dict[str, list[Mapping[str, Any]]] = {}
    for r in recs:
        if float(r.get("seconds") or 0) > 0:
            by.setdefault(str(r["kind"]), []).append(r)
    kinds: dict[str, Any] = {}
    for k, rs in sorted(by.items()):
        if len(rs) < min_n:
            kinds[k] = {"n": len(rs), "judged": False}
            continue
        errs = []
        for i, r in enumerate(rs):
            m = Model(k, rs[:i] + rs[i + 1:])
            p = m.predict(r.get("units"))["seconds"]
            errs.append(abs(p - float(r["seconds"])) / float(r["seconds"]))
        within = sum(1 for e in errs if e <= tol) / len(errs)
        kinds[k] = {"n": len(rs), "judged": True, "median_err": round(_median(errs), 3), "max_err": round(max(errs), 3),
                    "within_tol": round(within, 3), "pass": within >= 0.8, "mode": Model(k, rs).mode}
    judged = [v for v in kinds.values() if v["judged"]]
    allerr = [v["median_err"] for v in judged]
    return {"tol": tol, "kinds": kinds, "judged": len(judged), "passing": sum(1 for v in judged if v["pass"]),
            "median_of_median_err": round(_median(allerr), 3) if allerr else None}


def pc_history(state: Path) -> list[dict[str, Any]]:
    """The PC's own measured runs (creator.resources.record_task rows, with `units` when the caller gave them) as cost records."""
    from creator import resources as R
    return [record(str(r["kind"]), float(r["seconds"]), r.get("units"), "pc") for r in R._rows(Path(state) / R.TASKS_FILE)
            if r.get("kind") and float(r.get("seconds") or 0) > 0]


# ------------------------------------------------------------------------------------------------ placement

def gpu_usd(t_gpu_s: float, usd_h: float = DEFAULT_GPU_USD_H, setup_s: float = DEFAULT_GPU_SETUP_S, transfer_usd: float = 0.0) -> float:
    return round((t_gpu_s + setup_s) / 3600.0 * usd_h + transfer_usd, 4)


def place(job: str, kind: str, t_pc_s: float, *, t_gpu_s: Optional[float] = None, value_usd: float = 0.0, recur_per_week: float = 1.0,
          faster: Optional[Mapping[str, Any]] = None, deadline: str = "", deps: Sequence[str] = (), usd_h: float = DEFAULT_GPU_USD_H,
          setup_s: float = DEFAULT_GPU_SETUP_S, transfer_usd: float = 0.0, pc_hour_usd: float = PC_HOUR_USD, horizon_weeks: float = 4.0,
          effort_gpu_h: float = 1.0, basis: str = "") -> dict[str, Any]:
    """The 7.10 decision. `faster` = {name, t_pc_s (after the change), effort_h}: the improvement candidate (cache, smaller model, fewer
    rows ...). Returns {placement: 'pc' | 'pc_long' | 'faster' | 'gpu_wishlist', reason, options{...}, wishlist_entry?}; every field a
    later predicted-vs-actual check needs is in it."""
    d: dict[str, Any] = {"job": job, "kind": kind, "t_pc_s": round(t_pc_s, 1), "basis": basis, "pc_hour_usd": pc_hour_usd,
                         "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if t_pc_s <= PC_BACKGROUND_S:
        d.update(placement="pc", reason=f"T_pc {t_pc_s / 60:.1f} min <= {PC_BACKGROUND_S / 60:.0f} min: background on the PC")
        return d
    n = max(0.0, recur_per_week) * horizon_weeks if recur_per_week else 1.0
    n = max(n, 1.0)
    pc_cost = t_pc_s / 3600.0 * pc_hour_usd * n
    opts: dict[str, Any] = {"pc": {"usd": round(pc_cost, 3), "hours": round(t_pc_s / 3600.0 * n, 2)}}
    if faster:
        ta = float(faster["t_pc_s"])
        eff = float(faster.get("effort_h", 1.0))
        opts["faster"] = {"name": faster.get("name", ""), "t_pc_after_s": ta, "effort_h": eff,
                          "usd": round(eff * pc_hour_usd + ta / 3600.0 * pc_hour_usd * n, 3), "under_30min": ta <= PC_BACKGROUND_S}
    entry = None
    if t_gpu_s is not None:
        s = t_pc_s / max(1e-9, t_gpu_s)
        usd = gpu_usd(t_gpu_s, usd_h, setup_s, transfer_usd)
        entry = {"id": job, "kind": kind, "value_usd": round(value_usd or pc_cost, 3), "t_pc_s": round(t_pc_s, 1), "t_gpu_s": round(t_gpu_s, 1),
                 "speedup": round(s, 2), "usd": usd, "deadline": deadline, "deps": list(deps), "status": "wishlist"}
        if s >= GPU_MIN_DOLLAR_SPEEDUP:                              # a GPU that is barely faster than the PC is not worth the setup and transfer risk
            opts["gpu"] = {"usd": round(usd + effort_gpu_h * pc_hour_usd, 3), "speedup": round(s, 2), "t_gpu_s": round(t_gpu_s, 1)}
    rule_gpu = entry is not None and entry["speedup"] >= GPU_MIN_SPEEDUP and t_pc_s >= GPU_MIN_PC_S
    fast = opts.get("faster")
    fast_wins = bool(fast and fast["under_30min"] and ("gpu" not in opts or fast["effort_h"] <= effort_gpu_h or fast["usd"] <= opts["gpu"]["usd"]))
    if rule_gpu and not fast_wins:
        d.update(placement="gpu_wishlist", reason=f"S {entry['speedup']:.1f} >= {GPU_MIN_SPEEDUP:.0f} and T_pc {t_pc_s / 3600:.1f} h >= "
                 f"{GPU_MIN_PC_S / 3600:.0f} h: GPU wishlist", wishlist_entry=entry)
    elif fast_wins:
        d.update(placement="faster", reason=f"'{fast['name']}' brings T_pc to {fast['t_pc_after_s'] / 60:.0f} min (< 30) for effort "
                 f"{fast['effort_h']:.1f} h: build the improvement instead of renting", improvement=dict(fast))
        if entry:
            d["wishlist_entry"] = {**entry, "status": "alternative"}
    else:
        best = min(opts, key=lambda k: opts[k]["usd"])         # one currency: dollars
        if best == "gpu" and entry and "gpu" in opts:
            d.update(placement="gpu_wishlist", reason="cheapest option in dollars (S or T_pc below the rule of thumb)", wishlist_entry=entry)
        elif best == "faster" and fast:
            d.update(placement="faster", reason=f"cheapest in dollars: '{fast['name']}' ({fast['t_pc_after_s'] / 60:.0f} min after)", improvement=dict(fast))
        else:
            d.update(placement="pc_long", reason=f"T_pc {t_pc_s / 3600:.1f} h but no better option: run on the PC in slices at idle priority")
    d["options"] = opts
    return d


# ------------------------------------------------------------------------------------------------ wishlist and rental proposal

def load_wishlist(path: Path) -> dict[str, dict[str, Any]]:
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def add_wish(path: Path, entry: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    w = load_wishlist(path)
    w[str(entry["id"])] = dict(entry)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(w, indent=1, sort_keys=True), encoding="utf-8")
    return w


def proposal(wishlist: Mapping[str, Mapping[str, Any]], *, threshold: float = PROPOSE_VALUE_PER_USD, min_usd: float = PROPOSE_MIN_USD,
             deadline_days: Optional[float] = None, usd_h: float = DEFAULT_GPU_USD_H, setup_s: float = DEFAULT_GPU_SETUP_S,
             budget_usd: Optional[float] = None, plan_gpu_s: Optional[float] = None) -> Optional[dict[str, Any]]:
    """ONE digest entry (or None) when the open wishlist's value per $ passes `threshold`, or a deadline is within `deadline_days`.
    `plan_gpu_s` = the compiled package's own makespan (jobs overlap, so it is below the sum of the entries); used when every open entry is in the plan.
    Entries are ordered by value per $ and taken while their deps are inside the plan and the budget lasts. Never rents: rents is False
    and the entry only asks the owner for a budget."""
    open_ = [dict(e) for e in wishlist.values() if e.get("status", "wishlist") == "wishlist"]
    if not open_:
        return None
    open_.sort(key=lambda e: -float(e.get("value_usd", 0)) / max(1e-6, float(e.get("usd", 0.0)) or 1e-6))
    plan: list[dict[str, Any]] = []
    ids: set[str] = set()
    open_ids = {e["id"] for e in open_}
    t_gpu = 0.0
    progress = True
    while progress:                                   # dependencies first: an entry joins once every open dep is already in the plan
        progress = False
        for e in open_:
            if e["id"] in ids or any(d in open_ids and d not in ids for d in e.get("deps") or []):
                continue
            if budget_usd is not None and (t_gpu + float(e["t_gpu_s"]) + setup_s) / 3600.0 * usd_h > budget_usd:
                continue
            plan.append(e)
            ids.add(e["id"])
            t_gpu += float(e["t_gpu_s"])
            progress = True
    if not plan:
        return None
    if plan_gpu_s is not None and len(plan) == len(open_):
        t_gpu, setup_s = float(plan_gpu_s), 0.0
    total_usd = round((t_gpu + setup_s) / 3600.0 * usd_h, 3)
    total_value = round(sum(float(e.get("value_usd", 0)) for e in plan), 3)
    ratio = total_value / max(1e-9, total_usd)
    dl = [e["deadline"] for e in plan if e.get("deadline")]
    due = bool(dl) and deadline_days is not None and deadline_days <= 3
    if total_usd < min_usd and not due:
        return None
    if ratio < threshold and not due:
        return None
    return {"kind": "gpu_rental_proposal", "rents": False, "needs": "owner budget approval", "jobs": [e["id"] for e in plan],
            "t_gpu_h": round(t_gpu / 3600.0, 2), "usd": total_usd, "value_usd": total_value, "value_per_usd": round(ratio, 2),
            "trigger": "deadline" if due and ratio < threshold else "value_per_usd", "deadline": min(dl) if dl else ""}


# ------------------------------------------------------------------------------------------------ predicted vs actual

def record_outcome(path: Path, decision: Mapping[str, Any], actual_s: float, where: str = "pc") -> dict[str, Any]:
    p = float(decision.get("t_pc_s" if where == "pc" else "t_gpu_s", 0.0))
    r = {"job": decision.get("job"), "kind": decision.get("kind"), "where": where, "predicted_s": round(p, 2), "actual_s": round(float(actual_s), 2),
         "ratio": round(float(actual_s) / p, 4) if p > 0 else None, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "ab") as f:
        f.write((json.dumps(r, sort_keys=True) + "\n").encode("utf-8"))
    return r


def calibration(path: Path) -> dict[str, Any]:
    """Per kind: n, median |error|, share within 30%, and `factor` (median actual/predicted) once RECAL_MIN_N outcomes are persistently
    off by more than RECAL_TRIGGER; predict() callers multiply by it (CostModel(factors=...))."""
    by: dict[str, list[float]] = {}
    try:
        for ln in Path(path).read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if r.get("ratio"):
                by.setdefault(str(r.get("kind")), []).append(float(r["ratio"]))
    except OSError:
        pass
    out: dict[str, Any] = {}
    for k, xs in sorted(by.items()):
        err = [abs(x - 1.0) for x in xs]
        med = _median(xs)
        fac = med if len(xs) >= RECAL_MIN_N and abs(med - 1.0) > RECAL_TRIGGER and math.isfinite(med) else 1.0
        out[k] = {"n": len(xs), "median_err": round(_median(err), 3), "within_30pct": round(sum(1 for e in err if e <= 0.30) / len(err), 3),
                  "factor": round(fac, 4)}
    return out


def factors(path: Path) -> dict[str, float]:
    return {k: v["factor"] for k, v in calibration(path).items() if v["factor"] != 1.0}
