"""Path taxonomy: what a mover did NEXT (canon C67; contract C66 sections 4, 5, 9). IMPLEMENTED - NOT VALIDATED.

For every episode from engine.research.episodes, look 1, 2, 3 and 5 sessions forward and label the path. All thresholds are PARAMETERS measured in
the name's OWN trailing volatility (sigma, through the day before the move) and in the move day's own true range, so a sleepy utility and a
biotech are judged by their own yardstick:

  SPIKED_NEXT_DAY   the next session moved far the same way (signed next-day return >= spike_k sigma and >= spike_min)
  REVERSED_NEXT_DAY the next session moved far the other way
  CONTINUED         (h>1) neither of the above, but the whole window drifted far the same way
  RETRACED          (h>1) neither of the above, but the whole window drifted far back
  EXPANDED          the true range got way bigger: mean true range over the window >= expand_ratio x the move day's
  CONSOLIDATED      the true range contracted while the move held: mean true range <= contract_ratio x the move day's
  STOPPED           no follow-through: net signed move within stop_k sigma sqrt(h), range neither expanded nor contracted
  UNCLASSIFIED      everything else - deliberately. A path that fits no rule is not forced into one.

Intraday variants (same module, separate columns): 'volatile during the day, then got way more volatile' is the day-over-day true-range
ratio (RANGE_EXPANSION / RANGE_CONTRACTION); a gap that faded is GAP_AND_FADE (the day gave back at least fade_frac of the opening gap) and a
gap that kept going is GAP_AND_GO (closed beyond its open, high in the range). Gap type uses only the move day itself; what the NEXT day did
about it is reported in units of sigma.

A path whose window is not fully observed (end of data, halted names) is flagged ok_h=False and labelled UNCLASSIFIED - never guessed.
Everything here is MATURED_RESEARCH_STATE: the labels use the future by definition. `matured_at` is the date the longest window closed;
only records whose matured_at is strictly before `now` may be handed on (episodes.assert_no_future / MaturedRecord.gate)."""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from engine.research.core import FirewallBreach, Namespace, _StrEnum, as_date, stable_hash
from engine.research.episodes import EpisodeConfig, EpisodeError, Grid, lens_band_side, type_name

NAMESPACE = Namespace.MATURED_RESEARCH


class PathClass(_StrEnum):
    CONSOLIDATED = "CONSOLIDATED"
    EXPANDED = "EXPANDED"
    STOPPED = "STOPPED"
    SPIKED_NEXT_DAY = "SPIKED_NEXT_DAY"
    REVERSED_NEXT_DAY = "REVERSED_NEXT_DAY"
    CONTINUED = "CONTINUED"
    RETRACED = "RETRACED"
    UNCLASSIFIED = "UNCLASSIFIED"


class IntradayPath(_StrEnum):
    RANGE_EXPANSION = "RANGE_EXPANSION"          # volatile during the day, then got way more volatile
    RANGE_CONTRACTION = "RANGE_CONTRACTION"
    RANGE_SAME = "RANGE_SAME"
    UNKNOWN = "UNKNOWN"


class GapType(_StrEnum):
    GAP_AND_GO = "GAP_AND_GO"
    GAP_AND_FADE = "GAP_AND_FADE"
    GAP_HOLD = "GAP_HOLD"
    NO_GAP = "NO_GAP"


CLASS_CODES = {c.value: i for i, c in enumerate(PathClass)}
CLASS_ORDER = tuple(c.value for c in PathClass)


@dataclasses.dataclass(frozen=True)
class PathConfig:
    """Thresholds. `*_k` are multiples of the name's trailing daily sigma (scaled by sqrt(h) over a window of h sessions)."""
    spike_k: float = 1.5
    spike_min: float = 0.02                  # a spike must also be an absolute 2% move: sigma can be tiny for sleepy names
    reverse_k: float = 1.5
    reverse_min: float = 0.02
    cont_k: float = 2.0
    expand_ratio: float = 1.5
    contract_ratio: float = 0.5
    stop_k: float = 1.0
    intraday_expand: float = 1.5             # next-day true range / move-day true range
    intraday_contract: float = 0.5
    gap_min: float = 0.02                    # smaller opening gaps are NO_GAP
    fade_frac: float = 0.5                   # share of the gap given back between open and close
    go_loc: float = 0.66                     # close position in the range needed for GAP_AND_GO (mirrored for gap down)
    tr_floor: float = 0.005

    def validate(self) -> list[str]:
        errs = []
        for f in ("spike_k", "reverse_k", "cont_k", "expand_ratio", "stop_k", "spike_min", "reverse_min"):
            if getattr(self, f) <= 0:
                errs.append(f"{f} must be > 0")
        if not (0 < self.contract_ratio < self.expand_ratio):
            errs.append("need 0 < contract_ratio < expand_ratio")
        if not (0 < self.intraday_contract < self.intraday_expand):
            errs.append("need 0 < intraday_contract < intraday_expand")
        if not (0 <= self.fade_frac <= 1.5) or not (0.5 <= self.go_loc <= 1.0):
            errs.append("fade_frac / go_loc out of range")
        if self.stop_k >= self.spike_k or self.stop_k >= self.reverse_k:
            errs.append("stop_k must sit below spike_k and reverse_k or STOPPED overlaps SPIKED")
        return errs

    def require_valid(self) -> "PathConfig":
        errs = self.validate()
        if errs:
            raise EpisodeError("; ".join(errs))
        return self

    def digest(self) -> str:
        return stable_hash(dataclasses.asdict(self), 12)


# ------------------------------------------------------------------------------------------------------------------ forward windows
@dataclasses.dataclass
class Forward:
    """Forward bars of every episode, shape (n_episodes, H+1); column 0 is the move day, column k the k-th session after it. NaN where the
    session does not exist or the name had no bar."""
    C: np.ndarray
    O: np.ndarray
    H: np.ndarray
    L: np.ndarray
    date: np.ndarray                         # datetime64 of each column; NaT past the end of the data
    exists: np.ndarray                       # bool, the session exists in the grid

    def tr(self, k: int) -> np.ndarray:
        """True range of column k as a fraction of the previous close (includes the opening gap)."""
        prev = self.C[:, k - 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            return (np.maximum(self.H[:, k], prev) - np.minimum(self.L[:, k], prev)) / prev


def gather_forward(grid: Grid, ti: np.ndarray, nj: np.ndarray, horizon: int) -> Forward:
    T = grid.shape[0]
    idx = ti[:, None] + np.arange(horizon + 1)[None, :]
    exists = idx < T
    ic = np.minimum(idx, T - 1)
    nn = nj[:, None]
    pick = lambda a: np.where(exists, a[ic, nn], np.nan)
    dates = grid.dates.to_numpy()
    return Forward(pick(grid.C), pick(grid.O), pick(grid.H), pick(grid.L),
                   np.where(exists, dates[ic], np.datetime64("NaT")), exists)


# ------------------------------------------------------------------------------------------------------------------ gap type (same day only)
def classify_gap(gap: np.ndarray, o: np.ndarray, c: np.ndarray, prev_c: np.ndarray, loc: np.ndarray, pc: PathConfig) -> np.ndarray:
    """GAP_AND_GO / GAP_AND_FADE / GAP_HOLD / NO_GAP for move-day bars. Uses the move day's own open/close/location only."""
    out = np.full(len(gap), GapType.NO_GAP.value, dtype=object)
    with np.errstate(invalid="ignore", divide="ignore"):
        sign = np.sign(gap)
        big = np.isfinite(gap) & (np.abs(gap) >= pc.gap_min)
        gap_abs = np.abs(o - prev_c)
        given_back = np.where(gap_abs > 0, -sign * (c - o) / gap_abs, 0.0)       # 1.0 = closed back at the previous close
        favourable_loc = np.where(sign > 0, loc, 1.0 - loc)
        go = big & (given_back <= 0.0) & (favourable_loc >= pc.go_loc)
        fade = big & (given_back >= pc.fade_frac)
    out[big] = GapType.GAP_HOLD.value
    out[go] = GapType.GAP_AND_GO.value
    out[fade] = GapType.GAP_AND_FADE.value
    return out


def classify_intraday(ratio: np.ndarray, pc: PathConfig) -> np.ndarray:
    """Day-over-day true-range ratio -> RANGE_EXPANSION / RANGE_CONTRACTION / RANGE_SAME (UNKNOWN where the ratio is missing)."""
    out = np.full(len(ratio), IntradayPath.UNKNOWN.value, dtype=object)
    ok = np.isfinite(ratio)
    out[ok] = IntradayPath.RANGE_SAME.value
    out[ok & (ratio >= pc.intraday_expand)] = IntradayPath.RANGE_EXPANSION.value
    out[ok & (ratio <= pc.intraday_contract)] = IntradayPath.RANGE_CONTRACTION.value
    return out


# ------------------------------------------------------------------------------------------------------------------ the classifier
def classify_window(side: np.ndarray, sigma: np.ndarray, r1: np.ndarray, net: np.ndarray, ratio: np.ndarray, h: int,
                    pc: PathConfig, ok: np.ndarray) -> np.ndarray:
    """Labels for one horizon. `r1` and `net` are SIGNED (positive = the move's own direction); rules apply in priority order and the first one
    that fires wins, so every episode gets exactly one class."""
    n = len(side)
    label = np.full(n, PathClass.UNCLASSIFIED.value, dtype=object)
    done = ~ok                                                  # unobservable windows stay UNCLASSIFIED
    with np.errstate(invalid="ignore", divide="ignore"):
        r1z = r1 / sigma
        netz = net / (sigma * np.sqrt(h))
        rules = [
            (PathClass.SPIKED_NEXT_DAY, (r1 >= pc.spike_min) & (r1z >= pc.spike_k)),
            (PathClass.REVERSED_NEXT_DAY, (r1 <= -pc.reverse_min) & (r1z <= -pc.reverse_k)),
            (PathClass.CONTINUED, (netz >= pc.cont_k) if h > 1 else np.zeros(n, bool)),
            (PathClass.RETRACED, (netz <= -pc.cont_k) if h > 1 else np.zeros(n, bool)),
            (PathClass.EXPANDED, ratio >= pc.expand_ratio),
            (PathClass.CONSOLIDATED, ratio <= pc.contract_ratio),
            (PathClass.STOPPED, np.abs(netz) <= pc.stop_k),
        ]
    for cls, cond in rules:
        hit = cond & ~done & np.isfinite(sigma)
        label[hit] = cls.value
        done |= hit
    return label


def label_paths(grid: Grid, eps: pd.DataFrame, pc: PathConfig = PathConfig(), horizons: Sequence[int] | None = None) -> pd.DataFrame:
    """Path labels for every episode row (same order and length as `eps`). Columns:
       gap_type, r1, r1z, gap1z, tr_t, rr1, intraday_path, matured_at,
       and per horizon h: cls_h, ok_h, net_h, netz_h, rr_h, mfe_h, mae_h, tmfe_h (session of the best favourable excursion),
       retrig_h (True when the window itself contains another move of at least the lower band edge - the path is then partly a NEW episode)
       (signed: positive = the move's own direction)."""
    pc.require_valid()
    hs = tuple(horizons) if horizons is not None else grid.cfg.horizons
    if list(hs) != sorted(set(hs)) or hs[0] < 1:
        raise EpisodeError("horizons must be strictly increasing positive integers")
    n = len(eps)
    cols = ["gap_type", "r1", "r1z", "gap1z", "tr_t", "rr1", "intraday_path", "matured_at"]
    for h in hs:
        cols += [f"cls_{h}", f"ok_{h}", f"net_{h}", f"netz_{h}", f"rr_{h}", f"mfe_{h}", f"mae_{h}", f"tmfe_{h}", f"retrig_{h}"]
    if n == 0:
        return pd.DataFrame({c: pd.Series(dtype=object if c in ("gap_type", "intraday_path") or c.startswith("cls_") else "float64") for c in cols})
    ti, nj = eps["ti"].to_numpy(), eps["nj"].to_numpy()
    side = eps["side"].to_numpy().astype(float)
    side_lens = np.where(side == 0, np.nan, side)
    sigma = eps["sigma"].to_numpy()
    Hmax = int(hs[-1])
    fw = gather_forward(grid, ti, nj, Hmax)
    prev_c = eps["prev_c"].to_numpy()
    c_t = fw.C[:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        tr_t = (np.maximum(fw.H[:, 0], prev_c) - np.minimum(fw.L[:, 0], prev_c)) / prev_c
        tr_ref = np.maximum(tr_t, pc.tr_floor)
        out: dict[str, Any] = {"gap_type": classify_gap(eps["gap"].to_numpy(), fw.O[:, 0], c_t, prev_c, eps["loc"].to_numpy(), pc),
                               "tr_t": tr_t}
        r1 = side_lens * (fw.C[:, 1] / c_t - 1.0)
        out["r1"] = r1
        out["r1z"] = r1 / sigma
        out["gap1z"] = side_lens * (fw.O[:, 1] / c_t - 1.0) / sigma
        rr1 = fw.tr(1) / tr_ref
        out["rr1"] = rr1
        out["intraday_path"] = classify_intraday(rr1, pc)
        trs = np.stack([fw.tr(k) for k in range(1, Hmax + 1)], axis=1)               # (n, Hmax)
        dret = fw.C[:, 1:] / fw.C[:, :-1] - 1.0                                      # raw daily returns after the move day
        cum = side_lens[:, None] * (fw.C[:, 1:] / c_t[:, None] - 1.0)                # signed net move after k sessions
        hi_exc = side_lens[:, None] * (np.where(side[:, None] > 0, fw.H[:, 1:], fw.L[:, 1:]) / c_t[:, None] - 1.0)
        lo_exc = side_lens[:, None] * (np.where(side[:, None] > 0, fw.L[:, 1:], fw.H[:, 1:]) / c_t[:, None] - 1.0)
    seen_end = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    for h in hs:
        window_ok = np.all(np.isfinite(trs[:, :h]), axis=1) & np.all(np.isfinite(cum[:, :h]), axis=1) & np.isfinite(sigma) & (side != 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            net = cum[:, h - 1]
            ratio = trs[:, :h].mean(axis=1) / tr_ref
            mfe = np.fmax.reduce(hi_exc[:, :h], axis=1)
            mae = np.fmin.reduce(lo_exc[:, :h], axis=1)
        out[f"ok_{h}"] = window_ok
        out[f"net_{h}"] = np.where(window_ok, net, np.nan)
        out[f"netz_{h}"] = np.where(window_ok, net / (sigma * np.sqrt(h)), np.nan)
        out[f"rr_{h}"] = np.where(window_ok, ratio, np.nan)
        out[f"mfe_{h}"] = np.where(window_ok, mfe / sigma, np.nan)
        out[f"mae_{h}"] = np.where(window_ok, mae / sigma, np.nan)
        out[f"tmfe_{h}"] = np.where(window_ok, np.argmax(np.where(np.isfinite(hi_exc[:, :h]), hi_exc[:, :h], -np.inf), axis=1) + 1, np.nan)
        out[f"retrig_{h}"] = window_ok & np.any(np.abs(dret[:, :h]) >= grid.cfg.lo, axis=1)
        out[f"cls_{h}"] = classify_window(side, sigma, np.where(window_ok, r1, np.nan), np.where(window_ok, net, np.nan),
                                          np.where(window_ok, ratio, np.nan), h, pc, window_ok)
        seen_end = np.where(window_ok, fw.date[:, h], seen_end)
    out["matured_at"] = seen_end
    return pd.DataFrame({c: out[c] for c in cols})


def with_paths(eps: pd.DataFrame, paths: pd.DataFrame) -> pd.DataFrame:
    """Episodes joined to their labels (positional, lengths must match)."""
    if len(eps) != len(paths):
        raise EpisodeError(f"episodes ({len(eps)}) and paths ({len(paths)}) differ in length")
    return pd.concat([eps.reset_index(drop=True), paths.reset_index(drop=True)], axis=1)


def observable(paths: pd.DataFrame, now) -> pd.DataFrame:
    """Rows whose longest window closed strictly before `now`. Anything else is not yet a fact (fail-closed filter, not an exception)."""
    if len(paths) == 0:
        return paths
    m = pd.to_datetime(paths["matured_at"])
    return paths[(m.notna()) & (m < pd.Timestamp(as_date(now)))]


def require_matured(paths: pd.DataFrame, now, what: str = "path labels") -> None:
    """Raise FirewallBreach if any labelled row matures on or after `now`."""
    m = pd.to_datetime(paths["matured_at"]).dropna() if len(paths) else pd.Series(dtype="datetime64[ns]")
    if len(m) and m.max() >= pd.Timestamp(as_date(now)):
        raise FirewallBreach(f"{what}: a window closes {m.max().date()} which is not strictly before now={as_date(now)}")


# ------------------------------------------------------------------------------------------------------------------ summaries
def class_table(ep: pd.DataFrame, lens: str, horizon: int) -> pd.DataFrame:
    """Per (type, class): count, share within the type, mean signed net move (in sigma), mean range ratio. `ep` is episodes joined to paths."""
    col = f"cls_{horizon}"
    if len(ep) == 0 or col not in ep:
        return pd.DataFrame(columns=["type", "cls", "n", "share", "mean_netz", "mean_rr"])
    band, side = lens_band_side(ep, lens)
    d = pd.DataFrame({"type": [type_name(b, s, lens) for b, s in zip(band, side)], "cls": ep[col].to_numpy(),
                      "ok": ep[f"ok_{horizon}"].to_numpy(), "netz": ep[f"netz_{horizon}"].to_numpy(), "rr": ep[f"rr_{horizon}"].to_numpy()})
    d = d[(band > 0) & d["ok"]]
    if len(d) == 0:
        return pd.DataFrame(columns=["type", "cls", "n", "share", "mean_netz", "mean_rr"])
    g = d.groupby(["type", "cls"]).agg(n=("netz", "size"), mean_netz=("netz", "mean"), mean_rr=("rr", "mean")).reset_index()
    g["share"] = g["n"] / g.groupby("type")["n"].transform("sum")
    return g.sort_values(["type", "n"], ascending=[True, False]).reset_index(drop=True)


def transition_matrix(ep: pd.DataFrame, h_from: int, h_to: int) -> pd.DataFrame:
    """How labels at one horizon turn into labels at a longer one (rows: class at h_from; columns: class at h_to; row-normalised)."""
    if len(ep) == 0:
        return pd.DataFrame()
    ok = ep[f"ok_{h_from}"] & ep[f"ok_{h_to}"]
    if not ok.any():
        return pd.DataFrame()
    t = pd.crosstab(ep.loc[ok, f"cls_{h_from}"], ep.loc[ok, f"cls_{h_to}"])
    return t.div(t.sum(axis=1), axis=0)


def base_rates(ep: pd.DataFrame, horizon: int) -> dict[str, float]:
    """Share of each class among observable paths - the baseline any precursor has to beat, per horizon."""
    col, ok = f"cls_{horizon}", f"ok_{horizon}"
    if len(ep) == 0 or not ep[ok].any():
        return {c: float("nan") for c in CLASS_ORDER}
    v = ep.loc[ep[ok], col].value_counts(normalize=True)
    return {c: float(v.get(c, 0.0)) for c in CLASS_ORDER}


def gap_table(ep: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Gap-and-go vs gap-and-fade vs hold: what the next session did (mean signed next-day move in sigma, share reversed/spiked)."""
    if len(ep) == 0 or "gap_type" not in ep:
        return pd.DataFrame()
    d = ep[ep[f"ok_{horizon}"] & (ep["gap_type"] != GapType.NO_GAP.value)]
    if len(d) == 0:
        return pd.DataFrame()
    g = d.groupby("gap_type").agg(n=("r1z", "size"), mean_r1z=("r1z", "mean"), mean_gap1z=("gap1z", "mean"), mean_rr1=("rr1", "mean"))
    g["share_spiked"] = d.groupby("gap_type")[f"cls_{horizon}"].apply(lambda s: float((s == PathClass.SPIKED_NEXT_DAY.value).mean()))
    g["share_reversed"] = d.groupby("gap_type")[f"cls_{horizon}"].apply(lambda s: float((s == PathClass.REVERSED_NEXT_DAY.value).mean()))
    return g.reset_index()


def intraday_table(ep: pd.DataFrame, lens: str = "rng") -> pd.DataFrame:
    """Range expansion / contraction the next day, by episode type: the 'volatile then got way more volatile' question in one table."""
    if len(ep) == 0 or "intraday_path" not in ep:
        return pd.DataFrame()
    band, side = lens_band_side(ep, lens)
    d = pd.DataFrame({"type": [type_name(b, s, lens) for b, s in zip(band, side)], "path": ep["intraday_path"].to_numpy()})[band > 0]
    if len(d) == 0:
        return pd.DataFrame()
    t = pd.crosstab(d["type"], d["path"])
    return t.div(t.sum(axis=1), axis=0)


def class_rates_by_era(ep: pd.DataFrame, horizon: int, era: np.ndarray) -> pd.DataFrame:
    """Class shares per era: a path taxonomy whose base rates swing across eras cannot be read as one stable world."""
    ok = ep[f"ok_{horizon}"].to_numpy() if len(ep) else np.zeros(0, bool)
    if not ok.any():
        return pd.DataFrame()
    d = pd.DataFrame({"era": np.asarray(era)[ok], "cls": ep.loc[ok, f"cls_{horizon}"].to_numpy()})
    t = pd.crosstab(d["era"], d["cls"])
    return t.div(t.sum(axis=1), axis=0)


def paths_report(ep: pd.DataFrame, lens: str = "c2c", horizons: Sequence[int] = (1, 3, 5)) -> dict[str, Any]:
    """Compact, identity-free summary a research page can print."""
    rep: dict[str, Any] = {"episodes": int(len(ep)), "lens": lens, "base_rates": {}, "tables": {}}
    for h in horizons:
        if f"cls_{h}" not in ep:
            continue
        rep["base_rates"][h] = base_rates(ep, h)
        rep["tables"][h] = class_table(ep, lens, h).to_dict("records")
    gt = gap_table(ep)
    rep["gap"] = gt.to_dict("records") if len(gt) else []
    return rep


# ------------------------------------------------------------------------------------------------------------------ profiles and sensitivity
def daily_profile(grid: Grid, eps: pd.DataFrame, days: int | None = None) -> np.ndarray:
    """(episodes x days) signed daily close-to-close returns after the move day, in units of the name's own sigma (positive = the move's
    direction). NaN where a session or bar is missing. The raw material of 'what does an average 7% mover do over the next week'."""
    n = len(eps)
    days = int(days if days is not None else grid.cfg.max_horizon)
    if n == 0:
        return np.zeros((0, days))
    fw = gather_forward(grid, eps["ti"].to_numpy(), eps["nj"].to_numpy(), days)
    with np.errstate(invalid="ignore", divide="ignore"):
        dret = fw.C[:, 1:] / fw.C[:, :-1] - 1.0
        return eps["side"].to_numpy(float)[:, None] * dret / eps["sigma"].to_numpy()[:, None]


def mean_profile(profile: np.ndarray, mask: np.ndarray | None = None) -> pd.DataFrame:
    """Mean and standard error of the cumulative signed path by day (in sigma), over the episodes in `mask` that have the full window."""
    if profile.shape[0] == 0:
        return pd.DataFrame(columns=["day", "n", "mean_cum", "se_cum"])
    p = profile if mask is None else profile[mask]
    full = np.isfinite(p).all(axis=1)
    p = p[full]
    if len(p) == 0:
        return pd.DataFrame(columns=["day", "n", "mean_cum", "se_cum"])
    cum = np.cumsum(p, axis=1)
    return pd.DataFrame({"day": np.arange(1, cum.shape[1] + 1), "n": len(p), "mean_cum": cum.mean(0),
                         "se_cum": cum.std(0, ddof=1) / math.sqrt(len(p)) if len(p) > 1 else np.nan})


def label_sensitivity(grid: Grid, eps: pd.DataFrame, pc: PathConfig = PathConfig(), factor: float = 0.2, horizon: int = 1) -> dict[str, Any]:
    """How much do labels move when every threshold is nudged by +-`factor`? Rebuilds the labels with all sigma multiples and ratios scaled up
    and down and reports the share of episodes whose class changes. A taxonomy whose classes flip under a 20% nudge is measuring the
    threshold, not the market, and any precursor found against it inherits that fragility."""
    base = label_paths(grid, eps, pc, (horizon,))[f"cls_{horizon}"].to_numpy()
    out: dict[str, Any] = {"n": int(len(base)), "horizon": horizon, "changed": {}}
    for tag, m in (("tighter", 1.0 + factor), ("looser", 1.0 - factor)):
        q = dataclasses.replace(pc, spike_k=pc.spike_k * m, reverse_k=pc.reverse_k * m, cont_k=pc.cont_k * m, stop_k=min(pc.stop_k * m, pc.spike_k * m * 0.99),
                                expand_ratio=pc.expand_ratio * m, contract_ratio=min(pc.contract_ratio * (2.0 - m), pc.expand_ratio * m * 0.99))
        alt = label_paths(grid, eps, q.require_valid(), (horizon,))[f"cls_{horizon}"].to_numpy()
        diff = alt != base
        out["changed"][tag] = float(diff.mean()) if len(base) else float("nan")
        out.setdefault("by_class", {})[tag] = {c: float(diff[base == c].mean()) for c in CLASS_ORDER if (base == c).any()}
    return out


def binomial_excess(k: int, n: int, base_rate: float) -> dict[str, float]:
    """Exact two-sided binomial test of a subgroup's class share against the base rate. Descriptive only: episodes on the same day are not
    independent, so this p-value is a screen, never evidence (evidence is the clustered contrast in precursors.py)."""
    from scipy import stats as sps
    if n <= 0 or not (0.0 < base_rate < 1.0):
        return {"share": float("nan"), "p": float("nan"), "lift": float("nan")}
    return {"share": k / n, "p": float(sps.binomtest(int(k), int(n), float(base_rate)).pvalue), "lift": (k / n) / base_rate}


def class_significance(ep: pd.DataFrame, lens: str, horizon: int, min_n: int = 30) -> pd.DataFrame:
    """For each episode type and class: its share, the pooled base share, lift and the exact binomial screen. Types with fewer than `min_n`
    observable episodes are skipped."""
    col, ok = f"cls_{horizon}", f"ok_{horizon}"
    if len(ep) == 0 or col not in ep:
        return pd.DataFrame(columns=["type", "cls", "n", "k", "share", "base", "lift", "p_screen"])
    band, side = lens_band_side(ep, lens)
    d = pd.DataFrame({"type": [type_name(b, s, lens) for b, s in zip(band, side)], "cls": ep[col].to_numpy()})[(band > 0) & ep[ok].to_numpy()]
    if len(d) == 0:
        return pd.DataFrame(columns=["type", "cls", "n", "k", "share", "base", "lift", "p_screen"])
    base = d["cls"].value_counts(normalize=True)
    rows = []
    for typ, g in d.groupby("type"):
        if len(g) < min_n:
            continue
        for cls, k in g["cls"].value_counts().items():
            r = binomial_excess(int(k), len(g), float(base[cls]))
            rows.append({"type": typ, "cls": cls, "n": len(g), "k": int(k), "share": r["share"], "base": float(base[cls]), "lift": r["lift"], "p_screen": r["p"]})
    return pd.DataFrame(rows, columns=["type", "cls", "n", "k", "share", "base", "lift", "p_screen"])


# ------------------------------------------------------------------------------------------------------------------ planted paths
_NOISE = 0.01
# each entry: list of (open, close, high, low) as fractions of the PREVIOUS close, for the sessions after the move day
_PATHS: dict[str, list[tuple[float, float, float, float]]] = {
    "spike":       [(0.005, 0.03, 0.035, -0.002)] + [(0.0, 0.002, 0.006, -0.004)] * 4,
    "reverse":     [(-0.005, -0.03, 0.002, -0.035)] + [(0.0, 0.002, 0.006, -0.004)] * 4,
    "consolidate": [(0.0, 0.002, 0.005, -0.004), (0.0, -0.002, 0.004, -0.005), (0.0, 0.002, 0.005, -0.004), (0.0, 0.0, 0.004, -0.004),
                    (0.0, 0.001, 0.004, -0.004)],
    "expand":      [(0.0, 0.005, 0.06, -0.06), (0.0, -0.004, 0.06, -0.06), (0.0, 0.005, 0.06, -0.06), (0.0, -0.004, 0.06, -0.06),
                    (0.0, 0.004, 0.06, -0.06)],
    "stop":        [(0.0, 0.003, 0.025, -0.025), (0.0, -0.003, 0.025, -0.025), (0.0, 0.003, 0.025, -0.025), (0.0, -0.003, 0.025, -0.025),
                    (0.0, 0.002, 0.025, -0.025)],
    "continue3":   [(0.0, 0.008, 0.03, -0.03), (0.0, 0.015, 0.035, -0.02), (0.0, 0.015, 0.035, -0.02), (0.0, 0.0, 0.03, -0.03),
                    (0.0, 0.0, 0.03, -0.03)],
    "retrace3":    [(0.0, -0.008, 0.03, -0.03), (0.0, -0.015, 0.02, -0.035), (0.0, -0.015, 0.02, -0.035), (0.0, 0.0, 0.03, -0.03),
                    (0.0, 0.0, 0.03, -0.03)],
    "unclassified": [(0.0, 0.012, 0.03, -0.03), (0.0, -0.011, 0.03, -0.03), (0.0, 0.012, 0.03, -0.03), (0.0, -0.011, 0.03, -0.03),
                     (0.0, 0.012, 0.03, -0.03)],
}
# expected labels of each planted path at horizons 1, 3, 5
PLANTED_EXPECT: dict[str, dict[int, str]] = {
    "spike": {1: "SPIKED_NEXT_DAY", 3: "SPIKED_NEXT_DAY", 5: "SPIKED_NEXT_DAY"},
    "reverse": {1: "REVERSED_NEXT_DAY", 3: "REVERSED_NEXT_DAY", 5: "REVERSED_NEXT_DAY"},
    "consolidate": {1: "CONSOLIDATED", 3: "CONSOLIDATED", 5: "CONSOLIDATED"},
    "expand": {1: "EXPANDED", 3: "EXPANDED", 5: "EXPANDED"},
    "stop": {1: "STOPPED", 3: "STOPPED", 5: "STOPPED"},
    "continue3": {1: "STOPPED", 3: "CONTINUED"},
    "retrace3": {1: "STOPPED", 3: "RETRACED"},
    "unclassified": {1: "UNCLASSIFIED"},
}
MOVE_DAY = 70


def planted_path_world(kinds: Sequence[str] | None = None, side: int = 1, n_days: int = 90) -> tuple[dict[str, pd.DataFrame], dict[str, dict[int, str]]]:
    """One name per planted path kind: 70 sessions of alternating +-1% noise (a known sigma), then a 7% move on session 70 (up if side>0, else a
    mirrored down move and mirrored path), then the planted follow-through. Returns (bars, {ticker: {horizon: expected class}}). The expectation
    table is the spec the classifier must reproduce EXACTLY - a check that can fail."""
    kinds = list(kinds) if kinds is not None else list(_PATHS)
    dates = pd.bdate_range("2021-03-01", periods=n_days)
    O = np.zeros((n_days, len(kinds)))
    H, L, C = O.copy(), O.copy(), O.copy()
    tick, expect = [], {}
    for j, kind in enumerate(kinds):
        p = 50.0
        for t in range(n_days):
            if t < MOVE_DAY:
                r = _NOISE if t % 2 == 0 else -_NOISE
                o, c, h, l = p * (1 + r * 0.3), p * (1 + r), p * (1 + max(r, 0) + 0.004), p * (1 + min(r, 0) - 0.004)
            elif t == MOVE_DAY:
                o, c, h, l = p * 1.01, p * 1.07, p * 1.075, p * 1.0
                if side < 0:
                    o, c, h, l = p * 0.99, p * 0.93, p * 1.0, p * 0.925
            else:
                k = t - MOVE_DAY - 1
                oo, cc, hh, ll = _PATHS[kind][k] if k < len(_PATHS[kind]) else (0.0, 0.0, 0.004, -0.004)
                if side < 0:
                    oo, cc, hh, ll = -oo, -cc, -ll, -hh
                o, c, h, l = p * (1 + oo), p * (1 + cc), p * (1 + hh), p * (1 + ll)
            O[t, j], H[t, j], L[t, j], C[t, j] = o, h, l, c
            p = c
        tick.append(f"P_{kind}")
        expect[tick[-1]] = PLANTED_EXPECT[kind]
    V = np.full_like(C, 2.0e6)
    bars = {k: pd.DataFrame(v, index=dates, columns=tick) for k, v in (("Open", O), ("High", H), ("Low", L), ("Close", C), ("Volume", V))}
    return bars, expect


def episode_config_for_planted() -> EpisodeConfig:
    """The detector configuration under which the planted worlds behave (a 60-session volatility window fits inside 70 sessions of noise)."""
    return EpisodeConfig(warm=80)
