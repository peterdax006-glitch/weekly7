"""Competing volatility hypotheses H1-H9 (+ discovered H10+) for the volatility laboratory (RESEARCH_BRAIN_CONTRACT C66 section 9,
Objective 1; canon C66 + C63). IMPLEMENTED - NOT VALIDATED.

A hypothesis here is a MECHANISM CLAIM about why a stock makes a large move next week, expressed as (a) the point-in-time derived
features that mechanism says matter, (b) the sign it expects each to have, (c) an estimator, and (d) what would falsify it. The
lab (engine.research.volatility_lab) runs all of them side by side over the same walk-forward folds and never crowns one from a
single score: a hypothesis whose fitted signs contradict its own mechanism is flagged even when it predicts well.

Every derived feature is a function of the SAME row (or of the same-date cross-section, or of the past of the date-level market
series through an expanding rank) - nothing reads a later date. Fitting refuses rows whose outcome had not matured before `now`."""
from __future__ import annotations

import dataclasses as dc
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from engine.learning.core import Epistemic, FirewallBreach, _StrEnum, as_date, stable_hash

EPS = 1e-9
BASE_COLUMNS = ("r1", "r5", "r20", "r60", "vol20", "atr", "gap", "range20", "dist_hi", "dist_lo", "absr1", "absr5", "max5",
                "logp", "vol_surge", "log_dv", "m_r5", "m_vol", "m_r20", "m_breadth")
EVENT_COLUMNS = ("days_to_event", "insider_n30", "filing_n5", "analog_p", "sector")
OUTCOME_COLUMNS = ("touch", "absmove", "tday", "up", "close", "end")


class HypKind(_StrEnum):
    LOGIT = "LOGIT"                 # additive, sign-interpretable
    GBM = "GBM"                     # interactions allowed
    RULE = "RULE"                   # a discovered conjunction of thresholds
    RESIDUAL = "RESIDUAL"           # what the others miss (H9: the honest 'unknown mechanism' bucket)


class HypState(_StrEnum):
    ACTIVE = "ACTIVE"
    PROBATION = "PROBATION"         # discovered, not yet replicated on later data
    RETIRED = "RETIRED"             # never deleted: a retired mechanism can be revived by new evidence


# ---------------------------------------------------------------------------------------------------------------
# derived features: name -> (base columns required, function of the frame)
# ---------------------------------------------------------------------------------------------------------------
def _xs_rank(s: pd.Series) -> pd.Series:
    """Percentile rank within the decision date (same-date cross-section only)."""
    return s.groupby(level=0).rank(pct=True)


def _xs_center(s: pd.Series) -> pd.Series:
    return s - s.groupby(level=0).transform("median")


def expanding_pct(series: pd.Series, min_periods: int = 8) -> pd.Series:
    """Percentile of each value among the values seen up to and including it (point-in-time regime level; NaN until enough history)."""
    v = series.to_numpy(float)
    out = np.full(len(v), np.nan)
    hist: list[float] = []
    for i, x in enumerate(v):
        if np.isfinite(x):
            hist.append(x)
            if len(hist) >= min_periods:
                out[i] = float(np.mean(np.asarray(hist) <= x))
    return pd.Series(out, index=series.index)


def _date_level(F: pd.DataFrame, col: str, fn: Callable[[pd.Series], pd.Series]) -> pd.Series:
    """Apply `fn` to the date-level series of a market column, then broadcast back onto (date, ticker) rows."""
    d = F[col].groupby(level=0).first().sort_index()
    return fn(d).reindex(F.index.get_level_values(0)).set_axis(F.index)


def _sector_median(F: pd.DataFrame, col: str) -> pd.Series:
    return F.groupby([F.index.get_level_values(0), F["sector"].to_numpy()])[col].transform("median")


def _ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / (b.abs() + EPS)


DERIVED: dict[str, tuple[tuple[str, ...], Callable[[pd.DataFrame], pd.Series]]] = {
    "lv20": (("vol20",), lambda F: np.log(F["vol20"] + EPS)),
    "latr": (("atr",), lambda F: np.log(F["atr"] + EPS)),
    "shock1": (("absr1", "vol20"), lambda F: _ratio(F["absr1"], F["vol20"])),
    "vol_ratio_short": (("absr5", "vol20"), lambda F: _ratio(F["absr5"], F["vol20"] * np.sqrt(5))),
    "max5_ratio": (("max5", "vol20"), lambda F: _ratio(F["max5"], F["vol20"])),
    "atr_ratio": (("atr", "vol20"), lambda F: _ratio(F["atr"], F["vol20"])),
    "xs_vol_rank": (("vol20",), lambda F: _xs_rank(F["vol20"])),
    "xs_atr_rank": (("atr",), lambda F: _xs_rank(F["atr"])),
    "xs_range_rank": (("range20",), lambda F: _xs_rank(F["range20"])),
    "rel_r20": (("r20", "m_r20"), lambda F: F["r20"] - F["m_r20"]),
    "rel_r5": (("r5", "m_r5"), lambda F: F["r5"] - F["m_r5"]),
    "abs_rel_r20": (("r20", "m_r20"), lambda F: (F["r20"] - F["m_r20"]).abs()),
    "xs_disp": (("r5",), lambda F: F["r5"].groupby(level=0).transform("std")),
    "mkt_vol": (("m_vol",), lambda F: np.log(F["m_vol"] + EPS)),
    "mkt_stress": (("m_vol",), lambda F: _date_level(F, "m_vol", expanding_pct)),
    "vol_over_mkt": (("vol20", "m_vol"), lambda F: np.log((F["vol20"] + EPS) / (F["m_vol"] + EPS))),
    "lv20_x_stress": (("vol20", "m_vol"), lambda F: np.log(F["vol20"] + EPS) * _date_level(F, "m_vol", expanding_pct).fillna(0.5)),
    "breadth_dev": (("m_breadth",), lambda F: (F["m_breadth"] - 0.5).abs()),
    "compress": (("range20", "vol20"), lambda F: np.log(_ratio(F["range20"], F["vol20"] * np.sqrt(20)))),
    "squeeze_atr": (("atr", "range20"), lambda F: _ratio(F["atr"], F["range20"])),
    "breakout_prox": (("dist_hi", "dist_lo"), lambda F: np.minimum(F["dist_hi"].abs(), F["dist_lo"].abs())),
    "near_hi": (("dist_hi",), lambda F: F["dist_hi"]),
    "near_lo": (("dist_lo",), lambda F: F["dist_lo"]),
    "lvol_surge": (("vol_surge",), lambda F: np.log(F["vol_surge"].clip(lower=0.05) + EPS)),
    "surge_x_move": (("vol_surge", "absr1", "vol20"), lambda F: np.log(F["vol_surge"].clip(lower=0.05) + EPS) * _ratio(F["absr1"], F["vol20"])),
    "dv_level": (("log_dv",), lambda F: F["log_dv"]),
    "dv_rel": (("log_dv",), lambda F: _xs_center(F["log_dv"])),
    "price_low": (("logp",), lambda F: -F["logp"]),
    "gap_abs": (("gap",), lambda F: F["gap"].abs()),
    "gap_ratio": (("gap", "vol20"), lambda F: _ratio(F["gap"].abs(), F["vol20"])),
    "ev_soon": (("days_to_event",), lambda F: ((F["days_to_event"] >= 0) & (F["days_to_event"] <= 5)).astype(float)),
    "ev_days": (("days_to_event",), lambda F: F["days_to_event"].clip(0, 30).fillna(30.0) / 30.0),
    "insider_recent": (("insider_n30",), lambda F: np.log1p(F["insider_n30"].clip(lower=0))),
    "filing_recent": (("filing_n5",), lambda F: np.log1p(F["filing_n5"].clip(lower=0))),
    "analog_vol": (("analog_p",), lambda F: F["analog_p"]),
    "sector_rel_vol": (("vol20", "sector"), lambda F: np.log((F["vol20"] + EPS) / (_sector_median(F, "vol20") + EPS))),
    "sector_vol": (("vol20", "sector", "m_vol"), lambda F: np.log((_sector_median(F, "vol20") + EPS) / (F["m_vol"] + EPS))),
}


def required_columns(features: Sequence[str]) -> tuple[str, ...]:
    """Base frame columns needed to derive `features`; unknown derived-feature names are a programming error, not a missing column."""
    bad = [f for f in features if f not in DERIVED]
    if bad:
        raise KeyError(f"unknown derived features {bad}")
    return tuple(dict.fromkeys(c for f in features for c in DERIVED[f][0]))


def missing_columns(features: Sequence[str], columns) -> tuple[str, ...]:
    have = set(columns)
    return tuple(c for c in required_columns(features) if c not in have)


def derive(F: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """Derived feature matrix (float32) on F's index. Raises KeyError naming missing base columns: an unavailable input is
    reported as unavailable, never silently filled with zeros."""
    miss = missing_columns(features, F.columns)
    if miss:
        raise KeyError(f"cannot derive {list(features)}: base columns missing {list(miss)}")
    out = {}
    for f in features:
        v = DERIVED[f][1](F)
        out[f] = pd.Series(np.asarray(v, float), index=F.index).replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame(out, index=F.index).astype("float32")


# ---------------------------------------------------------------------------------------------------------------
# the hypothesis specification
# ---------------------------------------------------------------------------------------------------------------
@dc.dataclass(frozen=True)
class Hypothesis:
    hid: str
    name: str
    mechanism: str                          # the causal story in one sentence
    features: tuple[str, ...]
    kind: HypKind = HypKind.LOGIT
    priors: tuple[tuple[str, int], ...] = ()   # (feature, expected sign +1/-1): what the mechanism says
    falsifier: str = ""                     # an observation that would count against the mechanism
    origin: str = "seeded"                  # seeded | discovered
    parent: str = ""                        # hid or evidence id this one came from
    rule: tuple[tuple[str, str, float], ...] = ()   # RULE kind: ((feature, '>' | '<=', threshold), ...)
    state: HypState = HypState.ACTIVE
    weight_prior: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "kind", HypKind.parse(self.kind))
        object.__setattr__(self, "state", HypState.parse(self.state))

    def validate(self) -> list[str]:
        errs = []
        if not self.hid:
            errs.append("empty hid")
        if not self.mechanism:
            errs.append(f"{self.hid}: a hypothesis without a mechanism claim cannot be falsified")
        if self.kind != HypKind.RESIDUAL and not self.features:
            errs.append(f"{self.hid}: no features")
        bad = [f for f in self.features if f not in DERIVED]
        if bad:
            errs.append(f"{self.hid}: unknown features {bad}")
        for f, s in self.priors:
            if f not in self.features or s not in (-1, 1):
                errs.append(f"{self.hid}: bad prior ({f}, {s})")
        if self.kind == HypKind.RULE:
            if not self.rule:
                errs.append(f"{self.hid}: RULE hypothesis without clauses")
            for c in self.rule:
                if len(c) != 3 or c[1] not in (">", "<=") or c[0] not in DERIVED:
                    errs.append(f"{self.hid}: bad clause {c!r}")
        if self.origin == "discovered" and self.kind not in (HypKind.RULE, HypKind.RESIDUAL):
            errs.append(f"{self.hid}: discovered hypotheses are rules")
        return errs

    def needs(self) -> tuple[str, ...]:
        return required_columns(self.features) if self.features else ()

    def missing(self, columns) -> tuple[str, ...]:
        return missing_columns(self.features, columns) if self.features else ()

    def available(self, columns) -> bool:
        return not self.missing(columns)

    def fingerprint(self) -> str:
        return stable_hash([self.hid, self.features, str(self.kind), self.rule, self.priors])


def seeded_hypotheses() -> list[Hypothesis]:
    """H1-H9 as named by the owner (section 9). H9 is not a mechanism: it is the residual bucket that keeps 'unknown' visible."""
    return [
        Hypothesis("H1", "volatility clustering", "big moves and wide ranges recently make big moves next week likely (variance is persistent)",
                   ("lv20", "latr", "shock1", "vol_ratio_short", "max5_ratio"),
                   priors=(("lv20", 1), ("latr", 1), ("vol_ratio_short", 1), ("max5_ratio", 1)),
                   falsifier="once own-history volatility is controlled, recent shocks add nothing"),
        Hypothesis("H2", "event anticipation", "a scheduled catalyst or a burst of insider/filing activity is coming inside the holding window",
                   ("ev_soon", "ev_days", "insider_recent", "filing_recent", "analog_vol"),
                   priors=(("ev_soon", 1), ("ev_days", -1), ("insider_recent", 1), ("filing_recent", 1)),
                   falsifier="names with an event inside five days move no more than the same names without one"),
        Hypothesis("H3", "liquidity shock", "thin, low-priced stocks with wide ranges gap violently when order flow arrives",
                   ("dv_level", "dv_rel", "price_low", "atr_ratio", "gap_ratio"),
                   priors=(("dv_level", -1), ("dv_rel", -1), ("price_low", 1), ("gap_ratio", 1)),
                   falsifier="illiquidity carries no information once volatility level is known"),
        Hypothesis("H4", "cross-sectional relative movement", "stocks far from their peers (in rank, relative return, or sector) are the ones that move",
                   ("xs_vol_rank", "xs_range_rank", "rel_r20", "abs_rel_r20", "rel_r5", "sector_rel_vol", "xs_disp"),
                   priors=(("xs_vol_rank", 1), ("abs_rel_r20", 1), ("sector_rel_vol", 1)),
                   falsifier="cross-sectional distance from peers has no incremental value over own volatility"),
        Hypothesis("H5", "regime-dependent volatility", "the market regime scales every stock's chance of moving, and scales high-vol names most",
                   ("mkt_vol", "mkt_stress", "vol_over_mkt", "lv20_x_stress", "breadth_dev", "sector_vol"),
                   priors=(("mkt_vol", 1), ("mkt_stress", 1), ("lv20_x_stress", 1)),
                   falsifier="the mover rate is the same across market-volatility terciles after controlling for own volatility"),
        Hypothesis("H6", "technical compression to expansion", "a quiet, narrow range near a boundary precedes a breakout",
                   ("compress", "squeeze_atr", "breakout_prox", "near_hi", "near_lo", "vol_ratio_short"),
                   priors=(("compress", -1), ("breakout_prox", -1), ("squeeze_atr", -1)),
                   falsifier="compressed names move no more than uncompressed names of the same volatility level"),
        Hypothesis("H7", "unusual volume to movement", "abnormal volume, especially with a large price move, marks information arriving",
                   ("lvol_surge", "surge_x_move", "dv_rel", "shock1"),
                   priors=(("lvol_surge", 1), ("surge_x_move", 1)),
                   falsifier="volume surges do not precede larger next-week moves once the surge day's return is known"),
        Hypothesis("H8", "multi-factor interaction", "no single mechanism; combinations of the above (volume x compression, event x regime) carry the signal",
                   tuple(dict.fromkeys(("lv20", "latr", "shock1", "vol_ratio_short", "compress", "breakout_prox", "lvol_surge", "surge_x_move",
                                                    "dv_level", "dv_rel", "xs_vol_rank", "rel_r20", "mkt_vol", "mkt_stress", "gap_ratio", "atr_ratio",
                                                    "price_low", "near_hi", "near_lo"))),
                   kind=HypKind.GBM, falsifier="a gradient-boosted model over all features does not beat the additive H1-H7 models out of sample"),
        Hypothesis("H9", "unknown mechanism", "movers that none of H1-H8 anticipates share some structure we have not named yet",
                   (), kind=HypKind.RESIDUAL, falsifier="the residual is unpredictable from any feature (then 'unknown' stays unknown)"),
    ]


# ---------------------------------------------------------------------------------------------------------------
# fitting and prediction
# ---------------------------------------------------------------------------------------------------------------
@dc.dataclass(frozen=True)
class FitConfig:
    C: float = 0.5                       # logistic regularisation
    winsor: float = 5.0                  # standardised features clipped to +-winsor
    gbm_trees: int = 80
    gbm_leaves: int = 7
    gbm_min_child: int = 60
    min_rows: int = 400
    min_events: int = 30
    max_train_rows: int = 120_000
    rule_prior_n: float = 20.0           # beta-smoothing pseudo-rows for a rule's two-level estimate
    inner_blocks: int = 4                # blocked, forward-only cross-fitting for the residual hypothesis
    resid_trees: int = 60
    seed: int = 7

    def validate(self) -> list[str]:
        errs = []
        if self.C <= 0 or self.winsor <= 0:
            errs.append("C and winsor must be positive")
        if self.min_rows < 50 or self.min_events < 5:
            errs.append("min_rows/min_events too small to fit anything meaningful")
        if self.inner_blocks < 3:
            errs.append("inner_blocks must be >= 3")
        return errs


@dc.dataclass
class Standardiser:
    mean: np.ndarray
    std: np.ndarray
    fill: np.ndarray
    winsor: float

    @classmethod
    def fit(cls, X: np.ndarray, winsor: float) -> "Standardiser":
        fill = np.nanmedian(X, axis=0)
        fill = np.where(np.isfinite(fill), fill, 0.0)
        Xf = np.where(np.isfinite(X), X, fill)
        std = Xf.std(axis=0)
        return cls(Xf.mean(axis=0), np.where(std > 1e-12, std, 1.0), fill, winsor)

    def apply(self, X: np.ndarray) -> np.ndarray:
        Xf = np.where(np.isfinite(X), X, self.fill)
        return np.clip((Xf - self.mean) / self.std, -self.winsor, self.winsor)


@dc.dataclass(frozen=True)
class SignReport:
    """Did the fitted directions agree with what the mechanism claimed? A hypothesis that predicts well for the wrong reason is
    a coincidence candidate, not support for its mechanism."""
    checked: int
    agree: int
    contradicted: tuple[str, ...]
    agree_share: float

    @property
    def mechanism_consistent(self) -> bool:
        return self.checked == 0 or (self.agree_share >= 0.6 and len(self.contradicted) <= max(1, self.checked // 3))


def sign_report(h: Hypothesis, coef: Mapping[str, float], tol: float = 0.02) -> SignReport:
    checked, agree, bad = 0, 0, []
    for f, s in h.priors:
        c = coef.get(f)
        if c is None or abs(c) < tol:
            continue
        checked += 1
        if np.sign(c) == s:
            agree += 1
        else:
            bad.append(f)
    return SignReport(checked, agree, tuple(bad), agree / checked if checked else float("nan"))


def check_mature(F: pd.DataFrame, now, what: str = "training frame") -> None:
    """Fail closed: fitting on an outcome that had not finished before `now` is a look-ahead."""
    if len(F) == 0:
        return
    if "end" not in F:
        raise FirewallBreach(f"{what}: no 'end' (outcome maturity) column, so maturity before now={as_date(now)} cannot be verified")
    late = pd.to_datetime(F["end"]).dt.normalize() >= pd.Timestamp(as_date(now))
    if late.any():
        raise FirewallBreach(f"{what}: {int(late.sum())} rows have outcomes maturing at/after now={as_date(now)}")


def _subsample(F: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    if len(F) <= cap:
        return F
    return F.iloc[np.sort(np.random.default_rng(seed).choice(len(F), cap, replace=False))]


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _expit(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


@dc.dataclass
class FittedHypothesis:
    hyp: Hypothesis
    ok: bool
    reason: str
    n_train: int = 0
    n_events: int = 0
    base_rate: float = float("nan")
    trained_through: str = ""
    std: Standardiser | None = None
    clf: object = None
    mag: object = None
    coef: dict = dc.field(default_factory=dict)
    signs: SignReport | None = None
    rule_stats: dict = dc.field(default_factory=dict)
    inner: list = dc.field(default_factory=list)          # RESIDUAL: the fitted hypotheses it is the residual of
    resid_model: object = None
    resid_features: tuple[str, ...] = ()
    resid_std: Standardiser | None = None

    def predict(self, F: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(P(extreme move), expected log absolute move) for each row of F. Rows whose inputs are missing get the base rate,
        never a fabricated confident value. Uses only F's own columns."""
        n = len(F)
        if not self.ok:
            return np.full(n, self.base_rate if np.isfinite(self.base_rate) else 0.0), np.full(n, np.nan)
        h = self.hyp
        if h.kind == HypKind.RULE:
            m = rule_mask(F, h.rule)
            p_in, p_out = self.rule_stats["p_in"], self.rule_stats["p_out"]
            mag = np.where(m, self.rule_stats["mag_in"], self.rule_stats["mag_out"])
            return np.where(m, p_in, p_out), mag
        if h.kind == HypKind.RESIDUAL:
            return self._predict_residual(F)
        X = derive(F, h.features).to_numpy(float)
        Z = self.std.apply(X)
        p = self.clf.predict_proba(Z)[:, 1]
        mag = self.mag.predict(Z) if self.mag is not None else np.full(n, np.nan)
        return p, mag

    def _predict_residual(self, F: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        pf, mf = fuse_inner(self.inner, F)
        Z = self.resid_std.apply(derive(F, self.resid_features).to_numpy(float))
        adj = self.resid_model.predict(Z)
        return np.clip(pf + adj, 1e-4, 1 - 1e-4), mf


def rule_mask(F: pd.DataFrame, rule: Sequence[tuple[str, str, float]]) -> np.ndarray:
    feats = tuple(dict.fromkeys(c[0] for c in rule))
    D = derive(F, feats)
    m = np.ones(len(F), bool)
    for f, op, thr in rule:
        v = D[f].to_numpy(float)
        m &= np.where(np.isfinite(v), (v > thr) if op == ">" else (v <= thr), False)
    return m


def fuse_inner(fitted: Sequence[FittedHypothesis], F: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Average, in logit space, the predictions of the fitted hypotheses that could be fitted."""
    ps, ms = [], []
    for fh in fitted:
        if fh.ok:
            p, m = fh.predict(F)
            ps.append(_logit(p))
            ms.append(m)
    if not ps:
        return np.full(len(F), 0.1), np.full(len(F), np.nan)
    return _expit(np.mean(ps, axis=0)), np.nanmean(np.vstack(ms), axis=0) if any(np.isfinite(m).any() for m in ms) else np.full(len(F), np.nan)


def _outcomes(F: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y = F["touch"].to_numpy(float)
    lm = np.log(F["absmove"].to_numpy(float) + 1e-4)
    return y, lm


def fit_hypothesis(h: Hypothesis, F: pd.DataFrame, now, cfg: FitConfig = FitConfig(), others: Sequence[Hypothesis] = ()) -> FittedHypothesis:
    """Fit one hypothesis on the rows of F (all of whose outcomes matured before `now`). Returns a FittedHypothesis with ok=False and a
    reason instead of raising when the data cannot support a fit (too few rows or events, missing input columns). `others` is
    used only by the RESIDUAL kind."""
    errs = h.validate() + cfg.validate()
    if errs:
        raise ValueError("; ".join(errs))
    check_mature(F, now, f"fit {h.hid}")
    bad_lab = F["touch"].isna() if "touch" in F else None
    if bad_lab is None:
        raise KeyError("training frame lacks the 'touch' outcome")
    F = F[~bad_lab.to_numpy()]
    y_all = F["touch"].to_numpy(float)
    br = float(y_all.mean()) if len(y_all) else float("nan")
    through = str(pd.to_datetime(F["end"]).max().date()) if len(F) and "end" in F else ""
    base = FittedHypothesis(h, False, "", int(len(F)), int(y_all.sum()) if len(F) else 0, br, through)
    if h.kind != HypKind.RESIDUAL and h.missing(F.columns):
        base.reason = f"input columns unavailable: {list(h.missing(F.columns))}"
        return base
    if len(F) < cfg.min_rows or base.n_events < cfg.min_events or base.n_events > len(F) - cfg.min_events:
        base.reason = f"insufficient data: {len(F)} rows, {base.n_events} events"
        return base
    F = _subsample(F, cfg.max_train_rows, cfg.seed)
    y, lm = _outcomes(F)
    if h.kind == HypKind.RULE:
        return _fit_rule(h, F, y, lm, base, cfg)
    if h.kind == HypKind.RESIDUAL:
        return _fit_residual(h, F, now, others, base, cfg)
    X = derive(F, h.features).to_numpy(float)
    std = Standardiser.fit(X, cfg.winsor)
    Z = std.apply(X)
    if h.kind == HypKind.GBM:
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(objective="binary", n_estimators=cfg.gbm_trees, num_leaves=cfg.gbm_leaves, min_child_samples=cfg.gbm_min_child,
                                 learning_rate=0.06, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=cfg.seed, verbose=-1, n_jobs=1)
        clf.fit(Z, y)
        mag = lgb.LGBMRegressor(n_estimators=cfg.gbm_trees, num_leaves=cfg.gbm_leaves, min_child_samples=cfg.gbm_min_child, learning_rate=0.06,
                                random_state=cfg.seed, verbose=-1, n_jobs=1).fit(Z, lm)
        imp = dict(zip(h.features, (clf.booster_.feature_importance("gain") / max(1.0, clf.booster_.feature_importance("gain").sum())).tolist()))
        base.coef = imp
    else:
        from sklearn.linear_model import LogisticRegression, Ridge
        clf = LogisticRegression(C=cfg.C, max_iter=300).fit(Z, y)
        mag = Ridge(alpha=10.0).fit(Z, lm)
        base.coef = dict(zip(h.features, clf.coef_[0].tolist()))
        base.signs = sign_report(h, base.coef)
    base.std, base.clf, base.mag, base.ok, base.reason = std, clf, mag, True, "ok"
    return base


def _fit_rule(h: Hypothesis, F: pd.DataFrame, y, lm, base: FittedHypothesis, cfg: FitConfig) -> FittedHypothesis:
    m = rule_mask(F, h.rule)
    n_in = int(m.sum())
    if n_in < 20 or n_in > len(F) - 20:
        base.reason = f"rule selects {n_in} of {len(F)} rows: not estimable"
        return base
    k = cfg.rule_prior_n
    p_in = (y[m].sum() + k * base.base_rate) / (n_in + k)
    p_out = (y[~m].sum() + k * base.base_rate) / (len(F) - n_in + k)
    base.rule_stats = dict(p_in=float(p_in), p_out=float(p_out), n_in=n_in, mag_in=float(lm[m].mean()), mag_out=float(lm[~m].mean()),
                           lift=float(p_in / max(p_out, 1e-9)))
    base.ok, base.reason = True, "ok"
    return base


def _blocks(dates: np.ndarray, k: int) -> list[np.ndarray]:
    ud = np.unique(dates)
    return [ud[i * len(ud) // k:(i + 1) * len(ud) // k] for i in range(k)]


def _fit_residual(h: Hypothesis, F: pd.DataFrame, now, others: Sequence[Hypothesis], base: FittedHypothesis, cfg: FitConfig) -> FittedHypothesis:
    """H9. Inner expanding-window cross-fitting: block b is predicted by hypotheses trained on blocks < b only, so the residual the
    model learns is out-of-sample and forward-only. The residual model then predicts what the others systematically miss."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    usable = [o for o in others if o.hid != h.hid and o.kind != HypKind.RESIDUAL and o.state != HypState.RETIRED and o.available(F.columns)]
    if len(usable) < 2:
        base.reason = "fewer than two other hypotheses available: no residual is defined"
        return base
    dates = F.index.get_level_values(0).to_numpy()
    blocks = _blocks(dates, cfg.inner_blocks)
    resid_rows, resid_pred = [], []
    inner_cfg = dc.replace(cfg, max_train_rows=min(cfg.max_train_rows, 30_000))
    for b in range(1, len(blocks)):
        tr = F[np.isin(dates, np.concatenate(blocks[:b]))]
        te_mask = np.isin(dates, blocks[b])
        te = F[te_mask]
        last_train = pd.to_datetime(tr["end"]).max()
        te_ok = te[pd.to_datetime(te["end"]) > last_train]                     # purge: test rows start after training outcomes matured
        te_ok = te_ok[pd.to_datetime(te_ok.index.get_level_values(0)) > last_train]
        fitted = [fit_hypothesis(o, tr, last_train + pd.Timedelta(days=1), inner_cfg) for o in usable]
        if not any(f.ok for f in fitted) or len(te_ok) == 0:
            continue
        pf, _ = fuse_inner(fitted, te_ok)
        resid_rows.append(te_ok)
        resid_pred.append(te_ok["touch"].to_numpy(float) - pf)
    if not resid_rows:
        base.reason = "no forward inner blocks could be scored"
        return base
    R = pd.concat(resid_rows)
    r = np.concatenate(resid_pred)
    feats = tuple(dict.fromkeys(f for o in usable for f in o.features))
    feats = tuple(f for f in feats if not missing_columns((f,), R.columns))
    X = derive(R, feats).to_numpy(float)
    std = Standardiser.fit(X, cfg.winsor)
    model = HistGradientBoostingRegressor(max_iter=cfg.resid_trees, max_leaf_nodes=cfg.gbm_leaves, min_samples_leaf=cfg.gbm_min_child,
                                          learning_rate=0.05, l2_regularization=5.0, random_state=cfg.seed).fit(std.apply(X), r)
    inner_final = [fit_hypothesis(o, F, now, inner_cfg) for o in usable]
    if not any(f.ok for f in inner_final):
        base.reason = "no inner hypothesis could be fitted on the full training frame"
        return base
    base.inner, base.resid_model, base.resid_features, base.resid_std = inner_final, model, feats, std
    base.ok, base.reason = True, "ok"
    return base


# ---------------------------------------------------------------------------------------------------------------
# registry: H1-H9 seeded, H10+ discovered
# ---------------------------------------------------------------------------------------------------------------
class HypothesisRegistry:
    """Holds the competing hypotheses. Discovered hypotheses enter on PROBATION and become ACTIVE only after a replication on data
    that was not used to find them; nothing is deleted, only RETIRED (and revivable)."""

    def __init__(self, seeded: Sequence[Hypothesis] | None = None):
        self._h: dict[str, Hypothesis] = {}
        self.log: list[dict] = []
        for h in (seeded if seeded is not None else seeded_hypotheses()):
            self._add(h, "seeded")

    def _add(self, h: Hypothesis, why: str) -> None:
        errs = h.validate()
        if errs:
            raise ValueError("; ".join(errs))
        if h.hid in self._h:
            raise ValueError(f"duplicate hypothesis id {h.hid}")
        self._h[h.hid] = h
        self.log.append({"event": "add", "hid": h.hid, "why": why, "fingerprint": h.fingerprint()})

    def __len__(self) -> int:
        return len(self._h)

    def __contains__(self, hid: str) -> bool:
        return hid in self._h

    def get(self, hid: str) -> Hypothesis:
        return self._h[hid]

    def all(self, states: Sequence[HypState] | None = None) -> list[Hypothesis]:
        want = set(states) if states is not None else {HypState.ACTIVE, HypState.PROBATION}
        return [h for h in self._h.values() if h.state in want]

    def available(self, columns) -> list[Hypothesis]:
        return [h for h in self.all() if h.kind == HypKind.RESIDUAL or h.available(columns)]

    def unavailable(self, columns) -> dict[str, tuple[str, ...]]:
        """Which hypotheses cannot be tested with these columns and what is missing: reported as UNKNOWN by the lab, never as a failure."""
        return {h.hid: h.missing(columns) for h in self.all() if h.kind != HypKind.RESIDUAL and h.missing(columns)}

    def next_id(self) -> str:
        nums = [int(h[1:]) for h in self._h if h[1:].isdigit()]
        return f"H{max(nums + [9]) + 1}"

    def register_discovered(self, cand: Hypothesis, evidence_id: str) -> Hypothesis:
        """Assign the next free H-number to a discovered rule and enter it on PROBATION. Identical rules are refused (a second
        discovery of the same mechanism is a replication, not a new hypothesis)."""
        for h in self._h.values():
            if h.kind == HypKind.RULE and tuple(sorted(h.rule)) == tuple(sorted(cand.rule)):
                raise ValueError(f"rule already registered as {h.hid}")
        h = dc.replace(cand, hid=self.next_id(), origin="discovered", parent=evidence_id, state=HypState.PROBATION)
        self._add(h, f"discovered from {evidence_id}")
        return h

    def promote(self, hid: str, evidence: str) -> None:
        h = self._h[hid]
        if h.state != HypState.PROBATION:
            raise ValueError(f"{hid} is {h.state}, not on probation")
        self._h[hid] = dc.replace(h, state=HypState.ACTIVE)
        self.log.append({"event": "promote", "hid": hid, "why": evidence})

    def retire(self, hid: str, why: str) -> None:
        h = self._h[hid]
        self._h[hid] = dc.replace(h, state=HypState.RETIRED)
        self.log.append({"event": "retire", "hid": hid, "why": why})

    def revive(self, hid: str, why: str) -> None:
        h = self._h[hid]
        if h.state != HypState.RETIRED:
            raise ValueError(f"{hid} is not retired")
        self._h[hid] = dc.replace(h, state=HypState.PROBATION)
        self.log.append({"event": "revive", "hid": hid, "why": why})

    def lineage(self, hid: str) -> list[str]:
        chain, cur = [hid], self._h[hid]
        while cur.parent in self._h:
            chain.append(cur.parent)
            cur = self._h[cur.parent]
        return chain

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([{"hid": h.hid, "name": h.name, "kind": str(h.kind), "state": str(h.state), "origin": h.origin,
                              "n_features": len(h.features), "parent": h.parent} for h in self._h.values()])

    def snapshot(self) -> dict:
        return {"hypotheses": {k: dc.asdict(v) | {"kind": str(v.kind), "state": str(v.state)} for k, v in self._h.items()}, "log": list(self.log)}

    def fingerprint(self) -> str:
        return stable_hash([h.fingerprint() + str(h.state) for h in self._h.values()])


# ---------------------------------------------------------------------------------------------------------------
# discovery of H10+: rules for the movers nobody anticipated, checked on later data before they count
# ---------------------------------------------------------------------------------------------------------------
@dc.dataclass(frozen=True)
class RuleEvidence:
    rule: tuple[tuple[str, str, float], ...]
    n_in: int
    n_events_in: int
    rate_in: float
    rate_out: float
    lift: float
    overlap_with_known: float       # share of the rule's rows already in the top decile of some existing hypothesis
    duplicate_of: str = ""          # existing hid whose selection this mostly restates

    @property
    def rule_id(self) -> str:
        return "R" + stable_hash(sorted(self.rule), 10)


def extract_leaf_rules(tree, feature_names: Sequence[str], min_samples: int) -> list[tuple[tuple[tuple[str, str, float], ...], int, float]]:
    """(clauses, n, positive rate) for every leaf of a fitted sklearn decision tree with at least min_samples rows."""
    t = tree.tree_
    out = []

    def walk(node: int, path: tuple):
        if t.children_left[node] == t.children_right[node]:
            n = int(t.n_node_samples[node])
            val = t.value[node][0]
            if n >= min_samples:
                out.append((path, n, float(val[1] / max(val.sum(), 1e-12))))
            return
        f, thr = feature_names[t.feature[node]], float(t.threshold[node])
        walk(t.children_left[node], path + ((f, "<=", thr),))
        walk(t.children_right[node], path + ((f, ">", thr),))

    walk(0, ())
    return out


def _merge_clauses(path: tuple) -> tuple[tuple[str, str, float], ...]:
    """Collapse repeated splits on one feature into its tightest bound."""
    hi: dict[str, float] = {}
    lo: dict[str, float] = {}
    for f, op, thr in path:
        if op == "<=":
            hi[f] = min(hi.get(f, np.inf), thr)
        else:
            lo[f] = max(lo.get(f, -np.inf), thr)
    return tuple(sorted([(f, "<=", round(v, 5)) for f, v in hi.items()] + [(f, ">", round(v, 5)) for f, v in lo.items()]))


def propose_rules(F: pd.DataFrame, p_known: np.ndarray, now, *, features: Sequence[str] | None = None, max_depth: int = 3, min_leaf: int = 150,
                  min_lift: float = 1.6, known_top_frac: float = 0.10, max_overlap: float = 0.5, seed: int = 0,
                  known_selections: Mapping[str, np.ndarray] | None = None) -> list[tuple[Hypothesis, RuleEvidence]]:
    """Search the rows the existing hypotheses rank in their LOWER 70% for a region with a high mover rate. `p_known` is the fused
    out-of-sample probability of H1-H8 on F's rows (positionally aligned). A candidate is returned only if its lift over the
    outside is large, it is not just the known hypotheses' top decile again, and it is estimated on at least min_leaf rows. It is a
    CANDIDATE on probation: replicate_rule() on later data decides."""
    from sklearn.tree import DecisionTreeClassifier
    check_mature(F, now, "propose_rules")
    if len(p_known) != len(F):
        raise ValueError("p_known must align with F")
    feats = tuple(f for f in (features or DERIVED) if not missing_columns((f,), F.columns))
    if not feats or len(F) < 4 * min_leaf:
        return []
    ok = F["touch"].notna().to_numpy()
    date_rank = pd.Series(p_known, index=F.index).groupby(level=0).rank(pct=True).to_numpy()
    region = ok & (date_rank <= 0.7)
    D = derive(F.iloc[np.flatnonzero(region)], feats)
    y = F["touch"].to_numpy(float)[region]
    if y.sum() < 30 or y.sum() > len(y) - 30:
        return []
    Xs = np.nan_to_num(D.to_numpy(float), nan=0.0)
    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_leaf, class_weight=None, random_state=seed).fit(Xs, y)
    rate_all = float(y.mean())
    sel = dict(known_selections or {})
    out = []
    for path, n, rate in extract_leaf_rules(tree, feats, min_leaf):
        if rate < min_lift * rate_all:
            continue
        rule = _merge_clauses(path)
        m_full = rule_mask(F, rule) & ok
        n_in = int(m_full.sum())
        if n_in < min_leaf:
            continue
        rate_in, rate_out = float(y_full_rate(F, m_full)), float(y_full_rate(F, ~m_full & ok))
        top = pd.Series(p_known, index=F.index).groupby(level=0).rank(pct=True).to_numpy() > 1 - known_top_frac
        overlap = float((top & m_full).sum() / max(n_in, 1))
        dup = ""
        for hid, mask in sel.items():
            j = float((mask & m_full).sum() / max((mask | m_full).sum(), 1))
            if j > 0.6:
                dup = hid
        if overlap > max_overlap or dup or rate_in < min_lift * max(rate_out, 1e-9):
            continue
        ev = RuleEvidence(rule, n_in, int(F["touch"].to_numpy(float)[m_full].sum()), rate_in, rate_out, rate_in / max(rate_out, 1e-9), overlap, dup)
        cand = Hypothesis("X", f"discovered region {ev.rule_id}", "a region of feature space where movers concentrate that H1-H8 do not rank highly",
                          tuple(dict.fromkeys(c[0] for c in rule)), kind=HypKind.RULE, rule=rule, origin="discovered", parent=ev.rule_id,
                          falsifier="the lift disappears on later data", state=HypState.PROBATION)
        out.append((cand, ev))
    out.sort(key=lambda t: -t[1].lift * np.sqrt(t[1].n_in))
    return out


def y_full_rate(F: pd.DataFrame, mask: np.ndarray) -> float:
    y = F["touch"].to_numpy(float)[mask]
    return float(y.mean()) if len(y) else float("nan")


@dc.dataclass(frozen=True)
class Replication:
    rule_id: str
    n_in: int
    rate_in: float
    rate_out: float
    lift: float
    lift_lo: float
    replicated: bool
    reason: str


def replicate_rule(h: Hypothesis, F_later: pd.DataFrame, now, *, min_in: int = 100, min_lift: float = 1.3, n_boot: int = 300, seed: int = 0) -> Replication:
    """A discovered rule must show its lift again on rows dated after the rows used to find it. Bootstrap by date week so one hot
    week cannot replicate a rule. The lower bound of the lift must exceed 1 and the point lift must exceed min_lift."""
    if h.kind != HypKind.RULE:
        raise ValueError("only RULE hypotheses replicate")
    check_mature(F_later, now, "replicate_rule")
    F_later = F_later[F_later["touch"].notna()]
    rid = "R" + stable_hash(sorted(h.rule), 10)
    if h.missing(F_later.columns):
        return Replication(rid, 0, float("nan"), float("nan"), float("nan"), float("nan"), False, f"columns unavailable: {h.missing(F_later.columns)}")
    m = rule_mask(F_later, h.rule)
    if int(m.sum()) < min_in or int((~m).sum()) < min_in:
        return Replication(rid, int(m.sum()), float("nan"), float("nan"), float("nan"), float("nan"), False, "too few rows in or out of the rule")
    y = F_later["touch"].to_numpy(float)
    wk = pd.to_datetime(F_later.index.get_level_values(0)).strftime("%G-%V").to_numpy()
    codes, uniq = pd.factorize(wk)
    g = len(uniq)
    k_in, n_in = np.bincount(codes, weights=y * m, minlength=g), np.bincount(codes, weights=m.astype(float), minlength=g)
    k_out, n_out = np.bincount(codes, weights=y * ~m, minlength=g), np.bincount(codes, weights=(~m).astype(float), minlength=g)
    rin, rout = k_in.sum() / n_in.sum(), k_out.sum() / max(n_out.sum(), 1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, g, size=(n_boot, g))
    lifts = (k_in[idx].sum(1) / np.maximum(n_in[idx].sum(1), 1)) / np.maximum(k_out[idx].sum(1) / np.maximum(n_out[idx].sum(1), 1), 1e-9)
    lo = float(np.quantile(lifts, 0.05))
    lift = float(rin / max(rout, 1e-9))
    ok = lo > 1.0 and lift >= min_lift
    return Replication(rid, int(m.sum()), float(rin), float(rout), lift, lo, bool(ok), "replicated" if ok else "lift not reproduced on later data")


# ---------------------------------------------------------------------------------------------------------------
# interaction features (Q02/Q14): registered on demand so a pair study can fit them like any other derived feature
# ---------------------------------------------------------------------------------------------------------------
def _xsz(s: pd.Series) -> pd.Series:
    g = s.groupby(level=0)
    return (s - g.transform("mean")) / (g.transform("std") + EPS)


def interaction_name(a: str, b: str, null_seed: int | None = None) -> str:
    return f"ix__{a}__{b}" if null_seed is None else f"ixnull{null_seed}__{a}__{b}"


def register_interaction(a: str, b: str, null_seed: int | None = None) -> str:
    """Register (idempotently) the feature xsz(a)*xsz(b), the product of the two features' same-date z-scores. With null_seed the second
    factor is permuted across tickers WITHIN each date (marginals kept, relation destroyed): a control interaction that must show no
    incremental value. Returns the feature name."""
    for f in (a, b):
        if f not in DERIVED:
            raise KeyError(f"unknown derived feature {f}")
    name = interaction_name(a, b, null_seed)
    if name in DERIVED:
        return name

    def fn(F: pd.DataFrame, a=a, b=b, null_seed=null_seed) -> pd.Series:
        za, zb = _xsz(pd.Series(DERIVED[a][1](F), index=F.index)), _xsz(pd.Series(DERIVED[b][1](F), index=F.index))
        if null_seed is not None:
            rng = np.random.default_rng(null_seed)
            vals = zb.to_numpy(float).copy()
            codes = pd.factorize(F.index.get_level_values(0))[0]
            for c in np.unique(codes):
                ix = np.flatnonzero(codes == c)
                vals[ix] = vals[ix][rng.permutation(len(ix))]
            zb = pd.Series(vals, index=F.index)
        return (za * zb).fillna(0.0)

    DERIVED[name] = (tuple(dict.fromkeys(DERIVED[a][0] + DERIVED[b][0])), fn)
    return name


# ---------------------------------------------------------------------------------------------------------------
# does a discovered rule hold in every group, or only on average?
# ---------------------------------------------------------------------------------------------------------------
@dc.dataclass(frozen=True)
class RuleStability:
    rule_id: str
    by: str
    groups: tuple                 # ((label, n_in, lift), ...) for groups with enough rows
    share_above_one: float
    p_sign: float                 # binomial p that lift>1 in at least this many groups if the rule were noise (p=0.5 each)
    min_lift: float
    stable: bool
    reason: str


def rule_stability(h: Hypothesis, F: pd.DataFrame, by: str = "year", min_in: int = 40, min_groups: int = 3) -> RuleStability:
    """Lift of the rule (mover rate inside / outside) per calendar year or per sector. A rule is STABLE when it lifts in at least 75% of
    the groups it can be measured in (binomial p<0.1 against 50/50 luck) and the worst group is not below 0.8. Too few measurable
    groups is reported as not stable with that reason: absence of evidence is never 'stable'."""
    from scipy.stats import binomtest
    rid = "R" + stable_hash(sorted(h.rule), 10)
    if h.kind != HypKind.RULE:
        raise ValueError("only RULE hypotheses have a stability profile")
    nan = float("nan")
    F = F[F["touch"].notna()]
    if by == "year":
        lab = pd.to_datetime(F.index.get_level_values(0)).year.astype(str).to_numpy()
    elif by == "sector":
        if "sector" not in F:
            return RuleStability(rid, by, (), nan, nan, nan, False, "no sector column")
        lab = F["sector"].astype(str).to_numpy()
    else:
        raise ValueError("by must be 'year' or 'sector'")
    if h.missing(F.columns) or len(F) == 0:
        return RuleStability(rid, by, (), nan, nan, nan, False, "rule inputs unavailable or empty frame")
    m = rule_mask(F, h.rule)
    y = F["touch"].to_numpy(float)
    rows = []
    for g in np.unique(lab):
        sel = lab == g
        n_in, n_out = int((m & sel).sum()), int((~m & sel).sum())
        if n_in < min_in or n_out < min_in:
            continue
        rin, rout = y[m & sel].mean(), y[~m & sel].mean()
        rows.append((str(g), n_in, float(rin / max(rout, 1e-9))))
    if len(rows) < min_groups:
        return RuleStability(rid, by, tuple(rows), nan, nan, nan, False, f"only {len(rows)} measurable groups")
    lifts = np.array([r[2] for r in rows])
    k = int((lifts > 1).sum())
    share = k / len(rows)
    p = float(binomtest(k, len(rows), 0.5, alternative="greater").pvalue)
    stable = bool(share >= 0.75 and p < 0.1 and lifts.min() >= 0.8)
    return RuleStability(rid, by, tuple(rows), share, p, float(lifts.min()), stable,
                         "lifts in most groups" if stable else f"lift>1 in {k}/{len(rows)} groups, worst {lifts.min():.2f}")


# ---------------------------------------------------------------------------------------------------------------
# evidence -> epistemic status of a hypothesis (real / conditional / degraded / contradicted / unknown)
# ---------------------------------------------------------------------------------------------------------------
@dc.dataclass(frozen=True)
class HypothesisEvidence:
    hid: str
    increment: float | None = None          # mean per-date AUC gain over the B0 baseline
    increment_lo: float | None = None
    increment_hi: float | None = None
    q: float | None = None
    sign_consistent_share: float | None = None    # share of folds whose fitted signs agreed with the mechanism
    decay: str | None = None                       # STABLE | DECAYING | STOPPED | IMPROVING | UNTESTED
    regime_dependent: bool | None = None
    transfer: tuple[tuple[str, str], ...] = ()     # (axis, StudyVerdict) actually measured for this hypothesis
    direction: str | None = None
    null_control_ok: bool | None = None


@dc.dataclass(frozen=True)
class HypothesisAssessment:
    hid: str
    status: Epistemic
    reasons: tuple[str, ...]
    untested: tuple[str, ...]

    @property
    def usable(self) -> bool:
        return str(self.status) in ("SUPPORTED", "CONDITIONAL")


def assess_hypothesis(ev: HypothesisEvidence, alpha: float = 0.10) -> HypothesisAssessment:
    """Turn the lab's separate measurements into one status, without hiding what was not measured. Order of precedence:
    a failed null control or a significantly negative increment is CONTRADICTED; no increment at all is UNKNOWN; a fitted mechanism whose
    signs disagree in most folds is DEGRADED (it predicts for the wrong reason); STOPPED/DECAYING is DEGRADED; SUPPORTED needs a
    significant increment, consistent signs, no decay and at least one transfer axis measured and not failed; everything else that
    shows a positive increment is CONDITIONAL and says what is untested."""
    why: list[str] = []
    untested = [n for n, v in (("increment", ev.increment), ("sign_consistency", ev.sign_consistent_share),
                               ("decay", ev.decay if ev.decay not in (None, "UNTESTED") else None), ("transfer", ev.transfer or None),
                               ("null_control", ev.null_control_ok)) if v is None]
    if ev.null_control_ok is False:
        return HypothesisAssessment(ev.hid, Epistemic.CONTRADICTED, ("the shuffled-input control gained: the measurement is untrustworthy",), tuple(untested))
    if ev.increment is None or ev.increment_lo is None:
        return HypothesisAssessment(ev.hid, Epistemic.UNKNOWN, ("no incremental measurement over the baseline",), tuple(untested))
    if ev.increment_hi is not None and ev.increment_hi < 0:
        return HypothesisAssessment(ev.hid, Epistemic.CONTRADICTED, (f"significantly worse than own-volatility persistence (hi {ev.increment_hi:+.4f})",), tuple(untested))
    significant = ev.increment_lo > 0 and (ev.q is None or ev.q <= alpha)
    if not significant:
        return HypothesisAssessment(ev.hid, Epistemic.HYPOTHESIS, ("no significant gain over the baseline yet",), tuple(untested))
    if ev.sign_consistent_share is not None and ev.sign_consistent_share < 0.5:
        why.append(f"fitted signs agree with the mechanism in only {ev.sign_consistent_share:.0%} of folds: predicts, but not for the stated reason")
        return HypothesisAssessment(ev.hid, Epistemic.DEGRADED, tuple(why), tuple(untested))
    if ev.decay in ("STOPPED", "DECAYING"):
        return HypothesisAssessment(ev.hid, Epistemic.DEGRADED, (f"signal is {ev.decay}",), tuple(untested))
    failed = [a for a, v in ev.transfer if v in ("NOT_SUPPORTED", "INVALID")]
    passed = [a for a, v in ev.transfer if v == "SUPPORTED"]
    if failed:
        why.append(f"does not transfer across {failed}")
    if ev.regime_dependent:
        why.append("effect depends on the market regime")
    if passed and not failed and not ev.regime_dependent and ev.sign_consistent_share is not None and ev.decay in ("STABLE", "IMPROVING"):
        return HypothesisAssessment(ev.hid, Epistemic.SUPPORTED, ("significant gain, consistent mechanism, stable, transfers on " + ",".join(passed),), tuple(untested))
    if not ev.transfer:
        why.append("transfer not measured for this hypothesis")
    if ev.direction in ("DIRECTION_BLIND", "LOSS_SKEWED"):
        why.append(f"finds volatility but is {ev.direction}")
    return HypothesisAssessment(ev.hid, Epistemic.CONDITIONAL, tuple(why) or ("significant gain; some evidence still missing",), tuple(untested))
