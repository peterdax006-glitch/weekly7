"""Missed-winner detector and its evaluation (Bible Phase 14, canon C20).

The detector is an online logistic model: every closed week it studies the stocks that gained >= 7% - the ones the
system picked AND the hundreds it did not - and learns what their cross-sectional ranks looked like beforehand
(evidence ranks, mu_raw, vol20, max20, log_dv, r5 ...). It is trained weekly and is judged OUT OF SAMPLE first:
each week's predictions are scored against the week's outcome before the model is allowed to learn from it. Its
influence on the pick score starts at ZERO, can rise only when that out-of-sample skill is statistically supported,
moves by at most `det_step` per week, and can never exceed the hard cap.

This module owns the model (adaptive.MissedWinnerDetector re-exports it) and the four tests the Bible demands:
  * detector alone          - precision@k and rank IC of its own ranking
  * detector + base         - the blend at the weight the detector had EARNED at decision time
  * shuffled detector       - same model, predictions permuted across names: skill must collapse to ~0
  * future-scrambled        - labels after a cut replaced by noise: decisions before the cut must not change,
                              skill after it must vanish
plus the control the adapter's claim rests on: learn from SHUFFLED winners (labels permuted across names within the
week, features intact) and compare - a detector whose uplift does not beat that control learned nothing real.
Also the descriptive side: why a winner was missed (ineligible / ranked below the cut / picked), what missed winners
look like versus picks, split by type and by era. Deterministic; every draw takes an explicit seed."""
import hashlib
import json

import numpy as np
import pandas as pd

DET_FEATS = ["e_ear", "e_ins_buyers30", "e_ins_officer30", "e_dist_52wh", "e_ind_mom60", "e_frog", "e_mom_12_1",
             "e_r5_nonews", "e_max20", "e_skew60", "mu_raw", "vol20", "max20", "log_dv", "r5"]
DET_HARD_CAP = 0.5                       # no configuration may give the detector more than this share of the score
SUSPICIOUS_IC = 0.4                      # detector-alone rank IC above this is treated as leakage, not skill
WINNER = 0.07                            # a +7% week is a winner (the goal the whole system is aimed at)
DET_DEFAULT = {"det_lr": 0.5, "det_l2": 0.05, "det_min_weeks": 6, "det_max": 0.5, "det_z_min": 1.0, "det_z_span": 4.0,
               "det_step": float("inf"),   # max weight increase per week; inf = ramp is the z-score alone
               "det_steps": 5, "det_min_names": 30, "det_skill_half_life": None, "prior_weeks": 8, "det_winner": WINNER}
# an extra input is approved only if its name carries no forward-looking token (point-in-time rule, Phase 14)
_FORBIDDEN_TOKENS = ("fwd", "future", "next", "label", "target", "outcome", "ret_f", "lead")


def check_feature_pit(name):
    """True if `name` may enter the detector: lagged/ranked inputs only, nothing that names the future."""
    n = str(name).lower()
    return not any(t in n for t in _FORBIDDEN_TOKENS)


def rank_ic(a, b):
    """Spearman correlation of two aligned series; 0.0 when either is constant or there are too few points."""
    a, b = pd.Series(a), pd.Series(b)
    ok = a.notna().values & b.notna().values
    if ok.sum() < 5:
        return 0.0
    ra, rb = a[ok].rank().values, b[ok].rank().values
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


class MissedWinnerDetector:
    """Canon C20: learns, week by week, what the next +7% stocks looked like - including the hundreds it did
    not pick - and earns influence only by predicting LATER weeks correctly (thin data / coincidence guard).
    Online logistic regression on cross-sectional ranks; deterministic."""

    def __init__(self, meta=None, extra_features=()):
        self.meta = {**DET_DEFAULT, **(meta or {})}
        bad = [f for f in extra_features if not check_feature_pit(f)]
        if bad:
            raise ValueError(f"inputs fail the point-in-time rule: {bad}")
        self.feats = list(DET_FEATS) + [f for f in extra_features if f not in DET_FEATS]
        self.coef = np.zeros(len(self.feats))
        self.bias = 0.0
        self.skill = []                      # out-of-sample rank-IC of its predictions, one per closed period
        self.n = 0
        self.history = []                    # one dict per learned week: n names, winners, skill, weight after
        self.missing = {}                    # feature -> weeks in which the column was absent
        self._w = 0.0                        # the weight it has EARNED so far (ramp-limited)

    # ------------------------------------------------------------------ features and prediction
    def _x(self, p):
        cols = []
        for c in self.feats:
            if c in p:
                v = p[c]
            else:
                v = pd.Series(0.0, index=p.index)
                self.missing[c] = self.missing.get(c, 0) + 1
            cols.append(v.rank(pct=True).fillna(0.5).values - 0.5)
        return np.column_stack(cols)

    def predict(self, p):
        z = self._x(p) @ self.coef + self.bias
        return pd.Series(1 / (1 + np.exp(-z)), index=p.index)

    # ------------------------------------------------------------------ learning
    def _labels(self, fwd):
        thr = self.meta["det_winner"]
        y = (fwd >= thr).astype(float)
        if y.sum() == 0:                                   # quiet market: learn from the top 5% instead
            y = (fwd >= fwd.quantile(0.95)).astype(float)
        return y

    def learn(self, p0, fwd):
        """Judge the detector on this closed week BEFORE it learns from it, then take a few gradient steps.
        Returns the week's diagnostics dict, or None when the week had too few names to learn from."""
        ok = fwd.notna()
        if ok.sum() < self.meta["det_min_names"]:
            return None
        p0, f = p0[ok], fwd[ok]
        y = self._labels(f)
        diag = {"n": int(len(y)), "winners": int(y.sum()), "skill": None}
        # 1) judge the detector on this period BEFORE it learns from it (out-of-sample)
        if self.n > 0:
            pr = self.predict(p0)
            sk = float(pr.rank().corr(f.rank()))
            self.skill.append(sk)
            diag["skill"] = sk
        # 2) then learn: a few gradient steps, L2 toward zero, positives up-weighted
        X = self._x(p0)
        w = np.where(y.values > 0, 0.5 / max(y.mean(), 1e-3), 0.5 / max(1 - y.mean(), 1e-3))
        lr, lam = self.meta["det_lr"], self.meta["det_l2"]
        for _ in range(int(self.meta["det_steps"])):
            pr = 1 / (1 + np.exp(-(X @ self.coef + self.bias)))
            g = (pr - y.values) * w
            self.coef -= lr * (X.T @ g / len(g) + lam * self.coef)
            self.bias -= lr * float(g.mean())
        self.n += 1
        self._refresh_weight()
        diag["weight"] = self._w
        self.history.append(diag)
        return diag

    # ------------------------------------------------------------------ earned influence
    def skill_mean(self):
        return float(np.mean(self.skill)) if self.skill else 0.0

    def support(self):
        """The statistical case for trusting the detector: (weeks judged, shrunk mean skill, z of that mean).
        With det_skill_half_life set, recent weeks count more, so a detector whose edge has gone loses its weight
        within a few weeks instead of coasting on old skill (n_eff replaces k in the standard error)."""
        k = len(self.skill)
        if k == 0:
            return {"k": 0, "mean": 0.0, "z": 0.0, "supported": False}
        s = np.asarray(self.skill)
        hl = self.meta.get("det_skill_half_life")
        if hl:
            w = 0.5 ** ((k - 1 - np.arange(k)) / hl)
            m = float((w * s).sum() / (w.sum() + self.meta["prior_weeks"]))
            n_eff = float(w.sum() ** 2 / (w ** 2).sum())
            mu = (w * s).sum() / w.sum()
            sd = float(np.sqrt((w * (s - mu) ** 2).sum() / w.sum() * n_eff / max(n_eff - 1, 1e-9))) if k > 1 else 1.0
        else:
            m = float(s.sum()) / (k + self.meta["prior_weeks"])
            sd = float(np.std(s, ddof=1)) if k > 1 else 1.0
            n_eff = float(k)
        z = m / (sd / np.sqrt(n_eff)) if sd > 0 else 0.0
        return {"k": k, "mean": m, "z": float(z),
                "supported": bool(k >= self.meta["det_min_weeks"] and z > self.meta["det_z_min"])}

    def target_weight(self):
        """The weight its evidence would justify today, ignoring the ramp limit. 0 until enough later weeks were judged."""
        s = self.support()
        if s["k"] < self.meta["det_min_weeks"]:
            return 0.0
        cap = min(self.meta["det_max"], DET_HARD_CAP)
        return float(np.clip((s["z"] - self.meta["det_z_min"]) / self.meta["det_z_span"], 0.0, cap))

    def _refresh_weight(self):
        t = self.target_weight()
        # weight may increase only under statistical support, by at most det_step a week; it may fall at once
        self._w = t if t <= self._w else min(t, self._w + self.meta["det_step"])

    def weight(self):
        """Weight earned so far: 0 until it has predicted enough later weeks; grows with its shrunk, significant skill."""
        return float(self._w)

    # ------------------------------------------------------------------ inspection and state
    def coef_table(self):
        return pd.Series(self.coef, index=self.feats, name="coef").sort_values(key=np.abs, ascending=False)

    def state_dict(self):
        return {"feats": self.feats, "coef": [float(c) for c in self.coef], "bias": float(self.bias), "skill": list(self.skill),
                "n": self.n, "w": self._w}

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.state_dict(), sort_keys=True).encode()).hexdigest()


# ==================================================================================================================
# the descriptive side: what did the winners we did not pick look like, and why were they missed
# ==================================================================================================================
def why_missed(p0, fwd, picked, eligible=None, score=None, thr=WINNER, cut_rank=None):
    """One row per winner of the closed week with the reason it was not owned:
    picked | ineligible | below_cut (ranked, but under the top-k line; `rank_pct` says how far under) | unscored.
    eligible: bool Series over p0.index; score: the selection score (higher = better); cut_rank: k."""
    win = fwd[fwd >= thr].dropna()
    rows = []
    picked = set(picked)
    rk = score.rank(ascending=False, method="first") if score is not None else None
    for t, r in win.items():
        if t in picked:
            why = "picked"
        elif eligible is not None and t in eligible.index and not bool(eligible[t]):
            why = "ineligible"
        elif rk is not None and t in rk.index and np.isfinite(rk[t]):
            why = "below_cut"
        else:
            why = "unscored"
        liq = float(p0["log_dv"].rank(pct=True)[t]) if "log_dv" in p0 and t in p0.index else np.nan
        rows.append({"ticker": t, "ret": float(r), "why": why, "rank": float(rk[t]) if rk is not None and t in rk.index else np.nan,
                     "liq_pct": liq, "type": winner_type(p0, t)})
    cols = ["ticker", "ret", "why", "rank", "liq_pct", "type"]
    return pd.DataFrame(rows, columns=cols)


def winner_type(p0, t):
    """A coarse type for a winner, from what was knowable at the decision: a gap-prone microcap, an event stock, a
    momentum leader, a fast mover, or 'other'. Used only to break the missed-winner report down."""
    if t not in p0.index:
        return "other"
    row = p0.loc[t]
    g = lambda c: float(row[c]) if c in row and np.isfinite(row[c]) else np.nan
    if "log_dv" in p0 and p0["log_dv"].notna().any() and g("log_dv") < p0["log_dv"].quantile(0.25):
        return "illiquid"
    if g("e_ear") == g("e_ear") and g("e_ear") > 0.85 or g("ear") == g("ear") and g("ear") > 0:
        return "event"
    if "e_dist_52wh" in p0 and g("e_dist_52wh") > p0["e_dist_52wh"].quantile(0.8):
        return "momentum"
    if "vol20" in p0 and g("vol20") > p0["vol20"].quantile(0.8):
        return "fast_mover"
    return "other"


def missed_profile(p0, picked, missed, feats=None):
    """Mean rank-feature of the missed winners minus that of the picks: where the system's blind spot sits."""
    feats = feats or DET_FEATS
    out = {}
    if not len(missed) or not len(picked):
        return out
    for c in feats:
        if c in p0:
            v = p0[c].rank(pct=True)
            out[c] = float(v.reindex(list(missed)).mean() - v.reindex(list(picked)).mean())
    return out


class MissedLedger:
    """Running record of closed weeks: winners, caught, missed, and the missed-minus-picked profile. Adapter appends;
    summary() answers 'is the blind spot systematic?' with a per-feature t-statistic across weeks, by type and by era."""

    def __init__(self):
        self.rows = []

    def add(self, date, winners, missed, profile, detector_skill=0.0, detector_weight=0.0, types=None, era=None):
        self.rows.append({"date": str(date), "winners": int(winners), "missed": int(missed), "caught": int(winners - missed),
                          "missed_minus_picked": dict(profile), "detector_skill": float(detector_skill),
                          "detector_weight": float(detector_weight), "types": dict(types or {}), "era": era})

    def frame(self):
        return pd.DataFrame(self.rows)

    def catch_rate(self):
        w = sum(r["winners"] for r in self.rows)
        return float(sum(r["caught"] for r in self.rows) / w) if w else 0.0

    def summary(self, min_weeks=4):
        """Per feature: mean missed-minus-picked, t across weeks, weeks observed. A feature is 'systematic' at |t| >= 2."""
        acc = {}
        for r in self.rows:
            for c, v in r["missed_minus_picked"].items():
                acc.setdefault(c, []).append(v)
        out = []
        for c, v in acc.items():
            v = np.asarray(v)
            sd = v.std(ddof=1) if len(v) > 1 else np.nan
            t = v.mean() / (sd / np.sqrt(len(v))) if len(v) >= min_weeks and sd and sd > 0 else np.nan
            out.append({"feature": c, "mean_gap": float(v.mean()), "t": float(t) if t == t else np.nan, "weeks": len(v),
                        "systematic": bool(t == t and abs(t) >= 2.0)})
        return pd.DataFrame(out, columns=["feature", "mean_gap", "t", "weeks", "systematic"]).sort_values(
            "t", key=lambda s: s.abs(), ascending=False, na_position="last").reset_index(drop=True)

    def by_type(self):
        """Missed winners per type over all weeks (counts)."""
        tot = {}
        for r in self.rows:
            for k, n in r["types"].items():
                tot[k] = tot.get(k, 0) + n
        return pd.Series(tot, dtype=float).sort_values(ascending=False)

    def by_era(self):
        f = self.frame()
        if f.empty or "era" not in f or f["era"].isna().all():
            return pd.DataFrame(columns=["winners", "missed", "caught", "catch_rate"])
        g = f.groupby("era")[["winners", "missed", "caught"]].sum()
        g["catch_rate"] = g["caught"] / g["winners"].where(g["winners"] > 0)
        return g


# ==================================================================================================================
# evaluation: detector alone / + base / shuffled / future-scrambled / shuffled-winners control
# ==================================================================================================================
def _topk_metrics(score, fwd, k, thr):
    s = score.reindex(fwd.index)
    order = np.argsort(-s.fillna(-np.inf).values, kind="stable")[:k]
    top = fwd.iloc[order]
    return float((top >= thr).mean()), float(top.mean())


def _blend(base, det, w):
    if w <= 0:
        return base.rank(pct=True)
    return (1 - w) * base.rank(pct=True) + w * det.rank(pct=True).reindex(base.index).fillna(0.5)


def _base_score(p0):
    if "mu_raw" in p0:
        return p0["mu_raw"]
    if "evidence" in p0:
        return p0["evidence"]
    return pd.Series(0.0, index=p0.index)


def _scramble_returns(fwd, rng):
    """Replace a week's forward returns with noise of the same scale - the future-scramble control."""
    sd = float(fwd.std()) if len(fwd) > 1 else 0.05
    return pd.Series(rng.normal(0, sd if sd > 0 else 0.05, len(fwd)), index=fwd.index)


def walk_forward(weeks, meta=None, mode="honest", seed=0, k=10, thr=WINNER, cut=None):
    """Run the detector through `weeks` = [(date, p0, fwd), ...] in date order and score every out-of-sample week.

    mode
      honest            - real labels
      label_shuffled    - trains on winners permuted across names (features intact); scored on the TRUE outcome.
                          This is the 'shuffled-winners' control: any uplift it shows is luck.
      pred_shuffled     - the honest model's predictions permuted across names (shuffled detector)
      future_scrambled  - every outcome after index `cut` (default: the middle week) is noise
    Returns a DataFrame, one row per scored week: date, ic_det, ic_base, ic_blend, prec/ret at k for detector alone,
    base alone and blend, weight_used (what the detector had EARNED before this week), base_rate."""
    if mode not in ("honest", "label_shuffled", "pred_shuffled", "future_scrambled"):
        raise ValueError(f"unknown mode {mode!r}")
    rng = np.random.default_rng(seed)
    det = MissedWinnerDetector(meta)
    cut = len(weeks) // 2 if cut is None else cut
    rows = []
    for i, (date, p0, fwd) in enumerate(weeks):
        true = fwd
        if mode == "future_scrambled" and i > cut:
            true = _scramble_returns(fwd, rng)
        ok = true.notna()
        if ok.sum() < det.meta["det_min_names"]:
            continue
        p, f = p0[ok], true[ok]
        if det.n > 0:                                      # scored before learning: the model has not seen this week
            pr = det.predict(p)
            if mode == "pred_shuffled":
                pr = pd.Series(rng.permutation(pr.values), index=pr.index)
            base = _base_score(p)
            w = det.weight()
            bl = _blend(base, pr, w)
            row = {"date": date, "weight_used": w, "base_rate": float((f >= thr).mean()),
                   "ic_det": rank_ic(pr, f), "ic_base": rank_ic(base, f), "ic_blend": rank_ic(bl, f)}
            for tag, sc in (("det", pr), ("base", base), ("blend", bl)):
                pk, rk = _topk_metrics(sc, f, k, thr)
                row[f"prec_{tag}"], row[f"ret_{tag}"] = pk, rk
            rows.append(row)
        learn_y = f
        if mode == "label_shuffled":
            learn_y = pd.Series(rng.permutation(f.values), index=f.index)
        det.learn(p, learn_y)
    cols = ["date", "weight_used", "base_rate", "ic_det", "ic_base", "ic_blend", "prec_det", "ret_det", "prec_base",
            "ret_base", "prec_blend", "ret_blend"]
    out = pd.DataFrame(rows, columns=cols)
    out.attrs["fingerprint"] = det.fingerprint()
    out.attrs["final_weight"] = det.weight()
    return out


def _t(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return 0.0
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def sign_flip_p(diff, n=2000, seed=0):
    """One-sided sign-flip permutation p-value that the mean of `diff` is > 0 (paired weekly differences)."""
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 3:
        return 1.0
    rng = np.random.default_rng(seed)
    obs = d.mean()
    flips = rng.choice([-1.0, 1.0], size=(n, len(d)))
    return float((1 + int(((flips * d).mean(axis=1) >= obs - 1e-15).sum())) / (n + 1))


def boot_ci(x, n=2000, seed=0, level=0.95):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(m, a)), float(np.quantile(m, 1 - a))


def evaluate(weeks, meta=None, seed=0, n_controls=20, k=10, thr=WINNER):
    """The full Phase-14 evaluation. Returns a dict:
      alone / with_base      - mean rank IC and precision@k, detector alone and blended, vs base
      shuffled_detector      - the same metrics with predictions permuted
      future_scrambled       - skill after the cut, and whether decisions BEFORE the cut were identical (no look-ahead)
      control                - distribution of the blend uplift when the detector learns from shuffled winners
      verdict                - True only if the honest uplift beats the control's 95th percentile AND is positive"""
    hon = walk_forward(weeks, meta, "honest", seed, k, thr)
    out = {"weeks_scored": int(len(hon))}
    if hon.empty:
        out.update(verdict=False, reason="no weeks scored")
        return out
    out["alone"] = {"ic": float(hon["ic_det"].mean()), "ic_t": _t(hon["ic_det"]), "prec_at_k": float(hon["prec_det"].mean()),
                    "base_rate": float(hon["base_rate"].mean())}
    out["base"] = {"ic": float(hon["ic_base"].mean()), "prec_at_k": float(hon["prec_base"].mean())}
    dif = hon["ic_blend"] - hon["ic_base"]
    dprec = hon["prec_blend"] - hon["prec_base"]
    out["with_base"] = {"ic": float(hon["ic_blend"].mean()), "prec_at_k": float(hon["prec_blend"].mean()),
                        "ic_uplift": float(dif.mean()), "ic_uplift_ci": boot_ci(dif, seed=seed),
                        "prec_uplift": float(dprec.mean()), "p_uplift": sign_flip_p(dif, seed=seed),
                        "mean_weight": float(hon["weight_used"].mean()), "final_weight": float(hon.attrs["final_weight"])}
    ps = walk_forward(weeks, meta, "pred_shuffled", seed + 1, k, thr)
    out["shuffled_detector"] = {"ic": float(ps["ic_det"].mean()), "prec_at_k": float(ps["prec_det"].mean()),
                                "collapsed": bool(abs(_t(ps["ic_det"])) < 2.5)}
    cut = len(weeks) // 2
    fs = walk_forward(weeks, meta, "future_scrambled", seed + 2, k, thr, cut=cut)
    cut_date = weeks[cut][0]
    pre_h, pre_s = hon[hon["date"] <= cut_date], fs[fs["date"] <= cut_date]
    same = len(pre_h) == len(pre_s) and np.allclose(pre_h[["ic_det", "prec_det", "weight_used"]].values,
                                                    pre_s[["ic_det", "prec_det", "weight_used"]].values, atol=1e-12)
    post = fs[fs["date"] > cut_date]
    out["future_scrambled"] = {"pre_cut_identical": bool(same), "post_cut_ic": float(post["ic_det"].mean()) if len(post) else 0.0,
                               "post_cut_t": _t(post["ic_det"]) if len(post) else 0.0}
    ups = []
    for j in range(n_controls):
        c = walk_forward(weeks, meta, "label_shuffled", seed + 100 + j, k, thr)
        ups.append(float((c["ic_blend"] - c["ic_base"]).mean()) if len(c) else 0.0)
    ups = np.array(ups)
    q95 = float(np.quantile(ups, 0.95)) if len(ups) else 0.0
    out["control"] = {"uplifts": ups.tolist(), "mean": float(ups.mean()) if len(ups) else 0.0, "q95": q95,
                      "p_vs_control": float((1 + (ups >= out["with_base"]["ic_uplift"]).sum()) / (1 + len(ups)))}
    # a weekly cross-sectional rank IC this large is not something a stock model earns: look for a leaking input
    out["suspicious"] = bool(out["alone"]["ic"] > SUSPICIOUS_IC)
    out["verdict"] = bool(out["with_base"]["ic_uplift"] > q95 and out["with_base"]["ic_uplift"] > 0 and not out["suspicious"])
    return out


def summary_text(res):
    """Human-readable verdict for the run report."""
    if not res.get("weeks_scored"):
        return "missed-winner evaluation: no weeks scored"
    a, b, w, c = res["alone"], res["base"], res["with_base"], res["control"]
    lines = [f"weeks scored {res['weeks_scored']}",
             f"detector alone   IC {a['ic']:+.3f} (t {a['ic_t']:+.1f})  precision@k {a['prec_at_k']:.3f} vs base rate {a['base_rate']:.3f}",
             f"base             IC {b['ic']:+.3f}  precision@k {b['prec_at_k']:.3f}",
             f"detector + base  IC {w['ic']:+.3f}  uplift {w['ic_uplift']:+.4f} CI [{w['ic_uplift_ci'][0]:+.4f}, {w['ic_uplift_ci'][1]:+.4f}]"
             f"  p {w['p_uplift']:.3f}  mean weight {w['mean_weight']:.3f}",
             f"shuffled detector IC {res['shuffled_detector']['ic']:+.3f}  collapsed {res['shuffled_detector']['collapsed']}",
             f"future scrambled  pre-cut identical {res['future_scrambled']['pre_cut_identical']}  post-cut t {res['future_scrambled']['post_cut_t']:+.1f}",
             f"shuffled-winners control uplift mean {c['mean']:+.4f} q95 {c['q95']:+.4f}  p vs control {c['p_vs_control']:.3f}",
             f"suspiciously good (possible leak): {res['suspicious']}",
             f"VERDICT learning from missed winners helped: {res['verdict']}"]
    return "\n".join(lines)
