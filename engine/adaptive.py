"""Self-training inside a year (canon C15-C18). One pure, deterministic module used by BOTH the blind live
trader and the re-tester, so every adaptive decision can be replayed and audited.

Pre-season (C17): choose_default() scores candidate settings on warm-up weeks the model had NOT trained on
(a nested split of the warm-up years), then shrinks toward the loop's global prior.
Weekly (C15/C16): Adapter.step() is called at each decision with ONLY
  - today's snapshot (what the system sees today), and
  - prices up to today (closes_to_now) - enforced by an explicit time-fence check.
It scores one-step neighbours of the current settings and each indicator's information coefficient on the
last CLOSED period only, shrinks the evidence toward the defaults (thin data / coincidences), switches one small
step at a time after a cooling-off period, and reverts fast when a deviation suddenly stops working (C15).
The meta-parameters in META are the 'training basis' the outer loop tunes between rounds (C16)."""
import numpy as np
import pandas as pd

from . import policy

META_DEFAULT = {
    "half_life": 6,          # weeks: how fast old evidence fades
    "prior_weeks": 8,        # shrinkage strength toward the defaults (thin data / coincidence guard)
    "switch_z": 2.0,         # a neighbour must lead by this many standard errors to be adopted
    "min_weeks": 4,          # no switching before this much evidence
    "cooldown": 3,           # weeks between switches
    "revert_drop": 0.03,     # a deviation that loses this much vs default over 2 periods is reverted
    "ic_beta": 0.5,          # indicator weights barely matter (sensitivity study, 39 windows) - adapt gently
    "ic_clip": 1.0,          # weights stay within prior x [1-clip, 1+clip]
    # what the sensitivity study (C14) found worth adapting: w_model and liq_q significant+consistent;
    # k is the volatility lever (C21); pool_q slight. Everything else measured as noise.
    "adaptive_knobs": ["w_model", "liq_q", "k", "pool_q"],
}
STEPS = {"k": [2, 3, 4, 6, 8, 12, 16], "exit_q": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95], "w_model": [0.0, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0],
         "pool_q": [0.8, 0.9, 0.95, 0.98, 0.99], "liq_q": [0.0, 0.2, 0.3, 0.5, 0.7, 0.85],
         "stress_thr": [None, 0.9, 0.95, 1.0, 1.05, 1.1], "brake": [None, 0.03, 0.05, 0.08, 0.12, 0.2],
         "rebalance_weeks": [1, 2, 3, 4]}


class TimeFence(Exception):
    pass


def eligible(p, cfg):
    ok = ~((p["ev_red_flag"] > 0) | ((p["ev_offering"] > 0) & (p["log_dv"].rank(pct=True) < 0.5)))
    if cfg["vol_filter"]:
        ok &= ~((p["vol20"].rank(pct=True) > 0.9) | (p["max20"].rank(pct=True) > 0.9))
    ok &= p["log_dv"].rank(pct=True) >= cfg["liq_q"]
    return ok & policy.not_crypto(p.index)


def pick(p, cfg, held, divs, det=None):
    """The selection rule (same as the trader's): score -> eligible -> regime-aware top-k.
    det: optional missed-winner detector probabilities, blended in with weight cfg['det_w']."""
    if cfg.get("ew"):
        p = p.assign(evidence=policy.evidence_from(p, cfg["ew"]))
    s = policy.score(p, cfg["w_model"])
    if det is not None and cfg.get("det_w", 0) > 0:
        s = (1 - cfg["det_w"]) * s + cfg["det_w"] * det.reindex(s.index).rank(pct=True).fillna(0.5)
    mkt = {c: float(p[c].iloc[0]) for c in p.columns if c.startswith("m_")}
    return policy.regime_targets(s[eligible(p, cfg)], list(held), cfg, p["vol20"], divs, mkt)


def neighbours(cfg, knobs):
    out = []
    for k in knobs:
        vals = STEPS.get(k)
        if not vals or cfg.get(k) not in vals:
            continue
        i = vals.index(cfg.get(k))
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals):
                out.append((k, vals[j]))
    return out


class Adapter:
    def __init__(self, default_cfg, meta=None):
        self.meta = {**META_DEFAULT, **(meta or {})}
        self.default = dict(default_cfg)
        self.cfg = dict(default_cfg)
        self.cfg.setdefault("ew", policy.default_evidence_weights())
        self.default.setdefault("ew", policy.default_evidence_weights())
        self.prev = None                     # (date, snapshot) of the last decision
        self.stats = {}                      # arm -> [ewma sum of excess, ewma weight, ewma sq]
        self.ic = {}                         # indicator -> [ewma ic, ewma weight, n]
        self.since_switch = 99
        self.weeks = 0
        self.log = []
        self.recent = []                     # current deviation's excess vs default, last periods
        self.det = MissedWinnerDetector(self.meta)
        self.missed = []                     # per closed period: winners it did not pick, and their profile

    def _decay(self):
        return 0.5 ** (1.0 / self.meta["half_life"])

    def step(self, today, snap, closes_to_now, divs, held):
        """Called at each decision. closes_to_now must end at `today` (time fence)."""
        if closes_to_now.index.max() > pd.Timestamp(today):
            raise TimeFence(f"adapter was handed prices after {today}")
        if self.prev is not None:
            self._learn(self.prev[0], self.prev[1], today, closes_to_now, divs, held)
        self.prev = (today, snap)
        self.weeks += 1
        return dict(self.cfg)

    def _period_return(self, names, d0, d1, closes):
        if not len(names):
            return 0.0
        seg = closes.loc[d0:d1, [n for n in names if n in closes.columns]]
        if len(seg) < 2 or seg.shape[1] == 0:
            return 0.0
        return float((seg.iloc[-1] / seg.iloc[0] - 1).mean())

    def _learn(self, d0, p0, d1, closes, divs, held):
        m, dec = self.meta, self._decay()
        cur_names = list(pick(p0, self.cfg, held, divs).index)
        base_r = self._period_return(cur_names, d0, d1, closes)
        def_names = list(pick(p0, self.default, held, divs).index)
        def_r = self._period_return(def_names, d0, d1, closes)
        # 1) knob neighbours: one-step counterfactuals on the period that has just CLOSED
        for k, v in neighbours(self.cfg, m["adaptive_knobs"]):
            alt = {**self.cfg, k: v}
            r = self._period_return(list(pick(p0, alt, held, divs).index), d0, d1, closes)
            s = self.stats.setdefault((k, v), [0.0, 0.0, 0.0])
            x = r - base_r
            s[0], s[1], s[2] = s[0] * dec + x, s[1] * dec + 1, s[2] * dec + x * x
        # 2) indicator ICs on the closed period (whole eligible universe, not just holdings)
        fwd = (closes.loc[d1] / closes.loc[d0] - 1).reindex(p0.index)
        for c in list(self.cfg["ew"]):
            col = f"e_{c}"
            if col not in p0:
                continue
            ok = fwd.notna() & p0[col].notna()
            if ok.sum() < 30 or p0.loc[ok, col].nunique() < 3:
                continue
            ic = float(p0.loc[ok, col].rank().corr(fwd[ok].rank()))
            e = self.ic.setdefault(c, [0.0, 0.0, 0])
            e[0], e[1], e[2] = e[0] * dec + ic, e[1] * dec + 1, e[2] + 1
        self._reweight()
        # 2b) canon C20: study every winner of the closed period, picked or not
        self.det.learn(p0, fwd)
        self.cfg["det_w"] = self.det.weight()
        win = fwd[fwd >= 0.07].index
        missed = [t for t in win if t not in set(cur_names)]
        if len(win):
            prof = {c: float(p0.loc[missed, c].mean() - p0.loc[cur_names, c].mean())
                    for c in DET_FEATS if c in p0 and len(missed) and len(cur_names)}
            self.missed.append({"date": str(d1), "winners": int(len(win)), "missed": len(missed),
                                "caught": int(len(win) - len(missed)), "missed_minus_picked": prof,
                                "detector_skill": self.det.skill_mean(), "detector_weight": self.cfg["det_w"]})
        # 3) fast revert: the active deviation from the default suddenly stops working
        if self.cfg_diff():
            self.recent = (self.recent + [base_r - def_r])[-2:]
            if len(self.recent) == 2 and sum(self.recent) < -m["revert_drop"]:
                self.log.append({"date": str(d1), "action": "revert", "why": f"deviation lost {sum(self.recent):.1%} vs default"})
                ew = self.cfg["ew"]
                self.cfg = {**self.default, "ew": ew}
                self.stats, self.recent, self.since_switch = {}, [], 0
                return
        # 4) switch one step if a neighbour leads convincingly (shrunk toward zero by the prior)
        self.since_switch += 1
        if self.weeks < m["min_weeks"] or self.since_switch < m["cooldown"]:
            return
        best, best_z = None, m["switch_z"]
        for arm, (sx, w, sxx) in self.stats.items():
            if arm[0] not in m["adaptive_knobs"] or self.cfg.get(arm[0]) == arm[1]:
                continue
            mean = sx / (w + m["prior_weeks"])                     # shrinkage: thin evidence counts for little
            var = max(sxx / max(w, 1e-9) - (sx / max(w, 1e-9)) ** 2, 1e-8)
            se = np.sqrt(var / max(w, 1.0))
            z = mean / se
            if z > best_z:
                best, best_z = arm, z
        if best:
            self.log.append({"date": str(d1), "action": "switch", "knob": best[0], "from": self.cfg.get(best[0]),
                             "to": best[1], "z": round(float(best_z), 2)})
            self.cfg[best[0]] = best[1]
            self.stats = {a: s for a, s in self.stats.items() if a[0] != best[0]}
            self.since_switch, self.recent = 0, []

    def _reweight(self):
        m = self.meta
        ew = {}
        for c, w0 in self.default["ew"].items():
            e = self.ic.get(c)
            if not e or e[1] <= 0:
                ew[c] = w0
                continue
            mean_ic = e[0] / (e[1] + m["prior_weeks"])              # shrink toward 0 with thin data
            z = mean_ic * np.sqrt(max(e[2], 1)) / 0.1               # ~0.1 = typical weekly IC noise
            factor = 1 + np.clip(m["ic_beta"] * np.sign(w0) * z / 3, -m["ic_clip"], m["ic_clip"])
            ew[c] = w0 * float(factor)                              # never flips sign; can fade to zero
        self.cfg["ew"] = ew

    def detector_scores(self, p):
        return self.det.predict(p)

    def cfg_diff(self):
        return {k: v for k, v in self.cfg.items() if k != "ew" and self.default.get(k) != v}


def choose_default(warm_snaps, warm_closes, divs, prior_cfg, cost_bps, run_variant, candidates):
    """Pre-season study (C17): score candidate settings on the out-of-sample tail of the warm-up years,
    judge them by quarters (so one lucky stretch can't win), and keep the prior unless a candidate is
    better in most quarters."""
    if not warm_snaps or warm_closes is None or len(warm_closes) < 60:
        return dict(prior_cfg), {"reason": "not enough warm-up data - using the prior"}
    qs = pd.qcut(np.arange(len(warm_closes)), 4, labels=False)
    bounds = [(warm_closes.index[qs == q][0], warm_closes.index[qs == q][-1]) for q in range(4)]

    def quarters(cfg):
        out = []
        for a, b in bounds:
            sn = {k: v for k, v in warm_snaps.items() if a <= pd.Timestamp(k) <= b}
            if not sn:
                continue
            out.append(run_variant(cfg, sn, warm_closes.loc[a:b], cost_bps, divs)["mean_week"])
        return np.array(out)

    prior_q = quarters(prior_cfg)
    best, best_score, best_q = dict(prior_cfg), 0.0, prior_q
    for c in candidates:
        cq = quarters(c)
        if len(cq) != len(prior_q) or len(cq) == 0:
            continue
        diff = cq - prior_q
        wins = float((diff > 0).mean())
        score = float(diff.mean()) if wins >= 0.75 else -1.0
        if score > best_score:
            best, best_score, best_q = dict(c), score, cq
    return best, {"prior_quarters": prior_q.tolist(), "chosen_quarters": best_q.tolist(), "gain": best_score}



DET_FEATS = ["e_ear", "e_ins_buyers30", "e_ins_officer30", "e_dist_52wh", "e_ind_mom60", "e_frog", "e_mom_12_1",
             "e_r5_nonews", "e_max20", "e_skew60", "mu_raw", "vol20", "max20", "log_dv", "r5"]


class MissedWinnerDetector:
    """Canon C20: learns, week by week, what the next +7% stocks looked like - including the hundreds it did
    not pick - and earns influence only by predicting LATER weeks correctly (thin data / coincidence guard).
    Online logistic regression on cross-sectional ranks; deterministic."""

    def __init__(self, meta):
        self.meta = meta
        self.coef = np.zeros(len(DET_FEATS))
        self.bias = 0.0
        self.skill = []                      # out-of-sample rank-IC of its predictions, one per closed period
        self.n = 0

    def _x(self, p):
        cols = []
        for c in DET_FEATS:
            v = p[c] if c in p else pd.Series(0.0, index=p.index)
            cols.append(v.rank(pct=True).fillna(0.5).values - 0.5)
        return np.column_stack(cols)

    def predict(self, p):
        z = self._x(p) @ self.coef + self.bias
        return pd.Series(1 / (1 + np.exp(-z)), index=p.index)

    def learn(self, p0, fwd):
        ok = fwd.notna()
        if ok.sum() < 30:
            return
        p0, y = p0[ok], (fwd[ok] >= 0.07).astype(float)
        if y.sum() == 0:                                   # quiet market: learn from the top 5% instead
            y = (fwd[ok] >= fwd[ok].quantile(0.95)).astype(float)
        # 1) judge the detector on this period BEFORE it learns from it (out-of-sample)
        if self.n > 0:
            pr = self.predict(p0)
            self.skill.append(float(pr.rank().corr(fwd[ok].rank())))
        # 2) then learn: a few gradient steps, L2 toward zero, positives up-weighted
        X = self._x(p0)
        w = np.where(y.values > 0, 0.5 / max(y.mean(), 1e-3), 0.5 / max(1 - y.mean(), 1e-3))
        lr, lam = self.meta.get("det_lr", 0.5), self.meta.get("det_l2", 0.05)
        for _ in range(5):
            pr = 1 / (1 + np.exp(-(X @ self.coef + self.bias)))
            g = (pr - y.values) * w
            self.coef -= lr * (X.T @ g / len(g) + lam * self.coef)
            self.bias -= lr * float(g.mean())
        self.n += 1

    def skill_mean(self):
        return float(np.mean(self.skill)) if self.skill else 0.0

    def weight(self):
        """0 until it has predicted enough later weeks; then grows with its shrunk, significant skill."""
        k = len(self.skill)
        if k < self.meta.get("det_min_weeks", 6):
            return 0.0
        m = float(np.sum(self.skill)) / (k + self.meta["prior_weeks"])
        sd = float(np.std(self.skill, ddof=1)) if k > 1 else 1.0
        z = m / (sd / np.sqrt(k)) if sd > 0 else 0.0
        return float(np.clip((z - 1.0) / 4.0, 0.0, self.meta.get("det_max", 0.5)))


class Session:
    """The whole daily trading loop, shared verbatim by the blind live trader and the re-tester.
    Each day: on_day(date, prices_today, closes_to_now, next_is_new_week, snapshot_fn)."""

    def __init__(self, default_cfg, divs, cost_bps, start_cash=1000.0, adaptive=False, meta=None):
        from . import policy as _p
        self.P = _p
        self.cfg = dict(default_cfg)
        self.cfg.setdefault("ew", _p.default_evidence_weights())
        self.divs, self.bps, self.adaptive = divs, cost_bps, adaptive
        self.adapter = Adapter(self.cfg, meta) if adaptive else None
        self.cash, self.pos = start_cash, {}
        self.week_start, self.capped, self.wk = start_cash, False, 0
        self.days, self.weeks, self.decisions, self.orders, self.week_rows = [], [], [], [], []

    def equity(self, px):
        return self.cash + sum(q * px[t] for t, q in self.pos.items() if np.isfinite(px.get(t, np.nan)))

    def needs_snapshot(self, next_is_new_week):
        return (next_is_new_week and self.wk % self.cfg.get("rebalance_weeks", 1) == 0) or not self.pos

    def _trade(self, target, px, val, day, reason):
        for t in sorted(set(self.pos) | set(target.index), key=lambda c: target.get(c, 0.0)):
            pr = px.get(t, np.nan)
            if not np.isfinite(pr):
                continue
            dv = target.get(t, 0.0) * 0.985 * val - self.pos.get(t, 0.0) * pr
            if abs(dv) < 1.0:
                continue
            self.cash -= dv + abs(dv) * self.bps / 1e4
            self.pos[t] = self.pos.get(t, 0.0) + dv / pr
            self.orders.append((str(day.date()), t, round(float(dv), 2), reason))
            if abs(self.pos[t]) * pr < 0.5:
                self.pos.pop(t)

    def on_day(self, day, px, closes_to_now, next_is_new_week, snap=None):
        if closes_to_now is not None and closes_to_now.index.max() > day:
            raise TimeFence(f"session handed prices after {day}")
        val = self.equity(px)
        wr = val / self.week_start - 1
        if snap is not None:
            held = list(self.pos)
            if self.adapter is not None:
                self.cfg = self.adapter.step(day, snap, closes_to_now, self.divs, held)
                det = self.adapter.detector_scores(snap)
            else:
                det = None
            target = pick(snap, self.cfg, held, self.divs, det)
            self._trade(target, px, val, day, "rebalance" if held else "initial build")
            self.decisions.append((str(day.date()), sorted(target.index)))
        elif not self.capped and self.cfg.get("brake") and wr <= -self.cfg["brake"]:
            target = pd.Series({t: q * px[t] / val for t, q in self.pos.items()}) * self.P.TOPK["brake_exposure"]
            self._trade(target, px, val, day, f"weekly brake ({wr:.1%})")
            self.capped = True
        val = self.equity(px)
        self.days.append((str(day.date()), float(val)))
        if next_is_new_week:
            self.weeks.append(val / self.week_start - 1)
            self.week_rows.append({"week_end": str(day.date()), "ret": val / self.week_start - 1,
                                   "brake": self.capped, "holdings": sorted(self.pos)})
            self.week_start, self.capped = val, False
            self.wk += 1

    def result(self, start_cash=1000.0):
        e = pd.Series([v for _, v in self.days])
        w = np.array(self.weeks) if self.weeks else np.array([0.0])
        return {"mean_week": float(w.mean()), "weeks_ge_7": int((w >= 0.07).sum()), "weeks_le_m7": int((w <= -0.07).sum()),
                "year_return": float(e.iloc[-1] / start_cash - 1), "max_dd": float((e / e.cummax() - 1).min()),
                "adaptations": self.adapter.log if self.adapter else [], "missed_winners": self.adapter.missed if self.adapter else []}


def replay(default_cfg, snaps, closes, cost_bps, divs, adaptive=False, meta=None, scramble_after=None, seed=0):
    """Re-tester: drives the SAME Session through an archived window.
    scramble_after: anti-cheat test - replace every price after this date with noise; decisions up to that
    date must not change (if they do, something looked into the future)."""
    if scramble_after is not None:
        closes = closes.astype("float64").copy()
        rng = np.random.default_rng(seed)
        m = closes.index > pd.Timestamp(scramble_after)
        closes.loc[m] = closes.loc[m].values * np.exp(rng.normal(0, 0.2, closes.loc[m].shape))
    S = Session(default_cfg, divs, cost_bps, adaptive=adaptive, meta=meta)
    dec = {pd.Timestamp(k): v for k, v in snaps.items()}
    sessions = closes.index
    for i, d in enumerate(sessions):
        nxt_new = i + 1 >= len(sessions) or sessions[i + 1].isocalendar().week != d.isocalendar().week
        snap = dec.get(d) if S.needs_snapshot(nxt_new) else None
        if snap is None and S.needs_snapshot(nxt_new) and not S.pos:
            snap = None
        S.on_day(d, closes.loc[d], closes.loc[:d], nxt_new, snap)
    return S
