"""Factor-weighted memory for the self-learning system (canon C34).

Every learned outcome ("arm" X did this well in week t, under market context c) is stored as an episode.
When the system asks "how good is arm X right now?", each episode is weighted by:

  recency      0.5 ** (age / half_life)                       - old weeks fade (half_life is tuned)
  similarity   exp(-||c - c_now||^2 / (2 * bandwidth^2))       - past weeks in similar markets count more
  shock        episodes before a detected break in the arm's results are cut by shock_cut
  source       long-term episodes from EARLIER windows count prior_scale as much as this window's own

and the estimate is shrunk toward zero by a pseudo-count (reliability: thin or noisy evidence counts little).
Deterministic; uses only episodes already recorded (the caller records a week only after it has closed)."""
import numpy as np
import pandas as pd

CTX = ["m_vix", "m_vix_term", "m_spy_ma200", "m_spy_ma50", "m_breadth", "m_dispersion", "m_spy_r5"]
MEM_DEFAULT = {"mem_half_life": 8.0, "mem_bandwidth": 1.5, "mem_prior_scale": 0.3, "mem_shrink": 6.0,
               "mem_shock_k": 2.5, "mem_shock_cut": 0.25}


def context_of(snap):
    """Market context of a decision day, from the snapshot the system already sees."""
    return np.array([float(snap[c].iloc[0]) if c in snap and snap[c].notna().any() else 0.0 for c in CTX])


class Memory:
    def __init__(self, params=None, long_term=None):
        self.p = {**MEM_DEFAULT, **(params or {})}
        self.ep = []                         # (arm, week, ctx, outcome, source) ; source 0 = this window, 1 = long-term
        self.breaks = {}                     # arm -> week of the last detected break
        self.cusum = {}                      # arm -> (pos, neg, mean, var, n)
        self.scale = None
        if long_term is not None and len(long_term):
            import ast
            for r in long_term.itertuples(index=False):
                arm = ast.literal_eval(r.arm) if isinstance(r.arm, str) else r.arm      # stored as text
                self.ep.append((arm, -1e6, np.asarray(r.ctx, dtype=float), float(r.outcome), 1))
            self._fit_scale()

    def _fit_scale(self):
        C = np.array([e[2] for e in self.ep]) if self.ep else np.zeros((1, len(CTX)))
        sd = C.std(axis=0)
        self.scale = np.where(sd > 1e-9, sd, 1.0)

    def record(self, arm, week, ctx, outcome):
        self.ep.append((arm, float(week), np.asarray(ctx, dtype=float), float(outcome), 0))
        if self.scale is None or len(self.ep) % 25 == 0:
            self._fit_scale()
        # shock detection: two-sided CUSUM on this arm's own outcomes
        pos, neg, m, v, n = self.cusum.get(arm, (0.0, 0.0, 0.0, 0.0, 0))
        n += 1
        d = outcome - m
        m += d / n
        v += d * (outcome - m)
        sd = np.sqrt(v / max(n - 1, 1)) if n > 2 else None
        if sd and sd > 0:
            z = (outcome - m) / sd
            pos, neg = max(0.0, pos + z - 0.5), min(0.0, neg + z + 0.5)
            if pos > self.p["mem_shock_k"] * 2 or neg < -self.p["mem_shock_k"] * 2:
                self.breaks[arm] = week
                pos, neg = 0.0, 0.0
        self.cusum[arm] = (pos, neg, m, v, n)

    def weight(self, e, week_now, ctx_now):
        arm, wk, ctx, _, src = e
        age = 0.0 if src else max(0.0, week_now - wk)
        w = 0.5 ** (age / self.p["mem_half_life"])
        if self.scale is not None:
            d = (ctx - ctx_now) / self.scale
            w *= float(np.exp(-float(d @ d) / (2 * self.p["mem_bandwidth"] ** 2)))
        if src:
            w *= self.p["mem_prior_scale"]
        b = self.breaks.get(arm)
        if b is not None and not src and wk < b:
            w *= self.p["mem_shock_cut"]
        elif b is not None and src:
            w *= self.p["mem_shock_cut"]
        return w

    def estimate(self, arm, week_now, ctx_now):
        """(shrunk mean, standard error, effective sample size) for an arm, weighted as described above."""
        ws, xs = [], []
        for e in self.ep:
            if e[0] == arm:
                ws.append(self.weight(e, week_now, ctx_now)); xs.append(e[3])
        if not ws:
            return 0.0, np.inf, 0.0
        w, x = np.array(ws), np.array(xs)
        sw = w.sum()
        if sw <= 0:
            return 0.0, np.inf, 0.0
        n_eff = sw ** 2 / (w ** 2).sum()
        mean = float((w * x).sum() / (sw + self.p["mem_shrink"] * w.mean()))       # shrink toward 0
        var = float((w * (x - (w * x).sum() / sw) ** 2).sum() / sw)
        se = np.sqrt(max(var, 1e-8) / max(n_eff, 1.0))
        return mean, se, float(n_eff)

    def export(self):
        """This window's own episodes, for the long-term bank."""
        return pd.DataFrame([{"arm": repr(a), "ctx": [float(x) for x in c], "outcome": float(o)}
                             for a, _, c, o, s in self.ep if s == 0])
