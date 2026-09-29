"""Canon C34: route the Adapter's learning through engine.memory (factor-weighted memory) and add long-term memory."""
import pathlib

p = pathlib.Path("engine/adaptive.py"); s = p.read_text(encoding="utf-8")
rep = [
('''    def __init__(self, default_cfg, meta=None):
        self.meta = {**META_DEFAULT, **(meta or {})}''',
 '''    def __init__(self, default_cfg, meta=None, long_term=None):
        from .memory import Memory, MEM_DEFAULT
        self.meta = {**META_DEFAULT, **MEM_DEFAULT, **(meta or {})}
        self.mem = Memory({k: self.meta[k] for k in MEM_DEFAULT}, long_term=long_term)   # C34'''),
('''        if self.prev is not None:
            self._learn(self.prev[0], self.prev[1], today, closes_to_now, divs, held)''',
 '''        if self.prev is not None:
            from .memory import context_of
            self._learn(self.prev[0], self.prev[1], today, closes_to_now, divs, held, ctx_now=context_of(snap))'''),
('''    def _learn(self, d0, p0, d1, closes, divs, held):
        m, dec = self.meta, self._decay()''',
 '''    def _learn(self, d0, p0, d1, closes, divs, held, ctx_now=None):
        from .memory import context_of
        m, dec = self.meta, self._decay()
        ctx0 = context_of(p0)                          # the market the lesson was learned in
        ctx_now = ctx0 if ctx_now is None else ctx_now'''),
('''            s = self.stats.setdefault((k, v), [0.0, 0.0, 0.0])
            x = r - base_r
            s[0], s[1], s[2] = s[0] * dec + x, s[1] * dec + 1, s[2] * dec + x * x''',
 '''            x = r - base_r
            self.mem.record(("knob", k, self.cfg.get(k), v), self.weeks, ctx0, x)   # keyed by the base it was measured from'''),
('''            ic = float(p0.loc[ok, col].rank().corr(fwd[ok].rank()))
            e = self.ic.setdefault(c, [0.0, 0.0, 0])
            e[0], e[1], e[2] = e[0] * dec + ic, e[1] * dec + 1, e[2] + 1
        self._reweight()''',
 '''            ic = float(p0.loc[ok, col].rank().corr(fwd[ok].rank()))
            self.mem.record(("ic", c), self.weeks, ctx0, ic)
        self._reweight(ctx_now)'''),
('''                self.stats, self.recent, self.since_switch = {}, [], 0
                return''',
 '''                self.recent, self.since_switch = [], 0
                return'''),
('''        best, best_z = None, m["switch_z"]
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
            self.since_switch, self.recent = 0, []''',
 '''        best, best_z = None, m["switch_z"]
        for k, v in neighbours(self.cfg, m["adaptive_knobs"]):
            mean, se, n_eff = self.mem.estimate(("knob", k, self.cfg.get(k), v), self.weeks, ctx_now)
            if n_eff < m["min_weeks"] or not np.isfinite(se):
                continue
            z = mean / se
            if z > best_z:
                best, best_z = (k, v), z
        if best:
            self.log.append({"date": str(d1), "action": "switch", "knob": best[0], "from": self.cfg.get(best[0]),
                             "to": best[1], "z": round(float(best_z), 2)})
            self.cfg[best[0]] = best[1]
            self.since_switch, self.recent = 0, []'''),
('''    def _reweight(self):
        m = self.meta
        ew = {}
        for c, w0 in self.default["ew"].items():
            e = self.ic.get(c)
            if not e or e[1] <= 0:
                ew[c] = w0
                continue
            mean_ic = e[0] / (e[1] + m["prior_weeks"])              # shrink toward 0 with thin data
            z = mean_ic * np.sqrt(max(e[2], 1)) / 0.1               # ~0.1 = typical weekly IC noise''',
 '''    def _reweight(self, ctx_now=None):
        m = self.meta
        ew = {}
        for c, w0 in self.default["ew"].items():
            mean_ic, se, n_eff = self.mem.estimate(("ic", c), self.weeks, ctx_now if ctx_now is not None else np.zeros(7))
            if n_eff <= 0 or not np.isfinite(se):
                ew[c] = w0
                continue
            z = mean_ic / max(se, 1e-6)'''),
('''    def __init__(self, default_cfg, divs, cost_bps, start_cash=1000.0, adaptive=False, meta=None):''',
 '''    def __init__(self, default_cfg, divs, cost_bps, start_cash=1000.0, adaptive=False, meta=None, long_term=None):'''),
('''        self.adapter = Adapter(self.cfg, meta) if adaptive else None''',
 '''        self.adapter = Adapter(self.cfg, meta, long_term=long_term) if adaptive else None'''),
('''def replay(default_cfg, snaps, closes, cost_bps, divs, adaptive=False, meta=None, scramble_after=None, seed=0, opens=None):''',
 '''def replay(default_cfg, snaps, closes, cost_bps, divs, adaptive=False, meta=None, scramble_after=None, seed=0, opens=None,
           long_term=None):'''),
('''    S = Session(default_cfg, divs, cost_bps, adaptive=adaptive, meta=meta)''',
 '''    S = Session(default_cfg, divs, cost_bps, adaptive=adaptive, meta=meta, long_term=long_term)'''),
]
for a, b in rep:
    assert a in s, a[:70]
    s = s.replace(a, b)
p.write_text(s, encoding="utf-8", newline="\n")

# long-term bank: the feed filters it (knows real dates); the trader never sees dates
p = pathlib.Path("engine/livesim.py"); s = p.read_text(encoding="utf-8")
rep = [
('''    # ---- what the trader may see ----''',
 '''    # ---- long-term memory (C34): only lessons from windows that ENDED before this one began ----
    def long_term_memory(self):
        bank = DIR / "memory_bank.parquet"
        if not bank.exists():
            return None
        b = pd.read_parquet(bank)
        start = self.first_live - self._shift                    # real start date, known only to the feed
        b = b[pd.to_datetime(b["real_end"]) < start]
        return b[["arm", "ctx", "outcome"]].reset_index(drop=True) if len(b) else None

    def real_end(self):
        return str((self.sessions[-1] - self._shift).date())

    # ---- what the trader may see ----'''),
('''        self.session = self.A.Session(self.cfg, self.divs, self.feed.cost_bps, adaptive=self.adaptive, meta=self.meta)''',
 '''        self.long_term = self.feed.long_term_memory() if self.adaptive else None
        self.session = self.A.Session(self.cfg, self.divs, self.feed.cost_bps, adaptive=self.adaptive, meta=self.meta,
                                      long_term=self.long_term)'''),
]
for a, b in rep:
    assert a in s, a[:70]
    s = s.replace(a, b)
p.write_text(s, encoding="utf-8", newline="\n")

# loop2: archive the long-term memory each window used (so the re-tester reproduces it), add to the bank after,
# and let the outer loop tune the memory factors
p = pathlib.Path("scripts/livesim_loop2.py"); s = p.read_text(encoding="utf-8")
rep = [
('''    opens = pd.read_parquet(a / "opens_v2.parquet") if (a / "opens_v2.parquet").exists() else None''',
 '''    opens = pd.read_parquet(a / "opens_v2.parquet") if (a / "opens_v2.parquet").exists() else None
    ltm = pd.read_parquet(a / "ltm.parquet") if (a / "ltm.parquet").exists() else None'''),
('''    return {"id": a.name, "snaps": ws, "closes": closes, "opens": opens,''',
 '''    return {"id": a.name, "snaps": ws, "closes": closes, "opens": opens, "ltm": ltm,'''),
('''    S = A.replay(cfg, w["snaps"], w["closes"], w["bps"], w["divs"], adaptive=True, meta=meta, opens=w["opens"])''',
 '''    S = A.replay(cfg, w["snaps"], w["closes"], w["bps"], w["divs"], adaptive=True, meta=meta, opens=w["opens"],
                 long_term=w["ltm"])'''),
('''        S1 = A.replay(res["used_cfg"], w["snaps"], w["closes"], w["bps"], w["divs"], adaptive=True, meta=res["meta"], opens=w["opens"])''',
 '''        S1 = A.replay(res["used_cfg"], w["snaps"], w["closes"], w["bps"], w["divs"], adaptive=True, meta=res["meta"],
                      opens=w["opens"], long_term=w["ltm"])'''),
('''                      scramble_after=cut, seed=len(r), opens=w["opens"])''',
 '''                      scramble_after=cut, seed=len(r), opens=w["opens"], long_term=w["ltm"])'''),
('''    feed._stocks["Open"].loc[feed.first_live:].to_parquet(a / "opens_v2.parquet")
    for k, v in trader.snaps.items():''',
 '''    feed._stocks["Open"].loc[feed.first_live:].to_parquet(a / "opens_v2.parquet")
    if trader.long_term is not None:
        trader.long_term.to_parquet(a / "ltm.parquet")
    ep = trader.session.adapter.mem.export()
    if len(ep):                                           # add this window's lessons to the long-term bank
        ep["real_end"] = feed.real_end()
        ep["window"] = run_id
        bank = DIR / "memory_bank.parquet"
        (pd.concat([pd.read_parquet(bank), ep]) if bank.exists() else ep).to_parquet(bank)
    for k, v in trader.snaps.items():'''),
('''              "det_max": [0.0, 0.25, 0.5], "det_min_weeks": [4, 8]}''',
 '''              "det_max": [0.0, 0.25, 0.5], "det_min_weeks": [4, 8],
              # C34 memory factors: fade speed, market-similarity width, weight of earlier windows, shrinkage, shock
              "mem_half_life": [4, 8, 16, 32], "mem_bandwidth": [0.75, 1.5, 3.0], "mem_prior_scale": [0.0, 0.1, 0.3, 0.6],
              "mem_shrink": [2, 6, 12], "mem_shock_k": [1.5, 2.5, 4.0], "mem_shock_cut": [0.1, 0.25, 0.5]}'''),
]
for a, b in rep:
    assert a in s, a[:70]
    s = s.replace(a, b)
p.write_text(s, encoding="utf-8", newline="\n")
print("C34 wired")
