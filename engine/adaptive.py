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
The meta-parameters in META are the 'training basis' the outer loop tunes between rounds (C16).

Bible Phase 18 (weekly adapter). Guard rails, all explicit and all covered by tests/test_adapter.py:
  minimum evidence   no change before `min_weeks` weeks and `min_weeks` effective samples behind a candidate
  cooling off        `cooldown` weeks between any two changes; `knob_cooldown` extra weeks for the knob that moved
  significance       a neighbour must lead by `switch_z` standard errors AND by `min_improve` in absolute terms
  one step           only grid neighbours (STEPS index +-1) of adaptive knobs; anything else raises GuardRailError
  rapid revert       a deviation that loses `revert_drop` over 2 periods (or `revert_fast` in one) returns to the default;
                     `revert_lockout` weeks then block re-adopting the exact value that failed
  budget             `max_switches` changes per window (None = unlimited)
  hands off          set_knob() works only before the first decision; afterwards it raises ManualInterventionError
Every week ends in one hash-chained audit record (hold / switch / revert with the evidence table behind it), so a
replay of the same inputs must end on the same audit_digest() and the trail can be checked against the settings in force."""
import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import policy
from .missed_winners import DET_FEATS, MissedLedger, MissedWinnerDetector, missed_profile, winner_type   # Phase 14 lives there

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
    "adaptive_knobs": ["w_model", "liq_q", "k", "pool_q", "w_move", "w_mom"],
    # Phase 18 guard rails. Defaults keep the pre-Phase-18 behaviour; a study turns a rail on by name.
    "min_improve": 0.0,      # absolute mean weekly improvement a neighbour must show, on top of switch_z
    "knob_cooldown": 0,      # extra weeks before the SAME knob may move again
    "max_switches": None,    # cap on adopted changes per window
    "revert_lockout": 0,     # weeks a reverted (knob, value) may not be re-adopted
    "revert_fast": None,     # one-week loss vs default that reverts immediately (None = only the 2-period rule)
    "mem_score_errors": True,   # classify each lesson (correct / false positive / ...) against what memory predicted
    "mem_use_dates": False,     # feed calendar dates to the memory (era + modernity factors); off in blind windows
    # missed-winner detector (Phase 14): read by MissedWinnerDetector through meta; listed here with their effective values
    "det_max": 0.5,          # cap on the detector's share of the pick score (the hard cap in missed_winners is also 0.5)
    "det_min_weeks": 6,      # out-of-sample weeks it must have been judged on before it may earn any weight
    # timeline dial hook (Session only, OFF by default): set dial_on True (and optionally dial_params=timeline.DialParams)
    "dial_on": False,
}
STEPS = {"k": [1, 2, 3, 4, 6, 8, 12, 16], "exit_q": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95], "w_model": [0.0, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0],
         "pool_q": [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99], "liq_q": [0.0, 0.2, 0.3, 0.5, 0.7, 0.85],
         "stress_thr": [None, 0.9, 0.95, 1.0, 1.05, 1.1], "brake": [None, 0.03, 0.05, 0.08, 0.12, 0.2],
         "rebalance_weeks": [1, 2, 3, 4], "w_move": [0.0, 0.3, 0.5, 0.7], "w_mom": [0.0, 0.2, 0.4]}


def _wiring():
    from .learning import wiring                   # lazy: the learning package is a sink here, never a dependency of a decision
    return wiring


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
    s, p = pick_score(p, cfg, det)
    mkt = {c: float(p[c].iloc[0]) for c in p.columns if c.startswith("m_")}
    return policy.regime_targets(s[eligible(p, cfg)], list(held), cfg, p["vol20"], divs, mkt)


def pick_score(p, cfg, det=None):
    """The cross-sectional score pick() ranks by, over the WHOLE snapshot (before eligibility), and the snapshot it was computed
    on. The learning sinks read it to say how far below the cut a missed winner ranked."""
    if cfg.get("ew"):
        p = p.assign(evidence=policy.evidence_from(p, cfg["ew"]))
    s = policy.score(p, cfg["w_model"])
    # C30 paths: big-mover probability (volatility finder) and momentum continuation (strong stocks near highs)
    wm, wo = cfg.get("w_move", 0.0), cfg.get("w_mom", 0.0)
    if wm > 0 and "p_move" in p:
        s = (1 - wm) * s + wm * p["p_move"].rank(pct=True)
    if wo > 0 and "e_dist_52wh" in p and "r5" in p:
        mom = (p["e_dist_52wh"].rank(pct=True) + p["r5"].rank(pct=True)) / 2
        s = (1 - wo) * s + wo * mom
    if det is not None and cfg.get("det_w", 0) > 0:
        s = (1 - cfg["det_w"]) * s + cfg["det_w"] * det.reindex(s.index).rank(pct=True).fillna(0.5)
    return s, p


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


class GuardRailError(Exception):
    """A proposed adaptation broke a guard rail (off-grid value, not one step, unknown knob, time going backwards)."""


class ManualInterventionError(GuardRailError):
    """Someone tried to change a setting by hand after the adapter was sealed (a blind year is hands-off)."""


@dataclass
class KnobState:
    """Phase 18: everything the adapter knows about one knob as of the last closed week."""
    knob: str
    base: object = None            # the value in force
    neighbor: object = None        # the best-supported one-step neighbour (highest confidence), if any
    evidence: float = 0.0          # shrunk, factor-weighted mean improvement of that neighbour over the base
    n_eff: float = 0.0             # effective sample behind it
    confidence: float = 0.0        # evidence / standard error (z)
    se: float = float("inf")       # standard error of the improvement
    improvement: float = 0.0       # realised excess of the neighbour over the base in the LAST closed week
    cooldown: int = 0              # weeks until this knob may move again
    last_switch: int = -1          # adapter week of its last change (-1 never)
    revert_state: str = "none"     # none | reverted (has been rolled back) | locked (a re-adoption is blocked)
    reverts: int = 0
    switches: int = 0

    def row(self):
        return asdict(self)


def validate_cfg(cfg, knobs):
    """Every adaptive knob that is set must sit ON its grid (an unset knob is simply not adapted): the adapter moves along STEPS by one index and can move nowhere else."""
    bad = [(k, cfg[k]) for k in knobs if k in STEPS and k in cfg and cfg[k] not in STEPS[k]]
    if bad:
        raise GuardRailError(f"settings off the step grid: {bad}")


def one_step(knob, old, new):
    """True iff `new` is the grid neighbour of `old` (index +-1)."""
    vals = STEPS.get(knob)
    if not vals or old not in vals or new not in vals:
        return False
    return abs(vals.index(new) - vals.index(old)) == 1


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, float):
        return o if np.isfinite(o) else str(o)
    return o


class Adapter:
    def __init__(self, default_cfg, meta=None, long_term=None):
        from .memory import Memory, MEM_DEFAULT, MEM_EXT_DEFAULT
        self.meta = {**META_DEFAULT, **MEM_DEFAULT, **MEM_EXT_DEFAULT, **(meta or {})}
        self.mem = Memory({k: self.meta[k] for k in {**MEM_DEFAULT, **MEM_EXT_DEFAULT}}, long_term=long_term)   # C34
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
        self.ledger = MissedLedger()
        self.missed = self.ledger.rows       # per closed period: winners it did not pick, and their profile
        self._pm_pending = []                # closed weeks awaiting the failure post-mortem (learning sink only; never decides)
        self._lessons_sent = None            # memory state last exported to the learning sinks
        self.knobs = {k: KnobState(k, base=self.cfg.get(k)) for k in self.meta["adaptive_knobs"] if k in STEPS}
        self.audit = []                      # hash-chained record of every weekly decision (Phase 18 audit trail)
        self._chain = "genesis"
        self.locked = {}                     # (knob, value) -> adapter week until which re-adoption is blocked
        self.n_switch = 0
        self.sealed = False                  # True after the first decision: no more manual changes
        self._last_date = None
        self._closed = None                  # what the week that just closed looked like: excess vs default, settings in force
        # a knob whose starting value is off its grid cannot be stepped, so it is frozen (and the audit says so)
        self.frozen = sorted(k for k in self.meta["adaptive_knobs"] if k in STEPS and k in self.cfg and self.cfg[k] not in STEPS[k])
        if self.frozen:
            self._audit("frozen_knobs", {"knobs": self.frozen, "values": {k: self.cfg[k] for k in self.frozen}}, date=None)

    def _decay(self):
        return 0.5 ** (1.0 / self.meta["half_life"])

    # ------------------------------------------------------------------ public entry
    def step(self, today, snap, closes_to_now, divs, held):
        """Called at each decision. closes_to_now must end at `today` (time fence)."""
        if closes_to_now.index.max() > pd.Timestamp(today):
            raise TimeFence(f"adapter was handed prices after {today}")
        if self._last_date is not None and pd.Timestamp(today) < self._last_date:
            raise GuardRailError(f"decision at {today} is earlier than the previous one ({self._last_date.date()})")
        self._last_date = pd.Timestamp(today)
        self.sealed = True
        if self.meta.get("mem_use_dates"):
            self.mem.set_clock(today)
        self._closed = None
        self.flush_post_mortems(today)
        if self.prev is not None:
            from .memory import context_of
            self._learn(self.prev[0], self.prev[1], today, closes_to_now, divs, held, ctx_now=context_of(snap))
        self.prev = (today, snap)
        self.weeks += 1
        return dict(self.cfg)

    def flush_post_mortems(self, now):
        """S21b: closed weeks whose outcome matured strictly before `now` go to the learning failure post-mortem (a sink: it
        records, it never changes a decision). Weeks not yet mature stay queued."""
        if not self._pm_pending:
            return 0
        now_ts = pd.Timestamp(now)
        live = [fx for fx in self._pm_pending if len(fx[0])]          # a week with no outcomes has nothing to explain
        ready = [fx for fx in live if fx[0]["resolved"].iloc[0] < now_ts]
        self._pm_pending = [fx for fx in live if fx[0]["resolved"].iloc[0] >= now_ts]
        for frame, X in ready:
            _wiring().on_post_mortem(frame, X, now)
        return len(ready)

    def send_lessons(self):
        """S21b: export the memory's sanitised lesson bank to the learning sinks once per memory state (dedupes repeated calls)."""
        mark = (len(self.mem), self.mem.seq)
        if mark == self._lessons_sent:
            return 0
        self._lessons_sent = mark
        return _wiring().on_session_end(self.mem) or 0

    def set_knob(self, knob, value, who="manual"):
        """Manual override. Allowed only BEFORE the first decision (the pre-season setting); after that it is refused,
        because no one may intervene inside a blind year. Off-grid values are refused always."""
        if self.sealed:
            raise ManualInterventionError(f"{who} tried to set {knob}={value} after the adapter was sealed")
        if knob in STEPS and value not in STEPS[knob]:
            raise GuardRailError(f"{knob}={value} is not on the step grid {STEPS[knob]}")
        self.cfg[knob] = value
        self.default[knob] = value
        self._audit("preseason_set", {"knob": knob, "value": value, "who": who}, date=None)

    def _period_return(self, names, d0, d1, closes):
        if not len(names):
            return 0.0
        seg = closes.loc[d0:d1, [n for n in names if n in closes.columns]]
        if len(seg) < 2 or seg.shape[1] == 0:
            return 0.0
        return float((seg.iloc[-1] / seg.iloc[0] - 1).mean())

    # ------------------------------------------------------------------ audit trail
    def _audit(self, action, detail, date):
        """Append one record to the hash chain. Each record's hash covers the previous one, so a record cannot be
        edited, dropped or reordered without breaking every hash after it."""
        rec = {"i": len(self.audit), "week": self.weeks, "date": None if date is None else str(date), "action": action,
               "cfg_diff": _jsonable(self.cfg_diff()), "since_switch": self.since_switch,
               "mem": f"{len(self.mem)}:{self.mem.seq}:{len(self.mem.break_log)}", "closed": _jsonable(self._closed),
               "detail": _jsonable(detail)}
        blob = json.dumps(rec, sort_keys=True, default=str)
        self._chain = hashlib.sha256((self._chain + blob).encode()).hexdigest()
        rec["hash"] = self._chain
        self.audit.append(rec)

    # ------------------------------------------------------------------ checkpoint / resume
    def snapshot(self):
        """A deep copy of everything the adapter carries between weeks (memory, detector, settings, knob table, audit
        chain, lockouts, the last snapshot it was handed). from_snapshot() on it continues EXACTLY as an uninterrupted
        adapter would: the live process can be stopped over a weekend without changing a single later decision."""
        import copy
        return {"meta": copy.deepcopy(self.meta), "default": copy.deepcopy(self.default), "cfg": copy.deepcopy(self.cfg),
                "mem": self.mem.state_dict(), "det": self.det.full_state(), "prev": copy.deepcopy(self.prev),
                "since_switch": self.since_switch, "weeks": self.weeks, "log": copy.deepcopy(self.log),
                "recent": list(self.recent), "knobs": {k: asdict(v) for k, v in self.knobs.items()},
                "audit": copy.deepcopy(self.audit), "chain": self._chain, "locked": dict(self.locked),
                "n_switch": self.n_switch, "sealed": self.sealed, "last_date": self._last_date,
                "missed": copy.deepcopy(self.ledger.rows), "frozen": list(self.frozen)}

    @classmethod
    def from_snapshot(cls, st):
        import copy
        from .memory import Memory
        a = cls(st["default"], st["meta"])
        a.default, a.cfg = copy.deepcopy(st["default"]), copy.deepcopy(st["cfg"])
        a.mem = Memory.from_state(st["mem"])
        a.det = MissedWinnerDetector.from_full_state(st["det"])
        a.prev, a.since_switch, a.weeks = copy.deepcopy(st["prev"]), st["since_switch"], st["weeks"]
        a.log, a.recent = copy.deepcopy(st["log"]), list(st["recent"])
        a.knobs = {k: KnobState(**v) for k, v in st["knobs"].items()}
        a.audit, a._chain = copy.deepcopy(st["audit"]), st["chain"]
        a.locked, a.n_switch, a.sealed, a._last_date = dict(st["locked"]), st["n_switch"], st["sealed"], st["last_date"]
        a.ledger.rows[:] = copy.deepcopy(st["missed"])
        a.frozen = list(st["frozen"])
        return a

    def audit_digest(self):
        """The head of the chain: identical inputs must reproduce this string exactly."""
        return self._chain

    def audit_frame(self):
        return pd.DataFrame([{k: r[k] for k in ("i", "week", "date", "action", "since_switch", "hash")} | {"cfg_diff": r["cfg_diff"]}
                             for r in self.audit])

    def verify_audit(self):
        """Recompute the chain. (ok, index_of_first_bad_record or None)."""
        prev = "genesis"
        for r in self.audit:
            body = {k: v for k, v in r.items() if k != "hash"}
            prev = hashlib.sha256((prev + json.dumps(body, sort_keys=True, default=str)).encode()).hexdigest()
            if prev != r["hash"]:
                return False, r["i"]
        return True, None

    def audit_matches_cfg(self):
        """The configuration in force must be exactly what the audit trail says it should be: default + switches - reverts.
        Catches a setting changed outside step() (e.g. a hand edit of adapter.cfg)."""
        return reconstruct_cfg(self.default, self.audit) == {k: v for k, v in self.cfg.items() if k not in ("ew", "det_w")}

    def knob_table(self):
        return pd.DataFrame([s.row() for s in self.knobs.values()])

    # ------------------------------------------------------------------ learning
    def _learn(self, d0, p0, d1, closes, divs, held, ctx_now=None):
        from .memory import context_of
        m, dec = self.meta, self._decay()
        ctx0 = context_of(p0)                          # the market the lesson was learned in
        ctx_now = ctx0 if ctx_now is None else ctx_now
        cur_names = list(pick(p0, self.cfg, held, divs).index)
        base_r = self._period_return(cur_names, d0, d1, closes)
        def_names = list(pick(p0, self.default, held, divs).index)
        def_r = self._period_return(def_names, d0, d1, closes)
        # the closed week as it was: what the settings in force earned versus the defaults (before any decision below)
        self._closed = {"excess": base_r - def_r, "deviation": sorted(self.cfg_diff())}
        dates = m.get("mem_use_dates")
        last_gain = {}
        # 1) knob neighbours: one-step counterfactuals on the period that has just CLOSED
        for k, v in neighbours(self.cfg, m["adaptive_knobs"]):
            alt = {**self.cfg, k: v}
            r = self._period_return(list(pick(p0, alt, held, divs).index), d0, d1, closes)
            x = r - base_r
            arm = ("knob", k, self.cfg.get(k), v)   # keyed by the base it was measured from
            exp = self.mem.estimate(arm, self.weeks, ctx0)[0] if m["mem_score_errors"] else None
            self.mem.record(arm, self.weeks, ctx0, x, date=d0 if dates else None, source_experiment="adapter",
                            expected=exp)
            last_gain[(k, v)] = x
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
            self.mem.record(("ic", c), self.weeks, ctx0, ic, date=d0 if dates else None, source_experiment="adapter")
        self._reweight(ctx_now)
        # 2b) canon C20: study every winner of the closed period, picked or not
        self.det.learn(p0, fwd)
        self.cfg["det_w"] = self.det.weight()
        # S21b: observe() keeps the same ledger row as the old add() AND hands the week to the learning ledger (why each winner
        # was rejected); the pick score says how far below the cut each one ranked
        score0 = pick_score(p0, self.cfg)[0]
        self.ledger.observe(d0, d1, p0, fwd, cur_names, score=score0, thr=m.get("det_winner", 0.07),
                            detector_skill=self.det.skill_mean(), detector_weight=self.cfg["det_w"], k=int(self.cfg.get("k") or 10))
        # the same closed week goes to the failure post-mortem at the NEXT decision: its outcome matured ON d1, and a record may
        # only be classified strictly after it matured (require_past); pending weeks are sink-only state, not in the snapshot
        fx = _wiring().adapter_week_frame(d0, d1, p0, fwd, cur_names, score0)
        if fx is not None:
            self._pm_pending.append(fx)
        # 3) knob table: evidence for every one-step neighbour, whether or not anything may move this week
        cand = self._evaluate_candidates(ctx_now, last_gain)
        # 4) fast revert: the active deviation from the default suddenly stops working
        if self.cfg_diff():
            self.recent = (self.recent + [base_r - def_r])[-2:]
            fast = m.get("revert_fast")
            slow = len(self.recent) == 2 and sum(self.recent) < -m["revert_drop"]
            quick = fast is not None and self.recent[-1] < -fast
            if slow or quick:
                why = (f"deviation lost {self.recent[-1]:.1%} in one week" if quick and not slow
                       else f"deviation lost {sum(self.recent):.1%} vs default")
                self.log.append({"date": str(d1), "action": "revert", "why": why})
                gone = self.cfg_diff()
                for kn, val in gone.items():
                    if kn in self.knobs:
                        self.knobs[kn].reverts += 1
                        self.knobs[kn].revert_state = "reverted"
                    if m["revert_lockout"] > 0:
                        self.locked[(kn, val)] = self.weeks + m["revert_lockout"]
                ew = self.cfg["ew"]
                self.cfg = {**self.default, "ew": ew}
                self.recent, self.since_switch = [], 0
                for kn in gone:
                    if kn in self.knobs:
                        self.knobs[kn].last_switch = self.weeks
                self._audit("revert", {"why": why, "reverted": gone, "base_r": base_r, "default_r": def_r}, d1)
                return
        # 5) switch one step if a neighbour leads convincingly (shrunk toward zero by the prior)
        self.since_switch += 1
        gate = self._global_gate()
        if gate:
            self._audit("hold", {"why": gate, "top": self._top(cand)}, d1)
            return
        best, best_z = None, m["switch_z"]
        for c in cand:
            if c["blocked"]:
                continue
            if c["z"] > best_z:
                best, best_z = c, c["z"]
        if best:
            k, v = best["knob"], best["to"]
            if k not in m["adaptive_knobs"]:
                raise GuardRailError(f"{k} is not an adaptive knob")
            if not one_step(k, self.cfg.get(k), v):
                raise GuardRailError(f"{k}: {self.cfg.get(k)} -> {v} is not one step on the grid")
            self.log.append({"date": str(d1), "action": "switch", "knob": k, "from": self.cfg.get(k),
                             "to": v, "z": round(float(best_z), 2)})
            self._audit("switch", {"knob": k, "from": self.cfg.get(k), "to": v, "z": best_z, "mean": best["mean"],
                                   "se": best["se"], "n_eff": best["n_eff"], "table": self._top(cand, 5)}, d1)
            self.cfg[k] = v
            self.since_switch, self.recent = 0, []
            self.n_switch += 1
            s = self.knobs[k]
            s.base, s.last_switch, s.switches, s.revert_state = v, self.weeks, s.switches + 1, "none"
            s.cooldown = m["knob_cooldown"]
        else:
            self._audit("hold", {"why": "no neighbour cleared the bar", "top": self._top(cand)}, d1)

    def _global_gate(self):
        m = self.meta
        if self.weeks < m["min_weeks"]:
            return f"minimum evidence: {self.weeks} < {m['min_weeks']} weeks"
        if self.since_switch < m["cooldown"]:
            return f"cooling off: {self.since_switch} < {m['cooldown']} weeks since the last change"
        if m["max_switches"] is not None and self.n_switch >= m["max_switches"]:
            return "switch budget spent"
        return None

    def _evaluate_candidates(self, ctx_now, last_gain):
        """Score every one-step neighbour from memory and apply the per-candidate guard rails. Returns dicts in the
        deterministic order of neighbours(); 'blocked' lists the reasons a candidate may not be adopted this week."""
        m, out = self.meta, []
        for s in self.knobs.values():
            s.base = self.cfg.get(s.knob)
            s.cooldown = max(0, m["knob_cooldown"] - (self.weeks - s.last_switch)) if s.last_switch >= 0 else 0
            s.neighbor, s.evidence, s.n_eff, s.confidence, s.se, s.improvement = None, 0.0, 0.0, 0.0, float("inf"), 0.0
        for key, until in list(self.locked.items()):
            if until <= self.weeks:
                del self.locked[key]
        for s in self.knobs.values():
            if any(kk == s.knob for kk, _ in self.locked):
                s.revert_state = "locked"
            elif s.revert_state == "locked":
                s.revert_state = "reverted"
        for k, v in neighbours(self.cfg, m["adaptive_knobs"]):
            mean, se, n_eff = self.mem.estimate(("knob", k, self.cfg.get(k), v), self.weeks, ctx_now)
            finite = np.isfinite(se)
            z = mean / se if finite and n_eff > 0 else 0.0
            why = []
            if n_eff < m["min_weeks"] or not finite:
                why.append("thin evidence")
            if mean < m["min_improve"]:
                why.append("improvement below floor")
            if self.locked.get((k, v), -1) > self.weeks:
                why.append("locked after revert")
            st = self.knobs[k]
            if st.cooldown > 0:
                why.append("knob cooling off")
            c = {"knob": k, "from": self.cfg.get(k), "to": v, "mean": float(mean), "se": float(se) if finite else float("inf"),
                 "n_eff": float(n_eff), "z": float(z), "blocked": why}
            out.append(c)
            if z > st.confidence or st.neighbor is None:
                st.neighbor, st.evidence, st.n_eff, st.confidence, st.se = v, float(mean), float(n_eff), float(z), c["se"]
                st.improvement = float(last_gain.get((k, v), 0.0))
        return out

    @staticmethod
    def _top(cand, n=3):
        rows = sorted(cand, key=lambda c: -c["z"])[:n]
        return [{"knob": c["knob"], "to": c["to"], "z": round(c["z"], 3), "mean": round(c["mean"], 6), "n_eff": round(c["n_eff"], 2),
                 "blocked": c["blocked"]} for c in rows]

    def _reweight(self, ctx_now=None):
        m = self.meta
        ew = {}
        for c, w0 in self.default["ew"].items():
            mean_ic, se, n_eff = self.mem.estimate(("ic", c), self.weeks, ctx_now if ctx_now is not None else np.zeros(7))
            if n_eff <= 0 or not np.isfinite(se):
                ew[c] = w0
                continue
            z = mean_ic / max(se, 1e-6)
            factor = 1 + np.clip(m["ic_beta"] * np.sign(w0) * z / 3, -m["ic_clip"], m["ic_clip"])
            ew[c] = w0 * float(factor)                              # never flips sign; can fade to zero
        self.cfg["ew"] = ew

    def detector_scores(self, p):
        return self.det.predict(p)

    def cfg_diff(self):
        return {k: v for k, v in self.cfg.items() if k not in ("ew", "det_w") and self.default.get(k) != v}


def reconstruct_cfg(default, audit):
    """Replay the audit trail onto the default settings: the configuration it says is in force (without 'ew', which
    the indicator re-weighting owns). Pre-season sets, switches and reverts are the only actions that change it."""
    cfg = {k: v for k, v in default.items() if k not in ("ew", "det_w")}
    base = dict(cfg)
    for r in audit:
        a, d = r["action"], r["detail"]
        if a == "preseason_set":
            cfg[d["knob"]] = d["value"]
            base[d["knob"]] = d["value"]
        elif a == "switch":
            cfg[d["knob"]] = d["to"]
        elif a == "revert":
            cfg = dict(base)
    return cfg


def audits_equal(a, b):
    """Two audit trails are the same decision history iff their chains end on the same hash and have the same length."""
    return len(a) == len(b) and (not a or a[-1]["hash"] == b[-1]["hash"])


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


class Session:
    """The whole daily trading loop, shared verbatim by the blind live trader and the re-tester.
    Each day: on_day(date, prices_today, closes_to_now, next_is_new_week, snapshot_fn)."""

    def __init__(self, default_cfg, divs, cost_bps, start_cash=1000.0, adaptive=False, meta=None, long_term=None):
        from . import policy as _p
        self.P = _p
        self.cfg = dict(default_cfg)
        self.cfg.setdefault("ew", _p.default_evidence_weights())
        self.divs, self.bps, self.adaptive = divs, cost_bps, adaptive
        self.adapter = Adapter(self.cfg, meta, long_term=long_term) if adaptive else None
        self.dial_on = bool(adaptive and (meta or {}).get("dial_on"))     # the dial reads the adapter's cadence: adaptive only
        self.dial_params = (meta or {}).get("dial_params")
        self.dial_state, self.dial_log, self.exposure = None, [], 1.0
        self.cash, self.pos = start_cash, {}
        self.week_start, self.capped, self.wk = start_cash, False, 0
        self.days, self.weeks, self.decisions, self.orders, self.week_rows = [], [], [], [], []
        self.fills = []                              # one dict per executed order: links a fill back to its decision
        self._fill_n = {}
        self.pending = None

    def equity(self, px):
        return self.cash + sum(q * px[t] for t, q in self.pos.items() if np.isfinite(px.get(t, np.nan)))

    def needs_snapshot(self, next_is_new_week):
        return (next_is_new_week and self.wk % self.cfg.get("rebalance_weeks", 1) == 0) or (not self.pos and self.pending is None)

    def _trade(self, target, px, val, day, reason, decision_date=None):
        for t in sorted(set(self.pos) | set(target.index), key=lambda c: (target.get(c, 0.0), c)):
            pr = px.get(t, np.nan)
            if not np.isfinite(pr):
                continue
            dv = target.get(t, 0.0) * 0.985 * val - self.pos.get(t, 0.0) * pr
            if abs(dv) < 1.0:
                continue
            self.cash -= dv + abs(dv) * self.bps / 1e4
            self.pos[t] = self.pos.get(t, 0.0) + dv / pr
            self.orders.append((str(day.date()), t, round(float(dv), 2), reason))
            key = (decision_date, t)
            n = self._fill_n[key] = self._fill_n.get(key, 0) + 1
            self.fills.append({"order_id": f"{decision_date}:{t}:{n}", "ticker": t, "decision_date": decision_date,
                               "fill_date": str(day.date()), "fill_price": float(pr), "dv": round(float(dv), 2), "reason": reason})
            if abs(self.pos[t]) * pr < 0.5:
                self.pos.pop(t)

    def on_day(self, day, px, closes_to_now, next_is_new_week, snap=None, px_open=None):
        """Canon C33: decisions use information up to today's close, so they are executed at the NEXT session's
        open (regular hours only; never after-hours or weekends). px_open = today's opening prices."""
        if closes_to_now is not None and closes_to_now.index.max() > day:
            raise TimeFence(f"session handed prices after {day}")
        # 1) morning: fill yesterday's decision at today's open
        if self.pending is not None:
            target, reason, as_weights, decided = self.pending
            fill = px_open if px_open is not None else px
            val_open = self.equity(fill)
            if not as_weights:                          # brake: scale the current holdings by a factor
                target = pd.Series({t: q * fill.get(t, np.nan) / val_open for t, q in self.pos.items()}) * target
            self._trade(target, fill, val_open, day, reason, decided)
            self.pending = None
        # 2) after the close: look at the day and decide for tomorrow's open
        val = self.equity(px)
        wr = val / self.week_start - 1
        if snap is not None:
            held = list(self.pos)
            if self.adapter is not None:
                self.cfg = self.adapter.step(day, snap, closes_to_now, self.divs, held)
                det = self.adapter.detector_scores(snap)
                if self.dial_on:                             # timeline dial: sees completed weekly returns + the stress reading only
                    from . import timeline
                    vt = float(snap["m_vix_term"].iloc[0]) if "m_vix_term" in snap else float("nan")
                    out, self.dial_state = timeline.step(self.weeks, {"stress": vt} if np.isfinite(vt) else None, None,
                                                         self.dial_state, self.dial_params)
                    self.cfg = {**self.cfg, **timeline.cfg_overrides(out)}
                    self.exposure = out.exposure
                    self.dial_log.append((str(day.date()), out.k, round(out.exposure, 2), out.brake, round(out.aggr, 2)))
            else:
                det = None
            target = pick(snap, self.cfg, held, self.divs, det)
            if self.dial_on:
                target = target * self.exposure
            self.pending = (target, "rebalance" if held else "initial build", True, str(day.date()))
            self.decisions.append((str(day.date()), sorted(target.index)))
        elif not self.capped and self.cfg.get("brake") and wr <= -self.cfg["brake"]:
            self.pending = (self.P.TOPK["brake_exposure"], f"weekly brake ({wr:.1%})", False, str(day.date()))
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
        if self.adapter is not None and self.days:          # S21b: the run's learning sinks (record only; the result is unchanged)
            self.adapter.flush_post_mortems(self.days[-1][0])
            self.adapter.send_lessons()
        e = pd.Series([v for _, v in self.days])
        w = np.array(self.weeks) if self.weeks else np.array([0.0])
        return {"mean_week": float(w.mean()), "weeks_ge_7": int((w >= 0.07).sum()), "weeks_le_m7": int((w <= -0.07).sum()),
                "year_return": float(e.iloc[-1] / start_cash - 1), "max_dd": float((e / e.cummax() - 1).min()),
                "adaptations": self.adapter.log if self.adapter else [], "missed_winners": self.adapter.missed if self.adapter else [],
                "audit_digest": self.adapter.audit_digest() if self.adapter else None,
                "audit_len": len(self.adapter.audit) if self.adapter else 0}


def replay(default_cfg, snaps, closes, cost_bps, divs, adaptive=False, meta=None, scramble_after=None, seed=0, opens=None,
           long_term=None):
    """Re-tester: drives the SAME Session through an archived window.
    scramble_after: anti-cheat test - replace every price after this date with noise; decisions up to that
    date must not change (if they do, something looked into the future)."""
    if scramble_after is not None:
        closes = closes.astype("float64").copy()
        rng = np.random.default_rng(seed)
        m = closes.index > pd.Timestamp(scramble_after)
        noise = np.exp(rng.normal(0, 0.2, closes.loc[m].shape))
        closes.loc[m] = closes.loc[m].values * noise
        if opens is not None:
            opens = opens.astype("float64").copy()
            opens.loc[m] = opens.loc[m].values * noise
    S = Session(default_cfg, divs, cost_bps, adaptive=adaptive, meta=meta, long_term=long_term)
    dec = {pd.Timestamp(k): v for k, v in snaps.items()}
    sessions = closes.index
    for i, d in enumerate(sessions):
        nxt_new = i + 1 >= len(sessions) or sessions[i + 1].isocalendar().week != d.isocalendar().week
        snap = dec.get(d) if S.needs_snapshot(nxt_new) else None
        if snap is None and S.needs_snapshot(nxt_new) and not S.pos:
            snap = None
        S.on_day(d, closes.loc[d], closes.loc[:d], nxt_new, snap,
                 opens.loc[d] if opens is not None and d in opens.index else None)
    return S


# ---------------------------------------------------------------------------------------------------------------
# Phase 18 reporting: what did the adapter do, why did it hold, and does the same input give the same history
# ---------------------------------------------------------------------------------------------------------------
def hold_reasons(audit):
    """Histogram of why the adapter did nothing (minimum evidence, cooling off, budget, no candidate, ...)."""
    out = {}
    for r in audit:
        if r["action"] == "hold":
            why = str(r["detail"].get("why", "")).split(":")[0]
            out[why] = out.get(why, 0) + 1
    return out


def deviation_spells(audit):
    """Spells during which the settings differed from the default: [(start_week, end_week, ended_by)], ended_by being
    'revert' or 'open' (still deviating at the end). How long the adapter dwells away from the default it was given."""
    spells, start = [], None
    for r in audit:
        dev = bool(r["cfg_diff"])
        if dev and start is None:
            start = r["week"]
        elif not dev and start is not None:
            spells.append((start, r["week"], "revert"))
            start = None
    if start is not None:
        spells.append((start, audit[-1]["week"], "open"))
    return spells


def adaptation_report(adapter):
    """One dict summarising an adapter's run: counts by action, switches by knob, reverts, weeks off default, hold
    reasons, the per-knob state table, and the integrity checks (hash chain, settings vs audit)."""
    au = adapter.audit
    acts = {}
    for r in au:
        acts[r["action"]] = acts.get(r["action"], 0) + 1
    by_knob = {}
    for r in au:
        if r["action"] == "switch":
            k = r["detail"]["knob"]
            by_knob[k] = by_knob.get(k, 0) + 1
    spells = deviation_spells(au)
    off = sum(1 for r in au if r["cfg_diff"])
    ok, bad = adapter.verify_audit()
    return {"weeks": adapter.weeks, "records": len(au), "actions": acts, "switches_by_knob": by_knob,
            "reverts": acts.get("revert", 0), "revert_rate": acts.get("revert", 0) / max(acts.get("switch", 0), 1),
            "weeks_off_default": off, "share_off_default": off / max(len(au), 1), "spells": spells,
            "mean_spell_weeks": float(np.mean([b - a for a, b, _ in spells])) if spells else 0.0,
            "hold_reasons": hold_reasons(au), "knobs": adapter.knob_table(), "chain_ok": ok, "chain_bad_index": bad,
            "settings_match_audit": adapter.audit_matches_cfg(), "digest": adapter.audit_digest(),
            "memory_fingerprint": adapter.mem.fingerprint(), "detector_weight": adapter.det.weight()}


def adaptation_value(audit, n_flips=2000, seed=0):
    """Did the adapting pay? For every closed week the adapter spent OFF its default settings, the excess return of the
    settings in force over the defaults (both measured on the same closed week, picks scored on next-period prices).
    Returns weeks off default, total and mean excess, its t-statistic, a one-sided sign-flip p-value that the mean is
    positive, and the contribution split by knob (a week with two deviating knobs credits each half). Weeks on the
    default contribute exactly zero, so this is the whole in-sample value of the adaptations, before costs."""
    rows = [r["closed"] for r in audit if r.get("closed") and r["closed"]["deviation"]]
    ex = np.array([c["excess"] for c in rows], dtype=float)
    by = {}
    for c in rows:
        for k in c["deviation"]:
            by[k] = by.get(k, 0.0) + c["excess"] / len(c["deviation"])
    if len(ex) < 3:
        return {"weeks_off_default": len(ex), "total_excess": float(ex.sum()), "mean_excess": float(ex.mean()) if len(ex) else 0.0,
                "t": float("nan"), "p_helped": 1.0, "by_knob": by}
    from .missed_winners import sign_flip_p
    sd = ex.std(ddof=1)
    return {"weeks_off_default": len(ex), "total_excess": float(ex.sum()), "mean_excess": float(ex.mean()),
            "t": float(ex.mean() / (sd / np.sqrt(len(ex)))) if sd > 0 else float("nan"),
            "p_helped": sign_flip_p(ex, n=n_flips, seed=seed), "by_knob": by}


def replay_check(default_cfg, snaps, closes, cost_bps, divs, opens=None, meta=None, long_term=None, runs=2):
    """Determinism gate: replay the same inputs `runs` times; the audit digest, memory fingerprint, detector fingerprint,
    decisions, orders and result must all match. Returns {'identical': bool, 'differs': [names], 'digest': str}."""
    outs = []
    for _ in range(runs):
        S = replay(default_cfg, snaps, closes, cost_bps, divs, adaptive=True, meta=meta, opens=opens, long_term=long_term)
        outs.append({"audit": S.adapter.audit_digest(), "memory": S.adapter.mem.fingerprint(), "detector": S.adapter.det.fingerprint(),
                     "decisions": S.decisions, "orders": S.orders, "fills": S.fills, "result": S.result()})
    bad = sorted(k for k in outs[0] if any(o[k] != outs[0][k] for o in outs[1:]))
    return {"identical": not bad, "differs": bad, "digest": outs[0]["audit"], "runs": runs}


def rail_study(default_cfg, snaps, closes, cost_bps, divs, variants, opens=None, base_meta=None, long_term=None):
    """Run the same window under different guard-rail settings and tabulate what each rail does. `variants` is
    {name: meta overrides}; the first row is always the unmodified base. Columns: switches, reverts, weeks off default,
    year return, mean week, max drawdown, audit digest prefix."""
    rows = []
    for name, over in [("base", {})] + list(variants.items()):
        S = replay(default_cfg, snaps, closes, cost_bps, divs, adaptive=True, meta={**(base_meta or {}), **over}, opens=opens,
                   long_term=long_term)
        r, rep = S.result(), adaptation_report(S.adapter)
        rows.append({"variant": name, "switches": rep["actions"].get("switch", 0), "reverts": rep["reverts"],
                     "weeks_off_default": rep["weeks_off_default"], "year_return": r["year_return"], "mean_week": r["mean_week"],
                     "max_dd": r["max_dd"], "digest": rep["digest"][:10]})
    return pd.DataFrame(rows)
