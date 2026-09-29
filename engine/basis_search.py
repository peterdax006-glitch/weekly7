"""Outer training-basis search (Bible PHASE 19; canon C11, C15-C21, C34, C39).

Between sealed rounds the loop trains its starting defaults (cfg) and adaptation meta-parameters (meta):
24 random starting configurations are screened on 10 random archived windows, the top 3 (plus the incumbent) are
confirmed on every archived window, and the next basis is adopted only if the tiered objective firewall accepts it
AND a paired bootstrap over windows says the gain is not one lucky window. The anti-overfit penalty shrinks each
confirmed score by its window-to-window standard error and by how far it fell from its screening score.
The evaluator is injected (`evaluate(window, cfg, meta) -> row`), so the search itself is pure and testable; wiring to
adaptive.replay is the caller's job. Windows carry an `end`: with `as_of` given, later windows are invisible.
"""
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from engine import objective as O

META_SPACE = {"half_life": [3, 6, 12], "prior_weeks": [4, 8, 16], "switch_z": [1.5, 2.0, 3.0], "min_weeks": [3, 6],
              "cooldown": [2, 4], "revert_drop": [0.02, 0.04, 0.08], "ic_beta": [0.0, 0.5, 1.0, 2.0],
              "det_max": [0.0, 0.25, 0.5], "det_min_weeks": [4, 8],
              "mem_half_life": [4, 8, 16, 32], "mem_bandwidth": [0.75, 1.5, 3.0], "mem_prior_scale": [0.0, 0.1, 0.3, 0.6],
              "mem_shrink": [2, 6, 12], "mem_shock_k": [1.5, 2.5, 4.0], "mem_shock_cut": [0.1, 0.25, 0.5]}
CFG_SPACE = {"k": [1, 1, 2, 2, 3, 4], "exit_q": [0.5, 0.7, 0.8, 0.9], "rebalance_weeks": [1, 1, 2], "brake": [None, None, 0.15],
             "max_per_sector": [None, 2], "w_model": [0.85, 1.0, 1.0], "pick": ["hivol", "hivol", "top"],
             "pool_q": [0.3, 0.5, 0.7, 0.9, 0.95], "liq_q": [0.0, 0.0, 0.2], "vol_filter": [False, False, True],
             "stress_thr": [None, 1.0, 1.05], "stress_k": [2, 3, 4], "trend_filter": [None, -0.05], "trend_gross": [0.0, 0.5],
             "w_move": [0.0, 0.3, 0.5, 0.7], "w_mom": [0.0, 0.2, 0.4]}


@dataclass(frozen=True)
class SearchConfig:
    n_start: int = 24
    n_screen: int = 10
    n_top: int = 3
    penalty_se: float = 1.0          # anti-overfit: subtract this many standard errors of the per-window score
    penalty_gap: float = 0.5         # and this share of the screen-to-confirm drop
    n_boot: int = 400
    boot_q: float = 0.10             # one-sided level, divided by n_top (held-out) or n_start (no held-out)
    min_boot_lo: float = 0.0         # the bootstrap lower bound of the paired gain must exceed this
    min_win_share: float = 0.6       # and at least this share of windows must improve on the deciding tier
    min_holdout: int = 3             # windows the screen never saw needed to use them as the evidence
    min_windows: int = 3             # never adopt on fewer confirmation windows
    seed: int = 0


def validate_meta(m):
    """Reject parameter sets the adaptation engine would misuse (nonsense combinations found by search are not gains)."""
    bad = []
    if m.get("half_life", 1) <= 0 or m.get("mem_half_life", 1) <= 0:
        bad.append("half-lives must be positive")
    if m.get("min_weeks", 1) < 1 or m.get("det_min_weeks", 1) < 1 or m.get("cooldown", 0) < 0:
        bad.append("week counts out of range")
    if not 0 <= m.get("det_max", 0) <= 1 or not 0 <= m.get("mem_shock_cut", 0) <= 1:
        bad.append("share parameters outside [0, 1]")
    if m.get("mem_bandwidth", 1) <= 0 or m.get("switch_z", 1) <= 0:
        bad.append("bandwidth and switch threshold must be positive")
    if m.get("cooldown", 0) > m.get("prior_weeks", 99):
        bad.append("cooldown longer than the prior")
    return bad


def sample_candidate(rng, cfg, meta, cfg_space=None, meta_space=None, tries=50):
    """Draw one random starting configuration around the incumbent; redraw (bounded) if meta is invalid."""
    cs, ms = CFG_SPACE if cfg_space is None else cfg_space, META_SPACE if meta_space is None else meta_space
    for _ in range(tries):
        c = {**cfg, **{k: v[rng.integers(len(v))] for k, v in cs.items()}}
        m = {**meta, **{k: v[rng.integers(len(v))] for k, v in ms.items()}}
        if not validate_meta(m):
            return c, m
    raise ValueError("could not draw a valid candidate: meta space is inconsistent")


def fingerprint(cfg, meta):
    return json.dumps([cfg, meta], sort_keys=True, default=str)


def changed_keys(a, b):
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


@dataclass
class Candidate:
    cfg: dict
    meta: dict
    screen: object = None            # TierScore on the screening windows
    confirm: object = None           # TierScore on all eligible windows
    adjusted: float = float("-inf")  # confirm soft score after the anti-overfit penalty
    boot_lo: float = float("nan")
    share: float = float("nan")
    verdict: str = ""
    is_incumbent: bool = False


@dataclass
class SearchResult:
    adopted: bool
    cfg: dict
    meta: dict
    reason: str
    incumbent: Candidate
    winner: Candidate
    candidates: list = field(default_factory=list)
    n_windows: int = 0
    n_screen: int = 0
    n_evals: int = 0

    def to_record(self):
        f = lambda c: None if c is None or c.confirm is None else {"key": list(c.confirm.key), "t1": c.confirm.t1,
                                                                   "risk": c.confirm.risk, "t3": c.confirm.t3}
        return {"event": "basis_search", "adopted": self.adopted, "reason": self.reason, "n_windows": self.n_windows,
                "n_screen": self.n_screen, "n_evals": self.n_evals, "incumbent": f(self.incumbent), "winner": f(self.winner),
                "changed": changed_keys(self.incumbent.cfg, self.cfg) + changed_keys(self.incumbent.meta, self.meta),
                "confirmed": [{"cfg": c.cfg, "adjusted": c.adjusted, "boot_lo": c.boot_lo, "verdict": c.verdict}
                              for c in self.candidates if c.confirm is not None]}


class BasisSearch:
    def __init__(self, evaluate, windows, cfg, meta, config=None, cfg_space=None, meta_space=None, evaluate_batch=None):
        """evaluate_batch(list of (window, cfg, meta)) -> rows, optional: lets a caller run one candidate's windows in parallel."""
        self.evaluate, self.windows, self.batch = evaluate, list(windows), evaluate_batch
        self.cfg, self.meta = dict(cfg), dict(meta)
        self.sc = config or SearchConfig()
        self.cfg_space, self.meta_space = cfg_space, meta_space
        self._cache, self.n_evals = {}, 0

    def _row(self, w, c, m):
        key = (w["id"], fingerprint(c, m))
        if key not in self._cache:
            self._cache[key] = self.evaluate(w, c, m)
            self.n_evals += 1
        return self._cache[key]

    def _rows(self, ws, c, m):
        if self.batch is not None:
            fp = fingerprint(c, m)
            todo = [w for w in ws if (w["id"], fp) not in self._cache]
            if todo:
                for w, r in zip(todo, self.batch([(w, c, m) for w in todo])):
                    self._cache[(w["id"], fp)] = r
                self.n_evals += len(todo)
            return [self._cache[(w["id"], fp)] for w in ws]
        return [self._row(w, c, m) for w in ws]

    def run(self, as_of=None):
        sc = self.sc
        rng = np.random.default_rng(sc.seed)
        wins = [w for w in self.windows if as_of is None or w["end"] <= as_of]      # no window past as_of is ever touched
        inc = Candidate(self.cfg, self.meta, is_incumbent=True)
        if len(wins) < sc.min_windows:
            return SearchResult(False, self.cfg, self.meta, f"only {len(wins)} windows (< {sc.min_windows}): keep basis",
                                inc, inc, [inc], len(wins), 0, 0)
        screen = [wins[i] for i in sorted(rng.choice(len(wins), size=min(sc.n_screen, len(wins)), replace=False))]
        inc.screen = O.evaluate(self._rows(screen, self.cfg, self.meta))
        seen, cands = {fingerprint(self.cfg, self.meta)}, []
        for _ in range(sc.n_start):
            c, m = sample_candidate(rng, self.cfg, self.meta, self.cfg_space, self.meta_space)
            fp = fingerprint(c, m)
            if fp in seen:
                continue
            seen.add(fp)
            cand = Candidate(c, m)
            cand.screen = O.evaluate(self._rows(screen, c, m))
            cands.append(cand)
        cands.sort(key=lambda x: (x.screen.key, x.screen.soft), reverse=True)
        finalists = [inc] + cands[:sc.n_top]
        for c in cands[sc.n_top:]:
            c.verdict = "dropped at screening"
        inc_rows = self._rows(wins, self.cfg, self.meta)
        inc.confirm = O.evaluate(inc_rows)
        screen_ids = {w["id"] for w in screen}
        hold_idx = np.array([i for i, w in enumerate(wins) if w["id"] not in screen_ids], int)
        inc_w = O.per_window_soft(inc_rows)
        inc.adjusted = self._adjust(inc, inc_w)
        best = None
        for c in finalists[1:]:
            rows = self._rows(wins, c.cfg, c.meta)
            c.confirm = O.evaluate(rows)
            w = O.per_window_soft(rows)
            c.adjusted = self._adjust(c, w)
            tier, diffs = O.decisive_diffs(inc_rows, rows)
            # The screen picked this candidate BECAUSE it did well on the screening windows, so those windows cannot
            # testify for it. Evidence = windows the screen never saw (Bonferroni over the finalists); with too few of
            # them, fall back to all windows but correct for the whole pool that was screened.
            if hold_idx.size >= sc.min_holdout:
                diffs, q = diffs[hold_idx], sc.boot_q / max(1, sc.n_top)
            else:
                q = sc.boot_q / max(1, sc.n_start)
            c.boot_lo = O.paired_bootstrap(diffs, sc.n_boot, q, seed=sc.seed)
            c.share = float((diffs > 0).mean()) if len(diffs) else 0.0
            ok, why = O.firewall(inc.confirm, c.confirm, inc_rows, rows)      # refuses leverage-only / cash-hiding gains
            if not ok:
                c.verdict = f"firewall: {why}"
            elif c.share < sc.min_win_share:
                c.verdict = f"firewall ok but only {c.share:.0%} of windows improved on the deciding tier"
            elif c.boot_lo <= sc.min_boot_lo:
                c.verdict = f"firewall ok but gain not reliable across windows (bootstrap lower bound {c.boot_lo:+.3f})"
            elif c.adjusted <= inc.adjusted:
                c.verdict = "gain vanishes after the anti-overfit penalty"
            else:
                c.verdict = f"eligible: {why}"
                if best is None or c.adjusted > best.adjusted:
                    best = c
        if best is None:
            return SearchResult(False, self.cfg, self.meta, "no confirmed candidate beat the incumbent", inc, inc,
                                [inc] + cands, len(wins), len(screen), self.n_evals)
        best.verdict = "ADOPTED: " + best.verdict
        return SearchResult(True, best.cfg, best.meta, best.verdict, inc, best, [inc] + cands, len(wins), len(screen), self.n_evals)

    def _adjust(self, cand, per_window):
        se = float(per_window.std(ddof=1) / np.sqrt(len(per_window))) if len(per_window) > 1 else 0.0
        drop = max(0.0, cand.screen.soft - cand.confirm.soft) if cand.screen is not None else 0.0
        return cand.confirm.soft - self.sc.penalty_se * se - self.sc.penalty_gap * drop


# ------------------------------------------------------------------ version history and rollback
class HistoryError(RuntimeError):
    pass


class BasisHistory:
    """Every adopted basis (cfg + meta) as an append-only, hash-chained list, so the loop can say which basis produced
    which windows and can go back. Nothing is deleted: a rollback appends a new version that re-installs an older one.
    `verify()` recomputes the chain and fails on any edited, removed or reordered entry. The file is written atomically."""

    def __init__(self, path=None, cfg=None, meta=None):
        self.path = None if path is None else Path(path)
        self.entries = []
        if self.path is not None and self.path.exists():
            self.entries = json.loads(self.path.read_text())
            if not self.verify():
                raise HistoryError(f"{self.path}: history chain is broken")
        elif cfg is not None:
            self._append("initial", cfg, meta or {}, "starting basis", None, None)

    @staticmethod
    def _hash(prev, body):
        return hashlib.sha256((prev + json.dumps(body, sort_keys=True, default=str)).encode()).hexdigest()[:16]

    def _append(self, kind, cfg, meta, reason, parent, score):
        body = {"kind": kind, "cfg": cfg, "meta": meta, "reason": reason, "parent": parent, "score": score}
        prev = self.entries[-1]["hash"] if self.entries else ""
        self.entries.append({"version": len(self.entries) + 1, **body, "hash": self._hash(prev, body)})
        self._save()

    def _save(self):
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.entries, indent=1, default=str))
        os.replace(tmp, self.path)

    @property
    def current(self):
        if not self.entries:
            raise HistoryError("no basis recorded yet")
        return self.entries[-1]

    def adopt(self, result):
        """Record a SearchResult's basis as the next version. Refuses a result that was not an adoption."""
        if not result.adopted:
            raise HistoryError("cannot record a search that adopted nothing")
        c = result.winner.confirm
        self._append("adopted", result.cfg, result.meta, result.reason, self.current["version"],
                     {"key": list(c.key), "t1": c.t1, "risk": c.risk})
        return self.current

    def get(self, version):
        for e in self.entries:
            if e["version"] == version:
                return e
        raise HistoryError(f"no basis version {version}")

    def rollback(self, to_version=None, why=""):
        """Re-install an earlier basis (default: the one before the current). Recorded as a new version."""
        cur = self.current
        if to_version is None:
            to_version = cur["parent"]
        if to_version is None or to_version == cur["version"]:
            raise HistoryError("nothing to roll back to")
        old = self.get(to_version)
        self._append("rollback", old["cfg"], old["meta"], f"rollback to v{to_version}: {why}", cur["version"], old["score"])
        return self.current

    def verify(self):
        prev = ""
        for i, e in enumerate(self.entries):
            body = {k: e[k] for k in ("kind", "cfg", "meta", "reason", "parent", "score")}
            if e.get("version") != i + 1 or e.get("hash") != self._hash(prev, body):
                return False
            prev = e["hash"]
        return True

    def lineage(self, version=None):
        """Versions from the given one back to the start, following parents (a rollback's parent is the version it left)."""
        e, out = (self.current if version is None else self.get(version)), []
        while e is not None:
            out.append(e["version"])
            e = None if e["parent"] is None else self.get(e["parent"])
        return out


def review_current_basis(prev_rows, cur_rows, n_boot=400, q=0.10, seed=0):
    """Should the current basis be rolled back? Compares fresh window rows (paired) produced under the previous and the
    current basis. Rolls back only on evidence the newer one is WORSE: the firewall prefers the older one AND the paired
    deciding-tier bootstrap upper tail says the loss is not one unlucky window. Returns (rollback, reason)."""
    if len(prev_rows) != len(cur_rows) or len(prev_rows) < 3:
        return False, "need at least 3 paired windows: keep the current basis"
    a, b = O.evaluate(prev_rows), O.evaluate(cur_rows)
    ok, why = O.firewall(b, a, cur_rows, prev_rows)          # would the OLD basis replace the current one?
    if not ok:
        return False, f"no case for rollback ({why})"
    _, d = O.decisive_diffs(cur_rows, prev_rows)
    lo = O.paired_bootstrap(d, n_boot, q, seed)
    if lo <= 0:
        return False, f"old basis looks better but not reliably (bootstrap lower bound {lo:+.3f})"
    return True, f"previous basis is better ({why}; bootstrap lower bound {lo:+.3f})"
