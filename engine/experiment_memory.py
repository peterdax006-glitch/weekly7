"""Bible Phase 30 (experiment memory), the search-space half: a normalised index of WHAT WAS TRIED.

engine.registry.ExperimentMemory records the twelve answers per experiment. This module answers the question asked
before launching one: has this configuration, or something very close to it, already been run, and how did it go?

* configs live in a declared `Space` (numeric ranges and categorical choices); distance is the mean normalised
  difference over the union of parameters (a parameter one side lacks counts as a full difference);
* `check(cfg)` flags exact repeats and near-duplicates and says whether a near neighbour already FAILED;
* `untried_neighbours(cfg)` proposes one-parameter moves not already covered by a prior attempt, best-first;
* `negative_results()` is the table of rejected / failed attempts with the reason, worst first.
The index is an append-only jsonl so it survives restarts and is never edited."""
import json
import math
from pathlib import Path

from .registry import fingerprint


class Space:
    """spec: {param: (lo, hi, step)} for numeric, {param: [choices]} for categorical. `step` is also the grid on which
    two values are treated as the same (so 0.9 and 0.9004 do not count as different experiments)."""

    def __init__(self, spec):
        self.spec = {}
        for k, v in spec.items():
            if isinstance(v, tuple) and len(v) == 3 and v[0] < v[1] and v[2] > 0:
                self.spec[k] = ("num", float(v[0]), float(v[1]), float(v[2]))
            elif isinstance(v, (list, tuple)) and len(v) >= 2:
                self.spec[k] = ("cat", list(v))
            else:
                raise ValueError(f"bad space spec for {k!r}: {v!r}")

    def snap(self, cfg):
        """Canonical config: numerics snapped to the step grid, unknown params kept as-is (they still count)."""
        out = {}
        for k, v in cfg.items():
            s = self.spec.get(k)
            if s and s[0] == "num" and isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                out[k] = round(round((v - s[1]) / s[3]) * s[3] + s[1], 10)
            else:
                out[k] = v
        return out

    def key(self, cfg):
        return fingerprint(self.snap(cfg))

    def distance(self, a, b):
        keys = set(a) | set(b)
        if not keys:
            return 0.0
        tot = 0.0
        for k in keys:
            if k not in a or k not in b:
                tot += 1.0
                continue
            s = self.spec.get(k)
            if s and s[0] == "num" and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (a[k], b[k])):
                tot += min(1.0, abs(a[k] - b[k]) / (s[2] - s[1]))
            else:
                tot += 0.0 if a[k] == b[k] else 1.0
        return tot / len(keys)


class TriedIndex:
    def __init__(self, path, space):
        self.path, self.space = Path(path), space
        self.rows = []
        if self.path.exists():
            self.rows = [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def add(self, experiment_id, cfg, outcome, score=None, reason=None, now=None):
        """outcome: adopt / reject / continue_testing. score: higher is better (None when unmeasured).
        A reject without a reason is refused: an unexplained negative result cannot stop anyone repeating it."""
        if outcome not in ("adopt", "reject", "continue_testing"):
            raise ValueError(f"unknown outcome {outcome!r}")
        if outcome == "reject" and not reason:
            raise ValueError("a rejected attempt needs a reason")
        if any(r["experiment_id"] == experiment_id for r in self.rows):
            raise ValueError(f"{experiment_id} already indexed")
        if score is not None and not math.isfinite(score):
            raise ValueError("score must be finite or None")
        row = {"experiment_id": experiment_id, "key": self.space.key(cfg), "cfg": self.space.snap(cfg), "outcome": outcome,
               "score": score, "reason": reason, "t": None if now is None else str(now)}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        self.rows.append(row)
        return row

    def import_registry(self, registry, cfg_field="config", score_metric="metrics.score"):
        """Index registry records that carry a config dict, an outcome, and (for rejects) a reason. Returns counts."""
        from .registry import _dig, _get
        added = skipped = 0
        for r in registry.records:
            cfg, oc = r.get(cfg_field), str(_get(r, "outcome") or "").lower()
            eid = r.get("experiment_id")
            if not isinstance(cfg, dict) or not eid or oc not in ("adopt", "reject", "continue_testing") \
                    or any(x["experiment_id"] == eid for x in self.rows) or (oc == "reject" and not _get(r, "reason")):
                skipped += 1
                continue
            sc = _dig(r, score_metric)
            self.add(eid, cfg, oc, sc if isinstance(sc, (int, float)) and math.isfinite(sc) else None, _get(r, "reason"))
            added += 1
        return {"added": added, "skipped": skipped}

    def nearest(self, cfg, k=5):
        snapped = self.space.snap(cfg)
        ds = sorted(((self.space.distance(snapped, r["cfg"]), r) for r in self.rows), key=lambda x: x[0])
        return ds[:k]

    def check(self, cfg, near=0.08):
        """Pre-launch check. verdict: 'repeat' (identical after snapping), 'near_duplicate' (within `near` of a prior
        attempt), 'known_failure_nearby' (a near attempt was rejected and none adopted), or 'novel'."""
        key = self.space.key(cfg)
        exact = [r for r in self.rows if r["key"] == key]
        close = [(d, r) for d, r in self.nearest(cfg, 50) if d <= near and r["key"] != key]
        bad = [r for r in exact + [r for _, r in close] if r["outcome"] == "reject"]
        good = [r for r in exact + [r for _, r in close] if r["outcome"] == "adopt"]
        if exact:
            verdict = "repeat"
        elif close:
            verdict = "near_duplicate"
        else:
            verdict = "novel"
        if bad and not good and verdict != "novel":
            verdict = "repeat" if exact else "known_failure_nearby"
        return {"verdict": verdict, "block": verdict in ("repeat", "known_failure_nearby") and bool(bad) and not good,
                "exact": [r["experiment_id"] for r in exact], "close": [(round(d, 4), r["experiment_id"]) for d, r in close],
                "reasons": [r["reason"] for r in bad]}

    def untried_neighbours(self, cfg, n=10, near=0.08):
        """One-parameter moves from cfg (numeric +-1 and +-2 steps; every other categorical choice) that neither repeat nor
        sit near a prior attempt. Ranked by the best score among tried attempts within 3x `near` of the candidate
        (unknown = middle), so the search leans toward promising regions rather than uniformly."""
        base = self.space.snap(cfg)
        cands = {}
        for k, s in self.space.spec.items():
            if k not in base:
                continue
            opts = []
            if s[0] == "num":
                for m in (-2, -1, 1, 2):
                    v = round(base[k] + m * s[3], 10)
                    if s[1] <= v <= s[2]:
                        opts.append(v)
            else:
                opts = [c for c in s[1] if c != base[k]]
            for v in opts:
                c = {**base, k: v}
                cands[self.space.key(c)] = (k, c)
        out = []
        for key, (k, c) in cands.items():
            if self.check(c, near)["verdict"] != "novel":
                continue
            scores = [r["score"] for d, r in self.nearest(c, 50) if d <= 3 * near and r["score"] is not None]
            out.append({"param": k, "cfg": c, "prior": max(scores) if scores else None})
        known = [o["prior"] for o in out if o["prior"] is not None]
        mid = sum(known) / len(known) if known else 0.0
        out.sort(key=lambda o: -(o["prior"] if o["prior"] is not None else mid))
        return out[:n]

    def negative_results(self):
        """Rejected attempts, lowest score first (unscored last), with the reason each was rejected."""
        rows = [{"experiment_id": r["experiment_id"], "cfg": r["cfg"], "score": r["score"], "reason": r["reason"]}
                for r in self.rows if r["outcome"] == "reject"]
        return sorted(rows, key=lambda r: (r["score"] is None, r["score"] if r["score"] is not None else 0))

    def coverage(self):
        """Share of each declared parameter's range/choices that has been touched: where nobody has looked yet."""
        out = {}
        for k, s in self.space.spec.items():
            vals = [r["cfg"][k] for r in self.rows if k in r["cfg"]]
            if s[0] == "num":
                grid = int(round((s[2] - s[1]) / s[3])) + 1
                out[k] = min(1.0, len({round(v, 8) for v in vals}) / grid) if grid else 0.0
            else:
                out[k] = len({v for v in vals if v in s[1]}) / len(s[1])
        return out
