"""I03 property tests (contract sections 18, 46, 62; checklist I03) - IMPLEMENTED, NOT VALIDATED.

`hypothesis` is not installed in the venv, so properties are seeded randomized loops: case i of a property draws from
np.random.default_rng([seed, i]), which makes every failure replayable (`replay`). A property is a function
prop(rng, stats, impl) -> list of violations (empty = holds); `impl` is the implementation under test (None = the real one).
Every property is registered with at least one MUTANT: a deliberately broken `impl` that the same property must catch, so no property
is a check that cannot fail. `stats` counts witnesses so a property that never exercised its interesting branch is detected."""
import dataclasses
import datetime as dt
import hashlib
import json
import math
import time
from collections import Counter
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest

from engine.learning import belief as B
from engine.learning import calibration as CAL
from engine.learning import credit as CR
from engine.learning import curator as CU
from engine.learning import epistemic as EP
from engine.learning import firewalls as FW
from engine.learning import future_firewall as FF
from engine.learning import hierarchy as H
from engine.learning import knowledge as KN
from engine.learning import planted_world as PW
from engine.learning import same_year as SY
from engine.learning import trader_view as TV
from engine.learning import transfer_score as TS
from engine.learning.core import (Confidence, DecisionEffect, Epistemic, FirewallBreach, Lifecycle, Promotion, Provenance, TemporalClass,
                                  as_date, require_past, stable_hash)

# ------------------------------------------------------------------------------------------------ the mini framework


@dataclasses.dataclass(frozen=True)
class Failure:
    case: int
    detail: str


@dataclasses.dataclass(frozen=True)
class PropertyResult:
    name: str
    n_cases: int
    seed: int
    failures: tuple[Failure, ...]
    stats: Counter

    @property
    def ok(self) -> bool:
        return not self.failures

    def describe(self) -> str:
        if self.ok:
            return f"{self.name}: {self.n_cases} cases hold"
        f = self.failures[0]
        return f"{self.name}: {len(self.failures)}/{self.n_cases} cases fail; first is case {f.case} (replay(seed={self.seed}, case={f.case})): {f.detail}"


def case_rng(seed: int, case: int) -> np.random.Generator:
    return np.random.default_rng([seed, case])


def run_property(name: str, prop: Callable, n_cases: int, seed: int = 0, impl: Any = None, stop_after: int = 3) -> PropertyResult:
    """Run `prop` on n_cases seeded draws. An exception inside a property is a failure, never a skip."""
    stats: Counter = Counter()
    fails: list[Failure] = []
    for i in range(n_cases):
        try:
            viol = prop(case_rng(seed, i), stats, impl)
        except Exception as e:                                        # noqa: BLE001 - a crashing property is a failing property
            viol = [f"raised {type(e).__name__}: {e}"]
        if viol:
            fails.append(Failure(i, "; ".join(viol[:3])))
            if len(fails) >= stop_after:
                break
    return PropertyResult(name, n_cases, seed, tuple(fails), stats)


def replay(prop: Callable, seed: int, case: int, impl: Any = None) -> list[str]:
    return prop(case_rng(seed, case), Counter(), impl)


# ------------------------------------------------------------------------------------------------ generators

WORDS = ("momentum", "calm", "reversal", "high", "low", "breadth", "spread", "carry", "gap", "trend")


def rounded(rng, lo, hi, nd=6) -> float:
    return round(float(rng.uniform(lo, hi)), nd)


def gen_atom(rng):
    kind = int(rng.integers(0, 9))
    if kind == 0:
        return int(rng.integers(-10 ** 6, 10 ** 6))
    if kind == 1:
        return rounded(rng, -50, 50, 6)
    if kind == 2:
        return "".join(rng.choice(list("abcdefghij"), size=int(rng.integers(0, 8))))
    if kind == 3:
        return bool(rng.integers(0, 2))
    if kind == 4:
        return None
    if kind == 5:
        return Epistemic(list(Epistemic)[int(rng.integers(0, len(Epistemic)))].value)
    if kind == 6:
        return dt.date(2000, 1, 1) + dt.timedelta(days=int(rng.integers(0, 9000)))
    if kind == 7:
        return float(rng.choice([math.nan, math.inf, -math.inf]))
    return str(rng.choice(WORDS))


def gen_tree(rng, depth=0):
    kind = int(rng.integers(0, 5)) if depth < 3 else 0
    if kind == 1:
        return {f"k{int(rng.integers(0, 40))}_{j}": gen_tree(rng, depth + 1) for j in range(int(rng.integers(1, 6)))}
    if kind == 2:
        return [gen_tree(rng, depth + 1) for _ in range(int(rng.integers(0, 5)))]
    if kind == 3:
        return tuple(gen_tree(rng, depth + 1) for _ in range(int(rng.integers(0, 4))))
    if kind == 4:
        return {"".join(rng.choice(list("xyz"), size=3)) + str(j) for j in range(int(rng.integers(0, 5)))}
    return gen_atom(rng)


def reorder(tree, rng):
    """Same content, different insertion order / container type: dicts rebuilt in shuffled key order, tuples become lists."""
    if isinstance(tree, dict):
        keys = list(tree)
        return {keys[i]: reorder(tree[keys[i]], rng) for i in rng.permutation(len(keys))}
    if isinstance(tree, (list, tuple)):
        return [reorder(v, rng) for v in tree]
    return tree


def numpyfy(tree):
    if isinstance(tree, dict):
        return {k: numpyfy(v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(numpyfy(v) for v in tree)
    if isinstance(tree, bool) or tree is None:
        return tree
    if isinstance(tree, int):
        return np.int64(tree)
    if isinstance(tree, float):
        return np.float64(tree)
    return tree


def perturb_leaf(tree, rng):
    """Copy of `tree` with one int or str leaf changed; None if the tree has none."""
    paths = []

    def walk(x, path):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, path + [k])
        elif isinstance(x, (list, tuple)):
            for i, v in enumerate(x):
                walk(v, path + [i])
        elif isinstance(x, int) and not isinstance(x, bool) or isinstance(x, str) and not isinstance(x, Epistemic):
            paths.append(path)
    walk(tree, [])
    if not paths:
        return None
    target = paths[int(rng.integers(0, len(paths)))]

    def rebuild(x, path):
        if not path:
            return x + 1 if isinstance(x, int) else x + "!"
        k, rest = path[0], path[1:]
        if isinstance(x, dict):
            return {kk: rebuild(v, rest) if kk == k else v for kk, v in x.items()}
        out = [rebuild(v, rest) if i == k else v for i, v in enumerate(x)]
        return tuple(out) if isinstance(x, tuple) else out
    return rebuild(tree, target)


# ------------------------------------------------------------------------------------------------ 1. stable_hash

def prop_stable_hash(rng, stats, impl=None):
    h = impl or stable_hash
    tree = {"root": gen_tree(rng), "n": int(rng.integers(0, 10))}
    base, bad = h(tree), []
    if not (isinstance(base, str) and len(base) == 16 and all(c in "0123456789abcdef" for c in base)):
        bad.append(f"format: {base!r} is not 16 lowercase hex digits")
    if h(reorder(tree, rng)) != base:
        bad.append("order/container invariance: shuffled keys or list-for-tuple changed the hash")
    if h(numpyfy(tree)) != base:
        bad.append("numpy-type invariance: np.int64/np.float64 hashed differently from python numbers")
    if h(tree) != base:
        bad.append("not deterministic within one process")
    changed = perturb_leaf(tree, rng)
    if changed is not None:
        stats["perturbed"] += 1
        if h(changed) == base:
            bad.append("sensitivity: changing a leaf did not change the hash")
    return bad


def order_sensitive_hash(obj, n=16):
    """MUTANT: plain json.dumps with str() fallback: depends on key order and stringifies numpy scalars."""
    return hashlib.sha256(json.dumps(obj, default=str).encode()).hexdigest()[:n]


# ------------------------------------------------------------------------------------------------ 2. KnowledgeObject round trip

def gen_knowledge(rng) -> KN.KnowledgeObject:
    n_ev = int(rng.integers(30, 500))
    champion = bool(rng.integers(0, 2))
    prov = Provenance(created_real="2026-09-29T00:00:00+00:00", learned_at="2020-01-01", code_hash="abc123", data_hash="d1", config_hash="c1",
                      experiment_id="E1", seed=int(rng.integers(0, 99)), outcomes_seen_through="2020-01-01")
    tags = tuple(sorted(str(t) for t in rng.choice(list(WORDS), size=int(rng.integers(0, 4)), replace=False)))
    kw = dict(knowledge_id="K-" + "".join(rng.choice(list("abcdefgh"), size=6)), created_at="2020-01-02", provenance=prov,
              observation=" ".join(rng.choice(list(WORDS), size=3)), hypothesis="h " + str(rng.choice(WORDS)),
              contexts=KN.context_from_text("volatility: vix >= 20"),
              effect=KN.Effect(int(rng.choice([-1, 1])), rounded(rng, 0.0005, 0.05), rounded(rng, 0.0001, 0.01)),
              confidence=Confidence(*[rounded(rng, 0, 1, 4) for _ in range(6)]),
              evidence=KN.Evidence(n_ev, float(rng.integers(1, n_ev + 1)), rounded(rng, 0, 1, 3), rounded(rng, 0, 1, 3), "2020-12-01"),
              temporal_class=list(TemporalClass)[int(rng.integers(0, len(TemporalClass)))], mechanism_tags=tags)
    if champion:
        kw.update(decision_effect=(DecisionEffect.RANKING,), epistemic=Epistemic.SUPPORTED, lifecycle=Lifecycle.ACTIVE, promotion=Promotion.CHAMPION)
    else:
        kw.update(decision_effect=(DecisionEffect.NONE,), epistemic=Epistemic.HYPOTHESIS, lifecycle=Lifecycle.BIRTH, promotion=Promotion.RESEARCH)
    return KN.KnowledgeObject(**kw)


def prop_knowledge_roundtrip(rng, stats, impl=None):
    decode = impl or KN.KnowledgeObject.from_json
    obj = gen_knowledge(rng)
    problems = obj.validate()
    if problems:
        return [f"generator produced an invalid object (vacuous property): {problems[:2]}"]
    text = obj.to_json()
    back = decode(text)
    bad = []
    if back != obj:
        diff = [f.name for f in dataclasses.fields(obj) if getattr(back, f.name) != getattr(obj, f.name)]
        bad.append(f"round trip changed fields {diff}")
    if back.to_json() != text:
        bad.append("re-serialising the decoded object gave different JSON")
    if back.record_hash() != obj.record_hash():
        bad.append("record_hash changed across the round trip")
    stats["tags"] += len(obj.mechanism_tags)
    return bad


def lossy_decode(text):
    """MUTANT: a decoder that silently drops the mechanism tags."""
    return dataclasses.replace(KN.KnowledgeObject.from_json(text), mechanism_tags=())


# ------------------------------------------------------------------------------------------------ 3. epistemic transitions

def gen_evidence(rng) -> EP.EvidenceSummary:
    n = int(rng.integers(0, 400))
    ctxs = ("bull", "bear", "calm", "stress")
    works = tuple(sorted(rng.choice(ctxs, size=int(rng.integers(0, 3)), replace=False)))
    fails = tuple(c for c in ctxs if c not in works and rng.random() < 0.3)
    opt = lambda lo, hi: None if rng.random() < 0.2 else rounded(rng, lo, hi, 3)
    return EP.EvidenceSummary(n_events=n, n_eff=float(rng.uniform(0, n)), t_discovery=opt(-1, 6), t_confirm=opt(-2, 5), p_real=opt(0, 1),
                              contradiction_rate=opt(0, 1), recent_ratio=opt(-1, 1.5), reversal_t=opt(0, 4), works_in=works, fails_in=fails,
                              has_hypothesis=bool(rng.integers(0, 2)), has_prediction=bool(rng.integers(0, 2)),
                              gate_reason="g" if rng.random() < 0.5 else "", retire_reason="r" if rng.random() < 0.5 else "")


def real_apply(profile, kind, key, new, ev, now, revive):
    return profile.apply(kind, key, new, ev, now, reason="prop", revive=revive)


def forced_apply(profile, kind, key, new, ev, now, revive):
    """MUTANT: writes the transition (with a valid hash chain) without asking whether it is legal."""
    old = profile.state(kind, key)
    rec = EP.TransitionRecord(kind, key, old, new, str(as_date(now)), "forced", stable_hash(ev), revive,
                              profile.history[-1].hash if profile.history else "")
    rec = dataclasses.replace(rec, hash=rec.body_hash())
    kept = tuple(s for s in profile.scopes if not (s.kind == kind and s.key == key))
    scopes = tuple(sorted(kept + (EP.ScopedState(kind, key, new, rec.at, "forced"),), key=lambda s: (s.kind, s.key)))
    return EP.EpistemicProfile(scopes, profile.history + (rec,))


def illegal_moves(profile) -> list[str]:
    """Every recorded transition that the transition table forbids, plus state/history disagreements."""
    bad = []
    for r in profile.history:
        first_look = r.frm is None and r.kind != EP.HISTORICAL
        if not first_look and not EP.legal(r.frm, r.to):
            bad.append(f"{r.kind}/{r.key or '-'}: {r.frm.value if r.frm else 'START'} -> {r.to.value} is not in the table")
        if r.frm == Epistemic.RETIRED and not r.revive:
            bad.append(f"{r.kind}: left RETIRED without revive")
    last = {}
    for r in profile.history:
        last[(r.kind, r.key)] = r.to
    for s in profile.scopes:
        if last.get((s.kind, s.key)) != s.state:
            bad.append(f"{s.kind}/{s.key or '-'}: state {s.state.value} is not the last recorded transition")
    return bad


def prop_epistemic_walk(rng, stats, impl=None):
    apply = impl or real_apply
    profile, day = EP.EpistemicProfile(), dt.date(2020, 1, 1)
    for _ in range(14):
        kind = str(rng.choice(EP.SCOPE_KINDS))
        key = "" if kind != EP.CONTEXT else str(rng.choice(["regime=bull", "regime=bear"]))
        ev = gen_evidence(rng)
        new = EP.derive_state(ev, kind) if rng.random() < 0.6 else list(Epistemic)[int(rng.integers(0, len(Epistemic)))]
        day += dt.timedelta(days=int(rng.integers(1, 20)))
        try:
            profile = apply(profile, kind, key, new, ev, day, bool(rng.random() < 0.3))
            stats["accepted"] += 1
        except EP.EpistemicError:
            stats["refused"] += 1
    return illegal_moves(profile) + [f"verify: {e}" for e in profile.verify_history() + profile.coherence()]


# ------------------------------------------------------------------------------------------------ 4. belief posterior

def gen_evidence_list(rng, as_of="2021-01-01"):
    k = int(rng.integers(1, 7))
    kinds = list(B.EvidenceKind)
    out = []
    for i in range(k):
        out.append(B.Evidence("S", (as_date(as_of) - dt.timedelta(days=int(rng.integers(1, 700)))).isoformat(), rounded(rng, -0.1, 0.1),
                              rounded(rng, 0.005, 0.06), int(rng.integers(5, 300)), kinds[int(rng.integers(0, len(kinds)))],
                              rounded(rng, 0.3, 1, 3), rounded(rng, 0.5, 1, 3), int(rng.integers(1, 20)), f"src{i}"))
    return out


def prop_belief_posterior(rng, stats, impl=None):
    pool = impl or B.pool
    prior_mean, prior_sd = rounded(rng, -0.03, 0.03), rounded(rng, 0.005, 0.06)
    ev = gen_evidence_list(rng)
    r = pool(prior_mean, prior_sd, ev, "2021-01-01")
    xs = [e.estimate for e in ev]
    lo, hi = min(xs + [prior_mean]) - 1e-12, max(xs + [prior_mean]) + 1e-12
    bad = []
    if not lo <= r.mean <= hi:
        bad.append(f"overshoot: posterior {r.mean:.5f} outside the range [{lo:.5f}, {hi:.5f}] of prior and evidence")
    if not 0 < r.sd <= prior_sd + 1e-12:
        bad.append(f"evidence widened the posterior: sd {r.sd:.5f} vs prior sd {prior_sd:.5f}")
    if len(ev) == 1:                                                 # with one record the posterior sits between prior and record
        stats["single"] += 1
        d_post, d_ev = r.mean - prior_mean, ev[0].estimate - prior_mean
        if d_post * d_ev < -1e-15 or abs(d_post) > abs(d_ev) + 1e-12:
            bad.append(f"single record: moved {d_post:+.5f} against/beyond evidence {d_ev:+.5f}")
    return bad


def extrapolating_pool(prior_mean, prior_sd, evidence, as_of, cfg=B.DEFAULT_CFG):
    """MUTANT: extrapolates past the newest record instead of averaging with the prior."""
    real = B.pool(prior_mean, prior_sd, evidence, as_of, cfg)
    x = evidence[-1].estimate
    return dataclasses.replace(real, mean=x + 0.5 * (x - prior_mean))


def prop_beta_belief(rng, stats, impl=None):
    upd = impl or (lambda b, s, t, q: b.updated(s, t, q))
    b = B.BetaBelief(float(rng.integers(1, 20)), float(rng.integers(1, 20)))
    trials = float(rng.integers(1, 60))
    succ = float(rng.integers(0, int(trials) + 1))
    new = upd(b, succ, trials, rounded(rng, 0.05, 1, 3))
    freq = succ / trials
    lo, hi = min(b.mean, freq) - 1e-12, max(b.mean, freq) + 1e-12
    bad = []
    if not lo <= new.mean <= hi:
        bad.append(f"beta mean {new.mean:.4f} left [{lo:.4f}, {hi:.4f}] (old mean {b.mean:.4f}, observed {freq:.4f})")
    if new.strength < b.strength - 1e-12:
        bad.append("evidence reduced the belief's strength")
    return bad


def double_counting_update(b, s, t, q):
    """MUTANT: counts successes twice."""
    return b.updated(min(2 * s, t), t, q)


# ------------------------------------------------------------------------------------------------ 5. hierarchy shrinkage

def gen_hierarchy(rng):
    cells = [{"ctx": {"market": m, "sector": s}, "effect": rounded(rng, -0.01, 0.01, 4), "p": 1.0 if m == "bull" or rng.random() < 0.5 else rounded(rng, 0.03, 0.12, 3)}
             for m in ("bull", "bear") for s in ("tech", "energy")]
    df = H.simulate_panel(int(rng.integers(0, 10 ** 6)), cells, n_days=int(rng.integers(60, 100)), noise=0.02)
    return H.KnowledgeHierarchy().fit(df, "2021-01-01")


def real_posterior(h, path):
    return h.posterior(path)


def raw_child_posterior(h, path):
    """MUTANT: lets every node claim its own raw mean, no shrinkage, no override gate."""
    po = h.posterior(path)
    return po if po is None else dataclasses.replace(po, qualifies=True, used_path=po.path, used_mean=po.raw_mean, used_var=po.raw_var)


def prop_hierarchy_shrinks(rng, stats, impl=None):
    get = impl or real_posterior
    h = gen_hierarchy(rng)
    bad = []
    for path in h.paths():
        po = get(h, path)
        if not path:
            continue
        parent = h.posterior(path[:-1])
        used = h.posterior(po.used_path)
        if not 0.0 <= po.weight_own <= 1.0 + 1e-12:
            bad.append(f"{path}: weight_own {po.weight_own}")
        if po.clusters < h.rule(len(path), "min_clusters"):                # tiny n: may never override, must inherit the parent exactly
            stats["tiny"] += 1
            if po.qualifies:
                bad.append(f"{path}: {po.clusters} independent dates yet it overrides its parent")
            if abs(po.used_mean - parent.used_mean) > 1e-12 or po.used_var < parent.used_var - 1e-15:
                bad.append(f"{path}: tiny cell reports mean {po.used_mean:.5f}/var {po.used_var:.3g} instead of parent's "
                           f"{parent.used_mean:.5f}/{parent.used_var:.3g}")
        elif not po.qualifies and len(po.used_path) >= len(path):
            bad.append(f"{path}: not qualified but still uses its own estimate")
        if used.used_var <= 0:
            bad.append(f"{po.used_path}: non-positive variance")
    est = h.estimate({"market": "bear", "sector": "tech"})
    if est.is_known() and est.p_sign_capped > est.p_sign + 1e-12:
        bad.append("sample-size cap raised sign confidence instead of lowering it")
    return bad


# ------------------------------------------------------------------------------------------------ 6. trader_view

DATE_FORMS = ("iso", "iso_slash", "iso_dot", "iso_compact_str", "iso_prefixed", "dmy", "month_name", "year_str", "year_in_text", "quarter",
              "fiscal", "era", "year_int", "year_float", "compact_int", "epoch_s", "epoch_ms", "date_obj", "datetime_obj", "timestamp",
              "np_datetime", "date_key", "year_key")


def make_date_form(rng, form):
    y, m, d = int(rng.integers(1950, 2036)), int(rng.integers(1, 13)), int(rng.integers(1, 29))
    if form == "iso":
        return f"{y}-{m:02d}-{d:02d}"
    if form == "iso_slash":
        return f"{y}/{m:02d}/{d:02d}"
    if form == "iso_dot":
        return f"{y}.{m:02d}.{d:02d}"
    if form == "iso_compact_str":
        return f"{y}{m:02d}{d:02d}"
    if form == "iso_prefixed":
        return f"AAPL_{y}-{m:02d}-{d:02d}"
    if form == "dmy":
        return f"{d:02d}/{m:02d}/{y}"
    if form == "month_name":
        return f"{('Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec'.split())[m - 1]} {d}, {y}"
    if form == "year_str":
        return str(y)
    if form == "year_in_text":
        return f"since {y}"
    if form == "quarter":
        return f"Q{int(rng.integers(1, 5))} {y}"
    if form == "fiscal":
        return f"FY{y}"
    if form == "era":
        return str(rng.choice(["lehman", "dotcom", "covid", "gfc", "brexit", "volmageddon"]))
    if form == "year_int":
        return y
    if form == "year_float":
        return y + round(float(rng.random()), 2)
    if form == "compact_int":
        return y * 10000 + m * 100 + d
    if form == "epoch_s":
        return int(rng.integers(10 ** 9, 4 * 10 ** 9))
    if form == "epoch_ms":
        return int(rng.integers(10 ** 12, 4 * 10 ** 12))
    if form == "date_obj":
        return dt.date(y, m, d)
    if form == "datetime_obj":
        return dt.datetime(y, m, d, 9, 30)
    if form == "timestamp":
        return pd.Timestamp(y, m, d)
    if form == "np_datetime":
        return np.datetime64(f"{y}-{m:02d}-{d:02d}")
    raise ValueError(form)


def clean_payload(rng, depth=0):
    """A random nested dict of safe relative features (no year-range numbers, no date-like text)."""
    out = {}
    for w in rng.choice(list(WORDS), size=int(rng.integers(1, 4)), replace=False):
        if depth < 2 and rng.random() < 0.4:
            out[f"rel_{w}"] = clean_payload(rng, depth + 1)
        else:
            out[f"rel_{w}"] = rounded(rng, -3, 3, 4)
    return out


def plant(rng, form):
    """A clean nested payload with one date-like thing hidden at a random place (as a value, a list item, or a key)."""
    payload = {"features": clean_payload(rng), "lean": 0.01, "note": [clean_payload(rng)]}
    if form in ("date_key", "year_key"):
        key = f"{int(rng.integers(1950, 2036))}-0{int(rng.integers(1, 10))}-1{int(rng.integers(0, 10))}" if form == "date_key" else str(rng.choice(["year", "date", "as_of", "timestamp"]))
        payload["features"][key] = 0.1
        return payload
    bad = make_date_form(rng, form)
    where = int(rng.integers(0, 3))
    if where == 0:
        payload["features"]["rel_x"] = bad
    elif where == 1:
        payload["note"].append(bad)
    else:
        payload["deep"] = {"a": [{"b": bad}]}
    return payload


def prop_trader_blind(rng, stats, impl=None):
    find = impl or TV.find_violations
    form = DATE_FORMS[int(rng.integers(0, len(DATE_FORMS)))]
    stats[form] += 1
    payload = plant(rng, form)
    bad = []
    if not find(payload):
        bad.append(f"{form}: a date-like value got through find_violations")
        return bad
    if impl is None:
        try:
            TV.assert_trader_safe(payload)
            bad.append(f"{form}: assert_trader_safe did not raise")
        except FirewallBreach:
            pass
        clean, rep = TV.scrub(payload)
        if TV.find_violations(clean):
            bad.append(f"{form}: scrub output still carries {TV.find_violations(clean)[:2]}")
        if rep.clean:
            bad.append(f"{form}: scrub report claims the payload was clean")
    return bad


def prop_trader_clean_not_flagged(rng, stats, impl=None):
    payload = {"features": clean_payload(rng), "lean": rounded(rng, -1, 1, 4), "horizon": int(rng.integers(1, 60))}
    v = (impl or TV.find_violations)(payload)
    return [f"false positive on a clean payload: {v[:2]}"] if v else []


def naive_iso_only(obj, *a, **k):
    """MUTANT: only looks for yyyy-mm-dd strings and datetime.date objects."""
    text = json.dumps(obj, default=lambda o: o.isoformat() if isinstance(o, (dt.date,)) else "obj")
    import re
    return ["date"] if re.search(r"(19|20)\d\d-\d\d-\d\d", text) else []


# ------------------------------------------------------------------------------------------------ 7. curator release

def build_curator(rng, cls=CU.Curator):
    cur = cls(None, code_hash="prop", created_real="2020-01-01T00:00:00")
    mems = []
    for j in range(int(rng.integers(0, 14))):
        filed = pd.Timestamp("2008-01-01") + pd.Timedelta(days=int(rng.integers(0, 3200)))
        matured = filed + pd.Timedelta(days=int(rng.integers(1, 500)))
        name = f"m{int(rng.integers(0, 6))}"
        it = TV.TraderMemoryItem.make(TV.opaque_token(name), "pattern", 1.0, {"rel_vix": rounded(rng, -1, 1, 3), "rel_r5": rounded(rng, -1, 1, 3)},
                                      rounded(rng, -0.5, 0.5, 3), 5)
        m = cur.file(it, filed, filed.year, {"m_vix": rounded(rng, 10, 40, 2), "m_spy_r5": rounded(rng, -0.08, 0.08, 4)}, matured_at=matured,
                     reliability=rounded(rng, 0.3, 0.9, 2))
        mems.append(m)
    return cur, mems


class LeakyCurator(CU.Curator):
    """MUTANT: scores memories as if it were a year later than the real clock."""
    def relevance(self, real_now, market_state):
        return super().relevance(as_date(real_now) + dt.timedelta(days=400), market_state)


def prop_curator_release(rng, stats, impl=None):
    cur, mems = build_curator(rng, impl or CU.Curator)
    now = dt.date(2009, 1, 1) + dt.timedelta(days=int(rng.integers(0, 3800)))
    state = {"m_vix": rounded(rng, 10, 40, 2), "m_spy_r5": rounded(rng, -0.08, 0.08, 4)}
    cur.day(now, state)
    rel = cur.release(now, state, k=8)
    by_id = {m.mem_id: m for m in mems}
    audit = cur.audit[-1]
    bad = []
    for row in audit.released:
        m = by_id[row["best"]]
        stats["released"] += 1
        if as_date(m.matured_at) >= now:
            bad.append(f"released memory whose outcome matured {m.matured_at}, not before real_now {now}")
    n_unmatured = sum(1 for m in mems if as_date(m.matured_at) >= now)
    stats["unmatured"] += n_unmatured
    if audit.excluded_unmatured != n_unmatured:
        bad.append(f"audit excluded {audit.excluded_unmatured} unmatured memories; {n_unmatured} really were")
    if len(rel.items) and abs(sum(i.weight for i in rel.items) - 1.0) > 1e-9:
        bad.append("release weights do not sum to 1")
    if TV.find_violations(rel.to_dict()) if hasattr(rel, "to_dict") else False:
        bad.append("release carries date-like content")
    try:
        cur.release(now - dt.timedelta(days=1), state)
        bad.append("release for a day before the current one should never be served from a later clock")
    except FirewallBreach:
        pass
    except Exception:                                               # noqa: BLE001 - an earlier day is allowed to be served; only a later one is not
        pass
    try:
        cur.release(now + dt.timedelta(days=1), state)
        bad.append("release for a day the clock has not reached was served")
    except FirewallBreach:
        pass
    return [b for b in bad if not b.startswith("release for a day before")]


# ------------------------------------------------------------------------------------------------ 8. firewalls fail closed

FIELD_LAYERS = {"items": {FW.LayerName.MEMORY, FW.LayerName.PROVENANCE}, "X": {FW.LayerName.DATA, FW.LayerName.TIME}, "horizon": {FW.LayerName.TIME},
                "identity_report": {FW.LayerName.IDENTITY}, "experiment": {FW.LayerName.EXPERIMENT}, "evaluation": {FW.LayerName.EVALUATION},
                "code": {FW.LayerName.CODE_VERSION}}


REFERENCE_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)      # seeds 11, 17, 41 trip the data layer's implausible-IC check (a chance draw in the reference panel)


class LenientLayer(FW.FirewallLayer):
    """MUTANT: wraps a real layer but treats 'not supplied' as fine."""
    def __init__(self, inner):
        self.inner, self.name = inner, inner.name

    def inspect(self, ctx):
        found, n = self.inner.inspect(ctx)
        return [f for f in found if f.check != "input-missing"], n


def lenient_gate():
    return FW.LearningFirewallGate([LenientLayer(layer) for layer in FW.default_layers()])


def prop_firewall_fails_closed(rng, stats, impl=None):
    gate = impl() if impl else FW.LearningFirewallGate()
    ref = FW.reference_context(seed=int(rng.choice(REFERENCE_SEEDS)))
    bad = []
    if impl is None and not gate.evaluate(ref).passed:
        bad.append("the untouched reference context should pass (gate over-strict)")
    fields = list(FIELD_LAYERS)
    drop = [fields[i] for i in rng.permutation(len(fields))[: int(rng.integers(1, 4))]]
    ctx = dataclasses.replace(ref, **{f: None for f in drop})
    verdict = gate.evaluate(ctx)
    stats["dropped"] += len(drop)
    if verdict.passed:
        bad.append(f"gate PASSED with {drop} missing")
    expected = set().union(*(FIELD_LAYERS[f] for f in drop))
    silent = expected - set(verdict.failed_layers)
    if silent:
        bad.append(f"missing {drop} but layers {sorted(l.value for l in silent)} did not fail")
    return bad


def prop_inputs_fail_closed(rng, stats, impl=None):
    check = impl or FF.check_timestamp
    kinds = [k.value for k in FF.InputKind]
    variant = int(rng.integers(0, 4))
    if variant == 0:
        inp = FF.LearningInput("no_stamp", str(rng.choice(kinds)))                                  # no timestamp, no rows
    elif variant == 1:
        inp = FF.LearningInput("bad_kind", "".join(rng.choice(list("qwxz"), size=5)), "2019-01-01")
    elif variant == 2:
        inp = FF.LearningInput("future", str(rng.choice(["PRICE", "EVENT", "FEATURE"])), "2021-06-01")
    else:
        inp = FF.LearningInput("label_today", "LABEL", "2020-01-01")                                # a label that matures ON now is not known
    res = check([inp], "2020-01-01")
    stats[f"variant{variant}"] += 1
    return [] if res.findings else [f"variant {variant} produced no finding: the check let a malformed input through"]


def prop_require_past(rng, stats, impl=None):
    fn = impl or require_past
    now = dt.date(2020, 1, 1) + dt.timedelta(days=int(rng.integers(0, 2000)))
    offset = int(rng.integers(-30, 31))
    when = now + dt.timedelta(days=offset)
    raised = False
    try:
        fn(when, now)
    except FirewallBreach:
        raised = True
    bad = []
    if raised != (offset >= 0):
        bad.append(f"offset {offset}d: raised={raised}, but only dates strictly before now may pass")
    for missing in (None,):
        try:
            fn(missing, now)
            bad.append("a missing timestamp was accepted")
        except FirewallBreach:
            pass
    return bad


def lax_require_past(when, now, what="record"):
    """MUTANT: accepts dates ON now and missing timestamps."""
    if when is not None and as_date(when) > as_date(now):
        raise FirewallBreach(what)


# ------------------------------------------------------------------------------------------------ 9. transfer ratio guards

def gen_gain(rng):
    kind = int(rng.integers(0, 8))
    if kind == 0:
        return 0.0
    if kind == 1:
        return float(rng.choice([-TS.EPS_GAIN, TS.EPS_GAIN, TS.EPS_GAIN * 1.0001, -TS.EPS_GAIN * 1.0001]))
    if kind == 2:
        return float(rng.choice([1e-12, -1e-12, 1e-300, -1e-300]))
    if kind == 3:
        return float(rng.choice([1e6, -1e6, 1e150]))
    if kind == 4:
        return math.nan
    return float(rng.normal(0, 0.01))


def prop_transfer_ratio(rng, stats, impl=None):
    ratio = impl or TS.transfer_ratio
    cross, same = gen_gain(rng), gen_gain(rng)
    r = ratio(cross, same)
    bad = []
    for name, v in (("value", r.value), ("capped", r.capped)):
        if v is not None and not math.isfinite(v):
            bad.append(f"{name}={v} for cross={cross!r} same={same!r}")
    if r.capped is not None and abs(r.capped) > TS.RATIO_CAP + 1e-12:
        bad.append(f"capped {r.capped} beyond the cap")
    if math.isnan(cross) or math.isnan(same):
        stats["nan"] += 1
        if r.value is not None or r.status != TS.RatioStatus.INSUFFICIENT:
            bad.append("a NaN gain must be INSUFFICIENT with no ratio")
    elif not same > TS.EPS_GAIN:
        stats["degenerate_denominator"] += 1
        if r.value is not None:
            bad.append(f"denominator {same!r} is not a real positive gain but a ratio {r.value} was returned")
    elif r.value is not None and cross != 0 and math.copysign(1, r.value) != math.copysign(1, cross):
        bad.append(f"sign flipped: cross {cross!r}, ratio {r.value}")
    return bad


def naive_ratio(cross, same, **kw):
    """MUTANT: plain division with numpy semantics (inf / nan instead of a named status)."""
    with np.errstate(all="ignore"):
        v = float(np.float64(cross) / np.float64(same))
    return TS.RatioResult(v, v, TS.RatioStatus.OK, cross, same, "naive")


# ------------------------------------------------------------------------------------------------ 10. calibration

def gen_probs(rng):
    n = int(rng.choice([0, 1, 2, 5, 30, 120, 400]))
    mode = int(rng.integers(0, 4))
    if mode == 0:
        p = rng.random(n)
    elif mode == 1:
        p = rng.choice([0.0, 1.0, 0.5], size=n)
    elif mode == 2:
        p = np.clip(rng.normal(0.8, 0.1, n), 0, 1)
    else:
        p = rng.beta(0.3, 0.3, n)
    y = (rng.random(n) < (p if rng.random() < 0.6 else 0.5)).astype(float)
    return p, y


def prop_calibration_bounds(rng, stats, impl=None):
    ece = impl or CAL.ece
    p, y = gen_probs(rng)
    n_bins, strat = int(rng.integers(2, 21)), str(rng.choice(["uniform", "quantile"]))
    bins = CAL.reliability_diagram(p, y, n_bins, strat)
    bad = []
    e = ece(bins)
    if len(p) == 0:
        stats["empty"] += 1
        if bins or not math.isnan(e):
            bad.append("empty input should give no bins and a NaN ECE, not a made-up number")
        return bad
    if not 0.0 <= e <= 1.0:
        bad.append(f"ECE {e} outside [0,1] (n={len(p)}, bins={n_bins}, {strat})")
    if sum(b.n for b in bins) != len(p):
        bad.append("bins do not partition the samples")
    for b in bins:
        if not (0 <= b.freq <= 1 and 0 <= b.mean_p <= 1 and b.ci_lo - 1e-9 <= b.freq <= b.ci_hi + 1e-9):
            bad.append(f"bin {b.lo:.2f}-{b.hi:.2f}: freq {b.freq:.3f} mean_p {b.mean_p:.3f} ci [{b.ci_lo:.3f},{b.ci_hi:.3f}]")
    eq = CAL.ece_equal_mass(p, y, n_bins)
    if math.isnan(eq) != (len(p) < 3 * n_bins):                       # documented: NaN exactly when there are under 3 rows per bin
        bad.append(f"ece_equal_mass NaN={math.isnan(eq)} with n={len(p)} and {n_bins} bins (documented NaN below 3 rows per bin)")
    for name, v in (("brier", CAL.brier(p, y)), ("ece_equal_mass", eq)):
        if not (math.isnan(v) and name == "ece_equal_mass") and not 0.0 <= v <= 1.0:
            bad.append(f"{name}={v} outside [0,1]")
    stats["scored"] += 1
    return bad


def signed_ece(bins):
    """MUTANT: forgets the absolute value, so over- and under-confidence cancel (and can go negative)."""
    n = sum(b.n for b in bins)
    return float(sum(b.n * b.gap for b in bins) / n) if n else float("nan")


def prop_calibration_rejects_invalid(rng, stats, impl=None):
    n = int(rng.integers(3, 30))
    p, y = rng.random(n), (rng.random(n) < 0.5).astype(float)
    which = int(rng.integers(0, 3))
    if which == 0:
        p[int(rng.integers(0, n))] = float(rng.choice([-0.1, 1.2, math.nan]))
    elif which == 1:
        y[int(rng.integers(0, n))] = 0.5
    else:
        y = y[:-1]
    try:
        (impl or CAL.reliability_diagram)(p, y)
    except ValueError:
        return []
    return [f"invalid input variant {which} was accepted"]


def unchecked_diagram(p, y, *a, **k):
    """MUTANT: skips validation."""
    return []


# ------------------------------------------------------------------------------------------------ 11. determinism

@pytest.fixture(scope="module")
def tiny_world_spec():
    items = (PW.Item("strong", PW.STRONG, (("f0", 4),), 0.02), PW.Item("neg", PW.NEGATIVE, (("f1", 4),), -0.015), PW.Item("noise", PW.NOISE, (("f2", 4),)))
    return PW.WorldSpec("prop_world", items, weeks=30, stocks=30, n_feat=4, discovery_end=15, confirm_end=22).check()


def world_fingerprint(spec, seed):
    w = PW.make_world(spec, seed)
    return w.content_hash(), stable_hash([round(float(v), 8) for v in w.y_raw.to_numpy()[:200]])


def frame_fingerprint(panel):
    return stable_hash({"X": [round(float(v), 8) for v in panel.X.to_numpy().ravel()[:300]], "idx": [str(x) for x in panel.X.index[:60]],
                        "y": [round(float(v), 8) for v in panel.y.to_numpy()[:150]], "weeks": [(str(a), str(b)) for a, b in panel.weeks]})


def prop_world_deterministic(rng, stats, impl=None):
    spec = impl["spec"]
    seed = int(rng.integers(0, 10 ** 6))
    make = impl.get("make", world_fingerprint)
    bad = []
    if make(spec, seed) != make(spec, seed):
        bad.append(f"planted world differs across two builds with seed {seed}")
    if make(spec, seed) == make(spec, seed + 1):
        bad.append("different seeds produced the same world (seed ignored)")
    return bad


def prop_same_year_deterministic(rng, stats, impl=None):
    world = PW.make_world(impl["spec"], 5)
    seed, run = int(rng.integers(0, 10 ** 6)), int(rng.integers(0, 5))
    make = impl.get("make", lambda w, r, s: SY.make_run_panel(w, r, s))
    a, b = frame_fingerprint(make(world, run, seed)), frame_fingerprint(make(world, run, seed))
    bad = [] if a == b else ["same seed and run gave different disguised panels"]
    if a == frame_fingerprint(make(world, run + 1, seed)):
        bad.append("the next run got the identical disguise (runs are not independent presentations)")
    return bad


def prop_credit_deterministic(rng, stats, impl=None):
    led = CR.simulate_decisions(60, int(rng.integers(0, 1000)), lambda r, d: 0.3 * d.pattern * d.timing + 0.2 * d.analog, noise=0.3)
    ds, _ = led.mature("2035-01-01")
    frame = CR.DecisionFrame.build(ds, led.components(), CR.CreditConfig().neutral)

    def sample(seed):
        co = CR.Coalitions(frame, CR.StructuredCombiner({c: 1.0 for c in CR.SIGNAL_COMPONENTS}), CR.Utility.PAYOFF, CR.CreditConfig().neutral)
        return (impl or CR.shapley_sampled)(co, 12, np.random.default_rng(seed)), co
    seed = int(rng.integers(0, 10 ** 6))
    (a, co), (b, _) = sample(seed), sample(seed)
    bad = []
    if not np.array_equal(a, b):
        bad.append("sampled Shapley differs across identical seeds")
    if not np.allclose(a.sum(axis=1), co.v(co.full_mask) - co.v(0)):
        bad.append("efficiency broken: sampled credits do not sum to v(all) - v(none)")
    return bad


def unseeded_shapley(co, n_orders, rng):
    """MUTANT: ignores the generator it was given and draws its own entropy."""
    return CR.shapley_sampled(co, n_orders, np.random.default_rng())


def unseeded_make(spec, seed):
    return world_fingerprint(spec, int(time.time_ns() % 10 ** 6))


# ------------------------------------------------------------------------------------------------ registry: property -> (cases, mutants)

def registry(spec):
    return {
        "stable_hash": (prop_stable_hash, 250, None, {"key_order_sensitive_json": order_sensitive_hash}),
        "knowledge_roundtrip": (prop_knowledge_roundtrip, 60, None, {"drops_tags": lossy_decode}),
        "epistemic_walk": (prop_epistemic_walk, 120, None, {"ignores_legality": forced_apply}),
        "belief_posterior": (prop_belief_posterior, 200, None, {"extrapolates": extrapolating_pool}),
        "beta_belief": (prop_beta_belief, 150, None, {"double_counts": double_counting_update}),
        "hierarchy_shrinks": (prop_hierarchy_shrinks, 8, None, {"raw_child": raw_child_posterior}),
        "trader_blind": (prop_trader_blind, 250, None, {"iso_only": naive_iso_only}),
        "trader_clean_not_flagged": (prop_trader_clean_not_flagged, 150, None, {"flags_everything": lambda payload: ["x"]}),
        "curator_release": (prop_curator_release, 40, None, {"leaky_clock": LeakyCurator}),
        "firewall_fails_closed": (prop_firewall_fails_closed, 40, None, {"missing_is_fine": lenient_gate}),
        "inputs_fail_closed": (prop_inputs_fail_closed, 80, None, {"accepts_all": lambda inputs, now: FF.CheckResult("timestamp", True, (), 1)}),
        "require_past": (prop_require_past, 200, None, {"lax": lax_require_past}),
        "transfer_ratio": (prop_transfer_ratio, 300, None, {"naive_division": naive_ratio}),
        "calibration_bounds": (prop_calibration_bounds, 150, None, {"signed_ece": signed_ece}),
        "calibration_rejects_invalid": (prop_calibration_rejects_invalid, 60, None, {"no_validation": unchecked_diagram}),
        "world_deterministic": (prop_world_deterministic, 3, {"spec": spec}, {"unseeded": {"spec": spec, "make": unseeded_make}}),
        "same_year_deterministic": (prop_same_year_deterministic, 3, {"spec": spec},
                                    {"unseeded": {"spec": spec, "make": lambda w, r, s: SY.make_run_panel(w, r, int(time.time_ns() % 10 ** 6))}}),
        "credit_deterministic": (prop_credit_deterministic, 6, None, {"unseeded": unseeded_shapley}),
    }


# ------------------------------------------------------------------------------------------------ the tests

def _names():
    return list(registry(None))


@pytest.mark.parametrize("name", _names())
def test_property_holds_on_the_real_implementation(name, tiny_world_spec):
    prop, n, impl, _ = registry(tiny_world_spec)[name]
    res = run_property(name, prop, n, seed=11, impl=impl)
    assert res.ok, res.describe()


@pytest.mark.parametrize("name", _names())
def test_property_catches_its_planted_counterexample(name, tiny_world_spec):
    prop, n, _, mutants = registry(tiny_world_spec)[name]
    assert mutants, f"{name} has no mutant: a property that cannot fail proves nothing"
    for label, mutant in mutants.items():
        res = run_property(f"{name}[{label}]", prop, n, seed=11, impl=mutant, stop_after=1)
        assert not res.ok, f"mutant {label} survived {n} cases of {name}: the property is too weak"
        assert res.failures[0].detail, "a failure must say what was violated"


def test_properties_actually_exercised_their_interesting_branches(tiny_world_spec):
    """Non-vacuity: witnesses counted by the properties show the hard branches were reached, not skipped."""
    reg = registry(tiny_world_spec)
    want = {"stable_hash": ("perturbed", 100), "epistemic_walk": ("accepted", 60), "belief_posterior": ("single", 15),
            "hierarchy_shrinks": ("tiny", 6), "curator_release": ("released", 10), "transfer_ratio": ("degenerate_denominator", 60),
            "calibration_bounds": ("empty", 3), "firewall_fails_closed": ("dropped", 40)}
    for name, (key, minimum) in want.items():
        prop, n, impl, _ = reg[name]
        res = run_property(name, prop, n, seed=11, impl=impl)
        assert res.stats[key] >= minimum, f"{name}: only {res.stats[key]} witnesses of {key} (want >= {minimum})"
    res = run_property("trader_blind", prop_trader_blind, 250, seed=11)
    assert sum(1 for f in DATE_FORMS if res.stats[f]) >= len(DATE_FORMS) - 2, "most date forms should be drawn"
    res = run_property("epistemic_walk", prop_epistemic_walk, 120, seed=11)
    assert res.stats["refused"] > 20, "illegal moves must actually be attempted (and refused)"


def test_failures_are_replayable_and_runs_are_reproducible():
    bad = lambda rng, stats, impl: [f"draw {int(rng.integers(0, 100))}"] if rng.random() < 0.3 else []
    a, b = run_property("x", bad, 50, seed=4), run_property("x", bad, 50, seed=4)
    assert not a.ok and a.failures == b.failures
    f = a.failures[0]
    assert replay(bad, 4, f.case) and f.detail in "; ".join(replay(bad, 4, f.case))
    assert "replay(seed=4" in a.describe()
    assert run_property("x", bad, 50, seed=5).failures != a.failures


def test_a_crashing_property_is_a_failure_and_zero_cases_is_vacuous_not_ok_by_accident():
    def boom(rng, stats, impl):
        raise KeyError("missing")
    res = run_property("boom", boom, 3)
    assert not res.ok and "raised KeyError" in res.failures[0].detail
    empty = run_property("none", boom, 0)
    assert empty.ok and empty.n_cases == 0                       # callers must assert n_cases; the registry always does


def test_known_gap_infinite_gain_inputs_are_not_guarded():
    """Documented finding, not a pass: transfer_ratio guards zero/negative/NaN denominators but an infinite cross-context gain still
    yields value=inf (capped to +-3). The property with finite inputs holds; this asserts the gap so it is noticed if it is fixed."""
    r = TS.transfer_ratio(math.inf, 0.01)
    assert r.value is not None and math.isinf(r.value) and r.capped == TS.RATIO_CAP


def test_whole_file_stays_fast():
    t0 = time.time()
    run_property("h", prop_stable_hash, 50, seed=1)
    assert time.time() - t0 < 5
