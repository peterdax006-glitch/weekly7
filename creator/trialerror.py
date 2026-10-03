"""CONTINUOUS TRIAL AND ERROR (owner, 3 Oct 2026: "we want it to be improving as it drills so it can start the trial and error process as it
drills"). Loaded on demand (creator.registry 'trialerror'); nothing here is in the swarm's eager start load.

1 ONLINE ALLOCATION. Instead of fixed halving checkpoints, every scored batch updates each arm's posterior and the next batch's arms are picked by
  TOP-TWO THOMPSON SAMPLING (Russo 2016): Beta(1+k, 1+n-k) for an accuracy, Normal(mean, se) for a PAIRED gain over the control on the same
  subjects. The control and the incumbent are always asked (the floor), so every comparison stays paired. RACING: an arm whose paired 95% CI
  excludes any improvement over the control (upper bound <= 0, after RACE_MIN pairs) is dropped for the rest of the round. A winner is never
  promoted on the round that chose it: the callers keep their out-of-sample re-test on the next round's fresh subjects as the verdict.
  `simulate_*` shows the advantage on synthetic arms with known accuracies.
2 IDEAS, NOT ONLY KNOBS. Feature FAMILIES the drill search proposes like any knob ('fam' in a variant): walk-forward file / directory history
  of git commits (how often THIS file needed a fix before, how often it changed, how recently, change shape, the files themselves, interactions)
  and generic history keys for any source (how often the first key occurred before, gap since the previous item, run length). Every feature of
  an item is computed from items CREATED STRICTLY BEFORE it (never an outcome; the walk-forward learns outcomes only after they resolve).
3 FAST REACTION. `react(state, row)` runs when a drill job finishes: when the result changed its source's best (or enough new rows arrived) the
  source is re-diagnosed at once and targeted variants are queued (learn_queue.jsonl), at most REACT_MAX_PER_HOUR per source.
4 VISIBLE. `experiments_section(state)`: what was tried, why, select, held-out, kept/dropped - read by creator.thinking.trust and learnloop."""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from creator import thinking as T

# ------------------------------------------------------------------------------------------------ 1 online allocation
RACE_MIN = 20                      # paired subjects before an arm may be raced out
TOP_TWO_BETA = 0.5                 # P(take the leader) in top-two Thompson; otherwise the strongest challenger


def beta_sample(k: int, n: int, rnd: random.Random) -> float:
    return rnd.betavariate(1.0 + k, 1.0 + max(0, n - k))


def mean_se(d: Sequence[float]) -> tuple[float, float]:
    """Mean and standard error; few observations get a wide prior-like se (never a false certainty)."""
    n = len(d)
    if n == 0:
        return 0.0, 1.0
    m = sum(d) / n
    if n < 2:
        return m, 1.0
    var = sum((x - m) ** 2 for x in d) / (n - 1)
    return m, max(math.sqrt(var / n), 1e-4 / math.sqrt(n))


PRIOR_N = 2.0                      # pseudo-observations of the prior: a gain of 0 with the pooled variance


def posterior(n: int, s: float, ss: float, v0: float) -> tuple[float, float]:
    """Normal posterior of a mean gain from n observations (sum s, sum of squares ss), shrunk toward 0 by PRIOR_N pseudo-observations with the
    pooled variance v0 (two equal answers never make a false certainty). Returns (mean, sd)."""
    dev = max(0.0, ss - (s * s / n if n else 0.0))
    var = (dev + PRIOR_N * v0) / (max(0, n - 1) + PRIOR_N)
    return s / (n + PRIOR_N), math.sqrt(var / (n + PRIOR_N))


def pooled_var(gains: dict[str, Sequence[float]], default: float = 0.25) -> float:
    xs = [x for d in gains.values() for x in d]
    if len(xs) < 3:
        return default
    m = sum(xs) / len(xs)
    return max(1e-6, sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def raced_out(d: Sequence[float]) -> bool:
    """Paired gains over the control (positive = better): dropped once the 95% CI excludes any improvement."""
    if len(d) < RACE_MIN:
        return False
    m, se = mean_se(d)
    return m + 1.96 * se <= 0.0


def thompson_order(post: dict[str, Callable[[random.Random], float]], rnd: random.Random, k: int) -> list[str]:
    """Top-two Thompson: up to k distinct arms. Each pick samples every arm's posterior; with prob TOP_TWO_BETA the sampled leader is taken,
    otherwise the arm that wins a re-sample in which the leader is excluded (the challenger)."""
    left = dict(post)
    out: list[str] = []
    while left and len(out) < k:
        draw = {a: f(rnd) for a, f in left.items()}
        lead = max(draw, key=lambda a: draw[a])
        if len(left) > 1 and rnd.random() >= TOP_TWO_BETA:
            draw2 = {a: f(rnd) for a, f in left.items() if a != lead}
            lead = max(draw2, key=lambda a: draw2[a])
        out.append(lead)
        left.pop(lead)
    return out


def allocate(gains: dict[str, Sequence[float]], control: str, floor: Sequence[str], rnd: random.Random, k: int = 2,
             dropped: Sequence[str] = ()) -> list[str]:
    """The arms for the next batch: the floor (control, incumbent / pinned) always, plus k Thompson picks among the rest that are not raced out.
    `gains[arm]` = the arm's paired gains over the control so far (Normal posterior on their mean)."""
    out = list(dict.fromkeys([control, *floor]))
    cand = {a: d for a, d in gains.items() if a not in out and a not in dropped and not raced_out(d)}

    v0 = pooled_var(gains)

    def sampler(d: Sequence[float]) -> Callable[[random.Random], float]:
        m, sd = posterior(len(d), sum(d), sum(x * x for x in d), v0)
        return lambda r: r.gauss(m, sd)
    out += thompson_order({a: sampler(d) for a, d in cand.items()}, rnd, k)
    return out


def best_arm(gains: dict[str, Sequence[float]], min_n: int = 1) -> Optional[str]:
    """The arm with the highest posterior-mean paired gain (ties: more evidence first) among arms with at least min_n pairs."""
    v0 = pooled_var(gains)
    ok = {a: posterior(len(d), sum(d), sum(x * x for x in d), v0) for a, d in gains.items() if len(d) >= min_n}
    if not ok:
        return None
    return max(ok, key=lambda a: (ok[a][0], len(gains[a])))


def seeded(*parts: Any) -> random.Random:
    """A deterministic generator: an allocation replayed from the same records picks the same arms."""
    return random.Random(int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16))


# ------------------------------------------------------------------------------------------------ simulation: online vs halving
SIM_RHO = 0.5                      # share of a question's outcome that is common to all arms (an easy question is easy for every strategy)


class _Question:
    """One synthetic question: with prob SIM_RHO an arm's outcome uses the question's shared uniform draw (correlated arms, as real strategies
    answering the same question are), otherwise its own."""
    def __init__(self, rnd: random.Random, rho: float) -> None:
        self.u, self.rnd, self.rho = rnd.random(), rnd, rho

    def pull(self, p: float) -> int:
        return int((self.u if self.rnd.random() < self.rho else self.rnd.random()) < p)


def _pull(p: float, rnd: random.Random) -> int:
    return int(rnd.random() < p)


def sim_halving(ps: Sequence[float], scale: int, rnd: random.Random, schedule: Sequence[tuple[int, int]] = ((10, 4), (20, 2), (40, 1)),
                control: int = 0, rho: float = SIM_RHO, control_cost: int = 1, race: bool = False) -> tuple[int, int]:
    """The CURRENT successive halving (creator.reasondrills HALVING extended to one winner; the judgment rounds use the same shape): every live arm
    answers the same questions up to each cutoff x scale; then the best `keep` survive (the control is never cut). With `race`, after every
    question an arm whose paired gain over the control has a 95% CI <= 0 (after RACE_MIN questions) leaves at once; when none is left the
    search stops (chosen = -1: no arm beats the control). Returns (chosen, calls)."""
    live = list(range(len(ps)))
    k = [0] * len(ps)
    pr = {a: [0, 0.0, 0.0] for a in live}
    asked = 0
    calls = 0

    def out_(a: int) -> bool:
        n, s1, s2 = pr[a]
        if n < RACE_MIN:
            return False
        m = s1 / n
        return m + 1.96 * math.sqrt(max(0.0, (s2 - n * m * m) / (n - 1)) / n) <= 0.0
    for cutoff, keep in schedule:
        upto = cutoff * scale
        for _q in range(asked, upto):
            q = _Question(rnd, rho)
            ys = {}
            for a in live:
                ys[a] = q.pull(ps[a])
                k[a] += ys[a]
                calls += 1 if a != control else control_cost
            if race and control in ys:
                for a in live:
                    if a != control:
                        g = float(ys[a] - ys[control])
                        pr[a][0] += 1
                        pr[a][1] += g
                        pr[a][2] += g * g
                live = [a for a in live if a == control or not out_(a)]
                if live == [control]:
                    return -1, calls
        asked = upto
        rest = sorted((a for a in live if a != control), key=lambda a: -k[a])
        live = [control] + rest[:keep] if control in live else rest[:keep]
    best = max((a for a in live if a != control), key=lambda a: k[a])
    return best, calls


def sim_online(ps: Sequence[float], budget: int, rnd: random.Random, control: int = 0, per_batch: int = 2, mode: str = "beta",
               rho: float = SIM_RHO, control_every: int = 1, control_cost: int = 1) -> tuple[int, int]:
    """Online allocation on the same questions model: per question the control (the floor) plus per_batch top-two Thompson picks among the
    others, until `budget` calls. mode 'beta' = Beta posterior on each arm's accuracy; 'paired' = Normal posterior on the paired gain over the
    control. Racing drops arms whose paired CI excludes improvement. Pick: highest posterior mean. Returns (chosen, calls)."""
    others = [a for a in range(len(ps)) if a != control]
    st = {a: [0, 0.0, 0.0, 0, 0] for a in others}                 # running paired n, gain sum, gain sum of squares, successes, answers

    def ms(a: int) -> tuple[float, float]:
        n, s, q = st[a][0], st[a][1], st[a][2]
        if n == 0:
            return 0.0, 1.0
        m = s / n
        if n < 2:
            return m, 1.0
        return m, max(math.sqrt(max(0.0, (q - n * m * m) / (n - 1)) / n), 1e-4 / math.sqrt(n))

    def out_(a: int) -> bool:
        m, se = ms(a)
        return st[a][0] >= RACE_MIN and m + 1.96 * se <= 0.0

    def sampler(a: int) -> Callable[[random.Random], float]:
        if mode == "beta":
            n, kk = int(st[a][4]), int(st[a][3])
            return lambda r: r.betavariate(1.0 + kk, 1.0 + n - kk)
        tot = sum(int(x[0]) for x in st.values())
        v0 = (sum(x[2] for x in st.values()) / tot - (sum(x[1] for x in st.values()) / tot) ** 2) if tot > 2 else 0.25
        m, sd = posterior(int(st[a][0]), st[a][1], st[a][2], max(1e-6, v0))
        return lambda r: r.gauss(m, sd)
    calls = nq = 0
    while calls < budget:
        q = _Question(rnd, rho)
        nq += 1
        ctl = nq % control_every == 0
        yc = q.pull(ps[control]) if ctl else 0
        calls += int(ctl) * control_cost
        post = {str(a): sampler(a) for a in others if not out_(a)}
        for a in (int(x) for x in thompson_order(post, rnd, per_batch)):
            y = q.pull(ps[a])
            calls += 1
            st[a][3] += y
            st[a][4] += 1
            if not ctl:
                continue
            g = float(y - yc)
            st[a][0] += 1
            st[a][1] += g
            st[a][2] += g * g
    live = [a for a in others if not out_(a) and st[a][4]] or others
    if mode == "beta":
        return max(live, key=lambda a: ((st[a][3] + 1.0) / (st[a][4] + 2.0), st[a][4])), calls
    return max(live, key=lambda a: (posterior(int(st[a][0]), st[a][1], st[a][2], 0.25)[0], st[a][0])), calls


def sim_stop_when_none_better(ps: Sequence[float], rnd: random.Random, cap: int = 20000, per_batch: int = 2, rho: float = SIM_RHO,
                              control_cost: int = 0) -> int:
    """Calls the online allocation spends until EVERY non-control arm is raced out (the owner's audit case: no strategy beats the free
    statistical predictor). Successive halving has no such stop: it always spends its whole schedule."""
    others = [a for a in range(1, len(ps))]
    st = {a: [0, 0.0, 0.0] for a in others}

    def out_(a: int) -> bool:
        n, s1, s2 = st[a]
        if n < RACE_MIN:
            return False
        m = s1 / n
        se = math.sqrt(max(0.0, (s2 - n * m * m) / (n - 1)) / n)
        return m + 1.96 * se <= 0.0
    calls = 0
    while calls < cap:
        live = [a for a in others if not out_(a)]
        if not live:
            return calls
        q = _Question(rnd, rho)
        yc = q.pull(ps[0])
        calls += control_cost

        def smp(a: int) -> Callable[[random.Random], float]:
            m, sd = posterior(int(st[a][0]), st[a][1], st[a][2], 0.25)
            return lambda r: r.gauss(m, sd)
        for x in thompson_order({str(a): smp(a) for a in live}, rnd, per_batch):
            g = float(q.pull(ps[int(x)]) - yc)
            calls += 1
            st[int(x)][0] += 1
            st[int(x)][1] += g
            st[int(x)][2] += g * g
    return calls


def simulate(ps: Sequence[float], target_error: float = 0.1, sims: int = 300, seed: int = 7, mode: str = "beta",
             schedule: Sequence[tuple[int, int]] = ((10, 4), (20, 2), (40, 1)), scales: Sequence[int] = (1, 2, 3, 4, 6, 8, 12, 16),
             per_batch: int = 2, control_every: int = 1, control_cost: int = 1, rho: float = SIM_RHO) -> dict[str, Any]:
    """Calls each method needs to find the best non-control arm with error <= target_error (the smallest tested budget that reaches it).
    Halving's budget is set by its question scale; the online method is run at the same budgets plus a finer grid."""
    best = max(range(1, len(ps)), key=lambda a: ps[a])
    rnd = random.Random(seed)
    out: dict[str, Any] = {"arms": list(ps), "best": best, "target_error": target_error, "sims": sims, "mode": mode, "rho": rho, "control_cost": control_cost, "control_every": control_every,
                           "halving": [], "online": []}
    out["halving_racing"] = []
    for scale in scales:
        res = [sim_halving(ps, scale, rnd, schedule, rho=rho, control_cost=control_cost) for _ in range(sims)]
        out["halving"].append({"scale": scale, "calls": res[0][1], "error": round(sum(1 for c, _n in res if c != best) / sims, 3)})
        res = [sim_halving(ps, scale, rnd, schedule, rho=rho, control_cost=control_cost, race=True) for _ in range(sims)]
        out["halving_racing"].append({"scale": scale, "calls": round(sum(n for _c, n in res) / sims, 1),
                                      "error": round(sum(1 for c, _n in res if c != best) / sims, 3)})
    hb = [h["calls"] for h in out["halving"]]
    grid = sorted(set(hb) | {int(x * f) for x in hb for f in (0.5, 0.75)})
    for budget in grid:
        res = [sim_online(ps, budget, rnd, mode=mode, per_batch=per_batch, control_every=control_every, rho=rho, control_cost=control_cost) for _ in range(sims)]
        out["online"].append({"calls": budget, "error": round(sum(1 for c, _n in res if c != best) / sims, 3)})

    def need(rows: list[dict[str, Any]]) -> Optional[int]:
        ok = sorted(rows, key=lambda r: r["calls"])
        for i, r in enumerate(ok):                                   # the smallest budget from which the error stays at or below the target
            if all(x["error"] <= target_error for x in ok[i:]):
                return int(r["calls"])
        return None
    out["calls_needed"] = {"halving": need(out["halving"]), "online": need(out["online"]), "halving_racing": need(out["halving_racing"])}
    return out


# ------------------------------------------------------------------------------------------------ 2 feature families
GIT_FAMILIES = ("fixhist", "churnhist", "recency", "shape", "files", "fixhist+recency", "fixhist+churnhist", "fixhist*msg", "fixhist+files")
GENERIC_FAMILIES = ("keyfreq", "gap", "runlen", "keyfreq+gap")
GIT_PARTS = ("fixhist", "churnhist", "recency", "shape", "files")
# owner's auditor, 3 Oct: file/dir history on Nupen's own 715 commits gave no held-out gain (too little data); the git families are proposed on
# the cross-repo x sources only (tens of thousands of commits), the generic ones everywhere.
FAMILY_SOURCES: dict[str, tuple[str, ...]] = {"x_git_fixed": GIT_FAMILIES + GENERIC_FAMILIES, "x_git_churn": GIT_FAMILIES + GENERIC_FAMILIES,
                                              "git_fixed": GENERIC_FAMILIES, "git_churn": GENERIC_FAMILIES, "journal_persist": GENERIC_FAMILIES,
                                              "plan_choice": GENERIC_FAMILIES}


def _b(n: float) -> str:
    return str(int(math.log2(n + 1))) if n >= 0 else "-"


def _groups(items: Sequence[Any]) -> list[list[int]]:
    """Item indices in creation order, grouped by equal creation time (a group never sees itself)."""
    order = sorted(range(len(items)), key=lambda i: items[i].created)
    out: list[list[int]] = []
    for i in order:
        if out and items[out[-1][0]].created == items[i].created:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def git_family_keys(items: Sequence[Any], fam: str) -> list[tuple[str, ...]]:
    """Per item the extra keys of git family `fam`, from the setup facts (item.meta: files, sizes, message type, project) of items CREATED
    STRICTLY EARLIER only. Whether an earlier commit was a fix is known when that commit exists; no outcome (y, resolution) is ever read.
    File and directory identities are per project (the privacy-stripped hashes of two projects never mix)."""
    want: set[str] = set()
    for p in fam.split("+"):
        want.update(p.split("*"))
    chg: dict[str, int] = {}
    fixc: dict[str, int] = {}
    dfix: dict[str, int] = {}
    last: dict[str, float] = {}
    out: list[tuple[str, ...]] = [()] * len(items)

    def ids(m: dict[str, Any]) -> tuple[list[str], set[str]]:
        r = str(m.get("repo") or "")
        fs = sorted(f"{r}|{f}" for f in m.get("files") or ())
        return fs, {f.split("/")[0] for f in fs}
    for grp in _groups(items):
        for x in grp:                                              # features of the whole same-time group BEFORE any of it is counted
            m = items[x].meta
            if not m:
                continue
            fs, dirs = ids(m)
            keys: list[str] = []
            fx = max((fixc.get(f, 0) for f in fs), default=0)
            if "fixhist" in want:
                keys += [f"fx:{_b(fx)}", f"fxd:{_b(max((dfix.get(d, 0) for d in dirs), default=0))}"]
            if "churnhist" in want:
                new = sum(1 for f in fs if f not in chg)
                keys += [f"ch:{_b(max((chg.get(f, 0) for f in fs), default=0))}",
                         "new:" + ("none" if not fs else "all" if new == len(fs) else "some" if new else "0")]
            if "recency" in want:
                ages = [float(m["t"]) - last[f] for f in fs if f in last]
                keys.append(f"rc:{_b(min(ages) / 3600.0)}" if ages else "rc:new")
            if "shape" in want:
                na, nd = int(m.get("add") or 0), int(m.get("del") or 0)
                sh = ("none" if na + nd == 0 else "add" if nd == 0 else "del" if na == 0 else "madd" if na > 3 * nd else "mdel" if nd > 3 * na
                      else "mix")
                keys += [f"sh:{sh}", f"nd:{_b(len(dirs))}"]
            if "files" in want:
                keys += [f"f:{f}" for f in fs[:4]]
            if "msg" in want:
                keys.append(f"fx*m:{_b(fx)}|{str(m.get('s', '-')).split(' ')[0]}")
            out[x] = tuple(keys)
        for x in grp:
            m = items[x].meta
            if not m:
                continue
            fs, dirs = ids(m)
            isfix = str(m.get("s", "")).endswith(" fix")
            for f in fs:
                chg[f] = chg.get(f, 0) + 1
                last[f] = float(m["t"])
                if isfix:
                    fixc[f] = fixc.get(f, 0) + 1
            if isfix:
                for dd in dirs:
                    dfix[dd] = dfix.get(dd, 0) + 1
    return out


def generic_family_keys(items: Sequence[Any], fam: str) -> list[tuple[str, ...]]:
    """For any BItem list: from items CREATED STRICTLY BEFORE each one - how often its first key occurred (keyfreq), the gap since the previous
    item (gap, log2 hours), and how many items in a row before it shared its first key (runlen). Never an outcome."""
    want = set(fam.split("+"))
    freq: dict[str, int] = {}
    out: list[tuple[str, ...]] = [()] * len(items)
    prev_t: Optional[float] = None
    run_key, run_n = "", 0
    for grp in _groups(items):
        t0 = items[grp[0]].created
        for i in grp:
            k0 = items[i].keys[0] if items[i].keys else "-"
            keys: list[str] = []
            if "keyfreq" in want:
                keys.append(f"kf:{_b(freq.get(k0, 0))}")
            if "gap" in want:
                keys.append(f"gp:{_b((t0 - prev_t) / 3600.0) if prev_t is not None else 'first'}")
            if "runlen" in want:
                keys.append(f"rl:{_b(run_n) if run_key == k0 else 0}")
            out[i] = tuple(keys)
        for i in grp:
            k0 = items[i].keys[0] if items[i].keys else "-"
            freq[k0] = freq.get(k0, 0) + 1
            run_n = run_n + 1 if k0 == run_key else 1
            run_key = k0
        prev_t = t0
    return out


def with_family(items: Sequence[Any], fam: str) -> list[Any]:
    """`items` with family `fam` appended to every item's keys ('a+b' = both; 'fixhist*msg' = the interaction). Called by
    drillsources.apply_variant, so every consumer of a variant (drills, live predictions, learn loop, decisions) sees the same features."""
    from creator import drillsources as D
    parts = fam.split("+")
    g = "+".join(p for p in parts if p.split("*")[0] in GIT_PARTS)
    o = "+".join(p for p in parts if p.split("*")[0] not in GIT_PARTS)
    extra: list[tuple[str, ...]] = [()] * len(items)
    if g:
        extra = [a + b for a, b in zip(extra, git_family_keys(items, g))]
    if o:
        extra = [a + b for a, b in zip(extra, generic_family_keys(items, o))]
    return [D.BItem(it.keys + e, it.created, it.resolved, it.y, it.subject, it.meta) for it, e in zip(items, extra)]


def propose_family(source: str, best: Optional[dict[str, Any]], rnd: random.Random) -> Optional[dict[str, Any]]:
    """A family proposal: the best-on-select variant with a family added (or swapped). None when the source has no families."""
    fams = FAMILY_SOURCES.get(source)
    if not fams:
        return None
    base = {k: v for k, v in (best or {"decay": 0.97, "k": 3.0}).items() if k not in ("search", "learn", "why")}
    return {**base, "fam": rnd.choice([f for f in fams if f != base.get("fam")] or list(fams)), "search": 1}


# ------------------------------------------------------------------------------------------------ search hygiene (owner's audit, 3 Oct)
IMPROVE_SE_FRAC = 0.5              # a new best must beat the old by this fraction of the select gain's standard error
NOISE_MIN_ROWS = 12                # variants needed before the select-vs-held-out rank agreement is judged
NOISE_RHO = 0.1                    # below this the select part does not predict the held-out part: searching it only follows noise


def improve_threshold(row: dict[str, Any]) -> float:
    """Approximate paired SE of a select-Brier difference between two variants: the incumbent's select gain SE (from its 95% CI) is an upper
    bound for a paired difference between highly correlated variants; IMPROVE_SE_FRAC of it is required."""
    ci = (row.get("select") or {}).get("gain_ci95") or [None, None]
    if ci[0] is None or ci[1] is None:
        return 1e-9
    return max(1e-9, IMPROVE_SE_FRAC * (float(ci[1]) - float(ci[0])) / 3.92)


def spearman(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    n = len(a)
    if n < 3:
        return None

    def ranks(v: Sequence[float]) -> list[float]:
        o = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[o[j + 1]] == v[o[i]]:
                j += 1
            for x in range(i, j + 1):
                r[o[x]] = (i + j) / 2.0
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va, vb = sum((x - ma) ** 2 for x in ra), sum((y - mb) ** 2 for y in rb)
    return cov / math.sqrt(va * vb) if va > 0 and vb > 0 else None          # a constant side: agreement cannot be judged


def select_predicts_heldout(rows: Sequence[dict[str, Any]]) -> Optional[float]:
    """Rank correlation between the select and held-out Brier across a source's variants (None = too few to tell)."""
    ok = [r for r in rows if (r.get("select") or {}).get("n") and (r.get("heldout") or {}).get("n")]
    if len({(r["select"]["brier"], r["heldout"]["brier"]) for r in ok}) < NOISE_MIN_ROWS:   # distinct results, not repeated rows
        return None
    return spearman([r["select"]["brier"] for r in ok], [r["heldout"]["brier"] for r in ok])


# ------------------------------------------------------------------------------------------------ 3 fast reaction
REACT_MAX_PER_HOUR = 2
REACT_EVERY_ROWS = 6
REACT_MAX_VARIANTS = 6


def react_path(state: Path) -> Path:
    return Path(state) / "thinking" / "react.json"


def react(state: Path, row: dict[str, Any], repo: Path, journal: Path, research: Path, now: Optional[float] = None) -> Optional[dict[str, Any]]:
    """Called when a drill job finishes. If the row set a new best for its source, or REACT_EVERY_ROWS rows arrived since the last reaction, and
    the source has not reacted REACT_MAX_PER_HOUR times in the past hour: re-diagnose THAT source with its best variant (learnloop's grouping,
    no 30-min wait) and queue variants aimed at its top significant weakness. Never raises into the drill job."""
    from creator import drillsources as D
    from creator import learnloop as LL
    now = time.time() if now is None else now
    s = str(row.get("source") or "")
    if s not in LL.DIAG_SOURCES or not (row.get("select") or {}).get("n"):
        return None
    try:
        st = json.loads(react_path(state).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        st = {}
    me = st.setdefault(s, {"times": [], "rows_at": 0})
    rows = [r for r in T._jsonl(D.runs_path(Path(state))) if r.get("source") == s and (r.get("select") or {}).get("n")]
    b = D.best_variant(Path(state), s)
    new_best = b is not None and b["variant"] == row.get("variant")
    me["times"] = [t for t in me["times"] if now - float(t) < 3600.0]
    if not (new_best or len(rows) - int(me["rows_at"]) >= REACT_EVERY_ROWS) or len(me["times"]) >= REACT_MAX_PER_HOUR or b is None:
        return None
    me["times"].append(now)
    me["rows_at"] = len(rows)
    var = dict(b["variant"])
    raw = D.load(s, state, repo, journal, research)
    pairs = LL.predict(raw, s, var, journal)
    acc: dict[tuple[str, str], dict[str, Any]] = {}
    LL.drill_groups(s, pairs, acc)
    ws = [w for w in LL.rank(acc) if w.get("significant")]
    out: dict[str, Any] = {"source": s, "at": LL._iso(now), "trigger": "new_best" if new_best else "rows", "weaknesses": len(ws), "queued": 0}
    if ws:
        w = ws[0]
        rid = "RX-" + hashlib.sha1(f"{s}|{w['id']}|{now}".encode()).hexdigest()[:10]
        ran = {json.dumps({k: x for k, x in (r.get("variant") or {}).items() if k not in ("learn", "why")}, sort_keys=True) for r in rows}
        queued = {json.dumps({k: x for k, x in q["variant"].items() if k not in ("learn", "why")}, sort_keys=True) for q in LL.queue_rows(state)
                  if q.get("source") == s}
        vs = [v for v in LL.variants_for(w, var, rid)
              if json.dumps({k: x for k, x in v.items() if k not in ("learn", "why")}, sort_keys=True) not in ran | queued][:REACT_MAX_VARIANTS]
        if vs:
            LL._append(state, "queue", [{"source": s, "variant": v, "improvement": rid, "at": LL._iso(now), "why": f"react:{w['kind']} {w['where']}"}
                                        for v in vs])
        out.update(weakness={k: w.get(k) for k in ("id", "kind", "where", "n", "loss_total", "loss_ci95")}, queued=len(vs), id=rid)
    st[s] = me
    react_path(state).parent.mkdir(parents=True, exist_ok=True)
    react_path(state).write_text(json.dumps(st), encoding="utf-8")
    with (Path(state) / "thinking" / "react_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(out, default=str) + "\n")
    return out


# ------------------------------------------------------------------------------------------------ 4 visible
def why_of(v: dict[str, Any]) -> str:
    if v.get("learn"):
        return f"diagnosis {v['learn']}"
    if v.get("fam") and v.get("search"):
        return f"idea: feature family {v['fam']}"
    return "search" if v.get("search") else "grid"


def drill_experiments(state: Path, last: int = 15) -> dict[str, Any]:
    """Per drill source, its latest experiments in completion order: variant, why, select / held-out Brier, and whether it was KEPT (became the
    best on select by more than the improvement threshold) or dropped. Plus per feature family: tried, kept, best held-out vs no family."""
    from creator import drillsources as D
    out: dict[str, Any] = {}
    by: dict[str, list[dict[str, Any]]] = {}
    for r in T._jsonl(D.runs_path(Path(state))):
        if (r.get("select") or {}).get("n"):
            by.setdefault(str(r.get("source")), []).append(r)
    for s, rows in by.items():
        dg = rows[-1].get("digest")
        rows = [r for r in rows if r.get("digest") == dg]
        best: Optional[dict[str, Any]] = None
        exps = []
        fams: dict[str, dict[str, Any]] = {}
        for r in rows:
            kept = best is None or r["select"]["brier"] < best["select"]["brier"] - improve_threshold(best)
            if kept:
                best = r
            v = r.get("variant") or {}
            exps.append({"variant": {k: x for k, x in v.items() if k not in ("search", "learn")}, "why": r.get("why") or why_of(v),
                         "select": r["select"]["brier"], "heldout": (r.get("heldout") or {}).get("brier"),
                         "heldout_gain_ci95": (r.get("heldout") or {}).get("gain_ci95"), "kept": kept})
            f = str(v.get("fam") or "none")
            g = fams.setdefault(f, {"tried": 0, "kept": 0, "best_select": None, "heldout_of_best_select": None, "heldout_gain_ci95": None})
            g["tried"] += 1
            g["kept"] += int(kept)
            if g["best_select"] is None or r["select"]["brier"] < g["best_select"]:
                g.update(best_select=r["select"]["brier"], heldout_of_best_select=(r.get("heldout") or {}).get("brier"),
                         heldout_gain_ci95=(r.get("heldout") or {}).get("gain_ci95"), n_heldout=(r.get("heldout") or {}).get("n"))
        out[s] = {"tried": len(rows), "select_predicts_heldout_rho": select_predicts_heldout(rows), "latest": exps[-last:], "families": fams}
    return out


def experiments_section(state: Path) -> dict[str, Any]:
    """The compact 'experiments' section (trust.json / learnloop): drill experiments, the fast reactions, and the online judgment / reasoning
    allocations (arms asked, raced out, current leader) when those modules have records."""
    out: dict[str, Any] = {"drills": drill_experiments(state)}
    out["reactions"] = T._jsonl(Path(state) / "thinking" / "react_log.jsonl")[-10:]
    try:
        from creator import reasondrills as RD
        out["reasoning"] = RD.online_report(state)
    except Exception as e:                                           # noqa: BLE001 - a report never breaks the trust file
        out["reasoning"] = {"error": f"{type(e).__name__}: {e}"[:200]}
    try:
        from creator import judgment as J
        out["judgment"] = {t: J.online_report(state, t) for t in J.PUB_TOPICS}
    except Exception as e:                                           # noqa: BLE001
        out["judgment"] = {"error": f"{type(e).__name__}: {e}"[:200]}
    return out
