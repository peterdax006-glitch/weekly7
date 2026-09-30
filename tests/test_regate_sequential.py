"""F12 (C69 ledger W-06 / W-12; C69 sections 12-14, 27, 28, 31; C66 section 42 quality gate): re-gating a finding as its evidence
accumulates must neither retire a true effect for lack of time nor let repeated looks manufacture a pass.

W-12 finding: at default settings the genuine planted pattern DBR44ef231394d2 (feature lv20, planted truth 'genuine') was gated at
cycle 31 (FAILED: one unseen year), 45 and 59 (NEEDS_MORE_EVIDENCE: replication heterogeneity, failure episodes) and RETIRED by the
hard cap of three looks. Two defects: (1) the look cap; (2) the replication ledger froze at the first look (run ids '-w0' / '-w1' were
re-used, so no later look could add a run). Both are fixed (evidence.SequentialPlan, evidence.replicate blocks) and every mechanism here
has a planted case it must catch, a null case and the empty case. Fast tests are synthetic; the planted-world proofs are long
(W7_LONG_TESTS=1) and the multi-seed runner is this file's __main__:

    python tests/test_regate_sequential.py gate --seeds 0 1 2 3 4 5 --world default      sequential gate on genuine / coincidence / null
    python tests/test_regate_sequential.py loop --seed 3 --world default                  the full research loop at default settings
Results go to state/research/regate_sequential/ with provenance."""
from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.learning import promotion as PR                                   # noqa: E402
from engine.research import evidence as EV                                    # noqa: E402
from engine.research import loop as LP                                        # noqa: E402
from engine.research import quality_gate as QG                                # noqa: E402
from engine.research import replication as RP                                 # noqa: E402

warnings.filterwarnings("ignore")
LONG = os.environ.get("W7_LONG_TESTS") == "1"
OUT = Path(__file__).resolve().parents[1] / "state" / "research" / "regate_sequential"
NULL_PLANT = {"vol_state_sd": 0.0, "base_sigma": (0.015, 0.015), "coincidence_rate": 0.0, "earnings_shock": 0.0}   # nothing predicts


# ============================================================================================================ the alpha-spending plan
def test_alpha_spending_sums_below_alpha_for_any_number_of_looks():
    plan = EV.SequentialPlan()
    a = [plan.alpha_at(k) for k in range(1, 20001)]
    assert all(x > y for x, y in zip(a, a[1:]))                                 # each later look is stricter
    assert abs(a[0] - 0.05 * 6 / math.pi ** 2) < 1e-12 and sum(a) < plan.alpha and plan.spent(20000) < plan.alpha
    assert plan.spent(3) == pytest.approx(sum(a[:3]))
    with pytest.raises(ValueError):
        plan.alpha_at(0)
    assert EV.SequentialPlan(alpha=0.7).validate() and EV.SequentialPlan(horizon_dates=10).validate() and plan.validate() == []


def test_sequential_policies_only_ever_raise_thresholds():
    plan, base, rbase = EV.SequentialPlan(), PR.PromotionPolicy(), RP.DEFAULT_POLICY
    for k in (1, 2, 5, 40, 1000):
        p, r = plan.promotion_policy(k), plan.replication_policy(k)
        assert p.alpha <= base.alpha and p.min_oos_t >= base.min_oos_t and p.min_effect == base.min_effect
        assert r.alpha <= rbase.alpha and r.ci_level >= rbase.ci_level and r.max_i2 == rbase.max_i2
        assert p.validate() == [] and r.validate() == []
        q = plan.quality_policy(k, "c")
        assert q.promotion == p and q.min_failure_episodes == QG.QualityPolicy().min_failure_episodes
    assert plan.promotion_policy(50).min_oos_t > plan.promotion_policy(2).min_oos_t


def _stat_pass(x: np.ndarray, pol) -> bool:
    """The gate's own statistical-validity test (engine.learning.promotion) on a weekly effect series."""
    t = PR.t_stat(x)
    ev = PR.StatisticalEvidence(float(x.mean()), int(len(x)), t, PR.one_sided_p(t), 1, float(len(x)))
    return PR.gate_statistical_validity(ev, pol).status == PR.PASS and t >= pol.min_oos_t


def _looks(mu: float, looks: int, first: int, every: int, sims: int, plan, fixed: bool, seed: int) -> np.ndarray:
    """Look index (1-based) at which each simulated finding first passes the statistical gate, 0 = never. Effects accumulate: look k
    sees the first `first + (k-1) * every` weekly effects (the loop re-gates on the growing out-of-sample window)."""
    rng = np.random.default_rng(seed)
    out = np.zeros(sims, int)
    n_max = first + (looks - 1) * every
    for s in range(sims):
        x = rng.normal(mu, 0.1, n_max)
        for k in range(1, looks + 1):
            pol = PR.PromotionPolicy() if fixed else plan.promotion_policy(k)
            if _stat_pass(x[:first + (k - 1) * every], pol):
                out[s] = k
                break
    return out


def test_fixed_alpha_looks_manufacture_passes_and_the_plan_does_not():
    """Planted defect: looking again and again at a NULL effect with the fixed-sample alpha passes it far more often than alpha
    (optional stopping). Under the plan the rate across 20 looks stays within alpha (binomial margin)."""
    plan, sims = EV.SequentialPlan(), 600
    fixed = _looks(0.0, 20, 60, 13, sims, plan, True, 1)
    seq = _looks(0.0, 20, 60, 13, sims, plan, False, 1)
    r_fixed, r_seq = float((fixed > 0).mean()), float((seq > 0).mean())
    assert r_fixed > 0.10, r_fixed                                              # ~2-4x alpha: the defect is real and caught
    assert r_seq <= plan.alpha + 2.5 * math.sqrt(plan.alpha * (1 - plan.alpha) / sims), r_seq
    assert r_seq < r_fixed / 2


def test_a_true_effect_is_not_starved_of_alpha():
    """A modest true effect (0.03 per week, sd 0.1: t ~ 2.3 at 60 weeks) is promoted by the statistical gate at some look in almost
    every simulation: the plan delays some by a few looks, it does not kill them. The old rule (at most three looks) lost far more."""
    plan = EV.SequentialPlan()
    seq = _looks(0.03, 20, 60, 13, 300, plan, False, 2)
    fixed = _looks(0.03, 20, 60, 13, 300, plan, True, 2)
    capped = ((fixed > 0) & (fixed <= 3)).mean()                                 # W-12's cap: three looks at the fixed alpha
    assert (seq > 0).mean() >= 0.93 and (seq > 0).mean() >= capped, ((seq > 0).mean(), capped)
    assert np.median(seq[seq > 0]) <= 2


# ============================================================================================================ futility
def _bundle(effects=None, repl=None) -> EV.Bundle:
    oos = None
    if effects is not None:
        d = tuple(str(x.date()) for x in pd.date_range("2019-01-04", periods=len(effects), freq="W-FRI"))
        oos = QG.OOSBundle(PR.StatisticalEvidence(float(np.mean(effects)), len(effects), 1.0, 0.2, 1, float(len(effects))),
                           PR.OOSEvidence("2018-12-01", d, tuple(float(v) for v in effects), 0.1), (2018,))
    return EV.Bundle("S", QG.QualityEvidence(oos=oos, replication=repl), {}, {})


def test_futility_retires_only_on_measured_absence():
    plan, rng = EV.SequentialPlan(), np.random.default_rng(0)
    assert "measured absent" in plan.futility(_bundle(rng.normal(-0.05, 0.1, 40)))            # planted: an effect that is not there
    assert plan.futility(_bundle(rng.normal(0.05, 0.1, 40))) is None                         # a real one keeps being looked at
    assert plan.futility(_bundle(rng.normal(-0.05, 0.1, 10))) is None                        # too few periods to call it absent
    assert plan.futility(_bundle()) is None                                                  # empty evidence: keep looking, never retire
    ev, _ = QG.reference_evidence()
    failed = type(ev.replication)(**{**ev.replication.__dict__, "status": RP.Status.FAILED, "n_refuting": 2})
    assert "replication FAILED" in plan.futility(_bundle(rng.normal(0.05, 0.1, 40), failed))


# ============================================================================================================ replication blocks
def test_replication_blocks_are_complete_disjoint_and_wait_for_their_dates():
    dates = pd.date_range("2019-01-04", periods=40, freq="W-FRI")
    b = EV.replication_blocks(dates, "2019-01-01", 13)
    assert [len(x) for x in b] == [13, 13, 13]                                   # the 40th date waits for its block to complete
    assert all(pd.Timestamp(x[-1]) < pd.Timestamp(y[0]) for x, y in zip(b, b[1:]))
    later = EV.replication_blocks(dates, b[1][-1], 13)
    assert len(later) == 1 and pd.Timestamp(later[0][0]) == pd.Timestamp(b[2][0])   # only what follows the newest recorded window
    assert EV.replication_blocks([], "2019-01-01", 13) == [] and EV.replication_blocks(dates, "2030-01-01", 13) == []


def _synthetic(n_dates=110, n_names=40, strength=1.0, seed=0):
    """A small (date, ticker) frame: the touch outcome depends on the score with the given strength (0 = null)."""
    rng = np.random.default_rng(seed)
    d = pd.date_range("2017-01-06", periods=n_dates, freq="W-FRI")
    idx = pd.MultiIndex.from_product([d, [f"T{i:02d}" for i in range(n_names)]])
    s = rng.normal(0, 1, len(idx))
    y = (rng.random(len(idx)) < 1 / (1 + np.exp(-(strength * s - 1.0)))).astype(float)
    G = pd.DataFrame({"end": idx.get_level_values(0) + pd.Timedelta(days=7), "m_vol": rng.normal(0.01, 0.002, len(idx))}, index=idx)
    return G, pd.Series(s, index=idx), pd.Series(y, index=idx)


def _masks(G, upto, cfg):
    d = pd.to_datetime(G.index.get_level_values(0))
    k = int(len(pd.unique(d)) * cfg.orient_frac)
    train_end = sorted(pd.unique(d))[k - 1]
    live = np.asarray(d < pd.Timestamp(upto))
    return live & np.asarray(d <= train_end), live & np.asarray(d >= pd.Timestamp(train_end) + pd.Timedelta(days=cfg.purge_days))


def test_a_persistent_ledger_gains_runs_at_every_look():
    """Planted defect (W-12): with the ledger kept across looks, the old code re-used run ids and never added a run after look 1, so the
    first look's replication verdict was frozen. Now each look adds the complete new blocks, never re-records one, and the discovery is
    registered exactly once."""
    cfg, spec = EV.EvidenceConfig(), EV.FindingSpec("S_lv20", "lv20", 1.0)
    G, s, y = _synthetic()
    led = RP.ReplicationLedger()
    seen = []
    for upto in ("2018-01-05", "2018-04-06", "2018-07-06", "2019-01-11"):
        tr, te = _masks(G, upto, cfg)
        now = str((pd.Timestamp(upto) + pd.Timedelta(days=14)).date())
        a, p = EV.replicate(G, s, y, tr, te, spec, cfg, now, "c", "d", led)
        seen.append(p["repl_runs"])
    assert seen == sorted(seen) and seen[-1] > seen[0] >= 1, seen
    ids = [r.run_id for r in led.runs_for("ES_lv20", "2030-01-01")]
    assert len(ids) == len(set(ids)) and len(led.discoveries()) == 1
    wins = sorted(r.window for r in led.runs_for("ES_lv20", "2030-01-01"))
    assert all(a[1] < b[0] for a, b in zip(wins, wins[1:]))                     # disjoint fresh periods
    assert a.status == RP.Status.REPLICATED, (a.status, a.reasons)               # a real effect replicates once the quarters exist


def test_replication_of_a_null_score_never_replicates():
    cfg, spec = EV.EvidenceConfig(), EV.FindingSpec("S_null", "lv20", 1.0)
    G, s, y = _synthetic(strength=0.0, seed=4)
    led = RP.ReplicationLedger()
    for upto in ("2018-01-05", "2018-07-06", "2019-01-11"):
        tr, te = _masks(G, upto, cfg)
        a, p = EV.replicate(G, s, y, tr, te, spec, cfg, str((pd.Timestamp(upto) + pd.Timedelta(days=14)).date()), "c", "d", led)
    assert a is None or a.status != RP.Status.REPLICATED


def test_replication_on_an_empty_test_window_adds_nothing():
    cfg, spec = EV.EvidenceConfig(), EV.FindingSpec("S_e", "lv20", 1.0)
    G, s, y = _synthetic()
    tr, _ = _masks(G, "2018-01-05", cfg)
    led = RP.ReplicationLedger()
    a, p = EV.replicate(G, s, y, tr, np.zeros(len(G), bool), spec, cfg, "2018-01-20", "c", "d", led)
    assert p["repl_runs"] == 0 and p["repl_runs_added"] == 0 and a.status != RP.Status.REPLICATED


# ============================================================================================================ the sequential gate
def _ref_bundle():
    ev, _ = QG.reference_evidence()
    return EV.Bundle("REF", ev, {}, {})


def test_sequential_gate_tightens_with_the_look_number():
    """The clean reference bundle PROMOTEs at look 1; at a far later look its OOS t (~3) no longer clears z(1 - alpha_k), so the same
    evidence is NOT promoted - later looks demand more, which is what makes watching longer honest."""
    b, now = _ref_bundle(), "2021-06-01"
    assert EV.verdicts(EV.gate([b], now, "refcode"))["REF"] == "PROMOTE"          # fixed-sample gate unchanged
    assert EV.verdicts(EV.gate([b], now, "refcode", looks={"REF": 1}))["REF"] == "PROMOTE"
    late = EV.gate([b], now, "refcode", looks={"REF": 60})
    assert EV.verdicts(late)["REF"] != "PROMOTE" and "out_of_sample" in EV.blocking(late, "REF")
    with pytest.raises(ValueError, match="no look number"):
        EV.gate([b], now, "refcode", looks={})
    assert EV.gate([], now, "refcode", looks={}).decisions == ()                 # empty: nothing to gate, nothing promoted


def test_mixed_looks_in_one_call_keep_their_own_alpha():
    ev, _ = QG.reference_evidence()
    a, b = EV.Bundle("A", ev, {}, {}), EV.Bundle("B", ev, {}, {})
    rep = EV.gate([a, b], "2021-06-01", "refcode", looks={"A": 1, "B": 60})
    assert [d.subject_id for d in rep.decisions] == ["A", "B"] and rep.promoted == ("A",)


# ============================================================================================================ the loop's schedule
def test_regate_schedule_has_no_look_cap():
    """Planted defect (W-12): a finding with three looks used to be dropped however much evidence was still coming. Now it is due
    whenever a quarter of new dates has matured, at any look count; not due before that; never twice in one cycle."""
    d = pd.date_range("2018-01-05", periods=40, freq="W-FRI")
    M = pd.DataFrame({"x": 1.0}, index=pd.MultiIndex.from_product([d, ["A"]]))
    reg = {"D1": {"key": "k1", "through": "2018-01-05", "looks": 3}, "D2": {"key": "k2", "through": str(d[30].date()), "looks": 1},
           "D3": {"key": "k3", "through": "2018-01-05", "looks": 40}}
    assert LP.regate_due(reg, M) == ["k1", "k3"]
    assert LP.regate_due(reg, M, done=("k1",)) == ["k3"]
    assert LP.regate_due({}, M) == [] and LP.regate_due(reg, M.iloc[0:0]) == []
    assert not hasattr(LP, "REGATE_MAX_LOOKS") and isinstance(LP.REGATE_PLAN, EV.SequentialPlan)


# ============================================================================================================ planted-world runners
def _provenance(args: dict) -> dict:
    try:
        from engine import provenance
        return provenance.stamp(args, int(args.get("seed", 0)))
    except Exception as e:                                                      # noqa: BLE001 - recorded, never silent
        return {"error": f"{type(e).__name__}: {e}"}


def sequential_trace(seed: int, world: str = "default", features=("lv20", "price_low", "insider_recent"), start: str = "2017-12-15",
                     every: int = LP.REGATE_NEW_DATES, plan: EV.SequentialPlan = LP.REGATE_PLAN) -> list[dict]:
    """The loop's re-gate schedule on the planted world without the rest of the loop: every finding is gated from `start` every
    `every` decision dates, with its persistent replication ledger, numbered looks and the plan's alpha, until PROMOTE or retirement."""
    from engine.research import feeds as FD
    plant = {"seed": seed, **(NULL_PLANT if world == "null" else {})}
    feed = FD.planted_feed(FD.PlantConfig(**plant))
    truth = feed.source.world.truth if hasattr(feed.source, "world") else {}
    dates = [d for d in feed.dates() if d >= start][::every]
    leds = {f: RP.ReplicationLedger() for f in features}
    store = QG.QuarantineStore()
    live, rows = {f: 0 for f in features}, []
    for now in dates:
        F = feed.store.upto(now)
        M = F[pd.to_datetime(F["end"]) < pd.Timestamp(now)]
        for f in [f for f in features if live[f] is not None]:
            live[f] += 1
            spec = EV.FindingSpec("D_" + f, f, 1.0, "VOLATILITY", n_tests_searched=20, has_falsifier=True, seed=seed)
            b = EV.assemble(M, spec, now, code_hash="f12", data_hash=f"planted{seed}", created_real="2026-09-30T00:00:00+00:00",
                            ledger=leds[f], look=live[f], plan=plan)
            rep = EV.gate([b], now, "f12", store=store, looks={spec.subject_id: live[f]}, plan=plan)
            v = EV.verdicts(rep)[spec.subject_id]
            why = "promoted" if v == "PROMOTE" else "quarantined" if v == "QUARANTINED" else plan.futility(b)
            role = ("genuine" if f in truth.get("genuine", ()) else "coincidence" if f in truth.get("noise", ()) else "null")
            if world == "null":
                role = "null"
            rows.append({"seed": seed, "world": world, "feature": f, "role": role, "now": now, "look": live[f], "verdict": v,
                         "alpha": plan.alpha_at(live[f]), "n_test": b.parts.get("n_test"), "effect_test": b.parts.get("effect_test"),
                         "repl_runs": b.parts.get("repl_runs"), "repl_i2": b.parts.get("repl_i2"),
                         "blocking": sorted(EV.blocking(rep, spec.subject_id)), "stopped": why or ""})
            if why:
                live[f] = None
    return rows


def run_loop(seed: int, world: str = "default", max_cycles: int | None = None, root: Path | None = None) -> dict:
    """The full research loop at DEFAULT settings (scripts/research_loop.py defaults: LoopConfig(), TwoStageConfig(gate_min_weeks=26),
    the feed's own sweeps, the C68 stages registered) on planted world `seed`; resumable from its per-cycle checkpoint. Writes the
    per-cycle gate history and the promotion cycle of every finding to state/research/regate_sequential/loop_<world>_s<seed>.json."""
    from engine.research import error_loop as C68                               # noqa: F401 - registers the C68 stages (as the script)
    from engine.research import two_stage as TS
    plant = {"seed": seed, **(NULL_PLANT if world == "null" else {})}
    feed = LP.world_feed("planted", plant=plant)
    truth = feed.source.world.truth if hasattr(feed.source, "world") else {}
    run_id = f"f12_{world}_s{seed}"
    root = root or OUT / run_id
    root.mkdir(parents=True, exist_ok=True)
    cfg = LP.LoopConfig(run_id=run_id, seed=seed, checkpoint="cycle", two_stage=TS.TwoStageConfig(gate_min_weeks=26))
    out = OUT / f"loop_{world}_s{seed}.json"
    state, rt, info = LP.open_loop(feed, root, cfg, sweeps=feed.sweeps())
    t0, n = time.monotonic(), 0
    rec = {"seed": seed, "world": world, "code_hash": rt.code_hash, "resume": info.get("action"), "cycles": [],
           "provenance": _provenance({"seed": seed, "world": world, "run_id": run_id, "plan": str(LP.REGATE_PLAN)})}
    if out.exists():
        rec["cycles"] = json.loads(out.read_text(encoding="utf-8")).get("cycles", [])
    while max_cycles is None or n < max_cycles:
        rep = LP.step(state, rt)
        if rep is None:
            break
        n += 1
        gates = {k[5:]: {x: v.get(x) for x in ("feature", "verdict", "looks", "retired", "history", "blocking")}
                 for k, v in state.lineage.nodes.items() if v["kind"] == "GATE"}
        rec["cycles"].append({"cycle": rep["cycle"], "now": rep["now"], "knowledge": sorted(k["feature"] for k in state.knowledge.values()),
                              "failed": rep["failed"], "counters": {k: v for k, v in rep["counters"].items() if k.startswith("gate_")}})
        rec.update(gates=gates, knowledge={k: dict(v) for k, v in state.knowledge.items()}, seconds=round(time.monotonic() - t0, 1),
                   truth={k: truth.get(k) for k in ("genuine", "genuine_event", "noise", "noise_family")})
        out.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    return summarise_loop(rec)


def summarise_loop(rec: dict) -> dict:
    """First cycle each feature was promoted, its role in the planted truth, and whether anything outside the genuine sets was filed."""
    truth = rec.get("truth") or {}
    genuine = set(truth.get("genuine") or ()) | set(truth.get("genuine_event") or ())
    first = {}
    for c in rec["cycles"]:
        for f in c["knowledge"]:
            first.setdefault(f, (c["cycle"], c["now"]))
    wrong = sorted(f for f in first if rec["world"] == "null" or f not in genuine)
    return {"seed": rec["seed"], "world": rec["world"], "cycles": len(rec["cycles"]), "promoted": first, "false_promotions": wrong,
            "gates": {k: (v.get("feature"), v.get("verdict"), v.get("looks"), v.get("retired")) for k, v in (rec.get("gates") or {}).items()}}


@pytest.mark.integration
@pytest.mark.skipif(not LONG, reason="planted-world sequential gate: set W7_LONG_TESTS=1 (~15 min)")
def test_planted_worlds_promote_the_genuine_never_the_coincidence_or_null():
    rows = []
    for s in range(6):
        rows += sequential_trace(s, "default")
    for s in range(3):
        rows += sequential_trace(s, "null", features=("lv20", "price_low"))
    T = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    T.to_json(OUT / "sequential_trace_test.json", orient="records", indent=1)
    promoted = T[T["verdict"] == "PROMOTE"]
    assert set(promoted["role"]) <= {"genuine"}, promoted
    assert not len(T[(T["world"] == "null") & (T["verdict"] == "PROMOTE")])
    gen = T[T["role"] == "genuine"]
    assert gen.groupby("seed")["verdict"].apply(lambda v: (v == "PROMOTE").any()).mean() >= 5 / 6


@pytest.mark.integration
@pytest.mark.skipif(not LONG, reason="full research loop on the planted world: set W7_LONG_TESTS=1 (~1-2 h)")
def test_full_loop_at_default_settings_promotes_the_genuine_pattern(tmp_path):
    s = run_loop(int(os.environ.get("W7_F12_SEED", "1")), "default", root=tmp_path)
    assert s["promoted"] and not s["false_promotions"], s


# ============================================================================================================ CLI (detached runs)
def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="F12 planted-world proofs of the sequential re-gate")
    ap.add_argument("mode", choices=("gate", "loop"))
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--world", choices=("default", "null"), default="default")
    ap.add_argument("--max-cycles", type=int, default=None)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = [a.seed] if a.seed is not None else a.seeds
    if a.mode == "gate":
        for s in seeds:
            t0 = time.monotonic()
            rows = sequential_trace(s, a.world, features=("lv20", "price_low", "insider_recent"))
            (OUT / f"gate_{a.world}_s{s}.json").write_text(json.dumps({"rows": rows, "seconds": round(time.monotonic() - t0, 1),
                                                                       "provenance": _provenance({"seed": s, "world": a.world})},
                                                                      indent=1, default=str), encoding="utf-8")
            print(json.dumps([{k: r[k] for k in ("feature", "role", "now", "look", "verdict", "stopped")} for r in rows]), flush=True)
    else:
        for s in seeds:
            print(json.dumps(run_loop(s, a.world, a.max_cycles), default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
