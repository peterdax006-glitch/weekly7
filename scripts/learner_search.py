"""Real-cache search for learners that generalise (canon C54/C55/C56; Bible Phases 11, 21-23, 45; task B24).

usage: learner_search.py --tag NAME [--learners band_cfg,band_pool,regime_map,lessons,mover_use,chain] [--pairs 12]
                         [--memo-pairs 4] [--seed 1] [--min-hist 8] [--min-ram 2.5]

Each learner (engine.learners) is run through the learning-delta harness on PAST-ONLY pairs: the learning window W is played
(Run 1), the learner consults every revealed window that ENDED before W began plus W itself, and the resulting state is judged on
(i) the same real year under a fresh disguise (Run 2), (ii) the no-learning and random-relabel controls, and (iii) a later
transfer year B that starts at least three years after W ended. The verdict that counts is the transfer delta with a 95% CI above
zero (engine.learning_delta.headline_transfer); same-year gains and the memorisation gap are reported next to it.
Nothing is adopted by this script; it writes state/research/learners/<tag>/{summary.json,report.md,pairs/*.json}.
Every finished pair is checkpointed (a pair written by different code is redone, never mixed). Launch detached:
  Start-Process .venv/Scripts/python.exe -ArgumentList '-u','scripts/learner_search.py','--tag','ls1' -WindowStyle Hidden
Only revealed windows are read; the sealed windows in state/livesim are never touched."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd

from engine import config as K, provenance
from engine import learners as LR
from engine import learning_delta as L
from engine.improve import log_experiment

OUT = K.STATE / "research" / "learners"
DESIGN_METRIC = {"band_cfg": "in_band", "band_pool": "in_band", "regime_map": "in_band", "lessons": "mean_week",
                 "mover_use": "in_band", "chain": "in_band"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def choose_past_pairs(pool, n_pairs, seed, gap_years=3, min_hist=8):
    """(W, B) pairs where B starts >= gap_years after W ended and W has >= min_hist revealed windows behind it (so a learner has
    years to learn from). W is the latest still-unused window that ends early enough: the learner then sees as much history as a
    live system would have had just before B. Each W and each B is used once."""
    ps = sorted(pool, key=lambda p: (p["real_start"], p["id"]))
    ends = {p["id"]: p["real_end"] for p in ps}

    def hist(p):
        return sum(1 for q in ps if q["real_end"] < p["real_start"])

    Bs = [p for p in ps if any(q["real_end"].year <= p["real_start"].year - gap_years and hist(q) >= min_hist for q in ps)]
    rng = np.random.default_rng(seed)
    used, pairs = set(), []
    for i in rng.permutation(len(Bs)):
        if len(pairs) >= n_pairs:
            break
        B = Bs[int(i)]
        Ws = [q for q in ps if q["id"] not in used and q["id"] != B["id"] and q["real_end"].year <= B["real_start"].year - gap_years
              and hist(q) >= min_hist]
        if not Ws:
            continue
        W = max(Ws, key=lambda q: (ends[q["id"]], q["id"]))
        used.add(W["id"])
        pairs.append((W, B))
    return pairs


def make_book(pool, out):
    """A Book whose panels are built lazily from the archived windows and cached on disk (panels/<id>.npz)."""
    book = LR.Book(cache_dir=out / "ledgers")
    (out / "panels").mkdir(parents=True, exist_ok=True)

    def loader_for(e):
        def load():
            f = out / "panels" / f"{e['id']}.npz"
            if f.exists():
                return LR.load_panel(f)
            w = L.load_window(e)
            p = LR.build_panel(w.snaps, w.closes, e["id"], e["real_start"], e["real_end"], w.bps)
            LR.save_panel(p, f)
            return p
        return load

    for e in pool:
        book.register(e["id"], e["real_start"], e["real_end"], loader=loader_for(e))
    return book


def learner_selfcheck(seed=0):
    """Planted worlds through the same harness: a generalisable law must be learned and must transfer; a world with nothing to learn
    (null) and a world whose law flips sign with the year (flip) must leave the learner idle. Returns a dict and a valid flag."""
    res = {}
    loose = LR.GateParams(guard_worst5_tol=-0.2)             # a hotter pool has a worse worst week by construction in the band world
    for world, name, want, kw in (("band", "band_pool", "learns", dict(n_pairs=8, n_weeks=40, gp=loose)),
                                  ("lesson", "lessons", "learns", dict(n_pairs=4)), ("null", "band_pool", "idle", dict(n_pairs=4)),
                                  ("flip", "lessons", "idle", dict(n_pairs=4))):
        recs, lrn, _ = LR.planted_pairs(world, name, n_hist=10, seed=seed, **kw)
        agg = L.aggregate(recs, seed)
        metric = DESIGN_METRIC[name]
        tr = agg["metrics"][metric]["transfer"]
        changed = sum(r["s0_fp"] != r["s1_fp"] for r in recs)
        ok = (tr["lo"] > 0 and changed == len(recs)) if want == "learns" else (changed <= 1 and abs(tr["mean"]) < 0.02)
        res[f"{world}/{name}"] = {"want": want, "ok": bool(ok), "transfer": tr["mean"], "lo": tr["lo"], "hi": tr["hi"], "pairs_changed": changed}
        log(f"learner selfcheck {world}/{name}: {res[f'{world}/{name}']}")
    res["valid"] = all(v["ok"] for v in res.values())
    return res


def fmt(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:+.5f}"


def learner_report(name, recs, agg, design):
    h = agg["headline"]
    L_ = [f"## {name}", ""]
    t = h["metrics"][design]
    L_.append(f"Design metric **{design}**: past-only transfer {fmt(t['mean'])} [{fmt(t['lo'])}, {fmt(t['hi'])}] over {t['n']} pairs -> "
              f"**{'PASS' if t['lo'] > 0 else 'no pass'}**; headline verdict on weekly mean: {h['verdict']['label']}")
    L_ += ["", "| metric | transfer delta (past-only) | same-year delta | gap (memorisation) | noise ctrl |", "|---|---|---|---|---|"]
    for m in L.METRICS:
        r, ht = agg["metrics"][m], h["metrics"][m]
        c = lambda d: f"{fmt(d['mean'])} [{fmt(d['lo'])}, {fmt(d['hi'])}]"
        L_.append(f"| {m} | {fmt(ht['mean'])} [{fmt(ht['lo'])}, {fmt(ht['hi'])}] | {c(r['same'])} | {c(r['gap'])} | {fmt(r['noise']['mean'])} |")
    changed = [r for r in recs if r["s0_fp"] != r["s1_fp"]]
    L_ += ["", f"State changed in {len(changed)} of {len(recs)} pairs. Same-year verdict (weekly mean): {agg['verdict']['label']} - "
           f"{agg['verdict']['why']}. In-band verdict: {agg['verdict_in_band']['label']}.", "", "What it adopted:"]
    for r in recs:
        note = "; ".join(x for x in r.get("learner_notes", []) if "adopt" in x or "kept" in x or "map" in x) or "nothing"
        L_.append(f"- {r['window']} -> {r['transfer']}: {note[:230]}")
    return "\n".join(L_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--learners", default="band_cfg,band_pool,regime_map,lessons,mover_use,chain")
    ap.add_argument("--pairs", type=int, default=12)
    ap.add_argument("--memo-pairs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--min-hist", type=int, default=8)
    ap.add_argument("--min-ram", type=float, default=2.5)
    a = ap.parse_args()
    out = OUT / a.tag
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    (out / "pid.txt").write_text(str(os.getpid()))
    if not L.wait_for_ram(a.min_ram, log_fn=log):
        log("free RAM stayed under the floor for 20 minutes: not running")
        (out / "ABORTED.txt").write_text("RAM")
        return 2
    import scripts.learning_delta as LDS                        # state_for / read_state0 (S0 is past-only by construction)
    code = provenance.code_stamp()
    log(f"learner search '{a.tag}' seed {a.seed}, code {code.get('code_hash')}")
    sc = L.harness_selfcheck(a.seed, log_fn=log)
    lsc = learner_selfcheck(a.seed)
    (out / "selfcheck.json").write_text(json.dumps({"harness": sc, "learners": lsc}, indent=1, default=str))
    if not (sc["valid"] and lsc["valid"]):
        log("SELF-CHECK FAILED: verdicts will be marked VOID")
    pool = L.revealed_pool()
    pairs = choose_past_pairs(pool, a.pairs, a.seed, min_hist=a.min_hist)
    log(f"{len(pool)} revealed windows; {len(pairs)} past-only pairs: " + ", ".join(f"{w['id']}->{b['id']}" for w, b in pairs))
    st, bank = LDS.read_state0()
    book = make_book(pool, out)
    player = LR.PolicyReplayPlayer()
    summary = {"tag": a.tag, "seed": a.seed, "pairs": [{"window": w["id"], "transfer": b["id"]} for w, b in pairs],
               "selfcheck": {"harness_valid": sc["valid"], "learners_valid": lsc["valid"]}, "learners": {}, "provenance":
               provenance.stamp({"cfg": st["cfg"], "meta": st["meta"]}, seed=a.seed)}
    all_recs = {}
    for name in [x for x in a.learners.split(",") if x]:
        learner = LR.make_learner(name, book)
        recs = []
        for we, be in pairs:
            f = out / "pairs" / f"{name}__{we['id']}__{be['id']}.json"
            if f.exists():
                r = json.loads(f.read_text())
                if r.get("_code") == code.get("code_hash"):
                    recs.append(r)
                    continue
            if not L.wait_for_ram(a.min_ram, log_fn=log):
                log("RAM never freed up; stopping with what is checkpointed")
                break
            W, B = L.load_window(we), L.load_window(be)
            r = L.run_pair(W, B, player, learner, LDS.state_for(W, st, bank), L.derive_seed(a.seed, name), log_fn=log)
            r["learner_notes"] = list(getattr(learner, "last_notes", []) or [])
            r["history_windows"] = len(getattr(learner, "last_ids", []) or [])
            r["_code"] = code.get("code_hash")
            f.write_text(json.dumps(r, default=str))
            recs.append(r)
            book.drop_panels(keep={we["id"]})
            del W, B
        if not recs:
            continue
        all_recs[name] = recs
        agg = L.aggregate(recs, a.seed)
        summary["learners"][name] = {"aggregate": agg, "design_metric": DESIGN_METRIC[name], "n": len(recs)}
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        log(f"{name}: transfer {DESIGN_METRIC[name]} {agg['headline']['metrics'][DESIGN_METRIC[name]]['mean']:+.5f} "
            f"[{agg['headline']['metrics'][DESIGN_METRIC[name]]['lo']:+.5f}, {agg['headline']['metrics'][DESIGN_METRIC[name]]['hi']:+.5f}]")
    # memoriser control on real windows: the harness must be able to see memorisation with these very pairs
    mrecs = []
    for we, be in pairs[:a.memo_pairs]:
        f = out / "pairs" / f"memo__{we['id']}__{be['id']}.json"
        if f.exists() and json.loads(f.read_text()).get("_code") == code.get("code_hash"):
            mrecs.append(json.loads(f.read_text()))
            continue
        W, B = L.load_window(we), L.load_window(be)
        r = L.run_pair(W, B, L.MemorisingReplayPlayer(), L.MemoriserLearner(), LDS.state_for(W, st, bank), L.derive_seed(a.seed, "memo"), log_fn=log)
        r["_code"] = code.get("code_hash")
        f.write_text(json.dumps(r, default=str))
        mrecs.append(r)
    magg = L.aggregate(mrecs, a.seed) if mrecs else None
    memo_ok = None
    if magg:
        mm = magg["metrics"][L.PRIMARY]
        memo_ok = bool(mm["same"]["mean"] > 3 * max(abs(mm["transfer"]["mean"]), 1e-4) and mm["same"]["mean"] > 0)
    summary["memoriser_control"] = {"sees_memorisation": memo_ok, "same": None if not magg else magg["metrics"][L.PRIMARY]["same"]["mean"],
                                    "transfer": None if not magg else magg["metrics"][L.PRIMARY]["transfer"]["mean"]}
    valid = bool(sc["valid"] and lsc["valid"] and memo_ok is not False)
    summary["valid"] = valid
    lines = [f"# Generalising learners - {a.tag}", "",
             f"Validity: harness self-check {'VALID' if sc['valid'] else 'INVALID'}; learner self-check (planted worlds) "
             f"{'VALID' if lsc['valid'] else 'INVALID'}; real-window memoriser control "
             f"{'seen' if memo_ok else 'NOT seen' if memo_ok is False else 'not run'}. {'' if valid else 'NO VERDICT BELOW MAY BE USED.'}",
             "", "Pairs (learning window -> later transfer window): " + ", ".join(f"{p['window']}->{p['transfer']}" for p in summary["pairs"]), "",
             "A learner PASSES only if the past-only transfer delta of its design metric has a 95% CI above zero. Eight metrics x six "
             "learners are looked at, so one lone marginal pass is suggestive, not proof.", ""]
    passed = []
    for name, recs in all_recs.items():
        agg = summary["learners"][name]["aggregate"]
        lines += [learner_report(name, recs, agg, DESIGN_METRIC[name]), ""]
        if agg["headline"]["metrics"][DESIGN_METRIC[name]]["lo"] > 0:
            passed.append(name)
    lines += ["## Verdict", "", f"Passing learners: {', '.join(passed) if passed else 'none'}", "",
              "## What this does not prove", "",
              "- Ledgers are close-to-close, held-nothing weekly picks; the harness plays the real adaptive replay, so a ledger gain can vanish there.",
              "- Pairs share history (W is the latest window before B), so pair deltas are not independent and the CIs are optimistic.",
              "- Snapshots were produced by a model trained before each window; learners cannot fix that model, only how its output is used.",
              "- The price panel is survivor-only (memory: weekly7-price-panel-is-survivor-only): high-volatility pools look better than they were."]
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log_experiment({"event": "learner_search", "tag": a.tag, "learners": list(all_recs), "passed": passed, "valid": valid,
                    "window_ids": [p["window"] for p in summary["pairs"]], "outcome": "continue_testing",
                    "reason": f"passing: {passed or 'none'}"}, cfg={"cfg": st["cfg"], "meta": st["meta"]}, seed=a.seed)
    log(f"done: passing learners {passed or 'none'}; valid {valid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
