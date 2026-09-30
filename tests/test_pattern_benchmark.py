"""F19 real-vs-noise benchmark (canon C70-C75): the benchmark itself must be right before its scores mean anything. Truth labels are
exact; a perfect oracle scores 100% / 0; a random classifier scores what chance predicts; the detectability floor is monotone in the
effect; held-out seeds cannot reach tuning code; the sealed answer key (C72) exists before the answers, is never read by the system
and fails closed when missing, altered or late; a disguised world (C75 3M) gives the same screen. Synthetic, small, < 90 s."""
from __future__ import annotations

import builtins
import io
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.research import pattern_benchmark as PB                            # noqa: E402

warnings.filterwarnings("ignore")
SMALL_NOISE = (("null", 6), ("identity_null", 2), ("mt_winner", 3), ("autocorr_trap", 2), ("regime_corr", 2), ("vol_corr", 1),
               ("sample_size", 1), ("threshold_illusion", 2), ("near_pattern", 2), ("coincidence", 2), ("early_decay", 1),
               ("reversal", 1), ("delayed_coincidence", 1), ("interaction_decoy", 1), ("xor_trap", 1), ("selection_bias", 1),
               ("survivor_bias", 1), ("strong_nontransferable", 1), ("adversarial_near", 2), ("proxy", 2), ("fluke", 1), ("leak", 2))
SMALL = PB.BenchConfig(n_names=20, n_dates=90, frame_weeks=70, first_look=78, look_every=12, noise_counts=SMALL_NOISE,
                       bands=(("obvious", 0.9), ("faint", 0.1)), oracle_draws=8, mt_pool=6, null_world_share=0.0, tiers=(3,))


@pytest.fixture(scope="module")
def world():
    return PB.make_world(11, SMALL)


# ============================================================================================================ truth labels
def test_truth_labels_are_exact(world):
    key, F = world.key, world.frame
    P = pd.DataFrame(key["patterns"])
    real = P[P["label"] == PB.REAL]
    assert len(real) == len(SMALL.real_kinds) * len(SMALL.bands)
    assert set(real["kind"]) == set(PB.REAL_KINDS)
    counts = SMALL.counts_for(key["tier"])
    for k, n in counts.items():
        assert (P["kind"] == k).sum() == n, k
    planted = [c for c in F.columns if c.startswith(PB.PREFIX)]
    assert sorted(planted) == sorted(c for c in key["columns"] if c.startswith(PB.PREFIX)) and len(set(planted)) == len(planted)
    assert sorted(c for c in key["columns"] if not c.startswith(PB.PREFIX)) == sorted(f for f in PB.scan_universe(F.columns) if not f.startswith(PB.PREFIX))
    assert all(c in F.columns for p in key["patterns"] if p["kind"] != "base_null" for c in p["columns"])
    # every planted column belongs to exactly one pattern; interaction components are PART, the interaction itself has no column
    owners = {}
    for p in key["patterns"]:
        for c in p["columns"]:
            owners.setdefault(c, []).append(p["pid"])
    assert all(len(v) == 1 for c, v in owners.items())
    assert all(not p["columns"] for p in key["patterns"] if p["kind"] in ("interactive", "xor"))
    # the learner world carries no truth: no era / tier / probability column, no attrs, names reveal nothing
    assert not ({"era", "tier", "p_true", "truth", "label"} & set(F.columns)) and not F.attrs
    assert all(c[len(PB.PREFIX):].isdigit() for c in planted)
    kinds_by_name = [key["columns"][c] for c in sorted(planted)]
    assert kinds_by_name != sorted(kinds_by_name)                       # shuffled: name order is not pattern order
    assert {p["status"] for p in real.to_dict("records")} <= {PB.DETECTABLE, PB.UNDETECTABLE, PB.NOT_REPRESENTABLE}


def test_planted_effects_are_where_the_truth_says(world):
    """Planted-defect check on the generator: a strong linear REAL column carries a clear per-date effect in the planted direction; a
    leak column carries a big one (it is built from the outcome); a pure null carries none."""
    key, F = world.key, world.frame
    T, N = SMALL.n_dates, SMALL.n_names
    y = F["touch"].to_numpy(float).reshape(T, N) > 0.5

    def eff(c, s=1.0):
        return float(np.nanmean(PB.date_aucs(s * F[c].to_numpy(float).reshape(T, N), y[None])[0] - 0.5))
    lin = next(p for p in key["patterns"] if p["kind"] == "linear" and p["band"] == "obvious")
    assert eff(lin["columns"][0], lin["sign"]) > 0.05
    leak = [p for p in key["patterns"] if p["kind"] == "leak"]
    assert max(eff(p["columns"][0]) for p in leak) > 0.1
    nulls = [eff(p["columns"][0]) for p in key["patterns"] if p["kind"] == "null"]
    assert abs(np.mean(nulls)) < 0.03


def test_null_world_has_no_genuine_pattern():
    w = PB.make_world(2, PB.BenchConfig(**{**SMALL.__dict__, "null_world_share": 0.999}))
    P = pd.DataFrame(w.key["patterns"])
    assert w.key["tier"] == 0 and not (P["label"] == PB.REAL).any() and not (P["label"] == PB.PART).any()
    assert (P["label"] == PB.NOISE).sum() >= sum(n for _, n in SMALL_NOISE)
    rows, s = PB.score_world(w.key, PB.oracle_answers(w.key))
    assert s["real_total"] == 0 and s["false_positives"] == 0


def test_eras_and_year_are_hidden_and_era_changes_base_rate():
    cfg = PB.BenchConfig(**{**SMALL.__dict__, "single_era_share": 1.0})
    rates = {}
    for s in range(12):
        w = PB.make_world(s, cfg)
        rates.setdefault(w.key["eras"][0][1], []).append(w.key["base_rate"])
        assert str(w.key["year"]) not in "".join(c for c in w.frame.columns)
    means = {e: np.mean(v) for e, v in rates.items()}
    if "calm" in means and "crisis" in means:
        assert means["crisis"] > means["calm"] + 0.05
    assert len(rates) >= 3


# ============================================================================================================ scoring references
def test_perfect_oracle_scores_all_real_and_no_false_positive(world):
    rows, s = PB.score_world(world.key, PB.oracle_answers(world.key))
    assert s["real_right"] == s["real_detectable"] > 0
    assert s["false_positives"] == 0 and s["noise_rejected"] == s["noise_total"]
    R = pd.DataFrame(rows)
    assert (R.loc[(R["label"] == PB.REAL) & (R["status"] == PB.DETECTABLE), "credit"] == 1.0).all()


def test_random_classifier_scores_what_chance_predicts():
    q, hits, n_noise, fp, n_det, tp = 0.3, 0, 0, 0, 0, 0
    for s in range(4):
        w = PB.make_world(100 + s, SMALL)
        rows, summ = PB.score_world(w.key, PB.random_answers(w.key, q, seed=s))
        n_noise += summ["noise_total"]
        fp += summ["false_positives"]
        R = pd.DataFrame(rows)
        single = R[(R["label"] == PB.REAL) & ~R["kind"].isin(["interactive", "xor"]) & (R["status"] == PB.DETECTABLE)]
        n_det += len(single)
        tp += int(single["promoted"].sum())
    se = math.sqrt(q * (1 - q) / n_noise)
    assert abs(fp / n_noise - q) < 4 * se + 0.02, (fp, n_noise)
    assert n_det == 0 or abs(tp / n_det - q) < 4 * math.sqrt(q * (1 - q) / n_det) + 0.05


def test_score_catches_a_promoted_noise_and_names_its_kind(world):
    """Planted defect: an answer sheet that promotes the leak columns must show them as false positives of kind 'leak'; an empty
    answer sheet promotes nothing and misses every detectable real pattern."""
    ans = PB.oracle_answers(world.key)
    leaks = [p["columns"][0] for p in world.key["patterns"] if p["kind"] == "leak"]
    for c in leaks:
        ans["candidates"][c]["final"], ans["candidates"][c]["promoted_look"] = "PROMOTED", 0
    rows, s = PB.score_world(world.key, ans)
    assert s["false_positives"] == len(leaks) and s["fp_by_kind"] == {"leak": len(leaks)}
    empty = {"world_id": world.key["world_id"], "candidates": {}, "looks": []}
    rows, s = PB.score_world(world.key, empty)
    assert s["false_positives"] == 0 and s["real_right"] == 0 and len(s["misses"]) == s["real_detectable"]
    assert PB.aggregate([], []) == {}


def test_partial_credit_for_a_promoted_proxy(world):
    ans = PB.oracle_answers(world.key)
    prox = next((p for p in world.key["patterns"] if p["kind"] == "proxy" and p.get("parent")), None)
    if prox is None:
        pytest.skip("no proxy with a parent in this world")
    par = next(p for p in world.key["patterns"] if p["pid"] == prox["parent"])
    for c in par["columns"]:
        ans["candidates"][c]["final"], ans["candidates"][c]["promoted_look"] = "RETIRED", None
    ans["candidates"][prox["columns"][0]]["final"] = "PROMOTED"
    rows, _ = PB.score_world(world.key, ans)
    R = pd.DataFrame(rows).set_index("pid")
    assert R.loc[par["pid"], "credit"] == 0.5 and "proxy" in R.loc[par["pid"], "near_miss"]
    assert bool(R.loc[prox["pid"], "promoted"]) and "false positive" in R.loc[prox["pid"], "near_miss"]


# ============================================================================================================ detectability floor
def test_detectability_is_monotone_in_effect_size():
    rng = np.random.default_rng(0)
    T, N, draws = 120, 30, 16
    x = rng.normal(0, 1, (T, N))
    u = rng.random((draws, T, N))
    wins = [(0, 40, 118)]
    crit = PB.oracle_crit()
    powers, effects = [], []
    for beta in (0.0, 0.05, 0.12, 0.25, 0.5):
        p = 1 / (1 + np.exp(-(-1.8 + beta * x)))
        pw, ef = PB.power_curve(x, u < p[None], np.ones(T, bool), wins, crit)[0]
        powers.append(pw)
        effects.append(ef)
    assert powers == sorted(powers) and effects == sorted(effects), (powers, effects)
    assert powers[0] <= 0.2 and powers[-1] == 1.0
    assert PB.oracle_power(x, u < 0.2, np.zeros(T, bool), (40, 118), crit) == (0.0, 0.0)    # no date where the effect exists


def test_date_aucs_matches_the_gate_measure_and_ignores_missing():
    from engine.research import evidence as EV
    rng = np.random.default_rng(3)
    T, N = 12, 15
    s = rng.normal(0, 1, (T, N))
    s[rng.random((T, N)) < 0.2] = np.nan
    y = rng.random((T, N)) < 0.3
    idx = pd.MultiIndex.from_product([pd.date_range("2020-01-03", periods=T, freq="W-FRI"), [f"T{i}" for i in range(N)]])
    ref = EV.per_date_effect(pd.Series(s.reshape(-1), index=idx), pd.Series(y.reshape(-1).astype(float), index=idx), 8)
    mine = PB.date_aucs(s, y) - 0.5
    for d, v in ref.items():
        assert abs(mine[list(idx.levels[0]).index(d)] - v) < 1e-9


# ============================================================================================================ held-out seeds
def test_heldout_seeds_cannot_be_selected_by_tuning_code(tmp_path):
    held, dev = PB.heldout_seeds(300), PB.development_seeds(300)
    assert set(held).isdisjoint(dev) and sorted(held + dev) == list(range(300)) and 70 <= len(held) <= 130
    assert PB.tuning_seeds(dev[:20]) == dev[:20]
    with pytest.raises(PB.HeldOutAccess):
        PB.tuning_seeds(dev[:3] + held[:1])
    rec = PB.write_heldout_record(tmp_path / "h.json", 300)
    assert rec["heldout"] == held
    body = json.loads((tmp_path / "h.json").read_text())
    body["heldout"] = body["heldout"][1:]
    (tmp_path / "h.json").write_text(json.dumps(body))
    with pytest.raises(PB.HeldOutAccess):
        PB.write_heldout_record(tmp_path / "h.json", 300)
    assert PB.split_of(held[0]) == "heldout" and PB.split_of(dev[0]) == "development"


# ============================================================================================================ C72 sealed answer key
def _sealed(tmp_path, world):
    man = PB.Manifest(tmp_path / "manifest.jsonl")
    PB.seal_key(world.key, tmp_path, man)
    started = time.time_ns()
    ans = PB.oracle_answers(world.key)
    time.sleep(0.01)
    PB.save_answers(ans, tmp_path, man, started)
    return man


def test_key_opens_only_when_sealed_first_and_untouched(tmp_path, world):
    man = _sealed(tmp_path, world)
    key, ans = PB.open_key(tmp_path, world.key["world_id"], man)
    assert key["world_id"] == world.key["world_id"] and man.verify_chain() == []
    with pytest.raises(PB.SealError):
        PB.seal_key(world.key, tmp_path, man)                                   # a key is written once


def test_open_key_fails_closed_when_missing_altered_or_late(tmp_path, world):
    wid = world.key["world_id"]
    man = PB.Manifest(tmp_path / "m1.jsonl")
    with pytest.raises(PB.SealError, match="missing"):
        PB.open_key(tmp_path, wid, man)                                         # nothing sealed
    man = _sealed(tmp_path / "a", world)
    kp = PB.key_path(tmp_path / "a", wid)
    kp.write_bytes(kp.read_bytes().replace(b'"REAL"', b'"NOISE"', 1))           # altered key
    with pytest.raises(PB.SealError, match="altered"):
        PB.open_key(tmp_path / "a", wid, man)
    # answers saved BEFORE the key was written: the manifest order is wrong
    out = tmp_path / "b"
    man = PB.Manifest(out / "manifest.jsonl")
    started = time.time_ns()
    PB.save_answers(PB.oracle_answers(world.key), out, man, started - 10**9)
    time.sleep(0.01)
    PB.seal_key(world.key, out, man)
    with pytest.raises(PB.SealError):
        PB.open_key(out, wid, man)
    # a key rewritten after sealing (same bytes, new write) is caught by its mtime
    man = _sealed(tmp_path / "c", world)
    kp = PB.key_path(tmp_path / "c", wid)
    time.sleep(0.02)
    kp.write_bytes(kp.read_bytes())
    with pytest.raises(PB.SealError, match="rewritten"):
        PB.open_key(tmp_path / "c", wid, man)


def test_manifest_edit_breaks_the_chain(tmp_path, world):
    man = _sealed(tmp_path, world)
    lines = man.path.read_text().splitlines()
    row = json.loads(lines[0])
    row["sha256"] = "0" * 64
    lines[0] = json.dumps(row, sort_keys=True)
    man.path.write_text("\n".join(lines) + "\n")
    assert man.verify_chain()
    with pytest.raises(PB.SealError):
        PB.open_key(tmp_path, world.key["world_id"], man)


def test_the_system_never_reads_the_key(tmp_path, monkeypatch):
    """The system under test runs on a world whose key is already sealed on disk; every file it opens is recorded - the key is not
    among them, and neither the key reader nor the key path is referenced by the system's code."""
    cfg = PB.BenchConfig(**{**SMALL.__dict__, "first_look": 88, "look_every": 5, "screen_calls": 1})
    w = PB.make_world(5, cfg)
    man = PB.Manifest(tmp_path / "manifest.jsonl")
    PB.seal_key(w.key, tmp_path, man)
    kp = str(PB.key_path(tmp_path, w.key["world_id"]).resolve()).lower()
    seen = []
    real_open, real_io_open = builtins.open, io.open
    orig_rb, orig_rt = Path.read_bytes, Path.read_text

    def spy(f, *a, **k):
        seen.append(str(f))
        return real_open(f, *a, **k)

    def spy_rb(self):
        seen.append(str(self))
        return orig_rb(self)

    def spy_rt(self, *a, **k):
        seen.append(str(self))
        return orig_rt(self, *a, **k)
    monkeypatch.setattr(builtins, "open", spy)
    monkeypatch.setattr(io, "open", spy)
    monkeypatch.setattr(Path, "read_bytes", spy_rb)
    monkeypatch.setattr(Path, "read_text", spy_rt)
    ans = PB.run_system(w.frame, w.key["world_id"], 5, cfg)
    monkeypatch.undo()
    assert not any(str(Path(s).resolve()).lower() == kp for s in seen if s)
    assert not any(".key.json" in s for s in seen)
    names = set(PB.run_system.__code__.co_names) | set(PB._decision_summary.__code__.co_names)
    assert not names & {"open_key", "key_path", "seal_key", "Manifest", "make_world"}
    assert ans["candidates"] and ans["looks"] and all(k.startswith(PB.PREFIX) or k in PB.scan_universe(w.frame.columns)
                                                      for k in ans["candidates"])
    assert any(c.get("gate") for c in ans["candidates"].values())             # the real gate ran on at least one raised candidate


# ============================================================================================================ C75 3M / 3N / Firewall 10
def test_disguised_world_gives_the_same_screen(world):
    cfg = PB.BenchConfig(**{**SMALL.__dict__, "gate": False})
    a = PB.run_system(world.frame, "A", 1, cfg)
    G, back, tback = PB.disguise(world.frame, 3, shift_weeks=0)          # renamed and reordered, same calendar
    assert set(G.index.get_level_values(1)).isdisjoint(set(world.frame.index.get_level_values(1)))
    assert not G.index.get_level_values(1)[:5].equals(world.frame.index.get_level_values(1)[:5].map(lambda t: t))
    b = PB.run_system(G, "B", 1, cfg)
    inv = PB.answers_invariance(a, b, back)
    assert inv["invariant"], inv["diffs"][:5]
    # the invariance check itself can fail: a planted difference is reported
    b2 = json.loads(json.dumps(b))
    f = next(iter(b2["candidates"]))
    b2["candidates"][f]["final"] = "PROMOTED"
    assert not PB.answers_invariance(a, b2, back)["invariant"]


def test_mutations_are_valid_and_change_the_world():
    for name in PB.MUTATIONS:
        m = PB.mutate(SMALL if name not in ("fewer_names", "more_names") else PB.BenchConfig(), name)
        assert m.validate() == [] and m != SMALL
    with pytest.raises(KeyError):
        PB.mutate(SMALL, "nope")
    assert PB.full_scale_ok(PB.BenchConfig()) == [] and PB.full_scale_ok(SMALL)


def test_freeze_record_detects_a_changed_threshold():
    rec = PB.freeze_record(SMALL)
    PB.assert_frozen(rec, SMALL)
    with pytest.raises(PB.HeldOutAccess):
        PB.assert_frozen(rec, PB.BenchConfig(**{**SMALL.__dict__, "power_floor": 0.4}))


def test_aggregate_and_report_on_reference_sheets():
    rows, summ = [], []
    for s in range(3):
        w = PB.make_world(200 + s, SMALL)
        r, m = PB.score_world(w.key, PB.random_answers(w.key, 0.2, s) if s else PB.oracle_answers(w.key))
        rows += r
        summ.append(m)
    T = PB.aggregate(rows, summ)
    assert {"total", "era", "band", "noise", "closeness", "calibration", "failures"} <= set(T)
    md = PB.report_markdown(T, summ, {"note": "test"})
    assert "C72 score" in md and "| set |" in md
    lc = PB.learning_curve(summ, points=(1, 2))
    assert list(lc["history_worlds"]) == [1, 2]


def test_evidence_on_the_needed_columns_is_identical(world):
    """run_system hands evidence.assemble only the columns it reads (speed); the bundle and the verdict must be exactly the same."""
    from engine.research import evidence as EV
    from engine.research import loop as LP
    from engine.research import replication as RP
    F = world.frame
    dates = F.index.get_level_values(0).unique()
    now = dates[-2]
    M = F[(pd.to_datetime(F["end"]) < now)]
    planted = [c for c in F.columns if c.startswith(PB.PREFIX)]
    lin = next(p for p in world.key["patterns"] if p["kind"] == "linear" and p["band"] == "obvious")["columns"][0]
    with PB.registered(planted), PB.corpus_once():
        for f in (lin, "lv20"):
            spec = EV.FindingSpec("D_" + f, f, 1.0, "VOLATILITY", n_tests_searched=5, has_falsifier=True, seed=1)
            out = []
            for G in (M, M[PB.evidence_columns(f, M.columns)]):
                b = EV.assemble(G, spec, now, code_hash="c", data_hash="d", created_real="2026-09-30T00:00:00+00:00",
                                ledger=RP.ReplicationLedger(), look=1, plan=LP.REGATE_PLAN)
                rep = EV.gate([b], now, "c", looks={spec.subject_id: 1}, plan=LP.REGATE_PLAN)
                out.append((json.dumps(b.summary(), sort_keys=True, default=str), EV.verdicts(rep), EV.blocking(rep, spec.subject_id)))
            assert out[0] == out[1], f
