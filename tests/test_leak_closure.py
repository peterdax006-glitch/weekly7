"""F06 leak closure (C69 ledger W-10 / W-11; canon C55, C56, C58, C64; C66 sections 30-31). Synthetic data only; livesim.DIR is
redirected to tmp dirs, no sealed window or cache is opened.  Status: IMPLEMENTED - NOT VALIDATED.

Part 1  the training CALL is gated (channel 4): a basis search refuses any window that had not ended before the first real day
        the basis may be played; the loop splits the archive there and starts from a past-only incumbent; each link flipped in
        the source moves the computed verdict to LEAK; the design proof fails when the gate is removed.
Part 2  the untrained start is data-free in the meta too (NEUTRAL_META); old plays on the tuned META_DEFAULT are quarantined and
        keep the verdict at QUARANTINED; only a state without them is FIXED.
Part 3  channel 6 lookup part: verdict_year_lookup / combine_fingerprint_and_lookup flip on each planted defect and on a failed
        control; the real probes (curator, research firewall, memory bank, import closure) run and come out CLEAN.
Part 4  W-11: one real-shaped synthetic window replayed TWICE through the C64 test path on one shared curator store, with
        research filed under that year in between: nothing filed under the replayed year is released, no hindsight label reaches
        the trader side, every released memory had matured before its day, and trader_view refuses planted year/date content."""
import importlib.util
import json
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine import basis_search as B, blind_gates as BG, leak_audit as LA, livesim, objective as O
from engine.learning import test_path as TP
from engine.learning import trader_view as TV
from engine.learning.core import FirewallBreach
from engine.learning.curator import Curator

ROOT = Path(__file__).resolve().parent.parent
SRC = LA._read_sources()
PROOFS = LA.run_proofs()
T = pd.Timestamp


@pytest.fixture(scope="module")
def L(tmp_path_factory):
    """The loop module imported with livesim.DIR pointed at a temp dir (nothing real can be sealed, read or written)."""
    home = tmp_path_factory.mktemp("livesim")
    real, livesim.DIR = livesim.DIR, home
    argv, sys.argv = sys.argv, ["livesim_loop2.py"]
    try:
        spec = importlib.util.spec_from_file_location("livesim_loop2_f06", ROOT / "scripts" / "livesim_loop2.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        assert m.DIR == home and not list(home.iterdir())
        yield m
    finally:
        sys.argv = argv
        livesim.DIR = real


@pytest.fixture(scope="module")
def S():
    """The audit runner (scripts/leak_audit.py): the lookup probes live there because they import the curator/research."""
    spec = importlib.util.spec_from_file_location("leak_audit_runner_f06", ROOT / "scripts" / "leak_audit.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SPANS = {"w80": (T("1980-01-01"), T("1980-12-31")), "w95": (T("1995-01-01"), T("1995-12-31")),
         "w00": (T("2000-01-03"), T("2000-12-29")), "w05": (T("2005-01-03"), T("2005-12-30")), "w19": (T("2019-06-03"), T("2020-05-29"))}


def span(i):
    return SPANS[i]


def facts_with(old, new, key="loop2"):
    text = SRC[key]
    assert text.count(old) == 1, f"{old!r} occurs {text.count(old)}x"
    return LA.default_path_facts(sources={key: text.replace(old, new)})


def neutral_state(meta=None):
    """A state as the F06 loop writes it: w01 untrained on the neutral cfg + NEUTRAL_META; v1 trained on w01 for first use 1990."""
    f = LA.default_path_facts()
    nm = LA.default_neutral_meta(f["meta_default_literal"])
    cfg = LA.neutral_default_cfg(f["loop2_cfg_space"])
    st = {"windows": [{"run_id": "w01", "basis_version": 0, "untrained_basis": True, "prior_cfg": cfg, "meta": meta or nm}],
          "lineage": [{"version": 1, "cfg": {"k": 2}, "meta": nm, "trained_on": [["w01", "1980-01-01", "1980-12-31"]], "used_from": "1990-01-01"}]}
    return st, f


def v4(facts=None, state=None, proofs=None):
    f = facts or LA.default_path_facts()
    st = state if state is not None else neutral_state()[0]
    sc = LA.lineage_state_check(st, f["loop2_cfg_space"], f["meta_default_literal"])
    return LA.verdict_learned_state(f, proofs or PROOFS, sc)


# =====================================================================================================================
# Part 1 - the training call is gated
# =====================================================================================================================
def test_split_keeps_only_windows_that_ended_before_first_use_and_refuses_the_rest():
    kept, refused = LA.split_training_windows([{"id": i} for i in ("w80", "w95", "w00", "w19", "ghost")], "2000-12-29", span)
    assert [w["id"] for w in kept] == ["w80", "w95"]
    why = {r["id"]: r["why"] for r in refused}
    assert set(why) == {"w00", "w19", "ghost"}                                    # ends ON first use, after it, unknown span
    assert "on/after" in why["w00"] and "cannot be resolved" in why["ghost"]
    kept2, _ = LA.split_training_windows(["w80", "w00"], "2000-12-30", span)       # the day after its end: now it is past
    assert kept2 == ["w80", "w00"]


def test_refuse_late_training_raises_on_a_planted_future_window_and_passes_a_clean_set():
    with pytest.raises(LA.LateTrainingWindow, match="had not ended"):
        LA.refuse_late_training(["w80", "w19"], "2005-01-03", span)
    rec = LA.refuse_late_training(["w80", "w95", "w00"], "2005-01-03", span)
    assert rec == {"n_windows": 3, "used_from": "2005-01-03", "max_real_end": "2000-12-29"}
    assert issubclass(LA.LateTrainingWindow, FirewallBreach)


def test_training_gate_empty_and_missing_date_cases():
    assert LA.split_training_windows([], "2000-01-01", span) == ([], [])
    assert LA.refuse_late_training([], "2000-01-01", span)["n_windows"] == 0
    with pytest.raises(LA.LateTrainingWindow, match="first-use date"):
        LA.split_training_windows(["w80"], None, span)                            # no date is a refusal, never "everything"


def synth_windows(ids):
    return [{"id": i, "closes": pd.DataFrame({"a": [1.0, 2.0]}, index=pd.to_datetime(["2190-01-02", "2190-02-02"]))} for i in ids]


def planted_eval(w, cfg, meta):
    rng = np.random.default_rng(zlib.crc32(w["id"].encode()))
    return O.week_row(0.005 + 0.08 * cfg["k"] ** -0.5 * rng.standard_normal(52) * 0.45)


def test_the_loops_train_basis_call_itself_refuses_a_planted_future_window(L):
    seen = []

    def ev(w, c, m):
        seen.append(w["id"])
        return planted_eval(w, c, m)
    with pytest.raises(LA.LateTrainingWindow):
        L.train_basis(synth_windows(["w80", "w95", "w19"]), L.NEUTRAL_CFG, L.NEUTRAL_META, seed=1, evaluate=ev,
                      used_from="2005-01-03", span=span)
    assert seen == []                                                             # refused before a single replay
    r = L.train_basis(synth_windows(["w80", "w95", "w00"]), L.NEUTRAL_CFG, L.NEUTRAL_META, seed=1, evaluate=ev,
                      used_from="2005-01-03", span=span)
    assert r.n_windows == 3 and set(seen) <= {"w80", "w95", "w00"}


def test_main_seals_the_next_round_splits_at_its_first_day_and_starts_from_a_past_only_incumbent():
    src = SRC["loop2"]
    order = ["ANTI-CHEAT GATE FAILED", 'wins = [w for w in loaded if not w["thin"]]', "livesim.SealedYear(r)\n        used_from",
             "wins, late = LA.split_training_windows(wins, used_from, real_span)", "base = lineage.basis_for(used_from)",
             "res = train_basis(wins, base_cfg, base_meta", 'st["lineage"][-1]["used_from"]', 'reveal_round([w["']
    pos = [src.index(x) for x in order]
    assert pos == sorted(pos), dict(zip(order, pos))
    f = LA.default_path_facts()
    assert f["loop2_every_train_call_passes_used_from"] is True and f["loop2_train_basis_refuses_late_windows"] is True
    assert f["loop2_main_splits_training_set"] is True and f["loop2_train_call_uses_global_basis"] is False


def test_channel4_verdict_flips_when_any_link_of_the_training_gate_is_removed():
    call = "res = train_basis(wins, base_cfg, base_meta, seed=1000 * st[\"version\"] + rnd, used_from=used_from)"
    assert v4().status in (LA.FIXED, LA.QUARANTINED)
    flips = [(call, call.replace(", used_from=used_from", "")),
             (call, call.replace(", used_from=used_from", ", used_from=None")),
             (call, call.replace("base_cfg, base_meta", 'st["cfg"], st["meta"]')),
             ("        LA.refuse_late_training(wins, used_from, span)\n", "        pass\n"),
             ("        wins, late = LA.split_training_windows(wins, used_from, real_span)\n", "        late = []\n")]
    for old, new in flips:
        v = v4(facts_with(old, new))
        assert v.status == LA.LEAK, new
        assert any("gate:" in r for r in v.reasons), v.reasons


def test_training_gate_proof_holds_and_fails_when_the_gate_admits_everything(monkeypatch):
    p = LA.prove_training_gate()
    assert p["mechanism_holds"] and p["gated_violations"] == 0 and p["new_basis_not_eligible_for_its_round"] == 0
    assert p["trained_versions"] > 0 and p["share_of_plays_on_untrained_neutral_basis"] < 0.5    # the gate lets most plays train
    real_split = LA.split_training_windows
    monkeypatch.setattr(LA, "split_training_windows", lambda w, uf, sp, key="id": (list(w), []))  # a split that refuses nothing
    broken = LA.prove_training_gate()
    assert not broken["mechanism_holds"] and not broken["planted_late_window_refused"]
    assert broken["new_basis_not_eligible_for_its_round"] > 0      # the late-trained basis can no longer be played where it was meant to
    assert broken["gated_violations"] == 0                           # ... the play-time lineage gate still holds: two independent gates
    assert v4(proofs={**PROOFS, "training_gate": broken}).status == LA.LEAK
    monkeypatch.setattr(LA, "split_training_windows", real_split)


def test_lineage_state_check_flags_a_version_trained_past_its_first_use_and_resolves_revealed_plays():
    st, f = neutral_state()
    st["lineage"][0]["trained_on"].append(["w19", "2019-06-03", "2020-05-29"])     # planted: v1 saw 2020 but was first used in 1990
    sc = LA.lineage_state_check(st, f["loop2_cfg_space"], f["meta_default_literal"])
    assert sc["used_from_violations"] and sc["used_from_violations"][0]["late"] == ["w19"]
    assert LA.verdict_learned_state(f, PROOFS, sc).status == LA.LEAK
    st, f = neutral_state()
    st["windows"].append({"run_id": "w03", "basis_version": 1, "untrained_basis": False, "prior_cfg": {"k": 2}, "meta": st["lineage"][0]["meta"],
                          "real_start": "1991-02-01"})                            # never trained on: resolved by its revealed start
    sc = LA.lineage_state_check(st, f["loop2_cfg_space"], f["meta_default_literal"])
    assert sc["unresolved"] == [] and sc["violations"] == [] and sc["trained"] == ["w03"]
    st["windows"][-1]["real_start"] = "1980-06-01"                               # played inside its own basis' training window
    sc = LA.lineage_state_check(st, f["loop2_cfg_space"], f["meta_default_literal"])
    assert sc["violations"] == [{"window": "w03", "version": 1}]
    assert LA.lineage_state_check({"windows": [], "lineage": []}, {}, {})["used_from_violations"] == []


# =====================================================================================================================
# Part 2 - the data-free meta
# =====================================================================================================================
def test_neutral_meta_is_the_middle_of_the_search_space_every_step_knob_and_valid(L):
    nm = L.NEUTRAL_META
    assert not B.validate_meta(nm)
    for k, vals in B.META_SPACE.items():
        uniq = list(dict.fromkeys(vals))
        assert nm[k] == uniq[len(uniq) // 2], k
    assert nm["adaptive_knobs"] == list(L.A.STEPS) and set(nm["adaptive_knobs"]) <= set(L.CFG_SPACE)
    tuned = LA.tuned_meta_keys(L.A.META_DEFAULT, L.A.META_DEFAULT, nm)
    assert {"adaptive_knobs", "ic_beta", "min_weeks"} <= set(tuned)               # the study's choices are gone
    assert LA.tuned_meta_keys(nm, L.A.META_DEFAULT, nm) == []
    moved = {**L.A.META_DEFAULT, "ic_beta": 7.0, "min_weeks": 1, "adaptive_knobs": ["k"]}    # different tuned values ...
    assert LA.neutral_default_meta(B.META_SPACE, moved, list(L.A.STEPS)) == nm    # ... the same neutral meta: it reads no outcome


def test_tuned_meta_keys_empty_and_searched_cases():
    assert LA.tuned_meta_keys({}, {"a": 1}, {"a": 2}) == [] and LA.tuned_meta_keys(None, {}, {}) == []
    assert LA.tuned_meta_keys({"a": 1, "b": 1}, {"a": 1, "b": 1}, {"a": 2, "b": 2}, searched=("a",)) == ["b"]


def test_plan_round_hands_untrained_windows_the_neutral_meta_and_quarantine_marks_old_tuned_plays(L):
    lin = LA.BasisLineage()
    plan = L.plan_round(["w80"], lin, span=span)
    assert plan["w80"]["untrained"] and plan["w80"]["meta"] == L.NEUTRAL_META and plan["w80"]["cfg"] == L.NEUTRAL_CFG
    state = {"windows": [{"run_id": "a", "basis_version": 0, "untrained_basis": True, "meta": dict(L.A.META_DEFAULT)},
                         {"run_id": "b", "basis_version": 0, "untrained_basis": True, "meta": dict(L.NEUTRAL_META)},
                         {"run_id": "c", "legacy": True, "meta": dict(L.A.META_DEFAULT)},
                         {"run_id": "d", "basis_version": 2, "untrained_basis": False, "meta": {**L.NEUTRAL_META, "ic_beta": 0.5}}]}
    out = L.quarantine_tuned_meta(state)
    assert [w.get("legacy") for w in out["windows"]] == [LA.TUNED_META_LEGACY, None, True, None]   # d: ic_beta was searched
    assert L.quarantine_tuned_meta(out) == out and L.quarantine_tuned_meta({"windows": []}) == {"windows": []}
    headline = [w for w in out["windows"] if not w.get("thin") and not w.get("legacy")]
    assert [w["run_id"] for w in headline] == ["b", "d"]


def test_channel4_meta_residual_fixed_only_when_no_play_ran_on_the_tuned_meta():
    st, f = neutral_state()
    assert v4(state=st).status == LA.FIXED                                        # every gate + data-free meta + clean state
    tuned, _ = neutral_state(meta=dict(f["meta_default_literal"]))
    v = v4(state=tuned)
    assert v.status == LA.QUARANTINED and "META_DEFAULT" in v.reasons[0] and v.checks["plays_on_tuned_meta"] == ["w01"]
    tuned["windows"][0]["legacy"] = LA.TUNED_META_LEGACY                          # quarantined by the loop: still labelled
    v = v4(state=tuned)
    assert v.status == LA.QUARANTINED and v.checks["plays_on_tuned_meta_quarantined"] == ["w01"]
    back = facts_with('"meta": dict(NEUTRAL_META), "version": 0', '"meta": dict(A.META_DEFAULT), "version": 0')
    assert back["loop2_plan_round_untrained_meta_is_neutral"] is False
    v = v4(facts=back, state=st)
    assert v.status == LA.QUARANTINED and "no data-free replacement" in v.reasons[0]


# =====================================================================================================================
# Part 3 - channel 6: what can be looked up by year
# =====================================================================================================================
def good_lookup():
    return {"curator": {"unmatured_releases": 0, "after_window_released": False, "replayed_year_late_first_release_after_maturity": True,
                        "rerun_prefix_mismatches": 0, "planted_year_feature_refused": True, "past_control_released": True,
                        "control_curator_ignoring_maturity_mismatches": 12, "planted_extra_changes_releases": True,
                        "hindsight_feature_refused": True},
            "research": {"same_year_refused_during_replay": True, "hindsight_caught": True, "suite_passed": True, "suite_void": False,
                         "admitted_outside_replay": True},
            "memory_bank": {"planted_late_row_caught": True, "clean_bank_passes": True},
            "static": {"trader_closure_clean": True, "research_loop_release_passes_replay": True, "loop2_learner_default": "off",
                       "curator_writers": ["engine/learning/test_path.py"]}}


def test_lookup_verdict_is_clean_on_a_good_part_and_flips_on_each_planted_defect():
    assert LA.verdict_year_lookup(good_lookup()).status == LA.CLEAN
    defects = [("curator", "unmatured_releases", 3), ("curator", "after_window_released", True), ("curator", "rerun_prefix_mismatches", 2),
               ("curator", "planted_year_feature_refused", False), ("curator", "replayed_year_late_first_release_after_maturity", False),
               ("research", "same_year_refused_during_replay", False), ("research", "hindsight_caught", False), ("research", "suite_passed", False),
               ("memory_bank", "planted_late_row_caught", False), ("static", "trader_closure_clean", False)]
    for part, key, bad in defects:
        p = good_lookup()
        p[part][key] = bad
        v = LA.verdict_year_lookup(p)
        assert v.status == LA.LEAK, key
        assert any(r.startswith("year lookup reaches the trader") for r in v.reasons)


def test_lookup_verdict_is_unmeasured_when_a_control_fails_or_the_part_is_missing():
    for part, key, bad in (("curator", "past_control_released", False), ("curator", "control_curator_ignoring_maturity_mismatches", 0),
                           ("research", "admitted_outside_replay", False), ("research", "suite_void", True),
                           ("memory_bank", "clean_bank_passes", False), ("curator", "rerun_prefix_mismatches", None)):
        p = good_lookup()
        p[part][key] = bad
        assert LA.verdict_year_lookup(p).status == LA.UNMEASURED, key
    assert LA.verdict_year_lookup(None).status == LA.UNMEASURED and LA.verdict_year_lookup({}).status == LA.UNMEASURED


def test_latent_lookup_items_are_reported_but_do_not_move_the_status():
    p = good_lookup()
    p["curator"]["hindsight_feature_refused"] = False
    p["static"]["research_loop_release_passes_replay"] = False
    v = LA.verdict_year_lookup(p)
    assert v.status == LA.CLEAN and len([r for r in v.reasons if r.startswith("latent")]) == 2


def fp_parts(trader="identifiable"):
    return {"fingerprint": {"exposed_6y_warmup_plus_window": {"trader_inputs": {"verdict": trader}, "levels_raw": {"verdict": "identifiable"}},
                            "hidden_12_months_only": {"trader_inputs": {"verdict": "not identifiable"}},
                            "exposed_6y_warmup_plus_window__shuffled_control": {"levels_raw": {"skill": 0.01}}}}


def test_channel6_combines_the_fingerprint_with_the_lookup_part():
    f = LA.default_path_facts()
    v = LA.compute_verdicts({**fp_parts(), "lookup": good_lookup()}, f, PROOFS, None)["6"]
    assert v.status == LA.LEAK and any("owner's ruling" in r for r in v.reasons) and v.checks["year_lookup_status"] == LA.CLEAN
    bad = good_lookup()
    bad["research"]["same_year_refused_during_replay"] = False
    v = LA.compute_verdicts({**fp_parts("not identifiable"), "lookup": bad}, f, PROOFS, None)["6"]
    assert v.status == LA.LEAK and v.reasons[0].startswith("year lookup reaches the trader")      # was QUARANTINED on the probe alone
    v = LA.compute_verdicts({**fp_parts("not identifiable"), "lookup": {}}, f, PROOFS, None)["6"]
    assert v.status == LA.UNMEASURED                                              # an empty lookup part can never improve the probe
    v = LA.compute_verdicts(fp_parts("not identifiable"), f, PROOFS, None)["6"]
    assert v.status == LA.QUARANTINED and v.checks["year_lookup_part_present"] is False


def test_hindsight_feature_keys_names_conclusions_not_observables():
    got = LA.hindsight_feature_keys({"knowability_unpredictable": 0.2, "what_changed_score": 1.0, "error_class_gap": 1.0, "r5": 0.1,
                                     "vol20": 1.0, "atr_pct": 0.3, "m_vix": 0.1})
    assert got == ["knowability_unpredictable", "what_changed_score", "error_class_gap"]
    assert LA.hindsight_feature_keys({}) == []


@pytest.fixture(scope="module")
def lookup_part(S):
    return S.year_lookup_audit(seed=0, end="2011-09-30")                        # 7 months: still holds every plant


def test_the_audit_module_itself_stays_off_the_curator_and_research_import_path(lookup_part):
    s = lookup_part["static"]                                                     # trader_research_violations + trader_path over the closure
    assert s["trader_closure_clean"] and s["trader_research_violations"] == [] and s["trader_closure_reaches"] == []
    assert s["leak_audit_on_trader_path"] and s["trader_closure_modules"] > 50   # it IS on the trader path, so this matters


def test_real_lookup_probes_run_and_are_clean_with_live_controls(lookup_part):
    c, r, m, s = (lookup_part[k] for k in ("curator", "research", "memory_bank", "static"))
    assert c["past_control_released"] and c["unmatured_releases"] == 0 and not c["after_window_released"]
    assert c["first_release_real_day"]["replayed_year_late"] > "2011-09-08"      # released only after it matured
    assert c["rerun_prefix_mismatches"] == 0 and c["control_curator_ignoring_maturity_mismatches"] > 0
    assert c["memories_filed_by_run1"] == c["memories_after_rerun"] > 0 and c["planted_extra_changes_releases"]
    assert r["same_year_refused_during_replay"] and r["admitted_outside_replay"] and r["admitted_without_replay_context"]
    assert r["suite_passed"] and m["planted_late_row_caught"] and s["trader_closure_clean"] and s["loop2_learner_default"] == "off"
    v = LA.verdict_year_lookup(lookup_part)
    assert v.status == LA.CLEAN, v.reasons
    json.dumps(LA._jsonable(lookup_part), default=str)


# =====================================================================================================================
# Part 4 - W-11: the same real-shaped year replayed twice with research filed under it in between
# =====================================================================================================================
FEATS = ("r5", "r20", "vol20", "atr_pct", "log_dv")
START = "2022-05-01"
REAL_YEARS = re.compile(r"(?<![\d.])(2022|2023)(?!\d)")


def make_data(start, n=40, seed=3, years=1):
    start = pd.Timestamp(start)
    idx = pd.bdate_range(start - pd.DateOffset(years=years), start + pd.DateOffset(months=12) - pd.Timedelta(days=1))
    rng = np.random.default_rng(seed)
    tick = sorted(f"AB{chr(65 + i % 26)}{i}" for i in range(n))
    close = pd.DataFrame(50 * np.exp(rng.normal(0.0004, 0.015, (len(idx), len(tick))).cumsum(0)), index=idx, columns=tick)
    opn = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.002, close.shape))
    vol = pd.DataFrame(rng.uniform(5e5, 2e6, close.shape), index=idx, columns=tick)
    stocks = {"Close": close, "Open": opn, "High": close * 1.01, "Low": close * 0.99, "Volume": vol}
    mk = pd.DataFrame({"SPY": 100 * np.exp(rng.normal(0.0003, 0.01, len(idx)).cumsum()), "^VIX": 18 + rng.normal(0, 1, len(idx))}, index=idx)
    market = {"Close": mk, "Open": mk, "High": mk, "Low": mk, "Volume": mk * 0 + 1e6}
    at = pd.DatetimeIndex(idx[::9], tz="UTC") + pd.Timedelta(hours=13)
    ev = pd.DataFrame({"ticker": [tick[i % n] for i in range(len(at))], "accepted": at, "kind": "8K", "form": "8-K"})
    fd = idx[::7]
    ins = pd.DataFrame({"symbol": [tick[i % n] for i in range(len(fd))], "filed": fd, "tdate": fd - pd.Timedelta(days=2)})
    return stocks, market, ev, ins, pd.DataFrame({"ticker": tick, "sic": 3570})


def play(home, shift_weeks, store_root, on_decision=None, check_path=True):
    """One disguised replay of the SAME real months (2022-05 .. 2023-04) through the hardened feed and the C64 path runner, on a
    curator store that persists at `store_root` (as CURATOR_ROOT does for loop2 --learner legit). `on_decision(real, day)` is the
    trusted side's hook, called on every decision day before the trader acts."""
    real_dir, livesim.DIR = livesim.DIR, home
    try:
        rec = BG.seal_window([], 7, "2026-01-01", tag="t1")
        rec.update(start=START, shift_days=7 * shift_weeks)
        rec["digest"] = BG.seal_digest(rec)
        (home / "sealed_t1.json").write_text(json.dumps(rec))
        feed = LA.hardened_feed_class()(livesim.SealedYear("t1"), warmup_years=1, data=make_data(START), use_insider=False)
        feed.precompute_features()
        cfg = TP.PathConfig(min_rows=20, sample_names=None, preroll_days=40, features=FEATS, seed=5, check_path=check_path)
        runner = TP.PathRunner(feed, cfg, curator=Curator(store_root=store_root, code_hash="pinned-f06"), code_hash_fn=lambda: "pinned-f06")
        seen, orig = [], runner.trader.act

        def spy(day, data):
            real = runner.clock.real(data.now)
            seen.append((real, day))
            if on_decision is not None:
                on_decision(real, day)
            return orig(day, data)
        runner.trader.act = spy

        def tick():
            runner.on_tick()
            feed.mark_processed()
        livesim.drive(feed, tick)
        return feed, runner, seen
    finally:
        livesim.DIR = real_dir


def research_records():
    """Research the trusted side files under the replayed year between the two runs: an episode-path summary (a matured outcome,
    no hindsight class), a counterfactual knowability class, a what_changed conclusion and an error record - all dated inside
    the window, all matured long before the rerun reaches the day they would be asked for."""
    from engine.research import firewall as FWL
    from engine.research.core import Knowability, Namespace
    from engine.research.namespaces import InfoKind, InfoObject

    def rec(oid, learned, origin, **payload):
        p = {"features": {"vol20": 0.8, "r5": -0.1}, "lean": 0.2, "horizon": 5, "trader_kind": "pattern", **payload}
        return InfoObject(oid, InfoKind.RESEARCH_RESULT, Namespace.MATURED_RESEARCH, learned, p, FWL._prov(learned), origin=origin,
                          tags={"real_year": int(learned[:4])})
    return [rec("episodepath", "2022-06-15", "engine.research.episode_paths", path_shape="gap_then_fade", n_obs=41),
            rec("counterfactual", "2022-07-01", "engine.research.counterfactual", knowability=str(Knowability.EXTERNALLY_CAUSED)),
            rec("whatchanged", "2022-07-15", "engine.research.what_changed", conclusion="regime shift", cause="external"),
            rec("errorrecord", "2022-08-01", "engine.research.error_loop", decision_effect="NONE", surprise_z=2.4)]


@pytest.fixture(scope="module")
def rerun(tmp_path_factory):
    """Run 1 of the window, then research filed under its year, then run 2 of the SAME real months under another disguise on the
    same curator store, with the trusted side asking the research firewall for everything on each decision day."""
    from engine.research import firewall as FWL
    from engine.research.namespaces import ResearchStore
    store_root = tmp_path_factory.mktemp("curator_store")
    run1 = play(tmp_path_factory.mktemp("run1"), 9000, store_root)
    after_run1 = sorted(m.mem_id for m in run1[1].curator.store.memories())
    rs = ResearchStore("f06-rerun")
    objs = research_records()
    for o in objs:
        rs.put(o)
    fw = FWL.ResearchTraderFirewall(rs, "f06-rerun")
    replay = FWL.ReplayContext("t1", START, "2023-04-30", run_index=1)
    asked = {"days": 0, "admitted": [], "refused_channels": {}, "release_refused": 0, "bare_admitted": set()}

    def trusted_side(real, day):
        asked["days"] += 1
        if asked["days"] % 4:
            return
        now = str(real.date())
        for o in objs:
            d = fw.inspect(o.object_id, now, replay)
            if d.admitted:
                asked["admitted"].append((now, o.object_id))
            asked["refused_channels"].setdefault(o.object_id, set()).update(str(c) for c in d.channels)
            if fw.inspect(o.object_id, now, None).admitted:           # control: without the replay context the rule is inert
                asked["bare_admitted"].add(o.object_id)
        try:
            fw.release([o.object_id for o in objs], now, replay)
        except FirewallBreach:
            asked["release_refused"] += 1
    run2 = play(tmp_path_factory.mktemp("run2"), 9500, store_root, on_decision=trusted_side, check_path=False)   # same code: run 1 checked the path
    return {"run1": run1, "run2": run2, "after_run1": after_run1, "asked": asked, "fw": fw, "objs": objs}


def test_w11_no_research_filed_under_the_replayed_year_is_released_during_the_rerun(rerun):
    a = rerun["asked"]
    assert a["days"] > 40 and a["release_refused"] == a["days"] // 4 > 0          # every asked release was refused whole
    assert a["admitted"] == []
    assert all("SAME_YEAR_RERUN" in ch for ch in a["refused_channels"].values())
    assert "episodepath" in a["bare_admitted"]                                    # control: the same record, no replay context -> admitted
    from engine.research import firewall as FWL
    later = FWL.ReplayContext("t9", "2024-03-01", "2025-02-28", run_index=0)
    dec = {o.object_id: rerun["fw"].inspect(o.object_id, "2024-06-03", later) for o in rerun["objs"]}
    assert dec["episodepath"].admitted                                            # outside the replayed year it is ordinary matured research
    assert {str(c) for c in dec["counterfactual"].channels} >= {"HINDSIGHT_LABEL"}
    assert {str(c) for c in dec["whatchanged"].channels} >= {"HINDSIGHT_LABEL"}
    assert {str(c) for c in dec["errorrecord"].channels} >= {"RESEARCH_ONLY_KNOWLEDGE"}


def test_w11_no_hindsight_label_or_research_item_reaches_the_trader_side(rerun):
    from engine.research import firewall as FWL
    research_tokens = {FWL.project(o).item_id for o in rerun["objs"]}
    feed, runner, seen = rerun["run2"]
    assert len(seen) > 240
    words = FWL.HINDSIGHT_TOKENS | {"counterfactual", "what_changed", "whatchanged", "episode", "error_record", "conclusion"}
    for real, day in seen:
        text = day.json().lower()
        assert not any(w in text for w in words), (real, text[:200])
        assert not research_tokens & {i.item_id for i in day.release.items}
        assert not REAL_YEARS.search(day.json()) and TV.find_violations(day.to_dict()) == []
    for m in runner.curator.store.memories():                                    # what is filed can never carry a hindsight name either
        assert LA.hindsight_feature_keys(m.payload["features"]) == [], m.payload["features"]
    assert runner.release_hits == {} and TP.trader_side_violations() == []


def test_w11_every_memory_the_rerun_is_shown_had_matured_before_that_real_day(rerun):
    _, r1, s1 = rerun["run1"]
    _, r2, s2 = rerun["run2"]
    earliest = {}
    for m in r2.curator.store.memories():
        earliest[m.key] = min(earliest.get(m.key, m.matured_at), m.matured_at)
    shown = 0
    for real, day in s2:
        for it in day.release.items:
            assert pd.Timestamp(earliest[it.item_id]) < real
            shown += 1
    assert shown > 0 and r2.counts["releases"] > 0
    years = {m.real_year for m in r2.curator.store.memories()}
    assert years == {2022, 2023}                                                 # filed by the real year, both runs, one store
    assert set(rerun["after_run1"]) <= {m.mem_id for m in r2.curator.store.memories()}
    assert r1.curator.store.verify()["ok"] and r2.curator.store.verify()["ok"]


def test_w11_trader_view_refuses_planted_year_and_date_content_in_released_items(rerun):
    from engine.research import firewall as FWL
    from engine.research.namespaces import ResearchStore
    _, runner, seen = rerun["run2"]
    item = next(it for _, day in seen for it in day.release.items)
    for bad in ({**item.features, "year": 0.4}, {**item.features, "vol20": 2022.0}, {**item.features, "filed_q": 1.0}, {**item.features, "r5": 20220615}):
        with pytest.raises(FirewallBreach):
            TV.TraderMemoryItem.make(item.item_id, item.kind, 1.0, bad, item.lean, item.horizon)
    with pytest.raises(FirewallBreach):
        runner.curator.file({**item.to_dict(), "features": {**item.features, "regime_2022_like": 1.0}}, "2023-04-03", 2023,
                            {"m_vix": 0.1}, matured_at="2023-04-04")
    st = ResearchStore("f06-planted-year")
    st.put(FWL.clean_object("yr", learned="2019-11-29", features={"vol20": 1.0, "regime_2022_like": 1.0}))
    d = FWL.ResearchTraderFirewall(st, "f06-planted-year").inspect("yr", "2024-06-03", None)
    assert not d.admitted and "YEAR_IDENTITY" in {str(c) for c in d.channels}
    assert TV.find_violations({"features": {"vol20": 1.0}}) == []                 # the clean control passes


def test_quarantined_tuned_meta_lessons_never_reach_a_later_trader(L, tmp_path, monkeypatch):
    """F06 hook (channel 4): windows quarantined as tuned_meta are published, and Feed.long_term_memory drops their lessons."""
    state = L.quarantine_tuned_meta({"windows": [
        {"run_id": "w01a", "window": "w01a", "basis_version": 0, "untrained_basis": True, "meta": dict(L.A.META_DEFAULT)},
        {"run_id": "w01b", "window": "w01b", "basis_version": 0, "untrained_basis": True, "meta": dict(L.NEUTRAL_META)}]})
    assert L.publish_bank_exclusions(state, root=tmp_path) == ["w01a"]
    assert livesim.bank_exclusions(tmp_path) == frozenset({"w01a"}) and livesim.bank_exclusions(tmp_path / "none") == frozenset()
    bank = pd.DataFrame({"arm": ["x", "y"], "ctx": [[0.0], [0.0]], "outcome": [0.1, -0.1], "window": ["w01a", "w01b"],
                         "real_end": ["2001-01-05", "2001-01-05"]})
    bank.to_parquet(tmp_path / "memory_bank.parquet")
    monkeypatch.setattr(livesim, "DIR", tmp_path)
    feed = object.__new__(livesim.Feed)
    feed.first_live, feed._shift, feed.enforce = pd.Timestamp("2030-01-02"), pd.Timedelta(days=0), False
    out = feed.long_term_memory()
    assert list(out["arm"]) == ["y"]                                   # the tuned play's lesson is gone, the clean one stays
    (tmp_path / livesim.BANK_EXCLUSIONS).unlink()
    assert sorted(feed.long_term_memory()["arm"]) == ["x", "y"]        # the hook, removed, lets it through: the test can fail
