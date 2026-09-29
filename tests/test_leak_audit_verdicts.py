"""S18: channel verdicts are COMPUTED from the default blind path and the measurements (contract sections 30, 55, 85; canon C56).

Every test that matters flips one default (or plants one defect) and asserts the verdict moves: a status that cannot change is
typed, not computed. Sources are the real repository files read as text and edited in memory - nothing on disk changes and no
sealed window is opened. All synthetic, seconds."""
import json

import pytest

from engine import leak_audit as L

SRC = L._read_sources()
PROOFS = L.run_proofs()


def facts_with(key=None, old=None, new=None, count=1):
    """Facts of the real sources with ONE textual edit (the edit must apply exactly `count` time: a refactor that moves the
    line fails the test loudly instead of silently testing nothing)."""
    if key is None:
        return L.default_path_facts(sources={})
    text = SRC[key]
    assert text.count(old) == count, f"{old!r} occurs {text.count(old)}x in {key}"
    return L.default_path_facts(sources={key: text.replace(old, new)})


def good_state():
    """A loop state as the fixed loop writes it: v1 was trained on w01 (ended 1990), w02 was played untrained, w03 (1995) played v1."""
    space = L.default_path_facts()["loop2_cfg_space"]
    neutral = L.neutral_default_cfg(space)
    meta = L.default_path_facts()["meta_default_literal"]
    return {"windows": [{"run_id": "w01", "basis_version": 0, "untrained_basis": True, "prior_cfg": neutral, "meta": meta},
                        {"run_id": "w03", "basis_version": 1, "untrained_basis": False, "prior_cfg": {"k": 2}, "meta": meta}],
            "lineage": [{"version": 1, "cfg": {"k": 2}, "meta": meta, "trained_on": [["w01", "1980-01-01", "1980-12-31"]]},
                        {"version": 2, "cfg": {"k": 2}, "meta": meta, "trained_on": [["w01", "1980-01-01", "1980-12-31"], ["w03", "1995-01-01", "1995-12-31"]]}]}, space, meta


def state_check(state, space, meta):
    return L.lineage_state_check(state, space, meta)


def v4(facts=None, proofs=None, sc="ok"):
    facts = facts or L.default_path_facts()
    if sc == "ok":
        s, sp, m = good_state()
        sc = state_check(s, sp, m)
    return L.verdict_learned_state(facts, proofs or PROOFS, sc)


# ---------------------------------------------------------------------------------------------------------------------
# the facts really come from the source
# ---------------------------------------------------------------------------------------------------------------------
def test_facts_read_the_real_default_path():
    f = L.default_path_facts()
    assert all(f["sources_parsed"].values())
    assert f["feed_tradable_rule_default"] == "split_invariant" and f["blind_feed_hardened_default"] is True and f["run_hardened_default"] is True
    assert f["loop2_play_calls_without_per_id"] == [] and f["loop2_n_run_workers_calls"] >= 1
    assert f["loop2_train_basis_calls"] and not f["loop2_train_basis_calls"][0]["passes_as_of"]     # the training call takes no as_of: the gate is what is proven


def test_missing_or_unparsable_source_gives_none_not_a_guess():
    f = L.default_path_facts(sources={"livesim": None, "loop2": "def broken(:"})
    assert f["sources_parsed"]["livesim"] is False and f["sources_parsed"]["loop2"] is False
    assert L.verdict_adjusted_prices(f, PROOFS, None).status == L.UNMEASURED
    assert L.verdict_feed_shape(f, None).status == L.UNMEASURED
    assert L.verdict_learned_state(f, PROOFS, None).status == L.UNMEASURED


# ---------------------------------------------------------------------------------------------------------------------
# channel 2: flip each link of the split-invariant chain
# ---------------------------------------------------------------------------------------------------------------------
def test_channel2_is_fixed_on_the_real_default_and_flips_to_leak():
    assert L.verdict_adjusted_prices(L.default_path_facts(), PROOFS, None).status == L.FIXED
    flips = [("livesim", 'tradable_rule="split_invariant"):', "tradable_rule=None):"),
             ("livesim", "relative=True, rel_q=rel_q, tradable_rule=self.tradable_rule)", "relative=True, rel_q=rel_q)"),
             ("features", "tradable = split_invariant_tradable(C, V, rel_q[1])", "tradable = (C.rank(axis=1, pct=True) >= rel_q[0]) & C.notna()")]
    for key, old, new in flips:
        assert L.verdict_adjusted_prices(facts_with(key, old, new), PROOFS, None).status == L.LEAK, old


def test_channel2_proof_can_fail_when_the_rule_moves_under_a_planted_split(monkeypatch):
    p = L.prove_split_invariance()
    assert p["mechanism_neutralises_planted_leak"] and p["planted_leak_is_real"]
    monkeypatch.setattr(L, "split_invariant_tradable", lambda C, V, dv_q=0.4, window=20: C.rank(axis=1, pct=True) >= 0.2)   # a rule that reads the price level
    bad = L.prove_split_invariance()
    assert not bad["mechanism_neutralises_planted_leak"]
    assert L.verdict_adjusted_prices(L.default_path_facts(), {"split_invariance": bad}, None).status == L.LEAK


def test_channel2_without_the_planted_control_is_unmeasured_not_fixed():
    weak = {"split_invariance": {"mechanism_neutralises_planted_leak": True, "planted_leak_is_real": False}}
    v = L.verdict_adjusted_prices(L.default_path_facts(), weak, None)
    assert v.status == L.UNMEASURED                                    # the proof could not have failed: it does not count
    assert L.verdict_adjusted_prices(L.default_path_facts(), {}, None).status == L.UNMEASURED


# ---------------------------------------------------------------------------------------------------------------------
# channel 8c
# ---------------------------------------------------------------------------------------------------------------------
def causality(hard_spy=100.0, hard_rho=-0.02, hard_before=0, hard_pos=0.005, plain_rho=0.98):
    row = {"plain": {"spy_first_level": 20.0, "column_order_vs_real_alpha_rho": plain_rho, "columns_shown_before_listing": 12},
           "hardened": {"spy_first_level": hard_spy, "column_order_vs_real_alpha_rho": hard_rho, "columns_shown_before_listing": hard_before},
           "hardened_rerun_linkability": {"share_reidentified_by_column_position": hard_pos}}
    return {"feed_exposure_real_windows": {"1995-06-01": row, "2018-03-01": row}}


def test_channel8c_fixed_by_default_and_each_unsafe_setting_flips_it():
    f = L.default_path_facts()
    assert L.verdict_feed_shape(f, causality()).status == L.FIXED
    flips = [("livesim", "def blind_feed_class(hardened=True):", "def blind_feed_class(hardened=False):"),
             ("livesim", "meta=None, hardened=True):", "meta=None, hardened=False):"),
             ("livesim", "feed = blind_feed_class(hardened)(sealed)", "feed = Feed(sealed)"),
             ("loop2", "livesim.run(cfg, run_id,", "livesim.run(cfg, run_id, hardened=False,")]
    for key, old, new in flips:
        assert L.verdict_feed_shape(facts_with(key, old, new), causality()).status == L.LEAK, old


def test_channel8c_measured_exposure_flips_and_uninformative_control_does_not_pass():
    f = L.default_path_facts()
    for bad in (causality(hard_spy=6.1), causality(hard_rho=0.95), causality(hard_before=3), causality(hard_pos=0.9)):
        assert L.verdict_feed_shape(f, bad).status == L.LEAK
    blind_instrument = causality(plain_rho=0.0)
    blind_instrument["feed_exposure_real_windows"] = {k: {**v, "plain": {"spy_first_level": 100.0, "column_order_vs_real_alpha_rho": 0.0, "columns_shown_before_listing": 0}}
                                                      for k, v in blind_instrument["feed_exposure_real_windows"].items()}
    assert L.verdict_feed_shape(f, blind_instrument).status == L.UNMEASURED      # the plain feed shows nothing: the instrument cannot see, so no FIXED
    assert L.verdict_feed_shape(f, None).status == L.UNMEASURED and L.verdict_feed_shape(f, {}).status == L.UNMEASURED


# ---------------------------------------------------------------------------------------------------------------------
# channel 4
# ---------------------------------------------------------------------------------------------------------------------
def test_channel4_real_default_is_quarantined_only_by_the_tuned_meta_defaults():
    v = v4()
    assert v.status == L.QUARANTINED and v.checks["meta_default_tuned_on_real_outcomes"] is True
    assert "META_DEFAULT" in v.reasons[0]
    untuned = L.default_path_facts(sources={"adaptive": SRC["adaptive"].replace("sensitivity study", "coin flip")})
    assert v4(untuned).status == L.FIXED                                 # nothing else stands between the gate and FIXED


def test_channel4_verdict_flips_when_the_loop_passes_the_global_basis():
    first, second = 'run_workers(ids, st["cfg"], st["meta"], per_id=per_id)', 'run_workers(stale_ids, st["cfg"], st["meta"], per_id=per_id)'
    assert SRC["loop2"].count(first) == 1 and SRC["loop2"].count(second) == 1
    for old in (first, second):                                              # dropping the override at EITHER call site (incl. the stale-rerun path) is a leak
        v = v4(L.default_path_facts(sources={"loop2": SRC["loop2"].replace(old, old.replace(", per_id=per_id", ""))}))
        assert v.status == L.LEAK and any("global_basis" in r or "per_window" in r for r in v.reasons), old
    cls = 'classify_round(ids, st["cfg"], st["meta"], rnd, per_id=per_id)'
    assert cls in SRC["loop2"]
    assert v4(L.default_path_facts(sources={"loop2": SRC["loop2"].replace(cls, 'classify_round(ids, st["cfg"], st["meta"], rnd)', 1)})).status == L.LEAK


def test_channel4_verdict_flips_when_the_registered_set_differs_from_the_trained_set():
    f = facts_with("loop2", "register_basis(st, lineage, st[\"version\"], res.cfg, res.meta, wins)", "register_basis(st, lineage, st[\"version\"], res.cfg, res.meta, wins[:1])")
    v = v4(f)
    assert v.status == L.LEAK and v.checks["registered_training_set_equals_trained_set"] is False


def test_channel4_verdict_flips_when_planning_or_the_neutral_start_is_removed():
    plan = "plan = plan_round(ids, lineage)"
    assert v4(facts_with("loop2", plan, "plan = {r: {'cfg': st['cfg'], 'meta': st['meta'], 'version': st['version'], 'untrained': False} for r in ids}")).status == L.LEAK
    assert v4(facts_with("loop2", "NEUTRAL_CFG = LA.neutral_default_cfg(CFG_SPACE)", "NEUTRAL_CFG = {'k': 1}")).status == L.LEAK
    head = 'headline = [w for w in st["windows"] if not w.get("thin") and not w.get("legacy")]'
    assert v4(facts_with("loop2", head, 'headline = [w for w in st["windows"] if not w.get("thin")]')).status == L.LEAK


def test_channel4_proof_fails_if_the_lineage_gate_is_broken(monkeypatch):
    ok = L.prove_lineage_gate()
    assert ok["mechanism_holds"] and ok["planted_leak_is_real"] and ok["naive_share_of_plays_touched_by_future_training"] > 0.5
    monkeypatch.setattr(L.BasisLineage, "eligible", lambda self, rec, real_start, allow_same_window=False: True)      # a gate that admits everything
    broken = L.prove_lineage_gate()
    assert not broken["mechanism_holds"] and broken["gated_violations"] > 0
    assert v4(proofs={**PROOFS, "lineage_gate": broken}).status == L.LEAK


def test_lineage_state_check_flags_a_planted_future_trained_play():
    s, space, meta = good_state()
    assert state_check(s, space, meta)["violations"] == []
    s["windows"].append({"run_id": "w02", "basis_version": 1, "untrained_basis": False, "prior_cfg": {"k": 2}, "meta": meta})
    s["lineage"][1]["trained_on"].append(["w02", "1979-01-01", "1979-12-31"])           # w02 starts 1979, before v1's training window ended (1980-12-31)
    chk = state_check(s, space, meta)
    assert chk["violations"] == [{"window": "w02", "version": 1}]
    v = v4(sc=chk)
    assert v.status == L.LEAK and "w02" in " ".join(v.reasons)


def test_lineage_state_check_flags_cfg_meta_and_unresolved_plays():
    s, space, meta = good_state()
    s["windows"][0]["prior_cfg"] = {**s["windows"][0]["prior_cfg"], "k": 99}
    assert state_check(s, space, meta)["untrained_cfg_mismatch"] == ["w01"]
    assert v4(sc=state_check(s, space, meta)).status == L.LEAK
    s, space, meta = good_state()
    s["windows"][0]["meta"] = {**meta, "half_life": 999}
    assert state_check(s, space, meta)["untrained_meta_mismatch"] == ["w01"]
    s, space, meta = good_state()
    s["windows"].append({"run_id": "wX", "basis_version": 2, "untrained_basis": False, "prior_cfg": {}, "meta": meta})        # real start never recorded
    chk = state_check(s, space, meta)
    assert chk["unresolved"] == ["wX"] and v4(sc=chk).status == L.LEAK
    s, space, meta = good_state()
    s["windows"].append({"run_id": "wY", "legacy": True, "basis_version": 1})           # legacy plays are kept apart, not judged
    assert state_check(s, space, meta)["legacy"] == ["wY"]


def test_lineage_state_check_empty_and_missing_state():
    assert L.lineage_state_check(None, {}, {}) is None
    e = L.lineage_state_check({"windows": [], "lineage": []}, {"k": [1, 2, 3]}, {"a": 1})
    assert e["violations"] == [] and e["trained"] == [] and e["lineage_is_monotone"] is True
    assert v4(sc=None).status == L.UNMEASURED                                          # no state file read: never FIXED/QUARANTINED
    s, space, meta = good_state()
    s["lineage"][1]["trained_on"] = s["lineage"][1]["trained_on"][1:]                  # v2 forgot a window v1 saw
    assert state_check(s, space, meta)["lineage_is_monotone"] is False
    assert v4(sc=state_check(s, space, meta)).status == L.LEAK


# ---------------------------------------------------------------------------------------------------------------------
# the other channels, from measurements
# ---------------------------------------------------------------------------------------------------------------------
def parts_ok():
    return {"static": {"closure_modules": ["livesim"], "research_only_modules_reachable": [], "free_text_columns": {"events": []},
                       "model_embargo_sessions": 10, "label_reach_sessions": 6},
            "runtime": {"blocked_network_attempts": [], "livesim_state_files_opened": [], "data_cache_files_opened": []},
            "causality": {**causality(), "features_truncation_invariance_2012_sample": {"clean": True, "leaky_features": {}, "n_features": 53}},
            "fingerprint": {"exposed_6y_warmup_plus_window": {"trader_inputs": {"verdict": "not identifiable"}, "levels_raw": {"verdict": "identifiable"}},
                            "hidden_12_months_only": {"trader_inputs": {"verdict": "not identifiable"}},
                            "exposed_6y_warmup_plus_window__shuffled_control": {"levels_raw": {"skill": 0.01}}},
            "survivorship": {"registry_rows_with_last_close": 244, "dead_names_with_recovered_prices": 77},
            "metadata": {"crypto_matches_in_panel": [{"ticker": "MSTR"}], "not_crypto_on_blind_codes_filters_nothing": True}}


def verdicts(parts, state="ok"):
    s = None
    if state == "ok":
        st, sp, m = good_state()
        s = state_check(st, sp, m)
    return L.compute_verdicts(parts, L.default_path_facts(), PROOFS, s)


def test_planted_measurement_defects_flip_each_channel():
    base = {k: v.status for k, v in verdicts(parts_ok()).items()}
    assert base["7"] == L.CLEAN and base["8a"] == L.CLEAN and base["8b"] == L.CLEAN and base["8e"] == L.CLEAN and base["5"] == L.CLEAN
    assert base["2"] == L.FIXED and base["8c"] == L.FIXED and base["4"] == L.QUARANTINED and base["6"] == L.QUARANTINED and base["8d"] == L.CLEAN
    p = parts_ok(); p["runtime"]["blocked_network_attempts"] = [["connect", "1.2.3.4"]]
    assert verdicts(p)["7"].status == L.LEAK
    p = parts_ok(); p["static"]["free_text_columns"] = {"events": ["headline"]}
    assert verdicts(p)["8a"].status == L.LEAK
    p = parts_ok(); p["causality"]["features_truncation_invariance_2012_sample"] = {"clean": False, "leaky_features": {"mom20": 0.3}}
    v = verdicts(p)
    assert v["8b"].status == L.LEAK and v["8e"].status == L.LEAK
    p = parts_ok(); p["fingerprint"]["exposed_6y_warmup_plus_window"]["trader_inputs"]["verdict"] = "identifiable"
    assert verdicts(p)["6"].status == L.LEAK
    p = parts_ok(); p["fingerprint"]["exposed_6y_warmup_plus_window__shuffled_control"]["levels_raw"]["skill"] = 0.6
    assert verdicts(p)["6"].status == L.UNMEASURED                                 # a probe that finds skill in shuffled labels is not trusted
    p = parts_ok(); p["static"]["closure_modules"] = ["livesim", "analogs"]
    assert verdicts(p)["5"].status == L.LEAK
    p = parts_ok(); p["runtime"]["livesim_state_files_opened"] = ["state/livesim/loop2.json"]
    assert verdicts(p)["8d"].status == L.LEAK
    p = parts_ok(); p["static"]["research_only_modules_reachable"] = ["lessons"]
    assert verdicts(p)["8d"].status == L.LEAK
    p = parts_ok(); p["survivorship"]["dead_names_with_recovered_prices"] = 244
    assert verdicts(p)["1"].status == L.CLEAN
    p = parts_ok(); p["metadata"] = {"crypto_matches_in_panel": [], "hard_listed_crypto_tickers_traded_before_2018": []}
    assert verdicts(p)["3"].status == L.CLEAN


def test_missing_parts_are_unmeasured_never_clean():
    v = verdicts({}, state=None)
    assert v["2"].status == L.FIXED                                                  # computed from source + a proof, needs no cache part
    assert v["4"].status == L.UNMEASURED and v["8c"].status == L.UNMEASURED
    for k in ("1", "3", "5", "6", "7", "8a", "8b", "8d", "8e"):
        assert v[k].status == L.UNMEASURED, k
    assert not [k for k, x in v.items() if x.status == L.CLEAN]
    p = parts_ok(); del p["runtime"]
    assert verdicts(p)["7"].status == L.UNMEASURED and verdicts(p)["8d"].status == L.UNMEASURED


def test_network_verdict_flips_when_the_worker_stops_installing_the_guard():
    f = facts_with("loop2", "guard = LA.NetworkGuard().install()", "guard = None")
    v = L.compute_verdicts(parts_ok(), f, PROOFS, None)["7"]
    assert v.status == L.LEAK and v.checks["worker_installs_guard"] is False


def test_status_vocabulary_and_worst_of():
    assert L.worst_status(L.CLEAN, L.FIXED) == L.FIXED
    assert L.worst_status(L.FIXED, L.QUARANTINED) == L.QUARANTINED
    assert L.worst_status(L.QUARANTINED, L.UNMEASURED) == L.UNMEASURED
    assert L.worst_status(L.UNMEASURED, L.LEAK, L.CLEAN) == L.LEAK
    assert L.worst_status() == L.UNMEASURED
    with pytest.raises(ValueError):
        L.Verdict("MAYBE", {})
    a = L.Audit()
    a.add(L.Channel("x", "unmeasured", L.UNMEASURED))
    assert a.counts()["UNMEASURED"] == 1 and a.open_leaks() == [] and "UNMEASURED" in a.markdown()


def test_verdict_evidence_is_json_serialisable():
    v = verdicts(parts_ok())
    json.dumps({k: L._jsonable(x.as_evidence()) for k, x in v.items()}, default=str)
