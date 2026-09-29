"""Outer basis search (Phase 19): structure of the screen/confirm funnel, no look-ahead, planted effect, overfit guard."""
import zlib
import numpy as np
import pytest
from engine import basis_search as B, objective as O

CFG0 = {"k": 6, "exit_q": 0.8, "pool_q": 0.7, "brake": None}
META0 = {"half_life": 6, "prior_weeks": 8, "switch_z": 2.0, "min_weeks": 4, "cooldown": 3, "revert_drop": 0.03, "ic_beta": 0.5,
         "det_max": 0.25, "det_min_weeks": 4, "mem_half_life": 8, "mem_bandwidth": 1.5, "mem_prior_scale": 0.1,
         "mem_shrink": 6, "mem_shock_k": 2.5, "mem_shock_cut": 0.25}
CS = {"k": [1, 2, 3, 4, 5, 6], "exit_q": [0.5, 0.8], "pool_q": [0.5, 0.7, 0.9]}
MS = {k: v for k, v in B.META_SPACE.items()}


def wins(n, base=0):
    return [{"id": f"w{i}", "end": base + i} for i in range(n)]


def sigma_eval(w, cfg, meta):
    """Planted effect: smaller k means bigger weekly swings; in-band share peaks near k=2 (sd ~ 7%)."""
    rng = np.random.default_rng(zlib.crc32(w["id"].encode()))
    z = rng.standard_normal(52)
    return O.week_row(0.005 + 0.035 * (cfg["k"] ** -0.55) * 2.2 * z)


def test_meta_space_covers_bible_list():
    need = {"half_life", "prior_weeks", "switch_z", "min_weeks", "cooldown", "revert_drop", "ic_beta", "det_max",
            "det_min_weeks", "mem_half_life", "mem_bandwidth", "mem_prior_scale", "mem_shrink", "mem_shock_k", "mem_shock_cut"}
    assert need <= set(B.META_SPACE)
    from engine import adaptive as A
    from engine.memory import MEM_DEFAULT
    known = set(A.META_DEFAULT) | set(MEM_DEFAULT) | {"det_max", "det_min_weeks"}   # the detector reads these via meta.get()
    assert set(B.META_SPACE) <= known, set(B.META_SPACE) - known
    rng = np.random.default_rng(0)
    for _ in range(50):
        c, m = B.sample_candidate(rng, CFG0, META0)
        assert not B.validate_meta(m) and set(c) >= set(CFG0)


def test_validate_meta_catches_nonsense():
    assert B.validate_meta({**META0, "half_life": 0})
    assert B.validate_meta({**META0, "det_max": 1.5})
    assert B.validate_meta({**META0, "cooldown": 20})
    assert not B.validate_meta(META0)
    with pytest.raises(ValueError):
        B.sample_candidate(np.random.default_rng(0), CFG0, META0, meta_space={"half_life": [0, -1]})


def test_funnel_structure_and_eval_budget():
    calls = []
    def ev(w, c, m):
        calls.append((w["id"], B.fingerprint(c, m)))
        return sigma_eval(w, c, m)
    ws = wins(15)
    cfg = B.SearchConfig(seed=3)
    s = B.BasisSearch(ev, ws, CFG0, META0, cfg, CS, MS)
    r = s.run()
    assert r.n_screen == 10 and r.n_windows == 15
    assert len(calls) == len(set(calls)) == r.n_evals                     # cached: nothing evaluated twice
    per_cand = {}
    for wid, fp in calls:
        per_cand.setdefault(fp, set()).add(wid)
    full = [fp for fp, ids in per_cand.items() if len(ids) == 15]
    assert len(full) == 4                                                  # incumbent + top 3 confirmed on ALL windows
    assert all(len(ids) in (10, 15) for ids in per_cand.values())          # everyone else saw only the 10 screening windows
    assert sum(1 for c in r.candidates if c.confirm is not None) == 4


def test_planted_effect_is_found():
    r = B.BasisSearch(sigma_eval, wins(20), CFG0, META0, B.SearchConfig(seed=1), CS, MS).run()
    assert r.adopted and r.cfg["k"] < CFG0["k"], r.reason
    assert r.winner.confirm.t1 > r.incumbent.confirm.t1 and r.winner.boot_lo > 0
    assert r.reason.startswith("ADOPTED")


def test_pure_noise_is_rarely_adopted():
    """Every candidate is identical in distribution: selection alone must not be mistaken for improvement."""
    adopted = 0
    for seed in range(12):
        def noise(w, c, m, seed=seed):
            rng = np.random.default_rng(zlib.crc32((w["id"] + B.fingerprint(c, m)).encode()) + seed)
            return O.week_row(0.07 * rng.standard_normal(52))
        r = B.BasisSearch(noise, wins(26), CFG0, META0, B.SearchConfig(seed=seed), CS, MS).run()
        adopted += r.adopted
    assert adopted <= 2, adopted


def test_screen_luck_is_not_enough():
    """Candidate looks perfect on the screening windows only; confirmation on all windows must expose it."""
    ws = wins(20)
    def ev(w, c, m):
        rng = np.random.default_rng(zlib.crc32(w["id"].encode()))
        lucky = c["k"] != CFG0["k"] and int(w["id"][1:]) % 2 == 0          # 'good' only on half the windows
        sd = 0.065 if lucky else 0.02
        return O.week_row(sd * rng.standard_normal(52))
    r = B.BasisSearch(ev, ws, CFG0, META0, B.SearchConfig(seed=2, n_screen=10), CS, MS).run()
    if r.adopted:                                                           # an adoption must survive the all-window bootstrap
        assert r.winner.boot_lo > 0 and r.winner.confirm.t1 > r.incumbent.confirm.t1
    assert any("not reliable" in c.verdict or "penalty" in c.verdict or "firewall" in c.verdict or c.verdict.startswith("ADOPTED")
               for c in r.candidates if c.confirm is not None)


def test_no_lookahead_windows_after_as_of_never_evaluated():
    seen = set()
    def ev(w, c, m):
        seen.add(w["end"])
        return sigma_eval(w, c, m)
    B.BasisSearch(ev, wins(20), CFG0, META0, B.SearchConfig(seed=0, n_start=6), CS, MS).run(as_of=9)
    assert seen and max(seen) <= 9


def test_too_few_windows_keeps_basis():
    r = B.BasisSearch(sigma_eval, wins(2), CFG0, META0, B.SearchConfig(seed=0), CS, MS).run()
    assert not r.adopted and r.cfg == CFG0 and r.n_evals == 0
    r0 = B.BasisSearch(sigma_eval, [], CFG0, META0, B.SearchConfig(seed=0), CS, MS).run()
    assert not r0.adopted and r0.reason.startswith("only 0 windows")


def test_deterministic_and_record_is_json_safe():
    import json
    a = B.BasisSearch(sigma_eval, wins(14), CFG0, META0, B.SearchConfig(seed=5), CS, MS).run()
    b = B.BasisSearch(sigma_eval, wins(14), CFG0, META0, B.SearchConfig(seed=5), CS, MS).run()
    assert (a.adopted, a.cfg, a.meta) == (b.adopted, b.cfg, b.meta)
    rec = a.to_record()
    json.dumps(rec, default=str)
    assert rec["event"] == "basis_search" and rec["n_windows"] == 14


def test_incumbent_never_replaced_by_a_worse_confirmed_candidate():
    """The incumbent is already the planted optimum (k=2): whatever is adopted must not lose tier 1 to it."""
    inc = {**CFG0, "k": 2}
    r = B.BasisSearch(sigma_eval, wins(18), inc, META0, B.SearchConfig(seed=4), CS, MS).run()
    assert r.winner.confirm.key >= r.incumbent.confirm.key


def test_screening_windows_cannot_testify_for_their_own_winner():
    """Planted overfit: a candidate is great on exactly the windows the screen sees and bad elsewhere. Held-out evidence
    must reject it. The screen's windows are learned by asking the same-seed search which windows it evaluated first."""
    seen_first = []
    ws = wins(30)
    probe = B.BasisSearch(lambda w, c, m: (seen_first.append(w["id"]), sigma_eval(w, c, m))[1], ws, CFG0, META0,
                          B.SearchConfig(seed=6, n_start=1), CS, MS)
    probe.run()
    screen_ids = set(seen_first[:10])
    assert len(screen_ids) == 10

    def ev(w, c, m):
        rng = np.random.default_rng(zlib.crc32(w["id"].encode()))
        if c["k"] == CFG0["k"]:
            sd = 0.03
        else:                                                 # big win on the screened windows, small loss everywhere else
            sd = 0.07 if w["id"] in screen_ids else 0.028
        return O.week_row(sd * rng.standard_normal(52))
    r = B.BasisSearch(ev, ws, CFG0, META0, B.SearchConfig(seed=6, n_start=24), CS, MS).run()
    assert not r.adopted, r.reason
    verdicts = [c.verdict for c in r.candidates if c.confirm is not None]
    assert any(("windows improved" in v or "not reliable" in v or "penalty" in v) for v in verdicts), verdicts
    # the plant is real: with the held-out and overfit guards switched off, the same search is fooled and adopts
    naive = B.SearchConfig(seed=6, n_start=24, min_holdout=999, min_win_share=0.0, boot_q=0.9, penalty_se=0.0, penalty_gap=0.0)
    assert B.BasisSearch(ev, ws, CFG0, META0, naive, CS, MS).run().adopted


def test_fallback_without_holdout_is_stricter_than_with_it():
    """With no unseen windows the significance level is divided by n_start, not n_top: same data, fewer adoptions."""
    n_small = sum(B.BasisSearch(sigma_eval, wins(10), CFG0, META0, B.SearchConfig(seed=s, n_screen=10), CS, MS).run().adopted
                  for s in range(6))
    n_large = sum(B.BasisSearch(sigma_eval, wins(20), CFG0, META0, B.SearchConfig(seed=s, n_screen=10), CS, MS).run().adopted
                  for s in range(6))
    assert n_large >= n_small and n_large >= 4


def test_penalty_is_monotone_in_noise():
    s = B.BasisSearch(sigma_eval, wins(6), CFG0, META0)
    c = B.Candidate(CFG0, META0)
    c.screen = O.evaluate([O.week_row([0.06] * 20)])
    c.confirm = c.screen
    steady = s._adjust(c, np.full(8, 1.0))
    noisy = s._adjust(c, np.array([5.0, -3.0, 4.0, -2.0, 6.0, -4.0, 3.0, -1.0]))
    assert noisy < steady
    c.confirm = O.evaluate([O.week_row([0.01] * 20)])                       # fell hard from screen to confirm
    assert s._adjust(c, np.full(8, 1.0)) < steady


def test_invalid_candidates_never_reach_evaluation():
    seen = []
    def ev(w, c, m):
        seen.append(m)
        return sigma_eval(w, c, m)
    ms = {"half_life": [0, 3, 6], "cooldown": [0, 30]}
    B.BasisSearch(ev, wins(12), CFG0, META0, B.SearchConfig(seed=0), CS, ms).run()
    assert seen and all(not B.validate_meta(m) for m in seen)


def test_duplicate_candidates_are_evaluated_once():
    calls = []
    def ev(w, c, m):
        calls.append((w["id"], B.fingerprint(c, m)))
        return sigma_eval(w, c, m)
    B.BasisSearch(ev, wins(12), CFG0, META0, B.SearchConfig(seed=0, n_start=24), {"k": [1, 2]}, {}).run()   # only 2 distinct cfgs exist
    assert len({fp for _, fp in calls}) <= 3 and len(calls) == len(set(calls))


def test_batch_evaluator_gives_the_same_answer_and_is_used_once_per_candidate():
    sizes = []
    def batch(tasks):
        sizes.append(len(tasks))
        return [sigma_eval(w, c, m) for w, c, m in tasks]
    ws = wins(20)
    a = B.BasisSearch(sigma_eval, ws, CFG0, META0, B.SearchConfig(seed=1), CS, MS).run()
    b = B.BasisSearch(None, ws, CFG0, META0, B.SearchConfig(seed=1), CS, MS, evaluate_batch=batch).run()
    assert (a.adopted, a.cfg, a.meta, a.n_evals) == (b.adopted, b.cfg, b.meta, b.n_evals)
    assert sum(sizes) == b.n_evals and max(sizes) == 10   # never re-asks cached windows (confirm only adds the unscreened 10)

import json


# ---------------------------------------------------------------- Phase 19 items: funnel sizes, gaming, history, rollback
def test_default_funnel_is_24_starts_10_screen_3_finalists():
    c = B.SearchConfig()
    assert (c.n_start, c.n_screen, c.n_top) == (24, 10, 3)
    calls = {}
    def ev(w, cfg, m):
        calls.setdefault(B.fingerprint(cfg, m), set()).add(w["id"])
        return sigma_eval(w, cfg, m)
    B.BasisSearch(ev, wins(30), CFG0, META0, B.SearchConfig(seed=2)).run()        # default spaces, default sizes
    assert len(calls) <= 25                                                        # incumbent + at most 24 distinct draws
    assert sum(len(v) == 30 for v in calls.values()) == 4                          # incumbent + top 3 on every window
    assert sum(len(v) == 10 for v in calls.values()) == len(calls) - 4


def test_search_refuses_a_leverage_only_candidate_and_the_firewall_alone_would_pass_it(monkeypatch):
    def ev(w, c, m):
        rng = np.random.default_rng(zlib.crc32(w["id"].encode()))
        if c["k"] == CFG0["k"]:
            return O.week_row(0.008 + 0.045 * rng.standard_normal(52))              # honest, small edge
        return O.week_row(0.075 * rng.standard_normal(52))                          # zero edge, levered into the band
    cfg = B.SearchConfig(seed=3)
    r = B.BasisSearch(ev, wins(30), CFG0, META0, cfg, CS, MS).run()
    assert not r.adopted
    assert any("gaming: no_edge" in c.verdict for c in r.candidates if c.confirm is not None)
    monkeypatch.setattr(O, "gaming_flags", lambda rows, *a, **k: [])
    r2 = B.BasisSearch(ev, wins(30), CFG0, META0, cfg, CS, MS).run()
    v = [c.verdict for c in r2.candidates if c.confirm is not None]
    assert v and not any("gaming" in x or x.startswith("firewall:") for x in v), v   # the tiers alone let it through the firewall


def test_search_refuses_a_cash_hiding_candidate():
    def ev(w, c, m):
        rng = np.random.default_rng(zlib.crc32(w["id"].encode()))
        if c["k"] == CFG0["k"]:
            return O.week_row(0.008 + 0.03 * rng.standard_normal(52))
        w_ = np.where(rng.random(52) < 0.55, rng.normal(0, 0.001, 52), rng.choice([-1, 1], 52) * 0.07)
        return O.week_row(w_)
    r = B.BasisSearch(ev, wins(30), CFG0, META0, B.SearchConfig(seed=4), CS, MS).run()
    assert not r.adopted and any("cash_hiding" in c.verdict for c in r.candidates if c.confirm is not None)


def _adopted_result(seed=1):
    r = B.BasisSearch(sigma_eval, wins(20), CFG0, META0, B.SearchConfig(seed=seed), CS, MS).run()
    assert r.adopted
    return r


def test_history_records_adoptions_and_lineage(tmp_path):
    h = B.BasisHistory(tmp_path / "basis.json", CFG0, META0)
    assert h.current["version"] == 1 and h.current["kind"] == "initial" and h.current["cfg"] == CFG0
    r = _adopted_result()
    v2 = h.adopt(r)
    assert v2["version"] == 2 and v2["parent"] == 1 and v2["cfg"] == r.cfg and v2["score"]["t1"] > 0
    assert h.lineage() == [2, 1] and h.verify()
    kept = B.BasisSearch(sigma_eval, wins(2), CFG0, META0, B.SearchConfig(seed=0), CS, MS).run()
    with pytest.raises(B.HistoryError):
        h.adopt(kept)                                                            # a non-adoption cannot be recorded


def test_rollback_reinstalls_an_older_basis_without_deleting_anything(tmp_path):
    h = B.BasisHistory(tmp_path / "basis.json", CFG0, META0)
    h.adopt(_adopted_result())
    back = h.rollback(why="fresh windows got worse")
    assert back["kind"] == "rollback" and back["cfg"] == CFG0 and back["version"] == 3 and back["parent"] == 2
    assert len(h.entries) == 3 and h.get(2)["kind"] == "adopted" and h.verify()
    assert h.lineage() == [3, 2, 1] and "fresh windows got worse" in back["reason"]
    with pytest.raises(B.HistoryError):
        h.rollback(to_version=3)                                                 # cannot roll back to where you already are
    with pytest.raises(B.HistoryError):
        h.rollback(to_version=42)
    fresh = B.BasisHistory(cfg=CFG0)
    with pytest.raises(B.HistoryError):
        fresh.rollback()                                                         # v1 has no parent


def test_history_persists_and_tampering_is_detected(tmp_path):
    path = tmp_path / "basis.json"
    h = B.BasisHistory(path, CFG0, META0)
    h.adopt(_adopted_result())
    again = B.BasisHistory(path)
    assert again.entries == h.entries and again.current["version"] == 2            # reload does not start a new history
    data = json.loads(path.read_text())
    data[1]["cfg"]["k"] = 1 if data[1]["cfg"]["k"] != 1 else 2                     # edit an adopted basis after the fact
    path.write_text(json.dumps(data))
    with pytest.raises(B.HistoryError):
        B.BasisHistory(path)
    data = json.loads(path.read_text())
    path.write_text(json.dumps(data[1:]))                                          # drop the first entry
    with pytest.raises(B.HistoryError):
        B.BasisHistory(path)
    empty = B.BasisHistory()
    with pytest.raises(B.HistoryError):
        empty.current


def _fresh_rows(sd, n=12, seed=0, mean=0.008):
    rng = np.random.default_rng(seed)
    return [O.week_row(mean + sd * rng.standard_normal(52)) for _ in range(n)]


def test_review_rolls_back_only_on_reliable_evidence():
    good, bad = _fresh_rows(0.07, seed=1), _fresh_rows(0.02, seed=2)
    roll, why = B.review_current_basis(prev_rows=good, cur_rows=bad)               # current is much worse: roll back
    assert roll and "previous basis is better" in why
    roll, why = B.review_current_basis(prev_rows=bad, cur_rows=good)               # current is better: never roll back
    assert not roll and "no case" in why
    same = _fresh_rows(0.05, seed=3)
    assert B.review_current_basis(same, same)[0] is False
    assert B.review_current_basis(good[:2], bad[:2])[0] is False                    # too few windows
    assert B.review_current_basis(_fresh_rows(0.05, seed=4), _fresh_rows(0.05, seed=5))[0] is False   # two equal-quality bases: keep
    assert B.review_current_basis([], [])[0] is False
