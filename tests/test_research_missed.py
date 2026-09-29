"""Tests for engine/research/missed.py (RESEARCH_BRAIN_CONTRACT C66 section 6/8, canon C67). Synthetic worlds only; each test plants
an effect the engine must find, a null it must NOT find anything in, or the empty/degenerate case."""
import dataclasses
import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest

from engine.learning import missed_winners as MW
from engine.research import missed as M
from engine.research.core import Availability, FirewallBreach, Knowability, MaturedRecord, Namespace, ResearchState

FEATS = ["f0", "f1", "f2", "f3", "f4", "f5"]
START = dt.date(2020, 1, 6)


def make_day(i, rng, n=400, planted=True, skill=False, k=10, lag=2):
    """A cross-section. Planted: f0&f1 both high -> up mover half the time; f2 low & f5 high -> down mover; f4 high -> the mover
    reverses next day. skill=True makes the system's own score use f0+f1, so it owns the planted winners."""
    X = rng.random((n, len(FEATS)))
    up_sig = (X[:, 0] > 0.8) & (X[:, 1] > 0.8)
    dn_sig = (X[:, 2] < 0.12) & (X[:, 5] > 0.8)
    up = rng.random(n) < np.where(up_sig & planted, 0.5, 0.03)
    dn = (rng.random(n) < np.where(dn_sig & planted, 0.5, 0.03)) & ~up
    band = (rng.random(n) < 0.06) & ~up & ~dn
    fwd = np.clip(rng.normal(0, 0.02, n), -0.045, 0.045)
    fwd[up] = rng.uniform(0.07, 0.15, up.sum())
    fwd[dn] = -rng.uniform(0.07, 0.15, dn.sum())
    sgn = rng.choice([-1, 1], n)
    fwd[band] = sgn[band] * rng.uniform(0.05, 0.069, band.sum())
    score = (X[:, 0] + X[:, 1] if skill else X[:, 3]) + rng.normal(0, 0.3, n)
    order = np.argsort(-score, kind="stable")
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(1, n + 1)
    paths = M.PATHS
    obs = []
    d0, d1 = START + dt.timedelta(days=i), START + dt.timedelta(days=i + lag)
    for j in range(n):
        path = None
        if abs(fwd[j]) >= 0.05:
            if planted and X[j, 4] > 0.7 and rng.random() < 0.85:
                path = "reversal"
            else:
                path = paths[int(rng.choice(4, p=[0.4, 0.2, 0.25, 0.15]))]
        obs.append(M.MoveObs(cid=f"c{i}-{j}", decided_at=str(d0), matured_at=str(d1), fwd=float(fwd[j]),
                             features={f: float(X[j, c]) for c, f in enumerate(FEATS)}, picked=bool(rank[j] <= k), score=float(score[j]),
                             rank=int(rank[j]), next_path=path))
    return M.DayBook(str(d0), str(d1), tuple(obs), tuple(FEATS), k=k, era="A" if i < 30 else "B",
                     context={"m_x": float(rng.normal())})


def now_after(day):
    return M.default_now(day)


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(7)
    days = [make_day(i, rng) for i in range(45)]
    state, reports = M.replay(days, FEATS)
    return days, state, reports


@pytest.fixture(scope="module")
def null_world():
    rng = np.random.default_rng(8)
    days = [make_day(i, rng, planted=False) for i in range(45)]
    state, reports = M.replay(days, FEATS)
    return days, state, reports


def mo(**kw):
    base = dict(cid="x", decided_at="2020-02-03", matured_at="2020-02-05", fwd=0.10, features={"f0": 0.5})
    base.update(kw)
    return M.MoveObs(**base)


P = M.Params()


# ---------------------------------------------------------------- parameters and data model
def test_default_params_valid_and_bad_weights_caught():
    assert P.validate() == []
    bad = dataclasses.replace(P, weights={**P.weights, "magnitude": 0.9})
    assert any("sum" in e or "outside" in e for e in bad.validate())
    missing = dataclasses.replace(P, weights={"magnitude": 1.0})
    assert any("nine" in e for e in missing.validate())
    with pytest.raises(ValueError):
        M.new_state(FEATS, dataclasses.replace(P, weights={"magnitude": 1.0}))
    with pytest.raises(ValueError):
        M.new_state([])


def test_moveobs_validation_catches_identity_and_bad_values():
    assert mo().validate() == []
    assert any("identity" in e for e in mo(features={"ticker": 0.5}).validate())
    assert any("rank in [0, 1]" in e for e in mo(features={"f0": 1.7}).validate())
    assert any("follow" in e for e in mo(matured_at="2020-02-03").validate())
    assert any("pred_dir" in e for e in mo(pred_dir=3).validate())
    assert any("path class" in e for e in mo(next_path="sideways").validate())
    assert any("kind" in e for e in mo(info=(M.InfoItem("astrology", "x"),)).validate())


def test_norm_path_maps_aliases_and_unknown_stays_none():
    assert M.norm_path("spiked") == "expansion" and M.norm_path("Reversed") == "reversal" and M.norm_path(None) is None
    assert M.norm_path("mystery") is None


def test_availability_never_reads_missing_timestamp_as_known():
    a = lambda **kw: M.availability(M.InfoItem("event", "e", **kw), "2020-02-03", "2020-02-05")
    assert a(available_at="2020-02-01") == Availability.KNOWN_BEFORE_EVENT
    assert a(available_at="2020-02-03") == Availability.KNOWN_BEFORE_EVENT
    assert a(available_at="2020-02-03", after_close=True) == Availability.SIMULTANEOUS
    assert a(available_at="2020-02-04") == Availability.KNOWN_ONLY_AFTER_EVENT
    assert a(available_at=None) == Availability.UNAVAILABLE
    assert a(available_at=None, expected=False) == Availability.UNCERTAIN


def test_dayfrom_frame_hashes_names_and_ranks_features():
    df = pd.DataFrame({"a": [1.0, 5.0, 3.0, 9.0], "b": [4.0, 3.0, 2.0, 1.0], "fwd": [0.1, -0.1, 0.0, 0.2],
                       "picked": [True, False, False, True], "rank": [1, 4, 3, 2]}, index=["AAA", "BBB", "CCC", "DDD"])
    day = M.DayBook.from_frame(df, "2020-03-02", "2020-03-04", ["a", "b"], k=2)
    assert day.validate() == []
    assert all("AAA" not in o.cid for o in day.obs)
    assert day.obs[3].features["a"] == 1.0 and day.obs[0].features["a"] == 0.25
    assert day.obs[0].picked and not day.obs[1].picked and day.obs[1].rank == 4
    assert len({o.cid for o in day.obs}) == 4


# ---------------------------------------------------------------- capture ladder
def test_capture_winner_ladder():
    c = lambda **kw: M.classify_capture(mo(**kw), P)
    assert c(picked=True).capture == M.Capture.PREDICTED
    assert c(picked=True, weight=0.02).capture == M.Capture.PARTIAL_UNDERSIZED
    assert c(pred_prob=0.8).capture == M.Capture.PARTIAL_MOVEMENT_ONLY
    assert c(pred_prob=0.8, pred_dir=-1).capture == M.Capture.ANTI_PREDICTED
    assert c(rank=13).capture == M.Capture.PARTIAL_NEAR_BAND
    assert c(rank=13).coverage > c(rank=15).coverage > 0
    assert c(rank=200).capture == M.Capture.MISSED
    assert c(fwd=0.02).capture == M.Capture.NOT_A_MOVE


def test_capture_loser_ladder_separates_skill_from_luck():
    c = lambda **kw: M.classify_capture(mo(fwd=-0.12, **kw), P)
    assert c(risk_flag=True).capture == M.Capture.PREDICTED
    assert c().capture == M.Capture.LUCKY_AVOID and c().coverage < 0.5
    assert c(picked=True, realised=-0.03).capture == M.Capture.PARTIAL_PROTECTED
    assert c(picked=True, realised=-0.11).capture == M.Capture.MISSED
    assert c(picked=True, pred_dir=1, confidence=0.9).capture == M.Capture.ANTI_PREDICTED
    assert c(picked=True, risk_flag=True, weight=0.02).capture == M.Capture.PARTIAL_UNDERSIZED
    assert M.loss_incurred(mo(fwd=-0.12, picked=True), 10) > 0 and M.loss_incurred(mo(fwd=-0.12), 10) == 0


# ---------------------------------------------------------------- signatures and novelty
def test_signature_roundtrip_and_undistinguished_zero():
    X = np.array([[0.05, 0.5, 0.95, 0.5, 0.5, 0.5], [0.5] * 6], dtype=np.float32)
    codes = M.signature_codes(X, P)
    assert codes[1] == 0 and codes[0] > 0
    assert M.decode_signature(int(codes[0]), FEATS) == (("f0", "low"), ("f2", "high"))
    assert M.decode_signature(0, FEATS) == ()
    assert M.signature_codes(np.zeros((0, 6), np.float32), P).size == 0


def test_signature_uses_only_the_most_extreme_features():
    X = np.array([[0.01, 0.02, 0.03, 0.04, 0.5, 0.5]], dtype=np.float32)
    assert len(M.decode_signature(int(M.signature_codes(X, P)[0]), FEATS)) == P.sig_top


def test_novelty_index_empty_full_then_decays():
    idx = M.NoveltyIndex(3)
    v = np.array([1.0, 0.0, 0.0])
    assert idx.novelty(v) == 1.0
    idx.add(v)
    assert idx.novelty(v) == pytest.approx(0.0)
    assert idx.novelty(np.array([0.0, 1.0, 0.0])) == pytest.approx(1.0)
    idx.add(np.zeros(3))
    assert len(idx) == 1


def test_pattern_match_directional_and_partial():
    pat = M.PatternSig("p1", 1, {"f0": (0.8, 1.0), "f1": (0.8, 1.0)})
    assert pat.match({"f0": 0.9, "f1": 0.9}) == 1.0
    assert 0 < pat.match({"f0": 0.9, "f1": 0.7}) < 1.0
    assert pat.match({"f0": 0.9}) == 0.5
    assert M.best_pattern_match(mo(features={"f0": 0.9, "f1": 0.9}), [pat], -1) == (0.0, None)
    assert M.best_pattern_match(mo(features={"f0": 0.9, "f1": 0.9}), [pat], 1) == (1.0, "p1")


# ---------------------------------------------------------------- the pre-move audit cannot see the future
def test_view_excludes_days_matured_at_or_after_the_decision(world):
    days, _, _ = world
    store = M.LiftStore(FEATS, P)
    for d in days[:10]:
        store.add_day(d)
    v_before = store.view(days[8].decided_at)
    ev_before = v_before.evidence(days[8].obs[0].features, "up")
    for d in days[10:20]:
        store.add_day(d)
    v_after = store.view(days[8].decided_at)
    assert v_after.n_days == v_before.n_days
    assert v_after.evidence(days[8].obs[0].features, "up") == ev_before
    assert store.view(days[8].decided_at).n_days == 6          # matured strictly before day 8 (2-day lag): days 0..5
    with pytest.raises(ValueError):
        store.add_day(days[3])                                   # history is append-only


def test_planted_effect_is_called_predictable_and_null_is_not(world, null_world):
    days, state, _ = world
    view = state.lift.view(days[-1].decided_at)
    planted = [o for o in days[-1].obs if o.features["f0"] > 0.85 and o.features["f1"] > 0.85 and o.fwd >= 0.07]
    assert planted, "world must contain planted movers on the last day"
    kinds = [M.assess(o, view, P).knowability for o in planted]
    assert sum(k == Knowability.PREDICTABLE for k in kinds) >= 0.8 * len(kinds)
    ev = view.evidence(planted[0].features, "up")
    assert ev.enough and ev.lift > 2 and {c[0] for c in ev.contributors} >= {"f0", "f1"}
    nd, nstate, _ = null_world
    nview = nstate.lift.view(nd[-1].decided_at)
    movers = [o for o in nd[-1].obs if o.fwd >= 0.07]
    nk = [M.assess(o, nview, P).knowability for o in movers]
    assert sum(k == Knowability.PREDICTABLE for k in nk) <= max(1, 0.1 * len(nk))
    assert M.predictable_share(state) > 3 * max(M.predictable_share(nstate), 0.01)


def test_thin_history_is_unknown_with_zero_confidence_not_unpredictable():
    view = M.LiftStore(FEATS, P).view("2020-06-01")
    rep = M.assess(mo(features={f: 0.9 for f in FEATS}), view, P)
    assert rep.knowability == Knowability.UNKNOWN and rep.confidence_in_classification == 0.0 and rep.p_mover is None
    assert "thin" in rep.reasons[0]
    assert rep.future_information_used_by_auditor[0] == "outcome:fwd"


def test_data_failure_and_external_cause_and_unavailable(world):
    days, state, _ = world
    view = state.lift.view(days[-1].decided_at)
    plain = {f: 0.5 for f in FEATS}
    assert M.assess(mo(features=plain, missing_frac=0.7), view, P).knowability == Knowability.DATA_FAILURE
    ext = mo(features=plain, info=(M.InfoItem("event", "earnings_shock", available_at="2020-02-04", strength=0.9),))
    rep = M.assess(ext, view, P)
    assert rep.knowability == Knowability.EXTERNALLY_CAUSED
    assert any("earnings_shock" in s for s in rep.future_information_used_by_auditor)
    lacking = mo(features=plain, info=(M.InfoItem("filing", "missing_filing", available_at=None),))
    assert M.assess(lacking, view, P).knowability == Knowability.INFORMATIONALLY_UNAVAILABLE
    assert M.assess(mo(features=plain), view, P).knowability == Knowability.UNKNOWN


def test_unused_pre_move_information_makes_a_miss_potentially_predictable(world):
    days, state, _ = world
    view = state.lift.view(days[-1].decided_at)
    o = mo(features={f: 0.5 for f in FEATS}, info=(M.InfoItem("filing", "insider_buy", available_at="2020-02-01", strength=0.8),))
    rep = M.assess(o, view, P)
    assert rep.knowability == Knowability.POTENTIALLY_PREDICTABLE and "insider_buy" in rep.information_that_would_have_been_available


def test_knowability_record_travels_only_through_the_gate(world):
    days, state, reports = world
    rep = next(r for r in reports if r.audits)
    recs = M.matured_records(rep, "2026-09-29T00:00:00", seed=1)
    assert recs and all(isinstance(r, MaturedRecord) and r.namespace == Namespace.MATURED_RESEARCH for r in recs)
    r = recs[0]
    with pytest.raises(FirewallBreach):
        r.gate(r.matured_at)                                     # matures ON now: not yet known
    assert r.gate(M.default_now(days[0]) + dt.timedelta(days=400))["kind"] == "knowability"
    gate = M.ReleaseGate([as_year(r.matured_at)])
    with pytest.raises(FirewallBreach):
        gate.release(recs, "2030-01-01")                         # same-year rerun leak
    gate.set_replaying([1999])
    assert len(gate.release(recs, "2030-01-01")) == len(recs)


def as_year(s):
    return int(str(s)[:4])


# ---------------------------------------------------------------- engine step: maturity, streaming, resume
def test_step_processes_only_days_matured_strictly_before_now():
    rng = np.random.default_rng(1)
    days = [make_day(i, rng, n=120) for i in range(3)]
    st = M.new_state(FEATS)
    for d in days:
        M.submit(st, d)
    assert M.step(st, days[0].matured_at) == []                  # matured ON now: still the future
    assert len(st.pending) == 3
    out = M.step(st, as_date_plus(days[1].matured_at, 1))
    assert [r.decided_at for r in out] == [days[0].decided_at, days[1].decided_at] and len(st.pending) == 1
    with pytest.raises(FirewallBreach):
        M._process(st, days[2], days[2].matured_at)
    with pytest.raises(ValueError):
        M.submit(st, days[0])                                    # already processed


def as_date_plus(s, n):
    return dt.date.fromisoformat(s) + dt.timedelta(days=n)


def test_submit_rejects_malformed_days():
    st = M.new_state(FEATS)
    bad = M.DayBook("2020-01-06", "2020-01-05", (), tuple(FEATS))
    with pytest.raises(ValueError):
        M.submit(st, bad)
    with pytest.raises(ValueError):
        M.submit(st, M.DayBook("2020-01-06", "2020-01-08", (), ("zz",)))


def test_empty_day_produces_an_empty_report_not_an_error():
    st = M.new_state(FEATS)
    M.submit(st, M.DayBook("2020-01-06", "2020-01-08", (), tuple(FEATS)))
    rep = M.step(st, "2020-01-10")[0]
    assert rep.n_moves == 0 and rep.top == () and rep.snapshot.n_names == 0 and math.isnan(rep.catch_rate(1))
    assert M.select_exceptions(M.DayBook("2020-01-06", "2020-01-08", (), tuple(FEATS)), P) == ()
    assert st.stream.questions("2026-09-29", "2020-01-08") == []
    assert M.summary(st)["moves"] == 0 and M.symmetry([rep])["verdict"] == "INSUFFICIENT"
    assert M.stream_health(st) == {"topics": 0, "verdict": "EMPTY"}


def test_exceptions_keep_movers_picks_and_near_band_only(world):
    days, state, _ = world
    d = days[10]
    exc = M.select_exceptions(d, P)
    assert 0 < len(exc) < len(d.obs) / 2
    cids = {o.cid for o in exc}
    assert all(o.cid in cids for o in d.obs if o.picked)
    biggest = sorted((o for o in d.obs), key=lambda o: -abs(o.fwd))[:5]
    assert all(o.cid in cids for o in biggest)
    assert [o.cid for o in exc] == sorted(cids)
    assert len(state.exceptions) == len(state.done) == 45


def test_planted_world_report_counts_add_up(world):
    _, state, reports = world
    for r in reports:
        n = sum(sum(c.values()) for c in r.captures.values())
        assert n == r.n_moves
        assert r.n_unknowable_misses + r.n_learnable_misses <= r.n_moves
        assert r.namespace == Namespace.MATURED_RESEARCH
    txt = M.render_report(reports[-1])
    assert "MISSED OPPORTUNITIES" in txt and "winners" in txt and "losers" in txt


def test_sweep_resumes_after_a_kill_and_matches_an_uninterrupted_run(tmp_path):
    rng = np.random.default_rng(3)
    days = [make_day(i, rng, n=150) for i in range(14)]
    full, _ = M.replay(days, FEATS)
    ckpt = tmp_path / "ck.npz"
    a = M.new_state(FEATS)
    gen = M.sweep(a, days, checkpoint=ckpt, every=5)
    for _ in range(7):
        next(gen)                                                # "killed" after 7 days; the checkpoint holds the first 5
    b = M.load_checkpoint(ckpt)
    assert len(b.done) == 5
    list(M.sweep(b, days, checkpoint=ckpt, every=5))
    assert b.done == full.done
    assert b.stream.to_json() == full.stream.to_json()
    assert np.allclose(b.lift.view("2030-01-01").counts, full.lift.view("2030-01-01").counts)
    assert b.sigs.rows == full.sigs.rows and b.tally.cells == full.tally.cells
    assert not (tmp_path / "ck.npz.tmp").exists()


def test_checkpoint_roundtrip_preserves_everything(world, tmp_path):
    _, state, _ = world
    M.save_checkpoint(state, tmp_path / "s.npz")
    back = M.load_checkpoint(tmp_path / "s.npz")
    assert back.p == state.p and back.names == state.names and back.done == state.done
    assert back.stream.to_json() == state.stream.to_json()
    assert np.array_equal(back.novelty.buf, state.novelty.buf)
    assert back.near_up.stable() == state.near_up.stable()
    assert back.filters.rows == state.filters.rows and back.paths.days == state.paths.days
    assert M.summary(back)["moves"] == M.summary(state)["moves"]


def test_replay_is_deterministic(world):
    days, state, _ = world
    again, _ = M.replay(days, FEATS)
    assert again.stream.to_json() == state.stream.to_json()


# ---------------------------------------------------------------- ranking
def make_opp(cid, score, **kw):
    crit = {c: 0.5 for c in M.CRITERIA}
    d = dict(cid=cid, decided_at="2020-02-03", direction=1, capture=M.Capture.MISSED, coverage=0.0, fwd=0.1, knowability=Knowability.PREDICTABLE,
             signature=5, sig_id="s", criteria=crit, score=score, reason="UNKNOWN", effect=M.DecisionEffect.NONE, loss_cost=0.0, pattern_id=None)
    d.update(kw)
    return M.Opportunity(**d)


def test_ranking_ties_do_not_depend_on_input_order_or_name():
    opps = [make_opp(f"z{i}", 0.5) for i in range(20)]
    a = [x.cid for x in M.rank_opportunities(opps)]
    b = [x.cid for x in M.rank_opportunities(list(reversed(opps)))]
    assert a == b and a != sorted(a)


def test_unknowable_and_data_failure_rank_below_predictable_and_loss_boost_lifts_losers(world):
    _, state, reports = world
    rep = next(r for r in reversed(reports) if len(r.top) > 5)
    kinds = {x.knowability for x in rep.top}
    assert all(x.score == 0.0 for x in rep.top if x.knowability == Knowability.DATA_FAILURE)
    ranked = [x.score for x in rep.top]
    assert ranked == sorted(ranked, reverse=True)
    assert Knowability.PREDICTABLE in kinds or Knowability.POTENTIALLY_PREDICTABLE in kinds
    boosted = dataclasses.replace(P, loss_boost=1.0)
    assert P.loss_boost > boosted.loss_boost


def test_ranking_scores_the_nine_criteria_in_range(world):
    _, _, reports = world
    xs = [x for r in reports for x in r.top]
    assert xs
    for x in xs:
        assert set(x.criteria) == set(M.CRITERIA)
        assert all(0.0 <= v <= 1.0 for v in x.criteria.values()) and 0.0 <= x.score <= 1.0
    assert any(x.criteria["repeatability"] > 0 for x in xs) and any(x.criteria["novelty"] < 1.0 for x in xs)
    assert "score" in M.explain_rank(xs[0], P.weights)


def test_pareto_fronts_and_rank_stability():
    good = make_opp("g", 0.9, criteria={c: 0.9 for c in M.CRITERIA})
    bad = make_opp("b", 0.1, criteria={c: 0.1 for c in M.CRITERIA})
    mixed = make_opp("m", 0.5, criteria={**{c: 0.1 for c in M.CRITERIA}, "magnitude": 0.95})
    fr = M.pareto_fronts([good, bad, mixed])
    assert fr[0] == 0 and fr[1] > fr[0] and fr[2] == 0
    assert M.pareto_fronts([]) == [] and M.rank_stability([], P) == {}
    st = M.rank_stability([good, bad, mixed], P, seed=1, n=20, top=1)
    assert st["g"] == 1.0 and st["b"] == 0.0 and all(0 <= v <= 1 for v in st.values())


def test_criteria_redundancy_detects_duplicated_criteria():
    rng = np.random.default_rng(0)
    opps = []
    for i in range(30):
        v = float(rng.random())
        crit = {c: float(rng.random()) for c in M.CRITERIA}
        crit["magnitude"], crit["information"] = v, v
        opps.append(make_opp(f"o{i}", 0.5, criteria=crit))
    red = M.criteria_redundancy(opps)
    assert any({a, b} == {"magnitude", "information"} for a, b, _ in red)
    assert M.criteria_redundancy(opps[:3]) == []


# ---------------------------------------------------------------- why missed: winners (base module) and losers (mirror)
def test_winner_rejection_reasons_come_from_the_base_analyzer():
    o_filter = mo(cid="a", filters_hit=("risk_cap",), eligible=False, rank=4)
    o_rank = mo(cid="b", rank=13, features={"f0": 0.5})
    o_dir = mo(cid="c", rank=50, pred_dir=-1)
    o_blank = mo(cid="d", rank=300)
    day = M.DayBook("2020-02-03", "2020-02-05", (o_filter, o_rank, o_dir, o_blank), ("f0",), k=10)
    rej = M.explain_missed_winners(list(day.obs), day, P)
    assert rej["a"].primary == MW.RR.OVER_AGGRESSIVE_RISK_FILTER
    assert rej["b"].primary == MW.RR.WRONG_RANKING
    assert rej["c"].primary == MW.RR.DIRECTION_DISAGREEMENT
    assert rej["d"].primary in (MW.RR.WRONG_RANKING, MW.RR.UNKNOWN)
    assert set(rej) == {"a", "b", "c", "d"}


def test_loss_analyzer_reasons_and_precedence():
    la = M.LossAnalyzer(P)
    L = lambda **kw: la.explain(mo(fwd=-0.12, picked=True, **kw))
    assert L(risk_flag=True).primary == M.LossReason.RISK_FLAG_OVERRIDDEN
    assert L(gap=-0.11, risk_flag=True).primary == M.LossReason.GAP_UNAVOIDABLE          # a stop could not have helped
    assert L(gap=-0.11).avoidable_by_stop is False and L(risk_flag=True).avoidable_by_stop is True
    assert L(pred_dir=1, confidence=0.9).primary == M.LossReason.DIRECTION_WRONG
    assert L(anti_context=True).primary == M.LossReason.CONTEXT_IGNORED
    assert L(reliability=0.1).primary == M.LossReason.BAD_RELIABILITY_IGNORED
    assert L(confidence=0.95).primary == M.LossReason.OVERCONFIDENT_BET
    assert L(realised=-0.12).primary == M.LossReason.STOP_TOO_LOOSE
    assert L().primary == M.LossReason.NO_RISK_SIGNAL
    assert L(missing_frac=0.6, realised=-0.02).primary == M.LossReason.INSUFFICIENT_EVIDENCE
    with pytest.raises(ValueError):
        la.explain(mo(fwd=-0.12))
    assert la.explain_all([mo(fwd=-0.12), mo(cid="y", fwd=-0.12, picked=True), mo(cid="z", fwd=0.2, picked=True)]).keys() == {"y"}


# ---------------------------------------------------------------- near misses
def test_near_miss_ledger_finds_a_planted_separator_and_ignores_noise():
    rng = np.random.default_rng(5)
    led = M.NearMissLedger(FEATS)
    noise = M.NearMissLedger(FEATS)
    for _ in range(12):
        diffs = rng.normal(0, 0.2, (8, 6))
        diffs[:, 0] += 0.35                                       # winners just under the cut sit higher on f0 than same-rank controls
        led.add(diffs)
        noise.add(rng.normal(0, 0.2, (8, 6)))
    assert led.stable() == ["f0"]
    assert noise.stable() == []
    led.add(np.zeros((0, 6)))
    assert led.stats().shape[0] == 6


def test_paired_differences_use_rank_matching():
    rows = []
    for i in range(1, 30):
        rows.append(mo(cid=f"w{i}", fwd=0.10, rank=10 + i, features={"f0": 0.9}))
        rows.append(mo(cid=f"l{i}", fwd=-0.02, rank=10 + i, features={"f0": 0.4}))
    d = M.paired_differences(rows, ["f0"], 10, P, M.UP)
    assert len(d) > 5 and np.allclose(d, 0.5)
    assert M.paired_differences([], ["f0"], 10, P).shape == (0, 1)
    own = [mo(cid=f"a{i}", fwd=-0.1, picked=True, rank=i, features={"f0": 0.2}) for i in range(1, 5)] + \
          [mo(cid=f"b{i}", fwd=0.1, picked=True, rank=i, features={"f0": 0.8}) for i in range(1, 5)]
    assert np.allclose(M.paired_differences(own, ["f0"], 10, P, M.DOWN), -0.6)


def test_near_misses_flagged_informative_only_with_evidence_or_stable_separator(world):
    days, state, reports = world
    near = [n for r in reports for n in r.near_misses]
    assert near and {n.source for n in near} <= {"rank", "prob", "partial"}
    assert all(n.gap >= 0 for n in near)
    plain = mo(cid="n1", rank=13, features={"f0": 0.5})
    out = M.find_near_misses([plain], 10, P, {"n1": M.assess(plain, state.lift.view("2030-01-01"), P)}, ["f0"])
    assert out and out[0].informative is False and "nothing" in out[0].why


# ---------------------------------------------------------------- research stream
def test_stream_promotes_the_planted_signature_and_questions_are_identity_free(world):
    _, state, _ = world
    st = state.stream
    assert len(st) > 0
    promising = st.top(5, [ResearchState.PROMISING])
    assert promising, "the planted f0/f1 signature recurs on many days and is enriched among movers"
    feats = {f for e in promising for f, _ in e.features}
    assert {"f0", "f1"} & feats
    qs = M.stream_questions(state, "2026-09-29T00:00:00", 5)
    assert qs and all(q.problem in (M.Problem.VOLATILITY, M.Problem.LOSS_AVOIDANCE) for q in qs)
    from engine.learning import trader_view as TV
    assert all(TV.find_violations({"t": q.text}) == [] for q in qs)
    assert len({q.question_id for q in qs}) == len(qs)


def test_stream_drops_a_question_that_would_carry_an_identity():
    names = ["AAPL_2008", "f1", "f2"]
    stream = M.ResearchStream(names, P)
    code = int(M.signature_codes(np.array([[0.95, 0.5, 0.5]], dtype=np.float32), P)[0])
    e = M.StreamEntry("sid", code, 1, M.decode_signature(code, names), ResearchState.QUEUED, 1, 1, 3, 1, 1.0, 0.5, 0.3, 0.0, 1, 0, (), (), ())
    stream.entries["sid"] = e
    assert stream.questions("2026-09-29", "2020-01-01") == []


def test_stream_state_machine_cancel_dormant_and_promote():
    names = ["f0", "f1", "f2"]
    stats = M.SigStats(names, P)
    code = int(M.signature_codes(np.array([[0.95, 0.95, 0.5]], dtype=np.float32), P)[0])

    def opp(k, day):
        return make_opp(f"o{day}-{k}", 0.4, signature=code, sig_id="S", knowability=k, decided_at=f"2020-03-{day:02d}")

    unk = M.ResearchStream(names, P)
    for day in range(1, 8):
        unk.update([opp(Knowability.UNKNOWN, day), opp(Knowability.EXTERNALLY_CAUSED, day)], stats)
    assert unk.entries["S"].state == ResearchState.CANCELLED and unk.entries["S"].history[-1][1] == "CANCELLED"

    dorm = M.ResearchStream(names, P)
    dorm.update([opp(Knowability.PREDICTABLE, 1)], stats)
    assert dorm.entries["S"].state == ResearchState.QUEUED
    dorm.update([opp(Knowability.PREDICTABLE, 2)], stats)
    assert dorm.entries["S"].state == ResearchState.EXPLORING
    for _ in range(P.dormant_after):
        dorm.update([], stats)
    assert dorm.entries["S"].state == ResearchState.DORMANT
    dorm.update([opp(Knowability.PREDICTABLE, 3)], stats)
    assert dorm.entries["S"].state == ResearchState.EXPLORING          # a dormant topic wakes when it recurs

    dropped = M.ResearchStream(names, P)
    dropped.update([opp(Knowability.DATA_FAILURE, 1), make_opp("z", 0.9, signature=0, sig_id="Z")], stats)
    assert len(dropped) == 0                                             # data failures and undistinguished rows never enter


def test_stream_json_roundtrip(world):
    _, state, _ = world
    blob = state.stream.to_json()
    back = M.ResearchStream.from_json(blob, state.p)
    assert back.to_json() == blob and back.top(3) == state.stream.top(3)


def test_stream_health_and_topic_report(world):
    _, state, _ = world
    h = M.stream_health(state)
    assert h["topics"] == len(state.stream) and h["verdict"] in ("HEALTHY", "MONOCULTURE", "STALE") and h["up_topics"] + h["down_topics"] == h["topics"]
    e = state.stream.top(1)[0]
    rep = M.topic_report(state, e.sig_id)
    assert rep["observations"] == e.n_obs and rep["enrichment"] > 0
    with pytest.raises(KeyError):
        M.topic_report(state, "nope")


# ---------------------------------------------------------------- symmetry, priority, filters, regimes, breakdowns
def test_symmetry_flags_a_system_that_only_catches_winners():
    rng = np.random.default_rng(11)
    days = [make_day(i, rng, skill=True) for i in range(30)]
    _, reps = M.replay(days, FEATS)
    s = M.symmetry(reps, seed=2)
    assert s["winner_catch"] > s["loser_catch"] and s["verdict"] == "WINNER_BIASED"
    assert M.symmetry(reps[:3])["verdict"] == "INSUFFICIENT"


def test_symmetry_is_symmetric_when_nothing_is_skilled(null_world):
    _, _, reps = null_world
    assert M.symmetry(reps, seed=1)["verdict"] in ("SYMMETRIC", "LOSER_BIASED")


def test_priority_audit_and_tally_breakdowns(world):
    _, state, reports = world
    pa = M.priority_audit(reports)
    assert pa["verdict"] in ("OK", "UNDER_RANKED") and 0 <= pa["top_loser_share"] <= 1
    assert M.priority_audit([])["verdict"] == "NO_DATA"
    by_era = state.tally.breakdown("era")
    assert set(by_era.era) == {"A", "B"}
    row = by_era.iloc[0]
    assert row.moves == row.predicted + row.partial + row.missed
    assert row.learnable_miss + row.unknowable_miss <= row.missed
    assert not state.tally.breakdown("kind").empty
    assert M.Tally().breakdown().empty


def test_filter_ledger_calls_a_filter_that_removes_winners_costly():
    led = M.FilterLedger()
    obs = [mo(cid=f"w{i}", fwd=0.12, filters_hit=("risk_gap",), eligible=False) for i in range(6)] + \
          [mo(cid=f"l{i}", fwd=-0.08, filters_hit=("risk_gap",), eligible=False) for i in range(2)] + \
          [mo(cid=f"g{i}", fwd=-0.2, filters_hit=("good_filter",), eligible=False) for i in range(4)]
    day = M.DayBook("2020-02-03", "2020-02-05", tuple(obs), ("f0",))
    led.add_day(day, P)
    df = led.review().set_index("filter")
    assert df.loc["risk_gap", "verdict"] == "COSTLY" and df.loc["good_filter", "verdict"] == "PAYS"
    assert df.loc["risk_gap", "winners_foregone"] == 6 and M.FilterLedger().review().empty


def test_regime_table_covers_all_days(world):
    _, state, reports = world
    rt = M.regime_table(state)
    assert rt.days.sum() == len(reports) and set(rt.columns) >= {"regime", "up_catch", "down_catch"}
    assert M.regime_table(M.new_state(FEATS)).empty


# ---------------------------------------------------------------- C67 next-day path lab
def test_path_lab_finds_planted_reversal_and_nothing_in_the_null(world, null_world):
    _, state, reports = world
    edge = state.paths.edge(seed=1)
    assert edge["verdict"] == "EDGE" and edge["view_rate"] > edge["majority_rate"]
    assert reports[-1].path.n_labelled > 5 and set(reports[-1].path.distribution) <= set(M.PATHS)
    _, nstate, _ = null_world
    assert nstate.paths.edge(seed=1)["verdict"] == "NO_EDGE"
    assert M.PathLedger().edge()["verdict"] == "NO_DATA"


def test_path_topics_finds_the_dominant_path_of_a_stream_topic(world):
    _, state, _ = world
    topics = M.path_topics(state, n=5, min_obs=8, alpha=0.05)
    for t in topics:
        assert t["path"] in M.PATHS and t["share"] > t["base"] and t["p"] < 0.05


def test_model_path_predictions_are_scored():
    rng = np.random.default_rng(2)
    day = make_day(0, rng, n=200)
    obs = tuple(dataclasses.replace(o, path_pred=o.next_path) if o.next_path else o for o in day.obs)
    day = dataclasses.replace(day, obs=obs)
    pd_ = M.analyse_paths(day, M.LiftStore(FEATS, P).view("2030-01-01"), P)
    assert pd_.n_model == pd_.n_labelled > 0 and pd_.hits_model == pd_.n_model
    assert pd_.hits_view == 0 and pd_.mean_lift_realised is None       # no history yet: not scored, never faked


# ---------------------------------------------------------------- controls and audits
def test_null_control_planted_versus_shuffled_outcomes():
    rng = np.random.default_rng(21)
    days = [make_day(i, rng, n=300) for i in range(28)]
    out = M.null_control(days, FEATS, seed=4, reps=1)
    assert out["real_predictable_share"] > 2 * max(out["null_predictable_share"], 0.01)
    assert out["clean"] is True and out["real_path_edge"] > out["null_path_edge"]


def test_shuffle_outcomes_keeps_marginals_and_breaks_alignment():
    rng = np.random.default_rng(3)
    day = make_day(0, rng, n=200)
    sh = M.shuffle_outcomes(day, np.random.default_rng(0))
    assert sorted(o.fwd for o in sh.obs) == sorted(o.fwd for o in day.obs)
    assert [o.features for o in sh.obs] == [o.features for o in day.obs]
    assert sum(a.fwd != b.fwd for a, b in zip(sh.obs, day.obs)) > 100


def test_audit_catches_a_feature_that_is_the_outcome_and_passes_clean_days():
    rng = np.random.default_rng(9)
    day = make_day(0, rng, n=400)
    assert M.audit_day(day, P).ok
    f = day.fwd()
    rk = pd.Series(f).rank(pct=True).to_numpy()
    leaky = tuple(dataclasses.replace(o, features={**o.features, "f5": float(rk[i])}) for i, o in enumerate(day.obs))
    bad = dataclasses.replace(day, obs=leaky)
    a = M.audit_day(bad, P)
    assert not a.ok and a.leaks[0][0] == "f5" and abs(a.leaks[0][1]) > 0.9
    st = M.new_state(FEATS)
    with pytest.raises(FirewallBreach):
        M.submit_checked(st, bad)
    assert st.pending == []
    assert M.submit_checked(st, day).ok and len(st.pending) == 1
    empty = M.audit_day(M.DayBook("2020-01-06", "2020-01-08", (), ("f0",)), P)
    assert not empty.ok or empty.issues == ("empty day",)
    assert empty.issues == ("empty day",)


def test_audit_flags_structural_problems():
    obs = [mo(cid=f"c{i}", rank=1, fwd=(2.5 if i == 0 else 0.0), features={"f0": 0.5}) for i in range(5)]
    a = M.audit_day(M.DayBook("2020-02-03", "2020-02-05", tuple(obs), ("f0",)), P)
    assert any("constant" in i for i in a.issues) and any("duplicate ranks" in i for i in a.issues) and any("100%" in i for i in a.issues)


def test_discover_distinctions_is_gated_by_now_and_empty_when_thin(world):
    _, state, _ = world
    assert isinstance(M.discover_distinctions(state, "2030-01-01", seed=1), list)
    with pytest.raises(FirewallBreach):
        M.discover_distinctions(state, "2020-01-08")                 # some retained weeks resolve after this `now`
    assert M.discover_distinctions(M.new_state(FEATS), "2030-01-01") == []
    assert len(M.retained_weeks(state)) == len(state.done)


def test_summary_and_opportunity_frame(world):
    _, state, reports = world
    s = M.summary(state)
    assert s["days"] == 45 and s["moves"] > 0 and 0 <= s["unknowable_share"] <= 1 and s["stream"] == len(state.stream)
    df = M.opportunity_frame([x for r in reports for x in r.top])
    assert set(M.CRITERIA) <= set(df.columns) and len(df) > 0
    assert M.opportunity_frame([]).empty


def test_engine_stays_fast_and_small_on_a_market_wide_day():
    rng = np.random.default_rng(1)
    st = M.new_state(FEATS)
    day = make_day(0, rng, n=3000)
    M.submit(st, day)
    rep = M.step(st, "2020-02-01")[0]
    assert rep.snapshot.n_names == 3000
    assert len(st.exceptions[day.decided_at]) < 500                      # the full cross-section is not retained


# ---------------------------------------------------------------- eras, research targets, narrative
def test_era_consistency_marks_topics_seen_in_every_era(world):
    _, state, _ = world
    rows = M.era_consistency(state)
    assert rows and all(0 <= r["era_share"] <= 1 for r in rows)
    consistent = [r for r in rows if r["consistent"]]
    assert consistent and all(r["eras_seen"] == 2 for r in consistent)
    assert set(e.sig_id for e in M.cross_era_topics(state)) >= {r["sig_id"] for r in consistent}
    assert M.era_consistency(M.new_state(FEATS)) == []


def test_research_targets_carry_values_and_unknowable_targets_carry_none(world):
    _, _, reports = world
    rep = next(r for r in reversed(reports) if len(r.top) >= 5)
    ts = M.research_targets(rep, 5)
    assert len(ts) == 5
    for t, x in zip(ts, rep.top):
        assert t["value"].compute_cost > 0 and (t["value"].loss_reduction_value is None) == (not x.is_loss)
        if x.knowability in M.UNKNOWABLE:
            assert t["value"].decision_value == 0.0 and t["value"].information_gain == 0.0
        assert t["problem"] == (M.Problem.LOSS_AVOIDANCE if x.is_loss else M.Problem.VOLATILITY)
    assert M.research_targets(dataclasses.replace(rep, top=()), 3) == []


def test_narrative_names_capture_cause_and_information(world):
    days, state, _ = world
    view = state.lift.view(days[-1].decided_at)
    o = mo(fwd=-0.13, picked=True, risk_flag=True, features={f: 0.5 for f in FEATS},
           info=(M.InfoItem("filing", "insider_sale", available_at="2020-02-01", strength=0.8), M.InfoItem("event", "missing_call", available_at=None)))
    text = M.narrate(o, view, P)
    assert "loss of -13.0%" in text and "RISK_FLAG_OVERRIDDEN" in text and "insider_sale" in text and "missing_call" in text
    assert "non-move" in M.narrate(mo(fwd=0.0), view, P)


def test_coverage_curve_and_half_split_replication(world):
    days, state, reports = world
    cc = M.coverage_curve(reports, window=10)
    assert len(cc) == len(reports) - 9 and (cc.learnable_miss_rate.dropna() >= 0).all()
    assert M.coverage_curve(reports[:3]).empty
    rep = M.half_split_replication(days, FEATS, top=15)
    assert rep["verdict"] == "REPLICATES" and rep["overlap"] > rep["expected_overlap"]
    assert M.half_split_replication(days[:5], FEATS)["verdict"] == "INSUFFICIENT"


def test_half_split_does_not_replicate_in_a_null_world(null_world):
    days, _, _ = null_world
    rep = M.half_split_replication(days, FEATS, top=15)
    real = M.half_split_replication(make_days_for_replication(), FEATS, top=15)
    assert rep["p_overlap"] > real["p_overlap"]


def make_days_for_replication():
    rng = np.random.default_rng(31)
    return [make_day(i, rng) for i in range(45)]


def test_trader_handoff_refuses_research_objects_and_dated_payloads(world):
    days, state, reports = world
    rep = next(r for r in reports if r.audits)
    late = M.default_now(days[0]) + dt.timedelta(days=400)
    for leaked in (rep, rep.audits[0], rep.top[0], state.stream.top(1)[0], {"fwd": 0.1}):
        with pytest.raises(FirewallBreach):
            M.trader_handoff(leaked, late)                       # planted leaks: future label, audit, opportunity, stream entry
    rec = M.matured_records(rep, "2026-09-29T00:00:00")[0]
    assert M.trader_handoff(rec, late)["kind"] == "knowability"
    with pytest.raises(FirewallBreach):
        M.trader_handoff(rec, rec.matured_at)                    # not yet matured
    dated = MaturedRecord("r", rec.matured_at, {"note": "the 2008 crash repeated"}, rec.provenance)
    with pytest.raises(FirewallBreach):
        M.trader_handoff(dated, late)                            # year identity in the payload


def test_reports_export_as_json_lines(world, tmp_path):
    import json
    _, _, reports = world
    n = M.write_reports(reports[:5], tmp_path / "r.jsonl")
    lines = (tmp_path / "r.jsonl").read_text().splitlines()
    assert n == 5 == len(lines)
    row = json.loads(lines[-1])
    assert row["n_moves"] == reports[4].n_moves and set(row["engagement_lift"]) == {"1", "-1"}
    assert M.write_reports([], tmp_path / "e.jsonl") == 0
