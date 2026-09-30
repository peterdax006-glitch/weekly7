"""F29 (C75 1C, canon C67): completed outcome labels, the sweep that no longer goes idle, per-lens evidence, held-out checks, questions raised
into the research brain, the episode-level truncation audit and resume after a real process kill. Synthetic data only.
Each mechanism has a planted case it must catch, a null case where it must find nothing, and the empty case."""
import dataclasses
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.learning.trader_view import string_reasons
from engine.research import episode_paths as EP
from engine.research import episodes as E
from engine.research import precursors as P

NOW = "2002-06-03"
CFG = P.LabConfig(n_slices=1, lenses=("c2c",), n_perm=4, cluster_months=1, max_controls=20, seed=3)
NARROW = ("volume", "gap")                                     # what the research loop's world feed hands the lab


def narrow_registry(fams=NARROW):
    return P.PrecursorRegistry([s for s in P.default_registry(audit=False).specs.values() if s.family in fams], audit=False)


def world(plant=E.Plant("volume_before", frac=0.7), seed=5, n=90, mover_rate=0.02):
    return E.synthetic_bars(n, 4 * 260, seed=seed, start="1998-01-05", plant=plant, mover_rate=mover_rate)[0]


class CountingLoader:
    def __init__(self, inner):
        self.inner, self.calls = inner, []

    def __call__(self, unit, ecfg, extra_warm, extra_future):
        self.calls.append(unit.uid)
        return self.inner(unit, ecfg, extra_warm, extra_future)


# ------------------------------------------------------------------------------------------------------------------ labels
@pytest.mark.parametrize("side", [1, -1])
def test_collapse_and_acceleration_are_labelled_and_flagged(side):
    bars, expect = EP.planted_path_world(kinds=["collapse", "accelerate", "reverse", "spike"], side=side)
    g = E.build_grid(bars, EP.episode_config_for_planted())
    eps = E.episode_frame(g)
    eps = eps[(eps["date"] == g.dates[EP.MOVE_DAY]) & (eps["b_c2c"] != 0)].reset_index(drop=True)
    d = EP.with_paths(eps, EP.label_paths(g, eps)).set_index("ticker")
    for t, want in expect.items():
        for h, cls in want.items():
            assert d.loc[t, f"cls_{h}"] == cls, (t, h, d.loc[t, f"cls_{h}"], cls)
    assert d.loc["P_collapse", "coll_3"] and not d.loc["P_collapse", "coll_1"] and d.loc["P_collapse", "raw_3"] < -0.10
    assert d.loc["P_accelerate", "accel_3"] and not d.loc["P_accelerate", "coll_5"]
    assert not d.loc["P_reverse", "coll_5"] and not d.loc["P_spike", "accel_5"]                  # a 3% follow-through is neither
    assert d["nd_close_loc"].isin([c.value for c in EP.CloseLoc]).all()


def test_close_location_and_day_shape_classifiers():
    pc = EP.PathConfig()
    assert EP.classify_close_loc(np.array([0.9, 0.5, 0.1, np.nan]), pc).tolist() == ["CLOSED_STRONG", "CLOSED_MIDDLE", "CLOSED_WEAK", "UNKNOWN"]
    side = np.array([1, 1, 1, 1, -1, 1, 0])
    h0, l0 = np.full(7, 110.0), np.full(7, 100.0)
    h1 = np.array([108.0, 112.0, 112.0, 109.0, 112.0, 110.0, 108.0])
    l1 = np.array([102.0, 98.0, 101.0, 98.0, 101.0, 100.0, 102.0])
    got = EP.classify_day_shape(side, h0, l0, h1, l1).tolist()
    assert got == ["INSIDE_DAY", "OUTSIDE_DAY", "BREAKOUT_DAY", "BREAKDOWN_DAY", "BREAKDOWN_DAY", "OVERLAP_DAY", "UNKNOWN"]


def test_next_day_raw_measures_are_signed_by_the_move():
    bars, _ = EP.planted_path_world(kinds=["spike", "reverse"], side=-1)
    g = E.build_grid(bars, EP.episode_config_for_planted())
    eps = E.episode_frame(g)
    eps = eps[(eps["date"] == g.dates[EP.MOVE_DAY]) & (eps["b_c2c"] != 0)].reset_index(drop=True)
    d = EP.with_paths(eps, EP.label_paths(g, eps)).set_index("ticker")
    assert d.loc["P_spike", "nd_o2c"] > 0 and d.loc["P_reverse", "nd_o2c"] < 0                   # down mover: a further fall is favourable
    assert d.loc["P_spike", "nd_rng"] > 0.03 and d.loc["P_spike", "nd_close_loc"] == "CLOSED_STRONG"


def test_label_ledger_adds_up_merges_and_handles_empty():
    bars = world(E.Plant("class_separator"), seed=8, n=60)
    g = E.build_grid(bars, E.EpisodeConfig())
    eps = E.episode_frame(g)
    ep = EP.with_paths(eps, EP.label_paths(g, eps))
    c = EP.label_counts(ep, "c2c")
    n_types = sum(v for k, v in c.items() if k.endswith("|n|episodes"))
    band, side = E.lens_band_side(ep, "c2c")
    assert n_types == int(((band > 0) & (side != 0)).sum())
    for h in (1, 5):
        tab = EP.label_table(c, h)
        cls = [x for x in EP.CLASS_ORDER if x in tab.columns]
        assert np.allclose(tab[cls].sum(axis=1), 1.0)                                          # every observable window has one class
    both = EP.merge_label_counts([c, c])
    assert all(both[k] == 2 * v for k, v in c.items())
    assert EP.label_counts(ep.iloc[:0], "c2c") == {} and EP.label_table({}, 1).empty and EP.mean_table({}).empty
    means = EP.mean_table(c)
    assert "nd_o2c" in means.columns and means["nd_floc"].between(0, 1).all()


def _intraday_day(day: str, ticker: str, path: np.ndarray, o: float = 100.0) -> pd.DataFrame:
    ts = pd.date_range(f"{day} 09:30", periods=len(path), freq="5min")
    cl = o * (1 + path)
    op = np.r_[o, cl[:-1]]
    return pd.DataFrame({"ts": ts, "ticker": ticker, "Open": op, "High": np.maximum(op, cl) * 1.0005, "Low": np.minimum(op, cl) * 0.9995,
                         "Close": cl, "Volume": 1e4})


def test_intraday_bar_labels_find_planted_shapes_and_mark_the_rest_unmeasured():
    B = 78
    x = np.linspace(0, 1, B)
    trend = 0.06 * x
    fade = np.where(x < 0.3, 0.06 * x / 0.3, 0.06 - 0.05 * (x - 0.3) / 0.7)
    vrev = np.where(x < 0.3, -0.03 * x / 0.3, -0.03 + 0.07 * (x - 0.3) / 0.7)
    ib = pd.concat([_intraday_day("2026-07-06", "TRD", trend), _intraday_day("2026-07-06", "FAD", fade), _intraday_day("2026-07-06", "VRV", vrev),
                    _intraday_day("2026-07-07", "TRD", -0.5 * fade), _intraday_day("2026-07-06", "THN", trend[:10])])
    eps = pd.DataFrame({"date": pd.to_datetime(["2026-07-06"] * 5), "ticker": ["TRD", "FAD", "VRV", "THN", "NONE"], "side": [1, 1, 1, 1, 1]})
    sessions = pd.bdate_range("2026-07-01", "2026-07-10")
    lab = EP.intraday_bar_labels(eps, ib, sessions)
    assert lab["ib_d0_shape"].tolist() == ["TREND", "FADE", "V_REVERSAL", "UNMEASURED", "UNMEASURED"]
    assert lab["ib_measured"].tolist() == [True, True, True, False, False]
    assert lab.loc[0, "ib_d1_shape"] != "UNMEASURED" and lab.loc[1, "ib_d1_shape"] == "UNMEASURED"   # next-day bars only exist for TRD
    assert lab.loc[1, "ib_d0_t_fav"] < 150 and lab.loc[0, "ib_d0_t_fav"] > 300                   # the fade peaked early, the trend late
    assert np.isnan(lab.loc[3, "ib_d0_fav"])
    down = EP.intraday_bar_labels(eps.assign(side=-1).iloc[:1], ib, sessions)
    assert down.loc[0, "ib_d0_shape"] != "TREND"                                                 # the same path is adverse for a down mover
    empty = EP.intraday_bar_labels(eps.iloc[:0], ib, sessions)
    assert len(empty) == 0 and "ib_d0_shape" in empty
    assert (EP.intraday_bar_labels(eps, ib.iloc[:0], sessions)["ib_d0_shape"] == "UNMEASURED").all()
    assert set(EP.UNMEASURED_FIELDS) >= {"intraday_path_historic", "pre_and_after_hours", "delisted_outcomes"}


# ------------------------------------------------------------------------------------------------------------------ the idle sweep, fixed
def test_the_feed_sized_sweep_no_longer_goes_idle_and_raises_questions(tmp_path):
    """The research loop's world feed: 3 years, one slice, one lens, 13 volume/gap columns, first evaluation at 8 units. Before F29 it ran
    3 units, never evaluated (3 < 8), and the loop read a `promoted` field the EvalReport did not have: 0 candidates, 0 questions."""
    bars = world(E.Plant("volume_before", frac=0.8), n=80)
    st, store = P.open_state(tmp_path / "lab", [1998, 1999, 2000], CFG, registry=narrow_registry())
    assert st.widen and st.next_eval_at == 8
    reps = [P.step(st, NOW, P.frame_loader(bars), max_units=1, store=store) for _ in range(40)]
    assert sum(len(r.done) for r in reps) > 3                                                     # it kept producing work
    assert set(st.widened) >= {"range", "return", "candle"} and reps[-1].exhausted and not reps[-1].done
    assert len(st.registry.specs) == len(P.default_registry(audit=False).specs)
    evals = [r.evaluation for r in reps if r.evaluation is not None]
    assert evals and st.eval_seq >= 1
    promoted = [c for e in evals for c in e.promoted]
    assert promoted and all(st.candidates[c].status == P.CandidateStatus.CANDIDATE for c in promoted)
    assert any(st.candidates[c].family == "volume" for c in promoted)                             # the planted volume precursor
    questions = [P.to_question(st.candidates[c], "2002-06-01") for c in promoted[:3]]              # exactly what loop.st_precursors does
    assert questions and all(not string_reasons(q.text) for q in questions)
    back, _ = P.open_state(tmp_path / "lab", [1998, 1999, 2000], CFG, registry=narrow_registry())
    assert set(back.registry.specs) == set(st.registry.specs) and back.widened == st.widened    # widening survives a restart
    assert back.units_run == st.units_run and P.step(back, NOW, P.frame_loader(bars)).done == []


def test_without_widening_the_small_sweep_still_gets_judged_when_exhausted():
    st = P.new_state([1998, 1999, 2000], CFG, registry=narrow_registry(), widen=False)
    reps = P.sweep(st, NOW, P.frame_loader(world()))
    assert st.units_run == 3 < st.rules.min_clusters + 5 and st.eval_seq == 1                    # judged once, on exhaustion
    assert reps[-1].exhausted and any(r.evaluation is not None for r in reps)
    assert P.step(st, NOW, P.frame_loader(world())).evaluation is None                           # and never twice on the same evidence


def test_budget_stops_starting_units():
    st = P.new_state([1998, 1999, 2000], CFG, registry=narrow_registry(), widen=False)
    rep = P.step(st, NOW, P.frame_loader(world()), max_units=3, budget_s=0.0)
    assert rep.done == [] and st.units_done() == 0 and not rep.exhausted


def test_each_unit_counts_only_its_own_lens():
    """A unit is (year, slice, lens). Before F29 every unit computed every lens, so with two lenses each lens's evidence was added twice."""
    bars = world(n=70)
    one = P.new_state([1999], CFG, registry=narrow_registry(), widen=False)
    P.sweep(one, NOW, P.frame_loader(bars))
    two = P.new_state([1999], dataclasses.replace(CFG, lenses=("c2c", "rng")), registry=narrow_registry(), widen=False)
    P.sweep(two, NOW, P.frame_loader(bars))
    for key, cell in one.evidence.cells.items():
        a = cell.stacked()[1][:, :, 0, :]
        b = two.evidence.cells[key].stacked()[1][:, :, 0, :]
        assert np.allclose(a, b), key                                                              # the c2c evidence is not doubled
    assert any(k.startswith("rng|") for k in two.evidence.cells) and not any(k.startswith("rng|") for k in one.evidence.cells)
    assert sum(1 for u in two.market) == 1                                                         # date-level scan once per (year, slice)


def test_grouped_steps_read_bars_once_and_equal_ungrouped():
    bars = world(n=40)
    cfg = dataclasses.replace(CFG, lenses=("c2c", "gap"), n_slices=2)
    a = P.new_state([1998, 1999], cfg, registry=narrow_registry(), widen=False)
    la = CountingLoader(P.frame_loader(bars))
    P.sweep(a, NOW, la)
    b = P.new_state([1998, 1999], cfg, registry=narrow_registry(), widen=False)
    lb = CountingLoader(P.frame_loader(bars))
    P.sweep(b, NOW, lb, grouped=True)
    assert a.evidence.digest() == b.evidence.digest() and a.book.fraction_done() == b.book.fraction_done() == 1.0
    assert len(la.calls) == 8 and len(lb.calls) == 4                                              # 2 years x 2 slices, read once each
    assert a.labels.keys() == b.labels.keys() and all(a.labels[k] == b.labels[k] for k in a.labels)


# ------------------------------------------------------------------------------------------------------------------ out of sample, null, questions
YEARS = [1997, 1998, 1999, 2000, 2001]                         # two eras (edge at 2000): discovery before 2001 can pass the era check


@pytest.fixture(scope="module")
def planted():
    bars = E.synthetic_bars(80, 6 * 260, seed=41, start="1996-06-03", plant=E.Plant("volume_before", frac=0.8), mover_rate=0.02)[0]
    st = P.new_state(YEARS, CFG, first_eval_units=1, registry=P.registry_with_decoys(2, seed=41), widen=False)
    P.sweep(st, NOW, P.frame_loader(bars))
    return st


@pytest.fixture(scope="module")
def null():
    bars = E.synthetic_bars(80, 6 * 260, seed=42, start="1996-06-03", plant=E.Plant("none"), mover_rate=0.02)[0]
    st = P.new_state(YEARS, CFG, first_eval_units=1, registry=P.registry_with_decoys(3, seed=42), widen=False)
    P.sweep(st, NOW, P.frame_loader(bars))
    return st


def test_holdout_confirms_the_planted_precursor(planted):
    looks = planted.ledger.total_trials
    tab, s = P.holdout_check(planted, 2001, NOW)
    assert planted.ledger.total_trials > looks                                                     # the discovery look is counted
    vol = tab[(tab["family"] == "volume") & (tab["key"].str.endswith("|ctl|pre")) & (tab["status_discovery"] == "CANDIDATE")]
    assert len(vol) and vol["confirmed"].any() and vol["clusters_hold"].min() >= 12
    assert s["candidates"] >= 1 and s["candidates_confirmed"] >= 1 and s["decoy_confirmed"] == 0


def test_null_panel_stays_within_its_false_positive_budget(null):
    promoted = [c for c in null.candidates.values() if c.status == P.CandidateStatus.CANDIDATE]
    assert promoted == []
    d = P.decoy_report(null)
    assert d["decoy_tests"] > 0 and d["decoy_promoted"] == 0 and d["tracked_share"] <= 2 * null.rules.screen_q
    tab, s = P.holdout_check(null, 2001, NOW)
    assert s["candidates"] == 0 and s["candidates_confirmed"] == 0
    rep, _ = P.raise_questions(null, NOW, statuses=("CANDIDATE",))
    assert rep.questions == ()                                                                     # nothing real, nothing asked


def test_questions_are_raised_through_the_research_brain(planted):
    from engine.research import questions as Q
    rep, led = P.raise_questions(planted, NOW, min_abs_t=3.0)
    assert rep.questions and not rep.refused
    for qo in rep.questions:
        assert isinstance(qo, Q.QuestionObject) and not qo.check() and qo.source == "new_discovery"
        assert any(h.kind == "noise" for h in qo.hypotheses) and not string_reasons(qo.question.text)
    assert {r["qid"] for r in led.rows} == {q.qid for q in rep.questions}
    rep2, _ = P.raise_questions(planted, NOW, ledger=led, min_abs_t=3.0)
    assert len(led.rows) == len(rep.questions)                                                     # asked again: merged, not duplicated
    with pytest.raises(Exception):
        Q.generate(P.question_events(planted, "2099-01-01")[:1], "1990-01-01", Q.QuestionLedger())   # evidence after now is refused
    assert P.question_events(planted, "1990-01-01") == []                                           # nothing has matured before then


# ------------------------------------------------------------------------------------------------------------------ truncation audit
def test_truncation_audit_is_clean_on_the_registry_and_catches_a_planted_leak():
    bars = E.synthetic_bars(50, 420, seed=9, mover_rate=0.02)[0]
    ok = P.truncation_audit(bars, P.default_registry(), n_cuts=3, seed=1)
    assert ok["clean"] and ok["episodes_checked"] > 0 and ok["cells_compared"] > 0 and ok["labels_mature_after_entry"]
    leaky = P.PrecursorRegistry(audit=False)
    nxt = lambda c: np.vstack([c.g.C[1:] / c.g.C[:-1] - 1.0, np.full((1, c.g.C.shape[1]), np.nan)])   # tomorrow's return
    leaky.register(P.PrecursorSpec("tomorrow_ret", "return", nxt), audit=False)
    leaky.register(P.PrecursorSpec("today_vol", "volume", lambda c: np.log1p(c.g.V)), audit=False)
    bad = P.truncation_audit(bars, leaky, n_cuts=3, seed=1)
    assert not bad["clean"] and any(k.startswith("tomorrow_ret@0@post") for k in bad["mismatches"])
    assert not any(k.startswith("today_vol") for k in bad["mismatches"])                          # the honest feature is not blamed
    empty = P.truncation_audit(bars, leaky, cuts=[])
    assert empty["episodes_checked"] == 0 and empty["clean"]


# ------------------------------------------------------------------------------------------------------------------ store drift and a real kill
def test_changed_definitions_archive_the_old_sweep_instead_of_crashing(tmp_path):
    bars = world(n=60)
    st, store = P.open_state(tmp_path / "lab", [1998], CFG, registry=narrow_registry(), widen=False)
    P.step(st, NOW, P.frame_loader(bars), store=store)
    with pytest.raises(P.DefinitionDrift):
        P.open_state(tmp_path / "lab", [1998], CFG, pcfg=EP.PathConfig(collapse_min=0.2), registry=narrow_registry(), on_drift="raise")
    st2, _ = P.open_state(tmp_path / "lab", [1998], CFG, pcfg=EP.PathConfig(collapse_min=0.2), registry=narrow_registry())
    assert st2.units_done() == 0 and any("archived" in n for n in st2.notes)
    aside = [p for p in tmp_path.iterdir() if p.name.startswith("lab.superseded-")]
    assert len(aside) == 1 and (aside[0] / "manifest.json").exists()                               # the old work is kept, intact


CHILD = textwrap.dedent('''
    import sys, time
    sys.path.insert(0, {root!r})
    from engine.research import episodes as E, precursors as P
    bars = E.synthetic_bars(60, 4 * 260, seed=5, start="1998-01-05", plant=E.Plant("volume_before", frac=0.7), mover_rate=0.02)[0]
    reg = P.PrecursorRegistry([s for s in P.default_registry(audit=False).specs.values() if s.family in ("volume", "gap")], audit=False)
    cfg = P.LabConfig(n_slices=2, lenses=("c2c",), n_perm=4, cluster_months=1, max_controls=20, seed=3)
    st, store = P.open_state({lab!r}, [1998, 1999, 2000], cfg, registry=reg, widen=False)
    while True:
        r = P.step(st, "2002-06-03", P.frame_loader(bars), max_units=1, store=store)
        print("unit", r.done, flush=True)
        if not r.done:
            break
        time.sleep(2.0)
''')


def test_resume_after_a_killed_process_matches_an_uninterrupted_run(tmp_path):
    bars = E.synthetic_bars(60, 4 * 260, seed=5, start="1998-01-05", plant=E.Plant("volume_before", frac=0.7), mover_rate=0.02)[0]
    cfg = P.LabConfig(n_slices=2, lenses=("c2c",), n_perm=4, cluster_months=1, max_controls=20, seed=3)
    clean = P.new_state([1998, 1999, 2000], cfg, registry=narrow_registry(), widen=False)
    P.sweep(clean, NOW, P.frame_loader(bars))
    lab = tmp_path / "lab"
    script = tmp_path / "child.py"
    script.write_text(CHILD.format(root=str(Path(__file__).resolve().parent.parent), lab=str(lab)), encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t0 = time.time()
    try:
        while not (lab / "manifest.json").exists() and time.time() - t0 < 60 and proc.poll() is None:
            time.sleep(0.05)
        assert (lab / "manifest.json").exists(), proc.communicate()[0] if proc.poll() is not None else "child too slow"
    finally:
        proc.kill()                                                                                 # a hard kill, mid-sweep
        proc.wait()
    st, store = P.open_state(lab, [1998, 1999, 2000], cfg, registry=narrow_registry(), widen=False)
    had = set(st.book.records)
    assert 1 <= len(had) < 6                                                                        # killed part way through
    loader = CountingLoader(P.frame_loader(bars))
    P.sweep(st, NOW, loader, store=store)
    assert not (had & set(loader.calls)) and len(loader.calls) == 6 - len(had)                      # nothing redone, nothing skipped
    assert st.evidence.digest() == clean.evidence.digest() and st.book.fraction_done() == 1.0
