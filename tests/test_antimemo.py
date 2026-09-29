"""Phase 11 anti-memorisation firewall: genuine lessons pass, a (ticker, date) memoriser and a lesson that only helps
A are rejected, disguise is faithful, and degenerate inputs stay well-formed."""
import numpy as np
import pandas as pd

from engine.antimemo import (disguise, harmed, invariance_check, paired_delta, play_window, run_experiment,
                             DEFAULT_CFG)
from test_lessons import make_panel, score_fn

CFG = {"boot": 150, "n_disguises": 2}
PARAMS = {"max_lessons": 4}


def _windows(seed=11, b_trap=True):
    XA, yA = make_panel(seed)
    XB, yB = make_panel(seed + 100, trap=b_trap)
    return XA, yA, XB, yB


def test_disguise_is_faithful_and_fresh():
    X, y = make_panel(1, n_days=20, n_tk=10)
    X2, y2, tmap = disguise(X, y, seed=3)
    assert len(set(tmap.values())) == 10 and not set(tmap) & set(tmap.values())
    shifts = (X2.index.get_level_values(0) - X.index.get_level_values(0)).unique()
    assert len(shifts) == 1 and shifts[0].days % 7 == 0 and shifts[0].days > 0
    assert np.array_equal(X2.to_numpy(), X.to_numpy()) and np.array_equal(y2.to_numpy(), y.to_numpy())
    X3, _, tmap3 = disguise(X, y, seed=4)
    assert tmap3 != tmap and not X3.index.equals(X2.index)


def test_identity_free_play_is_unchanged_by_disguise():
    X, y = make_panel(2, n_days=40)
    a, _ = play_window(X, y, score_fn)
    X2, y2, _ = disguise(X, y, seed=9)
    b, _ = play_window(X2, y2, score_fn)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_genuine_lessons_survive_disguise_and_unseen_window():
    XA, yA, XB, yB = _windows()
    r = run_experiment(XA, yA, XB, yB, score_fn, cfg=CFG, seed=1, params=PARAMS)
    assert r["lesson_count"] > 0 and r["identity_invariant"] and r["invariance_max_diff"] == 0.0
    assert r["improvement_A"] > 0 and r["improvement_B"] > 0 and r["degradation_B"] == 0.0
    assert abs(r["memorisation_gap"]) < 1e-9                        # same rule, other names and dates: same gain
    assert r["accepted_lessons"] and r["final_improvement_B"] > 0 and r["verdict"] == "kept"
    for key in ("improvement_A", "improvement_A_disguised", "improvement_B", "degradation_B", "lesson_count",
                "accepted_lessons", "rejected_lessons"):
        assert key in r


class Memoriser:
    """Planted cheat: remembers, per (date, ticker), which rows won in window A and boosts them. Perfect on A itself."""
    def __init__(self, frame, chunks=3):
        win = frame[frame["side"] * frame["y"] > 0.02]
        self.boost = {i: 1 + 50.0 for i in win.index}
        self.owner = {i: hash(i[1]) % chunks for i in win.index}
        self.ids = list(range(chunks))
        self.on = set(self.ids)

    def items(self):
        return list(self.on)

    def subset(self, ids):
        m = Memoriser.__new__(Memoriser)
        m.boost, m.owner, m.ids, m.on = self.boost, self.owner, self.ids, set(ids)
        return m

    def factor(self, X):
        return pd.Series([self.boost[i] if i in self.boost and self.owner[i] in self.on else 1.0 for i in X.index],
                         index=X.index)

    def adjust(self, score, X):
        return score * self.factor(X).to_numpy()


def _memo_learner(frame, XA, now):
    return Memoriser(frame)


def test_memoriser_is_caught_by_the_disguise_rerun_alone():
    XA, yA, XB, yB = _windows(12)
    r = run_experiment(XA, yA, XB, yB, score_fn, cfg=CFG, seed=2, learner=_memo_learner, enforce_invariance=False)
    assert r["improvement_A"] > 0.02                                  # it "learned the answer" for A ...
    assert r["improvement_A_disguised"] < 0.25 * r["improvement_A"]   # ... which vanishes under a fresh disguise
    assert r["memorisation_gap"] > 0.01
    assert not r["accepted_lessons"] and all(x["reason"] in ("memoriser", "harms_B", "no_gain_A", "no_support_B")
                                             for x in r["rejected_lessons"])
    assert any(x["reason"] == "memoriser" for x in r["rejected_lessons"])


def test_memoriser_is_caught_structurally_by_invariance_check():
    XA, yA, XB, yB = _windows(13)
    r = run_experiment(XA, yA, XB, yB, score_fn, cfg=CFG, seed=3, learner=_memo_learner)
    assert not r["identity_invariant"] and r["invariance_max_diff"] > 1
    assert not r["accepted_lessons"] and {x["reason"] for x in r["rejected_lessons"]} == {"identity_keyed"}


def test_lesson_that_helps_A_but_harms_B_is_rejected():
    XA, yA = make_panel(14)                                            # trap present: the lesson helps A
    XB, yB = make_panel(114, trap=False)
    reg = (XB["m_vix"] > 0.3) & (XB["f2"] > 0.3)
    yB = yB.where(~reg, 0.05 * XB["f0"] + yB * 0.3)                    # in B the same region is where f0 works BEST
    r = run_experiment(XA, yA, XB, yB, score_fn, cfg=CFG, seed=4, params=PARAMS)
    assert r["improvement_A"] > 0 and r["improvement_B"] < 0 and r["degradation_B"] > 0
    assert not r["accepted_lessons"] and r["verdict"] == "none_kept"
    assert any(x["reason"] == "harms_B" for x in r["rejected_lessons"])


def test_pure_noise_yields_no_lessons_and_a_well_formed_report():
    XA, yA = make_panel(15, trap=False)
    XB, yB = make_panel(115, trap=False)
    yA = pd.Series(np.random.default_rng(1).normal(scale=0.03, size=len(yA)), index=yA.index)
    r = run_experiment(XA, yA, XB, yB, score_fn, cfg=CFG, seed=5, params={"max_lessons": 4, "fdr": 0.01})
    assert r["lesson_count"] == len(r["accepted_lessons"]) + len(r["rejected_lessons"]) or r["lesson_count"] == 0
    assert r["verdict"] == "none_kept" or r["final_improvement_B"] >= 0
    assert r["degradation_B"] >= 0


def test_paired_delta_and_harm_rule_on_known_series():
    idx = pd.RangeIndex(100)
    base = pd.Series(np.random.default_rng(0).normal(0, 0.01, 100), index=idx)
    better = base + 0.01
    worse = base - 0.01
    assert not harmed(paired_delta(better, base, boot=200), DEFAULT_CFG)
    s = paired_delta(worse, base, boot=200)
    assert harmed(s, DEFAULT_CFG) and s["mean"] == -0.01 + 0.0 or abs(s["mean"] + 0.01) < 1e-12
    assert not harmed(paired_delta(base, base, boot=200), DEFAULT_CFG)     # identical: no harm
    e = paired_delta(pd.Series(dtype=float), pd.Series(dtype=float))
    assert e["n"] == 0 and not harmed(e, DEFAULT_CFG)


def test_invariance_check_passes_for_an_empty_book():
    from engine.lessons import LessonBook
    X, _ = make_panel(16, n_days=10, n_tk=5)
    assert invariance_check(LessonBook(), X)[0]


def test_tercile_labeller_uses_only_reference_edges():
    from engine.antimemo import tercile_labeller
    ref = pd.Series(np.arange(300, dtype=float))
    lab = tercile_labeller(ref)
    later = pd.Series([-5.0, 150.0, 999.0], index=list("abc"))
    assert lab(later).tolist() == ["low", "mid", "high"]              # a wild later window cannot move the edges


def test_improvement_by_era_finds_a_planted_period_effect():
    from engine.antimemo import improvement_by_era
    idx = pd.bdate_range("2020-01-01", periods=520)
    base = pd.Series(0.0, index=idx)
    w = base.copy()
    w[idx.year == 2021] = 0.01
    e = improvement_by_era(base, w)
    assert e["improvement"].round(6).tolist() == [0.0, 0.01] and e["n"].sum() == 520
    assert improvement_by_era(pd.Series(dtype=float), pd.Series(dtype=float)).empty


def test_improvement_by_group_attributes_changed_picks_to_the_right_type():
    from engine.antimemo import improvement_by_group
    XA, yA = make_panel(60, n_days=80)
    from engine.lessons import LessonBook, Lesson
    kw = dict(k=5, collect_frame=True)
    _, f0 = play_window(XA, yA, score_fn, None, **kw)

    class OnlyHigh:                                                   # re-weights f2>1 rows only
        def adjust(self, s, X):
            return s * np.where(X["f2"].to_numpy() > 1.0, 0.0, 1.0)
    _, f1 = play_window(XA, yA, score_fn, OnlyHigh(), **kw)
    grp = pd.Series(np.where(XA["f2"] > 1.0, "hi", "lo"), index=XA.index)
    g = improvement_by_group(f0, f1, grp).set_index("type")
    assert g.loc["hi", "picks_with"] < g.loc["hi", "picks_base"] and g.loc["hi", "picks_changed"] > 0
    assert g.loc["hi", "mean_pick_pnl_with"] != g.loc["hi", "mean_pick_pnl_base"] or g.loc["hi", "picks_with"] == 0


def test_long_only_play_never_shorts_and_report_renders_every_required_output():
    XA, yA, XB, yB = _windows(70)
    _, fr = play_window(XA, yA, lambda Xd: Xd["f0"].rank(pct=True), long_only=True, collect_frame=True)
    assert (fr["side"] == 1.0).all() and fr.groupby(level=0)["taken"].sum().eq(5).all()
    from engine.antimemo import report_markdown
    r = run_experiment(XA, yA, XB, yB, score_fn, cfg=CFG, seed=6, params=PARAMS,
                       type_fn=lambda Z: pd.Series(np.where(Z["f3"] > 0, "up", "down"), index=Z.index))
    md = report_markdown(r)
    for needle in ("improvement on A", "disguised A", "unseen B", "degradation on B", "lessons:", "accepted:", "rejected:"):
        assert needle in md
    assert set(r["era_B"]) and {t["type"] for t in r["type_B"]} == {"up", "down"}
    assert r["episode_categories"] and "mining_rejections" in r


def test_memoriser_check_is_not_fooled_by_a_lesson_that_is_neutral_everywhere():
    class Neutral:
        def items(self): return [0]
        def subset(self, ids): return self
        def factor(self, X): return pd.Series(1.0, index=X.index)
        def adjust(self, s, X): return s
    XA, yA, XB, yB = _windows(71)
    r = run_experiment(XA, yA, XB, yB, score_fn, cfg=CFG, seed=7, learner=lambda f, X, n: Neutral())
    assert not r["accepted_lessons"] and r["rejected_lessons"][0]["reason"] == "no_gain_A"


def test_disguise_kinds_change_only_what_they_claim():
    X, y = make_panel(80, n_days=15, n_tk=8)
    Xn, _, _ = disguise(X, y, seed=1, kind="names")
    Xd, _, _ = disguise(X, y, seed=1, kind="dates")
    assert Xn.index.get_level_values(0).equals(X.index.get_level_values(0))
    assert not set(Xn.index.get_level_values(1)) & set(X.index.get_level_values(1))
    assert Xd.index.get_level_values(1).equals(X.index.get_level_values(1))
    assert not Xd.index.get_level_values(0).equals(X.index.get_level_values(0))


class TickerMemo:
    """Planted cheat #2: a per-ticker boost learned from window A (keyed on names only, never on dates)."""
    def __init__(self, frame, top=12):
        m = (frame["side"] * frame["y"]).groupby(level=1).mean().nlargest(top)
        self.boost = {t: 20.0 for t in m.index}

    def items(self): return [0]
    def subset(self, ids): return self
    def factor(self, X): return pd.Series([self.boost.get(t, 1.0) for t in X.index.get_level_values(1)], index=X.index)
    def adjust(self, s, X): return s * self.factor(X).to_numpy()


class NoLesson(TickerMemo):
    def __init__(self, frame): self.boost = {}
    def items(self): return []
    def subset(self, ids): return self


def _alpha_panel(seed, alpha_seed=5):
    X, y = make_panel(seed, trap=False)
    a = pd.Series(np.random.default_rng(alpha_seed).normal(0, 0.02, 40), index=[f"T{i:02d}" for i in range(40)])
    return X, y + X.index.get_level_values(1).map(a).to_numpy()      # every ticker carries a persistent alpha


def test_ticker_keyed_memory_is_attributed_to_names_not_dates():
    XA, yA = _alpha_panel(81)
    XB, yB = _alpha_panel(181)
    r = run_experiment(XA, yA, XB, yB, score_fn, cfg=CFG, seed=8, learner=lambda f, X, n: TickerMemo(f),
                       enforce_invariance=False)
    assert r["improvement_A"] > 0.005
    assert r["improvement_A_dates_only"] > 0.8 * r["improvement_A"]           # dates never mattered to it
    assert r["improvement_A_names_only"] < 0.2 * r["improvement_A"]           # renaming the stocks removed the gain
    assert not r["accepted_lessons"] and r["rejected_lessons"][0]["reason"] == "memoriser"


def test_holdout_window_c_scores_the_kept_book_once_and_flags_harm():
    XA, yA, XB, yB = _windows(82)
    XC, yC = make_panel(282)
    ok = run_experiment(XA, yA, XB, yB, score_fn, cfg=CFG, seed=9, params=PARAMS, XC=XC, yC=yC)
    assert ok["accepted_lessons"] and ok["holdout_C"]["mean"] > 0 and not ok["holdout_C"]["harmed"]
    XC2, yC2 = make_panel(283, trap=False)
    reg = (XC2["m_vix"] > 0.3) & (XC2["f2"] > 0.3)
    yC2 = yC2.where(~reg, 0.06 * XC2["f0"])                                    # in C the region rewards the model
    bad = run_experiment(XA, yA, XB, yB, score_fn, cfg=CFG, seed=9, params=PARAMS, XC=XC2, yC=yC2)
    assert bad["accepted_lessons"] and bad["improvement_B"] > 0                # B (same regime as A) let it through ...
    assert bad["holdout_C"]["mean"] < 0 and bad["holdout_C"]["harmed"]         # ... only C exposes the drift
    none = run_experiment(XA, yA, XB, yB, score_fn, cfg=CFG, seed=9, learner=lambda f, X, n: NoLesson(f), XC=XC, yC=yC)
    assert none["holdout_C"]["mean"] == 0.0 and not none["holdout_C"]["harmed"]


def test_optimism_control_separates_a_real_effect_from_what_overfitting_can_fake():
    from engine.antimemo import optimism_control
    XA, yA, XB, yB = _windows(83)
    r = run_experiment(XA, yA, XB, yB, score_fn, cfg={**CFG, "null_reps": 3}, seed=10, params=PARAMS)
    n = r["null_control"]
    assert len(n["null_improvements"]) == 3 and n["exceeds_null"] and n["p_value"] <= 0.25
    assert n["null_mean"] < 0.5 * r["improvement_A"]
    XN, yN = make_panel(84, trap=False)
    yN = pd.Series(np.random.default_rng(84).normal(scale=0.03, size=len(yN)), index=yN.index)
    o = optimism_control(XN, yN, score_fn, {**DEFAULT_CFG, "null_reps": 3}, 1, 0.02, PARAMS)   # a claimed 2% gain on noise
    assert o["null_max"] < 0.02 and o["exceeds_null"]                          # the null spread is what a claim is judged by
    lo = optimism_control(XN, yN, score_fn, {**DEFAULT_CFG, "null_reps": 3}, 1, -1.0, PARAMS)
    hi = optimism_control(XN, yN, score_fn, {**DEFAULT_CFG, "null_reps": 3}, 1, 10.0, PARAMS)
    assert lo["p_value"] == 1.0 and hi["p_value"] == 0.25 and not lo["exceeds_null"]     # p is bounded by 1/(reps+1)
