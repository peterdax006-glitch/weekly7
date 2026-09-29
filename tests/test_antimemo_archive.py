"""Phase 11 on archived windows: the wrapper, the disguise of a whole window, and the three-arm harness, on SYNTHETIC
windows built with the same shape as scripts/livesim_loop2.load_window (never the sealed archive)."""
import numpy as np
import pandas as pd
import pytest

from engine import policy
from engine.antimemo import (archive_experiment, archive_frame, archive_markdown, default_archive_learner,
                             disguise_window, recall_learner, RecallControl, window_panel, wrap_snaps, _arm)
from engine.lessons import Lesson, LessonBook

CFG = {"k": 3, "exit_q": 0.8, "rebalance_weeks": 1, "brake": None, "max_per_sector": None, "w_model": 1.0,
       "pick": "top", "pool_q": 0.7, "liq_q": 0.0, "vol_filter": False, "stress_thr": None, "stress_k": 2,
       "trend_filter": None, "trend_gross": 0.0}
E_COLS = [f"e_{c}" for c in policy.default_evidence_weights()]
M_COLS = ["m_spy_ma50", "m_spy_ma200", "m_spy_r5", "m_vix", "m_vix_chg5", "m_vix_term", "m_breadth", "m_dispersion"]


def make_window(seed, n_weeks=60, n_tk=60, trap=True, start="2190-01-05", tag=None):
    """Weekly snapshots (Friday closes) whose next-week return follows the model score, except in the trap
    (high vol20 and calm-to-stressed m_vix > 0) where it reverses. Opens equal the prior close, so the realised
    week is exactly the planted return."""
    rng = np.random.default_rng(seed)
    tks = [f"S{i:04d}" for i in range(n_tk)]
    monday = pd.Timestamp(start) + pd.offsets.Week(weekday=0)          # whole ISO weeks: snapshot = Friday close
    sess = pd.bdate_range(monday, periods=5 * (n_weeks + 2))
    C = pd.DataFrame(100.0, index=sess, columns=tks)
    snaps, weekly = {}, {}
    for w in range(1, n_weeks + 1):
        i = 5 * w - 1                                              # snapshot on the last session of week w
        f = pd.DataFrame(rng.normal(size=(n_tk, len(E_COLS) + 5)), index=tks, columns=E_COLS + ["p_move", "evidence", "vol20", "max20", "r5"])
        f["mu_raw"] = rng.normal(size=n_tk)
        f["log_dv"] = np.tile(np.linspace(10, 20, n_tk), 1)
        f["ev_red_flag"] = 0.0
        f["ev_offering"] = 0.0
        vix = rng.normal()
        for c in M_COLS:
            f[c] = vix if c == "m_vix" else 0.0
        score = f["mu_raw"].rank(pct=True)
        in_trap = (f["vol20"].rank(pct=True) > 0.6) & (vix > 0) if trap else pd.Series(False, index=tks)
        r = 0.06 * (score - 0.5) * np.where(in_trap, -1.0, 1.0) + rng.normal(scale=0.02, size=n_tk)
        snaps[str(sess[i].date())] = f[["mu_raw", "p_move"] + E_COLS + ["evidence", "vol20", "max20", "log_dv", "ev_red_flag", "ev_offering", "r5"] + M_COLS]
        for k in range(1, 6):
            C.iloc[i + k] = C.iloc[i] * (1 + r) ** (k / 5)
        weekly[str(sess[i].date())] = r
    O = C.shift(1).bfill()
    return {"id": tag or f"syn{seed}", "snaps": snaps, "closes": C, "opens": O, "ltm": None, "bps": 10,
            "divs": {t: "Finance" if i % 2 else "Technology" for i, t in enumerate(tks)}}, weekly


def test_window_panel_recovers_the_planted_weekly_returns_and_paths():
    w, weekly = make_window(1, n_weeks=8)
    X, Y = window_panel(w)
    d0 = sorted(weekly)[2]
    y = Y.xs(pd.Timestamp(d0), level=0)
    assert np.allclose(y["y"], weekly[d0], atol=1e-9)
    assert (y["mfe"] >= 0).all() and (y["mae"] <= 0).all()
    assert np.allclose(np.maximum(y["mfe"], y["y"].clip(lower=0)), y["mfe"] + 0 * y["y"])       # final close never beats the max
    assert X["score"].between(0, 1).all() and "mu_raw" not in X and "m_vix" in X
    last = Y.xs(pd.Timestamp(sorted(weekly)[-1]), level=0)
    assert last["y"].notna().all()


def test_window_panel_leaves_nan_when_the_window_ends_first():
    w, _ = make_window(2, n_weeks=4)
    w["closes"] = w["closes"].iloc[:-8]                          # cut the final week short
    X, Y = window_panel(w)
    assert Y.xs(pd.Timestamp(sorted(w["snaps"])[-1]), level=0)["y"].isna().all()
    assert archive_frame(X, Y, [])["y"].notna().all() and len(archive_frame(X, Y, [])) < len(X)


def test_wrapper_alone_changes_nothing_and_a_lesson_changes_the_picks():
    w, _ = make_window(3, n_weeks=20)
    X, _ = window_panel(w)
    from engine import adaptive as A
    S_raw = A.replay(CFG, w["snaps"], w["closes"], w["bps"], w["divs"], opens=w["opens"])
    S_wrap = _arm(w, X, None, CFG, {}, False)
    assert S_raw.decisions == S_wrap.decisions and np.allclose(S_raw.weeks, S_wrap.weeks)      # factor 1 = untouched policy
    L = Lesson("dn", [("score", ">", 0.9)], -1, 0.0, "ok", 50, 8, 9, -0.01, 3.0, 0.01, -0.01, 9.5, 0.5, 0, 100)
    b = LessonBook()
    b.lessons = {"dn": L}
    S_les = _arm(w, X, b, CFG, {}, False)
    assert S_les.decisions != S_wrap.decisions                    # a veto on the top decile moves the portfolio
    snaps = wrap_snaps(w["snaps"], b, X)
    d0 = sorted(snaps)[0]
    top = X.xs(pd.Timestamp(d0), level=0)["score"] > 0.9
    sc = X.xs(pd.Timestamp(d0), level=0)["score"]
    assert np.allclose(snaps[d0].loc[top[top].index, "mu_raw"], sc[top] * 0.25)          # veto, clipped at the 0.25 floor
    assert np.allclose(snaps[d0].loc[top[~top].index, "mu_raw"], sc[~top])               # everyone else untouched


def test_disguise_window_renames_and_shifts_everything_consistently():
    w, _ = make_window(4, n_weeks=12, n_tk=20)
    w2 = disguise_window(w, seed=5)
    assert w2["id"] != w["id"] and set(w2["closes"].columns).isdisjoint(w["closes"].columns)
    assert set(w2["divs"]) == set(w2["closes"].columns) and list(w2["divs"].values()) == list(w["divs"].values())
    assert set(w2["opens"].columns) == set(w2["closes"].columns)
    sh = (w2["closes"].index - w["closes"].index).unique()
    assert len(sh) == 1 and sh[0].days % 7 == 0 and sh[0].days > 0
    d = sorted(w["snaps"])[3]
    d2 = str((pd.Timestamp(d) + sh[0]).date())
    assert d2 in w2["snaps"] and np.array_equal(w2["snaps"][d2].to_numpy(), w["snaps"][d].to_numpy())
    assert set(w2["snaps"][d2].index) == set(w2["closes"].columns)
    other = disguise_window(w, seed=6)
    assert set(other["closes"].columns) != set(w2["closes"].columns)
    X, _ = window_panel(w)
    X2, _ = window_panel(w2)
    assert np.allclose(X.to_numpy(), X2.to_numpy()) and X2.index.get_level_values(1)[0].startswith("Z")


@pytest.fixture(scope="module")
def windows():
    return [make_window(10 + i, n_weeks=45, n_tk=50, tag=f"w{i}")[0] for i in range(6)]


def test_archive_experiment_real_lessons_transfer_and_are_not_memorisation(windows):
    res = archive_experiment(windows, CFG, {}, adaptive=False, folds=3, seed=1,
                             params={"min_support": 25, "min_weeks": 5, "min_tickers": 8})
    assert res["parity_ok"] and len(res["windows"]) == 6
    assert res["weekly_gain_a"]["mean"] > 0 and res["weekly_gain_a"]["lo"] > 0       # learned from OTHER windows, still helps
    assert res["weekly_gain_b"]["mean"] > 0
    for r in res["windows"]:                                                        # identity-free: b and c are the same rule
        assert abs(r["gain_b"] - r["gain_c"]) < 1e-9
    assert res["memorisation"]["windows_memorised"] == 0 and abs(res["memorisation"]["b_minus_c_mean"]) < 1e-9
    assert "arm a" in archive_markdown(res) and "| w0 |" in archive_markdown(res)


def test_archive_experiment_sees_a_planted_memoriser_and_never_flags_it_as_transfer(windows):
    res = archive_experiment(windows, CFG, {}, adaptive=False, folds=3, seed=1, learner=recall_learner, parity_windows=0)
    assert res["weekly_gain_b"]["mean"] > 0.005 and res["weekly_gain_b"]["lo"] > 0     # it aced the window it memorised
    assert res["weekly_gain_c"]["mean"] == 0.0 and res["weekly_gain_a"]["mean"] == 0.0   # and nothing else
    m = res["memorisation"]
    assert m["windows_memorised"] == 6 and m["b_minus_c_mean"] > 0.005 and m["b_minus_a_mean"] > 0.005
    assert all(r["picks_changed_c"] == 0 and r["picks_changed_b"] > 0 for r in res["windows"])


def test_archive_experiment_on_windows_with_no_structure_finds_nothing_to_apply():
    ws = [make_window(50 + i, n_weeks=40, n_tk=40, trap=False, tag=f"n{i}")[0] for i in range(4)]
    for w in ws:                                                  # returns unrelated to the score at all
        rng = np.random.default_rng(len(w["id"]) + int(w["id"][1:]))
        w["closes"] = w["closes"].apply(lambda c: c.iloc[0] * np.exp(np.cumsum(rng.normal(0, 0.01, len(c)))))
        w["opens"] = w["closes"].shift(1).bfill()
    res = archive_experiment(ws, CFG, {}, adaptive=False, folds=2, seed=2, params={"min_support": 25, "min_weeks": 5})
    assert res["parity_ok"] and abs(res["memorisation"]["b_minus_c_mean"]) < 1e-9
    assert res["weekly_gain_a"]["lo"] <= 0 <= res["weekly_gain_a"]["hi"]              # nothing to learn: CI straddles zero


def test_recall_control_keys_are_rows_not_rules():
    w, _ = make_window(7, n_weeks=10, n_tk=20)
    X, Y = window_panel(w)
    fr = archive_frame(X, Y, [])
    rc = RecallControl([fr], thresh=0.0)
    assert rc.factor(X).max() == 50.0 and (rc.factor(X)[fr["y"] <= 0] == 1.0).all()
    X2, _ = window_panel(disguise_window(w, 1))
    assert rc.factor(X2).eq(1.0).all()                           # same numbers, other names and dates: nothing fires
