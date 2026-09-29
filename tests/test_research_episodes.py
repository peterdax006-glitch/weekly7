"""Tests for engine/research/episodes.py, episode_paths.py and precursors.py (canon C67; contract C66 sections 4, 5, 6, 9, 13, 21-23).

Synthetic worlds only. Each mechanism has a planted case it must catch, a null case where it must find nothing, and the empty case:
  * detector          hand-built bars with known bands, measures, glitches, halts and illiquid names; counts exact
  * streaming         chunked run == one-shot run (episodes, counts, payload); out-of-order refused; snapshot/restore
  * path taxonomy     planted paths labelled EXACTLY at horizons 1, 3, 5, both directions; incomplete windows never guessed
  * precursors        a volume spike two sessions before a move is FOUND and promoted; a reversal-vs-continuation separator is FOUND at the
                      move-day anchor; noise-only and decoy features are never promoted; a feature that reads the move day's own close is REFUSED
  * always-on         resume after a crash neither redoes nor skips a unit; wider feature sets extend, never repeat; corrupted stores refused
  * firewall          sessions at/after now refused; candidates released only when matured and never from a replayed year; text is identity-free"""
import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from engine.learning.trader_view import string_reasons
from engine.research import episode_paths as EP
from engine.research import episodes as E
from engine.research import precursors as P
from engine.research.core import FirewallBreach, Knowability, MaturedRecord
from engine.learning.core import Health

NOW = "2002-06-03"
CFG = P.LabConfig(n_slices=1, lenses=("c2c",), n_perm=6, cluster_months=1, max_controls=20, seed=1)


# ------------------------------------------------------------------------------------------------------------------ hand-built worlds
def build_bars(rel: dict, T: int = 130, N: int = 6, seed: int = 0, vol: np.ndarray | None = None, nan_close: set | None = None):
    """OHLCV from relative bars. `rel[(t, j)] = (open, close, high, low)` as fractions of the previous close override a quiet random base."""
    rng = np.random.default_rng(seed)
    o = rng.normal(0, 0.002, (T, N))
    c = np.where(np.arange(T)[:, None] % 2 == 0, 0.008, -0.008) * np.ones((1, N)) + rng.normal(0, 0.001, (T, N))
    h = np.maximum(o, c) + 0.004
    l = np.minimum(o, c) - 0.004
    for (t, j), (oo, cc, hh, ll) in rel.items():
        o[t, j], c[t, j], h[t, j], l[t, j] = oo, cc, hh, ll
    C = np.zeros((T, N))
    prev = np.full(N, 50.0)
    O, H, L = np.zeros((T, N)), np.zeros((T, N)), np.zeros((T, N))
    for t in range(T):
        O[t], C[t], H[t], L[t] = prev * (1 + o[t]), prev * (1 + c[t]), prev * (1 + h[t]), prev * (1 + l[t])
        prev = C[t]
    V = np.full((T, N), 2.0e6) if vol is None else vol
    dates = pd.bdate_range("2021-01-04", periods=T)
    tick = [f"N{j}" for j in range(N)]
    out = {k: pd.DataFrame(v, index=dates, columns=tick) for k, v in (("Open", O), ("High", H), ("Low", L), ("Close", C), ("Volume", V))}
    for (t, j) in (nan_close or set()):
        for k in ("Open", "High", "Low", "Close"):
            out[k].iloc[t, j] = np.nan
    return out


EVENT_T = 100


def detector_world():
    rel = {(EVENT_T, 0): (0.005, 0.06, 0.065, 0.0),            # c2c +6%: 5-10% up
           (EVENT_T, 1): (-0.11, -0.12, -0.10, -0.125),         # c2c -12%, gap -11%: >10% down
           (EVENT_T, 2): (0.07, 0.01, 0.075, 0.0),              # gap +7% that faded to +1%
           (EVENT_T, 3): (0.0, 0.005, 0.04, -0.05),             # range 9%, tiny close move
           (EVENT_T, 5): (0.0, 0.07, 0.075, 0.0)}               # a 7% move in an illiquid name
    vol = np.full((130, 6), 2.0e6)
    vol[:, 5] = 100.0
    return build_bars(rel, vol=vol)


# ------------------------------------------------------------------------------------------------------------------ detector
def test_config_validation_refuses_nonsense():
    assert E.EpisodeConfig().validate() == []
    for bad in (E.EpisodeConfig(lo=0.1, hi=0.05), E.EpisodeConfig(max_move=0.08), E.EpisodeConfig(horizons=(3, 1)), E.EpisodeConfig(vol_min=2),
                E.EpisodeConfig(warm=10), E.EpisodeConfig(measures=("zzz",)), E.EpisodeConfig(n_cohorts=0)):
        assert bad.validate()
        with pytest.raises(E.EpisodeError):
            bad.require_valid()


def test_detector_bands_measures_and_counts_are_exact():
    blk = E.run_in_memory(detector_world())
    eps = blk.episodes
    day = eps[eps["date"] == blk.counts.index[EVENT_T]].set_index("ticker")
    assert set(day.index) == {"N0", "N1", "N2", "N3"}                       # the illiquid N5 is not an episode
    assert day.loc["N0", "b_c2c"] == 1 and day.loc["N1", "b_c2c"] == -2 and day.loc["N2", "b_c2c"] == 0 and day.loc["N3", "b_c2c"] == 0
    assert day.loc["N2", "b_gap"] == 1 and day.loc["N1", "b_gap"] == -2
    assert day.loc["N2", "b_o2c"] == -1                                     # gap-and-fade: closed below its open by ~5.6%
    assert day.loc["N3", "b_rng"] == 1 and day.loc["N3", "b_c2c"] == 0
    assert day.loc["N0", "loc"] > 0.9 and day.loc["N2", "body"] == -1
    row = blk.counts.iloc[EVENT_T]
    assert row["c2c_up_5_10"] == 1 and row["c2c_dn_gt10"] == 1 and row["c2c_dn_5_10"] == 0
    assert row["gap_up_5_10"] == 1 and row["gap_dn_gt10"] == 1
    assert row["rng_up_5_10"] >= 2 and row["rng_dn_5_10"] == 1              # N0 and N3 close up, N2 closes down
    assert row["n_illiquid"] == 1 and row["n_suspect"] == 0
    quiet = blk.counts.drop(blk.counts.index[EVENT_T])
    assert quiet.drop(columns=["n_names", "n_suspect", "n_illiquid"]).to_numpy().sum() == 0    # nothing else counted anywhere


def test_sigma_uses_only_the_days_before_the_move():
    blk = E.run_in_memory(detector_world())
    g = E.build_grid(detector_world(), E.EpisodeConfig())
    row = blk.episodes[(blk.episodes["ticker"] == "N0") & (blk.episodes["date"] == g.dates[EVENT_T])].iloc[0]
    hist = g.c2c[EVENT_T - 60:EVENT_T, 0]
    assert row["sigma"] == pytest.approx(np.std(hist, ddof=1))
    assert row["vol_z"] == pytest.approx(abs(row["c2c"]) / row["sigma"])


def test_glitch_split_and_halt_are_not_movers():
    rel = {(EVENT_T, 0): (-0.5, -0.5, -0.49, -0.51),                        # 2-for-1 split in an unadjusted feed
           (EVENT_T, 1): (0.0, 0.07, 0.075, 0.0)}                            # a real mover, but the name was halted the day before
    bars = build_bars(rel, nan_close={(EVENT_T - 1, 1)})
    blk = E.run_in_memory(bars)
    assert len(blk.episodes[blk.episodes["date"] == blk.counts.index[EVENT_T]]) == 0
    assert blk.counts.iloc[EVENT_T]["n_suspect"] >= 1
    assert E.split_like_mask(np.array([-0.5, -0.33, 0.07]), np.array([0.0, 0.0, 0.0]), np.array([-0.5, -0.33, 0.07])).tolist() == [True, False, False]
    health = E.bars_health(bars)
    assert "glitch bars present" in health["issues"] and health["split_like_bars"] >= 1


def test_empty_and_degenerate_inputs():
    empty = {k: pd.DataFrame(index=pd.DatetimeIndex([]), columns=["A", "B"], dtype=float) for k in E.FIELD_KEYS}
    blk = E.run_in_memory(empty)
    assert len(blk.episodes) == 0 and len(blk.counts) == 0
    assert E.hundreds_report(blk.counts)["days"] == 0
    assert E.repeat_movers(blk.episodes)["episodes"] == 0 and E.concentration_report(blk.episodes)["names"] == 0
    assert len(EP.label_paths(E.build_grid(empty, E.EpisodeConfig()), blk.episodes)) == 0
    with pytest.raises(E.EpisodeError):
        E.build_grid({"Close": pd.DataFrame()}, E.EpisodeConfig())
    assert E.EpisodeStream().feed(empty) == []


def test_hundreds_report_and_day_summaries():
    idx = pd.bdate_range("2020-01-01", periods=10)
    counts = pd.DataFrame({"n_names": 3000, "c2c_up_5_10": [150, 90, 120, 200, 99, 100, 101, 5, 300, 130],
                           "c2c_dn_5_10": [100] * 10}, index=idx)
    rep = E.hundreds_report(counts, 100)
    assert rep["share_days_at_least"]["c2c_up_5_10"] == pytest.approx(0.7)
    assert rep["both_sides_share_days_at_least"] == 1.0 and rep["universe_median"] == 3000
    rate = E.rate_by_era(counts.assign(x=0), ["c2c_up_5_10"])
    assert float(rate["c2c_up_5_10"].iloc[0]) == pytest.approx(counts["c2c_up_5_10"].mean() / 3.0)


def test_episode_summaries_on_synthetic_world():
    bars, _ = E.synthetic_bars(80, 200, seed=2)
    s = E.year_summary(bars, minimum=5)
    assert s["episodes"] > 100 and s["repeat"]["repeat_share"] < 0.5 and s["concentration"]["names"] > 40
    assert s["health"]["issues"] == [] or "calendar holes present" in s["health"]["issues"]
    assert 0 <= s["concentration"]["herfindahl"] <= 1


# ------------------------------------------------------------------------------------------------------------------ streaming
def _stream(bars, sizes, on_block=None, cfg=None):
    s = E.EpisodeStream(cfg or E.EpisodeConfig(), on_block)
    out, a = [], 0
    for n in sizes:
        out += s.feed({k: v.iloc[a:a + n] for k, v in bars.items()})
        a += n
    out += s.flush()
    return out, s


def test_stream_equals_in_memory_for_any_chunking():
    bars, _ = E.synthetic_bars(70, 240, seed=5, plant=E.Plant("volume_before"))
    ref = E.run_in_memory(bars)
    for sizes in ((240,), (37, 60, 100, 43), (120, 1, 1, 118), (10,) * 24):
        out, _ = _stream(bars, sizes)
        eps, counts = E.concat_blocks(out)
        assert E.episode_digest(eps) == E.episode_digest(ref.episodes)
        assert len(eps) == len(ref.episodes) and len(eps) > 20
        pd.testing.assert_frame_equal(counts, ref.counts, check_exact=False, rtol=1e-9)


def test_stream_with_path_payload_equals_in_memory():
    bars, _ = E.synthetic_bars(70, 240, seed=6)
    hook = lambda g, eps, lo, hi: EP.label_paths(g, eps)
    ref = E.run_in_memory(bars, on_block=hook)
    out, _ = _stream(bars, (50, 90, 100), on_block=hook)
    got = pd.concat([b.payload for b in out], ignore_index=True)
    assert len(got) == len(ref.payload)
    cols = [c for c in got.columns if c.startswith("cls_") or c.startswith("ok_")]
    assert (got[cols].to_numpy() == ref.payload[cols].to_numpy()).all()


def test_stream_refuses_disorder_and_field_changes():
    bars, _ = E.synthetic_bars(20, 120, seed=1)
    s = E.EpisodeStream()
    s.feed({k: v.iloc[:60] for k, v in bars.items()})
    with pytest.raises(E.EpisodeError):
        s.feed({k: v.iloc[50:100] for k, v in bars.items()})                # overlaps what was already fed
    with pytest.raises(E.EpisodeError):
        s.feed({k: v.iloc[60:90] for k, v in bars.items() if k != "Volume"})


def test_stream_snapshot_restore_mid_run(tmp_path):
    bars, _ = E.synthetic_bars(50, 200, seed=8)
    ref = E.run_in_memory(bars)
    s = E.EpisodeStream()
    first = s.feed({k: v.iloc[:110] for k, v in bars.items()})
    E.save_stream(s, tmp_path / "s.json")
    s2 = E.load_stream(tmp_path / "s.json")
    rest = s2.feed({k: v.iloc[110:] for k, v in bars.items()}) + s2.flush()
    eps, _ = E.concat_blocks(first + rest)
    assert E.episode_digest(eps) == E.episode_digest(ref.episodes)
    raw = json.loads((tmp_path / "s.json").read_text())
    raw["body"]["sessions_seen"] += 1
    (tmp_path / "s.json").write_text(json.dumps(raw))
    with pytest.raises(E.EpisodeError):
        E.load_stream(tmp_path / "s.json")


def test_stream_emits_only_when_the_look_forward_exists():
    bars, _ = E.synthetic_bars(30, 150, seed=2)
    s = E.EpisodeStream()
    blocks = s.feed({k: v.iloc[:100] for k, v in bars.items()})
    assert blocks and blocks[0].hi_date == bars["Close"].index[100 - 1 - 5]        # five sessions still unseen
    assert s.flush()[0].hi_date == bars["Close"].index[99]


# ------------------------------------------------------------------------------------------------------------------ path taxonomy
@pytest.mark.parametrize("side", [1, -1])
def test_planted_paths_are_labelled_exactly(side):
    bars, expect = EP.planted_path_world(side=side)
    g = E.build_grid(bars, EP.episode_config_for_planted())
    eps = E.episode_frame(g)
    eps = eps[(eps["date"] == g.dates[EP.MOVE_DAY]) & (eps["b_c2c"] != 0)].reset_index(drop=True)
    assert set(eps["ticker"]) == set(expect)
    d = EP.with_paths(eps, EP.label_paths(g, eps))
    for _, r in d.iterrows():
        for h, want in expect[r["ticker"]].items():
            assert r[f"cls_{h}"] == want, (r["ticker"], h, r[f"cls_{h}"], want)
        assert r["ok_5"] and r["net_1"] == pytest.approx(r["r1"])
    sp = d.set_index("ticker")
    assert sp.loc["P_spike", "r1z"] > 1.5 and sp.loc["P_reverse", "r1z"] < -1.5           # signed by the move's own direction
    assert sp.loc["P_expand", "intraday_path"] == "RANGE_EXPANSION" and sp.loc["P_consolidate", "intraday_path"] == "RANGE_CONTRACTION"
    assert sp.loc["P_continue3", "mfe_3"] > sp.loc["P_continue3", "netz_3"] * 0 and sp.loc["P_retrace3", "mae_3"] < 0


def test_unobservable_windows_are_flagged_not_guessed():
    bars, _ = EP.planted_path_world(kinds=["spike", "stop"], n_days=EP.MOVE_DAY + 3)         # only two sessions after the move
    g = E.build_grid(bars, EP.episode_config_for_planted())
    eps = E.episode_frame(g)
    eps = eps[(eps["date"] == g.dates[EP.MOVE_DAY]) & (eps["b_c2c"] != 0)].reset_index(drop=True)
    d = EP.with_paths(eps, EP.label_paths(g, eps))
    assert d["ok_1"].all() and not d["ok_3"].any() and not d["ok_5"].any()
    assert (d["cls_3"] == "UNCLASSIFIED").all() and d["net_3"].isna().all()
    assert (pd.to_datetime(d["matured_at"]) == g.dates[EP.MOVE_DAY + 2]).all()              # only the windows of one and two sessions closed


def test_gap_and_go_gap_and_fade_and_hold():
    pc = EP.PathConfig()
    prev, loc = np.array([100.0, 100.0, 100.0, 100.0, 100.0]), np.array([0.9, 0.2, 0.5, 0.8, 0.1])
    o = np.array([107.0, 107.0, 107.0, 100.5, 93.0])
    c = np.array([110.0, 101.0, 105.0, 101.0, 90.0])
    gap = o / prev - 1
    got = EP.classify_gap(gap, o, c, prev, loc, pc)
    assert got.tolist() == ["GAP_AND_GO", "GAP_AND_FADE", "GAP_HOLD", "NO_GAP", "GAP_AND_GO"]


def test_retrigger_and_excursion_timing_columns():
    bars, _ = EP.planted_path_world(kinds=["expand", "consolidate"])
    g = E.build_grid(bars, EP.episode_config_for_planted())
    eps = E.episode_frame(g)
    eps = eps[(eps["date"] == g.dates[EP.MOVE_DAY]) & (eps["b_c2c"] != 0)].reset_index(drop=True)
    d = EP.with_paths(eps, EP.label_paths(g, eps)).set_index("ticker")
    assert not d["retrig_5"].any()                                                             # no follow-on 5% close-to-close move
    assert d.loc["P_expand", "tmfe_5"] >= 1 and d.loc["P_expand", "tmfe_5"] <= 5
    prof = EP.daily_profile(g, eps)
    assert prof.shape == (2, 5) and np.isfinite(prof).all()
    m = EP.mean_profile(prof)
    assert list(m["day"]) == [1, 2, 3, 4, 5] and (m["n"] == 2).all()


def test_label_sensitivity_and_significance_tables():
    bars, _ = E.synthetic_bars(120, 300, seed=4, plant=E.Plant("class_separator"), mover_rate=0.02)
    g = E.build_grid(bars, E.EpisodeConfig())
    eps = E.episode_frame(g)
    sens = EP.label_sensitivity(g, eps, horizon=1)
    assert 0.0 <= sens["changed"]["tighter"] <= 1.0 and 0.0 <= sens["changed"]["looser"] <= 1.0
    d = EP.with_paths(eps, EP.label_paths(g, eps))
    tab = EP.class_significance(d, "c2c", 1)
    assert len(tab) and tab["lift"].max() > 1.0
    top = tab.sort_values("p_screen").iloc[0]
    assert top["cls"] in ("SPIKED_NEXT_DAY", "REVERSED_NEXT_DAY")                              # the planted follow-through dominates
    ct = EP.class_table(d, "c2c", 1)
    assert abs(ct.groupby("type")["share"].sum() - 1).max() < 1e-9
    assert EP.base_rates(d, 1)["SPIKED_NEXT_DAY"] > 0.05
    tm = EP.transition_matrix(d, 1, 5)
    assert np.allclose(tm.sum(axis=1), 1.0)


def test_path_matured_at_and_firewall_helpers():
    bars, _ = EP.planted_path_world(kinds=["spike"])
    g = E.build_grid(bars, EP.episode_config_for_planted())
    eps = E.episode_frame(g)
    eps = eps[(eps["date"] == g.dates[EP.MOVE_DAY]) & (eps["b_c2c"] != 0)].reset_index(drop=True)
    d = EP.label_paths(g, eps)
    close5 = g.dates[EP.MOVE_DAY + 5]
    assert pd.Timestamp(d["matured_at"].iloc[0]) == close5
    with pytest.raises(FirewallBreach):
        EP.require_matured(d, close5)                                                          # matures ON now: not yet known
    EP.require_matured(d, close5 + pd.Timedelta(days=1))
    assert len(EP.observable(d, close5)) == 0 and len(EP.observable(d, close5 + pd.Timedelta(days=1))) == 1
    with pytest.raises(FirewallBreach):
        E.assert_no_future(eps, g.dates[EP.MOVE_DAY])
    with pytest.raises(E.EpisodeError):
        EP.PathConfig(stop_k=2.0).require_valid()


# ------------------------------------------------------------------------------------------------------------------ registry and leak audit
def test_every_builtin_precursor_passes_the_future_audit():
    reg = P.default_registry()
    assert len(reg.columns()) > 55
    for s in reg.specs.values():
        rep = P.audit_spec(s)
        assert rep.ok, (s.name, rep.reasons)
    cat = P.precursor_catalog(reg)
    assert set(cat["family"]) >= {"volume", "compression", "range", "gap", "candle", "mover_history", "relative", "sector"}


def test_future_leaking_precursors_are_refused():
    def next_day_return(c):                                   # the move day's own close, seen from the day before
        out = np.full(c.g.C.shape, np.nan)
        out[:-1] = c.g.c2c[1:]
        return out

    def centred_mean(c):                                      # a centred window reads tomorrow
        return pd.DataFrame(c.g.C).rolling(5, center=True, min_periods=1).mean().to_numpy()

    def full_sample_z(c):                                     # normalising with the whole sample leaks the future into the past
        return (c.g.C - np.nanmean(c.g.C, axis=0)) / np.nanstd(c.g.C, axis=0)

    reg = P.PrecursorRegistry()
    for fn in (next_day_return, centred_mean, full_sample_z):
        with pytest.raises(FirewallBreach):
            reg.register(P.PrecursorSpec(fn.__name__, "test", fn))
    assert not reg.specs
    honest = P.PrecursorSpec("honest_ret", "return", lambda c: c.g.c2c)
    reg.register(honest)
    with pytest.raises(P.LabError):
        reg.register(honest)
    with pytest.raises(FirewallBreach):
        P.external_spec("bad_events", "event", lambda c: c.g.c2c, pub_lag=-1)
    lagged = P.external_spec("ok_events", "event", lambda c: c.g.c2c, pub_lag=2)
    reg.register(lagged)


def test_leaking_row_gather_is_impossible_from_the_pre_anchor():
    bars, _ = E.synthetic_bars(30, 120, seed=3)
    g = E.build_grid(bars, E.EpisodeConfig())
    panel = {"ret_1": np.arange(g.shape[0] * g.shape[1], dtype=np.float32).reshape(g.shape)}
    ti, nj = np.array([50, 0]), np.array([3, 3])
    pre = P.gather_features(panel, ["ret_1@0", "ret_1@2"], ti, nj, "pre")
    post = P.gather_features(panel, ["ret_1@0"], ti, nj, "post")
    assert pre[0, 0] == panel["ret_1"][49, 3] and pre[0, 1] == panel["ret_1"][47, 3]           # move day 50: anchor row 49
    assert post[0, 0] == panel["ret_1"][50, 3]
    assert np.isnan(pre[1]).all() and post[1, 0] == panel["ret_1"][0, 3]                       # no row before the grid
    with pytest.raises(P.LabError):
        P.gather_features(panel, ["ret_1@0"], ti, nj, "future")


def test_decoys_are_deterministic_noise_and_pass_the_audit():
    spec = P.decoy_spec(0, seed=3)
    assert P.audit_spec(spec).ok
    bars, _ = E.synthetic_bars(40, 120, seed=9)
    g = E.build_grid(bars, E.EpisodeConfig())
    a, b = spec.fn(P.Ctx(g)), P.decoy_spec(0, seed=3).fn(P.Ctx(g))
    assert (a == b).all() and 0.45 < a.mean() < 0.55 and a.min() >= 0 and a.max() < 1
    ret = np.where(np.isfinite(g.c2c), g.c2c, 0.0)
    assert abs(np.corrcoef(a.ravel(), ret.ravel())[0, 1]) < 0.05                                # independent of the market
    other = P.decoy_spec(1, seed=3).fn(P.Ctx(g))
    assert abs(np.corrcoef(a.ravel(), other.ravel())[0, 1]) < 0.1


# ------------------------------------------------------------------------------------------------------------------ contrast mathematics
def test_contrast_recovers_a_planted_shift_and_shuffles_do_not():
    rng = np.random.default_rng(0)
    n = 6000
    strata = np.repeat(np.arange(300), 20)
    cluster = strata // 25
    g = rng.random(n) < 0.3
    F = rng.random((n, 4)).astype(np.float32)
    F[:, 0] += 0.2 * g
    con = P.Contrast(F, strata, cluster)
    T = con.versions(g, 8, np.random.default_rng(1))
    real = P.tensor_stats(T[:, :, 0, :])
    assert real["effect"][0] == pytest.approx(0.2, abs=0.03) and real["t"][0] > 8
    assert np.abs(real["t"][1:]).max() < 4
    null_t = np.stack([P.tensor_stats(T[:, :, v, :])["t"] for v in range(1, 9)])
    assert np.abs(null_t).max() < 4.5 and np.abs(null_t).mean() < 1.5


def test_inactive_rows_take_no_part_in_the_contrast_or_the_shuffle():
    rng = np.random.default_rng(2)
    n = 4000
    strata = np.repeat(np.arange(200), 20)
    cluster = strata // 20
    kind = rng.integers(0, 3, n)                                   # 0 control, 1 group, 2 someone else entirely
    F = rng.random((n, 2)).astype(np.float32)
    F[kind == 2] = 50.0                                            # would ruin the estimate if it leaked in
    F[kind == 1, 0] += 0.15
    con = P.Contrast(F, strata, cluster)
    T = con.versions(kind == 1, 4, np.random.default_rng(3), active=kind != 2)
    real = P.tensor_stats(T[:, :, 0, :])
    assert real["effect"][0] == pytest.approx(0.15, abs=0.04) and abs(real["effect"][1]) < 0.05
    for v in range(1, 5):
        assert abs(P.tensor_stats(T[:, :, v, :])["effect"][0]) < 0.06
    assert real["n1"][0] == (kind == 1).sum() and real["n0"][0] == pytest.approx((kind == 0).sum(), rel=0.02)


def test_clustering_widens_the_uncertainty_of_repeated_days():
    rng = np.random.default_rng(4)
    strata = np.repeat(np.arange(60), 50)
    cluster_fine = strata
    cluster_coarse = strata // 15
    g = rng.random(len(strata)) < 0.4
    day_shift = np.repeat(rng.normal(0, 0.2, 60), 50)                # a shared day effect on the group only
    F = (rng.random((len(strata), 1)) + day_shift[:, None] * g[:, None]).astype(np.float32)
    fine = P.tensor_stats(P.Contrast(F, strata, cluster_fine).versions(g, 1, rng)[:, :, 0, :])
    coarse = P.tensor_stats(P.Contrast(F, strata, cluster_coarse).versions(g, 1, rng)[:, :, 0, :])
    assert coarse["se"][0] > 0 and fine["clusters"][0] == 60 and coarse["clusters"][0] == 4
    assert fine["effect"][0] == pytest.approx(coarse["effect"][0])


def _hand_cell(effects, noise=0.4, seed=0, n_perm=4, first=(1996, 1)):
    """A Cell with one feature whose per-month effect follows `effects`; other features are pure noise."""
    rng = np.random.default_rng(seed)
    cell = P.Cell(1 + n_perm)
    months = len(effects)
    ids = np.array([first[0] * 12 + first[1] - 1 + i for i in range(months)])
    T = np.zeros((months, 3, 1 + n_perm, 4))
    for v in range(1 + n_perm):
        for m in range(months):
            w = 400.0
            for f in range(3):
                e = effects[m] if (f == 0 and v == 0) else rng.normal(0, noise * 0.02)
                T[m, f, v] = [w * (e + rng.normal(0, 0.01)), w, 60.0, 60.0]
    cell.add(["real@0", "noise_a@0", "noise_b@0"], ids, T, 730000)
    return cell, ids


def test_walk_forward_and_era_checks_on_hand_built_evidence():
    rules, cfg = P.EvidenceRules(), dataclasses.replace(CFG, cluster_months=1, era_edges=(1997, 1999))
    steady, ids = _hand_cell([0.06] * 48)
    ids, T = steady.stacked()
    wf = P.walk_forward(T, 0, rules)
    assert wf["enough"] and wf["ok"] and wf["passed"] == wf["tested"] >= 2
    er = P.era_check(ids, T, 0, cfg, rules)
    assert er["enough"] and er["ok"] and len(er["eras"]) >= 2
    early, _ = _hand_cell([0.10] * 24 + [0.0] * 24)                     # an effect that died halfway
    _, Te = early.stacked()
    assert not P.walk_forward(Te, 0, rules)["ok"]
    one_era, ids1 = _hand_cell([0.08] * 12 + [-0.02] * 36)              # strong in the first year only
    ids1, T1 = one_era.stacked()
    assert not P.era_check(ids1, T1, 0, cfg, rules)["ok"]
    short, idss = _hand_cell([0.06] * 3)
    _, Ts = short.stacked()
    assert not P.walk_forward(Ts, 0, rules)["enough"]                   # too little to judge is not a pass


def test_cell_grows_features_without_disturbing_old_ones():
    c = P.Cell(3)
    a = np.ones((2, 1, 3, 4))
    c.add(["f1@0"], np.array([1, 2]), a, 10)
    before = c.stacked()[1].copy()
    c.add(["f1@0", "f2@0"], np.array([2, 3]), np.ones((2, 2, 3, 4)) * 2, 20)
    ids, T = c.stacked()
    assert list(ids) == [1, 2, 3] and c.features == ["f1@0", "f2@0"]
    assert (T[0, 0] == before[0, 0]).all() and (T[1, 0] == 3).all() and (T[2, 1] == 2).all() and (T[0, 1] == 0).all()
    assert c.last_ordinal == 20 and c.units == 2
    with pytest.raises(P.LabError):
        c.add(["f1@0"], np.array([9]), np.ones((1, 1, 5, 4)), 1)


def test_evidence_store_roundtrip_and_checksum(tmp_path):
    st = P.EvidenceStore(4)
    cell, _ = _hand_cell([0.05] * 12)
    st.cells["c2c|5_10_up|ctl|pre"] = cell
    sha = st.save(tmp_path / "e.npz")
    back = P.EvidenceStore.load(tmp_path / "e.npz", sha)
    assert back.digest() == st.digest() and back.n_tests() == 3
    with pytest.raises(P.LabError):
        P.EvidenceStore.load(tmp_path / "e.npz", "0" * 64)
    raw = bytearray((tmp_path / "e.npz").read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    (tmp_path / "bad.npz").write_bytes(bytes(raw))
    with pytest.raises(Exception):
        P.EvidenceStore.load(tmp_path / "bad.npz", sha)


def test_redundancy_groups_and_feature_health():
    rng = np.random.default_rng(0)
    base = rng.random((500, 1)).astype(np.float32)
    F = np.hstack([base, base + rng.normal(0, 0.01, (500, 1)), rng.random((500, 2)), np.full((500, 1), 0.5)]).astype(np.float32)
    F[:250, 2] = np.nan
    cols = ["a", "a_copy", "b", "c", "const"]
    assert P.redundancy_groups(F, cols) == [["a", "a_copy"]]
    h = P.feature_health(F, cols).set_index("column")
    assert h.loc["const", "constant"] and h.loc["b", "finite_share"] == pytest.approx(0.5) and not h.loc["a", "constant"]
    with pytest.raises(P.LabError):
        P.feature_health(F, cols[:-1])


# ------------------------------------------------------------------------------------------------------------------ end to end: planted, separator, null
def _sweep(plant, seed, decoys=3, years=(1998, 1999, 2000), store=None, mover_rate=0.02):
    bars, truth = E.synthetic_bars(120, 4 * 260, seed=seed, start="1998-01-05", plant=plant, mover_rate=mover_rate)
    st = P.new_state(list(years), CFG, first_eval_units=1, registry=P.registry_with_decoys(decoys, seed=seed))
    P.sweep(st, NOW, P.frame_loader(bars), store=store)
    P.evaluate_state(st, NOW, force=True)
    return st, bars


@pytest.fixture(scope="module")
def volume_world():
    return _sweep(E.Plant("volume_before", frac=0.7), seed=11)


@pytest.fixture(scope="module")
def separator_world():
    return _sweep(E.Plant("class_separator"), seed=12, decoys=0)


@pytest.fixture(scope="module")
def null_world():
    return _sweep(E.Plant("none"), seed=13, decoys=3)


def test_planted_volume_precursor_is_found_and_promoted(volume_world):
    st, _ = volume_world
    assert st.book.fraction_done(st.registry.columns()) == 1.0
    cands = [c for c in st.candidates.values() if c.status == P.CandidateStatus.CANDIDATE]
    vol = [c for c in cands if c.family == "volume" and c.comparison == "ctl" and c.anchor == "pre"]
    assert vol, P.lab_report(st)
    best = max(vol, key=lambda c: abs(c.t))
    assert best.effect > 0.15 and best.t > 10 and best.wf_tested >= 2 and len({e[0] for e in best.eras}) >= 2
    spike_two_before = [c for c in vol if c.feature in ("vol_z@1", "vol_burst@1")]              # the plant: volume spike two sessions before
    assert spike_two_before and all(c.effect > 0.2 for c in spike_two_before)
    assert P.knowability_of(best) in (Knowability.POTENTIALLY_PREDICTABLE, Knowability.WEAKLY_PREDICTABLE)
    assert not any(c.status == P.CandidateStatus.CANDIDATE for c in st.candidates.values() if c.feature.startswith(P.DECOY_PREFIX))
    # unrelated families do not become candidates from a volume plant
    assert not any(c.family in ("sector", "event") and c.status == P.CandidateStatus.CANDIDATE for c in st.candidates.values())


def test_planted_world_ledger_and_decoy_accounting(volume_world):
    st, _ = volume_world
    assert st.ledger.verify() == [] and st.ledger.total_trials >= st.evidence.n_tests() * 0.2
    rep = P.decoy_report(st)
    assert rep["decoy_tests"] > 30 and rep["decoy_promoted"] == 0 and rep["tracked_share"] <= rep["screen_q"] + 0.05
    um = P.unknown_map(st)
    assert len(um) and set(um["verdict"]) <= {"candidate(s)", "UNKNOWN: no precursor found", "too thin"}
    reps = P.family_representatives(st)
    assert set(reps) == set(st.candidates) and all(reps[r] == r for r in set(reps.values()))
    rep = P.lab_report(st)
    assert "mover-episode lab" in rep and "MATURED_RESEARCH_STATE" in rep and string_reasons(rep.splitlines()[0]) == []


def test_candidate_text_and_records_are_identity_free_and_gated(volume_world):
    st, _ = volume_world
    for c in st.candidates.values():
        assert string_reasons(c.text) == [] and string_reasons(c.candidate_id) == []
        P.assert_identity_free(c, ["S0001", "S0002"])
    cands = [c for c in st.candidates.values() if c.status == P.CandidateStatus.CANDIDATE]
    q = P.to_question(cands[0], "2026-09-29")
    assert q.problem.value == "VOLATILITY" and "S0" not in q.text and string_reasons(q.text) == []
    early, held_e = P.release(st, "1999-01-01")
    assert early == [] and all("matured" in w or "status" in w or "replay" in w for w in held_e.values())
    recs, held = P.release(st, NOW)
    assert recs and all(isinstance(r, MaturedRecord) for r in recs)
    assert all(r.payload["decision_effect"] == "NONE" for r in recs)
    for r in recs:
        r.gate(NOW)
        with pytest.raises(FirewallBreach):
            r.gate(r.matured_at)                                                                # maturing ON now is not yet known
    same_year, held_y = P.release(st, NOW, replay_windows=[("2000-03-01", "2000-04-01")])
    assert len(same_year) < len(recs) or not recs
    assert any("replayed" in w for w in held_y.values())
    assert P.replay_guard(st, [("2000-03-01", "2000-04-01")]) == {"c2c": [2000]}


def test_candidate_follow_up_timeline_and_health(volume_world):
    st, _ = volume_world
    cid = max((c for c in st.candidates.values() if c.status == P.CandidateStatus.CANDIDATE), key=lambda c: abs(c.t)).candidate_id
    tl = P.candidate_timeline(st, cid)
    assert list(tl["year"]) == [1998, 1999, 2000] and (tl["effect"] > 0.1).all()
    assert P.candidate_health(st, cid, recent_years=1)["health"] in (Health.HEALTHY, Health.DEGRADING)
    everything = P.health_check_all(st)
    assert cid in everything and all(isinstance(v, Health) for v in everything.values())


def test_health_detects_a_finding_that_died():
    cell, ids = _hand_cell([0.12] * 30 + [-0.05] * 18)
    st = P.new_state([1996, 1997, 1998, 1999], CFG, registry=P.PrecursorRegistry(audit=False))
    st.evidence.cells["c2c|5_10_up|ctl|pre"] = cell
    c = P.Candidate("PCx", "c2c|5_10_up|ctl|pre", "c2c", "5_10_up", "ctl", "pre", "real@0", "volume", "higher", 0.06, 5.0, 0.0, 0.0, 0.0, 100, 100, 48,
                    3, 3, (), P.CandidateStatus.CANDIDATE, (), 730000, (1996, 1997, 1998, 1999), 1, "x")
    st.candidates["PCx"] = c
    h = P.candidate_health(st, "PCx", recent_years=1)
    assert h["health"] == Health.BROKEN and h["recent"]["effect"] < 0


def test_reversal_versus_continuation_separator_is_found(separator_world):
    st, _ = separator_world
    pair = "REVERSED_NEXT_DAY~SPIKED_NEXT_DAY@h1"
    hits = [c for c in st.candidates.values() if c.comparison == pair and c.anchor == "post" and c.family == "volume"]
    assert hits, P.lab_report(st)
    best = max(hits, key=lambda c: abs(c.t))
    assert best.effect < -0.15 and best.status == P.CandidateStatus.CANDIDATE                # reversals had LOW move-day volume
    assert P.problem_of(best.comparison).value == "DIRECTION"
    pre = [c for c in st.candidates.values() if c.comparison == pair and c.anchor == "pre" and c.status == P.CandidateStatus.CANDIDATE
           and c.family == "volume"]
    assert not pre                                                                            # before the move nothing distinguishes them
    ctl_up = [c for c in st.candidates.values() if c.comparison == "ctl" and c.status == P.CandidateStatus.CANDIDATE and c.family == "volume"]
    assert not any(c.feature in ("vol_z@1", "vol_burst@1") for c in ctl_up)                   # no volume plant in this world


def test_noise_only_world_finds_nothing_after_correction(null_world):
    st, _ = null_world
    assert st.evidence.n_tests() >= 300
    promoted = [c for c in st.candidates.values() if c.status == P.CandidateStatus.CANDIDATE]
    assert promoted == [], [(c.key, c.feature, round(c.t, 1)) for c in promoted]
    rep = P.decoy_report(st)
    assert rep["decoy_promoted"] == 0
    assert st.ledger.expected_best_null_t() > 3.0                                             # the bar rises with the number of looks
    recs, _ = P.release(st, NOW)
    assert recs == []


def test_evaluation_is_idempotent_on_unchanged_evidence(volume_world):
    st, _ = volume_world
    before = (st.ledger.total_trials, st.eval_seq, len(st.candidates))
    assert P.evaluate_state(st, NOW, force=True) is None
    assert (st.ledger.total_trials, st.eval_seq, len(st.candidates)) == before


# ------------------------------------------------------------------------------------------------------------------ always-on: resume, extend, corrupt
class CountingLoader:
    def __init__(self, inner, boom_on_call=None):
        self.inner, self.calls, self.boom = inner, [], boom_on_call

    def __call__(self, unit, ecfg, extra_warm, extra_future):
        self.calls.append(unit.uid)
        if self.boom is not None and len(self.calls) == self.boom:
            raise RuntimeError("simulated crash while loading")
        return self.inner(unit, ecfg, extra_warm, extra_future)


def _small_state(store_dir=None, years=(1998, 1999, 2000), registry=None):
    bars, _ = E.synthetic_bars(70, 4 * 260, seed=21, start="1998-01-05", plant=E.Plant("volume_before"), mover_rate=0.02)
    cfg = dataclasses.replace(CFG, n_perm=3)
    reg = registry if registry is not None else P.PrecursorRegistry([s for s in P.default_registry(audit=False).specs.values()
                                                                    if s.family in ("volume", "gap")], audit=False)
    return bars, cfg, reg


def test_resume_after_a_crash_neither_redoes_nor_skips_work(tmp_path):
    bars, cfg, reg = _small_state()
    clean = P.new_state([1998, 1999, 2000], cfg, registry=reg)
    P.sweep(clean, NOW, P.frame_loader(bars))
    want = clean.evidence.digest()
    store = P.LabStore(tmp_path / "lab")
    st = P.new_state([1998, 1999, 2000], cfg, registry=reg)
    loader = CountingLoader(P.frame_loader(bars), boom_on_call=2)
    with pytest.raises(RuntimeError):
        P.sweep(st, NOW, loader, store=store)
    assert len(st.book.records) == 1 and len(loader.calls) == 2                               # unit 2 died before committing: no trace of it
    st2, store2 = P.open_state(tmp_path / "lab", [1998, 1999, 2000], cfg, registry=reg)
    assert set(st2.book.records) == set(st.book.records) and st2.evidence.digest() == st.evidence.digest()
    loader2 = CountingLoader(P.frame_loader(bars))
    P.sweep(st2, NOW, loader2, store=store2)
    finished_first = next(iter(st.book.records))
    assert finished_first not in loader2.calls                                                # the finished unit is not loaded again
    assert len(loader2.calls) == 2 and len(set(loader2.calls)) == 2
    assert st2.evidence.digest() == want                                                      # identical to the uninterrupted run
    st3, _ = P.open_state(tmp_path / "lab", [1998, 1999, 2000], cfg, registry=reg)
    assert st3.evidence.digest() == want and st3.book.digest() == st2.book.digest()
    assert P.step(st3, NOW, CountingLoader(P.frame_loader(bars))).done == []                   # nothing left to do, nothing redone


def test_step_visits_the_least_covered_area_first():
    bars, cfg, reg = _small_state()
    cfg = dataclasses.replace(cfg, n_slices=2)
    st = P.new_state([1998, 1999, 2000], cfg, registry=reg)
    loader = CountingLoader(P.frame_loader(bars))
    P.step(st, NOW, loader, max_units=1)
    P.step(st, NOW, loader, max_units=1)
    years = [E.Unit.parse(u).year for u in loader.calls]
    assert len(set(years)) == 2                                                               # the second unit went to a different year
    P.step(st, NOW, loader, max_units=1)
    assert len({E.Unit.parse(u).year for u in loader.calls}) == 3
    assert len(loader.calls) == len(set(loader.calls))


def test_a_wider_feature_set_extends_units_without_double_counting():
    bars, cfg, reg = _small_state()
    st = P.new_state([1998, 1999], cfg, registry=reg)
    P.sweep(st, NOW, P.frame_loader(bars))
    key = next(k for k in st.evidence.cells if k.endswith("|ctl|pre"))
    old_cols = list(st.evidence.cells[key].features)
    _, old_T = st.evidence.cells[key].stacked()
    old_n1 = old_T[:, 0, 0, 2].sum()
    st.registry.register(P.PrecursorSpec("extra_ret", "return", lambda c: c.g.c2c))
    assert st.book.fraction_done(st.registry.columns()) == 0.0
    loader = CountingLoader(P.frame_loader(bars))
    P.sweep(st, NOW, loader)
    assert len(loader.calls) == len(set(loader.calls)) == len(st.book.all_units())
    cell = st.evidence.cells[key]
    assert cell.features[:len(old_cols)] == old_cols and "extra_ret@0" in cell.features
    _, T = cell.stacked()
    assert T[:, 0, 0, 2].sum() == old_n1                                                      # the old feature was not counted twice
    assert T[:, cell.features.index("extra_ret@0"), 0, 2].sum() > 0
    assert P.step(st, NOW, loader).done == []


def test_empty_year_is_recorded_and_the_sweep_moves_on():
    bars, cfg, reg = _small_state()
    st = P.new_state([1990, 1998], cfg, registry=reg)
    rep = P.step(st, NOW, P.frame_loader(bars), max_units=2)
    assert "1990|0|c2c" in st.book.records and st.book.records["1990|0|c2c"].n_episodes == 0
    assert len(rep.done) == 2 and st.units_done() == 2
    dead = lambda u, e, w, f: {k: pd.DataFrame() for k in E.FIELD_KEYS}
    st2 = P.new_state([1998], cfg, registry=reg)
    assert P.step(st2, NOW, dead).done == ["1998|0|c2c"] and st2.book.records["1998|0|c2c"].n_days == 0


def test_a_year_without_its_full_future_waits_and_leaves_no_trace():
    bars, cfg, reg = _small_state()
    st = P.new_state([1998, 2001], cfg, registry=reg)                    # data ends inside 2001
    rep = P.step(st, NOW, P.frame_loader(bars), max_units=5)
    assert rep.waiting == ["2001|0|c2c"] and rep.done == ["1998|0|c2c"]
    assert "2001|0|c2c" not in st.book.records
    st.book.extend_years([2002])
    assert st.book.years == [1998, 2001, 2002]


def test_sessions_at_or_after_now_are_refused():
    bars, cfg, reg = _small_state()
    st = P.new_state([1998], cfg, registry=reg)
    with pytest.raises(FirewallBreach):
        P.step(st, "1999-01-05", P.frame_loader(bars), years_before=1999)                   # the loader's look-forward reaches past now
    assert st.units_done() == 0
    with pytest.raises(FirewallBreach):
        P.assert_before({"Close": pd.DataFrame(index=pd.DatetimeIndex(["2001-01-02"]))}, "2001-01-02")


def test_store_detects_corruption_and_definition_drift(tmp_path):
    bars, cfg, reg = _small_state()
    st = P.new_state([1998], cfg, registry=reg)
    store = P.LabStore(tmp_path / "lab")
    P.step(st, NOW, P.frame_loader(bars), store=store)
    (tmp_path / "lab" / "half_written.tmp999").write_text("junk")
    store.commit(st)
    assert not list((tmp_path / "lab").glob("*.tmp*")) and len(list((tmp_path / "lab").glob("evidence_*.npz"))) == 1
    with pytest.raises(P.LabError):
        store.load(dataclasses.replace(cfg, max_controls=cfg.max_controls + 1), registry=reg)      # a sweep may not change its definitions
    with pytest.raises(P.LabError):
        store.load(cfg, E.EpisodeConfig(lo=0.06), registry=reg)
    m = json.loads(store.manifest_path.read_text())
    m["body"]["seq"] += 5
    store.manifest_path.write_text(json.dumps(m))
    with pytest.raises(P.LabError):
        store.load(cfg, registry=reg)
    store2 = P.LabStore(tmp_path / "lab2")
    store2.commit(st)
    ev = next((tmp_path / "lab2").glob("evidence_*.npz"))
    raw = bytearray(ev.read_bytes())
    raw[len(raw) // 3] ^= 0x55
    ev.write_bytes(bytes(raw))
    with pytest.raises(P.LabError):
        store2.load(cfg, registry=reg)


# ------------------------------------------------------------------------------------------------------------------ coverage book and loaders
def test_coverage_book_orders_persists_and_refuses_repeats(tmp_path):
    b = E.CoverageBook([1990, 1991, 1992], n_slices=2, lenses=("c2c", "rng"))
    assert len(b.all_units()) == 12 and b.fraction_done() == 0.0
    first = b.pending(limit=1)[0]
    rec = E.UnitRecord(first.uid, "1990-12-31", True, 250, 100, 500, 0, {"c2c_up_5_10": 200}, ("a@0",), "cfg")
    b.mark(rec)
    with pytest.raises(E.EpisodeError):
        b.mark(rec)                                                                            # nothing new: refusing to record it again
    nxt = b.pending(("a@0",), limit=1)[0]
    assert nxt.year != first.year                                                              # least covered year first
    assert first not in b.pending(("a@0",)) and first in b.pending(("a@0", "b@0"))
    h = b.save(tmp_path / "cov.json")
    assert E.CoverageBook.load(tmp_path / "cov.json").digest() == b.digest()
    raw = json.loads((tmp_path / "cov.json").read_text())
    raw["body"]["records"][first.uid]["n_episodes"] = 1
    (tmp_path / "cov.json").write_text(json.dumps(raw))
    with pytest.raises(E.EpisodeError):
        E.CoverageBook.load(tmp_path / "cov.json")
    with pytest.raises(E.EpisodeError):
        b.mark(dataclasses.replace(rec, uid="2050|0|c2c", features_done=("z",)))
    rep = b.report(("a@0",), thin=300)
    assert rep["units_done"] == 1 and rep["years_untouched"] == [y for y in (1990, 1991, 1992) if y != first.year]
    assert rep["thin_cells"] == ([(first.year, "c2c_up_5_10", 200)] if first.lens == "c2c" else [])
    assert E.slice_of("AAPL", 4) == E.slice_of("AAPL", 4) and {E.slice_of(t, 3) for t in "ABCDEFGHIJKLMNOP"} == {0, 1, 2}
    with pytest.raises(E.EpisodeError):
        E.slice_tickers(["A"], 5, 4)


def test_cache_loader_reads_pre2000_and_modern_files_by_slice(tmp_path):
    rng = np.random.default_rng(0)
    old_idx = pd.bdate_range("1997-09-01", periods=60)
    new_idx = pd.bdate_range("1998-01-02", periods=60)
    tick = [f"T{j}" for j in range(12)]
    for name, idx in (("stocks_pre2000", old_idx), ("stocks", new_idx)):
        base = 10 + rng.random((len(idx), 12))
        for f, arr in (("open", base), ("high", base * 1.01), ("low", base * 0.99), ("close", base), ("volume", base * 1e5)):
            pd.DataFrame(arr, index=idx, columns=tick).to_parquet(tmp_path / f"{name}_{f}.parquet")
    assert E.cache_tickers(tmp_path) == sorted(tick)
    ld = P.cache_loader(3, cache_dir=tmp_path)
    got = ld(E.Unit(1998, 1, "c2c"), E.EpisodeConfig(), 0, 0)
    want = E.slice_tickers(tick, 1, 3)
    assert list(got["Close"].columns) == want and got["Close"].index.min() < pd.Timestamp("1998-01-02")      # warm-up sessions come along
    assert set(got) == set(E.FIELD_KEYS)
    both = E.load_bars("1997-01-01", "1998-12-31", cache_dir=tmp_path)
    assert both["Close"].index.is_monotonic_increasing and not both["Close"].index.has_duplicates
    assert E.load_bars("2010-01-01", "2010-12-31", cache_dir=tmp_path)["Close"].empty


def test_market_state_scan_finds_a_planted_date_level_dependence():
    bars, _ = E.synthetic_bars(60, 300, seed=7)
    g = E.build_grid(bars, E.EpisodeConfig())
    rng = np.random.default_rng(1)
    ctx = P.Ctx(g)
    state_series = rng.normal(0, 1, g.shape[0])
    ctx._m["market"] = {"ret": state_series, "breadth": rng.random(g.shape[0]), "disp": rng.random(g.shape[0])}
    y = 5 + 2.0 * np.r_[0.0, state_series[:-1]] + rng.normal(0, 1, g.shape[0])                 # tomorrow's count follows today's state
    counts = pd.DataFrame({"c2c_up_5_10": y, "n_names": 60}, index=g.dates)
    rows = {(r["count"], r["state"]): r for r in P.market_state_scan(counts, ctx)}
    assert rows[("c2c_up_5_10", "ret")]["r"] > 0.5 and rows[("c2c_up_5_10", "ret")]["p"] < 1e-6
    assert rows[("c2c_up_5_10", "breadth")]["p"] > 0.001
    with pytest.raises(P.LabError):
        P.market_state_scan(counts, ctx, lag=0)
    assert P.market_state_scan(counts.iloc[:10], ctx) == []


def test_ids_are_letters_only_and_never_read_as_dates():
    ids = {P.alpha_id("pc", {"i": i}) for i in range(3000)}
    assert len(ids) == 3000 and all(x.isalpha() for x in ids)
    assert not any(string_reasons(x) for x in ids)


def test_always_on_job_ticks_forever_and_goes_idle_when_done(tmp_path):
    bars, cfg, reg = _small_state()
    st = P.new_state([1998], cfg, registry=reg)
    job = P.AlwaysOn(st, CountingLoader(P.frame_loader(bars)), P.LabStore(tmp_path / "lab"), years_fn=lambda now: [1998, 1999])
    reports = list(job.run(lambda: NOW, max_ticks=6, stop_after_idle=2))
    assert reports[0].new_years == 1 and reports[0].done and not reports[0].idle           # a year appeared, work was done
    assert [r.idle for r in reports][-2:] == [True, True] and len(reports) <= 6
    assert sum(len(r.done) for r in reports) == 2 and st.units_done() == 2
    assert reports[-1].fraction_done == 1.0 and job.coverage()["units_done"] == 2
    st.registry.register(P.PrecursorSpec("late_ret", "return", lambda c: c.g.c2c))
    assert not job.tick(NOW).idle                                                          # a wider registry makes it busy again
    with pytest.raises(P.LabError):
        P.AlwaysOn(st, job.loader, units_per_tick=0)
    stopped = list(job.run(lambda: NOW, stop=lambda: True))
    assert stopped == []


def test_lab_config_and_rules_validate():
    assert P.LabConfig().validate() == []
    for bad in (P.LabConfig(cluster_months=5), P.LabConfig(lenses=("zzz",)), P.LabConfig(n_perm=0), P.LabConfig(pairs=(("STOPPED", "STOPPED"),)),
                P.LabConfig(pairs=(("STOPPED", "NOPE"),))):
        assert bad.validate()
        with pytest.raises(P.LabError):
            bad.require_valid()
    assert P.EvidenceRules().digest() != dataclasses.replace(P.EvidenceRules(), alpha_q=0.01).digest()
    assert P.cluster_of(pd.DatetimeIndex(["2001-02-10", "2001-05-01"]), 3).tolist() == [2001 * 4 + 0, 2001 * 4 + 1]
    assert P.cluster_year(2001 * 4 + 1, 3) == 2001


def test_event_and_sector_precursors_read_only_what_was_public():
    bars, _ = E.synthetic_bars(40, 160, seed=31)
    g = E.build_grid(bars, E.EpisodeConfig())
    day = g.dates[100]
    ev = pd.DataFrame({"kind": ["8-K", "8-K"], "ticker": [g.tickers[3], g.tickers[4]],
                       "accepted": [pd.Timestamp(day.date()).tz_localize("America/New_York") + pd.Timedelta(hours=10),
                                    pd.Timestamp(day.date()).tz_localize("America/New_York") + pd.Timedelta(hours=17)]})
    spec = P.event_days_spec("days_since_8k", ev, "8-K", pub_lag=1)
    panel = spec.fn(P.Ctx(g))
    assert panel[100, 3] == 60 and panel[101, 3] == 0 and panel[102, 3] == 1        # filed 10:00: session 100, read one session later
    assert panel[101, 4] == 60 and panel[102, 4] == 0                               # filed 17:00: public next session (101), read one later
    assert np.isnan(panel[0]).all()                                                 # nothing before the first session
    codes = (np.arange(len(g.tickers)) % 4).astype(int)
    ctx = P.Ctx(g, sector_codes=codes)
    rel = P.PrecursorRegistry(audit=False)
    rel.register(P.PrecursorSpec("sector_rel_ret_1", "sector", lambda c: P._sector_relative(c, c.g.c2c)), audit=False)
    panels, missing = P.ranked_panels(ctx, rel)
    assert not missing and "sector_rel_ret_1" in panels
    none_ctx = P.Ctx(g, sector_codes=None)
    assert P.ranked_panels(none_ctx, rel)[1] == ["sector_rel_ret_1"]                # no sector map: reported missing, never silently zero


def test_context_function_supplies_sector_codes_to_a_unit():
    bars, cfg, _ = _small_state()
    reg = P.PrecursorRegistry([s for s in P.default_registry(audit=False).specs.values() if s.family in ("sector", "volume")], audit=False)
    st = P.new_state([1998], cfg, registry=reg)
    P.step(st, NOW, P.frame_loader(bars), context_fn=lambda unit, grid: ((np.arange(len(grid.tickers)) % 5).astype(int), {}))
    assert not any("sector" in n for n in st.notes)
    st2 = P.new_state([1998], cfg, registry=reg)
    P.step(st2, NOW, P.frame_loader(bars))
    assert any("sector_rel_ret_1" in n for n in st2.notes)                            # without a map the gap is written down


def test_daily_counts_are_kept_per_slice_and_add_up_to_the_whole_universe(tmp_path):
    bars, cfg, reg = _small_state()
    cfg = dataclasses.replace(cfg, n_slices=3)
    st = P.new_state([1998, 1999], cfg, registry=reg)
    store = P.LabStore(tmp_path / "lab")
    P.sweep(st, NOW, P.frame_loader(bars), store=store)
    assert len(st.daily) == 6                                                                  # one table per slice-year (first lens only)
    whole = E.run_in_memory(E.year_bars(bars, 1998)).counts
    whole = whole[whole.index.year == 1998]
    merged = P.universe_counts(st, 1998)
    cols = [c for c in merged.columns if c != "n_illiquid"]
    assert (merged.index == whole.index).all()
    assert (merged[cols].to_numpy() == whole[cols].to_numpy()).all()                          # each name lives in exactly one slice
    back, _ = P.open_state(tmp_path / "lab", [1998, 1999], cfg, registry=reg)
    assert (P.universe_counts(back, 1999).to_numpy() == P.universe_counts(st, 1999).to_numpy()).all()
    table = P.hundreds_by_year(st, minimum=3)
    assert list(table["year"]) == [1998, 1999] and (table["universe_median"] > 60).all() and (table["sessions"] > 240).all()
