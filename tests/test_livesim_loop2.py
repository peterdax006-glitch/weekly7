"""Pure functions of scripts/livesim_loop2.py: objective choice, window selection, basis adoption, gates still wired."""
import importlib.util
import sys
import zlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from engine import objective as O, basis_search as B

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "scripts" / "livesim_loop2.py"


@pytest.fixture(scope="module")
def L():
    argv, sys.argv = sys.argv, ["livesim_loop2.py"]
    try:
        spec = importlib.util.spec_from_file_location("livesim_loop2_under_test", SRC)
        m = importlib.util.module_from_spec(spec)
        before = m_state_stamp()
        spec.loader.exec_module(m)
        assert m_state_stamp() == before, "importing the loop wrote its state file"
        return m
    finally:
        sys.argv = argv


def m_state_stamp():
    p = ROOT / "state" / "livesim" / "loop2.json"
    return (p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else None


def row(inb, over=0.0, w5=-0.05, dd=-0.1, pos=0.5):
    return {"in_band": inb, "over_band": over, "worst5": w5, "max_dd": dd, "pos_in_band": pos, "mean_week": 0.01, "sd_week": 0.04}


def test_import_does_not_start_the_loop(L):
    assert callable(L.main) and callable(L.train_basis) and callable(L.archive_dirs)


def test_tiered_is_the_library_objective(L):
    rows = [row(0.6), row(0.4, 0.1)]
    s = O.evaluate(rows)
    assert L.tiered(rows) == (s.scalar, s.t1, s.risk, s.t3)
    assert L.objective(rows, "volatility")[0] == s.scalar


def test_objective_is_lexicographic_through_the_loop_api(L):
    """Planted: a basis with worse tier 1 but a far better tail and direction must NOT win (old blended score allowed it)."""
    a = [row(0.20, 0.0, -0.01, -0.02, 1.0)]
    b = [row(0.30, 0.0, -0.30, -0.60, 0.0)]
    assert L.objective(b, "x")[0] > L.objective(a, "x")[0]


def test_objective_floor_and_empty(L):
    assert L.objective([row(0.9, dd=-0.995)], "x")[0] == -9.0
    assert L.objective([], "x") == (-9.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        L.objective([row(float("nan"))], "x")


def test_meta_space_is_the_bible_list_and_cfg_space_covers_the_start_cfg(L):
    assert L.META_SPACE is B.META_SPACE
    assert set(L.st["cfg"]) <= set(L.CFG_SPACE)
    rng = np.random.default_rng(0)
    for _ in range(30):
        c, m = B.sample_candidate(rng, L.st["cfg"], L.st["meta"], L.CFG_SPACE, L.META_SPACE)
        assert set(c) >= set(L.st["cfg"]) and not B.validate_meta(m)


def test_archive_dirs_skips_backups_empties_and_files(L, tmp_path):
    for name, snap in (("r01a", True), ("w02b", True), ("_w01c_original", True), ("r03c", False)):
        d = tmp_path / name
        d.mkdir()
        if snap:
            (d / "wsnap_2190-06-14.parquet").write_bytes(b"x")
    (tmp_path / "loop2.json").write_text("{}")
    assert [a.name for a in L.archive_dirs(tmp_path)] == ["r01a", "w02b"]
    assert L.archive_dirs(tmp_path / "missing") == []


def test_run_window_builds_tier_row_from_the_session(L, monkeypatch):
    class FakeSession:
        weeks = [0.07, -0.06, 0.02, 0.12, -0.30, 0.08]
        def result(self):
            return {"mean_week": 0.0, "max_dd": -0.42, "year_return": 0.1}
    monkeypatch.setattr(L.A, "replay", lambda *a, **k: FakeSession())
    w = {"snaps": {}, "closes": None, "bps": 5, "divs": {}, "opens": None, "ltm": None}
    r = L.run_window(w, {}, {})
    assert r["in_band"] == pytest.approx(0.5) and r["max_dd"] == -0.42          # daily drawdown kept, not the weekly one
    assert r["worst5"] == pytest.approx(np.quantile(FakeSession.weeks, 0.05)) and r["cat_rate"] == pytest.approx(1 / 6)
    assert O.evaluate([r]).key[0] > 0


def synth_windows(n):
    return [{"id": f"w{i}", "closes": pd.DataFrame({"a": [1.0, 2.0]}, index=pd.to_datetime([f"2190-01-{i + 1:02d}", f"2190-02-{i + 1:02d}"]))}
            for i in range(n)]


def planted_eval(w, cfg, meta):
    rng = np.random.default_rng(zlib.crc32(w["id"].encode()))
    return O.week_row(0.005 + 0.08 * cfg["k"] ** -0.5 * rng.standard_normal(52) * 0.45)


def test_train_basis_adopts_a_planted_improvement_and_tags_windows(L):
    start = {**L.st["cfg"], "k": 4}
    r = L.train_basis(synth_windows(20), start, L.st["meta"], seed=1, evaluate=planted_eval)
    assert r.adopted and r.cfg["k"] < 4 and r.n_windows == 20 and r.n_screen == L.SCREEN_N
    assert r.winner.confirm.t1 > r.incumbent.confirm.t1


def test_train_basis_keeps_incumbent_on_noise_and_is_seeded(L):
    def noise(w, c, m):
        return O.week_row(0.07 * np.random.default_rng(zlib.crc32((w["id"] + B.fingerprint(c, m)).encode())).standard_normal(52))
    a = L.train_basis(synth_windows(26), L.st["cfg"], L.st["meta"], seed=3, evaluate=noise)
    b = L.train_basis(synth_windows(26), L.st["cfg"], L.st["meta"], seed=3, evaluate=noise)
    assert (a.adopted, a.cfg) == (b.adopted, b.cfg)
    c = L.train_basis(synth_windows(2), L.st["cfg"], L.st["meta"], seed=3, evaluate=noise)
    assert not c.adopted and c.n_evals == 0                                      # too few windows: keep the basis


def test_train_basis_honours_as_of(L):
    seen = set()
    def ev(w, c, m):
        seen.add(w["id"])
        return planted_eval(w, c, m)
    ws = synth_windows(20)
    cut = ws[9]["closes"].index[-1]
    L.train_basis(ws, {**L.st["cfg"], "k": 4}, L.st["meta"], seed=1, evaluate=ev, as_of=cut)
    assert seen and max(int(i[1:]) for i in seen) <= 9


def test_safety_gates_are_still_wired_into_the_round():
    src = SRC.read_text(encoding="utf-8")
    for needle in ("provenance.stale(provenance.code_stamp())", "STALE CODE", "fill_audit.gate(S1", "future-scramble",
                   "gate_ok &= rep and scram and fill_ok", "ANTI-CHEAT GATE FAILED", "scramble_after=cut"):
        assert needle in src, f"loop lost a gate: {needle}"
    assert src.index("ANTI-CHEAT GATE FAILED") < src.index("res = train_basis(wins")    # gates precede any retraining
