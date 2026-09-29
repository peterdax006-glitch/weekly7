"""A6 minute collector (scripts/collect_intraday.py): only completed sessions are written, --recent limits the days,
re-running adds nothing when the day is unchanged, new bars merge without duplicates, and the manifest lists exactly
the files written. The network download is replaced by a fake."""
import importlib.util
import sys

import pandas as pd
import pytest

from engine import config as K

spec = importlib.util.spec_from_file_location("collect_intraday", K.ROOT / "scripts" / "collect_intraday.py")
CI = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CI)


def bars(days, tickers=("AAA", "BBB"), per_day=3):
    ts = [pd.Timestamp(f"{d} 09:30") + pd.Timedelta(minutes=m) for d in days for m in range(per_day)]
    idx = pd.MultiIndex.from_product([ts, list(tickers)], names=["ts", "ticker"])
    return pd.DataFrame({"Close": range(len(idx)), "Volume": 100}, index=idx, dtype=float)


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(CI, "OUT", tmp_path / "intraday")
    monkeypatch.setattr(CI, "pick_tickers", lambda n: ["AAA", "BBB"])
    state = {"D": bars(["2026-09-24", "2026-09-25", "2026-09-28"]), "now": pd.Timestamp("2026-09-28 17:00", tz="America/New_York")}
    monkeypatch.setattr(CI, "fetch", lambda t, iv, p: state["D"])
    real_now = pd.Timestamp.now
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda cls, tz=None: state["now"]))
    return tmp_path, state


def run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["collect_intraday.py", *args])
    CI.main()


def test_completed_days_excludes_an_open_session():
    D = bars(["2026-09-25", "2026-09-28"])
    assert CI.completed_days(D, pd.Timestamp("2026-09-28 14:00")) == [pd.Timestamp("2026-09-25")]
    assert len(CI.completed_days(D, pd.Timestamp("2026-09-28 16:20"))) == 2


def test_writes_completed_days_and_manifest(world, monkeypatch):
    tmp, state = world
    man = tmp / "written.txt"
    run(monkeypatch, "--manifest", str(man))
    files = sorted(p.name for p in (tmp / "intraday" / "1m").glob("*.parquet"))
    assert files == ["2026-09-24.parquet", "2026-09-25.parquet", "2026-09-28.parquet"]
    assert len(man.read_text().split()) == 6                     # 3 days x (1m, 5m)


def test_recent_limits_days(world, monkeypatch):
    tmp, _ = world
    run(monkeypatch, "--recent", "1")
    assert [p.name for p in (tmp / "intraday" / "1m").glob("*.parquet")] == ["2026-09-28.parquet"]


def test_rerun_with_nothing_new_writes_nothing(world, monkeypatch):
    tmp, _ = world
    run(monkeypatch)
    man = tmp / "again.txt"
    run(monkeypatch, "--manifest", str(man))
    assert man.read_text() == ""


def test_new_bars_merge_without_duplicates(world, monkeypatch):
    tmp, state = world
    run(monkeypatch)
    state["D"] = bars(["2026-09-28"], per_day=5)               # the same day with two more minutes
    run(monkeypatch)
    g = pd.read_parquet(tmp / "intraday" / "1m" / "2026-09-28.parquet")
    assert len(g) == 5 * 2 and not g.index.duplicated().any()


def test_intraday_run_does_not_save_today(world, monkeypatch):
    tmp, state = world
    state["now"] = pd.Timestamp("2026-09-28 11:00", tz="America/New_York")
    run(monkeypatch)
    assert not (tmp / "intraday" / "1m" / "2026-09-28.parquet").exists()
