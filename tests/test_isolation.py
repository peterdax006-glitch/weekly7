"""Live/research isolation (Phase 28, L5): static audit, runtime guards, activation drift."""
import builtins, json
from datetime import datetime, timezone

import pytest

from engine import isolation as I


def test_real_repo_has_no_unreviewed_coupling():
    rep = I.audit_imports()
    assert rep["modules"] > 15
    hard = [v for v in rep["violations"] if v["kind"] != "research_imports_live"]
    assert hard == [], hard                       # broker imports / sealed reads: no exceptions, ever
    # engine.parity_suite (another builder, 2026-09-28) lazily imports engine.live to run a live-path parity check.
    # It is pending owner review; anything ELSE new fails here.
    pending = {"parity_suite"}
    new = [v for v in rep["violations"] if v["module"] not in pending]
    assert new == [], new
    # known couplings really exist (the allow-list is not vacuous)
    assert rep["known_couplings"]
    assert {h["module"] for h in rep["known_couplings"]} <= set(I.KNOWN_CONSTANT_COUPLINGS)


def test_planted_research_broker_import_is_caught():
    for src in ("from .broker import get_broker\n", "from . import config, broker\n",
                "def f():\n    from engine.broker import LocalBroker\n"):
        v = I.audit_source_text("backtest", src)
        assert [x["kind"] for x in v] == ["research_imports_broker"], src


def test_planted_unreviewed_live_import_caught_and_known_allowed():
    src = "def f():\n    from .live import PRED_DIR\n"
    assert I.audit_source_text("newmodule", src)[0]["kind"] == "research_imports_live"
    assert I.audit_source_text("improve", src, known=I.KNOWN_CONSTANT_COUPLINGS) == []


def test_planted_live_sealed_read_caught(tmp_path):
    (tmp_path / "live.py").write_text("from . import livesim\n")
    (tmp_path / "broker.py").write_text("X = K.STATE / 'livesim' / 'sealed_a.json'\n")
    (tmp_path / "other.py").write_text("from . import config\n")
    kinds = sorted(v["kind"] for v in I.audit_imports(tmp_path)["violations"])
    assert kinds == ["live_reads_sealed", "live_reads_sealed"]


def test_audit_on_empty_dir(tmp_path):
    assert I.audit_imports(tmp_path) == {"violations": [], "known_couplings": [], "modules": 0}


def test_research_mode_blocks_and_restores():
    from engine import broker
    orig_get, orig_order = broker.get_broker, broker.LocalBroker.order
    with I.research_mode(broker):
        with pytest.raises(I.IsolationError):
            broker.get_broker()
        with pytest.raises(I.IsolationError):
            broker.LocalBroker.order(None, "AAPL", 1, 10, "x")
    assert broker.get_broker is orig_get and broker.LocalBroker.order is orig_order


def test_research_mode_restores_after_exception():
    from engine import broker
    orig = broker.get_broker
    with pytest.raises(ValueError):
        with I.research_mode(broker):
            raise ValueError("boom")
    assert broker.get_broker is orig


def test_live_mode_blocks_sealed_files_only(tmp_path):
    sd = tmp_path / "livesim"
    sd.mkdir()
    (sd / "sealed_x.json").write_text("{}")
    ok = tmp_path / "equity.json"
    ok.write_text("[]")
    real_open = builtins.open
    with I.live_mode(sd):
        assert json.loads(ok.read_text()) == []
        with pytest.raises(I.IsolationError):
            open(sd / "sealed_x.json")
        with pytest.raises(I.IsolationError):
            (sd / "sealed_x.json").read_text()
        with pytest.raises(I.IsolationError):
            open(tmp_path / "elsewhere" / ".." / "sealed_y.json")     # name pattern alone is enough
    assert builtins.open is real_open
    assert (sd / "sealed_x.json").read_text() == "{}"


def test_state_write_audit_real_repo_only_known_promotion_path():
    r = I.audit_state_writes()
    assert r["violations"] == [], r["violations"]
    assert [(k["module"], k["file"]) for k in r["known"]] == [("improve", "config_overrides.json")]
    assert I.audit_credentials() == []


def test_planted_research_write_to_live_state_caught(tmp_path):
    (tmp_path / "sneaky.py").write_text(
        "import json\nfrom . import config as K\n"
        "def go():\n    (K.STATE / 'positions_meta.json').write_text('{}')\n"
        "def peek():\n    return json.loads((K.STATE / 'equity.json').read_text())\n")
    (tmp_path / "live.py").write_text("X = 'equity.json'\nopen(X, 'w')\n")       # live may write its own state
    r = I.audit_state_writes(tmp_path)
    assert [(v["module"], v["file"]) for v in r["violations"]] == [("sneaky", "positions_meta.json")]
    assert [(v["module"], v["file"]) for v in r["reads"]] == [("sneaky", "equity.json")]
    (tmp_path / "sneaky.py").write_text("import os\nk = os.environ['ALPACA_KEY']\n")
    assert I.audit_credentials(tmp_path)[0]["module"] == "sneaky"


def test_research_mode_scrubs_and_restores_credentials(monkeypatch):
    import os
    from engine import broker
    monkeypatch.setenv("ALPACA_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET", "s")
    with I.research_mode(broker):
        assert "ALPACA_KEY" not in os.environ and "ALPACA_SECRET" not in os.environ
    assert os.environ["ALPACA_KEY"] == "k" and os.environ["ALPACA_SECRET"] == "s"


def test_paper_tree_audit_finds_planted_live_client(tmp_path):
    (tmp_path / "engine").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "engine" / "ok.py").write_text("c = TradingClient(a, b, paper=True)\n")
    (tmp_path / "scripts" / "bad.py").write_text("c = TradingClient(a, b, paper=False)\n")
    (tmp_path / "scripts" / "broken.py").write_text("TradingClient(\n")
    r = I.audit_paper_only_tree(tmp_path)
    assert r["engine/ok.py"]["ok"] and not r["scripts/bad.py"]["ok"] and "unparseable" in r["scripts/broken.py"]["problems"][0]


def test_data_currency_uses_sessions_not_calendar_days():
    from datetime import date
    mon = datetime(2026, 9, 28, 11, tzinfo=I.ET)
    assert I.data_is_current(date(2026, 9, 28), mon) and I.data_is_current(date(2026, 9, 25), mon)   # Fri = prev session
    assert not I.data_is_current(date(2026, 9, 24), mon) and not I.data_is_current(date(2026, 9, 29), mon)
    sat = datetime(2026, 9, 26, 11, tzinfo=I.ET)
    assert I.data_is_current(date(2026, 9, 25), sat) and not I.data_is_current(date(2026, 9, 23), sat)
    after_thanks = datetime(2026, 11, 27, 10, tzinfo=I.ET)              # day after Thanksgiving; Thursday closed
    assert I.previous_session(date(2026, 11, 27)) == date(2026, 11, 25)
    assert I.data_is_current(date(2026, 11, 25), after_thanks)


def test_preflight_composes_every_gate(tmp_path):
    from datetime import date
    now = datetime(2026, 9, 28, 11, tzinfo=I.ET)
    act = tmp_path / "act.json"
    good = dict(now=now, activation_path=act, state_dir=tmp_path, last_bar=date(2026, 9, 28),
                orders=[("AAPL", 1.0, 100.0)], equity=1000.0, held={})
    r = I.live_preflight(**good)
    assert not r["go"] and r["reasons"] == ["no activation record"]      # explicit activation is required
    I.write_activation(act, tmp_path, "dax", now)
    assert I.live_preflight(**good) == {"go": True, "reasons": [], "refused_orders": []}
    for key, val, frag in [("now", datetime(2026, 12, 25, 11, tzinfo=I.ET), "outside regular session"),
                           ("now", datetime(2026, 9, 28, 17, tzinfo=I.ET), "outside regular session"),
                           ("last_bar", date(2026, 9, 21), "stale data"),
                           ("orders", [("AAPL", -1.0, 100.0)], "broker safety")]:
        r = I.live_preflight(**{**good, key: val})
        assert not r["go"] and any(frag in x for x in r["reasons"]), (key, r)
    (tmp_path / "config_overrides.json").write_text("{}")               # config drift after activation
    assert not I.live_preflight(**good)["go"]


def test_activation_roundtrip_and_drift(tmp_path):
    st = tmp_path
    (st / "config_overrides.json").write_text('{"STOP_ATR": 2.0}')
    act = st / "activation.json"
    now = datetime(2026, 9, 28, 14, tzinfo=timezone.utc)
    assert I.verify_activation(act, st) == {"ok": False, "problems": ["no activation record"]}
    I.write_activation(act, st, "dax", now)
    assert I.verify_activation(act, st)["ok"]
    (st / "config_overrides.json").write_text('{"STOP_ATR": 3.0}')      # a "research run" rewrites live config
    r = I.verify_activation(act, st)
    assert not r["ok"] and "config_overrides.json" in r["problems"][0]


def test_activation_refuses_non_paper_and_anonymous(tmp_path):
    now = datetime(2026, 9, 28, tzinfo=timezone.utc)
    with pytest.raises(I.IsolationError):
        I.write_activation(tmp_path / "a.json", tmp_path, "dax", now, paper=False)
    with pytest.raises(ValueError):
        I.write_activation(tmp_path / "a.json", tmp_path, "  ", now)


def test_activation_tampered_record_fails(tmp_path):
    a = tmp_path / "a.json"
    I.write_activation(a, tmp_path, "dax", datetime(2026, 9, 28, tzinfo=timezone.utc))
    rec = json.loads(a.read_text())
    rec["paper"] = False
    a.write_text(json.dumps(rec))
    assert not I.verify_activation(a, tmp_path)["ok"]
    a.write_text("{not json")
    assert I.verify_activation(a, tmp_path)["problems"] == ["activation record unreadable"]
    a.write_text(json.dumps({"paper": True, "activated_by": "x", "config_hashes": {}}))
    assert not I.verify_activation(a, tmp_path)["ok"]         # empty hash map must not pass
