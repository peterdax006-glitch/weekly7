"""Tests for the research-side / trader-side firewall (C66 sections 29-31). Synthetic data only; no network, no caches."""
import dataclasses
import datetime as dt
import importlib
import json

import numpy as np
import pandas as pd
import pytest

from engine.learning import firewalls as FW
from engine.learning import trader_view as TV
from engine.learning.core import FirewallBreach
from engine.research import firewall as F
from engine.research import namespaces as N
from engine.research.core import Namespace

NOW = F.REF_NOW


def store_with(*objs, name="t"):
    st = N.ResearchStore(name)
    for o in objs:
        st.put(o)
    return st


# ------------------------------------------------------------------------------------------------ planted leaks
def test_planted_suite_catches_every_channel_and_releases_clean(tmp_path):
    s = F.run_planted_suite(import_root=tmp_path)
    assert not s.void, "clean control must be released"
    assert s.missed == [], s.missed
    assert s.passed
    for ch in F.REQUIRED_CHANNELS:
        assert s.results[ch].caught, ch
    cov = F.coverage(s)
    assert cov["unplanted"] == [] and cov["unproducible"] == [] and cov["import_planted"]
    assert "FUTURE_PRICE" in F.suite_markdown(s)
    assert len(s.to_frame()) == len(s.results)


def test_time_channels_caught_without_any_replay():
    # the time checks must not lean on the same-year rule: with no replay in play they still refuse
    time_channels = [c for c in F.REQUIRED_CHANNELS if c != F.LeakChannel.YEAR_IDENTITY] + [
        F.LeakChannel.FUTURE_VOLUME, F.LeakChannel.FUTURE_FILING, F.LeakChannel.FUTURE_EARNINGS, F.LeakChannel.FUTURE_METADATA,
        F.LeakChannel.FUTURE_CORPORATE_ACTION, F.LeakChannel.FUTURE_MARKET_STATE, F.LeakChannel.FUTURE_PATTERN_HEALTH]
    s = F.run_planted_suite(replay=None, channels=time_channels)
    assert s.missed == []
    for ch in time_channels:
        assert "SAME_YEAR_RERUN" not in s.results[ch].channels


@pytest.mark.parametrize("removed,must_miss", [
    ("maturity", {"FUTURE_LABEL", "FUTURE_EVENT", "FUTURE_CORPORATE_ACTION"}),
    ("projection", {"YEAR_IDENTITY", "MALFORMED"}), ("identity", {"TICKER_IDENTITY"}),
    ("fingerprint", {"NUMERIC_FINGERPRINT"}), ("structure", {"DATASET_STRUCTURE", "TRAINING_RUN_MARKER"}),
    ("cache_state", {"HIDDEN_CACHE", "IMPLICIT_SHARED_STATE"}), ("hindsight", {"HINDSIGHT_LABEL", "RESEARCH_ONLY_KNOWLEDGE"}),
    ("filing", {"FUTURE_FILING"})])
def test_ablation_every_check_is_load_bearing(removed, must_miss):
    checks = tuple(c for c in F.CHECKS if c[0] != removed)
    s = F.run_planted_suite(checks=checks)
    assert must_miss <= set(s.missed), (removed, s.missed)


def test_same_year_needs_both_same_year_and_existence_removed():
    one = F.run_planted_suite(checks=tuple(c for c in F.CHECKS if c[0] != "same_year"), channels=[F.LeakChannel.SAME_YEAR_RERUN])
    assert one.missed == []                                  # memory_firewall's eval-window check backs it up
    both = F.run_planted_suite(checks=tuple(c for c in F.CHECKS if c[0] not in ("same_year", "existence")),
                               channels=[F.LeakChannel.SAME_YEAR_RERUN])
    assert both.missed == ["SAME_YEAR_RERUN"]


def test_unfiled_object_never_admitted_even_without_filing_check():
    fw = F.ResearchTraderFirewall(store_with(), "s", checks=tuple(c for c in F.CHECKS if c[0] != "filing"))
    d = fw.inspect(F.clean_object("stray"), NOW)
    assert not d.admitted and F.LeakChannel.MISSING_PROVENANCE in d.channels


def test_crashing_check_fails_closed():
    def boom(ctx):
        raise RuntimeError("bug")
    fw = F.ResearchTraderFirewall(store_with(F.clean_object()), "s", checks=F.CHECKS + (("boom", boom),))
    d = fw.inspect("clean", NOW)
    assert not d.admitted and F.LeakChannel.MALFORMED in d.channels


# ------------------------------------------------------------------------------------------------ null and empty cases
def test_null_case_clean_research_released_and_filed_live():
    p, fw = F.pair("run", "secret")
    p.research.put_many([F.clean_object("a"), F.clean_object("b", features={"vol_z": -0.7, "r5": 0.01})])
    res = F.run_day(fw, NOW, replay=F.reference_replay(), live=p.live)
    assert len(res.release) == 2 and not res.refused
    assert abs(sum(i.weight for i in res.release.items) - 1) < 1e-9
    assert TV.release_year_hits(res.release) == {}
    TV.assert_trader_safe(res.release.to_dict())
    assert len(p.live) == 2 and p.check() == []
    gw = F.TraderGateway(p.live)
    mem = gw.memory(NOW, step=3)
    assert {i.item_id for i in mem.items} == {i.item_id for i in res.release.items}
    assert gw.access_frame()["served"].tolist() == [2]


def test_empty_cases():
    fw = F.ResearchTraderFirewall(store_with(), "s")
    res = fw.release([], NOW)
    assert len(res.release) == 0 and fw.audit.entries[-1].outcome == "EMPTY"
    assert F.census_summary(F.release_census(fw, NOW))["n"] == 0
    assert F.release_schedule(fw).empty
    assert N.provenance_completeness(N.ResearchStore())["n"] == 0
    assert not F.boundary_ok(pd.DataFrame(columns=["offset", "expected", "admitted"]))
    assert F.load_audit("does/not/exist.jsonl").entries == []
    with pytest.raises(FirewallBreach):
        fw.inspect("clean", None)
    assert not fw.inspect("nope", NOW).admitted
    with pytest.raises(ValueError):
        F.ResearchTraderFirewall(store_with(), "s", checks=())
    with pytest.raises(ValueError):
        F.ResearchTraderFirewall(store_with(), "")


def test_strict_refuses_whole_release_non_strict_filters():
    st = store_with(F.clean_object(), F.clean_object("same", learned="2020-02-14"))
    fw = F.ResearchTraderFirewall(st, "s")
    with pytest.raises(FirewallBreach):
        fw.release(["clean", "same"], NOW, F.reference_replay())
    assert fw.audit.entries[-1].outcome == "REFUSED"
    lax = F.ResearchTraderFirewall(st, "s", F.FirewallPolicy(strict=False))
    res = lax.release(["clean", "same"], NOW, F.reference_replay())
    assert len(res.release) == 1 and [d.object_id for d in res.refused] == ["same"]
    assert lax.audit.entries[-1].outcome == "PARTIAL"
    assert not F.decision_frame(res.decisions).empty


# ------------------------------------------------------------------------------------------------ time rules
def test_boundary_is_exact_and_monotone():
    sw = F.boundary_sweep(F.clean_object(), NOW)
    assert F.boundary_ok(sw)
    assert sw.loc[sw.offset == -1, "admitted"].item() and not sw.loc[sw.offset == 0, "admitted"].item()
    broken = sw.copy()
    broken.loc[broken.offset == 2, "admitted"] = True          # a planted re-admission after refusal
    assert not F.boundary_ok(broken)


def test_moment_refuses_ambiguous_and_moves_after_close():
    for bad in (None, "", "2019", "2019-11", 1577836800, 2019.0, "11/03/2019"):
        with pytest.raises(N.AmbiguousTimestamp):
            N.moment(bad)
    assert N.moment("2020-01-02") == dt.date(2020, 1, 2)
    assert N.moment(dt.datetime(2020, 1, 2, 17, 30)) == dt.date(2020, 1, 3)
    assert N.moment(dt.datetime(2020, 1, 2, 15, 59)) == dt.date(2020, 1, 2)
    assert N.moment(pd.Timestamp("2020-01-02 22:00", tz="UTC")) == dt.date(2020, 1, 3)


def test_payload_dates_find_hidden_dates():
    p = {"a": {"b": ["x", "held through 2021-03-04"]}, "f": pd.DataFrame({"v": [1, 2]}, index=pd.to_datetime(["2019-01-02", "2019-05-06"]))}
    ds = N.payload_dates(p)
    assert max(ds) == dt.date(2021, 3, 4) and dt.date(2019, 5, 6) in ds
    assert N.payload_dates({}) == []


def test_frame_timestamp_findings():
    good = pd.DataFrame({"v": [1.0, 2.0]}, index=pd.to_datetime(["2019-01-02", "2019-01-03"]))
    assert N.frame_timestamp_findings(good) == [] and N.frame_timestamp_findings(good.iloc[:0]) == []
    dup = pd.DataFrame({"v": [1.0, 2.0]}, index=pd.to_datetime(["2019-01-03", "2019-01-03"]))
    assert any("duplicated" in e for e in N.frame_timestamp_findings(dup))
    late = pd.DataFrame({"v": [1.0]}, index=pd.to_datetime(["2019-01-03 17:30"]))
    assert any("after the" in e for e in N.frame_timestamp_findings(late))
    assert N.frame_timestamp_findings(pd.DataFrame({"v": [1.0]}, index=["x"]))
    obj = F._kind("dupx", N.InfoKind.PRICE, "2019-01-03", "2019-01-03", bars=dup)
    assert any("timestamp" in e for e in obj.validate())


def test_knowable_rules_per_kind():
    lab = F._kind("l", N.InfoKind.LABEL, "2020-01-10", "2020-01-10")
    assert lab.knowable_at() == dt.date(2020, 1, 11)
    fil = F._kind("f", N.InfoKind.FILING, "2020-01-10", "2020-01-09")
    assert fil.knowable_at() == dt.date(2020, 1, 11)                     # SEC filings: next session at the earliest
    eps = F._kind("e", N.InfoKind.EARNINGS, "2019-12-31", "2019-11-01")
    assert eps.knowable_at() >= dt.date(2020, 1, 20)


# ------------------------------------------------------------------------------------------------ stores
def test_store_is_append_only_frozen_and_chained():
    raw = {"features": {"vol_z": 1.0}, "lean": 0.1, "horizon": 5, "lst": [1, 2]}
    o = dataclasses.replace(F.clean_object("x"), payload=raw)
    st = store_with(o)
    raw["lst"].append(3)                                           # the caller's container is not the store's
    assert st.get("x").payload["lst"] == (1, 2)
    with pytest.raises(TypeError):
        st.get("x").payload["lean"] = 9
    assert st.put(dataclasses.replace(o, payload={**raw, "lst": [1, 2]})) and len(st) == 1   # same content: idempotent
    with pytest.raises(FirewallBreach):
        st.put(o)                                                  # the caller mutated its list: different content now
    with pytest.raises(FirewallBreach):
        st.put(dataclasses.replace(o, payload={**raw, "lean": 0.9}))
    assert st.verify() == []
    st._objs["x"] = dataclasses.replace(st.get("x"), payload=N.deep_freeze({**raw, "lean": 0.5}))   # back-door edit
    assert any("changed" in e for e in st.verify())


def test_store_refuses_wrong_namespace_orphans_and_live_objects():
    st = N.ResearchStore()
    with pytest.raises(FirewallBreach):
        st.put(F.clean_object().with_namespace(Namespace.LIVE_POINT_IN_TIME, tags={}))
    with pytest.raises(FirewallBreach):
        st.put(dataclasses.replace(F.clean_object("orphan"), parents=("ghost",)))
    with pytest.raises(N.SharedStateError):
        N.deep_freeze({"s": N.ResearchStore()})
    with pytest.raises(N.SharedStateError):
        N.deep_freeze({"f": lambda: 1})


def test_lineage_taint_and_matured_record():
    st = store_with(F._kind("p", N.InfoKind.EXPERIMENT_RESULT, "2020-08-14", "2020-08-14"),
                    F._kind("c", N.InfoKind.RESEARCH_RESULT, "2019-10-01", "2019-10-01", parents=("p",)))
    k, who = st.effective_knowable("c")
    assert who == "p" and k == dt.date(2020, 8, 15)
    assert st.matured_record("c").matured_at == "2020-08-14"
    assert st.filed_years("c") == (2019, 2020)
    assert set(st.by_year()) == {2019, 2020}
    anc, missing, cycle = st.lineage("c")
    assert anc == ["p"] and not missing and not cycle


def test_live_store_clock_and_future_refusal():
    live = N.LiveStore("l")
    px = N.InfoObject("px", N.InfoKind.PRICE, Namespace.LIVE_POINT_IN_TIME, "2020-06-01", {"close_z": 0.3},
                      F._prov("2019-01-01"))
    with pytest.raises(FirewallBreach):
        live.put(px)                                              # no clock
    live.advance("2020-05-29")
    with pytest.raises(FirewallBreach):
        live.put(px)                                              # future-derived state cannot even be stored
    live.advance("2020-06-01")
    live.put(px)
    with pytest.raises(FirewallBreach):
        live.advance("2020-05-01")
    with pytest.raises(FirewallBreach):
        live.view("2020-06-02")
    assert [o.object_id for o in live.view()] == ["px"]
    assert [o.object_id for o in live.view("2020-05-31")] == []
    yr = N.InfoObject("yr", N.InfoKind.YEAR_IDENTITY, Namespace.LIVE_POINT_IN_TIME, "2019-01-01", {}, F._prov("2019-01-01"))
    with pytest.raises(FirewallBreach):
        live.put(yr)
    gw = F.TraderGateway(live)
    obs = gw.observe(NOW)
    assert obs == [{"kind": "price", "values": {"close_z": 0.3}}]
    with pytest.raises(FirewallBreach):
        gw.observe(NOW, kinds=[N.InfoKind.LABEL])


def test_tickets_single_use_bound_and_signed():
    p, fw = F.pair("run", "secret")
    p.research.put(F.clean_object())
    res = fw.release(["clean"], NOW)
    p.live.advance(NOW)
    obj, t = res.live_objects[0], res.tickets[0]
    with pytest.raises(FirewallBreach):
        p.live.put(obj)                                           # research origin without a ticket
    forged = dataclasses.replace(t, signature="0" * 32)
    with pytest.raises(FirewallBreach):
        p.live.accept(obj, forged)
    other = N.sign_ticket("wrong-secret", obj.object_id, obj.digest(), NOW, res.release_id)
    with pytest.raises(FirewallBreach):
        p.live.accept(obj, other)
    with pytest.raises(FirewallBreach):
        p.live.accept(dataclasses.replace(obj, payload={**obj.payload, "lean": -1.0}), t)
    p.live.accept(obj, t)
    with pytest.raises(FirewallBreach):
        p.live.accept(obj, t)                                     # replayed
    early = N.LiveStore("e", fw.verifier())
    early.advance("2020-01-01")
    with pytest.raises(FirewallBreach):
        early.accept(obj, t)                                      # decided after this clock


def test_shared_state_audits():
    a, b = N.ResearchStore("a"), N.ResearchStore("b")
    a.put(F.clean_object())
    b.put(F.clean_object())
    assert N.shared_references(a, b) == []
    shared = [1, 2]
    a._objs["clean"] = dataclasses.replace(a.get("clean"), payload={"x": shared})
    b._objs["clean"] = dataclasses.replace(b.get("clean"), payload={"y": shared})
    assert N.shared_references(a, b)
    assert N.shared_references(a, a)

    class Holder:
        def __init__(self, s):
            self.cache = {"k": s}
    assert N.reachable_stores(Holder(a))
    assert N.reachable_stores(lambda: a)
    assert N.reachable_stores({"plain": [1, 2]}) == []


def test_module_state_scan():
    src = "import types\nA = {}\nB = frozenset()\nC: list = list()\nD = types.MappingProxyType({})\ndef f():\n    global B\n"
    found = N.module_state_violations(src)
    assert {r.detail for r in found} == {"module-level mutable A", "module-level mutable C", "function rebinds globals B"}
    assert N.package_state_violations([N.__file__, F.__file__]) == []


def test_provenance_census_flags_implausible():
    bad = dataclasses.replace(F.clean_object("late"), provenance=dataclasses.replace(F._prov("2019-11-29"), created_real="2019-01-01"))
    st = store_with(F.clean_object(), bad)
    rep = N.provenance_completeness(st)
    assert rep["n"] == 2 and rep["implausible"] == ["late"] and rep["shares"]["code_hash"] == 1.0
    assert len(N.store_frame(st)) == 2


# ------------------------------------------------------------------------------------------------ import guards
def test_real_trader_path_never_reaches_research():
    assert F.trader_research_violations() == []
    assert F.assert_trader_research_clean() > 0
    assert "engine.research.firewall" in F.research_modules()


def test_planted_import_and_symbol_caught(tmp_path):
    v = F.plant_import_leak(tmp_path)
    assert any(x.target == "engine.research.namespaces" for x in v)
    (tmp_path / "engine" / "livesim.py").write_text("def f(s):\n    return s.MaturedRecord\n", encoding="utf-8")
    assert any(x.kind == "symbol" for x in F.trader_research_violations(("engine/livesim.py",), tmp_path))
    (tmp_path / "engine" / "livesim.py").write_text("X = 1\n", encoding="utf-8")
    assert F.trader_research_violations(("engine/livesim.py",), tmp_path) == []
    with pytest.raises(FirewallBreach):
        F.assert_trader_research_clean(("engine/missing.py",), tmp_path)


def test_runtime_import_blocker_blocks_and_restores():
    with F.ResearchImportBlocker() as b:
        with pytest.raises(F.ResearchImportBlocked):
            importlib.import_module("engine.research.namespaces")
        with pytest.raises(ImportError):
            importlib.import_module("engine.research.core")
        importlib.import_module("engine.learning.core")          # the rest of the engine is untouched
    assert len(b.attempts) == 2 and all(a.startswith("engine.research") for a in b.attempts)
    assert importlib.import_module("engine.research.namespaces") is N


# ------------------------------------------------------------------------------------------------ training gate
def test_training_gate_dates_hindsight_labels():
    df = pd.DataFrame({"date": ["2019-03-01", "2019-12-02", "2020-02-03", "2019-05-01"],
                       "matured_at": ["2019-03-08", "2020-06-02", "2020-02-10", "2019-05-08"],
                       "knowability": ["UNKNOWN", "PREDICTABLE", "EXTERNALLY_CAUSED", "UNKNOWN"], "x": [1.0, 2.0, 3.0, 4.0]})
    g = F.TrainingGate()
    kept, rep = g.admit(df, NOW, F.reference_replay())
    assert kept["x"].tolist() == [1.0, 4.0]
    assert (rep.n_in, rep.n_admitted, rep.n_unmatured, rep.n_same_year) == (4, 2, 1, 1)
    blind = g.blind_rows(kept, ["x", "knowability"])
    assert list(blind.columns) == ["x", "knowability"]
    with pytest.raises(FirewallBreach):
        g.blind_rows(kept, ["x", "date"])
    with pytest.raises(FirewallBreach):
        g.admit(df.assign(matured_at=["2019-02-01", *df.matured_at[1:]]), NOW)       # label before its row
    with pytest.raises(FirewallBreach):
        g.admit(df.assign(matured_at=["garbage", *df.matured_at[1:]]), NOW)
    with pytest.raises(FirewallBreach):
        g.admit(df.drop(columns=["matured_at"]), NOW)
    empty, rep0 = g.admit(df.iloc[:0], NOW)
    assert empty.empty and rep0.n_in == 0


# ------------------------------------------------------------------------------------------------ section 30 disguise audits
def _producer(calendar_leak: bool):
    def make(offset: int):
        d = (dt.date(2019, 11, 29) + dt.timedelta(days=offset)).isoformat()
        feats = {"vol_z": 1.25, "r20": -0.03}
        if calendar_leak:
            feats["season"] = (dt.date.fromisoformat(d).year % 10) / 10.0      # encodes the calendar without naming it
        return [F.clean_object("a", learned=d, features=feats)]
    return make


def test_year_blindness_clean_producer_passes_leaky_one_caught():
    ok = F.year_blindness_probe(_producer(False), NOW, F.reference_replay(), "s")
    assert ok.passed and len(set(ok.digests.values())) == 1
    bad = F.year_blindness_probe(_producer(True), NOW, F.reference_replay(), "s")
    assert not bad.passed


def test_run_invariance_and_planted_run_marker():
    fw = F.ResearchTraderFirewall(store_with(F.clean_object()), "s")
    assert F.run_invariance(fw, ["clean"], NOW, F.reference_replay())["invariant"]

    class Leaky(F.ResearchTraderFirewall):
        def release(self, ids, now, replay=None, step=None, weights=None):
            w = {i: 1.0 + (replay.run_index if replay else 0) for i in ids}
            res = super().release(list(ids) + ["extra"], now, replay, step, weights={**w, "extra": 1.0})
            return res
    st = store_with(F.clean_object(), F.clean_object("extra", features={"vol_z": -1.0}))
    assert not F.run_invariance(Leaky(st, "s"), ["clean"], NOW, F.reference_replay())["invariant"]


def test_release_text_findings_and_overlap():
    fw = F.ResearchTraderFirewall(store_with(F.clean_object()), "s")
    rel = fw.release(["clean"], NOW).release
    assert F.release_text_findings([rel], ["MSFT"], [2019]) == []
    ov = F.cross_run_overlap([rel], [rel])
    assert ov["jaccard"] == 1.0
    assert np.isnan(F.cross_run_overlap([], [])["jaccard"])


def test_shift_object_moves_every_date():
    o = F._kind("s", N.InfoKind.RESEARCH_RESULT, "2019-10-01", "2019-10-01", note="seen 2019-09-30")
    s = F.shift_object(o, 364)
    assert s.evidence_through() == o.evidence_through() + dt.timedelta(days=364)
    assert "2020-09-28" in s.payload["note"]


# ------------------------------------------------------------------------------------------------ gate layer, audit, reports
def test_research_layer_in_learning_gate():
    fw = F.ResearchTraderFirewall(store_with(F.clean_object(), F.clean_object("same", learned="2020-02-14")), "s")
    gate = F.research_gate()
    base = dict(now=NOW, relevant=frozenset({F.ResearchLayerName.RESEARCH}), firewall=fw, replay=F.reference_replay())
    ok = gate.evaluate(F.ResearchGateContext(research_ids=["clean"], **base))
    assert ok.passed
    bad = gate.evaluate(F.ResearchGateContext(research_ids=["clean", "same"], **base))
    assert not bad.passed and F.ResearchLayerName.RESEARCH in bad.failed_layers
    assert all(f.validate() == [] for f in bad.findings(FW.Severity.FAIL))
    missing = gate.evaluate(F.ResearchGateContext(**{**base, "firewall": None}))
    assert not missing.passed
    with pytest.raises(FirewallBreach):
        bad.require()


def test_audit_chain_save_load_and_tamper(tmp_path):
    fw = F.ResearchTraderFirewall(store_with(F.clean_object(), F.clean_object("same", learned="2020-02-14")), "s")
    fw.release(["clean"], NOW)
    with pytest.raises(FirewallBreach):
        fw.release(["same"], NOW, F.reference_replay())
    assert fw.audit.verify() == [] and len(fw.audit.to_frame()) == 2
    assert fw.audit.channel_counts()["SAME_YEAR_RERUN"] >= 1
    path = tmp_path / "audit.jsonl"
    assert F.save_audit(fw.audit, path) == 2
    assert len(F.load_audit(path).entries) == 2
    rows = path.read_text(encoding="utf-8").splitlines()
    d = json.loads(rows[0])
    d["outcome"] = "REFUSED"
    path.write_text("\n".join([json.dumps(d)] + rows[1:]) + "\n", encoding="utf-8")
    with pytest.raises(FirewallBreach):
        F.load_audit(path)


def test_census_and_schedule():
    st = store_with(F.clean_object(), F.clean_object("same", learned="2020-02-14"),
                    F._kind("fut", N.InfoKind.LABEL, "2020-07-01", "2019-06-03"))
    fw = F.ResearchTraderFirewall(st, "s")
    cen = F.release_census(fw, NOW, F.reference_replay())
    summ = F.census_summary(cen)
    assert summ["n"] == 3 and abs(summ["admitted_share"] - 1 / 3) < 1e-9
    assert summ["by_channel"]["SAME_YEAR_RERUN"] >= 1 and summ["by_kind"]["LABEL"] == 0.0
    sch = F.release_schedule(fw, F.reference_replay())
    assert sch["object_id"].tolist()[0] == "clean"
    assert sch.set_index("object_id").loc["same", "first_release"] is None
    assert F.first_release_date(fw, "clean") == dt.date(2019, 11, 30)


def test_projection_carries_only_trader_fields():
    o = F.clean_object()
    item = F.project(o)
    assert set(item.to_dict()) == {"item_id", "kind", "weight", "features", "lean", "horizon"}
    assert "hit_rate" not in json.dumps(item.to_dict()) and "AAPL" not in json.dumps(item.to_dict())
    assert F.project(F.shift_object(o, 7 * 52 * 3)).item_id == item.item_id      # same content, other calendar, same token


def test_policy_and_replay_validation():
    assert F.FirewallPolicy(fingerprint_decimals=0).validate()
    assert F.ReplayContext("", "2020-02-01", "2020-01-01").validate()
    fw = F.ResearchTraderFirewall(store_with(F.clean_object()), "s")
    d = fw.inspect("clean", NOW, F.ReplayContext("w", "2020-13-01", "2020-12-31"))
    assert not d.admitted and F.LeakChannel.TIMESTAMP_AMBIGUITY in d.channels
