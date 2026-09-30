"""F23 part 3 (C75 Phase 1E, C66 RF28 gap 13): released research knowledge changes a decision of the PRODUCTION decision-maker
(engine.adaptive.Session) - and only when it arrives through the firewall doors (engine.research.firewall.run_day or
Bridge.release -> MaturedRecord.gate), only as a trader_view.TraderRelease, and never when it is absent, unmatured or refused.
Synthetic data only (the tests/test_session.py world); no caches, no network."""
import numpy as np
import pandas as pd
import pytest

from engine import adaptive as A
from engine.learning import trader_view as TV
from engine.learning.core import FirewallBreach, Provenance
from engine.research import decision_bridge as DB
from engine.research import firewall as F
from engine.research import namespaces as N
from engine.research.core import MaturedRecord

CFG = {"k": 2, "exit_q": 0.8, "rebalance_weeks": 1, "brake": None, "max_per_sector": None, "w_model": 1.0,
       "pick": "top", "pool_q": 0.7, "liq_q": 0.0, "vol_filter": False, "stress_thr": None, "stress_k": 2,
       "trend_filter": None, "trend_gross": 0.0}
TICK = [f"T{i:02d}" for i in range(12)]
NOW = F.REF_NOW                                    # 2020-06-01: the firewall's reference clock


def world(n=45, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-03-02", periods=n)
    closes = pd.DataFrame(20 * np.exp(np.cumsum(rng.normal(0, 0.02, (n, len(TICK))), axis=0)), index=idx, columns=TICK)
    opens = closes.shift(1).fillna(closes.iloc[0]) * np.exp(rng.normal(0, 0.01, closes.shape))
    return closes, opens


def snaps(closes, seed=1):
    """Model score mu_raw is noise; vol20 is a fixed per-name ladder (T11 the most volatile), r5 is past momentum only."""
    rng = np.random.default_rng(seed)
    out = {}
    for d in closes.index:
        c = closes.loc[:d]
        r5 = (c.iloc[-1] / c.iloc[-6] - 1).to_numpy() if len(c) > 5 else np.zeros(len(TICK))
        out[str(d.date())] = pd.DataFrame(
            {"mu_raw": rng.normal(0, 1, len(TICK)), "evidence": 0.5, "vol20": np.linspace(0.01, 0.05, len(TICK)), "max20": 0.05,
             "log_dv": 18.0, "ev_red_flag": 0.0, "ev_offering": 0.0, "r5": r5, "m_vix": 0.5, "m_vix_term": 0.9, "m_spy_ma200": 1.05},
            index=pd.Index(TICK, name="ticker"))
    return out


def run(release=None, cfg=CFG):
    closes, opens = world()
    return A.replay(cfg, snaps(closes), closes, 0.0, {}, opens=opens, release=release)


def item(features, kind="pattern", lean=0.0, weight=1.0):
    return TV.TraderMemoryItem.make(TV.opaque_token({"f": features, "k": kind, "l": lean}), kind, weight, features, lean, 5)


def picks(S):
    return [tuple(t) for _, t in S.decisions]


# ------------------------------------------------------------------------------------------------ the Session hook
def test_no_knowledge_is_the_old_decision_path_exactly():
    base = run()
    empty = run(TV.TraderRelease(0, ()))
    assert picks(base) == picks(empty) and base.orders == empty.orders
    assert base.knowledge_log == [] and all(r["items"] == 0 for r in empty.knowledge_log)


def test_released_volatility_pattern_changes_the_decision():
    base, kn = run(), run(TV.TraderRelease(0, (item({"lv20": 1.0}),)))       # lv20 is the research name of log(vol20)
    assert picks(base) != picks(kn)
    top_vol = set(TICK[-4:])
    share = lambda S: np.mean([len(set(t) & top_vol) / max(1, len(t)) for t in picks(S)])      # noqa: E731
    assert share(kn) > share(base)                                        # it moved TOWARD what the knowledge says
    assert all(r["used"] == 1 and r["unusable"] == 0 for r in kn.knowledge_log)


def test_lesson_veto_removes_exactly_the_flagged_names():
    S = run(TV.TraderRelease(0, (item({"vol20": 1.0}, "lesson", -1.0),)))
    vetoed = {t for r in S.knowledge_log for t in r["vetoed"]}
    assert vetoed and vetoed <= set(TICK[-3:])                            # z(vol20) >= 1 is the top of the ladder only
    assert not any(set(t) & vetoed for t in picks(S))


def test_context_veto_cuts_exposure_only_when_the_market_breaches_it():
    hit = run(TV.TraderRelease(0, (item({"m_vix": 0.4}, "context", -1.0),)))     # m_vix 0.5 >= 0.4: fires
    miss = run(TV.TraderRelease(0, (item({"m_vix": 0.9}, "context", -1.0),)))    # 0.5 < 0.9: silent
    assert all(r["ctx_hit"] and r["scale"] == A.KNOWLEDGE_CTX_CUT for r in hit.knowledge_log)
    assert not any(r["ctx_hit"] for r in miss.knowledge_log) and picks(miss) == picks(run())
    assert hit.fills and max(abs(f["dv"]) for f in hit.fills) < max(abs(f["dv"]) for f in run().fills)


def test_unexpressible_knowledge_is_counted_never_guessed():
    S = run(TV.TraderRelease(0, (item({"sector_rel_vol": 1.0}),)))
    assert picks(S) == picks(run()) and all(r["unusable"] == 1 and r["used"] == 0 for r in S.knowledge_log)


def test_session_refuses_anything_that_did_not_pass_the_firewall():
    S = A.Session(CFG, {}, 0.0)
    class TraderRelease:                                                  # a look-alike built outside trader_view
        items = (item({"lv20": 1.0}),)
    for bad in ({"items": [{"kind": "pattern", "features": {"lv20": 1.0}}]}, [item({"lv20": 1.0})], "lv20", TraderRelease()):
        with pytest.raises(A.KnowledgeRefused):
            S.receive(bad)
    assert S.release is None
    with pytest.raises(FirewallBreach):                                   # a date cannot even be put into a release
        TV.TraderMemoryItem.make("abcdefg", "pattern", 1.0, {"lv20": 20200105.0}, 0.0, 5)


def test_knowledge_weight_is_capped():
    cfg = {**CFG, "knowledge_w": 5.0}
    p = snaps(world()[0])[str(world()[0].index[10].date())]
    s, _ = A.pick_score(p, cfg, None, pd.Series(1.0, index=p.index))
    s0, _ = A.pick_score(p, cfg)
    assert np.allclose(s, (1 - A.KNOWLEDGE_W_MAX) * s0 + A.KNOWLEDGE_W_MAX)


# ------------------------------------------------------------------------------------------------ research doors -> Session
def fw_with(*objs):
    st = N.ResearchStore("f23")
    for o in objs:
        st.put(o)
    return F.ResearchTraderFirewall(st, "f23-secret")


def test_research_firewall_release_reaches_the_session_and_changes_its_decision():
    fw = fw_with(F.clean_object("K1", features={"lv20": 1.0}, lean=0.0))
    sr = DB.session_release(fw, NOW)
    assert isinstance(sr.release, TV.TraderRelease) and sr.n_items == 1
    assert picks(run(sr.release)) != picks(run())


def test_unmatured_research_is_refused_at_the_door_and_changes_nothing():
    fw = fw_with(F.clean_object("K2", learned="2020-09-30", features={"lv20": 1.0}, lean=0.0))     # learned AFTER now
    with pytest.raises(FirewallBreach):                                   # strict firewall: the whole release is refused
        DB.session_release(fw, NOW)
    S = A.Session(CFG, {}, 0.0)
    with pytest.raises(FirewallBreach):
        DB.release_to_session(S, None, (), NOW, "2026-09-30T00:00:00+00:00",
                              records=[MaturedRecord("R9", "2020-09-30", {"features": {"lv20": 1.0}}, _prov("2020-09-30"))])
    assert S.release is None                                              # nothing installed: the old decision path stands


def _prov(learned="2019-11-29"):
    return Provenance(created_real="2026-09-30T00:00:00+00:00", learned_at=learned, code_hash="f23", outcomes_seen_through=learned)


def test_matured_records_convert_claims_and_refuse_future_ones():
    ok = MaturedRecord("R1", "2019-11-29", {"claims": [
        {"output": "VOLATILITY_RANKING", "target": "lv20", "direction": 1, "magnitude": 0.2, "conditions": []},
        {"output": "RISK_PENALTY", "target": "lv20", "direction": 1, "magnitude": 0.5, "conditions": ["volatility: vol20 ge 1.5"]},
        {"output": "CONFIDENCE", "target": "", "direction": 1, "magnitude": 0.1, "conditions": []}]}, _prov())
    sr = DB.records_to_release([ok], NOW)
    kinds = sorted(i.kind for i in sr.release.items)
    assert kinds == ["lesson", "pattern"] and len(sr.skipped) == 1 and "CONFIDENCE" in sr.skipped[0]
    assert abs(sum(i.weight for i in sr.release.items) - 1) < 1e-9
    late = MaturedRecord("R2", "2020-07-01", {"features": {"lv20": 1.0}}, _prov("2020-07-01"))
    with pytest.raises(FirewallBreach):
        DB.records_to_release([late], NOW)
    with pytest.raises(FirewallBreach):
        DB.records_to_release([{"features": {"lv20": 1.0}}], NOW)                   # not a gated record


def test_release_to_session_installs_it_and_the_empty_case_is_neutral():
    S = A.Session(CFG, {}, 0.0)
    sr = DB.release_to_session(S, None, (), NOW, "2026-09-30T00:00:00+00:00",
                               records=[MaturedRecord("R3", "2019-11-29", {"features": {"lv20": 1.0}, "trader_kind": "pattern"}, _prov())])
    assert S.release is sr.release and sr.n_items == 1
    S2 = A.Session(CFG, {}, 0.0)
    e = DB.release_to_session(S2, None, (), NOW, "2026-09-30T00:00:00+00:00")
    assert e.n_items == 0 and S2.release is not None and len(S2.release) == 0


def test_bridge_entry_releases_through_bridge_release_only():
    b = DB.Bridge()
    claim = DB.Claim(DB.Output.VOLATILITY_RANKING, "lv20", 1, 0.2,
                     measurement=DB.Measurement(DB.dc.BINDINGS[DB.E.RANKING].metric, 0.02, 0.01, 60, 2, True))
    d = DB.Discovery.make("higher trailing volatility ranks movers better", "f23", "2019-11-29", _prov(), (claim,))
    e = b.submit(d, "2019-12-02")
    assert e.disposition == DB.Disposition.DECISION_CHANGING
    closes, opens = world()
    S = A.Session(CFG, {}, 0.0)
    sr = DB.release_to_session(S, b, [e.entry_id], NOW, "2026-09-30T00:00:00+00:00")
    assert sr.n_items == 1
    assert picks(A.replay(CFG, snaps(closes), closes, 0.0, {}, opens=opens, release=S.release)) != picks(run())
    with pytest.raises(FirewallBreach):                                   # the same-year rerun leak is refused at Bridge.release
        DB.release_to_session(A.Session(CFG, {}, 0.0), b, [e.entry_id], NOW, "2026-09-30T00:00:00+00:00", replaying=(2019,))


def test_adaptive_imports_no_research_module():
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(A))
    mods = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    mods += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not [m for m in mods if "research" in m or "curator" in m or m.split(".")[-1].startswith(("pattern", "trust", "analog"))]
