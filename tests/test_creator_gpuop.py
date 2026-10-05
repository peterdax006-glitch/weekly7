"""P1.7: the GPU operator in replay mode, on scrubbed synthetic excerpts shaped like the 3-4 Oct logs."""
from __future__ import annotations

import json

from creator import gpuop as G

IP = ".".join(["192", "0", "2", "9"])
HEALTH = """\
09:56:55 util=0 disk=8G guard=1 stop=0 runner=3 queued=2 lastfails=1 sync_age=7min problems: gpu-low(0%)
09:59:55 util=0 disk=8G guard=1 stop=0 runner=3 queued=2 lastfails=1 sync_age=10min problems: gpu-low(0%)
10:03:05 util=99 disk=8G guard=1 stop=0 runner=3 queued=2 lastfails=1 sync_age=13min problems: none
14:22:21 util=74 disk=2G guard=2 stop=0 runner=1 queued=0 lastfails=1 sync_age=20min problems: gpu-low(74%) disk-low(2G)
14:25:21 util=98 disk=15G guard=2 stop=0 runner=1 queued=0 lastfails=1 sync_age=23min problems: none
"""
RUNNER = """\
09:30:00 module 10_calib_x: start
09:33:00 module 10_calib_x: guardian released
10:05:00 module 10_calib_x: DONE rc=0 "job": "ext:ft_x", "rc": 0
"""
SYNC = """\
Welcome banner for host {ip}
Connection to {ip} closed by remote host.
09:40:00 sync out_roles got=1
"""
JOB = (
    '{"job": "ext:ft_calib_x", "rc": 0, "wall_s": 1800.0, "usd_spent_total": 5.5, "monitor": {"util_mean": 98.2}, '
    '"result": {"skipped": "VRAM: 14256 MiB free after 1800 s, needs 26624"}}\n'
    '{"job": "ext:ft_thin_x", "rc": 0, "wall_s": 100.0, "usd_spent_total": 5.6, "monitor": {"util_mean": 30.0}, '
    '"result": {"sft": {"steps": 28}}}\n'
    "job ext:roleeval_x_tuned skipped: m.gguf could not be served: pod-local model l.gguf is not on the pod - refusing to serve it\n"
)


def _root(tmp_path):
    (tmp_path / "gpu").mkdir()
    q = tmp_path / "gpuday" / "trainmix" / "queue" / "logs"
    q.mkdir(parents=True)
    (tmp_path / "gpu" / "health.log").write_text(HEALTH, encoding="utf-8")
    (tmp_path / "gpu" / "module_runner.out").write_text(RUNNER, encoding="utf-8")
    (tmp_path / "gpu" / "sync_loop.out").write_text(SYNC.replace("{ip}", IP), encoding="utf-8")
    (q / "10_calib_x.log").write_text(JOB, encoding="utf-8")
    return tmp_path


def test_reconstruct_telemetry_and_scrub(tmp_path):
    recs = G.reconstruct(_root(tmp_path))
    ts = [r.t for r in recs if r.t is not None]
    assert ts == sorted(ts)
    assert {"health", "runner", "sync", "job"} <= {r.source for r in recs}
    assert not any(IP in r.text for r in recs)
    syn = [r for r in recs if r.synthetic]
    assert len(syn) == 1 and syn[0].vram_mib == {"free": 14256, "need": 26624}
    assert syn[0].t == G._tod("09:30:00") + G.FT_WAIT_PROBE_S
    assert G.scrub("C:" + "\\Users\\bob\\x and /c/" + "Users/bob/y " + IP) == "<path> and <path> <ip>"


def test_operator_diagnoses_each_incident_and_picks_claudes_fix(tmp_path):
    rep = G.replay_report(_root(tmp_path))
    by = {r["kind"]: r for r in rep["incidents"]}
    for kind in ("feeder_stall", "disk_full", "vram_starvation", "data_starved", "missing_lora", "transfer_stall"):
        assert kind in by, kind
    assert by["feeder_stall"]["remedy"] == "restart_feeder" and by["feeder_stall"]["verdict"] == "matches"
    assert by["vram_starvation"]["remedy"] == "hold_guardian_stop" and by["vram_starvation"]["verdict"] == "matches"
    assert by["missing_lora"]["remedy"] == "place_lora"
    assert by["transfer_stall"]["remedy"] == "sync_chunks_direct"
    # a starved trainer is diagnosed 120 s after its start, long before the 1800 s skip line existed
    assert by["vram_starvation"]["ttd_s"] == G.FT_WAIT_PROBE_S
    assert by["vram_starvation"]["claude_ref_s"] > 1700
    # the feeder stall is gone by 10:03; the measured effect is taken from the following samples
    assert by["feeder_stall"]["measured"]["cleared_within_window"] is True
    assert by["feeder_stall"]["claude_ref_s"] == G._tod("10:03:05") - G._tod("09:56:55")


def test_ranking_orders_by_gain_minus_risk_and_levels_gate_risk():
    ranked = G.rank_remedies("feeder_stall", level="A2")
    assert [r.action for r in ranked] == ["restart_feeder", "raise_feeder_workers", "move_client_to_pod"]
    assert all(ranked[i].score >= ranked[i + 1].score for i in range(len(ranked) - 1))
    a1 = G.rank_remedies("feeder_stall", level="A1")
    assert all(r.risk == "low" for r in a1) and "move_client_to_pod" not in {r.action for r in a1}
    assert G.rank_remedies("feeder_stall", level="A0") == []                      # A0 proposes only
    assert "recreate_instance" not in {r.action for r in G.rank_remedies("disk_full", level="A3")}     # high risk is never automatic
    assert all(len(G.rank_remedies(k.kind)) >= 1 for k in G.KF.TABLE if k.kind in G.PLAYBOOK)
    assert set(G.PLAYBOOK) <= set(G.KF.KINDS)


def test_every_blueprint_kind_has_a_playbook_and_component():
    for kind in G.KF.KINDS:
        assert kind in G.PLAYBOOK and kind in G.KIND_COMPONENT, kind
    assert set(G.KIND_COMPONENT.values()) <= set(G.COMPONENTS)


def test_revert_when_the_effect_is_worse():
    op = G.Operator(G.ReplayBackend(), level="A2")
    h = lambda t, util, guard=1, stop=0: G.Telemetry(t, "health", util=util, disk_g=8, guard=guard, stop=stop, runner=3, queued=1,
                                                      sync_age_min=3, text="x util= problems: gpu-low")
    pre = [h(100 + i * 60, 90) for i in range(5)]
    bad = G.Telemetry(400, "health", util=0, disk_g=8, guard=1, stop=0, runner=3, queued=1, sync_age_min=3,
                      text="400 util=0 problems: gpu-low(0%)")
    post = [h(460 + i * 60, 5) for i in range(5)]
    op.run(pre + [bad] + post)
    inc = op.incidents[0]
    assert inc.kind == "feeder_stall" and inc.reverted is True
    assert ("revert", "restart_feeder", 400) in op.backend.log


def test_learner_pulls_ranking_and_appends_evidence(tmp_path):
    p = tmp_path / "ev.jsonl"
    L = G.Learner(p)
    for _ in range(5):
        L.record({"kind": "feeder_stall", "action": "restart_feeder", "ok": False})
        L.record({"kind": "feeder_stall", "action": "raise_feeder_workers", "ok": True})
    assert G.rank_remedies("feeder_stall", L)[0].action == "raise_feeder_workers"
    assert len(p.read_text().splitlines()) == 10
    assert G.Learner(p).blended("feeder_stall", "restart_feeder", 0.9) == 0.0


def test_lifecycle_deadman_sync_before_stop_budget_and_empty_wishlist():
    s = G.PodState(usd_spent=3.0, usd_per_hour=1.0, deadman_armed=False, sync_age_min=40, wishlist_len=2, jobs_running=1)
    assert [a.key for a in G.lifecycle_actions(s, budget_usd=50)] == ["arm_deadman"]
    s = G.PodState(usd_spent=3.0, usd_per_hour=1.0, deadman_armed=True, sync_age_min=40, wishlist_len=0, jobs_running=0)
    assert [a.key for a in G.lifecycle_actions(s, budget_usd=50)] == ["sync_now", "stop_instance"]
    s = G.PodState(usd_spent=3.0, usd_per_hour=1.0, deadman_armed=True, sync_age_min=2, wishlist_len=0, jobs_running=0)
    assert [a.key for a in G.lifecycle_actions(s, budget_usd=50)] == ["stop_instance"]
    s = G.PodState(usd_spent=49.8, usd_per_hour=2.0, deadman_armed=True, sync_age_min=2, wishlist_len=9, jobs_running=2)
    acts = G.lifecycle_actions(s, budget_usd=50)
    assert acts[-1].key == "stop_instance" and dict(acts[-1].params)["reason"] == "budget"
    assert G.within_budget(49.0, 0.5, 50) and not G.within_budget(49.9, 0.5, 50)


def test_budget_cap_refuses_action_at_any_level():
    op = G.Operator(G.ReplayBackend(), level="A3", budget_usd=10.0)
    op.spent = 10.5
    op.observe(G.Telemetry(1.0, "health", util=0, disk_g=8, guard=1, stop=0, runner=3, queued=1, sync_age_min=3,
                           text="util=0 problems: gpu-low(0%)"))
    inc = op.incidents[0]
    assert inc.chosen is None and inc.refused == "budget cap" and op.backend.log == []


def test_live_backend_is_dry_run_unless_armed_and_holds_no_addresses(tmp_path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"ssh_target": "u@h", "ssh_port": 1, "guardian_dir": "/g", "work_dir": "/w",
                               "local": {"restart_feeder": ["feeder", "--restart"]}}), encoding="utf-8")
    calls = []

    def fake(argv, **kw):
        calls.append(argv)
        return type("P", (), {"returncode": 0})()

    b = G.LiveBackend(cfg, armed=False, runner=fake)
    r = b.do(G.Action("hold_guardian_stop"))
    assert r.executed is False and r.command[-1] == "touch /g/STOP" and calls == []
    assert b.do(G.Action("restart_feeder")).command == ("feeder", "--restart") and calls == []
    assert b.revert(G.Action("hold_guardian_stop")).command[-1] == "rm -f /g/STOP"
    a = G.LiveBackend(cfg, armed=True, runner=fake)
    assert a.do(G.Action("hold_guardian_stop")).executed is True and len(calls) == 1
    assert G.LiveBackend(None).do(G.Action("restart_feeder")).command == ()          # no config: nothing to run
    src = open(G.__file__, encoding="utf-8").read()
    import re
    assert not re.search(r"\d{1,3}(?:\.\d{1,3}){3}", src)
