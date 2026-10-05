"""P0.6: the known-fix table and replay (synthetic, scrubbed log excerpts) and the policy diff checker."""
from __future__ import annotations

from pathlib import Path

from creator import knownfix as K
from creator.tools import policy as PO

MODULE_RUNNER = """\
09:35:34 module 10_calib_x: start
09:38:21 module 15_role_x: guardian released
mv: cannot stat '<home>/queue/15_role_x.json': No such file or directory
09:41:30 module 15_role_x: FAILED rc=0 "job": "ext:ft_role_x", "rc": 5
"""
HEALTH = """\
09:56:55 util=0 disk=8G guard=1 stop=0 runner=3 queued=2 lastfails=1 sync_age=7min problems: gpu-low(0%)
10:31:34 util=? disk=?G guard=? stop=? runner=0 queued=0 lastfails=1 sync_age=41min problems: pod-unreachable sync-stale(41min)
10:33:34 util=? disk=?G guard=? stop=? runner=0 queued=0 lastfails=1 sync_age=43min problems: pod-unreachable sync-stale(43min)
14:22:21 util=74 disk=1G guard=2 stop=0 runner=1 queued=0 lastfails=1 sync_age=273min problems: disk-low(1G)
15:32:10 util=98 disk=15G guard=2 stop=0 runner=5 queued=3 lastfails=1 sync_age=17min problems: none
"""
SYNC = """\
Connection to <host> closed by remote host.
18:39:53 sync out_roles got=1
18:50:15 sync out/model.gguf.jsonl 199412458/199245455
18:51:00 sync out/model.gguf.jsonl 45000000/207676329
"""
TASK = """\
{"job": "ext:ft_x", "rc": 0, "result": {
   "skipped": "VRAM: 14256 MiB free after 1800 s, needs 26624"
}}
 "skipped": "model not served: pod-local model m-Q4_K_M.gguf: pod-local model m-lora.gguf is not on the pod - refusing to serve it",
    "steps": 28,
    "adapter": "/dev/shm/x/ft_x/adapter",
    "steps": 5,
    "adapter": "/dev/shm/x/smoke_x/adapter",
job ext:roleeval_x skipped: m.gguf could not be served: remote step failed (rc 5): boom
"""
AUTOGEN = """\
09:37:16 autogen: pass failed: AttributeError: module has no attribute 'x'
17:38:47 autogen: aider_17b not yet (aider_sft has 45 rows, needs 300)
09:55:51 autogen: queue/hold has files - not queuing (failure storm guard)
"""


def _write(tmp: Path) -> list[Path]:
    out = []
    for name, text in (("module_runner.out", MODULE_RUNNER), ("health.log", HEALTH), ("sync_loop.out", SYNC), ("task.log", TASK),
                       ("autogen.log", AUTOGEN)):
        p = tmp / name
        p.write_text(text, encoding="utf-8")
        out.append(p)
    return out


def test_replay_recognises_each_incident_kind(tmp_path: Path) -> None:
    rep = K.replay(_write(tmp_path))
    c = rep["counts"]
    for kind in ("smoke_release_race", "module_failed", "feeder_stall", "pod_unreachable", "sync_stale", "disk_full", "transfer_stall",
                 "vram_starvation", "missing_lora", "serve_failed", "data_starved", "autogen_crash", "failure_storm"):
        assert c.get(kind, 0) >= 1, (kind, c)
    assert c["pod_unreachable"] == 1                       # two consecutive alerts are one episode
    assert c["transfer_stall"] == 1                        # the drop and the partial 45 MB copy form one episode; the growing file is not a stall
    assert c["data_starved"] == 2                          # ft with 28 steps and the thin autogen mix; the 5-step smoke is not one
    assert set(rep["missed"]) >= {"deadman_unarmed", "pkill_self_match"}
    inc = [i for i in rep["incidents"] if i.kind == "feeder_stall"][0]
    assert inc.ts == "09:56:55" and inc.source == "health.log" and inc.line == 1


def test_signatures_for_incidents_that_are_not_in_the_logs() -> None:
    cases = {
        "unseen_kind_vram": "planner: unseen kind xyz, no VRAM reserve",
        "pkill_self_match": "pkill -f nupen matched its own command line (self-match)",
        "autogen_no_state": "10:00:00 autogen: state is empty, nothing to do",
        "deadman_unarmed": "guardian: deadman not armed",
        "base_download_wait": "waiting for the base to finish",
        "eval_too_small": '  "n": 12,',
    }
    for kind, line in cases.items():
        assert kind in {k.kind for k in K.match_line(line)}, kind
    assert K.match_line("everything is fine, util=98") == []


def test_every_entry_is_complete_and_remedies_are_ranked() -> None:
    assert set(K.BLUEPRINT_KINDS) <= set(K.KINDS)
    assert len({k.id for k in K.TABLE}) == len(K.TABLE)
    for k in K.TABLE:
        assert k.diagnosis and k.verification and k.remedies, k.id
        assert all(r.action and r.expected_effect and r.risk in ("low", "medium", "high") for r in k.remedies)
        k.compiled()
    assert K.remedies_for("vram_starvation")[0].risk == "low"


def test_metric_conditions() -> None:
    rec = K.parse_health(HEALTH.splitlines()[0])
    assert rec and rec["util"] == 0 and rec["guard"] == 1
    assert [k.kind for k in K.match_metrics(rec)] == ["feeder_stall"]
    assert K.parse_health("09:00:00 module a: start") is None


# ---------------------------------------------------------------------------------------------------------- diff checker
def _diff(path: str, added: list[str], removed: int = 0) -> str:
    body = "".join(f"-old{i}\n" for i in range(removed)) + "".join(f"+{a}\n" for a in added)
    return f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1,{removed} +1,{len(added)} @@\n{body}"


def test_diff_ok() -> None:
    v = PO.check_diff(_diff("creator/foo.py", ["x = 1", "y = '10.x.y'", "ver = '1.2.3'"], removed=3))
    assert v.verdict == "ok" and v.deleted == 3 and v.added == 3 and bool(v)


def test_diff_catches_seeded_violations() -> None:
    fake_key = "sk-" + "a1b2c3d4e5" * 3
    cases = {
        "protected": _diff("creator/sandbox.py", ["x = 1"]),
        "protected-glob": _diff("creator/audit/z.py", ["x = 1"]),
        "sealed-livesim": _diff("state/livesim/a.json", ["{}"]),
        "sealed-masterstock": _diff("Masterstock/notes.md", ["hi"]),
        "sealed-oldpc": _diff("docs/oldpc/a.txt", ["hi"]),
        "api-key": _diff("creator/a.py", [f'KEY = "{fake_key}"']),
        "token": _diff("creator/a.py", ["t = 'ghp_" + "A" * 30 + "'"]),
        # seeded violations are assembled at run time so this test file itself never trips the deploy gate (check_diff over git diff)
        "private-key": _diff("creator/a.py", ["-----BEGIN RSA " + "PRIVATE KEY-----"]),
        "ip": _diff("creator/a.py", ["host = '" + ".".join(["203", "0", "113", "77"]) + "'"]),
        "home": _diff("creator/a.py", ["P = '" + "C:/" + "Users/someone/data'"]),
        "home-posix": _diff("creator/a.py", ["P = '/" + "home/someone/data'"]),
        "creds-url": _diff("creator/a.py", ["u = 'https://" + "bob:hunter22" + "@example.org/x'"]),
        "ssh": _diff("creator/a.sh", ["ssh -p 22 " + "root" + "@pod.example.org uptime"]),
        "assigned": _diff("creator/a.py", ["api_" + "token = '" + "abcdefghijklmnopqrstuvwx'"]),
    }
    for name, d in cases.items():
        assert PO.check_diff(d).verdict == "block", name
    assert PO.check_diff(cases["protected"], allow_protected=True).verdict == "ok"
    assert PO.check_diff(cases["sealed-livesim"], allow_protected=True).verdict == "block"   # sealed is never allowed


def test_diff_review_rules_and_reasons_hide_the_secret() -> None:
    assert PO.check_diff(_diff("creator/a.py", [], removed=500)).verdict == "review"
    assert PO.check_diff(_diff("creator/a.py", [], removed=500), max_deleted=1000).verdict == "ok"
    assert PO.check_diff(_diff("creator/big.py", ["x" * 300] * 20), max_file_bytes=1000).verdict == "review"
    assert PO.check_diff(_diff("creator/blob.py", ["y" * 30000])).verdict == "review"
    d = "diff --git a/i.png b/i.png\nBinary files a/i.png and b/i.png differ\n"
    assert PO.check_diff(d).verdict == "review"
    v = PO.check_diff(_diff("creator/a.py", ['K = "sk-' + "q9" * 15 + '"']))
    assert v.reasons and all("q9q9" not in r for r in v.reasons)
    assert PO.check_diff("").verdict == "ok"


def test_diff_checker_is_deterministic_and_fast() -> None:
    import time
    d = _diff("creator/a.py", [f"line {i} = {i}" for i in range(5000)])
    t = time.perf_counter()
    a, b = PO.check_diff(d), PO.check_diff(d)
    assert a == b and time.perf_counter() - t < 1.0
