"""The practice runner (supervisor), the retrain-on-new-rows rule, predictions that travel with a claimed change, the practice metric,
and the shadow rule ignoring practice rows. Subprocesses are fakes."""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from creator import action_student as A
from creator import chooser as CH
from creator import constraints as CON
from creator import shadow
from tests.test_creator_action_student import PKG, PLAN, FakeLLM, work  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _load(name: str):                                                       # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)                             # type: ignore[arg-type]
    spec.loader.exec_module(mod)                                            # type: ignore[union-attr]
    return mod


class Proc:
    def __init__(self) -> None:
        self.pid, self.returncode = 7, None

    def poll(self):                                                         # type: ignore[no-untyped-def]
        return self.returncode


def runner(tmp_path: Path, idle: list, free: list, lm: list | None = None):  # type: ignore[no-untyped-def]
    svc = _load("nupen_service")
    spawned, killed, clock = [], [], [1000.0]
    rows = tmp_path / "rows.jsonl"
    pr = svc.PracticeRunner(python="py", idle=lambda: idle[0], free_gb=lambda: free[0], spawn=lambda c: (spawned.append(c), Proc())[1],
                            stop=lambda p: (killed.append(p), setattr(p, "returncode", 1)), clock=lambda: clock[0], log=lambda m: None,
                            lm_running=lambda: bool(lm and lm[0]), rows_path=rows, gap_s=60.0, empty_gap_s=600.0)
    return svc, pr, spawned, killed, clock, rows


def test_practice_runs_only_when_idle_and_ram_allow(tmp_path: Path) -> None:
    idle, free = [30.0], [8.0]
    _s, pr, spawned, _k, _c, _r = runner(tmp_path, idle, free)
    pr.tick()
    assert not spawned
    idle[0], free[0] = 3600.0, 1.0
    pr.tick()
    assert not spawned
    free[0] = 8.0
    pr.tick()
    assert len(spawned) == 1 and spawned[0][1:3] == ["-u", str(ROOT / "scripts" / "practice.py")] and "--rev" in spawned[0]


def test_practice_never_runs_alongside_the_lm_trainer_and_blocks_it(tmp_path: Path) -> None:
    lm = [True]
    svc, pr, spawned, _k, _c, _r = runner(tmp_path, [3600.0], [8.0], lm)
    pr.tick()
    assert not spawned                                                      # the LM trainer holds the one background slot
    lm[0] = False
    pr.tick()
    assert len(spawned) == 1
    py = tmp_path / "python.exe"
    py.write_text("", encoding="utf-8")
    started = []
    tr = svc.LMTrainer(python=py, idle=lambda: 3600.0, free_gb=lambda: 8.0, spawn=lambda c: (started.append(c), Proc())[1],
                       stop_file=tmp_path / "STOP", clock=lambda: 1000.0, log=lambda m: None, blocked=pr.running)
    tr.tick()
    assert not started                                                      # practice running: the trainer waits
    pr.proc.returncode = 0
    tr.blocked = pr.running
    tr.tick()
    assert len(started) == 1


def test_practice_stops_when_the_owner_returns(tmp_path: Path) -> None:
    idle = [3600.0]
    _s, pr, spawned, killed, clock, _r = runner(tmp_path, idle, [8.0])
    pr.tick()
    idle[0] = 5.0
    pr.tick()
    assert len(killed) == 1 and pr.proc is None
    pr.tick()
    assert len(spawned) == 1                                                # still not idle: nothing new
    idle[0] = 3600.0
    clock[0] += 100
    pr.tick()
    assert len(spawned) == 2


def test_nupen_stop_ends_practice_and_the_retrain_follows_only_new_rows(tmp_path: Path) -> None:
    _s, pr, spawned, killed, clock, rows = runner(tmp_path, [3600.0], [8.0])
    pr.tick()
    pr.tick(halt=True)
    assert len(killed) == 1
    clock[0] += 100
    pr.tick()
    assert len(spawned) == 2
    pr.proc.returncode = 0                                                  # a run that measured nothing new: no retrain, long wait
    pr.tick()
    assert len(spawned) == 2 and pr.proc is None
    clock[0] += 100
    pr.tick()
    assert len(spawned) == 2
    clock[0] += 600
    pr.tick()
    assert len(spawned) == 3
    rows.write_text('{"source": "prescreen"}\n', encoding="utf-8")          # now rows arrive: the retrain job follows
    pr.proc.returncode = 0
    pr.tick()
    assert len(spawned) == 4 and spawned[3][-2:] == [str(ROOT / "scripts" / "train_chooser.py"), "--practice"]


def _practice_state(state: Path) -> None:
    from creator import action_student as AS
    src = "import csv\nimport os\n\n\ndef rows(t):\n    return list(csv.reader(t))\n\n\ndef dead():\n    return 1\n"
    h = "abc123"
    (state / "practice_src").mkdir(parents=True, exist_ok=True)
    (state / "practice_src" / f"{h}.py").write_text(src, encoding="utf-8")
    acts = AS.enumerate_actions(src, "app/u.py")
    lines = []
    for f in ("app/u.py", "app/v.py", "app/w.py"):
        for i, a in enumerate(acts):
            lines.append(json.dumps({"source": "prescreen", "path": f, "src_hash": h, "metric": "size", "objective": "shrink " + f,
                                     "action": {**a.to_dict(), "path": f}, "good": i % 2 == 0, "at": dt.datetime.now().isoformat(timespec="seconds")}))
    (state / "practice_rows.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_retrain_only_when_new_rows_arrived(tmp_path: Path) -> None:
    import train_chooser as TC
    state = tmp_path
    assert TC.retrain_with_practice(state, epochs=20) is None               # no rows at all
    _practice_state(state)
    rec = TC.retrain_with_practice(state, epochs=20, holdout_folds=3)
    assert rec and (state / "chooser.json").is_file() and rec["practice_rows"] > 0
    assert "chooser_top1" in rec["before"] and "chooser_top1" in rec["after"]
    log = (state / "practice_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(log) == 1
    first = (state / "chooser.json").read_bytes()
    assert TC.retrain_with_practice(state, epochs=20, holdout_folds=3) is None      # nothing new: no retrain, no second record
    assert len((state / "practice_log.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert (state / "chooser.json").read_bytes() == first
    with (state / "practice_rows.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"source": "prescreen", "path": "app/z.py", "src_hash": "abc123", "metric": "size", "objective": "shrink z",
                            "action": json.loads((state / "practice_rows.jsonl").read_text(encoding="utf-8").splitlines()[0])["action"], "good": True}) + "\n")
    assert TC.retrain_with_practice(state, epochs=20, holdout_folds=3) is not None


def test_a_claimed_change_carries_its_prediction(work: Path, tmp_path: Path) -> None:   # noqa: F811
    st = A.ActionStudent(tmp_path / "n.jsonl", llm=FakeLLM("CHOICE: 1\nWHY: shrink it"), state_dir=tmp_path / "s", prescreen=True, prescreen_tests=False)
    res = st(PLAN, PKG, work)
    if not res.claimed_done:
        pytest.skip(res.notes)
    p = A.parse_prediction(res.notes)
    assert p is not None and p["metric"] in ("size", "activation") and isinstance(p["size_delta"], int)
    assert A.parse_prediction(res.reasoning) == p
    assert A.parse_prediction("no prediction here") is None


def test_practice_is_counted_separately_from_real_verdicts(tmp_path: Path) -> None:
    now = dt.datetime.now()
    _practice_state(tmp_path)
    (tmp_path / "practice_log.jsonl").write_text(json.dumps({"before": {"chooser_top1": 0.3}, "after": {"chooser_top1": 0.5, "random_top1": 0.4, "decisions": 9}}) + "\n",
                                                 encoding="utf-8")
    m = CON.practice_metric(tmp_path, now + dt.timedelta(minutes=1), 24.0)
    assert m.name == "practice_signal" and m.value > 0 and m.loss == 0.0 and m.score == 0.0
    assert m.detail["chooser_holdout_after"] == 0.5 and m.detail["chooser_holdout_before"] == 0.3 and m.detail["real_verdicts_separate"]
    names = [x.name for x in CON.learning_metrics(tmp_path, now, 24.0)]
    assert "practice_signal" in names and names[0] == "learning_signal"
    real = [x for x in CON.learning_metrics(tmp_path, now, 24.0) if x.name == "learning_signal"][0]
    assert real.detail["attempts"] == 0                                     # practice rows are no attempts


def test_shadow_rule_still_ignores_practice_rows(tmp_path: Path) -> None:
    _practice_state(tmp_path)
    s = shadow.update_policy(tmp_path, lessons_path=tmp_path / "lessons.jsonl")
    assert s["n"] == 0 and shadow.read_policy(tmp_path) is None
    assert CH.practice_rows(tmp_path / "practice_rows.jsonl", tmp_path / "practice_src")
    src = Path(shadow.__file__).read_text(encoding="utf-8")
    assert "practice_rows" not in src and "prescreen" not in src
