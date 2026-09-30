"""F17 (C69 sections 12-14, 27, 28, 31): the research loop filed less after F14 + F16 - diagnosis.

Finding: the regression is NOT in the gate. In every 'after' run (F14+F16 code) the very branches that filed in the 'before' runs (same
branch ids: seed 1 BR5224b8909968 xs_atr_rank and BRfb5bbfb1ce25 xs_vol_rank, seed 3 BRfb5bbfb1ce25, seed 4 BR9838ffe9fade xs_range_rank,
seed 5 BR2d20b91b1f58 vol_over_mkt) died in their FIRST ladder rung, cycles 1-6, with 'reconcile rejected: stale_code', and never reached
the gate. The runner (tests/test_regate_sequential.py run_loop, --fresh) reuses the run folder, so the compute ledger and the attempt folders
of the previous run are still there; the new run submits the same experiment keys, the ledger supersedes the old DONE entry and the rerun
writes attempt_02 (the 1a41b240 folder fix) - but engine.learning.compute.reconcile reads EVERY attempt folder of the key, finds the
superseded run's attempt_01 produced by the old code and rejects the fresh, valid result as stale_code. The 'before' runs were poisoned the
same way by the F12 runs (their early lv20 branches died), which is why they filed xs_* and the 'after' runs file lv20: neither run set is
a clean measurement of the gate.

The defect lives in engine/learning/compute.py (read-only for F17) and is documented here by a strict xfail that flips when it is fixed;
the null case (a result produced ONLY by old code is still stale) and the empty case must keep passing."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.learning import compute as CM                                     # noqa: E402

OLD, NEW = "code_before", "code_after"


def _spec(name="BR5224b8909968_s1"):
    return CM.ExperimentSpec(name, seed=1, as_of="2017-05-19", params={"branch": name[:14], "stage": "STAGE1_CHEAP_SCREEN"})


def _fn(sp, ctx):
    return {"effect": float(ctx.rng("main").normal()), "n": 5}


def _rerun_under_new_code(tmp_path):
    led, out = CM.ExperimentLedger(tmp_path / "ledger.json"), tmp_path / "out"
    sp = _spec()
    led.submit(sp, 0.0, OLD)
    assert CM.run_worker(sp, _fn, led, None, out, "w_before", 1.0, OLD)["state"] == CM.DONE     # the previous run, same folder
    assert led.submit(sp, 2.0, NEW).action == "superseding"                                      # the new run: same key, new code
    res = CM.run_worker(sp, _fn, led, None, out, "w_after", 3.0, NEW)
    assert res["state"] == CM.DONE and res["dir"].endswith("attempt_02")
    return led, out, sp


@pytest.mark.xfail(strict=True, reason="F17 defect in engine/learning/compute.reconcile (read-only here): a superseded run's attempt "
                                      "folder makes the current code's valid rerun 'stale_code' - the loop's first rungs die")
def test_a_rerun_under_new_code_is_accepted_despite_the_superseded_attempt(tmp_path):
    led, out, sp = _rerun_under_new_code(tmp_path)
    rec = CM.reconcile(led, out, NEW)
    assert sp.key in dict(rec.accepted), rec.rejected


def test_the_defect_as_it_hits_the_loop_today(tmp_path):
    """Planted: the exact sequence of the after-runs. Today the only finished attempt of the NEW code is rejected, which is what killed
    the xs_* / vol_over_mkt branches in cycles 1-6. When compute.reconcile is fixed this test must be updated with the xfail above."""
    led, out, sp = _rerun_under_new_code(tmp_path)
    rej = {k: why for k, why, _ in CM.reconcile(led, out, NEW).rejected}
    assert rej.get(sp.key) == "stale_code"
    # the new code's own result is whole and valid: the rejection is caused solely by the superseded attempt_01
    envs = CM._read_attempts(Path(out), sp.key)
    assert [e["code_hash"] for e in envs] == [OLD, NEW]
    only_new = [e for e in envs if int(Path(e["_dir"]).name.split("_")[1]) > CM.attempt_serial(led.get(sp.key)) - led.get(sp.key)["attempts"]]
    assert [e["code_hash"] for e in only_new] == [NEW] and only_new[0]["result_hash"] == led.get(sp.key)["result_hash"]


def test_null_a_result_made_only_by_old_code_stays_stale(tmp_path):
    """Null case: without a rerun, an old-code result must never be accepted under the new code (any fix must keep this)."""
    led, out = CM.ExperimentLedger(tmp_path / "ledger.json"), tmp_path / "out"
    sp = _spec("BR2d20b91b1f58_s1")
    led.submit(sp, 0.0, OLD)
    CM.run_worker(sp, _fn, led, None, out, "w", 1.0, OLD)
    rec = CM.reconcile(led, out, NEW)
    assert not rec.accepted and [(k, why) for k, why, _ in rec.rejected] == [(sp.key, "stale_code")]
    assert [k for k, _ in CM.reconcile(led, out, OLD).accepted] == [sp.key]            # under its own code it is valid


def test_empty_ledger_and_missing_folder_reconcile_to_nothing(tmp_path):
    rec = CM.reconcile(CM.ExperimentLedger(tmp_path / "ledger.json"), tmp_path / "no_such_out", NEW)
    assert rec.accepted == () and rec.rejected == () and rec.orphans == ()


def test_same_code_rerun_disagreement_is_still_caught(tmp_path):
    """The reconcile rule the fix must not weaken: two finished attempts of the SAME code that disagree prove non-determinism."""
    led, out = CM.ExperimentLedger(tmp_path / "ledger.json"), tmp_path / "out"
    sp = _spec("BR9838ffe9fade_s1")
    led.submit(sp, 0.0, NEW)
    CM.run_worker(sp, _fn, led, None, out, "w", 1.0, NEW)
    d2 = CM.attempt_dir(out, sp.key, 2)
    d1 = CM.attempt_dir(out, sp.key, 1)
    d2.mkdir(parents=True)
    import json
    env = json.loads((d1 / "result.json").read_text(encoding="utf-8"))
    env["result"] = {"effect": float(np.pi), "n": 5}
    env["result_hash"] = CM.stable_hash(env["result"], 20)
    (d2 / "result.json").write_text(json.dumps(env), encoding="utf-8")
    (d2 / "DONE").write_text("", encoding="utf-8")
    assert [why for _, why, _ in CM.reconcile(led, out, NEW).rejected] == ["nondeterministic"]
