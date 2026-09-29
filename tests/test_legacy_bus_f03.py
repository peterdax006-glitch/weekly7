"""C69 F03: legacy hook bus and the fifth mover model. Each test fails on the pre-F03 behaviour (hub sinks never persisted from an entry
script; a learning-type challenger skipped the composite gate; the GBM hypothesis trained its own LightGBM). Synthetic data only.
IMPLEMENTED - NOT VALIDATED."""
import inspect
import json

import numpy as np
import pandas as pd
import pytest

from engine import fv_pipeline as FV
from engine.learning import wiring as W
from engine.research import vol_hypotheses as VH
from engine.research import volatility_lab as L
from tests.test_learning_integration import fresh_hub, lesson_panel  # noqa: F401  (fixture reuse)


# ---------------------------------------------------------------- (1) the hub persists from an entry script
def _closed_frame():
    from engine.antimemo import play_window
    X, y = lesson_panel()
    _, fr = play_window(X, y, lambda Xd: Xd["f0"].copy(), None, k=5, collect_frame=True, horizon_days=7)
    return X, fr


def test_configure_from_env_makes_records_land_on_disk(fresh_hub, tmp_path):
    X, fr = _closed_frame()
    assert W.HUB.root is None                                       # the old state of every production entry point
    W.on_post_mortem(fr, X, fr["resolved"].max() + pd.Timedelta(days=1))
    assert not list(tmp_path.rglob("*.jsonl"))                     # nothing persisted before configure
    W.HUB.reset()
    hub = W.configure_from_env("loop_a", environ={W.ENV_HUB_ROOT: str(tmp_path / "hub")})
    assert hub is W.HUB and hub.lane == "loop_a" and hub.root == tmp_path / "hub" / "loop_a"
    W.on_post_mortem(fr, X, fr["resolved"].max() + pd.Timedelta(days=1))
    f = tmp_path / "hub" / "loop_a" / "failures.jsonl"
    rows = [json.loads(l) for l in f.read_text().splitlines()]
    assert rows and all("cause" in r["body"] for r in rows)


def test_configure_from_env_defaults_to_the_production_root_and_honours_the_kill_switch(fresh_hub):
    hub = W.configure_from_env(environ={})
    assert hub.root == W.production_root() / "research_loop"
    W.HUB.reset()
    assert W.configure_from_env("x", environ={W.ENV_HUB: "off"}) is None and W.HUB.root is None
    with pytest.raises(ValueError, match="lane"):
        W.configure_from_env("../escape", environ={})


# ---------------------------------------------------------------- (2) weight is board-scaled and its pass-through is visible
def test_weight_passthrough_is_counted_and_board_scaling_is_counted_separately(fresh_hub, tmp_path):
    from tests.test_learning_integration import board_with
    assert W.effective_weight("k", 0.7) == 0.7
    assert W.HUB.delivered["weight_unregistered_passthrough"] == 1 and W.HUB.delivered["weight_board_scaled"] == 0
    W.configure(board=board_with(tmp_path))
    assert W.effective_weight("lesson-shadow", 0.7) == 0.0
    assert W.effective_weight("lesson-champ", 0.7) == 0.7
    assert W.HUB.delivered["weight_board_scaled"] == 2


# ---------------------------------------------------------------- (3) promotion gate: unknown kinds are learning claims
@pytest.mark.parametrize("ch", [{"id": "C1", "kind": "learner"}, {"id": "C2"}, {"id": "C3", "kind": "model", "claims_learning": False}])
def test_a_non_legacy_challenger_cannot_skip_the_composite_gate(fresh_hub, ch):
    assert W.challenger_claims_learning(ch)
    v = W.promotion_allowed(ch, "2026-09-29")
    assert v.claims_learning and not v.allowed and [c.name for c in v.checks] == ["scorecard", "firewalls", "identity"]


@pytest.mark.parametrize("kind", sorted(W.LEGACY_CHALLENGER_KINDS))
def test_legacy_kinds_stay_on_the_live_shadow_test(fresh_hub, kind):
    assert not W.challenger_claims_learning({"id": "L", "kind": kind})
    assert W.promotion_allowed({"id": "L", "kind": kind}, "2026-09-29").allowed


def test_registered_evidence_makes_even_a_legacy_kind_a_learning_claim(fresh_hub):
    W.register_evidence("C9", W.LearningEvidence())
    v = W.promotion_allowed({"id": "C9", "kind": "meta"}, "2026-09-29")
    assert v.claims_learning and not v.allowed                  # empty evidence = failed checks, not a pass


# ---------------------------------------------------------------- (4) one canonical mover model
FIT = VH.FitConfig(min_rows=300, min_events=20, gbm_trees=30, max_train_rows=20000)


@pytest.fixture(scope="module")
def world():
    return L.planted_frame("H1", n_dates=84, n_tickers=50, seed=11, effect=1.4)


def test_vol_hypotheses_has_no_private_lightgbm():
    assert "lightgbm" not in inspect.getsource(VH) and "lgb." not in inspect.getsource(VH)


def test_gbm_hypothesis_runs_on_fv_pipeline_moverstage_and_matches_it_exactly(world):
    h = next(x for x in VH.seeded_hypotheses() if x.kind == VH.HypKind.GBM)
    now = world.index.get_level_values(0).max() + pd.Timedelta(days=30)
    fh = VH.fit_hypothesis(h, world, now, FIT)
    assert fh.ok, fh.reason
    assert isinstance(fh.clf, VH.CanonicalMoverModel) and isinstance(fh.clf.stage, FV.MoverStage)
    p, _ = fh.predict(world)
    Z = fh.std.apply(VH.derive(world, h.features).to_numpy(float))
    ref = FV.MoverStage(fh.clf.stage.cfg)                        # the canonical model, fitted directly on the same rows
    F = world[~world["touch"].isna()]
    ref.fit(pd.DataFrame(Z, columns=list(h.features)), F["touch"].to_numpy(float), F.index.get_level_values(0).to_numpy())
    assert np.allclose(p, ref.score(pd.DataFrame(Z, columns=list(h.features))))


def test_gbm_hypothesis_declines_honestly_when_the_canonical_model_has_too_little_data(world):
    h = next(x for x in VH.seeded_hypotheses() if x.kind == VH.HypKind.GBM)
    tiny = world.iloc[:350]
    fh = VH.fit_hypothesis(h, tiny, world.index.get_level_values(0).max() + pd.Timedelta(days=30), FIT)
    assert not fh.ok and fh.reason
