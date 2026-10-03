"""Continuous trial and error (3 Oct 2026, creator.trialerror): online allocation + racing, feature families (walk-forward, no leakage),
search hygiene (threshold, noise stop, journal digest), fast reaction, the experiments section, and the start-load guard."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from creator import drillsources as D
from creator import thinking as T
from creator import trialerror as TE


# ------------------------------------------------------------------------------------------------ 1 allocation
def test_allocate_keeps_the_floor_and_never_asks_a_raced_out_arm() -> None:
    gains = {"bad": [-1.0] * 30, "good": [0.3, 0.1, 0.2] * 10, "new": [], "meh": [0.0, 0.1, -0.1] * 5}
    for seed in range(20):
        arms = TE.allocate(gains, "ctl", ["inc"], random.Random(seed), k=2)
        assert arms[:2] == ["ctl", "inc"] and len(arms) == 4 and len(set(arms)) == 4
        assert "bad" not in arms                                               # raced out: its CI excludes any gain
    assert TE.raced_out([-1.0] * TE.RACE_MIN) and not TE.raced_out([-1.0] * (TE.RACE_MIN - 1))
    assert not TE.raced_out([0.5, -0.5] * 20)                                  # an undecided arm stays in


def test_thompson_concentrates_on_the_best_arm_and_is_seeded() -> None:
    gains = {"a": [0.0, 0.1] * 20, "b": [0.4, 0.5] * 20, "c": [0.1, 0.0] * 20}
    picks = [TE.thompson_order({k: (lambda r, d=d: r.gauss(*TE.posterior(len(d), sum(d), sum(x * x for x in d), 0.05))) for k, d in gains.items()},
                               random.Random(i), 1)[0] for i in range(200)]
    assert picks.count("b") > 80                                              # the leader is taken with prob TOP_TWO_BETA, challengers otherwise
    assert TE.allocate(gains, "x", [], TE.seeded(1, 2), 2) == TE.allocate(gains, "x", [], TE.seeded(1, 2), 2)
    assert TE.best_arm(gains) == "b" and TE.best_arm({"a": [1.0], "b": [0.2] * 30}, min_n=20) == "b"


def test_posterior_never_claims_certainty_from_two_equal_answers() -> None:
    m, sd = TE.posterior(2, 2.0, 2.0, 0.25)
    assert sd > 0.15 and 0 < m < 1.0


def test_simulation_online_beats_halving_at_equal_calls_on_accuracy_arms() -> None:
    """Reasoning-like arms (accuracy; control 0.40; one clearly best, the rest worse than the control): at the same number of calls the
    online allocation (control + 3 top-two Thompson picks per question) errs less often than the current halving (fixed seeds)."""
    ps = [0.40, 0.25, 0.28, 0.30, 0.33, 0.36, 0.50]
    for scale in (1, 2):
        rnd = random.Random(21)
        sims = 200
        h = [TE.sim_halving(ps, scale, rnd) for _ in range(sims)]
        budget = h[0][1]
        o = [TE.sim_online(ps, budget, rnd, mode="beta", control_every=4, per_batch=3) for _ in range(sims)]
        err_h = sum(c != 6 for c, _n in h) / sims
        err_o = sum(c != 6 for c, _n in o) / sims
        assert err_o < err_h - (0.05 if scale == 1 else 0.0), (scale, err_o, err_h)


def test_simulation_racing_stops_early_when_no_strategy_beats_the_free_predictor() -> None:
    """The audited judgment case: every strategy is worse than the statistical predictor. Plain halving spends its whole schedule; racing on
    the paired gain over the (free) predictor stops far earlier and most often concludes 'none better'."""
    ps = [0.60, 0.40, 0.42, 0.45, 0.48, 0.50, 0.52, 0.55, 0.56, 0.57]
    sched = ((4, 8), (10, 4), (20, 2), (40, 1))
    rnd = random.Random(5)
    plain = [TE.sim_halving(ps, 10, rnd, sched, rho=0.8, control_cost=0) for _ in range(60)]
    raced = [TE.sim_halving(ps, 10, rnd, sched, rho=0.8, control_cost=0, race=True) for _ in range(60)]
    assert sum(n for _c, n in raced) < 0.7 * sum(n for _c, n in plain)
    assert sum(1 for c, _n in raced if c == -1) > 10


def test_online_simulation_ends_when_every_arm_is_raced_out_with_a_free_control() -> None:
    c, n = TE.sim_online([0.95, 0.05, 0.05], 10_000, random.Random(1), control_cost=0, mode="paired")
    assert n < 10_000                                                       # stopped: nothing left to ask, never spins


# ------------------------------------------------------------------------------------------------ 2 feature families
def _commits(n: int = 120, seed: int = 3) -> list[dict[str, Any]]:
    rnd = random.Random(seed)
    out = []
    for i in range(n):
        f = {f"d{rnd.randrange(3)}/f{rnd.randrange(6)}.py"}
        s = "fix crash" if rnd.random() < 0.3 else "add thing"
        out.append({"h": f"{i:010d}", "t": 1000.0 + i * 60 - (i % 2) * 60 * 0, "s": s, "files": f, "lines": 3, "add": 2, "del": 1})
    return out


def _x(items: list[D.BItem], repo: str = "r") -> list[D.BItem]:
    return [D.BItem(it.keys, it.created, it.resolved, it.y, it.subject, {**it.meta, "repo": repo}) for it in items]


@pytest.mark.parametrize("fam", TE.GIT_FAMILIES + TE.GENERIC_FAMILIES)
def test_families_use_only_items_created_before(fam: str) -> None:
    cs = _commits()
    items = D._git_events(cs, D.FIX_WINDOW, "fixed")
    base = TE.with_family(items, fam)
    # rewrite the FUTURE (from item 60 on): other files, other messages, flipped outcomes, removed items - nothing before 60 may change
    rnd = random.Random(9)
    fut = [D.BItem(it.keys, it.created, it.resolved, 1 - it.y, it.subject,
                   {**it.meta, "files": {f"z/{rnd.randrange(99)}"}, "s": "fix -"} if i >= 60 else it.meta) for i, it in enumerate(items)]
    for cut in (fut, items[:60]):
        again = TE.with_family(cut, fam)
        assert [it.keys for it in again[:60]] == [it.keys for it in base[:60]], fam
    # outcomes are never read: flipping every y changes no key
    flipped = [D.BItem(it.keys, it.created, it.resolved, 1 - it.y, it.subject, it.meta) for it in items]
    assert [it.keys for it in TE.with_family(flipped, fam)] == [it.keys for it in base]


def test_fix_history_counts_earlier_fixes_of_the_same_file_only() -> None:
    cs = [{"h": "a" * 10, "t": 1.0, "s": "fix bug", "files": {"x/a.py"}, "lines": 1, "add": 1, "del": 0},
          {"h": "b" * 10, "t": 2.0, "s": "fix again", "files": {"x/a.py"}, "lines": 1, "add": 1, "del": 0},
          {"h": "c" * 10, "t": 2.0, "s": "add", "files": {"x/a.py"}, "lines": 1, "add": 1, "del": 0},     # same time as b: does not see b
          {"h": "d" * 10, "t": 3.0, "s": "add", "files": {"x/a.py"}, "lines": 1, "add": 1, "del": 0},
          {"h": "e" * 10, "t": 4.0, "s": "add", "files": {"y/b.py"}, "lines": 1, "add": 1, "del": 0}]
    ks = [it.keys for it in TE.with_family(D._git_events(cs, 1, "fixed"), "fixhist")]
    assert ks[0][-2] == "fx:0" and ks[1][-2] == "fx:1" and ks[2][-2] == "fx:1" and ks[3][-2] == "fx:1" and ks[4][-2] == "fx:0"
    assert ks[3][-1] == "fxd:1"                                                  # directory x/ had 2 fixes before t=3 -> log2(3)=1


def test_file_identity_is_per_project() -> None:
    cs = [{"h": "a" * 10, "t": 1.0, "s": "fix", "files": {"h1/h2"}, "lines": 1, "add": 1, "del": 0},
          {"h": "b" * 10, "t": 2.0, "s": "add", "files": {"h1/h2"}, "lines": 1, "add": 1, "del": 0}]
    a = _x(D._git_events(cs[:1], 1, "fixed"), "p1")
    b = _x(D._git_events(cs[1:], 1, "fixed"), "p2")
    ks = [it.keys for it in TE.with_family(a + b, "fixhist")]
    assert ks[1][-2] == "fx:0"                                                   # the same hashed path in ANOTHER project is another file


def test_apply_variant_appends_the_family_and_live_paths_see_it(tmp_path: Path) -> None:
    items = D._git_events(_commits(), D.FIX_WINDOW, "fixed")
    out = D.apply_variant(items, {"decay": 0.97, "k": 3.0, "fam": "fixhist+gap"}, tmp_path / "none.md")
    assert all(len(o.keys) == len(i.keys) + 3 for o, i in zip(out, items)) and out[0].meta is items[0].meta
    preds = D.predict_open(out, "git_fixed", {"decay": 0.97, "k": 3.0, "fam": "fixhist"})
    assert preds and all(0 < p.p < 1 for p in preds)


def test_family_predictions_never_see_future_outcomes() -> None:
    items = D._git_events(_commits(200), D.FIX_WINDOW, "fixed")
    v = {"decay": 0.97, "k": 3.0, "fam": "fixhist+churnhist"}
    a = D.walk_forward(D.apply_variant(items, v, Path("none")), "x", 0.97, 3.0)
    late = [D.BItem(it.keys, it.created, it.resolved, 1 - it.y if i >= 120 else it.y, it.subject, it.meta) for i, it in enumerate(items)]
    b = D.walk_forward(D.apply_variant(late, v, Path("none")), "x", 0.97, 3.0)
    cut = items[120].created
    assert [p.p for p in a if p.made_at < cut - 1] == [p.p for p in b if p.made_at < cut - 1]


def test_propose_tries_families_git_ones_only_on_the_cross_repo_sources() -> None:
    rnd = random.Random(1)
    xs = [D.propose("x_git_fixed", [], rnd) for _ in range(300)]
    own = [D.propose("git_fixed", [], rnd) for _ in range(300)]
    assert sum(1 for v in xs if v.get("fam")) > 50
    assert any(str(v.get("fam", "")).split("+")[0] in TE.GIT_PARTS for v in xs)
    assert all(str(v["fam"]).split("+")[0] not in TE.GIT_PARTS for v in own if v.get("fam"))
    assert D._proposal_why({"fam": "gap"}).startswith("idea")


# ------------------------------------------------------------------------------------------------ search hygiene
def _row(sel: float, held: float, search: int = 1, ci: tuple[float, float] = (0.0, 0.04), dg: str = "d") -> dict[str, Any]:
    return {"digest": dg, "search": search, "select": {"n": 100, "brier": sel, "gain_ci95": list(ci)}, "heldout": {"n": 50, "brier": held},
            "variant": {"decay": sel}}


def test_a_gain_inside_the_noise_does_not_reset_the_stop_rule() -> None:
    rows = [_row(0.2000, 0.2)] + [_row(0.2000 - 0.0001 * (i + 1), 0.2) for i in range(D.SEARCH_STOP_K)]
    st = D.search_state(rows, "d")
    assert st["consecutive_fails"] == D.SEARCH_STOP_K and st["stopped"]           # old rule: every 1e-4 gain reset it to 0
    big = rows[:3] + [_row(0.18, 0.18)]
    assert D.search_state(big, "d")["consecutive_fails"] == 0


def test_a_source_whose_select_part_does_not_predict_heldout_stops() -> None:
    rnd = random.Random(4)
    noise = [_row(0.25 + 0.05 * rnd.random(), 0.25 + 0.05 * rnd.random(), ci=(0.0, 0.0001)) for _ in range(20)]
    noise.sort(key=lambda r: -r["select"]["brier"])                                # every row a new best: fails never accumulate
    st = D.search_state(noise, "d")
    assert st["consecutive_fails"] == 0 and st["select_heldout_rho"] is not None
    good = [_row(0.2 + 0.001 * i, 0.2 + 0.001 * i, ci=(0.0, 0.0001)) for i in range(20)][::-1]
    assert not D.search_state(good, "d")["noise_stopped"]
    if st["select_heldout_rho"] < TE.NOISE_RHO:
        assert st["noise_stopped"] and st["stopped"]


def test_journal_digest_follows_the_items_not_the_bytes(tmp_path: Path) -> None:
    j = tmp_path / "JOURNAL.md"
    j.write_text("- 2026-10-01 nupen kernel work\n- 2026-10-02 stock backtest\n", encoding="utf-8")
    d1 = D.source_digest("journal_persist", tmp_path, tmp_path, j, tmp_path)
    j.write_text("# heading\n\n- 2026-10-01 nupen kernel work\n- 2026-10-02 stock backtest\n\n", encoding="utf-8")
    assert D.source_digest("journal_persist", tmp_path, tmp_path, j, tmp_path) == d1
    j.write_text("- 2026-10-01 nupen kernel work\n- 2026-10-02 stock backtest\n- 2026-10-03 spanish app\n", encoding="utf-8")
    assert D.source_digest("journal_persist", tmp_path, tmp_path, j, tmp_path) != d1


# ------------------------------------------------------------------------------------------------ 3 fast reaction + 4 visible
def _git_repo_state(tmp_path: Path, monkeypatch: Any) -> tuple[Path, list[dict[str, Any]]]:
    state = tmp_path / "st"
    cs = _commits(400, seed=8)
    for i, c in enumerate(cs):                                                      # a planted weakness: d0 files are fixed far more often
        if "d0/" in next(iter(c["files"])):
            c["s"] = "fix d0"
    monkeypatch.setattr(D, "git_commits", lambda repo, st=None, allow_full=False: cs)
    monkeypatch.setattr(D, "_git_head", lambda repo, timeout=60.0: "h" * 12)
    (state / "thinking").mkdir(parents=True)
    return state, cs


def test_react_requeues_targeted_variants_at_once_and_is_rate_limited(tmp_path: Path, monkeypatch: Any) -> None:
    state, _cs = _git_repo_state(tmp_path, monkeypatch)
    row = D.run_job("git_fixed", {"decay": 1.0, "k": 50.0, "cap": 0.02}, state, tmp_path, tmp_path / "J.md", tmp_path)
    assert row["select"]["n"]
    out = TE.react(state, row, tmp_path, tmp_path / "J.md", tmp_path, now=1000.0)
    assert out is not None and out["trigger"] == "new_best"
    from creator import learnloop as LL
    q = LL.queue_rows(state)
    assert out["queued"] == len(q) and all(x["source"] == "git_fixed" and x["improvement"].startswith("RX-") for x in q)
    assert {x["source"] for x in LL.queued_variants(state)} <= {"git_fixed"}
    assert TE.react(state, row, tmp_path, tmp_path / "J.md", tmp_path, now=1001.0) is not None   # the 2nd within the hour
    assert TE.react(state, row, tmp_path, tmp_path / "J.md", tmp_path, now=1002.0) is None       # the 3rd: rate limited
    assert TE.react(state, row, tmp_path, tmp_path / "J.md", tmp_path, now=1000.0 + 3700.0) is not None


def test_filler_reacts_after_a_job_and_records_why(tmp_path: Path, monkeypatch: Any) -> None:
    state, _cs = _git_repo_state(tmp_path, monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr(TE, "react", lambda st, row, *a, **k: calls.append(row["source"]))
    nj = D.drill_filler(state, tmp_path, tmp_path / "J.md", tmp_path, sources=["git_fixed"], seed=1)
    for _ in range(len(D.VARIANTS) + 6):
        j = nj()
        if j is None:
            break
        j()
    rows = T._jsonl(D.runs_path(state))
    assert calls and len(calls) == len([r for r in rows if r.get("select")])
    assert any(str(r.get("why", "")).startswith(("search", "idea")) for r in rows if r.get("search"))


def test_experiments_section_shows_tried_why_select_heldout_kept(tmp_path: Path, monkeypatch: Any) -> None:
    state, _cs = _git_repo_state(tmp_path, monkeypatch)
    for v in ({"decay": 1.0, "k": 3.0}, {"decay": 0.9, "k": 3.0, "fam": "gap", "search": 1}, {"decay": 1.0, "k": 3.0, "learn": "IM-x", "search": 1}):
        D.run_job("git_fixed", v, state, tmp_path, tmp_path / "J.md", tmp_path)
    ex = TE.experiments_section(state)
    g = ex["drills"]["git_fixed"]
    assert g["tried"] == 3 and len(g["latest"]) == 3
    assert {e["why"] for e in g["latest"]} == {"grid", "idea: feature family gap", "diagnosis IM-x"}
    assert all({"select", "heldout", "kept"} <= set(e) for e in g["latest"]) and g["latest"][0]["kept"]
    assert set(g["families"]) == {"none", "gap"} and g["families"]["gap"]["tried"] == 1
    tj = T.trust(state, write=False, owner_dir=tmp_path)
    assert "drills" in tj["experiments"]
    from creator import learnloop as LL
    assert "experiments" in LL.KNOWN_SECTIONS and LL.experiments(state)["git_fixed"]["tried"] == 3


# ------------------------------------------------------------------------------------------------ start load
def test_trial_and_error_stays_out_of_the_swarm_start_load() -> None:
    from creator import efficiency as E
    root = Path(__file__).resolve().parents[1]
    load = E.start_load(root)
    assert load["scripts/creator_swarm.py"] < 85_000, load
    eager = E.eager_graph(root, E.START_ENTRIES)
    closure = E._closure(eager, "scripts/creator_swarm.py")
    assert not {"creator/trialerror.py", "creator/drillsources.py", "creator/judgment.py", "creator/reasondrills.py", "creator/benchfill.py"} & set(closure)
