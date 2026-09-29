"""Phase 42 claim standard, plus the Phase 30 experiment-memory answers (unseen / worsened / meaningful / survived).
Every check has a planted defect that must be caught and a clean control that must not be."""
import json

import numpy as np
import pytest

from engine import claims as C
from engine import experiment_memory as EM


# ---------------------------------------------------------------- fixtures
def artefact(tmp_path, **data):
    p = tmp_path / "art.json"
    p.write_text(json.dumps(data or {"mean_week": 0.0123, "nested": {"x": 0.5}}), encoding="utf-8")
    return p


def good(tmp_path, **over):
    art = artefact(tmp_path)
    kw = dict(claim_id="C1", statement="Mean weekly return of the candidate", value=0.0123, unit="%", artefact_path="art.json",
              definition="mean of simple weekly returns over all replayed windows, equal weight", root=tmp_path,
              artefact_key="mean_week",
              cost=C.CostAdjustment(applied=True, bps=10.0),
              comparison=C.Comparison(baseline="frozen baseline", candidate="candidate v2", statistic="paired bootstrap of mean weekly difference",
                                      estimate=0.004, ci_low=0.001, ci_high=0.007, n=180, p_value=0.01),
              control=C.Control(name="500 shuffled-label runs", result=0.0002, note="null mean"))
    kw.update(over)
    return C.make_claim(**kw)


# ---------------------------------------------------------------- claim rendering
def test_complete_claim_renders_the_whole_trail(tmp_path):
    c = good(tmp_path)
    txt = c.render(tmp_path)
    for needle in ("[C1]", "1.23%", "definition:", "sha256:" + c.artefact_hash[:16], "net of 10 bps", "frozen baseline", "n=180", "500 shuffled-label runs"):
        assert needle in txt
    assert c.problems(tmp_path) == []


@pytest.mark.parametrize("mutate,expect", [
    (lambda c: setattr(c, "artefact_hash", ""), "no artefact hash"),
    (lambda c: setattr(c, "artefact_path", ""), "no artefact path"),
    (lambda c: setattr(c, "definition", ""), "no exact definition"),
    (lambda c: setattr(c, "definition", "the number"), "too thin"),
    (lambda c: setattr(c.cost, "applied", None), "cost adjustment not stated"),
    (lambda c: (setattr(c.cost, "applied", False), setattr(c.cost, "note", "")), "gross of costs with no stated reason"),
    (lambda c: setattr(c.cost, "bps", None), "bps missing"),
    (lambda c: setattr(c.comparison, "baseline", ""), "no baseline"),
    (lambda c: setattr(c.comparison, "statistic", ""), "no statistic"),
    (lambda c: setattr(c.comparison, "ci_low", None), "no interval"),
    (lambda c: setattr(c.comparison, "n", 0), "sample size"),
    (lambda c: setattr(c.comparison, "p_value", 1.5), "p-value"),
    (lambda c: setattr(c.control, "name", ""), "no control"),
    (lambda c: setattr(c.control, "result", None), "control has no result"),
    (lambda c: setattr(c, "value", float("nan")), "value missing"),
    (lambda c: setattr(c, "statement", ""), "no statement"),
    (lambda c: setattr(c, "kind", "wild"), "unknown kind"),
])
def test_render_refuses_a_claim_missing_any_required_element(tmp_path, mutate, expect):
    c = good(tmp_path)
    mutate(c)
    with pytest.raises(C.ClaimIncomplete) as ei:
        c.render(tmp_path)
    assert expect in str(ei.value)


def test_all_gaps_are_listed_not_just_the_first(tmp_path):
    c = good(tmp_path)
    c.definition, c.control.name = "", ""
    c.cost.applied = None
    with pytest.raises(C.ClaimIncomplete) as ei:
        c.render(tmp_path)
    assert len(ei.value.problems) >= 3


def test_tampered_artefact_is_caught_by_the_hash(tmp_path):
    c = good(tmp_path)
    (tmp_path / "art.json").write_text(json.dumps({"mean_week": 0.0999}), encoding="utf-8")
    assert any("hash does not match" in p for p in c.problems(tmp_path))
    (tmp_path / "art.json").unlink()
    assert any("does not exist" in p for p in c.problems(tmp_path))


def test_claimed_value_must_be_what_the_artefact_holds(tmp_path):
    art = tmp_path / "art.json"
    art.write_text(json.dumps({"mean_week": 0.0123}), encoding="utf-8")
    c = good(tmp_path, value=0.0456)                                # artefact says 0.0123
    assert any("not the claimed" in p for p in c.problems(tmp_path))
    assert any("not found" in p for p in good(tmp_path, artefact_key="nope").problems(tmp_path))
    assert good(tmp_path, artefact_key="mean_week").problems(tmp_path) == []       # control


def test_drafting_a_claim_about_a_missing_file_fails_immediately(tmp_path):
    with pytest.raises(FileNotFoundError):
        C.make_claim("X", "s", 1.0, "ghost.json", "a long enough definition here", root=tmp_path)


def test_control_that_matches_or_swallows_the_claim_is_refused(tmp_path):
    c = good(tmp_path)
    c.control.result = c.comparison.estimate
    assert any("not distinguishable" in p for p in c.problems(tmp_path))
    c = good(tmp_path)
    c.control.result = 0.003                                        # inside [0.001, 0.007]
    assert any("does not beat its control" in p for p in c.problems(tmp_path))
    c = good(tmp_path)
    c.control.result = -0.001                                       # outside the interval: fine
    assert c.problems(tmp_path) == []


def test_comparison_sanity(tmp_path):
    assert any("reversed" in p for p in good(tmp_path, comparison=C.Comparison("a", "b", "s", 0.004, 0.007, 0.001, 10, 0.5)).problems(tmp_path))
    assert any("outside its own interval" in p for p in good(tmp_path, comparison=C.Comparison("a", "b", "s", 0.02, 0.001, 0.007, 10, 0.5)).problems(tmp_path))
    assert any("same thing" in p for p in good(tmp_path, comparison=C.Comparison("a", "a", "s", 0.004, 0.001, 0.007, 10, 0.5)).problems(tmp_path))


def test_gross_claim_is_rendered_as_gross(tmp_path):
    c = good(tmp_path, cost=C.CostAdjustment(applied=False, note="cost model not available for 1962-1990"))
    assert "GROSS of costs (cost model not available" in c.render(tmp_path)


def test_pattern_and_setting_claims_need_their_phase42_lists(tmp_path):
    c = good(tmp_path, kind="pattern")
    with pytest.raises(C.ClaimIncomplete) as ei:
        c.render(tmp_path)
    assert all(f"missing '{k}'" in str(ei.value) for k in C.PATTERN_EXTRAS)
    c.extras = {k: f"see {k}" for k in C.PATTERN_EXTRAS}
    assert "permutation evidence: see permutation_evidence" in c.render(tmp_path)
    s = good(tmp_path, kind="setting", extras={"baseline": "k=4"})
    assert len(s.problems(tmp_path)) == len(C.SETTING_EXTRAS) - 1


def test_claim_book_is_all_or_nothing_and_roundtrips(tmp_path):
    b = C.ClaimBook(tmp_path)
    b.add(good(tmp_path))
    bad = good(tmp_path, claim_id="C2")
    bad.definition = ""
    b.add(bad)
    with pytest.raises(ValueError, match="duplicate"):
        b.add(good(tmp_path))
    with pytest.raises(C.ClaimIncomplete, match="C2"):
        b.render_all()
    assert list(b.audit()) == ["C2"]
    b.claims["C2"].definition = "mean of the weekly returns of the second candidate over test windows"
    assert "[C1]" in b.render_all() and "[C2]" in b.render_all()
    path = tmp_path / "book.jsonl"
    b.save(path)
    assert C.ClaimBook.load(path, tmp_path).render_all() == b.render_all()


# ---------------------------------------------------------------- report scanner
def book_with(tmp_path, *values):
    b = C.ClaimBook(tmp_path)
    for i, v in enumerate(values):
        b.add(good(tmp_path, claim_id=f"K{i}", value=v, artefact_key=""))
    return b


def test_scanner_flags_a_planted_unclaimed_number_and_passes_the_claimed_ones(tmp_path):
    b = C.ClaimBook(tmp_path)
    b.add(good(tmp_path))          # value 0.0123 (1.23%), interval 0.1%..0.7%, n=180, p=0.01, control 0.02%, bps 10
    text = "Mean weekly return was 1.23% (n = 180 weeks).\nThe candidate also reached 8.75% in 2019.\n"
    res = C.scan_report(text, b)
    toks = [u["token"] for u in res["uncovered"]]
    assert "8.75%" in toks and "1.23%" not in toks
    assert not res["ok"] and res["covered"][0]["claim"] == "C1"
    assert "8.75%" in C.format_scan(res)


def test_scanner_is_clean_when_every_number_is_claimed_and_ignores_identifiers(tmp_path):
    b = C.ClaimBook(tmp_path)
    b.add(good(tmp_path))
    text = ("Run R12 on 2026-09-28 at 12:30 used seed 7 and commit abc1234def (v1.2.3).\n"
            "See Phase 42, item 3 in B21. Mean weekly return 1.23%; interval 0.10% to 0.70%; 180 weeks; p = 0.01.\n"
            "1. First\n2) Second\n")
    res = C.scan_report(text, b)
    assert res["ok"], C.format_scan(res)
    assert res["numbers"] >= 4


def test_scanner_matches_at_the_precision_printed_and_not_looser(tmp_path):
    b = book_with(tmp_path, 0.0123)
    assert C.scan_report("gain 1.2%", b)["ok"]                    # 1.23 rounds to 1.2
    assert C.scan_report("gain 1.23%", b)["ok"]
    assert not C.scan_report("gain 1.3%", b)["ok"]                # 1.3 is not a rounding of 1.23
    assert not C.scan_report("gain 12.3%", b)["ok"]               # a slipped decimal point is caught
    assert C.covers(0.5, "50%") and not C.covers(0.5, "5%") and not C.covers(float("nan"), "5%")


def test_scanner_empty_and_number_free_text_is_ok():
    assert C.scan_report("", [])["ok"] and C.scan_report("no figures here at all", [])["coverage"] == 1.0
    res = C.scan_report("we made 12.5% last year", [])
    assert not res["ok"] and res["coverage"] == 0.0


def test_scanner_allow_list_is_visible_not_silent(tmp_path):
    res = C.scan_report("the panel has 5,243 tickers", [], allow=("5,243",))
    assert res["ok"] and res["allowed"][0]["token"] == "5,243"


# ---------------------------------------------------------------- Phase 30 answers
def rec(**kw):
    base = {"experiment_id": "E1", "train_range": "2008-01-01..2015-12-31", "test_range": "2016-01-01..2019-12-31",
            "validation_range": {"start": "2016-01-01", "end": "2016-12-31"}, "metrics": {"mean_week": 0.006, "max_dd": -0.4}}
    base.update(kw)
    return base


def test_unseen_detects_planted_overlap_and_unknown_cases():
    ok = EM.unseen(rec(test_range="2017-01-01..2019-12-31", validation_range="2016-01-01..2016-12-31"))
    assert ok["clean"] and ok["held_out"] == ["test", "validation"]
    leak = EM.unseen(rec(test_range="2014-01-01..2019-12-31"))          # starts inside training
    assert leak["overlap"] == ["test"] and not leak["clean"]
    vague = EM.unseen(rec(test_range="holdout from 2023-12-22", validation_range=None))
    assert vague["unverifiable"] == ["test"] and not vague["clean"]
    none = EM.unseen({"experiment_id": "E"})
    assert none["known"] is False and "no train" in none["why"]


def test_parse_range_forms():
    assert EM.parse_range("2008-01-01..2015-12-31") == ("2008-01-01", "2015-12-31")
    assert EM.parse_range(["2008-01-01", "2009-01-01"]) == ("2008-01-01", "2009-01-01")
    assert EM.parse_range("holdout from 2023-12-22") == ("2023-12-22", None)
    assert EM.parse_range("rolling-origin folds") is None and EM.parse_range(None) is None


def test_worsened_uses_metric_polarity():
    base = {"metrics": {"mean_week": 0.005, "max_dd": -0.40, "turnover": 0.5, "win_weeks": 0.55, "mystery": 1.0}}
    cand = {"metrics": {"mean_week": 0.009, "max_dd": -0.55, "turnover": 0.3, "win_weeks": 0.55, "mystery": 0.5, "only_cand": 3}}
    r = EM.worsened(cand, base)
    assert [x["metric"] for x in r["improved"]] == ["mean_week", "turnover"]       # lower turnover is better
    assert [x["metric"] for x in r["worsened"]] == ["mystery", "max_dd"] or {x["metric"] for x in r["worsened"]} == {"max_dd", "mystery"}
    assert r["risk_worsened"] == ["max_dd"]                                        # the drawdown got deeper
    assert [x["metric"] for x in r["flat"]] == ["win_weeks"]
    assert r["assumed_direction"] == ["mystery"] and r["unshared"] == ["only_cand"]


def test_worsened_with_nothing_shared_is_unknown_not_clean():
    r = EM.worsened({"metrics": {"a": 1.0}}, {"metrics": {"b": 1.0}})
    assert r["known"] is False and r["worsened"] == []


def test_bootstrap_finds_a_planted_improvement_and_not_a_null_one():
    rng = np.random.default_rng(3)
    base = rng.normal(0.0, 0.03, 200)
    better = base + 0.01 + rng.normal(0, 0.002, 200)
    real = EM.meaningful(better, base, seed=1)
    assert real["meaningful"] and real["ci_low"] > 0 and abs(real["diff"] - 0.01) < 0.002
    null = EM.meaningful(base + rng.normal(0, 0.03, 200), base, seed=1)             # pure noise on top
    assert null["meaningful"] is False and null["ci_low"] < 0 < null["ci_high"]
    worse = EM.meaningful(base - 0.02, base, seed=1)
    assert worse["significantly_worse"] and not worse["meaningful"]
    assert EM.meaningful(base - 0.02, base, higher_is_better=False, seed=1)["meaningful"]       # lower is better


def test_bootstrap_is_deterministic_and_block_widens_interval_for_autocorrelated_data():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(0, 1, 300)) * 0.01 + 0.05          # strongly autocorrelated series
    a, b = EM.bootstrap_diff(x, seed=5), EM.bootstrap_diff(x, seed=5)
    assert a == b
    blk = EM.bootstrap_diff(x, seed=5, block=20)
    assert (blk["ci_high"] - blk["ci_low"]) > 1.5 * (a["ci_high"] - a["ci_low"])


def test_meaningful_degenerate_cases_are_unknown_not_no():
    assert EM.meaningful([], [])["meaningful"] is None
    assert EM.meaningful([0.1], [0.0])["meaningful"] is None
    few = EM.meaningful(np.full(10, 0.1) + np.arange(10) * 1e-3, np.zeros(10))
    assert few["meaningful"] is None and "only 10" in few["why"]
    assert EM.meaningful([np.nan, np.nan, 1.0], [0, 0, 0])["known"] is False
    with pytest.raises(ValueError):
        EM.bootstrap_diff([1, 2, 3], [1, 2])


def test_survival_confirms_a_real_effect_and_fails_one_found_by_luck():
    rng = np.random.default_rng(11)
    real = [{"id": f"w{i}", "diff": rng.normal(0.01, 0.02, 60)} for i in range(4)]
    assert EM.survived_another_window(real, seed=2)["verdict"] == "confirmed"
    # found in window 0 by selection: strong there, nothing anywhere else
    luck = [{"id": "w0", "diff": rng.normal(0.03, 0.02, 60)}] + [{"id": f"w{i}", "diff": rng.normal(0.0, 0.02, 60)} for i in (1, 2, 3)]
    r = EM.survived_another_window(luck, seed=2)
    assert r["verdict"] in ("failed", "direction_only") and not r["survived"]
    assert r["windows"][0]["discovery"] and r["windows"][0]["diff"] > 0.02
    flipped = [{"id": "w0", "diff": np.full(60, 0.02)}, {"id": "w1", "diff": rng.normal(-0.01, 0.01, 60)}]
    assert EM.survived_another_window(flipped, seed=2)["verdict"] == "failed"


def test_survival_unknown_without_another_window():
    assert EM.survived_another_window([])["verdict"] == "unknown"
    one = EM.survived_another_window([{"id": "w0", "diff": np.ones(50)}])
    assert one["known"] is False and one["verdict"] == "unknown"
    short = EM.survived_another_window([{"id": "w0", "diff": np.ones(50)}, {"id": "w1", "diff": np.ones(5)}])
    assert short["verdict"] == "unknown"


def test_answer_phase30_assembles_and_marks_missing_inputs_unknown():
    rng = np.random.default_rng(1)
    base_w = rng.normal(0, 0.03, 100)
    full = EM.answer_phase30(rec(), rec(metrics={"mean_week": 0.004, "max_dd": -0.3}),
                             series={"cand": base_w + 0.01, "base": base_w},
                             windows=[{"id": "a", "diff": np.full(30, 0.01) + rng.normal(0, 0.005, 30)}, {"id": "b", "diff": np.full(30, 0.01) + rng.normal(0, 0.005, 30)}])
    assert full["data_unseen"]["known"] and full["worsened"]["risk_worsened"] == ["max_dd"]
    assert full["statistically_meaningful"]["meaningful"] and full["survived_another_window"]["verdict"] == "confirmed"
    empty = EM.answer_phase30({"experiment_id": "E"})
    assert not empty["data_unseen"]["known"] and empty["worsened"]["known"] is False
    assert empty["statistically_meaningful"]["meaningful"] is None and empty["survived_another_window"]["verdict"] == "unknown"


def test_load_experiments_counts_torn_lines_and_coverage(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text(json.dumps(rec()) + "\n{torn\n\n" + json.dumps({"event": "x", "metrics": None}) + "\n[1,2]\n", encoding="utf-8")
    rows, bad = EM.load_experiments(p)
    assert len(rows) == 2 and bad == 2
    cov = EM.answer_coverage(rows)
    assert cov["n"] == 2 and cov["with_ranges"] == 1 and cov["with_metrics"] == 1 and cov["without_experiment_id"] == 1
    assert EM.load_experiments(tmp_path / "missing.jsonl") == ([], 0)
