"""Bible Phase 29: the public explanation site. Each mechanism has a planted defect the check must catch:
sealed-tree reads, secrets, disguised dates, unrevealed windows, stale pages, external scripts, formatter leaks,
multiple-testing honesty, negative results shown, empty inputs. Synthetic data only; no network; runs in seconds."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts import site_audit as A, site_build as B, site_publish as PUB, site_safety as S, site_sources as SRC
from scripts import site_charts as C, site_render as R

AT = "2026-01-01 00:00 UTC"


# ---------------------------------------------------------------- synthetic state tree
def _patterns_frame(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    def row(name, status, p, effect, t, scope=None, dup=None):
        rows.append({"m_disc": effect, "t_disc": t, "m_conf": effect * .8, "t_conf": t * .7, "m_all": effect, "t_all": t, "n_eff": 500.0,
                     "m_recent": effect * .5, "t_recent": t * .4, "n_recent": 90.0, "key": "('p',1,4,2,0)", "key_named": name,
                     "p_coincidence": .01, "fdr_pass": True, "p_hallucinated": .02, "p_real": p, "confirmed": p >= .8,
                     "effect": effect, "status": status, "scope": scope, "duplicate_of": dup})
    row("ear q4 & max20 q0", "active", .97, .004, 4.1)
    row("frog q4", "rescoped", .91, .002, 3.0, scope="(0, 'high', 12.5, 20.1)")
    row("mom_12_1 q0", "discarded", .88, .003, 3.3)
    for i in range(30):
        row(f"skew60 q{i % 5} & dist_52wh q{(i + 1) % 5}", "rejected", float(rng.uniform(0, .5)), float(rng.normal(0, .002)), float(rng.normal(0, 2)))
    row("nan case q1", "rejected", float("nan"), float("nan"), float("nan"))
    return pd.DataFrame(rows)


def make_tree(root, cycles=None, registry=None, sens=None, checklist=None, patterns=True):
    root = Path(root)
    algo = root / "state" / "research" / "algorithm"
    algo.mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "state" / "livesim").mkdir()
    (root / "state" / "reports").mkdir()
    if patterns:
        _patterns_frame().to_parquet(algo / "patterns_2020-01-01_v2.parquet")
        (algo / "test_2020-01-01_v2.json").write_text(json.dumps({"ic_mean": .01, "ic_t": 1.2, "weeks": 52, "tested": 34, "fdr_pass": 20,
                                                                  "null_t_95pct": 1.9, "real_t_95pct": 3.5}))
        (algo / "planted_calibration.json").write_text(json.dumps({"seeds": 2, "recovery": {"strong +1.2%": 1.0, "regime-only +1.0%": .5},
                                                                    "false_active_per_run": .1, "active_per_run": 5.0, "false_share_of_active": .02,
                                                                    "calibration": {"(-0.01, 0.2]": {"share_truly_real": .2, "n": 100}, "(0.8, 0.9]": {"share_truly_real": .5, "n": 40},
                                                                                    "(0.9, 1.0]": {"share_truly_real": .96, "n": 300}}}))
    (root / "state" / "livesim" / "cycles.json").write_text(json.dumps({"cycles": cycles if cycles is not None else [], "rounds": []}))
    (root / "state" / "experiments.jsonl").write_text("\n".join(json.dumps(r) for r in (registry or [])) + ("\n" if registry else ""))
    (root / "state" / "CHECKLIST.md").write_text(checklist if checklist is not None else CHECKLIST)
    if sens is not None:
        (root / "state" / "research" / "sensitivity.json").write_text(json.dumps(sens))
    return B.default_paths(root)


CHECKLIST = """# Weekly7 master checklist
## Phase 0: control
- [x] P0.1 Canon integrity. Evidence: `verify` passes.
- [~] P0.2 Registry with provenance (built; writers pending)
- [ ] P0.3 Checkpoint bundle
## Test
- [?] T9 Re-tester parity (cause found; unproven)
- [!] T99 Broken thing
this line is not an item
- [ nonsense
"""


def make_sens(n_years=20, seed=1):
    rng = np.random.default_rng(seed)
    def var(group, knob, value, mean, sd=0.002):
        py = rng.normal(mean, sd, n_years)
        t = float(py.mean() / (py.std(ddof=1) / np.sqrt(n_years)))
        return {"group": group, "knob": knob, "value": value, "d_mean_week": float(py.mean()), "t": t,
                "share_years_same_way": float((np.sign(py) == np.sign(py.mean())).mean()), "d_weeks_ge7_per_year": .1, "d_worst_dd": -.01,
                "size": "significant", "consistency": "consistent", "per_year": [float(x) for x in py]}
    vs = [var("knob", "k", 2, -0.003), var("knob", "k", 12, 0.0), var("knob", "brake", .05, 0.01, .001), var("knob", "brake", .12, 0.0002),
          var("knob", "trend_filter", None, -0.001)]
    vs += [var("knob", "pad", i, rng.normal(0, .0005)) for i in range(40)]           # many tries, so the correction bites
    return {"hidden_years": n_years, "champion": {"k": 8, "brake": .08, "trend_filter": None, "ew": {"ear": 1.0, "max20": -.8}},
            "champion_mean_week": .003, "variants": vs, "by_knob": [{"group": "knob", "knob": "k", "range_of_effect": .003, "class": "significant"}]}


@pytest.fixture
def tree(tmp_path):
    return make_tree(tmp_path, sens=make_sens(), registry=[
        {"t": "2026-01-01T00:00:00", "event": "frontier", "rows": [1, 2], "experiment_id": "E1", "git_commit": "abc123", "seed": 3},
        {"t": "2026-01-02T00:00:00", "event": "rolled_back", "id": "v1.3", "reason": "full history worse than SPY"},
        {"t": "2026-01-03T00:00:00", "event": "voided_adjustment", "reason": "re-tester mismatch"}])


# ---------------------------------------------------------------- safety
def test_sealed_tree_is_unreachable_except_revealed_cycle_list():
    for name in ("sealed_m01.json", "r01a/result.json", "r05c/snap_2190-06-14.parquet", "memory_bank.parquet"):
        with pytest.raises(S.SealedAccess):
            S.guard_path(S.SEALED_DIR / name)
    with pytest.raises(S.SealedAccess):
        S.guard_path(S.ROOT / "data" / "cache" / "prices.parquet")
    assert S.guard_path(S.SEALED_DIR / "cycles.json").name == "cycles.json"
    with pytest.raises(S.SealedAccess):
        S.safe_read_text(S.SEALED_DIR / "sealed_m02.json")


def test_private_looking_files_are_refused(tmp_path):
    f = tmp_path / ".env"
    f.write_text("APCA_API_SECRET_KEY=abcdefghijklmnop")
    with pytest.raises(S.SealedAccess):
        S.safe_read_text(f)


def test_audit_catches_planted_secrets_and_sealed_marks_but_passes_clean_text():
    clean = "Portfolio value $2,150 after 52 weeks; 2190 shares; year 2016; S&P 500; week ending 2019-03-01."
    assert S.audit_text("ok", clean)
    for bad in ("key PKABCDEFGHIJKLMNOPQR here", "token ghp_" + "a" * 36, "-----BEGIN RSA PRIVATE KEY-----",
                "api_key = 'abcdefghijklmnop1234'", "window 2190-06-14 loss", "held S0174 for a week", "in year 2191 the model"):
        with pytest.raises(S.Leak):
            S.audit_text("bad", bad)


def test_unrevealed_windows_and_position_detail_never_pass_public_cycles():
    cyc = [{"run_id": "r01a", "revealed_year": "1987", "config": {"k": 4}, "config_version": 1,
            "diagnosis": {"mean_week": .001, "worst_positions": [{"code": "S0174", "week_end": "2190-06-18"}], "weeks": 52}},
           {"run_id": "r99z", "config": {"k": 4}, "diagnosis": {"mean_week": .5}}]
    pub = S.public_cycles(cyc)
    assert [c["run_id"] for c in pub] == ["r01a"]
    assert "worst_positions" not in pub[0] and pub[0]["weeks"] == 52
    assert "S0174" not in json.dumps(pub)


def test_scrub_removes_secret_keys_and_values():
    out = S.scrub({"api_key": "x", "note": "use PKABCDEFGHIJKLMNOPQR now", "nested": [{"password": "p", "ok": 1}]})
    assert "api_key" not in out and "[removed]" in out["note"] and out["nested"] == [{"ok": 1}]


# ---------------------------------------------------------------- full build
def test_build_writes_four_pages_manifest_and_passes_structural_audit(tree, tmp_path):
    m = B.build(tree, built_at=AT)
    for n in B.PAGES:
        assert (tree["out"] / n).exists()
    assert set(m["pages"]) == set(B.PAGES)
    assert A.audit_site(tree["out"]) == [] or all(x["severity"] != "error" for x in A.audit_site(tree["out"]))
    mf = json.loads((tmp_path / "state" / "research" / "site" / "manifest.json").read_text())
    assert mf["pages"]["explorer.html"]["sha256"] == m["pages"]["explorer.html"]["sha256"]


def test_explorer_never_hides_failed_patterns_and_marks_nan_as_dash(tree):
    B.build(tree, built_at=AT)
    t = (tree["out"] / "explorer.html").read_text(encoding="utf-8")
    assert 'data-status="discarded"' in t and 'data-status="rejected"' in t and 'data-status="active"' in t and 'data-status="rescoped"' in t
    assert "only when fear index (VIX) is above 20.1" in t                       # scope decoded
    assert "ear in the top fifth and max20 in the bottom fifth" in t.replace("&amp;", "&") or "top fifth" in t
    assert ">nan<" not in t and "nan%" not in t


def test_rejected_are_capped_but_counted_in_full(tmp_path):
    tr = make_tree(tmp_path)
    runs = SRC.load_pattern_runs(tr["algo"], caps={"rejected": 5})
    r = runs[0]
    assert r["counts"]["rejected"] == 31 and sum(1 for p in r["patterns"] if p["status"] == "rejected") == 5
    assert r["tested"] == 34 and r["listed"] == 3 + 5 - 0


def test_build_refuses_and_writes_nothing_when_a_sealed_date_reaches_a_page(tmp_path):
    tr = make_tree(tmp_path, sens=make_sens(), registry=[{"t": "x", "event": "note", "text": "worst week ended 2190-06-18"}])
    with pytest.raises(S.Leak):
        B.build(tr, built_at=AT)
    assert not any((tr["out"] / n).exists() for n in B.PAGES)


def test_secret_in_registry_is_scrubbed_not_published(tmp_path):
    tr = make_tree(tmp_path, sens=make_sens(), registry=[{"t": "x", "event": "note", "text": "paper key PKABCDEFGHIJKLMNOPQR", "api_key": "zzz"}])
    B.build(tr, built_at=AT)
    t = (tr["out"] / "runs.html").read_text(encoding="utf-8")
    assert "PKABCDEFGHIJKLMNOPQR" not in t and "zzz" not in t and "[removed]" in t


def test_unrevealed_cycle_content_absent_from_runs_page(tmp_path):
    cyc = [{"run_id": "r01a", "revealed_year": "1987", "config": {}, "diagnosis": {"mean_week": .002, "weeks_ge_7": 2, "year_return": .1, "market_year_return": .05}},
           {"run_id": "r77x", "config": {}, "diagnosis": {"mean_week": .0424242, "year_return": 9.99}}]
    tr = make_tree(tmp_path, cycles=cyc, sens=make_sens())
    B.build(tr, built_at=AT)
    t = (tr["out"] / "runs.html").read_text(encoding="utf-8")
    assert "r01a" in t and "r77x" not in t and "9.99" not in t and "1 still sealed" in t


def test_build_is_deterministic(tree):
    a = B.render_all(B.load_all(tree), AT)[0]
    b = B.render_all(B.load_all(tree), AT)[0]
    assert a == b


def test_empty_state_builds_with_visible_problems_not_crash(tmp_path):
    tr = make_tree(tmp_path, patterns=False, checklist="")
    m = B.build(tr, built_at=AT)
    t = (tr["out"] / "explorer.html").read_text(encoding="utf-8")
    assert "No pattern is currently live" in t and "no miner output" in t
    assert "Nothing recorded yet." in (tr["out"] / "runs.html").read_text(encoding="utf-8")
    assert m["problems"]["sens"] and m["problems"]["check"]
    # no data means the content promises (e.g. "a rejected pattern is listed") cannot hold, but the page itself must stay well-formed
    assert all(x["check"] == "promise" for x in A.audit_site(tr["out"]) if x["severity"] == "error")


def test_stale_detection_and_edited_page_detection(tree):
    assert B.check_fresh(tree)[0] is False                                         # never built
    B.build(tree, built_at=AT)
    assert B.check_fresh(tree) == (True, [])
    with open(tree["checklist"], "a") as f:
        f.write("- [x] X1 New item. Evidence: yes\n")
    fresh, why = B.check_fresh(tree)
    assert not fresh and any("CHECKLIST" in w for w in why)
    B.build(tree, built_at=AT)
    (tree["out"] / "runs.html").write_text("tampered")
    assert any("edited after the build" in w for w in B.check_fresh(tree)[1])


def test_verify_flags_a_leak_in_a_published_page(tree):
    B.build(tree, built_at=AT)
    assert B.verify(tree)[0] == []
    p = tree["out"] / "checklist.html"
    p.write_text(p.read_text(encoding="utf-8") + " <p>held S0174</p>", encoding="utf-8")
    assert B.verify(tree)[0]


# ---------------------------------------------------------------- sources
def test_checklist_parse_counts_states_and_reports_unparsed_lines():
    ck = SRC.parse_checklist(CHECKLIST)
    assert ck["counts"] == {"x": 1, "~": 1, " ": 1, "?": 1, "!": 1} and ck["total"] == 5
    assert ck["unparsed"] == 1
    assert ck["phases"][0]["items"][0]["evidence"].startswith("`verify`")
    un = SRC.build_unproven(ck, [{"event": "rolled_back", "reason": "worse", "id": "v1.3", "t": "t"}])
    assert [i["id"] for i in un["items"]][:2] == ["T99", "T9"] and un["negatives"][0]["id"] == "v1.3"


def test_sensitivity_multiple_testing_bar_and_negative_results():
    s = SRC.build_sensitivity(make_sens())
    assert s["n_variants"] == 45 and 3.0 < s["bonferroni_t"] < 4.0
    brake = next(k for k in s["knobs"] if k["knob"] == "brake")
    strong = next(v for v in brake["values"] if v["value"] == 0.05)
    assert strong["beats_multiple_testing"] and "beats the champion after multiple-testing" in brake["reason"]
    k = next(k for k in s["knobs"] if k["knob"] == "k")
    assert k["selected"] == 8 and k["negative"] >= 1 and next(v for v in k["values"] if v["value"] == 2)["effect"] < 0
    assert next(k for k in s["knobs"] if k["knob"] == "trend_filter")["selected"] == "off"
    lo, hi = strong["lo"], strong["hi"]
    assert lo < strong["effect"] < hi


def test_sensitivity_missing_input_reports_a_problem():
    probs = []
    s = SRC.build_sensitivity(None, probs)
    assert s["knobs"] == [] and probs


def test_sensitivity_page_lists_worse_alternatives(tree):
    B.build(tree, built_at=AT)
    t = (tree["out"] / "sensitivity2.html").read_text(encoding="utf-8")
    assert "Negative results (never hidden)" in t and "k = 2" in t and "survives correction" in t


def test_scope_and_expression_readers():
    txt, obj = SRC.parse_scope("(2, 'low', 0.98, 1.02)")
    assert "below 0.98" in txt and obj["context"] == "m_spy_ma200"
    assert SRC.parse_scope(None) == (None, None) and SRC.parse_scope(float("nan")) == (None, None)
    assert SRC.parse_scope("garbage(")[0] == "garbage("
    assert SRC.plain_expr("a q4 & b q0 unless c q2") == "a in the top fifth and b in the bottom fifth unless c in the middle fifth"


def test_registry_reader_counts_bad_lines_and_truncates_detail(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text('{"event":"a","t":"1"}\nnot json\n{"event":"b","t":"2","blob":"' + "x" * 6000 + '"}\n')
    probs = []
    recs = SRC.load_registry(p, probs)
    assert len(recs) == 2 and any("1 line" in x for x in probs)
    v = SRC.registry_view(recs[1])
    assert len(v["detail"]) < 2700 and "truncated" in v["detail"]
    assert SRC.provenance_coverage(recs)["seed"] == 0.0
    assert SRC.load_registry(tmp_path / "missing.jsonl", probs) == []


def test_bank_history_is_shown_when_a_bank_exists(tmp_path):
    from engine.pattern_bank import PatternBank
    bank = PatternBank(tmp_path / "bank")
    rec = {"id": "abc", "names": ["ear", "max20"], "name": "ear q4 & max20 q0", "state": "discarded", "scope": None, "effect": .002,
           "first_seen": "2018-01-01", "changed": "2020-01-01", "discovery": {"p_real": .93, "t_conf": 2.5, "n_eff": 400}, "discard_reason": "no context held",
           "windows": [{"window_end": "2019-01-01", "t": 2.0, "m": .002, "n": 300, "verdict": "hold", "source": "review"}],
           "last_validation": {"m_recent": -.001, "t_recent": -1.7, "n_recent": 40},
           "history": [{"as_of": "2018-01-01", "kind": "transition", "frm": None, "to": "active", "reason": "found"},
                       {"as_of": "2020-01-01", "kind": "transition", "frm": "active", "to": "discarded", "reason": "recent stretch contradicts"}]}
    bank._write({"schema": 1, "version": 1, "parent": None, "records": {"abc": rec}})
    b = SRC.load_bank(tmp_path / "bank")
    assert b["available"] and b["patterns"][0]["history"][-1]["to"] == "discarded" and b["verify"]["ok"]
    assert SRC.load_bank(tmp_path / "nobank")["available"] is False
    assert not (tmp_path / "nobank").exists()                                       # reading never creates state


# ---------------------------------------------------------------- structural audit catches planted defects
GOOD = ('<!doctype html><html><head><meta name="viewport" content="width=device-width, initial-scale=1"><title>t</title></head>'
        '<body><h1>x</h1><p>ok</p></body></html>')


def test_structural_audit_catches_each_planted_defect(tmp_path):
    def kinds(html, name="x.html"):
        return {f["check"] for f in A.audit_page(name, html, tmp_path)}
    assert kinds(GOOD) == set()
    assert "viewport" in kinds(GOOD.replace("width=device-width", "width=500"))
    assert "h1" in kinds(GOOD.replace("<h1>x</h1>", "<h1>x</h1><h1>y</h1>"))
    assert "external script" in kinds(GOOD.replace("<p>", '<script src="https://evil.example/x.js"></script><p>'))
    assert "external script" not in kinds(GOOD.replace("<p>", '<script src="https://cdnjs.cloudflare.com/x.js"></script><p>'))
    assert "formatter leak" in kinds(GOOD.replace("<p>ok</p>", "<td>nan</td>"))
    assert "duplicate ids" in kinds(GOOD.replace("<p>", '<i id="a"></i><i id="a"></i><p>'))
    assert "dangling filter" in kinds(GOOD.replace("<p>", '<div data-table="nope"></div><p>'))
    assert "broken link" in kinds(GOOD.replace("<p>", '<a href="missing.html">m</a><p>'))
    assert "size" in kinds(GOOD + "x" * (A.MAX_BYTES + 1))
    assert {"promise"} <= kinds(GOOD, "explorer.html")


def test_missing_page_is_an_error(tmp_path):
    assert A.audit_site(tmp_path, ["runs.html"])[0]["check"] == "missing"


# ---------------------------------------------------------------- render and charts
def test_formatters_never_print_nan_or_none():
    for f in (R.fmt_num, R.fmt_pct, R.fmt_int, R.fmt_t):
        assert f(None) == "—" and f(float("nan")) == "—" and f(float("inf")) == "—"
    assert R.esc("<script>") == "&lt;script&gt;" and R.esc(None) == ""
    assert "&lt;b&gt;" in R.table([("h", True)], [[R.cell(R.esc("<b>"))]], "t")


def test_charts_degrade_on_empty_input():
    assert "n/a" in C.sparkline([]) and "n/a" in C.sign_bars([]) and "nothing recorded" in C.stacked_bar([("a", 0, "red")])
    assert "No tested values" in C.interval_plot([{"label": "a", "est": float("nan")}]) and "No calibration" in C.calibration_dots([])
    svg = C.interval_plot([{"label": "a", "est": .1, "lo": .05, "hi": .2, "kind": "selected"}, {"label": "b", "est": -.1}])
    assert svg.count("<circle") == 2 and svg.count("<line") == 2                   # zero line + one interval; the bare dot has no interval
    assert "<svg" in C.sparkline([1, 2, 0, 3]) and C.sparkline([5, 5, 5]).startswith("<svg")


def test_era_breakdown_groups_by_decade():
    cs = [{"year": "1987", "mean_week": .01, "year_return": .2, "market_year_return": .1, "weeks_ge_7": 2, "max_dd": -.1},
          {"year": "Mar 1988 - Feb 1989", "mean_week": -.01, "year_return": -.2, "market_year_return": .1, "weeks_ge_7": 0, "max_dd": -.3},
          {"year": "2016", "mean_week": .0, "year_return": .1, "market_year_return": .0, "weeks_ge_7": 1, "max_dd": -.05}]
    e = {x["era"]: x for x in SRC.era_breakdown(cs)}
    assert e["1980s"]["windows"] == 2 and e["1980s"]["beat_market"] == .5 and e["1980s"]["worst_dd"] == -.3 and e["2010s"]["windows"] == 1


# ---------------------------------------------------------------- publisher
def test_publisher_refuses_on_leak_and_does_not_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(PUB, "LOG", tmp_path / "publish.log")
    tr = make_tree(tmp_path / "t", sens=make_sens(), registry=[{"t": "x", "event": "n", "text": "2190-06-18"}])
    msg = PUB.cycle(tr, commit=False)
    assert "REFUSED" in msg and (tmp_path / "publish.log").read_text().count("REFUSED") == 1


def test_publisher_builds_audits_then_reports_fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(PUB, "LOG", tmp_path / "publish.log")
    tr = make_tree(tmp_path / "t", sens=make_sens())
    assert "built and audited clean" in PUB.cycle(tr, commit=False)
    assert PUB.cycle(tr, commit=False).startswith("fresh")
