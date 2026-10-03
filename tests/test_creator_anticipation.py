"""F7: the anticipation rate (top-level trust metric): honest matching, echo removal, backlog, drill, and its place in trust.json / BLUEPRINT.md."""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

from creator import anticipation as A
from creator import thinking as T


def owner(tmp: Path) -> Path:
    d = tmp / "owner"
    d.mkdir()
    (d / "OWNER_MESSAGES.md").write_text(
        "# OWNER MESSAGES\n\n### 2026-10-01 10:00 UTC (session x)\n\n> keep a roster of every capability and measure the calibration of each prediction\n\n"
        "### 2026-10-01 12:00 UTC (session x)\n\n> build a quantum entanglement dashboard for the greenhouse sensors immediately\n", encoding="utf-8")
    (d / "JOURNAL.md").write_text("## Journal\n\n## 1 Oct 2026, 16:00-17:45 agents\n- 17:05 OWNER: the system must forecast limiting bottlenecks automatically\n"
                                  "- 23:20 OWNER: rotate hibernation snapshots weekly\n- 07:52 OWNER: later directive next morning\n", encoding="utf-8")
    return d


def test_directive_parsing_and_dates(tmp_path: Path) -> None:
    ds = A.directives(owner(tmp_path))
    assert len(ds) == 5
    j = [d for d in ds if d.source == "journal"]
    assert dt.datetime.fromtimestamp(j[0].t, dt.timezone.utc).isoformat().startswith("2026-10-01T23:05")          # 17:05 MDT = 23:05 UTC
    assert dt.datetime.fromtimestamp(j[2].t, dt.timezone.utc).isoformat().startswith("2026-10-02T13:52")          # rolled over to the next day (MDT)


def test_a_prior_artefact_naming_the_change_counts_and_is_stored_for_audit() -> None:
    d = A.Doc(2000.0, "the system must forecast limiting bottlenecks automatically and rank every bottleneck", "journal", "d1")
    good = A.Doc(1000.0, "forecast limiting bottleneck ranking: automatically rank each bottleneck", "blueprint", "b1")
    late = A.Doc(3000.0, "forecast limiting bottleneck automatically rank", "blueprint", "b2")
    r = A.anticipate([d], [good, late])
    assert r["anticipated"] == 1 and r["rate"] == 1.0
    p = r["pairs"][0]["matched"]
    assert p["artefact"] == "blueprint:b1" and p["lead_s"] == 1000 and len(p["shared_words"]) >= A.MIN_SHARED
    assert A.anticipate([d], [late])["anticipated"] == 0                      # an artefact AFTER the directive anticipates nothing


def test_echo_of_the_owner_is_not_anticipation() -> None:
    said = A.Doc(500.0, "forecast limiting bottleneck automatically rank", "owner_messages", "o1")
    d = A.Doc(2000.0, "forecast limiting bottleneck automatically rank", "journal", "d1")
    art = A.Doc(1000.0, "Focus: forecast limiting bottleneck automatically rank", "blueprint", "b1")   # merely repeats what the owner said at 500
    r = A.anticipate([said, d], [art])
    assert r["anticipated"] == 0 and len(r["backlog"]) == 2


def test_unrelated_overlap_and_thin_overlap_do_not_match() -> None:
    d = A.Doc(2000.0, "rotate hibernation snapshots weekly across storage volumes", "journal", "d")
    thin = A.Doc(1000.0, "weekly report of storage", "plan", "p")
    assert A.anticipate([d], [thin])["anticipated"] == 0


def test_backlog_lists_nearest_earlier_signals_and_eligibility() -> None:
    d0 = A.Doc(10.0, "something about weather", "journal", "pre")
    art = A.Doc(100.0, "sensors dashboards calibrated", "goal_proposal", "G1")
    d1 = A.Doc(200.0, "make greenhouse sensors dashboards nicer please", "journal", "post")
    assert A.tokens(art.text) & A.tokens(d1.text)
    r = A.anticipate([d0, d1], [art])
    assert r["directives"] == 2 and r["eligible_directives"] == 1
    post = [b for b in r["backlog"] if b["directive"] == "post"]
    assert post and post[0]["eligible"] and post[0]["closest_earlier_signals"][0]["artefact"] == "goal_proposal:G1"


def test_drill_scores_against_the_frequency_baseline() -> None:
    ds = [A.Doc(float(i), f"calibration roster capability measure topic{i % 3} unique{i}", "journal", f"d{i}") for i in range(30)]
    r = A.drill(ds, [])
    assert r["n"] == 27 and r["precision_at_10_model"] >= r["precision_at_10_baseline"] - 0.2 and len(r["gain_ci95"]) == 2


def test_snapshots_are_dated_and_become_artefacts(tmp_path: Path) -> None:
    st = tmp_path / "st"
    p = A.snapshot_blueprint(st, "# bp\nforecast bottleneck", now=1_790_000_000.0)
    assert p.name.startswith("BLUEPRINT_2026") and p.exists()
    arts = A.nupen_artefacts(st)
    assert [a.source for a in arts] == ["blueprint"] and abs(arts[0].t - 1_790_000_000.0) < 1


def test_trust_json_carries_the_rate_first_and_the_report_is_audited(tmp_path: Path) -> None:
    st = tmp_path / "st"
    st.mkdir()
    od = owner(tmp_path)
    rep = T.trust(st, write=True, owner_dir=od)
    raw = (st / "trust.json").read_text(encoding="utf-8")
    assert next(iter(json.loads(raw))) == "anticipation" and raw.index('"anticipation"') < raw.index('"topics"')
    assert rep["anticipation"]["directives"] == 5 and rep["anticipation"]["rate"] == 0.0
    full = json.loads((st / "thinking" / "anticipation.json").read_text(encoding="utf-8"))
    assert len(full["backlog"]) == 5


def test_blueprint_leads_with_the_anticipation_rate_and_snapshots_itself(tmp_path: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import nupen_blueprint as B
    st = tmp_path / "st"
    st.mkdir()
    od = owner(tmp_path)
    text = B.build(st, od)
    assert text.index("## 0. Anticipation rate") < text.index("## 1. Vision") and "Anticipation rate: 0 of 5 directives" in text
    out = tmp_path / "bp.md"
    assert B.main(["--state", str(st), "--out", str(out), "--owner-dir", str(od)]) == 0
    assert list((st / "blueprints").glob("BLUEPRINT_*.md"))
