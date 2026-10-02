"""Read-only report: (a) what the pre-screen would have rejected among the students' PAST choices (real lessons.jsonl, metric stage:
the change's AST-size / kernel-start-activation delta, no tests run), (b) the chooser's held-out accuracy on practice rows.

    python scripts/prescreen_report.py --lessons <repo>/state/creator/lessons.jsonl --tree <repo or snapshot> --practice <rows.jsonl>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from creator import chooser as CH            # noqa: E402
from creator import curriculum as CUR        # noqa: E402
from creator import prescreen as PS          # noqa: E402


def past_choices(lessons_path: Path, tree: Path) -> dict[str, object]:
    foot = PS.Footing(tree)
    out = []
    for les in CUR.LessonLog(lessons_path).lessons():
        if les.solver in CUR.TEACHER or not les.files_after or les.task_kind not in ("shrink", "activation"):
            continue
        metric = "activation" if les.task_kind == "activation" else "size"
        for rel, after in les.files_after.items():
            before = les.files_before.get(rel)
            if not rel.endswith(".py") or not before:
                continue
            foot.texts[rel], foot.sizes[rel] = before, max(PS.E.ast_size(before), 0)
            foot.eager[rel] = PS._eager_deps(before, foot.mod_of) or set()
            foot._base = foot._loaded(foot.eager, foot.sizes)
            v = PS.metric_verdict(foot, rel, after, metric)
            out.append({"solver": les.solver, "kind": les.task_kind, "path": rel, "size_delta": v.size_delta, "act_delta": v.act_delta,
                        "rejected": not v.ok, "adopted": les.adopted})
    rej = sum(1 for r in out if r["rejected"])
    return {"past_student_choices": len(out), "metric_rejected": rej, "rejected_fraction": rej / len(out) if out else None,
            "rows": out}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lessons", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--practice", default="")
    a = ap.parse_args(argv)
    rep: dict[str, object] = {"past": past_choices(Path(a.lessons), Path(a.tree))}
    if a.practice:
        p = Path(a.practice)
        rows = CH.practice_rows(p, p.parent / "practice_src")
        rep["practice_rows"] = {"decisions": len(rows), "positive": sum(r.sign > 0 for r in rows)}
        rep["practice_holdout"] = CH.practice_holdout(rows)
    print(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
