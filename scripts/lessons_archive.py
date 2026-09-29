"""Phase 11 on the REAL Test archive (state/livesim windows), read-only, through adaptive.replay.

Every archived window is replayed four ways (see engine.antimemo.archive_experiment): no lessons, lessons learned from
OTHER windows only (leave-one-fold-out), lessons learned from the SAME window, and those same-window lessons replayed
on the window with tickers renamed and dates shifted. Memorisation = the same-window arm beating both the other-windows
arm and the disguised rerun. A memorising RecallControl runs through the identical harness as a positive control.

usage: lessons_archive.py [--adaptive] [--control] [--folds 6] [--seed 0] [--limit N] [--tag name]
Lessons are applied by wrapping snapshot scores before replay; engine/adaptive.py and livesim_loop2.py are only imported.
Writes state/research/lessons/archive_<tag>.{json,md} (provenance-stamped) and one new version in state/lessons/."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from engine import config as K
from engine.antimemo import archive_experiment, archive_markdown, default_archive_learner, recall_learner
from engine.lessons import LessonBook, LessonStore
from engine.provenance import stamp
import livesim_loop2 as L


def window_dirs(root):
    return sorted(d for d in Path(root).iterdir()
                  if d.is_dir() and not d.name.startswith("_") and any(d.glob("wsnap_*.parquet")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adaptive", action="store_true")
    ap.add_argument("--control", action="store_true", help="also run the memorising positive control")
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="v1")
    a = ap.parse_args()
    t0 = time.time()
    st = json.loads((L.DIR / "loop2.json").read_text())                 # the current training basis, read-only
    cfg, meta = st["cfg"], st["meta"]
    dirs = window_dirs(L.DIR)[: a.limit or None]
    wins = [L.load_window(d) for d in dirs]
    say = lambda m: print(f"{time.time() - t0:6.0f}s {m}", flush=True)
    say(f"{len(wins)} windows, adaptive={a.adaptive}, basis v{st.get('version')}")
    params = {"min_support": 30}
    learner = default_archive_learner(params, a.seed)
    out = {"stamp": stamp({"cfg": cfg, "meta": meta, "adaptive": a.adaptive, "folds": a.folds, "params": params}, a.seed),
           "windows": [w["id"] for w in wins]}
    out["lessons"] = archive_experiment(wins, cfg, meta, a.adaptive, learner, folds=a.folds, seed=a.seed, progress=say)
    if a.control:
        say("positive control: RecallControl (memorises winning rows)")
        out["control"] = archive_experiment(wins, cfg, meta, a.adaptive, recall_learner, folds=a.folds, seed=a.seed,
                                            progress=say, parity_windows=0)
    dest = K.STATE / "research" / "lessons"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"archive_{a.tag}.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    md = [archive_markdown(out["lessons"], "Lessons on the Test archive")]
    if "control" in out:
        md += ["", archive_markdown(out["control"], "POSITIVE CONTROL: a memoriser through the same harness")]
    (dest / f"archive_{a.tag}.md").write_text("\n".join(md), encoding="utf-8")
    # one versioned book learned on every window, for the record (episodes are not kept: they are recomputable)
    from engine.antimemo import window_panel, archive_frame, _arm
    from engine.lessons import post_mortem
    items = []
    for i, w in enumerate(wins):
        X, Y = window_panel(w)
        base = _arm(w, X, None, cfg, meta, a.adaptive)
        fr = archive_frame(X, Y, base.decisions)
        items.append({"episodes": post_mortem(fr, X, fr["resolved"].max(), {"untaken_frac": 0.03}, a.seed + i), "frame": fr})
    book = learner(items)
    v = LessonStore().save(book, note=f"archive_{a.tag}: all {len(wins)} windows, adaptive={a.adaptive}", episodes=False,
                           stamp=out["stamp"])
    say(f"saved lesson book version {v} ({len(book.lessons)} lessons)")
    m = out["lessons"]["memorisation"]
    print(json.dumps({"gain_a": out["lessons"]["weekly_gain_a"]["mean"], "gain_b": out["lessons"]["weekly_gain_b"]["mean"],
                      "gain_c": out["lessons"]["weekly_gain_c"]["mean"], "b_minus_c": m["b_minus_c_mean"],
                      "memorised": m["windows_memorised"], "with_lessons": m["windows_with_lessons"],
                      "control_b_minus_c": out.get("control", {}).get("memorisation", {}).get("b_minus_c_mean"),
                      "seconds": round(time.time() - t0)}), flush=True)


if __name__ == "__main__":
    main()
