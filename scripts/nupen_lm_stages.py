"""Run the five language-stage tests against Nupen's current model; print the scorecard and save it to lmckpt/stages.json.

    python scripts/nupen_lm_stages.py [--generated 20] [--seed-file]   (run through scripts/lowprio.py)
The teacher grades stage 5 by setting "stage5_grade" (0..1) in stages.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from creator.lm import dialogue as D  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", type=int, default=20, help="templated held-out items per stage (stages 3 and 4) added to the teacher's")
    ap.add_argument("--seed-file", action="store_true", help="create teacher_dialogues.jsonl with the 60 seed lessons if absent")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args(argv)
    if args.seed_file:
        print("seed lessons written:", D.write_seed_file())
    try:
        from creator.lm import api
        lm = api.load_current()
    except ImportError:
        lm = None
    items = D.load_items(generated=args.generated)
    card = D.run_stages(lm, items, prior=D.load_state())
    if not args.no_save:
        D.save_state(card)
    print(json.dumps(card, indent=1))
    for k, s in card["stages"].items():
        print(f"stage {k} {s['name']}: {s.get('score', '-')} (need {'<=' if s['lower_better'] else '>='} {s['threshold']}) -> {'PASS' if s['passed'] else s.get('status', 'not passed')}")
    print("training data unlocked up to stage", card["unlocked_stage"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
