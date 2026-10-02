"""Talk to Nupen in the terminal (layer 1: rule-based; no language model needed).

    python scripts/nupen.py                       # interactive; 'quit' to leave
    python scripts/nupen.py --once "how are you doing"
    python scripts/nupen.py --root <repo copy>    # read another checkout's state (read-only experiments); pause/resume/approve write there
    python scripts/nupen.py --once "pause" --yes  # scripted confirmation of an action
    python scripts/nupen.py --export out.jsonl    # logged exchanges in the creator.lm.dialogue format (source nupen-terminal, not reviewed)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from creator import conversation as CV  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", help="send one message, print the answer, exit")
    ap.add_argument("--root", default=str(ROOT), help="repo root whose state/ is read (default: this checkout)")
    ap.add_argument("--yes", action="store_true", help="auto-confirm pause/resume/approve (scripting)")
    ap.add_argument("--no-log", action="store_true", help="do not append to conversations.jsonl")
    ap.add_argument("--export", help="write the logged exchanges as dialogue JSONL to this path and exit")
    a = ap.parse_args(argv)
    root = Path(a.root)
    if a.export:
        n = CV.export_dialogues(root / "state" / "creator", Path(a.export))
        print(f"exported {n} exchanges to {a.export}")
        return 0
    conv = CV.Conversation(root, auto_confirm=a.yes, log=not a.no_log)
    if a.once:
        print(conv.reply(a.once))
        return 0
    print(CV.banner())
    while True:
        try:
            line = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line.lower() in ("quit", "exit", "/quit", "bye"):
            print("Nupen: goodbye.")
            return 0
        print("Nupen:", conv.reply(line))


if __name__ == "__main__":
    sys.exit(main())
