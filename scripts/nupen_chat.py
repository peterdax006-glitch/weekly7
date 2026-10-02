"""Talk to Nupen in the terminal. Every exchange is appended to chat_transcripts.jsonl for the teacher to grade and turn into lessons.

    python scripts/nupen_chat.py            # interactive; /quit to leave, /reset to forget the conversation
    python scripts/nupen_chat.py --once "Hello"
"""
from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from creator.lm import dialogue as D  # noqa: E402


def load_model() -> Any:
    try:
        from creator.lm import api
    except ImportError:
        return None
    return api.load_current()


def exchange(lm: Any, history: list[dict[str, str]], user_text: str, session: str, transcripts: Path | None = None) -> str:
    history.append({"role": "user", "text": user_text})
    prompt = D.prompt_for(history)
    raw = lm.generate(prompt, max_new_tokens=96, temperature=0.7)
    reply, ok = D.extract_reply(raw, prompt)
    history.append({"role": "nupen", "text": reply})
    D.append_transcript({"ts": time.time(), "session": session, "turn": len(history) // 2, "source": "chat", "user": user_text,
                         "nupen": reply, "raw": raw, "turn_ok": ok, "context": history[:-2][-6:], "grade": None}, transcripts)
    return reply


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", help="send one message and exit")
    args = ap.parse_args(argv)
    lm = load_model()
    if lm is None:
        print("Nupen's language model is not trained yet - nothing to talk to. (Run the stage tests later; the trainer is still working.)")
        return 0
    session = uuid.uuid4().hex[:8]
    history: list[dict[str, str]] = []
    if args.once:
        print("Nupen:", exchange(lm, history, args.once, session))
        return 0
    print("Talking to Nupen. /quit to leave, /reset to start over.")
    while True:
        try:
            line = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line == "/quit":
            return 0
        if line == "/reset":
            history.clear()
            continue
        print("Nupen:", exchange(lm, history, line, session))


if __name__ == "__main__":
    raise SystemExit(main())
