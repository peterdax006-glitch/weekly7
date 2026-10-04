"""Talk to Nupen in the terminal, LAYER 2: a small local model is Nupen's voice; every answer is built from Nupen's own records.

    python scripts/nupen_talk.py                          # interactive (Qwen3-1.7B voice); 'quit' leaves, '/stats' shows the last turn
    python scripts/nupen_talk.py --once "what are you stuck on?"
    python scripts/nupen_talk.py --model 4b               # a bigger voice (or a GGUF path); the GPU pod answers when pulseroute routes it
    python scripts/nupen_talk.py --rules                  # layer 1 only (no model), with layer 2's open-question search
    python scripts/nupen_talk.py --root <checkout>        # read another checkout's records (default: this checkout)
    python scripts/nupen_talk.py --eval [--model 4b]      # the held-out conversation eval (creator/talkeval.py); --eval-layers 1,1.7b,4b

The voice needs no GPU: it leases a warm llama.cpp server of the model when the swarm's pool has one idle, else starts one (RAM-gated).
With no model available the rules answer - talking never fails because a model is missing."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from creator import talk as T  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", help="send one message, print the answer, exit")
    ap.add_argument("--model", default="1.7b", help="voice: 1.7b | 4b | 8b | a GGUF path (default 1.7b)")
    ap.add_argument("--ctx", type=int, default=0, help="context size (default: 8192 for 1.7b so a warm pool server is leased, else 4096)")
    ap.add_argument("--servers", type=int, default=0, help="model-server slots this machine may run (default: device setting llama_servers)")
    ap.add_argument("--route", default="", choices=("", "auto"), help="auto: use a healthy GPU pod route even if the device setting is off")
    ap.add_argument("--rules", action="store_true", help="no model: layer-1 understanding and rule speaking")
    ap.add_argument("--understand", default="hybrid", choices=("hybrid", "model", "rules"), help="who maps text to an intent")
    ap.add_argument("--root", default=str(ROOT), help="repo root whose state/ is read")
    ap.add_argument("--yes", action="store_true", help="auto-confirm pause/resume/approve (scripting)")
    ap.add_argument("--no-log", action="store_true", help="do not append to conversations.jsonl")
    ap.add_argument("--stats", action="store_true", help="print the turn's numbers (latency, tokens, who understood/spoke) after each answer")
    ap.add_argument("--eval", action="store_true", help="run the held-out conversation eval and exit")
    ap.add_argument("--eval-layers", default="", help="comma list for --eval, e.g. 1,1.7b,4b (default: 1 and --model)")
    ap.add_argument("--eval-limit", type=int, default=0, help="--eval: only the first N questions")
    a = ap.parse_args(argv)
    root = Path(a.root)
    if a.eval:
        from creator import talkeval as TE
        layers = [x.strip() for x in (a.eval_layers or f"1,{a.model}").split(",") if x.strip()]
        rep = TE.run(root, layers, limit=a.eval_limit, ctx=a.ctx, servers=a.servers or None, route=a.route)
        print(TE.table(rep))
        print(f"report: {rep['file']}")
        return 0
    voice = None
    if not a.rules:
        model = T.resolve_model(a.model)
        voice = T.Voice(model, ctx=a.ctx or (8192 if a.model == "1.7b" else 4096), servers=a.servers or None, route=a.route)
    conv = T.conversation(root, voice, mode="rules" if a.rules else a.understand, log=not a.no_log, auto_confirm=a.yes)
    sess: T.Session = conv.talk_session  # type: ignore[attr-defined]
    try:
        if a.once:
            print(conv.reply(a.once))
            if a.stats:
                print(json.dumps(dict(sess.turn, voice_at=voice.where() if voice else "")))
            return 0
        print(T.banner(voice))
        if voice is not None and not voice.open():
            print(f"(no voice: {voice.error} - the rules answer)")
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
            if line == "/stats":
                print(json.dumps(dict(sess.turn, voice_at=voice.where() if voice else "")))
                continue
            print("Nupen:", conv.reply(line))
            if a.stats:
                print(json.dumps(sess.turn))
    finally:
        if voice is not None:
            voice.close()


if __name__ == "__main__":
    sys.exit(main())
