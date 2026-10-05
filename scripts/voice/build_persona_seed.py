"""PC-side persona seed: curated-template expansion + local small-model generation (+ merge) -> filter.

Steps (each resumable, run at idle priority):
  templates  expand curated pair templates with slot values -> raw_templates.jsonl
  generate   call a local llama-server (1.7B/0.6B) with category few-shot prompts -> raw_llm.jsonl
  merge      add rows from an optional conv_rows.jsonl (owner/reply or chat messages) -> raw_conv.jsonl
  filter     persona_filter over all raw files -> persona_seed.jsonl (+ report)
Usage: python build_persona_seed.py STEP WORKDIR [--url http://127.0.0.1:8791] [--calls N] [--target N]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import persona_data as pd  # noqa: E402
import persona_filter as pf  # noqa: E402


def fill(t: str, slots: dict[str, str]) -> str:
    return re.sub(r"\{(\w+)\}", lambda m: slots[m.group(1)], t)


def cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def expand_templates(per_template: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for cat, c in pd.CATEGORIES.items():
        for owner_t, reply_t in c["pairs"]:
            names = sorted(set(re.findall(r"\{(\w+)\}", owner_t + reply_t)))
            combos = set()
            for _ in range(per_template * 4):
                if len(combos) >= (per_template if names else 1):
                    break
                combos.add(tuple(rng.choice(pd.SLOTS[n]) for n in names))
            for cb in combos:
                s = dict(zip(names, cb))
                o, r = cap(fill(owner_t, s)), cap(fill(reply_t, s))
                rows.append({"owner": o, "reply": r, "category": cat, "origin": "template"})
                codas = pd.CODAS.get(c["group"], [])
                if codas and len(pf.sentences(r)) == 1:  # curated closer on one-sentence replies -> second phrasing
                    for coda in rng.sample(codas, min(2, len(codas))):
                        rows.append({"owner": o, "reply": f"{r} {coda}", "category": cat, "origin": "template+coda"})
    return rows


def prompt_for(cat: str, rng: random.Random) -> list[dict]:
    c = pd.CATEGORIES[cat]
    shots = []
    for o, r in rng.sample(c["pairs"], min(3, len(c["pairs"]))):
        names = sorted(set(re.findall(r"\{(\w+)\}", o + r)))
        s = {n: rng.choice(pd.SLOTS[n]) for n in names}
        shots.append((cap(fill(o, s)), cap(fill(r, s))))
    topic = rng.choice(pd.TOPICS)
    rules = ("Nupen: calm, precise British butler-style assistant, light dry wit, says \"sir\", at most two short "
             "sentences, honest, no exclamation marks, no film quotes or character names.")
    ex = "\n".join(f"OWNER: {o}\nNUPEN: {r}\n" for o, r in shots)
    user = (
        f"{rules}\nSituation: {c['intent']} (inspiration: {topic}).\nExamples:\n{ex}\n"
        "Write 4 NEW different exchanges in exactly this format, nothing else:\nOWNER: ...\nNUPEN: ...\n/no_think"
    )
    return [{"role": "user", "content": user}]


PAIR_RE = re.compile(r"OWNER:\s*(.+?)\s*\n\s*NUPEN:\s*(.+?)\s*(?=\n\s*OWNER:|\Z)", re.S)


def call(url: str, messages: list[dict], temperature: float, seed: int) -> str:
    body = json.dumps({"messages": messages, "temperature": temperature, "top_p": 0.95, "max_tokens": 300, "seed": seed}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def generate(work: Path, url: str, calls: int, seed: int, workers: int) -> None:
    out = work / "raw_llm.jsonl"
    done = sum(1 for _ in out.open(encoding="utf-8")) if out.exists() else 0
    wsum = sum(c["weight"] for c in pd.CATEGORIES.values())
    names = list(pd.CATEGORIES)
    weights = [pd.CATEGORIES[n]["weight"] / wsum for n in names]
    rng = random.Random(seed)
    jobs = [(rng.choices(names, weights)[0], rng.randrange(1 << 30), rng.random()) for _ in range(calls)]
    state = work / "gen_calls_done.txt"
    start = int(state.read_text()) if state.exists() else 0

    def one(j):
        cat, sd, tr = j
        r = random.Random(sd)
        text = call(url, prompt_for(cat, r), 0.7 + 0.3 * tr, sd)
        return cat, [{"owner": o.strip(), "reply": a.strip().split("\n")[0].strip(), "category": cat, "origin": "llm"}
                     for o, a in PAIR_RE.findall(text)]

    n = start
    with ThreadPoolExecutor(workers) as ex, out.open("a", encoding="utf-8") as fh:
        for i in range(start, calls, workers * 4):
            batch = jobs[i:i + workers * 4]
            for cat, rows in ex.map(one, batch):
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=True) + "\n")
            fh.flush()
            n = i + len(batch)
            state.write_text(str(n))
            print(f"calls {n}/{calls}", flush=True)


def merge_conv(work: Path, src: Path) -> None:
    out = work / "raw_conv.jsonl"
    rows = []
    if src.exists():
        for line in src.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            if "messages" in d:
                u = [m["content"] for m in d["messages"] if m["role"] == "user"]
                a = [m["content"] for m in d["messages"] if m["role"] == "assistant"]
                if u and a:
                    rows.append({"owner": u[-1], "reply": a[-1], "category": d.get("category", "conv"), "origin": "conv_rows"})
            elif "owner" in d and "reply" in d:
                rows.append({**d, "origin": "conv_rows"})
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print("conv rows", len(rows))


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["templates", "generate", "merge", "filter"])
    ap.add_argument("work")
    ap.add_argument("--url", default="http://127.0.0.1:8791")
    ap.add_argument("--calls", type=int, default=700)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--per-template", type=int, default=4)
    ap.add_argument("--target", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--conv", default="")
    a = ap.parse_args(argv)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    if a.step == "templates":
        rows = expand_templates(a.per_template, a.seed)
        (work / "raw_templates.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        print("template rows", len(rows), "categories", len(pd.CATEGORIES))
    elif a.step == "generate":
        generate(work, a.url, a.calls, a.seed, a.workers)
    elif a.step == "merge":
        merge_conv(work, Path(a.conv))
    else:
        rows = []
        for f in ("raw_templates.jsonl", "raw_conv.jsonl", "raw_llm.jsonl"):
            p = work / f
            if p.exists():
                rows += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        chat, rep = pf.run(rows, a.target, a.seed, 0.03, "persona_seed")
        (work / "persona_seed.jsonl").write_text("".join(json.dumps(c) + "\n" for c in chat), encoding="utf-8")
        (work / "persona_seed.report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(json.dumps({k: rep[k] for k in ("in", "after_dedupe", "out", "rejected", "sir_position")}))


if __name__ == "__main__":
    main(sys.argv[1:])
