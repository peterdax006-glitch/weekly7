"""THE THINKING BENCHMARK (owner, 2 Oct 2026, night: "make it so when I wake up Nupen can think and reason much much better than when i fell asleep").

An HONEST, FROZEN measurement of how well Nupen thinks, run tonight and again in the morning on EXACTLY the same items, so the change is a number with
a confidence interval and not an impression. The benchmark MEASURES; it never trains, never writes anything but its own files under
state/creator/thinkbench/, and uses whatever predictor / strategy / procedure Nupen has at the moment it runs.

FOUR PARTS (every part reports n, score, baseline, 95% CI, and a per-item vector so a later run can be compared item by item):
  a  PREDICTION   held-out items of creator.thinking (verdict) and creator.drillsources (git_fixed, git_churn, journal_persist), scored by Brier against
                  the running base rate, with the predictor Nupen has now (for the drill sources: the best variant in drill_runs.jsonl).
  b  JUDGMENT     the same subjects (verdict, git_fixed) answered by the LOCAL model through creator.judgment's current best strategy (and a fixed
                  reference strategy, so a change of the model's skill is not confused with a change of the chosen strategy): Brier and accuracy.
  c  REASONING    multiple-choice questions with exact answers mined from Nupen's own history (which of 4 packages was adopted, which test covers a
                  module, which capability a module belongs to, which constraint was the top one at a snapshot), answered by the local model under
                  Nupen's reasoning procedure (creator.reasoning.procedure_text). Accuracy against chance (and, for constraints, against persistence).
  d  PLANNING     at the start of a window of kernel packages the local model ranks the candidates; the score is the Spearman rank correlation with what
                  turned out to matter (adopted, plus how closely the package matches what the owner asked for in the next 24 h). Baselines: random (0)
                  and the order the kernel planned them in.

NO FUTURE INFORMATION: every prompt uses only what existed before the item's own time (histories resolved strictly before, snapshots before the
window, directives before the window). The items, their questions and their answers are chosen ONCE (`freeze`) and stored with a content hash in
items.json; `load_frozen` refuses an edited file. Results go to baseline_<ISO>.json / now_<ISO>.json. Model calls are made with at most two local
servers that this module starts itself (pidfiles under the temp directory) and stops when done."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import queue
import random
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from creator import anticipation as A
from creator import drillsources as D
from creator import judgment as J
from creator import reasoning as R
from creator import thinking as T

VERSION = 1
SEED = 14
DRILL_SOURCES = ("git_fixed", "git_churn", "journal_persist")
PER_SOURCE = 40                     # prediction items per source (evenly spaced over the held-out 40% of the walk-forward)
JUDGE_PER_TOPIC = 25
Q_PER_KIND = 10
WINDOW = 6                          # packages per planning window
MAX_WINDOWS = 12
STRIDE = 2
REF_STRATEGY = {"shots": 2, "hint": 1}
ANSWER = re.compile(r"ANSWER\s*[:=]\s*\(?([A-D])\b", re.I)
RANKLINE = re.compile(r"RANK\s*[:=]\s*([0-9 ,>\-]+)", re.I)


def bench_dir(state: Path) -> Path:
    return Path(state) / "thinkbench"


# ------------------------------------------------------------------------------------------------ statistics
_T975 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23, 12: 2.18, 15: 2.13, 20: 2.09, 30: 2.04}


def tcrit(df: int) -> float:
    if df >= 60:
        return 1.96
    for k in sorted(_T975, reverse=True):
        if df >= k:
            return _T975[k] if df < 60 else 1.96
    return 12.71


def mean_ci(xs: Sequence[float]) -> list[Optional[float]]:
    """[mean, lo, hi]: mean with a 95% t interval (None bounds when n < 2)."""
    n = len(xs)
    if not n:
        return [None, None, None]
    m = sum(xs) / n
    if n < 2:
        return [round(m, 4), None, None]
    se = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n)
    h = tcrit(n - 1) * se
    return [round(m, 4), round(m - h, 4), round(m + h, 4)]


def wilson(k: int, n: int) -> list[Optional[float]]:
    """[rate, lo, hi]: Wilson 95% interval of a proportion."""
    if not n:
        return [None, None, None]
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(p, 4), round(c - h, 4), round(c + h, 4)]


def ranks(xs: Sequence[float]) -> list[float]:
    """Average ranks (1 = smallest); ties share the mean rank."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Rank correlation with ties; None when either side is constant."""
    if len(a) != len(b) or len(a) < 3:
        return None
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    va, vb = sum((x - ma) ** 2 for x in ra), sum((x - mb) ** 2 for x in rb)
    if va == 0 or vb == 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / math.sqrt(va * vb)


def _even(xs: Sequence[Any], k: int) -> list[Any]:
    if len(xs) <= k:
        return list(xs)
    return [xs[int(i * len(xs) / k)] for i in range(k)]


def _iso(t: float) -> str:
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%d %H:%M")


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------------------------ sources of the item set (read only)
def _drill_preds(source: str, state: Path, repo: Path, owner_dir: Path) -> tuple[list[T.Pred], dict[str, Any]]:
    """The predictor Nupen has NOW for a drill source: the best variant found by the drill search (default variant when none), walked forward."""
    best = D.best_variant(state, source)
    variant: dict[str, Any] = dict(best["variant"]) if best else {"decay": 0.97, "k": 3.0}
    items = D.apply_variant(D.load(source, state, repo, owner_dir / "JOURNAL.md", owner_dir), variant, owner_dir / "JOURNAL.md")
    preds = D.walk_forward(items, source, float(variant.get("decay", 0.97)), float(variant.get("k", 3.0)), str(variant.get("agg", "mean")),
                           float(variant.get("cap", 0.02)))
    return preds, variant


def _all_preds(state: Path, repo: Path, owner_dir: Path) -> dict[str, tuple[list[T.Pred], dict[str, Any]]]:
    out: dict[str, tuple[list[T.Pred], dict[str, Any]]] = {"verdict": (T.replay(T.load_items(state), "verdict"), {"model": "thinking.replay"})}
    for s in DRILL_SOURCES:
        try:
            out[s] = _drill_preds(s, state, repo, owner_dir)
        except Exception as e:                                       # noqa: BLE001 - a source without data is skipped, recorded
            out[s] = ([], {"error": f"{type(e).__name__}: {e}"[:200]})
    return out


def _wp_rows(state: Path) -> dict[str, dict[str, Any]]:
    wp: dict[str, dict[str, Any]] = {}
    for r in T._jsonl(state / "ledger.jsonl"):
        d = r.get("data") or {}
        if r.get("rtype") == "WorkPackage" and d.get("package_id") and d["package_id"] not in wp:
            wp[d["package_id"]] = d
    return wp


def _pkg_line(it: T.Item, d: dict[str, Any]) -> str:
    outs = d.get("outputs") or []
    return (f"requirement {it.req or '?'} - {str(d.get('objective', ''))[:110]} (why: {str(d.get('why_it_exists', ''))[:150]}; "
            f"{len(outs)} output files, e.g. {outs[0] if outs else '-'}; spec {it.spec_len} chars; planned {_iso(it.created)} UTC)")


_MOD = re.compile(r"modules (.+?); tests (.+?); floor")


def _capabilities(state: Path) -> list[dict[str, Any]]:
    out = []
    for r in T._jsonl(state / "ledger.jsonl"):
        if r.get("rtype") != "Capability":
            continue
        d = r.get("data") or {}
        m = _MOD.search(str(d.get("description", "")))
        if m:
            out.append({"component": d.get("component"), "name": str(d.get("name", "")), "modules": [x.strip() for x in m.group(1).split(",")],
                        "tests": [x.strip() for x in m.group(2).split(",")]})
    return out


def _snapshots(state: Path) -> list[dict[str, Any]]:
    out = []
    for r in T._jsonl(state / "constraints.jsonl"):
        if r.get("at") and r.get("event") in ("snapshot", "act"):
            out.append({**r, "t": T._local_ts(r["at"])})
    return sorted(out, key=lambda r: r["t"])


def _opts(rng: random.Random, truth: str, others: Sequence[str]) -> tuple[list[str], int]:
    pool = [o for o in dict.fromkeys(others) if o != truth]
    rng.shuffle(pool)
    opts = pool[:3] + [truth]
    rng.shuffle(opts)
    return opts, opts.index(truth)


def _questions(state: Path, rng: random.Random) -> list[dict[str, Any]]:
    """Reasoning questions with exact answers, mined from the ledger, kernel log and constraint log. Each stores its full prompt (frozen)."""
    qs: list[dict[str, Any]] = []
    letters = "ABCD"

    def add(kind: str, qid: str, text: str, opts: list[str], ans: int, baseline: Optional[int] = None) -> None:
        body = text + "\n" + "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(opts))
        qs.append({"id": f"c:{kind}:{qid}", "kind": kind, "prompt": body, "answer": ans, "baseline_pick": baseline})

    caps = _capabilities(state)
    mod_cap: dict[str, list[dict[str, Any]]] = {}
    for c in caps:
        for m in c["modules"]:
            mod_cap.setdefault(m, []).append(c)
    all_tests = sorted({t for c in caps for t in c["tests"]})
    single = [(m, cs[0]) for m, cs in sorted(mod_cap.items()) if len(cs) == 1 and len(caps) >= 4]
    rng.shuffle(single)
    picked = 0
    for m, c in single:
        stem = m.rsplit("/", 1)[-1][:-3]
        truth = next((t for t in c["tests"] if t.endswith(f"test_creator_{stem}.py")), None)
        if truth and picked < Q_PER_KIND:
            opts, ans = _opts(rng, truth, [t for t in all_tests if t not in c["tests"]])
            if len(opts) == 4:
                add("test_file", m, f"Which test file covers the module {m}?", opts, ans)
                picked += 1
    names = sorted({c["name"] for c in caps})
    rng.shuffle(single)
    for m, c in single[:Q_PER_KIND]:
        opts, ans = _opts(rng, c["name"], names)
        if len(opts) == 4:
            add("capability", m, f"Nupen's development ledger groups modules into capabilities. Which capability does the module {m} belong to?", opts, ans)
    items = [it for it in T.load_items(state) if it.resolved is not None and it.outcome]
    wp = _wp_rows(state)
    won = [i for i in items if i.outcome == "ADOPTED"]
    lost = [i for i in items if i.outcome != "ADOPTED"]
    for w in _even(won, Q_PER_KIND):
        near = sorted(lost, key=lambda x: abs(x.created - w.created))[:3]
        if len(near) < 3:
            continue
        grp = near + [w]
        rng.shuffle(grp)
        add("adopted", w.pkg, "Four development packages were planned around the same time. Exactly one was ADOPTED by the kernel (the other three were "
            "rejected, errored, cancelled or deferred). Which one was adopted?", [_pkg_line(x, wp.get(x.pkg, {})) for x in grp], grp.index(w))
    snaps = [s for s in _snapshots(state) if s.get("event") == "snapshot" and s.get("top")]
    cnames = sorted({str(x.get("name")) for s in snaps for x in s.get("ranked", [])})
    for k in range(1, len(snaps)):
        s, prev = snaps[k], snaps[k - 1]
        acts = [a for a in _snapshots(state) if a.get("event") == "act" and a["t"] < s["t"]][-3:]
        ctx = (f"Constraint snapshots so far (oldest first): " + "; ".join(f"{_iso(p['t'])} top={p['top']}" for p in snaps[max(0, k - 4):k]) +
               f". Previous snapshot ranking: " + ", ".join(f"{x['name']}={x['value']}" for x in prev.get("ranked", [])[:5]) +
               (". Actions taken on constraints: " + "; ".join(f"{a.get('constraint')} ({a.get('remedy')})" for a in acts) if acts else "") + ".")
        opts, ans = _opts(rng, str(s["top"]), cnames)
        if len(opts) == 4:
            add("constraint", _iso(s["t"]), f"{ctx}\nAt the snapshot taken {_iso(s['t'])} UTC, which limiting factor was ranked TOP?", opts, ans,
                opts.index(str(prev["top"])) if str(prev["top"]) in opts else None)
    return qs


def _windows(state: Path, owner_dir: Path, rng: random.Random) -> list[dict[str, Any]]:
    """Planning windows: WINDOW consecutive resolved packages (with both adopted and not-adopted ones); the state shown is that BEFORE the first one."""
    items = [it for it in T.load_items(state) if it.resolved is not None and it.outcome]
    wp = _wp_rows(state)
    snaps = [s for s in _snapshots(state) if s.get("event") == "snapshot" and s.get("top")]
    try:
        dirs = A.directives(owner_dir)
    except Exception:                                               # noqa: BLE001 - no owner log: relevance is adoption alone
        dirs = []
    docs = {it.pkg: A.Doc(it.created, f"{wp.get(it.pkg, {}).get('objective', '')} {wp.get(it.pkg, {}).get('why_it_exists', '')}", "pkg", it.pkg) for it in items}
    idf = A._idf(list(dirs) + list(docs.values())) if dirs else {}
    wins = [items[i:i + WINDOW] for i in range(0, len(items) - WINDOW + 1, STRIDE)]     # overlapping windows: only ~50 packages are resolved
    wins = [w for w in wins if 0 < sum(x.outcome == "ADOPTED" for x in w) < len(w)]
    out: list[dict[str, Any]] = []
    for w in _even(wins, MAX_WINDOWS):
        t0 = w[0].created
        before = [it for it in items if it.resolved is not None and it.resolved < t0][-10:]
        sb = [s for s in snaps if s["t"] < t0][-1:]
        ctx = [f"Time: {_iso(t0)} UTC."]
        if before:
            ctx.append(f"Of the last {len(before)} resolved packages {sum(x.outcome == 'ADOPTED' for x in before)} were adopted.")
        if sb:
            ctx.append("Latest constraint ranking: " + ", ".join(f"{x['name']}={x['value']}" for x in sb[0].get("ranked", [])[:4]) + ".")
        said = [d for d in dirs if d.t < t0][-2:]
        if said:
            ctx.append("Latest owner directives: " + " | ".join(d.text[:220].replace("\n", " ") for d in said))
        nxt = [d for d in dirs if t0 <= d.t <= t0 + 86400.0]
        rel: dict[str, float] = {}
        for it in w:
            m = max((A.match(d, docs[it.pkg], set(), idf)[0] for d in nxt), default=0.0) if nxt else 0.0
            rel[it.pkg] = round(float(it.outcome == "ADOPTED") + min(1.0, m), 4)
        shown = list(w)
        rng.shuffle(shown)
        out.append({"id": f"d:{w[0].pkg}", "context": " ".join(ctx), "candidates": [{"pkg": x.pkg, "text": _pkg_line(x, wp.get(x.pkg, {}))} for x in shown],
                    "relevance": rel, "planned_order": [x.pkg for x in w], "directives_after": len(nxt)})
    return out


def freeze(state: Path, repo: Path, owner_dir: Path, force: bool = False) -> dict[str, Any]:
    """Choose the benchmark items ONCE and store them with a content hash. Refuses to overwrite an existing frozen set unless `force`."""
    state = Path(state)
    p = bench_dir(state) / "items.json"
    if p.exists() and not force:
        return load_frozen(state)
    rng = random.Random(SEED)
    a: list[dict[str, Any]] = []
    for src, (preds, _meta) in _all_preds(state, repo, owner_dir).items():
        ps = [x for x in preds if x.outcome is not None]
        held = ps[int(len(ps) * D.SELECT_SPLIT):]
        for x in _even(held, PER_SOURCE):
            a.append({"id": f"a:{src}:{x.subject}", "source": src, "subject": x.subject, "created": x.made_at, "y": int(x.outcome or 0)})
    b = [{"id": f"b:{t}:{x['subject']}", "topic": t, "subject": x["subject"], "y": x["y"]}
         for t in J.TOPICS for x in _even([x for x in a if x["source"] == ("verdict" if t == "verdict" else "git_fixed")], JUDGE_PER_TOPIC)]
    body = {"version": VERSION, "seed": SEED, "frozen_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "state": str(state), "repo": str(repo), "a": a, "b": b, "c": _questions(state, rng), "d": _windows(state, owner_dir, rng)}
    body["hash"] = _sha({k: v for k, v in body.items() if k not in ("hash", "frozen_at")})
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(body, indent=1, sort_keys=True), encoding="utf-8")
    return body


def refreeze_planning(state: Path, owner_dir: Path) -> dict[str, Any]:
    """Rebuild ONLY part d of the frozen set (allowed only while no baseline has been written: a baseline pins the items for good)."""
    if list(bench_dir(state).glob("baseline_*.json")):
        raise ValueError("a baseline exists: the frozen items may not change")
    body = load_frozen(state)
    body["d"] = _windows(Path(state), owner_dir, random.Random(SEED))
    body["hash"] = _sha({k: v for k, v in body.items() if k not in ("hash", "frozen_at")})
    (bench_dir(state) / "items.json").write_text(json.dumps(body, indent=1, sort_keys=True), encoding="utf-8")
    return body


def load_frozen(state: Path) -> dict[str, Any]:
    body = json.loads((bench_dir(state) / "items.json").read_text(encoding="utf-8"))
    if _sha({k: v for k, v in body.items() if k not in ("hash", "frozen_at")}) != body.get("hash"):
        raise ValueError("the frozen thinking benchmark was edited after it was frozen (content hash mismatch)")
    return body


# ------------------------------------------------------------------------------------------------ model calls (at most two servers we start)
def default_llm_factory(i: int) -> Any:
    from creator import generator as G
    d = Path(tempfile.gettempdir()) / f"thinkbench_srv{i}"
    d.mkdir(parents=True, exist_ok=True)
    return G.LocalModel(pidfile=d / "llama_server.pid", servers=1, threads=6, startup_s=300.0)


class _Thinking:
    """A thinking-model server whose answers go through judgment.chat_text (the model's /no_think switch; <think> blocks stripped)."""

    def __init__(self, lm: Any) -> None:
        self.lm, self.model = lm, getattr(lm, "model", "")

    def __enter__(self) -> "_Thinking":
        self.lm.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.lm.__exit__(*exc)

    def chat(self, messages: list[dict[str, str]], **kw: Any) -> str:
        return J.chat_text(self.lm, messages, **kw)


def thinker_llm_factory(i: int) -> Any:
    """The THINKING model (device 'think_model') on its own private server, same threads and startup as the default; KeyError when none is
    configured or on disk (a benchmark of 'the thinker' must never silently fall back to the fast model)."""
    from creator import device as DEV
    from creator import generator as G
    path = DEV.think_model_path()
    if path is None:
        raise KeyError("no thinking model configured or on disk (device setting 'think_model')")
    d = Path(tempfile.gettempdir()) / f"thinkbench_think{i}"
    d.mkdir(parents=True, exist_ok=True)
    return _Thinking(G.LocalModel(model=path, pidfile=d / "llama_server.pid", servers=1, threads=6, startup_s=300.0))


def model_path(model: str) -> Optional[Path]:
    """'fast' -> the default code model; 'thinker' -> device 'think_model' (None when absent)."""
    from creator import generator as G
    if model == "thinker":
        from creator import device as DEV
        return DEV.think_model_path()
    return Path(G.DEFAULT_MODEL)


def ask_all(jobs: Sequence[tuple[str, list[dict[str, str]], int, float]], llm_factory: Callable[[int], Any], workers: int = 2,
            progress: Optional[Callable[[int, int], None]] = None) -> dict[str, Optional[str]]:
    """Run (key, messages, max_tokens, temperature) jobs on up to `workers` servers; a failed call maps to None. The servers stop when this returns."""
    q: queue.Queue[tuple[str, list[dict[str, str]], int, float]] = queue.Queue()
    for j in jobs:
        q.put(j)
    out: dict[str, Optional[str]] = {}
    lock = threading.Lock()

    def work(i: int) -> None:
        try:
            ctx = llm_factory(i)
            with ctx as llm:
                while True:
                    try:
                        key, msgs, mt, temp = q.get_nowait()
                    except queue.Empty:
                        return
                    try:
                        r: Optional[str] = llm.chat(msgs, max_tokens=mt, temperature=temp, seed=0, timeout=300.0)
                    except Exception:                               # noqa: BLE001 - one bad call is one missing answer
                        r = None
                    with lock:
                        out[key] = r
                        if progress:
                            progress(len(out), len(jobs))
        except Exception:                                           # noqa: BLE001 - a server that cannot start leaves its jobs to the others
            return

    ts = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(max(1, min(workers, 2)))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out


def parse_choice(reply: Optional[str]) -> Optional[int]:
    m = ANSWER.findall(reply or "")
    return "ABCD".index(m[-1].upper()) if m else None


def parse_rank(reply: Optional[str], n: int) -> Optional[list[int]]:
    """Candidate numbers 1..n best-first from 'RANK: 3, 1, 2'; missing ones are appended in shown order; None when nothing parses."""
    m = RANKLINE.findall(reply or "")
    if not m:
        return None
    seen: list[int] = []
    for x in re.findall(r"\d+", m[-1]):
        v = int(x)
        if 1 <= v <= n and v not in seen:
            seen.append(v)
    return seen + [i for i in range(1, n + 1) if i not in seen] if seen else None


# ------------------------------------------------------------------------------------------------ scoring
def _part(metric: str, higher_better: bool, n: int, score: Optional[float], baseline: Optional[float], gains: Sequence[float], per_item: dict[str, float],
          detail: dict[str, Any], extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """gain = improvement over the baseline (positive is better), with its 95% interval over the items."""
    m = mean_ci(gains)
    out = {"n": n, "metric": metric, "higher_is_better": higher_better, "score": score, "baseline": baseline, "gain": m[0], "gain_ci95": m[1:],
           "per_item": per_item, "detail": detail}
    out.update(extra or {})
    return out


def score_prediction(frozen: dict[str, Any], preds: dict[str, tuple[list[T.Pred], dict[str, Any]]]) -> dict[str, Any]:
    by = {s: {x.subject: x for x in ps} for s, (ps, _m) in preds.items()}
    per: dict[str, float] = {}
    gains: list[float] = []
    model_b: list[float] = []
    base_b: list[float] = []
    det: dict[str, dict[str, list[float]]] = {}
    miss = 0
    for it in frozen["a"]:
        x = by.get(it["source"], {}).get(it["subject"])
        if x is None or x.outcome is None:
            miss += 1
            continue
        bm, bb = T.brier(x.p, x.outcome), T.brier(x.base, x.outcome)
        per[it["id"]] = bm
        model_b.append(bm)
        base_b.append(bb)
        gains.append(bb - bm)
        d = det.setdefault(it["source"], {"model": [], "base": [], "last": []})
        d["model"].append(bm)
        d["base"].append(bb)
        d["last"].append(T.brier(x.last, x.outcome))
    detail = {s: {"n": len(d["model"]), "brier": round(sum(d["model"]) / len(d["model"]), 4), "brier_base_rate": round(sum(d["base"]) / len(d["base"]), 4),
                  "brier_last_value": round(sum(d["last"]) / len(d["last"]), 4), "gain_vs_base_rate": mean_ci([b - m for b, m in zip(d["base"], d["model"])]),
                  "predictor": preds[s][1]} for s, d in det.items() if d["model"]}
    n = len(model_b)
    return _part("Brier (lower is better) vs running base rate", False, n, round(sum(model_b) / n, 4) if n else None,
                 round(sum(base_b) / n, 4) if n else None, gains, per, detail, {"missing_items": miss})


def score_judgment(frozen: dict[str, Any], answers: dict[str, dict[str, Optional[float]]], preds: dict[str, tuple[list[T.Pred], dict[str, Any]]],
                   strategies: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """answers[strategy_name][item id] = probability (None = unusable reply, scored as 0.5). Primary strategy = 'current'."""
    by = {s: {x.subject: x for x in ps} for s, (ps, _m) in preds.items()}
    out: dict[str, Any] = {}
    for sname, ans in answers.items():
        gains, jb, bb, sb, acc, unusable = [], [], [], [], [], 0
        per: dict[str, float] = {}
        for it in frozen["b"]:
            if it["id"] not in ans:
                continue
            p = ans[it["id"]]
            if p is None:
                unusable += 1
                p = 0.5
            src = "verdict" if it["topic"] == "verdict" else "git_fixed"
            x = by.get(src, {}).get(it["subject"])
            if x is None:
                continue
            y = it["y"]
            per[it["id"]] = T.brier(p, y)
            jb.append(per[it["id"]])
            bb.append(T.brier(x.base, y))
            sb.append(T.brier(x.p, y))
            gains.append(bb[-1] - jb[-1])
            acc.append(int((p > 0.5) == bool(y)))
        n = len(jb)
        out[sname] = _part("Brier (lower is better) vs running base rate", False, n, round(sum(jb) / n, 4) if n else None, round(sum(bb) / n, 4) if n else None,
                           gains, per, {"strategy": strategies.get(sname), "accuracy": wilson(sum(acc), n), "unusable_replies": unusable,
                                        "statistical_predictor_brier": round(sum(sb) / n, 4) if n else None})
    primary = dict(out.get("current") or {"n": 0})
    primary["reference_strategy"] = {k: {kk: vv for kk, vv in v.items() if kk != "per_item"} for k, v in out.items() if k != "current"}
    return primary


def score_reasoning(frozen: dict[str, Any], replies: dict[str, Optional[str]]) -> dict[str, Any]:
    per: dict[str, float] = {}
    kinds: dict[str, list[int]] = {}
    base_hits: list[int] = []
    unparsed = 0
    for q in frozen["c"]:
        if q["id"] not in replies:
            continue
        c = parse_choice(replies[q["id"]])
        unparsed += c is None
        ok = int(c == q["answer"])
        per[q["id"]] = float(ok)
        kinds.setdefault(q["kind"], []).append(ok)
        if q.get("baseline_pick") is not None or q["kind"] == "constraint":
            base_hits.append(int(q.get("baseline_pick") == q["answer"]))
    allv = [v for xs in kinds.values() for v in xs]
    n = len(allv)
    detail = {k: {"n": len(v), "accuracy": wilson(sum(v), len(v)), "chance": 0.25} for k, v in kinds.items()}
    if base_hits:
        detail["constraint"]["persistence_baseline_accuracy"] = wilson(sum(base_hits), len(base_hits))
    acc = wilson(sum(allv), n)
    return _part("accuracy (higher is better) vs chance 0.25", True, n, acc[0], 0.25, [v - 0.25 for v in allv], per, detail,
                 {"score_ci95": acc[1:], "unparsed_replies": unparsed})


def score_planning(frozen: dict[str, Any], replies: dict[str, Optional[str]]) -> dict[str, Any]:
    per: dict[str, float] = {}
    kern, unparsed = [], 0
    for w in frozen["d"]:
        if w["id"] not in replies:
            continue
        cands = [c["pkg"] for c in w["candidates"]]
        order = parse_rank(replies[w["id"]], len(cands))
        if order is None:
            unparsed += 1
            order = list(range(1, len(cands) + 1))                 # an unusable reply is the shown (random) order: no skill, not a free pass
        pos = {cands[i - 1]: len(cands) - r for r, i in enumerate(order)}          # higher = ranked earlier
        rel = [w["relevance"][c] for c in cands]
        s = spearman([pos[c] for c in cands], rel)
        kb = spearman([len(w["planned_order"]) - w["planned_order"].index(c) for c in cands], rel)
        if s is not None:
            per[w["id"]] = s
            kern.append(kb if kb is not None else 0.0)
    n = len(per)
    m = mean_ci(list(per.values()))
    return _part("Spearman correlation with what mattered (higher is better) vs random 0", True, n, m[0], 0.0, list(per.values()), per,
                 {"kernel_plan_order_baseline": mean_ci(kern), "windows_frozen": len(frozen["d"])}, {"score_ci95": m[1:], "unparsed_replies": unparsed})


# ------------------------------------------------------------------------------------------------ prompts
def _c_messages(q: dict[str, Any]) -> list[dict[str, str]]:
    sysm = (R.procedure_text(True) + "\n\nYou now answer a multiple-choice question about the development history of this system. Use the procedure: think "
            "briefly (at most three short lines), then finish with exactly 'ANSWER: <letter>'.")
    return [{"role": "system", "content": sysm}, {"role": "user", "content": q["prompt"]}]


def _d_messages(w: dict[str, Any]) -> list[dict[str, str]]:
    lines = [f"[{i + 1}] {c['text']}" for i, c in enumerate(w["candidates"])]
    sysm = (R.procedure_text(True) + "\n\nYou plan the next development step of this system. Rank the candidate packages by what will matter most (what the "
            "kernel will adopt and what the owner is asking for). Think briefly, then finish with exactly 'RANK: ' and the candidate numbers best first, "
            "comma-separated, for example 'RANK: 3, 1, 2, 4, 6, 5'.")
    return [{"role": "system", "content": sysm}, {"role": "user", "content": f"State: {w['context']}\nCandidates:\n" + "\n".join(lines)}]


def current_strategy(state: Path, repo: Path, topic: str) -> dict[str, Any]:
    """The judgment strategy Nupen would use now: best in trust_section for the topic, else the first live strategy, else the default."""
    try:
        t = J.trust_section(state, repo).get(topic)
        if t and t.get("best_strategy"):
            return dict(t["best_strategy"])
    except Exception:                                               # noqa: BLE001
        pass
    try:
        recs = [r for r in T._jsonl(J.path(state)) if r.get("topic") == topic and r.get("p") is not None]
        live = J.alive(recs)
        if recs and live:
            return dict(live[0])
    except Exception:                                               # noqa: BLE001
        pass
    return dict(REF_STRATEGY)


def versions(state: Path, repo: Path, model: str = "fast") -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    files = ("thinking", "drillsources", "judgment", "reasoning", "anticipation", "thinkbench")
    code = {f: hashlib.sha256((root / f"{f}.py").read_bytes()).hexdigest()[:12] for f in files if (root / f"{f}.py").is_file()}
    mp = None
    try:
        mp = model_path(model)
        st = mp.stat()                                                  # type: ignore[union-attr]
        local = {"file": mp.name, "bytes": st.st_size, "mtime": _iso(st.st_mtime), "role": model}   # type: ignore[union-attr]
    except Exception:                                               # noqa: BLE001
        local = {"file": str(mp), "error": "not found", "role": model}
    try:
        import subprocess
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:                                               # noqa: BLE001
        head = "?"
    return {"thinkbench": VERSION, "reasoning_procedure": getattr(R, "VERSION", None), "procedure_sha": hashlib.sha256(R.procedure_text(True).encode()).hexdigest()[:12],
            "local_model": local, "code_sha": code, "repo_head": head,
            "judgment_strategies": {t: current_strategy(state, repo, t) for t in J.TOPICS}, "judgment_reference_strategy": REF_STRATEGY}


def run(state: Path, repo: Path, owner_dir: Path, frozen: Optional[dict[str, Any]] = None, llm_factory: Optional[Callable[[int], Any]] = None,
        use_model: bool = True, workers: int = 2, progress: Optional[Callable[[int, int], None]] = None, model: str = "fast") -> dict[str, Any]:
    """One full run of the frozen benchmark with whatever Nupen has now. Parts a needs no model; b, c, d use the local model (skipped with use_model=False)."""
    state = Path(state)
    t0 = time.monotonic()
    frozen = frozen or load_frozen(state)
    preds = _all_preds(state, repo, owner_dir)
    res: dict[str, Any] = {"at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "items_hash": frozen["hash"], "versions": versions(state, repo, model),
                           "parts": {"a_prediction": score_prediction(frozen, preds)}}
    if use_model:
        wsub = {t: current_strategy(state, repo, t) for t in J.TOPICS}
        strategies = {"current": wsub, "reference": dict(REF_STRATEGY)}
        cases = {t: {c.subject: c for c in J.load_cases(t, state, repo)} for t in J.TOPICS}
        allc = {t: list(cases[t].values()) for t in J.TOPICS}
        jobs: list[tuple[str, list[dict[str, str]], int, float]] = []
        for it in frozen["b"]:
            c = cases[it["topic"]].get(it["subject"])
            if c is None:
                continue
            for sname in ("current", "reference"):
                s = wsub[it["topic"]] if sname == "current" else REF_STRATEGY
                if sname == "reference" and s == wsub[it["topic"]]:
                    continue
                jobs.append((f"b|{sname}|{it['id']}", J.build_prompt(s, allc[it["topic"]], c), 120, 0.2))
        jobs += [(q["id"], _c_messages(q), 220, 0.0) for q in frozen["c"]]
        jobs += [(w["id"], _d_messages(w), 160, 0.0) for w in frozen["d"]]
        rep = ask_all(jobs, llm_factory or (thinker_llm_factory if model == "thinker" else default_llm_factory), workers, progress)
        ans: dict[str, dict[str, Optional[float]]] = {"current": {}, "reference": {}}
        for k, r in rep.items():
            if k.startswith("b|"):
                _b, sname, iid = k.split("|", 2)
                ans[sname][iid] = J.parse(r) if r else None
        for iid_item in frozen["b"]:                                  # reference == current for a topic: the same answers count for both
            for sname in ("reference",):
                if iid_item["id"] not in ans[sname] and ans["current"].get(iid_item["id"], "x") != "x" and REF_STRATEGY == wsub[iid_item["topic"]]:
                    ans[sname][iid_item["id"]] = ans["current"][iid_item["id"]]
        res["parts"]["b_judgment"] = score_judgment(frozen, ans, preds, {"current": wsub, "reference": REF_STRATEGY})
        res["parts"]["c_reasoning"] = score_reasoning(frozen, {k: v for k, v in rep.items() if k.startswith("c:")})
        res["parts"]["d_planning"] = score_planning(frozen, {k: v for k, v in rep.items() if k.startswith("d:")})
        res["calls"] = {"jobs": len(jobs), "answered": sum(1 for v in rep.values() if v is not None)}
    res["runtime_s"] = round(time.monotonic() - t0, 1)
    return res


def save(state: Path, res: dict[str, Any], prefix: str) -> Path:
    p = bench_dir(state) / f"{prefix}_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1, sort_keys=True), encoding="utf-8")
    return p


# ------------------------------------------------------------------------------------------------ comparison
def compare_part(base: dict[str, Any], now: dict[str, Any]) -> dict[str, Any]:
    """Paired comparison on the items both runs scored: change = (now - baseline) in the direction of 'better' (positive = Nupen improved)."""
    bi, ni = base.get("per_item") or {}, now.get("per_item") or {}
    ids = sorted(set(bi) & set(ni))
    sign = 1.0 if now.get("higher_is_better") else -1.0
    d = [sign * (ni[i] - bi[i]) for i in ids]
    m = mean_ci(d)
    verdict = "no data"
    if m[0] is not None:
        if m[1] is None:
            verdict = "too few items to tell"
        elif m[1] > 0:
            verdict = "SIGNIFICANTLY BETTER"
        elif m[2] is not None and m[2] < 0:
            verdict = "SIGNIFICANTLY WORSE"
        else:
            verdict = "no significant change"
    return {"n_paired": len(ids), "change": m[0], "change_ci95": m[1:], "verdict": verdict, "score_before": base.get("score"), "score_now": now.get("score"),
            "baseline_before": base.get("baseline"), "baseline_now": now.get("baseline")}


def compare(base: dict[str, Any], now: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"same_items": base.get("items_hash") == now.get("items_hash"), "parts": {}}
    for k, bp in (base.get("parts") or {}).items():
        np_ = (now.get("parts") or {}).get(k)
        if np_ and bp.get("n") and np_.get("n"):
            out["parts"][k] = compare_part(bp, np_)
    return out
