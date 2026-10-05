"""Training-data curation for the GPU modules (owner 5 Oct: "10x faster, WITHOUT losing quality"): fewer tokens, same signal. CPU only.

Every step is a proxy that needs no GPU and is reported per step (rows and tokens dropped, and why):
  length    rows above the module's cap (measured p99.5 of its own mix, never below the p99) - they cost the most padding and teach the least
  exact     identical conversations
  variant   the same user prompt answered several ways (the thinker mix carries every question twice: a worked trace AND the one-line answer):
            keep the preferred variant only
  eval      rows the eval-suite exclusion list names (creator.tools.evalexclude; recompiling honours EXCLUDE.json)
  template  planted-bug rows that restate one bug (same file, line, mutation: only the failing test differs): at most `max_per_template`
  easy      rows a code-only tool already gets right (PINPOINT label 1 = the spectrum ranker's top candidate): kept at `easy_keep` of their share;
            the hard rows (label >= 2) are ALL kept
  balance   per-group round-robin (file, repo, mutation kind) up to the budget, so no one source fills the budget
  near      near-duplicates inside the kept set (creator.trainmix.NearIndex)
Rows removed by `easy` and `balance` go to a RESERVE file in priority order: if a module's held-out result misses its adopt rule, the next
GPU job tops the module up from the reserve instead of regenerating anything. Rows removed as dup/over-length/eval are never reserved.

Calibration mixes are deduplicated and length-capped ONLY: their answer distribution is the thing being learned (ECE), so it is never rebalanced."""
from __future__ import annotations

import collections
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

CHARS_PER_TOKEN = 3.6                # the repo-wide estimate (creator.gpueff.length_stats)


def tokens(row: Mapping[str, Any]) -> float:
    return sum(len(m.get("content", "")) for m in row.get("messages") or []) / CHARS_PER_TOKEN


def _h(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def conv_key(row: Mapping[str, Any]) -> str:
    return _h(json.dumps(row.get("messages"), sort_keys=True))


def user_text(row: Mapping[str, Any]) -> str:
    return "\n".join(m["content"] for m in (row.get("messages") or [])[:-1] if m.get("role") == "user")


def order_key(row: Mapping[str, Any], i: int) -> str:
    """Deterministic pseudo-random priority (a hash of the row), so the same inputs always keep the same rows."""
    return _h(str(row.get("id") or i) + conv_key(row)[:12])


def percentile(vals: Sequence[float], p: float) -> float:
    v = sorted(vals)
    return v[min(len(v) - 1, int(p * len(v)))] if v else 0.0


@dataclass
class Policy:
    name: str
    len_cap: Optional[float] = None                         # tokens; None -> the p99.5 of the mix itself
    len_floor_pct: float = 0.995
    variant_key: Optional[Callable[[Mapping[str, Any]], str]] = None        # rows sharing this key are variants of one item
    variant_pref: Optional[Callable[[Mapping[str, Any]], float]] = None     # higher = preferred; default = the shorter answer
    template_key: Optional[Callable[[Mapping[str, Any]], str]] = None
    max_per_template: int = 0                               # 0 = no cap
    is_easy: Optional[Callable[[Mapping[str, Any]], bool]] = None
    easy_keep: float = 1.0                                  # fraction of the easy rows kept
    group_key: Optional[Callable[[Mapping[str, Any]], str]] = None
    row_cap: int = 0                                        # 0 = keep everything that survived
    near_dup: bool = True
    reserve_template: bool = False                          # rows over the template cap go to the reserve instead of being dropped
    extra: dict[str, Any] = field(default_factory=dict)


def _step(rep: list[dict[str, Any]], name: str, before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]], why: str) -> None:
    rep.append({"step": name, "rows_in": len(before), "rows_out": len(after), "tokens_in": round(sum(tokens(r) for r in before)),
                "tokens_out": round(sum(tokens(r) for r in after)), "why": why})


def _round_robin(rows: Sequence[Mapping[str, Any]], key: Callable[[Mapping[str, Any]], str], cap: int) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Take rows group by group in turn (each group in its hash order) until `cap`; the rest, in the same round-robin order, is the reserve."""
    groups: dict[str, list[tuple[str, Mapping[str, Any]]]] = collections.defaultdict(list)
    for i, r in enumerate(rows):
        groups[key(r)].append((order_key(r, i), r))
    for g in groups.values():
        g.sort(key=lambda t: t[0])
    ordered: list[Mapping[str, Any]] = []
    names = sorted(groups, key=lambda k: (len(groups[k]), k))
    depth = 0
    while len(ordered) < len(rows):
        for k in names:
            if depth < len(groups[k]):
                ordered.append(groups[k][depth][1])
        depth += 1
    return ordered[:cap], ordered[cap:]


def curate(rows: Sequence[Mapping[str, Any]], pol: Policy, exclusions: Any = None) -> dict[str, Any]:
    """rows -> {"kept", "reserve", "report"}. Pure CPU; deterministic."""
    rep: list[dict[str, Any]] = []
    cur = list(rows)
    toks = [tokens(r) for r in cur]
    _step(rep, "input", cur, cur, "rows as given")
    reserve: list[Mapping[str, Any]] = []
    # length
    cap = pol.len_cap if pol.len_cap else max(percentile(toks, pol.len_floor_pct), percentile(toks, 0.99))
    nxt = [r for r, t in zip(cur, toks) if t <= cap]
    _step(rep, "length", cur, nxt, f"cap {cap:.0f} tok (p99.5 of the mix)" if not pol.len_cap else f"cap {cap:.0f} tok")
    cur = nxt
    # exact duplicates
    seen: set[str] = set()
    nxt = []
    for r in cur:
        k = conv_key(r)
        if k not in seen:
            seen.add(k)
            nxt.append(r)
    _step(rep, "exact", cur, nxt, "identical conversations")
    cur = nxt
    # variants of one item
    if pol.variant_key:
        pref = pol.variant_pref or (lambda r: -len(r["messages"][-1].get("content", "")))
        best: dict[str, Mapping[str, Any]] = {}
        for r in cur:
            k = pol.variant_key(r)
            if k not in best or pref(r) > pref(best[k]):
                best[k] = r
        keep = {id(v) for v in best.values()}
        nxt = [r for r in cur if id(r) in keep]
        _step(rep, "variant", cur, nxt, "one answer per prompt (the preferred variant)")
        cur = nxt
    # eval-suite exclusion
    if exclusions:
        nxt = [r for r in cur if not exclusions.violation(user_text(r) + "\n" + (r["messages"][-1].get("content", "") if r.get("messages") else ""), str(r.get("id") or ""))]
        _step(rep, "eval", cur, nxt, "eval-suite exclusion list (EXCLUDE.json)")
        cur = nxt
    # template duplicates
    if pol.template_key and pol.max_per_template:
        per: dict[str, list[tuple[str, Mapping[str, Any]]]] = collections.defaultdict(list)
        for i, r in enumerate(cur):
            per[pol.template_key(r)].append((order_key(r, i), r))
        keep = set()
        for g in per.values():
            g.sort(key=lambda t: t[0])
            keep |= {id(r) for _, r in g[:pol.max_per_template]}
        nxt = [r for r in cur if id(r) in keep]
        if pol.reserve_template:
            reserve += [r for r in cur if id(r) not in keep]
        _step(rep, "template", cur, nxt, f"at most {pol.max_per_template} rows per restated item")
        cur = nxt
    # easy rows
    if pol.is_easy and pol.easy_keep < 1.0:
        easy = [r for r in cur if pol.is_easy(r)]
        hard = [r for r in cur if not pol.is_easy(r)]
        easy.sort(key=lambda r: order_key(r, 0))
        n_keep = int(round(len(easy) * pol.easy_keep))
        covered = {pol.template_key(r) for r in hard} if pol.template_key else set()
        chosen: list[Mapping[str, Any]] = []
        if pol.template_key:                                             # coverage first: every distinct item keeps at least one row
            for r in easy:
                k = pol.template_key(r)
                if k not in covered:
                    covered.add(k)
                    chosen.append(r)
        ids = {id(r) for r in chosen}
        chosen += [r for r in easy if id(r) not in ids][:max(0, n_keep - len(chosen))]
        keep_ids = {id(r) for r in chosen}
        nxt = hard + [r for r in easy if id(r) in keep_ids]
        reserve += [r for r in easy if id(r) not in keep_ids]
        _step(rep, "easy", cur, nxt, f"rows the code tools already solve kept at {pol.easy_keep:.0%} (and one per distinct item); all hard rows kept")
        cur = nxt
    # balance to the budget
    if pol.row_cap and len(cur) > pol.row_cap:
        keep_rows, rest = _round_robin(cur, pol.group_key or (lambda r: ""), pol.row_cap)
        reserve += rest
        _step(rep, "balance", cur, keep_rows, f"round-robin over groups to the {pol.row_cap}-row budget")
        cur = list(keep_rows)
    # near duplicates (last: only the survivors are hashed)
    if pol.near_dup and cur:
        from creator import trainmix as TM
        idx = TM.NearIndex()
        nxt = []
        for i, r in enumerate(cur):
            txt = user_text(r)
            if len(txt) < 40 or idx.match(txt) is None:
                idx.add(str(i), txt)
                nxt.append(r)
        _step(rep, "near", cur, nxt, f"near-duplicates (MinHash >= {idx.t})")
        cur = nxt
    toks_out = [tokens(r) for r in cur]
    return {"kept": cur, "reserve": reserve, "report": rep, "policy": pol.name,
            "tokens_before": rep[0]["tokens_in"], "tokens_after": round(sum(toks_out)), "rows_before": len(rows), "rows_after": len(cur),
            "len": {"mean": round(sum(toks_out) / max(1, len(cur)), 1), "p90": round(percentile(toks_out, 0.9), 1), "p99": round(percentile(toks_out, 0.99), 1),
                    "max": round(max(toks_out, default=0.0), 1)}}


# ------------------------------------------------------------------------------------------------ the Phase 2 policies
def _meta(r: Mapping[str, Any], k: str) -> str:
    return str((r.get("meta") or {}).get(k, ""))


def stem_key(r: Mapping[str, Any]) -> str:
    """The question template of a multiple-choice row: the text before the options with names, numbers and paths masked."""
    import re
    s = user_text(r).split("\nA. ")[0]
    s = re.sub(r"'[^']*'", "'#'", s)
    s = re.sub(r"`[^`]*`", "`#`", s)
    s = re.sub(r"\d[\d\-:. ]*", "#", s)
    return s[:90]


def coarse_stem(r: Mapping[str, Any]) -> str:
    """stem_key without the repository and cut to 60 characters: the question FAMILY (e.g. 'which of four commits came first')."""
    import re
    s = re.sub(r"Repository '[^']*'\.?\s*", "", user_text(r).split("\nA. ")[0])
    s = re.sub(r"'[^']*'", "'#'", s)
    s = re.sub(r"`[^`]*`", "`#`", s)
    return re.sub(r"\d[\d\-:. ]*", "#", s)[:60]


def repo_key(r: Mapping[str, Any]) -> str:
    import re
    m = re.search(r"Repository '([^']*)'", user_text(r))
    return m.group(1) if m else ""


def is_plain_answer(r: Mapping[str, Any]) -> float:
    return 1.0 if len(r["messages"][-1].get("content", "")) < 40 else 0.0


def phase2_policies(budgets: Optional[Mapping[str, int]] = None) -> dict[str, Policy]:
    """One policy per source. `budgets` (rows) come from the measured plateau of the family's dev loss (creator.gpueff.plateau_fraction)."""
    b = dict(budgets or {})
    return {
        "pinpoint": Policy("pinpoint", template_key=lambda r: f"{_meta(r, 'file')}:{_meta(r, 'line')}:{_meta(r, 'mutation')}", max_per_template=2, near_dup=False,
                           is_easy=lambda r: str(r["messages"][-1].get("content", "")).strip() == "1", easy_keep=0.35,
                           group_key=lambda r: _meta(r, "file") + "|" + _meta(r, "mutation"), row_cap=b.get("pinpoint", 0)),
        "debug_fix": Policy("debug_fix", template_key=lambda r: f"{_meta(r, 'file')}:{_meta(r, 'line')}:{_meta(r, 'mutation')}", max_per_template=1, near_dup=False,
                            group_key=lambda r: _meta(r, "file") + "|" + _meta(r, "mutation"), row_cap=b.get("debug_fix", 0)),
        "code": Policy("code", group_key=lambda r: _meta(r, "group")[:12], row_cap=b.get("code", 0)),
        "coderonly": Policy("coderonly", len_cap=1800.0, group_key=lambda r: r["messages"][0].get("content", "")[:30], row_cap=b.get("coderonly", 0)),
        "calib": Policy("calib"),                                          # dedupe + length only: the answer distribution is the signal
        "judge": Policy("judge"),
        "thinker": Policy("thinker", variant_key=user_text, variant_pref=is_plain_answer, template_key=coarse_stem, max_per_template=b.get("thinker_per_template", 2500),
                          reserve_template=True, group_key=lambda r: stem_key(r) + "|" + repo_key(r), row_cap=b.get("thinker", 0), near_dup=False),
    }


def iter_jsonl(p: Path) -> Iterable[dict[str, Any]]:
    with open(p, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                try:
                    yield json.loads(ln)
                except ValueError:
                    continue


def write_mix(out: Path, name: str, kept: Sequence[Mapping[str, Any]], dev: Sequence[Mapping[str, Any]], reserve: Sequence[Mapping[str, Any]],
              report: Mapping[str, Any], *, size_tag: str = "") -> dict[str, Any]:
    """A trainmix-shaped directory (MANIFEST.json + train.jsonl + dev.jsonl) plus reserve.jsonl and the curation report."""
    d = Path(out) / name
    d.mkdir(parents=True, exist_ok=True)
    for fn, rows in (("train.jsonl", kept), ("dev.jsonl", dev), ("reserve.jsonl", reserve)):
        with open(d / fn, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
    with open(d / "ids.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in kept:
            f.write(json.dumps({"id": r.get("id", ""), "source": r.get("source", ""), "kind": r.get("kind", "")}) + "\n")
    man = {"target": name, "rows": {"train": len(kept), "dev": len(dev), "pref": 0, "reserve": len(reserve)},
           "tokens_train": round(sum(tokens(r) for r in kept)), "curation": report, "size": size_tag}
    (d / "MANIFEST.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    return man


def _dev_pick(rows: Sequence[Mapping[str, Any]], cap: int, key: Callable[[Mapping[str, Any]], str]) -> list[Mapping[str, Any]]:
    """A small dev set (eval time is GPU time): the same round-robin over groups, deterministic."""
    return _round_robin(rows, key, cap)[0] if len(rows) > cap else list(rows)


def build_phase2_mixes(out: Path, gpuday: Path, exclusions: Any = None, budgets: Optional[Mapping[str, int]] = None) -> dict[str, Any]:
    """Curate every Phase 2 source into a trainmix-shaped directory under `out` (name '<mix>.c') and return the per-module token accounting.
    Sources: <gpuday>/trainmix/<mix>/{train,dev}.jsonl and <gpuday>/phase2_data/all/{pinpoint,debug_fix,code}.jsonl (split 'train' / 'eval').
    Honours the eval-suite exclusion list (creator.tools.evalexclude, EXCLUDE.json) on train AND dev rows."""
    if exclusions is None:
        from creator.tools import evalexclude as EX
        exclusions = EX.load()
    pol = phase2_policies(budgets)
    tm, pd = Path(gpuday) / "trainmix", Path(gpuday) / "phase2_data" / "all"
    report: dict[str, Any] = {}

    def screen(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        if not exclusions:
            return list(rows)
        return [r for r in rows if not exclusions.violation(user_text(r) + "\n" + (r["messages"][-1].get("content", "") if r.get("messages") else ""), str(r.get("id") or ""))]

    def one(name: str, policy: str, train: Sequence[Mapping[str, Any]], dev: Sequence[Mapping[str, Any]], dev_cap: int, dev_key: Callable[[Mapping[str, Any]], str]) -> None:
        c = curate(train, pol[policy], exclusions)
        dv = _dev_pick(screen(dev), dev_cap, dev_key)
        man = write_mix(out, name, c["kept"], dv, c["reserve"], {k: c[k] for k in ("report", "policy", "tokens_before", "tokens_after", "rows_before", "rows_after", "len")})
        report[name] = {"rows_before": c["rows_before"], "rows_after": c["rows_after"], "tokens_before": c["tokens_before"], "tokens_after": c["tokens_after"],
                        "reserve_rows": len(c["reserve"]), "dev_rows": len(dv), "len": c["len"], "steps": c["report"], "manifest_rows": man["rows"]}

    for name, src, policy, dev_cap in (("calib_17b.c", "calib_17b", "calib", 1000), ("judge_06b.c", "judge_06b", "judge", 300),
                                       ("coderonly_17b.c", "coderonly_17b", "coderonly", 344), ("thinker_17b.c", "thinker_17b", "thinker", 1200)):
        one(name, policy, list(iter_jsonl(tm / src / "train.jsonl")), list(iter_jsonl(tm / src / "dev.jsonl")), dev_cap,
            stem_key if policy == "thinker" else (lambda r: ""))
    for name, src, policy, dev_cap in (("pinpoint.c", "pinpoint", "pinpoint", 400), ("debug_fix.c", "debug_fix", "debug_fix", 300)):
        rows = list(iter_jsonl(pd / f"{src}.jsonl"))
        dev = [r for r in rows if r.get("split") == "eval"]
        dev_c = curate(dev, Policy(name + "-dev", template_key=pol[policy].template_key, max_per_template=1, near_dup=False), exclusions)["kept"]
        one(name, policy, [r for r in rows if r.get("split") == "train"], dev_c, dev_cap, lambda r: _meta(r, "file") + "|" + _meta(r, "mutation"))
    rows = [r for r in iter_jsonl(pd / "code.jsonl")]
    held = [r for r in rows if int(_h(_meta(r, "group"))[:4], 16) % 12 == 0]            # ~8% of the verified code rows by task group: the dev set
    held_ids = {id(r) for r in held}
    one("code_verified.c", "code", [r for r in rows if id(r) not in held_ids], held, 120, lambda r: "")
    report["_totals"] = {"tokens_before": sum(v["tokens_before"] for k, v in report.items() if k != "_totals"),
                         "tokens_after": sum(v["tokens_after"] for k, v in report.items() if k != "_totals")}
    return report
