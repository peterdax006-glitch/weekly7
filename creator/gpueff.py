"""GPU efficiency model, built from REAL run records (P1.6; MASTER_BLUEPRINT 8.5a). Read-only over files outside the repository:

  runs      <outputs>/<pulse>/ft_*/gpuday/runs/ft_*/result.json   steps, seconds, steps/s, peak VRAM, batch/accum, flags, eval-loss curve
  mixes     <runtime>/gpuday/trainmix/<name>/MANIFEST.json + train.jsonl   rows, tokens, length distribution (mean/p90/p99/max)
  jobs      <runtime>/gpuday/trainmix/queue/logs/*.log   one JSON line per pod job: wall_s, rc, usd_spent_total

The table it learns: {model size, rows, mean/p90 length, epochs, batch, grad-ckpt, packing, grouping} -> {tok/s, peak VRAM, quality}.
What it answers: fit_rates() seconds = load + trained_tokens / rate(size); epoch_gain() what the 2nd epoch bought; peak_gb() VRAM for a
mix; settings() the fastest setting that kept quality (1 epoch where the 2nd epoch gave < 10% held-out loss; flags nobody has measured
stay at the proven values and are listed as pending, never assumed). Nothing here contacts a GPU."""
from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from creator import jobcost as JC

RUNTIME = Path.home() / "creator_runtime"
SIZES = ("0.6b", "1.7b", "4b")
SIZE_TAG = {"06b": "0.6b", "17b": "1.7b", "4b": "4b"}
PARAMS_B = {"0.6b": 0.6, "1.7b": 1.7, "4b": 4.0}
EPOCH_GAIN_MIN = 0.10             # a 2nd epoch must buy >= 10% held-out loss to be worth its time (measured: code mixes 1-8%, calib 36-51%)
THIN_STEPS = 200                  # thin-mix rule: fewer optimizer steps than this and the module does not go to the GPU
DEFAULT_BATCH = 16                # batch 4 x accum 4: the settings proven on the real GPU (SAFE MODE)


def outputs_root() -> Path:
    return RUNTIME / "gpu" / "outputs"


def mix_root() -> Path:
    return RUNTIME / "gpuday" / "trainmix"


# ------------------------------------------------------------------------------------------------ readers

def size_of(name: str) -> str:
    m = re.search(r"_(06b|17b|4b)(?:_|$)", name)
    return SIZE_TAG[m.group(1)] if m else ""


def length_stats(train: Path, chars_per_token: float = 3.6) -> dict[str, float]:
    """Token-length distribution of a mix's train.jsonl (chars / 3.6, the repo-wide estimate)."""
    L: list[float] = []
    try:
        with open(train, encoding="utf-8") as f:
            for ln in f:
                try:
                    m = json.loads(ln).get("messages") or []
                except ValueError:
                    continue
                L.append(sum(len(x.get("content", "")) for x in m) / chars_per_token)
    except OSError:
        return {}
    if not L:
        return {}
    L.sort()
    q = lambda p: L[min(len(L) - 1, int(p * len(L)))]               # noqa: E731
    return {"rows": len(L), "mean": round(statistics.mean(L), 1), "p90": round(q(0.9), 1), "p80": round(q(0.8), 1), "p99": round(q(0.99), 1), "max": round(L[-1], 1)}


def padded_tokens(trained_rows: float, stats: Optional[Mapping[str, float]], fallback_tokens: float = 0.0) -> float:
    """Tokens the GPU really processes: rows x the p80 length. Micro-batches of 4 are padded to their longest row, which averages near the
    80th percentile of the length distribution; measured tok/s is constant to +-6% per size in this unit (raw tokens: +-25%)."""
    p80 = (stats or {}).get("p80")
    return float(trained_rows) * float(p80) if p80 else fallback_tokens * 1.25


def load_runs(out_root: Optional[Path] = None, mixes: Optional[Path] = None, lengths: bool = True) -> list[dict[str, Any]]:
    """One dict per fine-tune result.json, joined with its mix manifest. trained_tokens = tokens_train x (steps x batch x accum / rows)."""
    out_root, mixes = Path(out_root or outputs_root()), Path(mixes or mix_root())
    runs: list[dict[str, Any]] = []
    for f in sorted(out_root.glob("*/ft_*/gpuday/runs/ft_*/result.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            s = d["sft"]
        except (OSError, ValueError, KeyError):
            continue
        name = f.parent.name[3:]
        man: dict[str, Any] = {}
        try:
            man = json.loads((mixes / name / "MANIFEST.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        rows = int(s.get("rows") or 0)
        bs = int(s.get("batch") or 4) * int(s.get("accum") or 4)
        steps = int(s.get("steps") or 0)
        epochs = steps * bs / rows if rows else 0.0
        tok_train = float((man.get("tokens_train") or 0))
        r: dict[str, Any] = {
            "name": name, "size": size_of(name) or {"Qwen/Qwen3-0.6B": "0.6b", "Qwen/Qwen3-1.7B": "1.7b", "Qwen/Qwen3-4B": "4b"}.get(s.get("base"), ""),
            "rows": rows, "steps": steps, "epochs": round(epochs, 2), "seconds": float(s.get("seconds") or 0), "steps_per_s": s.get("steps_per_s"),
            "peak_gb": s.get("peak_vram_gb"), "batch": s.get("batch"), "accum": s.get("accum"), "grad_ckpt": s.get("grad_ckpt"),
            "packing": s.get("packing"), "group_by_length": s.get("group_by_length"), "tokens_train": tok_train,
            "trained_tokens": tok_train * steps * bs / rows if rows else 0.0, "stopped_early": bool(s.get("stopped_early")),
            "eval_loss": [list(e) for e in s.get("eval_loss") or []], "merge_s": (d.get("merge") or {}).get("seconds"),
            "gguf_s": (d.get("gguf") or {}).get("seconds"), "lora_gguf_s": (d.get("lora_gguf") or {}).get("seconds"),
            "max_seq": man.get("max_seq"), "pred_train_min": ((man.get("gpu_minutes") or {}).get("train")),
        }
        if lengths:
            r["len"] = length_stats(mixes / name / "train.jsonl")
        r["padded_tokens"] = padded_tokens(steps * bs, r.get("len"), r["trained_tokens"])
        runs.append(r)
    return runs


_JOB = re.compile(r'"job":\s*"([^"]+)"')
_RC = re.compile(r'"rc":\s*(-?\d+)')
_WALL = re.compile(r'"wall_s":\s*([\d.]+)')
_BYTES = re.compile(r'"bytes":\s*(\d+)')
_USD = re.compile(r'"usd_spent_total":\s*([\d.]+)')


def parse_logs(logs: Optional[Path] = None) -> list[dict[str, Any]]:
    """The one-line job records of the queue logs (lines are cut at 600 chars, so fields are read by pattern, not by json)."""
    logs = Path(logs or (mix_root() / "queue" / "logs"))
    out: list[dict[str, Any]] = []
    for f in sorted(logs.glob("*.log")):
        try:
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for ln in lines:
            if not ln.startswith('{"job"'):
                continue
            j, rc, w, u = _JOB.search(ln), _RC.search(ln), _WALL.search(ln), _USD.search(ln)
            if j and w:
                out.append({"job": j.group(1).split(":", 1)[-1], "rc": int(rc.group(1)) if rc else 0, "wall_s": float(w.group(1)),
                            "usd_total": float(u.group(1)) if u else None, "skipped": '"skipped"' in ln, "log": f.name,
                            "bytes": int(b.group(1)) if (b := _BYTES.search(ln)) and "trainmix_upload" in j.group(1) else None})
    return out


def job_kind(name: str) -> str:
    """'roleeval_calib_17b_base' -> 'roleeval_base'; 'ft_judge_06b' -> 'ft_0.6b'; 'register_gen_calib_17b' -> 'register_gen'."""
    head = name.split("_", 1)[0]
    sz = size_of(name)
    if head in ("ft", "smoke"):
        return f"{head}_{sz or 'x'}"
    if head in ("roleeval", "talkeval"):
        v = name.rsplit("_", 1)[-1]
        return f"{head}_{v if v in ('base', 'generalist', 'tuned') else 'x'}"
    if name.startswith("register_gen"):
        return "register_gen"
    if name.startswith("stopgen"):
        return "stopgen"
    if name.startswith("trainmix_upload"):
        return "trainmix_upload"
    return head


def usd_per_hour(jobs: Sequence[Mapping[str, Any]]) -> Optional[float]:
    """Rented-hour price implied by the ledger: sum of usd_spent_total deltas / sum of the wall time of the jobs between them (per log)."""
    d_usd = d_s = 0.0
    prev: Optional[Mapping[str, Any]] = None
    for j in jobs:
        if prev is not None and prev["log"] == j["log"] and j["usd_total"] is not None and prev["usd_total"] is not None:
            du = j["usd_total"] - prev["usd_total"]
            if du > 0 and j["wall_s"] > 60:                           # long jobs only: short ones are rounded by the ledger
                d_usd += du
                d_s += j["wall_s"]
        prev = j
    return round(d_usd / d_s * 3600.0, 3) if d_s > 0 else None


def history(runs: Sequence[Mapping[str, Any]], jobs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Cost records for jobcost: every logged pod job (rc 0, not skipped) by kind, ft_* priced in trained tokens; plus the pure training
    time of each result.json as 'train_<size>' (units = trained tokens)."""
    by_name = {r["name"]: r for r in runs}
    recs: list[dict[str, Any]] = []
    for j in jobs:
        if j["rc"] != 0 or j["skipped"] or j["wall_s"] <= 0:
            continue
        k = job_kind(j["job"])
        units = None
        if k.startswith("ft_"):                                      # wall = pure training (train_<size>) + a roughly constant overhead
            run = by_name.get(j["job"][3:])
            if not run or not (run["seconds"] <= j["wall_s"] <= run["seconds"] * 3):
                continue                                              # a retried / resumed fine-tune whose result is not this wall time
            recs.append(JC.record("ft_overhead", j["wall_s"] - run["seconds"], None, "gpu", job=j["job"], run_s=run["seconds"]))
            continue
        if k == "trainmix_upload" and j.get("bytes"):
            units = j["bytes"]
        recs.append(JC.record(k, j["wall_s"], units, "gpu", job=j["job"]))
    base_wall: dict[str, list[float]] = {}                       # an eval of a tuned/generalist model costs ~ the base model's eval of the same role
    for j in jobs:
        if j["rc"] == 0 and j["job"].startswith("roleeval_") and j["job"].endswith("_base"):
            base_wall.setdefault(j["job"][len("roleeval_"):-5], []).append(j["wall_s"])
    for rec in recs:
        n = str(rec.get("job", ""))
        if rec["kind"] in ("roleeval_generalist", "roleeval_tuned") and n.rsplit("_", 1)[0][len("roleeval_"):] in base_wall:
            rec["units"] = JC._median(base_wall[n.rsplit("_", 1)[0][len("roleeval_"):]])
    for r in runs:
        if r["size"] and r["seconds"] > 0 and r["padded_tokens"] > 0:
            recs.append(JC.record(f"train_{r['size']}", r["seconds"], r["padded_tokens"], "gpu", job=r["name"]))
    return recs


# ------------------------------------------------------------------------------------------------ the efficiency model

def epoch_gain(run: Mapping[str, Any]) -> Optional[float]:
    """What the 2nd epoch bought: relative drop of held-out loss from the end of epoch 1 to the end. Needs a >= 1.9-epoch run."""
    ev = run.get("eval_loss") or []
    if run.get("epochs", 0) < 1.9 or len(ev) < 3 or run.get("stopped_early"):
        return None
    half = run["steps"] / run["epochs"]
    at = min(ev, key=lambda e: abs(e[0] - half))
    last = min(e[1] for e in ev if e[0] >= ev[-1][0] - 1) if ev else None
    return round((at[1] - last) / at[1], 4) if at[1] else None


class Eff:
    """rates (tok/s per size), load seconds, epoch gains, VRAM peaks - all from the runs it was built with."""

    def __init__(self, runs: Sequence[Mapping[str, Any]]):
        self.runs = list(runs)
        self.rate: dict[str, float] = {}
        self.load_s: dict[str, float] = {}
        for sz in SIZES:
            rs = [r for r in self.runs if r["size"] == sz and r["padded_tokens"] > 0 and r["seconds"] > 0]
            big = [r for r in rs if r["steps"] >= 100]
            pts = [(r["padded_tokens"], r["seconds"]) for r in rs]
            fit = JC._ols(pts) if len(pts) >= 3 else None
            if fit:
                self.load_s[sz], self.rate[sz] = fit[0], 1.0 / fit[1]
            elif big:
                self.rate[sz] = JC._median([r["padded_tokens"] / r["seconds"] for r in big])
                small = [r["seconds"] - r["padded_tokens"] / self.rate[sz] for r in rs if r["steps"] < 100]
                self.load_s[sz] = max(0.0, JC._median(small)) if small else 60.0
        for sz in SIZES:                                              # a size never run: scale the nearest measured one (cost ~ params^0.5)
            if sz not in self.rate and self.rate:
                ref = min(self.rate, key=lambda k: abs(PARAMS_B[k] - PARAMS_B[sz]))
                self.rate[sz] = self.rate[ref] * (PARAMS_B[ref] / PARAMS_B[sz]) ** 0.5
                self.load_s[sz] = self.load_s.get(ref, 60.0)

    def train_s(self, size: str, padded: float) -> float:
        return self.load_s.get(size, 60.0) + padded / self.rate[size]

    def gains(self) -> dict[str, float]:
        return {r["name"]: g for r in self.runs if (g := epoch_gain(r)) is not None}

    def peak_gb(self, size: str, max_tokens: float) -> dict[str, Any]:
        """VRAM peak (gradient checkpointing, batch 4 x 4). Measured peaks of the same size at a sequence length >= this mix's (the worst of
        them); with none, the smallest measured peak of any size scaled by sqrt of the weights ratio - an ESTIMATE, a smoke run measures it."""
        same = [r for r in self.runs if r["size"] == size and r.get("peak_gb") and (r.get("len") or {}).get("max")]
        up = [r for r in same if r["len"]["max"] >= max_tokens]
        if up:
            r = max(up, key=lambda x: x["peak_gb"])
            return {"gb": float(r["peak_gb"]), "basis": f"measured {r['name']} (max len {r['len']['max']:.0f} >= {max_tokens:.0f})"}
        if same:
            r = max(same, key=lambda x: x["peak_gb"])
            return {"gb": float(r["peak_gb"]), "basis": f"measured {r['name']} (longest seen {r['len']['max']:.0f} < {max_tokens:.0f}: lower bound)"}
        allr = [r for r in self.runs if r.get("peak_gb") and (r.get("len") or {}).get("max") and r["len"]["max"] >= max_tokens]
        if allr:
            r = min(allr, key=lambda x: x["peak_gb"])
            return {"gb": round(float(r["peak_gb"]) * math.sqrt(PARAMS_B[size] / PARAMS_B[r["size"]]), 2),
                    "basis": f"estimate from {r['name']} scaled by sqrt(params): smoke run to confirm"}
        return {"gb": 30.0, "basis": "no data: assume the whole card"}

    def validate(self, jobs: Optional[Sequence[Mapping[str, Any]]] = None) -> dict[str, Any]:
        """Leave-one-run-out error of train seconds, per size and overall, against the OLD trainmix.minutes() prediction."""
        errs, old = [], []
        for i, r in enumerate(self.runs):
            if not (r["size"] and r["seconds"] > 0 and r["padded_tokens"] > 0):
                continue
            rest = Eff(self.runs[:i] + self.runs[i + 1:])
            if r["size"] not in rest.rate:
                continue
            p = rest.train_s(r["size"], r["padded_tokens"])
            errs.append((r["name"], r["size"], round(p / r["seconds"] - 1, 3)))
            if r.get("pred_train_min"):
                old.append((r["name"], round(r["pred_train_min"] * 60 / (r["seconds"] + (r.get("lora_gguf_s") or 0)) - 1, 3)))
        ae = [abs(e[2]) for e in errs]
        wall = {j["job"][3:]: j["wall_s"] for j in (jobs or []) if j["job"].startswith("ft_") and j["rc"] == 0 and not j["skipped"]}
        ft = []
        for r in self.runs:                                                # whole fine-tune job: LOO training + LOO median overhead
            w = wall.get(r["name"])
            e = next((x for x in errs if x[0] == r["name"]), None)
            if w and e and r["seconds"] <= w <= r["seconds"] * 3:
                oh = [wall[o["name"]] - o["seconds"] for o in self.runs if o["name"] != r["name"] and o["name"] in wall
                      and o["seconds"] <= wall[o["name"]] <= o["seconds"] * 3]
                if oh:
                    pred = r["seconds"] * (1 + e[2]) + JC._median(oh)
                    ft.append((r["name"], round(pred / w - 1, 3)))
        fta = [abs(x[1]) for x in ft]
        return {"ft_total_loo": ft, "ft_total_median_abs_err": round(JC._median(fta), 3) if fta else None,
            "ft_total_within_30pct": sum(1 for x in fta if x <= 0.3), "loo": errs, "median_abs_err": round(JC._median(ae), 3) if ae else None, "within_30pct": sum(1 for e in ae if e <= 0.3), "n": len(ae),
                "old_model_signed_err": old, "old_median_abs_err": round(JC._median([abs(o[1]) for o in old]), 3) if old else None}


FAMILY_EPOCHS = {"code": 1, "pinpoint": 1, "thinker": 1, "calib": 2, "judge": 2, "checker": 2}      # measured gains decide below; these are the priors


def settings(eff: Eff, size: str, family: str, rows: int, tokens_train: float, stats: Mapping[str, float], *, nockpt_measured: bool = False) -> dict[str, Any]:
    """The fastest setting that kept quality, from what was measured. epochs: 1 where the family's 2nd epoch gave < 10%, else 2. batch 4 x 4,
    gradient checkpointing on, no packing, no length grouping (the only combination ever run on the GPU, 3-4 Oct); anything faster is listed
    under `pending` with the speed-plan's expectation and is applied by the scheduler only after a smoke run measured it."""
    gains = [g for r in eff.runs if (g := epoch_gain(r)) is not None if _family_of(r["name"]) == family]
    if gains:
        epochs = 2 if statistics.median(gains) >= EPOCH_GAIN_MIN else 1
        basis = f"measured 2nd-epoch gain median {statistics.median(gains):.1%} ({len(gains)} runs) -> {epochs} epoch(s)"
    else:
        epochs = FAMILY_EPOCHS.get(family, 1)
        basis = f"no measured 2nd-epoch gain for '{family}': family prior {epochs}"
    steps = math.ceil(rows / DEFAULT_BATCH) * epochs
    return {"epochs": epochs, "epochs_basis": basis, "batch": 4, "accum": 4, "grad_ckpt": not nockpt_measured, "packing": False,
            "group_by_length": False, "max_seq": int(min(4096, max(1024, 2 ** math.ceil(math.log2(max(2.0, stats.get("p99", 1024.0) * 1.1)))))),
            "steps": steps, "trained_tokens": tokens_train * epochs,
            "pending": [{"flag": "group_by_length", "expect": "~1.25x (padding ~25%)"}, {"flag": "no grad-ckpt at max_seq 2048", "expect": "~1.3x"},
                        {"flag": "flash-attn on the pod", "expect": "1.1-1.5x"}]}


def _family_of(name: str) -> str:
    n = name.lower()
    for fam, keys in (("calib", ("calib",)), ("judge", ("judge",)), ("pinpoint", ("pinpoint",)), ("thinker", ("thinker",)),
                      ("code", ("coder", "pipeline", "role_", "locate", "brevity", "promptbake"))):
        if any(k in n for k in keys):
            return fam
    return "other"


def build(out_root: Optional[Path] = None, mixes: Optional[Path] = None, logs: Optional[Path] = None) -> dict[str, Any]:
    runs = load_runs(out_root, mixes)
    jobs = parse_logs(logs)
    return {"runs": runs, "jobs": jobs, "eff": Eff(runs), "history": history(runs, jobs), "usd_h": usd_per_hour(jobs)}
