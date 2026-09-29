"""Bible Phase 33 (required reproducibility); serves the deterministic-experiment canon.

Run the same experiment twice and require identical: configuration hash, seed, candidate ordering, model configuration,
predictions, trades, metrics. Comparison is by exact byte hash - there is no tolerance. When two runs differ the module
does not stop at "differ": `diagnose` names the first differing element and classifies the cause (ordering only,
float noise, shape, NaN pattern, dtype, missing key) and `run_twice` deliberately poisons the global RNGs differently
before each run, so an experiment that leaks global random state fails here instead of in production.

An experiment is any callable  fn(cfg, seed) -> dict  of named artifacts (DataFrames, arrays, lists, dicts, scalars)."""
import copy
import hashlib
import json
import os
import random
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ("config_hash", "seed", "candidate_order", "model_config", "predictions", "trades", "metrics")


class ReproError(AssertionError):
    pass


# ------------------------------------------------------------------ canonical hashing
def _feed(h, o, path="root"):
    """Feed a type-tagged canonical byte encoding of `o` into hash `h`. Unsupported types raise: silently hashing
    repr() of an unknown object would make different runs look identical or identical runs look different."""
    if o is None:
        h.update(b"N")
    elif isinstance(o, (bool, np.bool_)):
        h.update(b"B1" if o else b"B0")
    elif isinstance(o, (int, np.integer)):
        h.update(b"I" + str(int(o)).encode() + b";")
    elif isinstance(o, (float, np.floating)):
        f = float(o)
        h.update(b"F" + (b"nan" if f != f else struct.pack("<d", f)))
    elif isinstance(o, str):
        b = o.encode("utf-8"); h.update(b"S" + str(len(b)).encode() + b":" + b)
    elif isinstance(o, bytes):
        h.update(b"Y" + str(len(o)).encode() + b":" + o)
    elif isinstance(o, (pd.Timestamp, np.datetime64)):
        h.update(b"T" + str(pd.Timestamp(o).value).encode() + b";")
    elif isinstance(o, Path):
        h.update(b"P"); h.update(file_hash(o).encode())
    elif isinstance(o, np.ndarray):
        h.update(b"A" + str(o.dtype).encode() + str(o.shape).encode())
        if o.dtype == object:
            for i, v in enumerate(o.ravel()):
                _feed(h, v, f"{path}[{i}]")
        else:
            a = np.ascontiguousarray(o)
            if a.dtype.kind == "f":                        # canonical NaN so equal arrays hash equal
                a = np.where(np.isnan(a), np.nan, a)
            h.update(a.tobytes())
    elif isinstance(o, pd.MultiIndex):
        h.update(b"MI" + str(o.nlevels).encode())
        for lv in range(o.nlevels):
            _feed(h, np.asarray(o.get_level_values(lv)), path)
        _feed(h, list(o.names), path)
    elif isinstance(o, pd.Index):
        h.update(b"IX"); _feed(h, np.asarray(o), path); _feed(h, o.name, path)
    elif isinstance(o, pd.Series):
        h.update(b"SR"); _feed(h, o.name, path); _feed(h, o.index, path); _feed(h, o.to_numpy(), path)
    elif isinstance(o, pd.DataFrame):
        h.update(b"DF"); _feed(h, list(map(str, o.columns)), path); _feed(h, o.index, path)
        for c in range(o.shape[1]):
            _feed(h, o.iloc[:, c].to_numpy(), f"{path}[:,{c}]")
    elif isinstance(o, dict):
        h.update(b"D" + str(len(o)).encode())
        for k in sorted(o, key=lambda k: (type(k).__name__, str(k))):
            _feed(h, k, path); _feed(h, o[k], f"{path}.{k}")
    elif isinstance(o, (list, tuple)):
        h.update((b"L" if isinstance(o, list) else b"U") + str(len(o)).encode())
        for i, v in enumerate(o):                          # order is part of the artifact (candidate ordering)
            _feed(h, v, f"{path}[{i}]")
    elif isinstance(o, (set, frozenset)):
        h.update(b"Z" + str(len(o)).encode())
        for d in sorted(artifact_hash(v) for v in o):
            h.update(d.encode())
    else:
        raise TypeError(f"cannot canonically hash {type(o).__name__} at {path}")


def artifact_hash(obj):
    h = hashlib.sha256()
    _feed(h, obj)
    return h.hexdigest()


def file_hash(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def hash_artifacts(art):
    return {k: artifact_hash(v) for k, v in art.items()}


def config_hash(cfg):
    """Same definition as engine.provenance.config_hash (kept local so this module imports nothing heavy)."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ------------------------------------------------------------------ diagnosis
def _first_diff(a, b, path="root"):
    """Return (path, description) of the first difference between two artifacts, or None."""
    if type(a) is not type(b) and not (isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number))):
        return path, f"type {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, pd.DataFrame):
        if list(a.columns) != list(b.columns):
            return path, f"columns differ: {list(a.columns)[:6]} vs {list(b.columns)[:6]}"
        if a.shape != b.shape:
            return path, f"shape {a.shape} vs {b.shape}"
        if not a.index.equals(b.index):
            return path, "index differs"
        for c in a.columns:
            r = _first_diff(a[c], b[c], f"{path}[{c!r}]")
            if r:
                return r
        return None
    if isinstance(a, pd.Series):
        if a.shape != b.shape:
            return path, f"length {len(a)} vs {len(b)}"
        if not a.index.equals(b.index):
            return path, "index differs"
        return _first_diff(a.to_numpy(), b.to_numpy(), path)
    if isinstance(a, np.ndarray):
        if a.shape != b.shape:
            return path, f"shape {a.shape} vs {b.shape}"
        if a.dtype != b.dtype:
            return path, f"dtype {a.dtype} vs {b.dtype}"
        if a.dtype.kind == "f":
            an, bn = np.isnan(a), np.isnan(b)
            if (an != bn).any():
                i = int(np.flatnonzero((an != bn).ravel())[0])
                return f"{path}[{i}]", "NaN pattern differs"
            ne = (a != b) & ~an
        else:
            ne = a != b
        if ne.any():
            i = int(np.flatnonzero(np.asarray(ne).ravel())[0])
            return f"{path}[{i}]", f"{a.ravel()[i]!r} vs {b.ravel()[i]!r}"
        return None
    if isinstance(a, dict):
        if set(a) != set(b):
            return path, f"keys differ: only-a {sorted(map(str, set(a) - set(b)))[:5]}, only-b {sorted(map(str, set(b) - set(a)))[:5]}"
        for k in sorted(a, key=str):
            r = _first_diff(a[k], b[k], f"{path}.{k}")
            if r:
                return r
        return None
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return path, f"length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            r = _first_diff(x, y, f"{path}[{i}]")
            if r:
                return r
        return None
    if isinstance(a, float) or isinstance(b, float):
        if (a != a) and (b != b):
            return None
        return None if a == b else (path, f"{a!r} vs {b!r}")
    return None if a == b else (path, f"{a!r} vs {b!r}")


def _numeric_leaves(o):
    if isinstance(o, pd.DataFrame):
        return o.select_dtypes("number").to_numpy(dtype=float).ravel()
    if isinstance(o, (pd.Series, np.ndarray)):
        a = np.asarray(o)
        return a.astype(float).ravel() if a.dtype.kind in "iuf" else None
    if isinstance(o, (list, tuple)) and o and all(isinstance(v, (int, float, np.number)) for v in o):
        return np.asarray(o, dtype=float)
    if isinstance(o, (int, float, np.number)) and not isinstance(o, bool):
        return np.asarray([o], dtype=float)
    if isinstance(o, dict):
        parts = [_numeric_leaves(o[k]) for k in sorted(o, key=str)]
        parts = [p for p in parts if p is not None]
        return np.concatenate(parts) if parts else None
    return None


def diagnose(name, a, b):
    """Classify why artifact `name` differs between two runs. Never says 'within tolerance': a difference is a bug."""
    if artifact_hash(a) == artifact_hash(b):
        return {"name": name, "same": True}
    where, what = _first_diff(a, b) or ("root", "hash differs but no element differs (dtype/index metadata)")
    causes = []
    if isinstance(a, (pd.DataFrame, pd.Series)) and isinstance(b, type(a)) and a.shape == b.shape:
        try:
            sa, sb = a.sort_index(), b.sort_index()
            if not a.index.equals(b.index) and artifact_hash(sa) == artifact_hash(sb):
                causes.append("ordering_only: identical rows in a different order")
        except Exception:
            pass
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b):
        try:
            if sorted(map(repr, a)) == sorted(map(repr, b)):
                causes.append("ordering_only: same candidates, different order (set/dict iteration or unstable sort?)")
        except Exception:
            pass
    na, nb = _numeric_leaves(a), _numeric_leaves(b)
    max_abs = rel = None
    if na is not None and nb is not None and na.shape == nb.shape and na.size:
        fin = np.isfinite(na) & np.isfinite(nb)
        if fin.any():
            d = np.abs(na[fin] - nb[fin])
            max_abs = float(d.max())
            rel = float((d / np.maximum(np.abs(na[fin]), 1e-300)).max())
            if 0 < rel < 1e-9:
                causes.append("float_noise: relative gap under 1e-9 - summation order, threading or BLAS differences")
            elif rel >= 1e-9:
                causes.append("value_change: real numeric difference - unseeded randomness or shared mutable state")
    if "shape" in what or "length" in what:
        causes.append("size_change")
    if "keys differ" in what:
        causes.append("structure_change")
    if "NaN pattern" in what:
        causes.append("nan_pattern")
    if "dtype" in what or "type " in what:
        causes.append("type_change")
    return {"name": name, "same": False, "first_difference": where, "detail": what, "causes": causes,
            "max_abs_diff": max_abs, "max_rel_diff": rel}


# ------------------------------------------------------------------ global RNG probes
def _poison(seed):
    np.random.seed(seed % (2 ** 32))
    random.seed(seed)


def uses_global_rng(fn, cfg, seed):
    """Run fn once and report which global generators it consumed (numpy legacy state, python `random`)."""
    _poison(123456)
    s0 = np.random.get_state(); p0 = random.getstate()
    fn(copy.deepcopy(cfg), seed)
    s1 = np.random.get_state(); p1 = random.getstate()
    out = []
    if s0[1].tobytes() != s1[1].tobytes() or s0[2] != s1[2]:
        out.append("numpy.random global state")
    if p0 != p1:
        out.append("python random global state")
    return out


# ------------------------------------------------------------------ run twice
def run_twice(fn, cfg, seed, required=REQUIRED, poison=True):
    """Execute fn(cfg, seed) twice on deep copies of cfg. Returns a report with `reproducible` (bool), per-artifact
    hashes and a diagnosis for each artifact that differs. With poison=True the global RNGs are seeded differently
    before each run so any reliance on them shows up as a difference."""
    cfg_hash0 = config_hash(cfg)
    runs = []
    for k in range(2):
        if poison:
            _poison(1000 + 7919 * k)
        c = copy.deepcopy(cfg)
        art = dict(fn(c, seed))
        mutated = config_hash(c) != cfg_hash0
        art.setdefault("config_hash", cfg_hash0)
        art.setdefault("seed", seed)
        runs.append((art, mutated))
    (a, ma), (b, mb) = runs
    rep = {"config_hash": cfg_hash0, "seed": seed, "artifacts": {}, "problems": [], "diagnoses": []}
    if ma or mb:
        rep["problems"].append("experiment mutated its own cfg (later runs then start from different settings)")
    for name in required:
        if name not in a or name not in b:
            rep["artifacts"][name] = {"same": False, "missing": True}
            rep["problems"].append(f"required artifact '{name}' not produced by run {'1' if name not in a else '2'}")
            continue
        ha, hb = artifact_hash(a[name]), artifact_hash(b[name])
        rep["artifacts"][name] = {"same": ha == hb, "hash": ha[:16], "hash_b": hb[:16]}
        if ha != hb:
            d = diagnose(name, a[name], b[name])
            rep["diagnoses"].append(d)
            rep["problems"].append(f"'{name}' differs at {d['first_difference']}: {d['detail']} {d['causes']}")
    for name in sorted((set(a) | set(b)) - set(required)):        # extras are checked too, but not mandatory
        if name in a and name in b:
            same = artifact_hash(a[name]) == artifact_hash(b[name])
            rep["artifacts"][name] = {"same": same}
            if not same:
                d = diagnose(name, a[name], b[name])
                rep["diagnoses"].append(d)
                rep["problems"].append(f"extra artifact '{name}' differs at {d['first_difference']}")
    try:
        rep["global_rng_use"] = uses_global_rng(fn, cfg, seed)
    except Exception as e:
        rep["global_rng_use"] = [f"probe failed: {e!r}"]
    if rep["global_rng_use"] and not any(s.startswith("probe") for s in rep["global_rng_use"]):
        rep["problems"].append("experiment consumes global RNG state: " + ", ".join(rep["global_rng_use"]))
    rep["reproducible"] = not rep["problems"]
    return rep


def assert_reproducible(fn, cfg, seed, **kw):
    rep = run_twice(fn, cfg, seed, **kw)
    if not rep["reproducible"]:
        raise ReproError("; ".join(rep["problems"]))
    return rep


# ------------------------------------------------------------------ fresh-process runs (PYTHONHASHSEED-sensitive bugs)
_CHILD = r"""
import sys, json, importlib
sys.path[:0] = json.loads(sys.argv[3])
mod, fn = sys.argv[1].split(":")
f = getattr(importlib.import_module(mod), fn)
from engine.repro import hash_artifacts
cfg = json.loads(sys.argv[2]); seed = int(sys.argv[4])
print("HASHES=" + json.dumps(hash_artifacts(f(cfg, seed))))
"""


def run_in_subprocess(target, cfg, seed, hashseeds=("0", "1"), pythonpath=(), timeout=300):
    """Run 'module:function' in fresh interpreters with different PYTHONHASHSEED values (set/dict-of-str iteration
    order changes with it) and compare artifact hashes. Catches nondeterminism an in-process rerun cannot see."""
    root = str(Path(__file__).resolve().parent.parent)
    outs = []
    for hs in hashseeds:
        env = {**os.environ, "PYTHONHASHSEED": hs, "PYTHONPATH": os.pathsep.join([root, *map(str, pythonpath)])}
        p = subprocess.run([sys.executable, "-c", _CHILD, target, json.dumps(cfg, default=str), json.dumps([root, *map(str, pythonpath)]),
                            str(seed)], capture_output=True, text=True, env=env, timeout=timeout)
        line = [ln for ln in p.stdout.splitlines() if ln.startswith("HASHES=")]
        if p.returncode != 0 or not line:
            raise ReproError(f"child failed (PYTHONHASHSEED={hs}): {p.stderr.strip()[-400:]}")
        outs.append(json.loads(line[0][7:]))
    diff = sorted({k for o in outs[1:] for k in set(o) | set(outs[0]) if o.get(k) != outs[0].get(k)})
    return {"hashseeds": list(hashseeds), "hashes": outs, "differs": diff, "reproducible": not diff}


# ------------------------------------------------------------------ manifests (cross-session reproducibility)
def write_manifest(art, path, cfg=None, seed=None):
    m = {"config_hash": config_hash(cfg) if cfg is not None else None, "seed": seed, "hashes": hash_artifacts(art)}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(m, indent=1, sort_keys=True), encoding="utf-8")
    return m


def check_manifest(art, path):
    """Compare artifacts with a manifest written by an earlier run. Returns the list of names that differ; a missing
    manifest file raises (an absent record must not read as 'nothing differs')."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no manifest at {p}")
    old = json.loads(p.read_text(encoding="utf-8"))["hashes"]
    new = hash_artifacts(art)
    return sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
