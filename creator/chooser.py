"""A TRAINABLE chooser for ActionStudent: a small listwise linear policy (pure numpy) over the candidate actions.

Lessons used to be only shown to the 1.5B model in its prompt; nothing trained on them. Here every row is (objective, the candidate
actions of one file, which of them was the good/bad choice) and the policy learns a score per candidate from hashed features:
the action kind, the objective's words and bigrams (candidate names masked as OWN/OTH so it generalises across names) crossed with
the kind and with structural facts about the target (how many functions use the import, callers/size of a function, reads of a
temporary), the lexical prior, the rich label's words and, when known, the language model's own pick. Training is deterministic
full-batch Adam on a softmax-over-candidates loss: positive rows pull the chosen action up, a measured-rejection row pushes the
action the student took down (cancelled / errored cycles carry no signal and are dropped). The kernel's measurement still decides
adoption; this module only chooses which exact, parse-checked action to try."""
from __future__ import annotations

import ast
import dataclasses
import json
import re
import zlib
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from creator import action_student as A
from creator import student as ST
from creator.curriculum import TEACHER, Lesson, is_skill_signal

DIM = 1 << 15
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "state/creator/chooser.json"
MAX_LINES = 1000                                    # larger files are skipped (explain() is quadratic)
MODEL_WEIGHT = 1.5                                  # logit bonus for the language model's own pick when it is known


def _h(*parts: object) -> int:
    return zlib.crc32("\x1f".join(str(p) for p in parts).encode("utf-8")) % DIM


def _tokens(text: str) -> list[str]:
    return [w for w in re.split(r"[^a-z0-9]+", text.lower()) if w]


def _names(a: A.Action) -> set[str]:
    return {v for k, v in a.params if k in ("alias", "name", "function", "variable", "param") and v}


def _mask(objective: str, own: set[str], others: set[str]) -> list[str]:
    """Objective tokens with candidate names replaced: whole-word matches only, so 'helper' survives as a word."""
    text = " " + objective.lower().replace("`", " ").replace("'", " ") + " "
    for nm, tag in [(n, " OWN ") for n in sorted(own, key=len, reverse=True)] + [(n, " OTH ") for n in sorted(others - own, key=len, reverse=True)]:
        text = re.sub(r"(?<![a-z0-9_])" + re.escape(nm.lower()) + r"(?![a-z0-9_])", tag, text)
    return _tokens(text.replace("OWN", " ownx ").replace("OTH", " othx "))


def _structure_ref(src: str, a: A.Action) -> list[str]:
    """The reference definition of `structure` (re-walks the AST per call; kept so a test proves the indexed version equal on real rows)."""
    tree = ST._parse(src)
    out: list[str] = []
    if tree is None:
        return out
    p = a.p
    funcs = {f.name: f for f in ast.walk(tree) if isinstance(f, ST.FuncT)}

    def refs(name: str) -> int:
        return sum(1 for x in ast.walk(tree) if isinstance(x, ast.Name) and x.id == name)

    if a.kind in ("lazy_import", "drop_unused_import"):
        nm = p.get("alias") or p.get("name") or ""
        users = [f.name for f in tree.body if isinstance(f, ST.FuncT) and any(isinstance(x, ast.Name) and x.id == nm for x in ast.walk(f))]
        top = any(isinstance(x, ast.Name) and x.id == nm for st in tree.body if not isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Import, ast.ImportFrom))
                  for x in ast.walk(st))
        out += [f"users={min(len(users), 3)}", f"module_use={int(top)}", f"used={int(bool(users) or top)}"]
    elif a.kind == "remove_unused":
        fn = funcs.get(p.get("name", ""))
        calls = refs(p.get("name", ""))
        size = (getattr(fn, "end_lineno", 0) or 0) - (getattr(fn, "lineno", 0) or 0) + 1 if fn else 0
        out += [f"refs={min(calls, 3)}", f"size={'small' if size <= 3 else 'mid' if size <= 8 else 'big'}",
                f"doc={int(bool(fn and ast.get_docstring(fn)))}"] + [f"nm:{t}" for t in _tokens(p.get("name", "").replace("_", " "))[:3]]
    elif a.kind == "inline_temp":
        fn = funcs.get(p.get("function", ""))
        reads = sum(1 for x in ast.walk(fn) if isinstance(x, ast.Name) and x.id == p.get("variable") and isinstance(x.ctx, ast.Load)) if fn else 0
        out += [f"reads={min(reads, 3)}", f"callers={min(refs(p.get('function', '')), 2)}"]
    else:
        out += ["guard"]
    return out


class _SrcIndex:
    """Everything `structure` reads from one source, from ONE parse (h62, 3 Oct 2026: the chooser retrain spent 1,184 of 1,515 s
    re-parsing and re-walking the same file's AST for every candidate in every one of its 7 fits). Only plain facts are kept (no AST), so
    many files fit in memory. Same answers as _structure_ref by construction: Name counts over ast.walk(tree), the last-wins funcs dict of
    ast.walk order, per-function size / docstring / Load-context Name counts, the same top-level user and module-use sets."""
    __slots__ = ("ok", "funcs", "refs", "body_users", "top_names")

    def __init__(self, src: str) -> None:
        tree = ST._parse(src)
        self.ok = tree is not None
        self.funcs: dict[str, tuple[int, bool, dict[str, int]]] = {}  # name -> (size in lines, has docstring, Load Name id counts)
        self.refs: dict[str, int] = {}
        self.body_users: list[tuple[str, set[str]]] = []           # top-level functions in body order: (name, Name ids inside)
        self.top_names: set[str] = set()                            # Name ids in top-level statements that are not defs or imports
        if tree is None:
            return
        fnodes: dict[str, Any] = {}
        for x in ast.walk(tree):
            if isinstance(x, ST.FuncT):
                fnodes[x.name] = x
            elif isinstance(x, ast.Name):
                self.refs[x.id] = self.refs.get(x.id, 0) + 1
        for name, fn in fnodes.items():
            loads: dict[str, int] = {}
            for x in ast.walk(fn):
                if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load):
                    loads[x.id] = loads.get(x.id, 0) + 1
            size = (getattr(fn, "end_lineno", 0) or 0) - (getattr(fn, "lineno", 0) or 0) + 1
            self.funcs[name] = (size, bool(ast.get_docstring(fn)), loads)
        for st in tree.body:
            if isinstance(st, ST.FuncT):
                self.body_users.append((st.name, {x.id for x in ast.walk(st) if isinstance(x, ast.Name)}))
            elif not isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Import, ast.ImportFrom)):
                self.top_names.update(x.id for x in ast.walk(st) if isinstance(x, ast.Name))


_INDEX: dict[str, _SrcIndex] = {}
_INDEX_MAX = 4096                                   # sources kept per process (a retrain sees ~1,400 distinct texts)


def _index(src: str) -> _SrcIndex:
    ix = _INDEX.get(src)
    if ix is None:
        if len(_INDEX) >= _INDEX_MAX:
            _INDEX.clear()
        ix = _INDEX[src] = _SrcIndex(src)
    return ix


def structure(src: str, a: A.Action) -> list[str]:
    """Structural facts about the target of `a` (computed from the AST of `src`; never from the objective). Identical to _structure_ref."""
    ix = _index(src)
    out: list[str] = []
    if not ix.ok:
        return out
    p = a.p
    if a.kind in ("lazy_import", "drop_unused_import"):
        nm = p.get("alias") or p.get("name") or ""
        users = [n for n, names in ix.body_users if nm in names]
        top = nm in ix.top_names
        out += [f"users={min(len(users), 3)}", f"module_use={int(top)}", f"used={int(bool(users) or top)}"]
    elif a.kind == "remove_unused":
        f = ix.funcs.get(p.get("name", ""))
        calls = ix.refs.get(p.get("name", ""), 0)
        size = f[0] if f else 0
        out += [f"refs={min(calls, 3)}", f"size={'small' if size <= 3 else 'mid' if size <= 8 else 'big'}",
                f"doc={int(bool(f and f[1]))}"] + [f"nm:{t}" for t in _tokens(p.get("name", "").replace("_", " "))[:3]]
    elif a.kind == "inline_temp":
        f = ix.funcs.get(p.get("function", ""))
        var = p.get("variable")
        reads = f[2].get(var, 0) if f and isinstance(var, str) else 0
        out += [f"reads={min(reads, 3)}", f"callers={min(ix.refs.get(p.get('function', ''), 0), 2)}"]
    else:
        out += ["guard"]
    return out


def featurize(objective: str, a: A.Action, src: str, cands: list[A.Action], lex: float = 0.0, model_pick: Optional[bool] = None) -> dict[int, float]:
    """Sparse hashed feature vector of one candidate in the context of its row."""
    f: dict[int, float] = {}

    def put(*parts: object, v: float = 1.0) -> None:
        i = _h(*parts)
        f[i] = f.get(i, 0.0) + v

    kind = a.kind
    own = _names(a)
    others = set().union(*[_names(c) for c in cands]) if cands else set()
    ws = _mask(objective, own, others)
    st = structure(src, a)
    put("kind", kind)
    put("nc", min(len(cands), 6), kind)
    put("lex", kind, v=lex / 3.0)
    put("lexpos", kind, v=1.0 if lex > 0 else 0.0)
    if "ownx" in ws:
        put("own", kind)
    for w in set(ws):
        put("w", w, kind)
        for s in st:
            put("x", w, s)
    for b in {f"{x} {y}" for x, y in zip(ws, ws[1:])}:
        put("b", b, kind)
    for s in st:
        put("st", s, kind)
    for w in set(_tokens(a.label(True))):
        put("lab", w)
    if model_pick is not None:
        put("model", v=1.0 if model_pick else 0.0)
    return f


# ------------------------------------------------------------------------------------------------ rows


@dataclasses.dataclass
class Row:
    """One decision: the candidates of a file, the indices that were chosen, +1 (good outcome) / -1 (measured bad), a weight."""
    objective: str
    src: str
    cands: list[A.Action]
    chosen: list[int]                  # 0-based
    sign: int = 1
    weight: float = 1.0
    source: str = ""


_CHOICE_SALT: Optional[str] = None


def _choice_salt() -> str:
    """Hash of the code that maps a lesson's file change onto the action space (h62: a retrain re-derived every lesson's candidates and its
    greedy explanation from scratch - 233 s of 1,515 - although they change only when a lesson or this code changes)."""
    global _CHOICE_SALT
    if _CHOICE_SALT is None:
        import hashlib
        h = hashlib.sha256()
        for f in (A.__file__, ST.__file__):
            try:
                h.update(Path(f).read_bytes())
            except OSError:
                h.update(b"?")
        h.update(b"|file_choice-v1")
        _CHOICE_SALT = h.hexdigest()[:24]
    return _CHOICE_SALT


def _file_choice(before: str, after: str, path: str) -> tuple[list[A.Action], list[int]]:
    """(candidates of `before`, indices of the actions that explain before -> after): disk-cached on (before, after, path) + the code hash."""
    from creator import diskcache as DC
    key = DC.key_of(path, before, after)
    hit = DC.get("chooser_choice", _choice_salt(), key)
    if isinstance(hit, tuple) and len(hit) == 2:
        return list(hit[0]), list(hit[1])
    cands = A.enumerate_actions(before, path)
    chosen, _ = A.explain(before, after, path)
    idx = [cands.index(a) for a in chosen if a in cands]
    DC.put("chooser_choice", _choice_salt(), key, (tuple(cands), tuple(idx)))
    return cands, idx


def lesson_rows(lessons: Iterable[Lesson], neg_weight: float = 0.5, teacher_negatives: bool = False) -> list[Row]:
    """Lessons -> rows. Adopted lessons (teacher or student) map their change onto the action space (positive rows); a MEASURED
    rejection of a student's attempt gives a negative row for the actions it took. adopted=None and cancelled/errored verdicts
    (curriculum.is_skill_signal False) carry no signal and produce nothing."""
    rows: list[Row] = []
    for les in lessons:
        if les.adopted is None or not is_skill_signal(les):
            continue
        if les.adopted is False and (les.solver in TEACHER) and not teacher_negatives:
            continue
        for p, after in sorted(les.files_after.items()):
            before = les.files_before.get(p)
            if not p.endswith(".py") or not before or len(before.splitlines()) > MAX_LINES:
                continue
            cands, idx = _file_choice(before, after, p)
            if cands and idx:
                rows.append(Row(les.objective, before, cands, idx, 1 if les.adopted else -1, 1.0 if les.adopted else neg_weight,
                                f"{'adopted' if les.adopted else 'rejected'}:{les.solver}"))
    return rows


PRACTICE_SOURCE = "prescreen"
PRACTICE_WEIGHT = 0.3                               # practice rows are real measurements but not kernel adoptions: a weaker, separate source


def practice_rows(path: Path, src_dir: Path, weight: float = PRACTICE_WEIGHT, neg_weight: float = 0.5,
                  exclude_paths: Iterable[str] = ()) -> list[Row]:
    """state/creator/practice_rows.jsonl (scripts/practice.py) -> rows, one decision per (target file, source, metric): the good
    candidates (measured delta < 0 and the direct tests pass) are a positive row, the measured-bad ones a negative row. These are
    SEPARATE from `lesson_rows` (the shadow switch rule trains and scores on real kernel outcomes only): source = 'prescreen'.
    Sources come from `src_dir/<src_hash>.py` (written once per file by the practice run)."""
    skip = set(exclude_paths)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for ln in lines:
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if r.get("source") != PRACTICE_SOURCE or r.get("path") in skip or r.get("inconclusive"):         # a test timeout is no outcome
            continue
        groups.setdefault((r["path"], r["src_hash"], r["metric"]), []).append(r)
    rows: list[Row] = []
    for (_, h, _), g in sorted(groups.items()):
        try:
            src = (Path(src_dir) / f"{h}.py").read_text(encoding="utf-8")
        except OSError:
            continue
        cands = [A.Action(x["action"]["kind"], tuple(sorted(x["action"]["params"].items())), x["action"]["path"]) for x in g]
        good = [i for i, x in enumerate(g) if x.get("good")]
        bad = [i for i, x in enumerate(g) if not x.get("good")]
        obj = str(g[0]["objective"])
        if good:
            rows.append(Row(obj, src, cands, good, 1, weight, PRACTICE_SOURCE))
        if bad:
            rows.append(Row(obj, src, cands, bad, -1, weight * neg_weight, PRACTICE_SOURCE))
    return rows


def practice_holdout(rows: list[Row], folds: int = 5, epochs: int = 200) -> dict[str, Any]:
    """Held-out accuracy by target FILE (a file's rows are never in its own training fold): on each decision that has at least one
    good and one bad candidate, does the chooser's top pick land on a candidate that is measured good (delta < 0, tests pass)?
    Compared with picking at random (the share of good candidates) and with always picking the first candidate."""
    pos = [r for r in rows if r.sign > 0]
    files = sorted({r.cands[0].path for r in rows if r.cands})
    fold_of = {f: i % folds for i, f in enumerate(files)}
    hit = rand = first = n = 0.0
    for k in range(folds):
        train = [r for r in rows if fold_of.get(r.cands[0].path) != k]
        test = [r for r in pos if fold_of.get(r.cands[0].path) == k and len(r.chosen) < len(r.cands)]
        if not test or not train:
            continue
        ch = Chooser().fit(train, epochs=epochs)
        for r in test:
            p = ch.pick(r.objective, r.cands, {c.path: r.src for c in r.cands})
            hit += p is not None and (p - 1) in r.chosen
            rand += len(r.chosen) / len(r.cands)
            first += 0 in r.chosen
            n += 1
    return {"decisions": int(n), "chooser_top1": hit / n if n else None, "random_top1": rand / n if n else None,
            "first_candidate_top1": first / n if n else None, "folds": folds, "train_rows": len(rows)}


# ------------------------------------------------------------------------------------------------ the policy


class Chooser:
    def __init__(self, w: Optional[np.ndarray] = None, meta: Optional[dict[str, Any]] = None) -> None:
        self.w = np.zeros(DIM) if w is None else w
        self.meta = meta or {}

    # -- scoring
    def scores(self, objective: str, cands: list[A.Action], srcs: dict[str, str], model_pick: Optional[int] = None,
               model_weight: float = MODEL_WEIGHT) -> list[float]:
        lex = A.lexical_scores(objective, cands)
        out: list[float] = []
        for i, a in enumerate(cands):
            fv = featurize(objective, a, srcs.get(a.path, next(iter(srcs.values()), "")), cands, lex[i], None)
            s = sum(self.w[k] * v for k, v in fv.items())
            if model_pick is not None and model_pick == i + 1:
                s += model_weight
            out.append(float(s))
        return out

    def pick(self, objective: str, cands: list[A.Action], srcs: dict[str, str], model_pick: Optional[int] = None,
             model_weight: float = MODEL_WEIGHT) -> Optional[int]:
        """1-based index of the best candidate (ties: the earliest); None for no candidates or an untrained chooser."""
        if not cands or not self.trained:
            return None
        sc = self.scores(objective, cands, srcs, model_pick, model_weight)
        return sc.index(max(sc)) + 1

    @property
    def trained(self) -> bool:
        return bool(np.any(self.w))

    # -- training
    def fit(self, rows: list[Row], epochs: int = 300, lr: float = 0.05, l2: float = 1e-3) -> "Chooser":
        """Deterministic full-batch Adam on the listwise softmax loss (negative rows: -log(1-p) of the taken action)."""
        idx_l: list[int] = []
        val_l: list[float] = []
        cid_l: list[int] = []
        starts: list[int] = []
        c = 0
        for r in rows:
            starts.append(c)
            lex = A.lexical_scores(r.objective, r.cands)
            for i, a in enumerate(r.cands):
                for k, v in sorted(featurize(r.objective, a, r.src, r.cands, lex[i]).items()):
                    idx_l.append(k)
                    val_l.append(v)
                    cid_l.append(c)
                c += 1
        idx, val, cid = np.array(idx_l, dtype=np.int64), np.array(val_l), np.array(cid_l, dtype=np.int64)
        w = np.zeros(DIM)
        m, v2 = np.zeros(DIM), np.zeros(DIM)
        n_rows = max(len(rows), 1)
        for t in range(1, epochs + 1):
            s = np.bincount(cid, weights=val * w[idx], minlength=c)
            g = np.zeros(c)
            for r, st in zip(rows, starts):
                n = len(r.cands)
                z = s[st:st + n]
                p = np.exp(z - z.max())
                p /= p.sum()
                if r.sign > 0:
                    tgt = np.zeros(n)
                    tgt[r.chosen] = 1.0 / len(r.chosen)
                    g[st:st + n] = r.weight * (p - tgt)
                else:
                    gg = np.zeros(n)
                    for i in r.chosen:
                        pi = min(float(p[i]), 0.99)
                        e = np.zeros(n)
                        e[i] = 1.0
                        gg += (pi / (1 - pi)) * (e - p) / len(r.chosen)
                    g[st:st + n] = r.weight * gg
            grad = np.bincount(idx, weights=val * g[cid], minlength=DIM) / n_rows + l2 * w
            m = 0.9 * m + 0.1 * grad
            v2 = 0.999 * v2 + 0.001 * grad * grad
            w = w - lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v2 / (1 - 0.999 ** t)) + 1e-8)
        self.w = w
        self.meta = {"rows": len(rows), "positive": sum(r.sign > 0 for r in rows), "negative": sum(r.sign < 0 for r in rows),
                     "epochs": epochs, "lr": lr, "l2": l2, "dim": DIM}
        return self

    # -- persistence
    def to_dict(self) -> dict[str, Any]:
        nz = np.nonzero(self.w)[0]
        return {"dim": DIM, "meta": self.meta, "index": [int(i) for i in nz], "weight": [round(float(self.w[i]), 8) for i in nz]}

    def save(self, path: Path = DEFAULT_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), sort_keys=True), encoding="utf-8", newline="\n")

    @staticmethod
    def load(path: Path = DEFAULT_PATH) -> Optional["Chooser"]:
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
            if int(d["dim"]) != DIM:
                return None
            w = np.zeros(DIM)
            w[np.array(d["index"], dtype=np.int64)] = np.array(d["weight"])
            return Chooser(w, d.get("meta", {}))
        except (OSError, ValueError, KeyError, TypeError):
            return None
