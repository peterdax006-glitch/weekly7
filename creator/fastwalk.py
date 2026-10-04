"""Compiled walk-forward (owner, 3 Oct 2026: "feel free to install whatever will speed things up the most").

drillsources.walk_forward is the reference: an event loop (resolutions teach, creations predict) over per-key decayed counts. Pure Python spent
~180 us per item - fine for Nupen's own 850 commits, minutes per job once the public histories (hundreds of thousands of commits) are drilled.
This is the same arithmetic in the same order, compiled with numba: feature keys become integer ids, counts live in arrays. tests/test_fastwalk.py
holds it to the reference prediction by prediction. Without numba (or with NUPEN_FASTWALK=0) callers keep the reference.
"""
from __future__ import annotations

import math
import os
from typing import Any, Optional, Sequence

import numpy as np

numba: Any
try:
    import numba
    HAVE_NUMBA = True
except ImportError:                                                     # pragma: no cover - the reference loop stays available
    numba = None
    HAVE_NUMBA = False


def enabled() -> bool:
    return HAVE_NUMBA and os.environ.get("NUPEN_FASTWALK", "1") != "0"


def _core(ev_t: Any, ev_kind: Any, ev_i: Any, ys: Any, kstart: Any, kids: Any, nkeys: int,
          decay: float, k: float, logit: bool, cap: float) -> Any:
    """Events already in the reference order (time, kind: 0 create before 1 resolve, index). Returns per creation event:
    (item index, time, p, laplace base, last value)."""
    n_ev = ev_t.shape[0]
    cnt_n = np.zeros(nkeys)
    cnt_p = np.zeros(nkeys)
    seen = np.zeros(nkeys, dtype=np.bool_)
    out_i = np.empty(n_ev, dtype=np.int64)
    out_t = np.empty(n_ev)
    out_p = np.empty(n_ev)
    out_b = np.empty(n_ev)
    out_l = np.empty(n_ev)
    m = 0
    scale = 1.0
    gn = 0.0
    gp = 0.0
    hn = 0
    hpos = 0
    hlast = -1
    for e in range(n_ev):
        i = ev_i[e]
        if ev_kind[e] == 1:                                             # resolution: the model learns
            scale *= decay
            w = 1.0 / scale
            y = ys[i]
            for j in range(kstart[i], kstart[i + 1]):
                key = kids[j]
                seen[key] = True
                cnt_n[key] += w
                cnt_p[key] += w * y
            gn += w
            gp += w * y
            hn += 1
            hpos += y
            hlast = y
            if scale < 1e-100:                                          # renormalise exactly as the reference does
                for key in range(nkeys):
                    cnt_n[key] *= scale
                    cnt_p[key] *= scale
                gn *= scale
                gp *= scale
                scale = 1.0
        else:                                                           # creation: predict from what is resolved so far
            glob = (gp * scale + 1.0) / (gn * scale + 2.0)
            s = 0.0
            c = 0
            for j in range(kstart[i], kstart[i + 1]):
                key = kids[j]
                if seen[key]:
                    q = (cnt_p[key] * scale + k * glob) / (cnt_n[key] * scale + k)
                    if logit:
                        s += math.log(max(q, 1e-6) / max(1.0 - q, 1e-6))
                    else:
                        s += q
                    c += 1
            if c > 0 and logit:
                p = 1.0 / (1.0 + math.exp(-s / c))
            elif c > 0:
                p = s / c
            else:
                p = glob
            p = min(1.0 - cap, max(cap, p))
            out_i[m] = i
            out_t[m] = ev_t[e]
            out_p[m] = p
            out_b[m] = (hpos + 1.0) / (hn + 2.0)
            out_l[m] = 0.5 if hn == 0 else (0.75 if hlast == 1 else 0.25)
            m += 1
    return out_i[:m], out_t[:m], out_p[:m], out_b[:m], out_l[:m]


_compiled: Optional[Any] = None


def _kernel() -> Any:
    global _compiled
    if _compiled is None:
        _compiled = numba.njit(cache=True, nogil=True)(_core) if HAVE_NUMBA else _core
    return _compiled


def walk_forward(items: Sequence[Any], topic: str, decay: float = 0.97, k: float = 3.0, agg: str = "mean", cap: float = 0.02) -> list[Any]:
    """Same contract and result as drillsources.walk_forward (items with .keys/.created/.resolved/.y/.subject)."""
    from creator import thinking as T
    n = len(items)
    ids: dict[str, int] = {}
    kstart = np.zeros(n + 1, dtype=np.int64)
    flat: list[int] = []
    ys = np.zeros(n, dtype=np.int64)
    ev: list[tuple[float, int, int]] = []
    for i, it in enumerate(items):
        for key in it.keys:
            flat.append(ids.setdefault(key, len(ids)))
        kstart[i + 1] = len(flat)
        ys[i] = int(it.y)
        if it.resolved is not None:
            ev.append((it.created, 0, i))                               # create before resolve at equal time (h58 leak fix)
            ev.append((it.resolved, 1, i))
    ev.sort()                                                           # exactly the reference's tuple order
    if not ev:
        return []
    ev_t = np.array([e[0] for e in ev], dtype=np.float64)
    ev_kind = np.array([e[1] for e in ev], dtype=np.int64)
    ev_i = np.array([e[2] for e in ev], dtype=np.int64)
    oi, ot, op, ob, ol = _kernel()(ev_t, ev_kind, ev_i, ys, kstart, np.array(flat, dtype=np.int64), max(1, len(ids)),
                                   float(decay), float(k), agg == "logit", float(cap))
    return [T.Pred(topic, items[i].subject, float(t), float(p), float(b), float(lv), int(items[i].y))
            for i, t, p, b, lv in zip(oi.tolist(), ot.tolist(), op.tolist(), ob.tolist(), ol.tolist())]
