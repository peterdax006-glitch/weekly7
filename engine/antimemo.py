"""Bible Phase 11 - rerun anti-memorisation experiment (firewall against "learning the answer").

Exactly the Bible's six steps, in run_experiment():
  1 play window A with no lessons          2 post-mortem -> lessons (learner)
  3 rerun A twice-over: as is, and under a FRESH disguise (tickers renamed to opaque codes, every date shifted by a
    whole number of weeks; features and returns untouched)           4 measure improvement (paired by day)
  5 play unseen window B                   6 keep a lesson only when B is not harmed
A lesson that memorised A improves A but not disguised A (the same days under other names and dates), so the gap
`improvement_A - improvement_A_disguised` is the memorisation signature. A structural check backs it up:
invariance_check() applies the book to the panel and to a disguised copy and demands identical multipliers row by
row - any identity-, date- or ticker-keyed recall fails it and the whole book is rejected.

Learners plug in through a small interface (record/learn/items/subset/adjust/tick/factor), so the same firewall
audits LessonBook and anything else that claims to learn from a window. B is used to reject lessons, so it is
consumed as a selection set: the report says so, and a further window is needed for an unbiased final figure."""
import math

import numpy as np
import pandas as pd

from .lessons import LessonBook, post_mortem

DEFAULT_CFG = {"k": 5, "cost": 0.0005, "horizon_days": 7, "n_disguises": 2, "boot": 400, "block": 5,
               "harm_alpha": 0.2, "null_reps": 0, "long_only": False, "tol_harm": 0.0, "memo_ratio": 0.5, "memo_min_gap": 1e-4}


# ------------------------------------------------------------------ disguise
def disguise(X, y=None, seed=0, shift_weeks=(40, 400), kind="both"):
    """Fresh disguise: opaque ticker codes (a seeded random injection) and a whole-weeks date shift. Row order, feature
    values and returns are untouched, so an identity-free decision rule must behave exactly as before.
    kind: "both", "names" (tickers only) or "dates" (dates only) - to attribute a memoriser's failure to what it keyed on."""
    rng = np.random.default_rng(seed)
    tk = pd.Index(X.index.get_level_values(1).unique())
    codes = set()
    while len(codes) < len(tk):
        codes.add("Z" + "".join(rng.choice(list("0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"), 6)))
    codes = sorted(codes)
    codes = [codes[i] for i in rng.permutation(len(codes))]
    tmap = dict(zip(tk, codes)) if kind in ("both", "names") else {t: t for t in tk}
    shift = pd.Timedelta(days=7 * int(rng.integers(*shift_weeks))) if kind in ("both", "dates") else pd.Timedelta(0)

    def remap(idx):
        return pd.MultiIndex.from_arrays([idx.get_level_values(0) + shift,
                                          idx.get_level_values(1).map(tmap)], names=idx.names)

    X2 = X.copy()
    X2.index = remap(X.index)
    if y is None:
        return X2, None, tmap
    y2 = y.copy()
    y2.index = X2.index
    return X2, y2, tmap


# ------------------------------------------------------------------ play a window
def play_window(X, y, score_fn, book=None, k=5, cost=0.0005, horizon_days=7, collect_frame=False, long_only=False):
    """Play a window day by day. score_fn(Xday) -> Series on Xday.index (must be row-wise or cross-sectional within the
    day: it never sees another day). The book, if any, only re-weights that day's scores. The k largest |score| are
    taken, long if score>0 else short (long_only: the k largest score, always long), and settle at y less cost.
    Returns (daily pnl Series, candidate frame). The frame holds the BASE score, so a post-mortem judges the model, and
    `taken` marks what was actually picked (with the book, if one was given)."""
    groups = X.groupby(level=0, sort=True).indices
    pnl, frames = {}, []
    for d, pos in groups.items():
        Xd = X.iloc[pos]
        base = score_fn(Xd)
        s = book.adjust(base, Xd) if book is not None else base
        rank_key = s.to_numpy() if long_only else np.abs(s.to_numpy())
        take = np.zeros(len(s), bool)
        take[np.argsort(-rank_key, kind="stable")[:min(k, len(s))]] = True
        side = np.ones(len(s)) if long_only else np.where(s.to_numpy() >= 0, 1.0, -1.0)
        yd = y.iloc[pos].to_numpy()
        pnl[d] = float(np.mean(side[take] * yd[take] - cost)) if take.any() else 0.0
        if collect_frame:
            frames.append(pd.DataFrame({"score": base.to_numpy(), "y": yd, "taken": take, "side": side,
                                        "resolved": pd.Timestamp(d) + pd.Timedelta(days=horizon_days)}, index=Xd.index))
    daily = pd.Series(pnl).sort_index()
    return daily, (pd.concat(frames) if frames else pd.DataFrame())


# ------------------------------------------------------------------ statistics
def paired_delta(with_l, base, seed=0, boot=400, block=5):
    """Mean of the per-day paired difference, a moving-block bootstrap CI (blocks keep the overlap between adjacent
    days' horizons), and the one-sided bootstrap probability that the true mean is >= 0 (p_harm = P(mean >= 0) is
    small when the book truly hurts)."""
    d = (with_l - base).dropna().to_numpy()
    if len(d) == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "p_harm": 1.0, "n": 0}
    rng = np.random.default_rng(seed)
    b = max(1, min(block, len(d)))
    nb = math.ceil(len(d) / b)
    starts = rng.integers(0, len(d) - b + 1, size=(boot, nb))
    means = np.array([np.concatenate([d[s:s + b] for s in row])[:len(d)].mean() for row in starts])
    return {"mean": float(d.mean()), "lo": float(np.quantile(means, 0.05)), "hi": float(np.quantile(means, 0.95)),
            "p_harm": float((means >= 0).mean()), "n": len(d)}


def harmed(stat, cfg):
    """B is harmed when the mean is worse than -tol_harm AND the bootstrap makes a true harm likely (p_harm is the
    probability the true mean is >= 0). A mean loss beyond three times the tolerance counts on its own."""
    m, tol = stat["mean"], cfg["tol_harm"]
    if m >= -tol:
        return False
    return stat["p_harm"] < cfg["harm_alpha"] or (tol > 0 and m < -3 * tol)


# ------------------------------------------------------------------ breakdowns (per era, per stock type)
def tercile_labeller(ref, labels=("low", "mid", "high")):
    """Bucket function whose edges are fitted on `ref` (window A) and applied unchanged to later data, so a stock-type
    label never peeks at the window it labels."""
    edges = np.quantile(ref.dropna().to_numpy(), np.linspace(0, 1, len(labels) + 1)[1:-1])
    return lambda x: pd.Series(np.array(labels, dtype=object)[np.searchsorted(edges, x.to_numpy(float))], index=x.index)


def improvement_by_era(base, with_l, freq="YE"):
    """Mean daily/weekly pnl difference per calendar period, with the number of decision days in each."""
    d = (with_l - base).dropna()
    if d.empty:
        return pd.DataFrame(columns=["n", "improvement", "base", "with"])
    g = d.groupby(pd.Grouper(freq=freq))
    out = pd.DataFrame({"n": g.size(), "improvement": g.mean(),
                        "base": base.reindex(d.index).groupby(pd.Grouper(freq=freq)).mean(),
                        "with": with_l.reindex(d.index).groupby(pd.Grouper(freq=freq)).mean()})
    return out[out["n"] > 0]


def improvement_by_group(frame_base, frame_with, groups, cost=0.0005):
    """Per stock type: mean pnl of the picks made without / with the book, and how many picks the book changed.
    groups: Series on the frame index (a type label per row, e.g. a volatility tercile)."""
    rows = []
    for gname in sorted(set(groups.dropna())):
        idx = groups.index[groups == gname]
        b = frame_base.reindex(idx)
        w = frame_with.reindex(idx)
        pb = (b["side"] * b["y"] - cost)[b["taken"].fillna(False).astype(bool)]
        pw = (w["side"] * w["y"] - cost)[w["taken"].fillna(False).astype(bool)]
        rows.append({"type": gname, "rows": len(idx), "picks_base": len(pb), "picks_with": len(pw),
                     "mean_pick_pnl_base": float(pb.mean()) if len(pb) else np.nan,
                     "mean_pick_pnl_with": float(pw.mean()) if len(pw) else np.nan,
                     "picks_changed": int((b["taken"].fillna(False).astype(bool) != w["taken"].fillna(False).astype(bool)).sum())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ structural check
def invariance_check(book, X, seed=0, atol=1e-9):
    """Apply the book to X and to a disguised copy; the multipliers must match row for row. Returns (ok, max_abs_diff)."""
    X2, _, _ = disguise(X, None, seed=seed + 991)
    a = book.factor(X).to_numpy()
    b = book.factor(X2).to_numpy()
    diff = float(np.max(np.abs(a - b))) if len(a) else 0.0
    return diff <= atol, diff


def _default_learner(cfg, params=None, seed=0):
    def learn(frame, XA, now):
        book = LessonBook(params, seed)
        book.record(post_mortem(frame, XA, now, params, seed))
        book.learn()
        return book
    return learn


# ------------------------------------------------------------------ the experiment
def run_experiment(XA, yA, XB, yB, score_fn, cfg=None, seed=0, learner=None, params=None, enforce_invariance=True,
                   type_fn=None, XC=None, yC=None):
    """Bible Phase 11. `learner(frame, XA, now) -> book` (default: post_mortem + LessonBook.learn). Returns the
    required outputs: improvement_A, improvement_A_disguised, improvement_B, degradation_B, lesson_count,
    accepted_lessons, rejected_lessons, plus memorisation_gap, identity_invariant, per-lesson evidence and the
    firewall's final book (only lessons that passed), and, for the kept book on B, improvement per era and (when
    type_fn(X) -> label Series is given) per stock type. cfg["long_only"] plays long-only.
    Optional window C (XC, yC) is never used for any decision: the kept book is scored there once, which is the
    unbiased figure B cannot give. cfg["null_reps"] > 0 adds optimism_control(): what the same learner "gains" on A
    when the returns are shuffled within each day (pure overfit)."""
    c = {**DEFAULT_CFG, **(cfg or {})}
    kw = dict(k=c["k"], cost=c["cost"], horizon_days=c["horizon_days"], long_only=c["long_only"])
    bs = dict(boot=c["boot"], block=c["block"])
    learner = learner or _default_learner(c, params, seed)

    baseA, frame = play_window(XA, yA, score_fn, None, collect_frame=True, **kw)             # 1
    now = frame["resolved"].max() if len(frame) else pd.Timestamp("1970-01-01")
    book = learner(frame, XA, now)                                                           # 2
    dis = [disguise(XA, yA, seed=seed * 1000 + i) for i in range(c["n_disguises"])]          # 3 (fresh per rerun)
    dis_base = [play_window(Xd, yd, score_fn, None, **kw)[0] for Xd, yd, _ in dis]
    baseB, _ = play_window(XB, yB, score_fn, None, **kw)                                     # 5

    def evaluate(bk, tag):
        wa = play_window(XA, yA, score_fn, bk, **kw)[0]
        wd = [play_window(Xd, yd, score_fn, bk, **kw)[0] for Xd, yd, _ in dis]
        wb = play_window(XB, yB, score_fn, bk, **kw)[0]
        sa = paired_delta(wa, baseA, seed + 1, **bs)
        sd = [paired_delta(w, b, seed + 2 + i, **bs) for i, (w, b) in enumerate(zip(wd, dis_base))]
        sb = paired_delta(wb, baseB, seed + 9, **bs)
        return {"A": sa, "A_dis": {"mean": float(np.mean([s["mean"] for s in sd])),
                                   "min": float(min(s["mean"] for s in sd))}, "B": sb}                 # 4

    ids = book.items()
    invariant, inv_diff = invariance_check(book, XA, seed) if ids else (True, 0.0)
    full = evaluate(book, "full") if ids else None
    res = {"lesson_count": len(ids), "identity_invariant": bool(invariant), "invariance_max_diff": inv_diff,
           "improvement_A": full["A"]["mean"] if full else 0.0,
           "improvement_A_disguised": full["A_dis"]["mean"] if full else 0.0,
           "improvement_B": full["B"]["mean"] if full else 0.0,
           "degradation_B": max(0.0, -full["B"]["mean"]) if full else 0.0,
           "baseline_A": float(baseA.mean()), "baseline_B": float(baseB.mean()),
           "b_selection_note": "B is used to reject lessons; holdout_C (when a window C is given) is the unbiased figure"}
    res["memorisation_gap"] = res["improvement_A"] - res["improvement_A_disguised"]
    if ids:                                          # which identity did it key on? rename-only vs shift-only reruns
        for kind in ("names", "dates"):
            Xk, yk, _ = disguise(XA, yA, seed=seed * 1000 + 77, kind=kind)
            wk_ = play_window(Xk, yk, score_fn, book, **kw)[0]
            bk_ = play_window(Xk, yk, score_fn, None, **kw)[0]
            res[f"improvement_A_{kind}_only"] = float((wk_ - bk_).mean())
    for attr, key in (("summary", "book_summary"), ("category_counts", "episode_categories")):
        if hasattr(book, attr):
            res[key] = getattr(book, attr)()
    res["mining_rejections"] = dict(getattr(book, "rejected_mining", {}))
    accepted, rejected, per = [], [], {}
    for lid in ids:
        ev = evaluate(book.subset([lid]), lid)
        per[lid] = ev
        iA, iD, iB = ev["A"]["mean"], ev["A_dis"]["min"], ev["B"]["mean"]
        if enforce_invariance and not invariant:
            reason = "identity_keyed"
        elif iA > c["memo_min_gap"] and iA - iD > c["memo_min_gap"] and iD < c["memo_ratio"] * iA:
            reason = "memoriser"                 # improves A, not A under a disguise (most informative reason first)
        elif harmed(ev["B"], c):
            reason = "harms_B"
        elif iA <= 0:
            reason = "no_gain_A"
        elif iB < 0:
            reason = "no_support_B"
        else:
            accepted.append(lid)
            continue
        rejected.append({"lid": lid, "reason": reason, "A": iA, "A_disguised": iD, "B": iB})
    keep = list(accepted)
    while keep:                                  # the kept set as a whole must not harm B either
        fin = evaluate(book.subset(keep), "final")
        if not harmed(fin["B"], c):
            break
        worst = min(keep, key=lambda i: per[i]["B"]["mean"])
        keep.remove(worst)
        rejected.append({"lid": worst, "reason": "combined_harms_B", "A": per[worst]["A"]["mean"],
                         "A_disguised": per[worst]["A_dis"]["min"], "B": per[worst]["B"]["mean"]})
    final = evaluate(book.subset(keep), "final") if keep else None
    if keep:
        wB, fB = play_window(XB, yB, score_fn, book.subset(keep), collect_frame=True, **kw)
        era = improvement_by_era(baseB, wB)
        era.index = era.index.strftime("%Y")
        res["era_B"] = era.round(6).to_dict("index")
        if type_fn is not None:
            _, f0 = play_window(XB, yB, score_fn, None, collect_frame=True, **kw)
            res["type_B"] = improvement_by_group(f0, fB, type_fn(XB), c["cost"]).round(6).to_dict("records")
    if XC is not None and len(XC):
        bc_ = play_window(XC, yC, score_fn, None, **kw)[0]
        wc_ = play_window(XC, yC, score_fn, book.subset(keep) if keep else None, **kw)[0]
        res["holdout_C"] = paired_delta(wc_, bc_, seed + 13, **bs)
        res["holdout_C"]["harmed"] = bool(keep and harmed(res["holdout_C"], c))
    if c["null_reps"]:
        res["null_control"] = optimism_control(XA, yA, score_fn, c, seed, res["improvement_A"], params)
    res.update({"accepted_lessons": keep, "rejected_lessons": rejected, "per_lesson": per,
                "final_improvement_A": final["A"]["mean"] if final else 0.0,
                "final_improvement_A_disguised": final["A_dis"]["mean"] if final else 0.0,
                "final_improvement_B": final["B"]["mean"] if final else 0.0,
                "final_book": book.subset(keep) if hasattr(book, "subset") else None,
                "verdict": "kept" if keep else "none_kept"})
    return res


def optimism_control(XA, yA, score_fn, cfg, seed, real_improvement_A, params=None):
    """How much can this learner "improve" A when there is nothing to learn? Returns are shuffled across tickers within
    each day (the score and features keep their marginals, their link to the outcome is destroyed), lessons are learned
    and replayed on that same window exactly as for the real A. The real improvement must beat the null spread."""
    c = {**DEFAULT_CFG, **cfg}
    kw = dict(k=c["k"], cost=c["cost"], horizon_days=c["horizon_days"], long_only=c["long_only"])
    vals = []
    for i in range(c["null_reps"]):
        rng = np.random.default_rng(seed * 7919 + i)
        yn = yA.groupby(level=0, group_keys=False).transform(lambda v: rng.permutation(v.to_numpy()))
        base, fr = play_window(XA, yn, score_fn, None, collect_frame=True, **kw)
        book = LessonBook(params, seed + i)
        book.record(post_mortem(fr, XA, fr["resolved"].max(), params, seed + i))
        book.learn()
        w = play_window(XA, yn, score_fn, book, **kw)[0] if book.items() else base
        vals.append(float((w - base).mean()))
    v = np.array(vals)
    return {"null_improvements": vals, "null_mean": float(v.mean()), "null_max": float(v.max()),
            "real_improvement_A": float(real_improvement_A),
            "p_value": float((1 + (v >= real_improvement_A).sum()) / (len(v) + 1)),
            "exceeds_null": bool(real_improvement_A > v.max())}


# ------------------------------------------------------------------ Phase 11 on the real Test archive
# A "window" is what scripts/livesim_loop2.load_window returns: weekly snapshots (ticker x columns), closes, opens,
# cost, sector divisions. Lessons are applied by WRAPPING the snapshot scores before adaptive.replay sees them
# (engine/adaptive.py is untouched): mu_raw is replaced by score_percentile * lesson_factor, and the policy re-ranks it.
def snap_features(snap):
    """Numeric features of one weekly snapshot, plus score = percentile rank of the model's mu_raw."""
    f = snap.select_dtypes("number").drop(columns=["mu_raw"], errors="ignore").astype("float64")
    f["score"] = snap["mu_raw"].rank(pct=True).astype("float64")
    return f


def window_panel(w, horizon=5):
    """(X, Y) for one window. X: features on a (date, ticker) index. Y: y = close(t+horizon)/open(t+1) - 1 (bought at the
    next session's open, canon C33), mfe / mae = best / worst close on the way. NaN where the window ends first."""
    C = w["closes"].astype("float64")
    O = (w["opens"] if w["opens"] is not None else w["closes"]).astype("float64")
    sess = C.index
    xs, ys = [], []
    for ds, snap in sorted(w["snaps"].items()):
        d = pd.Timestamp(ds)
        f = snap_features(snap)
        f.index = pd.MultiIndex.from_arrays([[d] * len(f), f.index], names=["date", "ticker"])
        i = int(sess.searchsorted(d))
        tk = snap.index
        if i + horizon < len(sess):
            entry = O.iloc[i + 1].reindex(tk)
            path = C.iloc[i + 1:i + horizon + 1].reindex(columns=tk)
            y = pd.DataFrame({"y": path.iloc[-1] / entry - 1, "mfe": (path.max() / entry - 1).clip(lower=0),
                              "mae": (path.min() / entry - 1).clip(upper=0)})
        else:
            y = pd.DataFrame(np.nan, index=tk, columns=["y", "mfe", "mae"])
        y.index = f.index
        xs.append(f)
        ys.append(y)
    return pd.concat(xs), pd.concat(ys)


def archive_frame(X, Y, decisions, resolve_days=8):
    """Candidate frame for post_mortem: base score, outcome, and whether the REAL replayed system held the name
    (decisions: Session.decisions, a list of (date string, tickers))."""
    held = {pd.Timestamp(d): set(t) for d, t in decisions}
    idx = X.index
    taken = np.array([tk in held.get(d, ()) for d, tk in idx])
    fr = pd.DataFrame({"score": X["score"].to_numpy(), "y": Y["y"].to_numpy(), "mfe": Y["mfe"].to_numpy(),
                       "mae": Y["mae"].to_numpy(), "taken": taken, "side": 1.0,
                       "resolved": idx.get_level_values(0) + pd.Timedelta(days=resolve_days)}, index=idx)
    return fr[fr["y"].notna()]


def wrap_snaps(snaps, book, X):
    """Snapshots whose mu_raw is score-percentile x lesson factor (factor 1 when book is None, which leaves the policy's
    ordering exactly as before). X: the window's panel from window_panel."""
    out = {}
    by_date = X.groupby(level=0).indices
    for ds, snap in snaps.items():
        d = pd.Timestamp(ds)
        Xd = X.iloc[by_date[d]]
        fac = book.factor(Xd).to_numpy() if book is not None else 1.0
        s = snap.copy()
        s["mu_raw"] = Xd["score"].to_numpy() * fac
        out[ds] = s
    return out


def disguise_window(w, seed=0, shift_weeks=(40, 400)):
    """A fresh disguise of a whole archived window: every ticker renamed to an opaque code (snapshots, closes, opens and
    the sector map), every date shifted by a whole number of weeks. Nothing else changes."""
    rng = np.random.default_rng(seed)
    tk = sorted(set(w["closes"].columns) | {t for s in w["snaps"].values() for t in s.index})
    alphabet = list("0123456789ABCDEFGHJKLMNPQRSTUVWXYZ")
    codes = set()
    while len(codes) < len(tk):
        codes.add("Z" + "".join(rng.choice(alphabet, 6)))
    codes = sorted(codes)
    tmap = dict(zip(tk, [codes[i] for i in rng.permutation(len(codes))]))
    shift = pd.Timedelta(days=7 * int(rng.integers(*shift_weeks)))
    def cols(df):
        df = df.rename(columns=tmap)
        df.index = df.index + shift
        return df
    return {**w, "id": f"{w['id']}~dis{seed}", "closes": cols(w["closes"]),
            "opens": cols(w["opens"]) if w["opens"] is not None else None,
            "snaps": {str((pd.Timestamp(k) + shift).date()): s.rename(index=tmap) for k, s in w["snaps"].items()},
            "divs": {tmap.get(t, t): v for t, v in w["divs"].items()}}


class RecallControl:
    """POSITIVE CONTROL for the archive experiment: a deliberately memorising "lesson book" that boosts every
    (date, ticker) row that won in its training window. It must beat baseline on the window it memorised and fail
    everywhere its keys do not exist (other windows, renamed/shifted rerun) - proving the harness can see memorisation."""
    def __init__(self, frames, thresh=0.02, boost=50.0):
        self.keys = {i for fr in frames for i in fr.index[(fr["y"] > thresh).to_numpy()]}
        self.boost = boost

    def factor(self, X):
        return pd.Series([self.boost if i in self.keys else 1.0 for i in X.index], index=X.index)


def default_archive_learner(params=None, seed=0):
    """items: [{"episodes": [...], "frame": DataFrame}] -> a LessonBook mined by pnl and by mistake kind."""
    def learn(items):
        b = LessonBook(params, seed)
        for it in items:
            b.record(it["episodes"])
        b.learn()
        b.learn_by_kind()
        return b
    return learn


def recall_learner(items):
    return RecallControl([it["frame"] for it in items])


def _arm(w, X, book, cfg, meta, adaptive):
    from . import adaptive as A
    snaps = wrap_snaps(w["snaps"], book, X)
    S = A.replay(cfg, snaps, w["closes"], w["bps"], w["divs"], adaptive=adaptive, meta=meta, opens=w["opens"],
                 long_term=w["ltm"])
    return S


def _weeks(S):
    return np.asarray(S.weeks, float)


def _stat(diffs, seed, boot=400, block=4):
    """paired_delta on a plain array of weekly differences."""
    s = pd.Series(np.asarray(diffs, float))
    return paired_delta(s, s * 0.0, seed=seed, boot=boot, block=block)


def archive_experiment(windows, cfg, meta, adaptive=False, learner=None, params=None, folds=6, seed=0,
                       post_params=None, progress=None, parity_windows=2):
    """Phase 11 on real archived windows, through adaptive.replay. Per window:
       base  the window replayed with no lessons (also proves the wrapper alone changes nothing: parity)
       a     lessons learned from OTHER windows only (leave-one-fold-out)
       b     lessons learned from THIS window, replayed on it          <- the memorisation arm
       c     the same lessons replayed on this window disguised (tickers renamed, dates shifted) against its own base
    A rule that learned a pattern gains the same in b and c; a memoriser gains in b only. Report per window the weekly
    mean improvement of each arm, the memorisation gaps (b - c) and (b - a), and bootstrap CIs over weeks."""
    from . import adaptive as A
    learner = learner or default_archive_learner(params, seed)
    pp = {**{"untaken_frac": 0.03}, **(post_params or {})}
    say = progress or (lambda *_: None)
    stash = []
    for n, w in enumerate(windows):
        X, Y = window_panel(w)
        base = _arm(w, X, None, cfg, meta, adaptive)
        parity = None
        if n < parity_windows:
            raw = A.replay(cfg, w["snaps"], w["closes"], w["bps"], w["divs"], adaptive=adaptive, meta=meta,
                           opens=w["opens"], long_term=w["ltm"])
            parity = bool(np.allclose(_weeks(raw), _weeks(base)) and raw.decisions == base.decisions)
        fr = archive_frame(X, Y, base.decisions)
        eps = post_mortem(fr, X, fr["resolved"].max() if len(fr) else pd.Timestamp("1970-01-01"), pp, seed + n)
        stash.append({"w": w, "X": X, "base": base, "item": {"episodes": eps, "frame": fr}, "parity": parity})
        say(f"[{n + 1}/{len(windows)}] {w['id']} base {_weeks(base).mean():+.3%}/wk, {len(eps)} episodes"
            + ("" if parity is None else f", wrapper parity {parity}"))
    fold_of = [i % folds for i in range(len(stash))]
    book_a = {}
    for f in sorted(set(fold_of)):
        book_a[f] = learner([s["item"] for i, s in enumerate(stash) if fold_of[i] != f])
        say(f"fold {f}: other-windows book has {len(book_a[f].items()) if hasattr(book_a[f], 'items') else '?'} lessons")
    rows = []
    for i, s in enumerate(stash):
        w, X, base = s["w"], s["X"], s["base"]
        bb = learner([s["item"]])
        Sa = _arm(w, X, book_a[fold_of[i]], cfg, meta, adaptive)
        Sb = _arm(w, X, bb, cfg, meta, adaptive)
        w2 = disguise_window(w, seed=seed * 1000 + i)
        X2, _ = window_panel(w2)
        base2 = _arm(w2, X2, None, cfg, meta, adaptive)
        Sc = _arm(w2, X2, bb, cfg, meta, adaptive)
        wk = {k: _weeks(v) for k, v in dict(base=base, a=Sa, b=Sb, c_base=base2, c=Sc).items()}
        n_l = lambda bk: len(bk.items()) if hasattr(bk, "items") and callable(bk.items) else 0
        row = {"window": w["id"], "weeks": int(len(wk["base"])), "base": float(wk["base"].mean()),
               "gain_a": float((wk["a"] - wk["base"]).mean()), "gain_b": float((wk["b"] - wk["base"]).mean()),
               "gain_c": float((wk["c"] - wk["c_base"]).mean()),
               "lessons_a": n_l(book_a[fold_of[i]]), "lessons_b": n_l(bb),
               "picks_changed_a": sum(x != y for x, y in zip(Sa.decisions, base.decisions)),
               "picks_changed_b": sum(x != y for x, y in zip(Sb.decisions, base.decisions)),
               "picks_changed_c": sum(x != y for x, y in zip(Sc.decisions, base2.decisions)),
               "_d_a": wk["a"] - wk["base"], "_d_b": wk["b"] - wk["base"], "_d_c": wk["c"] - wk["c_base"]}
        row["memorisation_b_minus_c"] = row["gain_b"] - row["gain_c"]
        row["memorisation_b_minus_a"] = row["gain_b"] - row["gain_a"]
        rows.append(row)
        say(f"  {w['id']}: a {row['gain_a']:+.3%} b {row['gain_b']:+.3%} c {row['gain_c']:+.3%} "
            f"(lessons a/b {row['lessons_a']}/{row['lessons_b']}, picks changed a/b/c "
            f"{row['picks_changed_a']}/{row['picks_changed_b']}/{row['picks_changed_c']})")
    out = {"windows": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
           "parity_ok": all(s["parity"] for s in stash if s["parity"] is not None), "adaptive": adaptive}
    for arm in ("a", "b", "c"):
        d = np.concatenate([r[f"_d_{arm}"] for r in rows]) if rows else np.array([])
        out[f"weekly_gain_{arm}"] = _stat(d, seed + ord(arm))
    mem_c = np.array([r["memorisation_b_minus_c"] for r in rows])
    mem_a = np.array([r["memorisation_b_minus_a"] for r in rows])
    out["memorisation"] = {"b_minus_c_mean": float(mem_c.mean()) if len(mem_c) else 0.0,
                           "b_minus_c_ci": [float(np.quantile(mem_c, .05)), float(np.quantile(mem_c, .95))] if len(mem_c) else [0, 0],
                           "b_minus_a_mean": float(mem_a.mean()) if len(mem_a) else 0.0,
                           "windows_memorised": int(sum(r["memorisation_b_minus_c"] > 1e-9 and r["gain_b"] > 0 for r in rows)),
                           "windows_with_lessons": int(sum(r["lessons_b"] > 0 for r in rows))}
    return out


def archive_markdown(res, title="Phase 11 on the Test archive"):
    f = lambda v: f"{v * 100:+.3f}%"
    L = [f"# {title}", "", f"adaptive replay: {res['adaptive']}  |  wrapper parity: {res['parity_ok']}", ""]
    for arm, what in (("a", "lessons from other windows only"), ("b", "lessons from the same window"),
                      ("c", "same window, renamed and shifted")):
        s = res[f"weekly_gain_{arm}"]
        L.append(f"- arm {arm} ({what}): {f(s['mean'])} per week over {s['n']} weeks, 90% CI {f(s['lo'])} .. {f(s['hi'])}")
    m = res["memorisation"]
    L += [f"- memorisation (b - c): {f(m['b_minus_c_mean'])} (90% CI {f(m['b_minus_c_ci'][0])} .. {f(m['b_minus_c_ci'][1])}); "
          f"(b - a): {f(m['b_minus_a_mean'])}; windows memorised: {m['windows_memorised']} of {m['windows_with_lessons']} with lessons",
          "", "| window | base/wk | a | b | c | b-c | lessons a/b | picks changed a/b/c |", "|---|---|---|---|---|---|---|---|"]
    for r in res["windows"]:
        L.append(f"| {r['window']} | {f(r['base'])} | {f(r['gain_a'])} | {f(r['gain_b'])} | {f(r['gain_c'])} | "
                 f"{f(r['memorisation_b_minus_c'])} | {r['lessons_a']}/{r['lessons_b']} | "
                 f"{r['picks_changed_a']}/{r['picks_changed_b']}/{r['picks_changed_c']} |")
    return "\n".join(L)


def report_markdown(res, title="Anti-memorisation experiment"):
    """Human-readable report of one run: the Bible's required outputs first, then why each lesson was kept or rejected."""
    f = lambda v: f"{v * 100:+.3f}%"
    L = [f"# {title}", "",
         f"- improvement on A: {f(res['improvement_A'])}  |  on disguised A: {f(res['improvement_A_disguised'])}"
         f"  |  memorisation gap: {f(res['memorisation_gap'])}",
         f"- improvement on unseen B: {f(res['improvement_B'])}  |  degradation on B: {f(-res['degradation_B'])}"
         f" (full book, before the firewall)",
         f"- lessons: {res['lesson_count']}  accepted: {len(res['accepted_lessons'])}  "
         f"rejected: {len(res['rejected_lessons'])}  |  identity-invariant: {res['identity_invariant']}",
         f"- after the firewall: A {f(res['final_improvement_A'])}, disguised A {f(res['final_improvement_A_disguised'])},"
         f" B {f(res['final_improvement_B'])}  ->  {res['verdict']}", "",
         "| lesson | verdict | A | A disguised | B |", "|---|---|---|---|---|"]
    for lid in res["accepted_lessons"]:
        e = res["per_lesson"][lid]
        L.append(f"| {lid} | kept | {f(e['A']['mean'])} | {f(e['A_dis']['min'])} | {f(e['B']['mean'])} |")
    for r in res["rejected_lessons"]:
        L.append(f"| {r['lid']} | {r['reason']} | {f(r['A'])} | {f(r['A_disguised'])} | {f(r['B'])} |")
    for key, head in (("era_B", "Per period on B (kept book)"), ("type_B", "Per stock type on B (kept book)")):
        if res.get(key):
            L += ["", f"## {head}", "```", pd.DataFrame(res[key]).T.to_string() if isinstance(res[key], dict)
                  else pd.DataFrame(res[key]).to_string(index=False), "```"]
    if "improvement_A_names_only" in res:
        L.append(f"- keyed on names? rename-only rerun {f(res['improvement_A_names_only'])}; "
                 f"on dates? shift-only rerun {f(res['improvement_A_dates_only'])}")
    if "holdout_C" in res:
        h = res["holdout_C"]
        L.append(f"- never-used window C: {f(h['mean'])} (90% CI {f(h['lo'])} .. {f(h['hi'])}), harmed: {h['harmed']}")
    if "null_control" in res:
        n = res["null_control"]
        L.append(f"- optimism control (returns shuffled within day): null mean {f(n['null_mean'])}, max {f(n['null_max'])}, "
                 f"real A {f(n['real_improvement_A'])}, p {n['p_value']:.3f}, exceeds null: {n['exceeds_null']}")
    L += ["", f"_{res['b_selection_note']}_"]
    return "\n".join(L)


def log_result(res, cfg=None, seed=None):
    """Append the experiment to the registry (provenance-stamped). Imported lazily: it pulls in the model stack."""
    from .improve import log_experiment
    slim = {k: v for k, v in res.items() if k not in ("per_lesson", "final_book")}
    log_experiment({"event": "antimemo", **slim}, cfg=cfg, seed=seed)
    return slim
