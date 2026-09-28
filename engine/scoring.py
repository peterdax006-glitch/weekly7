"""Universe-wide prediction scoring (Blueprint Part H2 level 3, Part M1 'Selection'/'Forecast').

Each prediction file becomes scorable 5 sessions later. For every scorable day we record the
rank IC of every signal column against the realised 5-day return, and the calibration of
P(+7% before stop). ~2,500 predictions a day gives real statistical power within a month."""
import json
import numpy as np
import pandas as pd

from . import config as K

SCORES = K.STATE / "scores.jsonl"
H = 5


def _outcomes(stocks, day, tickers):
    C, Hh, L = stocks["Close"], stocks["High"], stocks["Low"]
    idx = C.index
    i = idx.get_loc(pd.Timestamp(day))
    if i + H >= len(idx):
        return None
    win = slice(idx[i + 1], idx[i + H])
    c0 = C.loc[idx[i], tickers]
    fwd = np.log(C.loc[idx[i + H], tickers] / c0)
    hi = Hh.loc[win, tickers].max() / c0 - 1
    return pd.DataFrame({"fwd": fwd, "touched7": (hi >= 0.07).astype(float)})


def score_predictions(stocks):
    from .live import PRED_DIR
    done = set()
    if SCORES.exists():
        done = {json.loads(l)["date"] for l in SCORES.read_text().splitlines() if l.strip()}
    for f in sorted(PRED_DIR.glob("*.parquet")):
        day = f.stem
        if day in done:
            continue
        P = pd.read_parquet(f)
        out = _outcomes(stocks, day, [t for t in P.index if t in stocks["Close"].columns])
        if out is None:
            continue
        df = P.join(out, how="inner").dropna(subset=["fwd"])
        cols = [c for c in df.columns if c in ("score", "evidence", "rank_raw", "mu_raw", "p_target", "tilt")
                or c.startswith("f_") or c.startswith("o_")]
        ic = {}
        for c in cols:
            s = pd.to_numeric(df[c], errors="coerce")
            ok = s.notna()
            if ok.sum() >= 30 and s[ok].nunique() > 2:
                ic[c] = float(s[ok].rank().corr(df.loc[ok, "fwd"].rank()))
        p = df["p_target"].clip(0, 1)
        rec = {"date": day, "n": int(len(df)), "ic": ic,
               "brier": float(((p - df["touched7"]) ** 2).mean()),
               "pred_rate": float(p.mean()), "actual_rate": float(df["touched7"].mean()),
               "top20_fwd": float(df.nlargest(20, "score")["fwd"].mean()),
               "all_fwd": float(df["fwd"].mean())}
        with open(SCORES, "a") as fh:
            fh.write(json.dumps(rec) + "\n")


def load_scores() -> pd.DataFrame:
    if not SCORES.exists():
        return pd.DataFrame()
    rows = [json.loads(l) for l in SCORES.read_text().splitlines() if l.strip()]
    return pd.DataFrame(rows)
