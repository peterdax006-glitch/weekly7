"""Multi-timeframe candle features (canon C35): daily, weekly (5 sessions) and monthly (21 sessions) candles and
small-print signals - momentary reversals, inside days, engulfing, streaks, gap fills. All computed from data up to
and including day t (point-in-time). Minute candles: free history does not exist for past decades; see
scripts/collect_intraday.py for the forward-looking collector."""
import numpy as np
import pandas as pd


def _candle(O, H, L, C, n, tag):
    if n == 1:
        o, h, l = O, H, L
    else:
        o, h, l = O.shift(n - 1), H.rolling(n).max(), L.rolling(n).min()
    rng = (h - l).replace(0, np.nan)
    return {
        f"{tag}_body": (C - o) / rng,                         # -1 (full red) .. +1 (full green)
        f"{tag}_upwick": (h - np.maximum(C, o)) / rng,
        f"{tag}_lowwick": (np.minimum(C, o) - l) / rng,
        f"{tag}_pos": (C - l) / rng,                          # where the close sits in the candle's range
        f"{tag}_range": rng / C,
    }


def build(stocks):
    O, H, L, C = (stocks[k].astype("float32") for k in ("Open", "High", "Low", "Close"))
    F = {}
    for n, tag in ((1, "cd"), (5, "cw"), (21, "cm")):
        F.update(_candle(O, H, L, C, n, tag))
    r = C / C.shift(1) - 1
    up = (r > 0).astype("float32")
    # streak of consecutive up (+) or down (-) days
    u = up.values
    run = np.ones_like(u)
    for t in range(1, len(u)):                               # run length, vectorised across all stocks
        run[t] = np.where(u[t] == u[t - 1], run[t - 1] + 1, 1)
    streak = pd.DataFrame(run, index=up.index, columns=up.columns)
    F["streak"] = streak.where(up > 0, -streak)
    F["reversal_1d"] = (np.sign(r) != np.sign(r.shift(1))).astype("float32") * np.sign(r)   # momentary reversal
    F["reversal_vs_week"] = np.sign(r) * -np.sign(C / C.shift(5) - 1)                       # today against the week
    F["inside_day"] = ((H <= H.shift(1)) & (L >= L.shift(1))).astype("float32")
    F["outside_day"] = ((H > H.shift(1)) & (L < L.shift(1))).astype("float32")
    body = C - O
    F["engulf"] = (((body > 0) & (body.shift(1) < 0) & (C > O.shift(1)) & (O < C.shift(1))).astype("float32")
                   - ((body < 0) & (body.shift(1) > 0) & (C < O.shift(1)) & (O > C.shift(1))).astype("float32"))
    gap = O / C.shift(1) - 1
    F["gap"] = gap
    F["gap_filled"] = (((gap > 0) & (L <= C.shift(1))) | ((gap < 0) & (H >= C.shift(1)))).astype("float32")
    F["week_vs_month_pos"] = F["cw_pos"] - F["cm_pos"]                                     # timeframe disagreement
    F["day_vs_week_body"] = F["cd_body"] - F["cw_body"]
    return F
