"""Canon C30: three-way pattern study for the Test project, over every played window.

For each decision week: the system's picks (replayed exactly with the re-tester) and every stock's return over
the following holding period. Groups:
  PICKED_WON   picked and the stock rose >= +7% in the period
  PICKED_LOST  picked and the stock fell
  MISSED_WON   not picked, rose >= +7%
Compares every snapshot indicator (ranked within the day, so eras are comparable), dollar volume ("quantity"),
price level, and the model's own expected return. Prints the biggest separations and a verdict on what to change.
Writes state/research/three_way.json (and docs/three_way.json for the site)."""
import glob, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from engine import config as K, adaptive as A
from engine.live import plain

src = open("scripts/livesim_loop2.py", encoding="utf-8").read()
ns = {"__file__": "scripts/livesim_loop2.py"}
_a = sys.argv; sys.argv = ["x"]
exec(compile(src[:src.index("def worker(")], "l2", "exec"), ns)
sys.argv = _a
cfg, meta = ns["st"]["cfg"], ns["st"]["meta"]
rows = []
for a in sorted(glob.glob(str(K.STATE / "livesim" / "*"))):
    if not Path(a).is_dir() or not list(Path(a).glob("wsnap_*")):
        continue
    w = ns["load_window"](a)
    S = A.replay(cfg, w["snaps"], w["closes"], w["bps"], w["divs"], adaptive=True, meta=meta)
    dec = dict(S.decisions)
    days = sorted(dec)
    C = w["closes"]
    for i, d in enumerate(days):
        d0 = pd.Timestamp(d)
        d1 = pd.Timestamp(days[i + 1]) if i + 1 < len(days) else C.index[-1]
        if d0 not in w["snaps"] and d not in w["snaps"]:
            continue
        snap = w["snaps"].get(d, w["snaps"].get(str(d0.date())))
        if snap is None or d1 <= d0:
            continue
        ret = (C.loc[d1] / C.loc[d0] - 1).reindex(snap.index)
        R = snap.drop(columns=[c for c in snap.columns if c.startswith("m_")]).rank(pct=True)
        R["price_level"] = C.loc[d0].reindex(snap.index).rank(pct=True)
        R["ret"] = ret.values
        R["picked"] = snap.index.isin(dec[d])
        R["window"] = Path(a).name
        rows.append(R.dropna(subset=["ret"]))
D = pd.concat(rows)
D["group"] = np.where(D["picked"] & (D["ret"] >= 0.07), "PICKED_WON",
             np.where(D["picked"] & (D["ret"] < 0), "PICKED_LOST",
             np.where(~D["picked"] & (D["ret"] >= 0.07), "MISSED_WON", "other")))
feats = [c for c in D.columns if c not in ("ret", "picked", "window", "group")]
g = D.groupby("group")[feats].mean()
sd = D[feats].std().replace(0, np.nan)
won_vs_lost = ((g.loc["PICKED_WON"] - g.loc["PICKED_LOST"]) / sd).sort_values(key=abs, ascending=False)
missed_vs_won = ((g.loc["MISSED_WON"] - g.loc["PICKED_WON"]) / sd).sort_values(key=abs, ascending=False)
counts = D["group"].value_counts().to_dict()
picked = D[D["picked"]]
out = {"windows": int(D["window"].nunique()), "counts": counts,
       "pick_hit_rate_7": float((picked["ret"] >= 0.07).mean()), "pick_win_rate": float((picked["ret"] > 0).mean()),
       "universe_hit_rate_7": float((D["ret"] >= 0.07).mean()),
       "won_vs_lost": [{"feature": f, "name": plain(f.replace("e_", "")), "gap_sd": float(v)} for f, v in won_vs_lost.head(15).items()],
       "missed_vs_won": [{"feature": f, "name": plain(f.replace("e_", "")), "gap_sd": float(v)} for f, v in missed_vs_won.head(15).items()],
       "profiles": g.round(3).to_dict()}
for p in (K.STATE / "research" / "three_way.json", K.SITE / "three_way.json"):
    p.write_text(json.dumps(out, indent=1, default=float))
print(f"{out['windows']} windows | groups {counts}")
print(f"system picks: {out['pick_hit_rate_7']:.1%} rose >= +7% over the holding period, {out['pick_win_rate']:.1%} rose at all "
      f"(all stocks: {out['universe_hit_rate_7']:.1%} rose >= +7%)")
print("\nWHAT SEPARATES PICKED WINNERS FROM PICKED LOSERS (+ = winners had more):")
for r in out["won_vs_lost"][:8]:
    print(f"   {r['gap_sd']:+.2f} sd  {r['name']} ({r['feature']})")
print("\nWHAT THE SYSTEM IS BLIND TO: MISSED WINNERS vs PICKED WINNERS (+ = missed winners had more):")
for r in out["missed_vs_won"][:8]:
    print(f"   {r['gap_sd']:+.2f} sd  {r['name']} ({r['feature']})")
