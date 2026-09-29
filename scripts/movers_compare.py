import json, glob, sys, numpy as np, pandas as pd
tags = sys.argv[1:] or [""]
for tag in tags:
    rs = [json.load(open(p)) for p in sorted(glob.glob(f"state/movers/m??{tag}/result.json"))]
    if not rs: continue
    hits = picks = weeks = 0
    for p in sorted(glob.glob(f"state/movers/m??{tag}/weeks.parquet")):
        D = pd.read_parquet(p)
        for d, g in D.groupby("date"):
            P = g[g["prob"] >= 0.92].nlargest(10, "prob"); weeks += 1
            picks += len(P); hits += int(((P["up"] >= .1) | (P["dn"] <= -.1)).sum())
    print(f"{tag or 'base':9s} n={len(rs):2d} top-10 {np.mean([r['hit_touch'] for r in rs]):.1%} worst {min(r['hit_touch'] for r in rs):.1%} "
          f">=95%: {sum(r['hit_touch'] >= .95 for r in rs):2d} | 92% bar: {hits/max(picks,1):.1%} on {picks/max(weeks,1):.1f}/wk")
