"""Algorithm -> Find volatility: inside each blind window, a pattern miner learns MOVEMENT-SIZE patterns (tier 1) from
warm-up weeks only (labels closed before the window), and its score becomes an input to the mover model."""
import pathlib
p = pathlib.Path("scripts/movers.py"); s = p.read_text(encoding="utf-8")
old = '''    cols = [c for c in X.columns if not c.startswith(DROP)]
    S = feed._stocks'''
new = '''    if v.get("patterns"):
        from engine import candles
        from engine.patterns import PatternMiner
        cf = candles.build(feed._stocks)
        X = X.copy()
        for k2, fr in cf.items():
            X[k2] = fr.stack(future_stack=True).reindex(X.index).astype("float32").values
        del cf
    cols = [c for c in X.columns if not c.startswith(DROP)]
    S = feed._stocks'''
if old not in s:
    old = '''    cols = [c for c in X.columns if not c.startswith(DROP)]'''
    new = new.replace("\n    S = feed._stocks", "")
assert old in s; s = s.replace(old, new, 1)
old = '''    dates = X.index.get_level_values(0)
    Xt = X[dates.isin(warm)][cols]'''
new = '''    dates = X.index.get_level_values(0)
    if v.get("patterns"):
        # the miner sees only warm-up week-ends whose 5-session window has closed
        wk_warm = [d0 for d0 in week_ends(sessions[sessions < first]) if d0 <= cutoff]
        Xm = X[dates.isin(wk_warm)][cols]
        swing = np.maximum(up, -dn).stack(future_stack=True).reindex(Xm.index)
        swing = swing - swing.groupby(level=0).transform("mean")
        miner = PatternMiner({"max_rows": 400_000, "null_reps": 1}).fit(Xm, swing, now=first)
        pat = pd.Series(0.0, index=X.index)
        for d0 in sorted(set(dates)):
            if d0 in set(wk_warm) or d0 >= first:
                Xd = X.xs(d0, level=0)[cols]
                pat.loc[d0] = miner.score(Xd).values
        X = X.assign(pat_move=pat.values)
        cols = cols + ["pat_move"]
        run_window.miner_report = miner.report
    Xt = X[dates.isin(warm)][cols]'''
assert old in s; s = s.replace(old, new, 1)
old = '''    res.update({"id": rid, "train_base_rate": base_rate_train, "variant": v, "seconds": round(time.time() - t0)})'''
new = '''    res.update({"id": rid, "train_base_rate": base_rate_train, "variant": v, "seconds": round(time.time() - t0),
                "miner": getattr(run_window, "miner_report", None) if v.get("patterns") else None})'''
assert old in s; s = s.replace(old, new, 1)
p.write_text(s, encoding="utf-8", newline="\n")
print("movers can use the pattern miner")
