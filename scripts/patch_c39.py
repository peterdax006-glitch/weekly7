"""Canon C38/C39 tiered objective in the Test loop (a necessary Test change: the loop must judge by the owner's order).
Tier 1: most weeks move ~7% (|r| in 5%-10%). Tier 2: minimise risk. Tier 3: raise the share of in-band weeks that are positive."""
import pathlib
p = pathlib.Path("scripts/livesim_loop2.py"); s = p.read_text(encoding="utf-8")
old = '''    r["sd_week"] = float(wk.std()) if len(wk) > 1 else 0.0
    r["win_weeks"] = float((wk > 0).mean()) if len(wk) else 0.5
    return r


def objective(rows, phase):'''
new = '''    r["sd_week"] = float(wk.std()) if len(wk) > 1 else 0.0
    r["win_weeks"] = float((wk > 0).mean()) if len(wk) else 0.5
    band = (np.abs(wk) >= BAND[0]) & (np.abs(wk) <= BAND[1])
    r["in_band"] = float(band.mean()) if len(wk) else 0.0                 # tier 1: share of ~7% weeks
    r["over_band"] = float((np.abs(wk) > BAND[1]).mean()) if len(wk) else 0.0
    r["worst5"] = float(np.quantile(wk, 0.05)) if len(wk) > 5 else 0.0   # tier 2: tail risk
    r["pos_in_band"] = float((wk[band] > 0).mean()) if band.any() else 0.0  # tier 3
    return r


BAND = (0.05, 0.10)          # "about 7%": below 5% is too low, above 10% too risky (C38)


def tiered(rows):
    """C39: tier 1 dominates until most weeks are ~7% (majority); then risk; then the positive share."""
    t1 = float(np.mean([r["in_band"] for r in rows]))
    over = float(np.mean([r["over_band"] for r in rows]))
    risk = float(np.mean([r["worst5"] for r in rows])) + float(min(r["max_dd"] for r in rows)) / 4
    t3 = float(np.mean([r["pos_in_band"] for r in rows]))
    reached = t1 >= 0.5
    return (100 * min(t1, 0.5) - 50 * over + (10 * (risk + 0.3) + t3 if reached else 0.0)), t1, risk, t3


def objective(rows, phase):
    """Kept for the old call sites; now defers to the tiered objective."""
    mw = float(np.mean([r["mean_week"] for r in rows]))
    sd = float(np.mean([r["sd_week"] for r in rows]))
    worst = min(r["max_dd"] for r in rows)
    if worst < -0.99:
        return -9.0, mw, sd, worst
    return tiered(rows)[0], mw, sd, worst


def _old_objective(rows, phase):'''
assert old in s; s = s.replace(old, new)
old = '''        print(f"  [{r}] avg week {res['mean_week']:+.2%} | swing (sd) {res['sd_week']:.2%} |'''
new = '''        wk_ = np.array(res.get("weekly_returns", [])) if res.get("weekly_returns") else None
        print(f"  [{r}] ~7% weeks (5-10% moves) {res.get('in_band', float('nan')):.0%} | avg week {res['mean_week']:+.2%} | swing (sd) {res['sd_week']:.2%} |'''
assert old in s; s = s.replace(old, new)
old = '''    r = trader.session.result()
    wk = np.array(trader.session.weeks)
    r["sd_week"] = float(wk.std()) if len(wk) > 1 else 0.0'''
new = '''    r = trader.session.result()
    wk = np.array(trader.session.weeks)
    r["sd_week"] = float(wk.std()) if len(wk) > 1 else 0.0
    band = (np.abs(wk) >= BAND[0]) & (np.abs(wk) <= BAND[1])
    r["in_band"] = float(band.mean()) if len(wk) else 0.0
    r["pos_in_band"] = float((wk[band] > 0).mean()) if band.any() else 0.0
    r["weekly_returns"] = [float(x) for x in wk]'''
assert old in s; s = s.replace(old, new)
p.write_text(s, encoding="utf-8", newline="\n")
print("tiered objective in")
