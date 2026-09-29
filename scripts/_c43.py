import pathlib
p = pathlib.Path("engine/patterns.py"); s = p.read_text(encoding="utf-8")
old = '''        R["status"] = np.where(~R["confirmed"], "rejected", np.where(dying, "benched", "active"))'''
new = '''        R["status"] = np.where(~R["confirmed"], "rejected", np.where(dying, "failed", "active"))   # C43'''
assert old in s; s = s.replace(old, new)
old = '''            for i in R.index[R["status"] == "benched"][:50]:'''
new = '''            for i in R.index[R["status"] == "failed"][:100]:'''
assert old in s; s = s.replace(old, new)
old = '''                        if np.sign(a[0]) == sgn and abs(a[1]) >= 2 and np.sign(rr[0]) == sgn and abs(rr[1]) >= 1:'''
new = '''                        # improved form must be consistent through the WHOLE history up to now: long-run,
                        # discovery half, confirmation half and the recent stretch all agree (C43)
                        dsc = _wstats(w[mask & cm & disc], yv[mask & cm & disc]) if (mask & cm & disc).sum() >= 30 else (0, 0, 0)
                        cnf = _wstats(w[mask & cm & conf], yv[mask & cm & conf]) if (mask & cm & conf).sum() >= 30 else (0, 0, 0)
                        if (np.sign(a[0]) == sgn and abs(a[1]) >= 2 and np.sign(rr[0]) == sgn and abs(rr[1]) >= 1
                                and np.sign(dsc[0]) == sgn and np.sign(cnf[0]) == sgn):'''
assert old in s; s = s.replace(old, new)
old = '''        # redundancy: patterns firing on nearly the same rows are one idea - keep the strongest'''
new = '''        R.loc[R["status"] == "failed", "status"] = "discarded"      # C43: never hold a failed pattern
        # redundancy: patterns firing on nearly the same rows are one idea - keep the strongest'''
assert old in s; s = s.replace(old, new)
p.write_text(s, encoding="utf-8", newline="\n")
print("C43 in")
