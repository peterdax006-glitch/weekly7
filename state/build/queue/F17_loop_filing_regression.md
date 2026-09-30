# F17_loop_filing_regression  [model: opus]
C69 sections 12-14, 27, 28, 31 (diagnose before changing anything). Read state/build/CONTEXT.md (rules 1-29).
Facts (committed results; compare them, do not re-run blindly):
- final-code loops (after F12, before F14/F16), git show cac30e76:state/research/regate_sequential/loop_default_s*.json and
  SUMMARY_final_loops.md: genuine knowledge filed on 4/6 default seeds (s1 xs_atr_rank c58 + xs_vol_rank c102; s3 xs_vol_rank c77 +
  lv20 c112; s4 xs_range_rank c109; s5 vol_over_mkt c47), nulls 0/3.
- after F14 (calendar cut + failure floor) and F16 (rolling 156-week frame), the SAME runner at default settings (current
  state/research/regate_sequential/loop_*_s*.json, logs/f16_loop_*.log): filed on 3/6 seeds, lv20 only (s0 c44, s1 c61, s3 c77);
  s2, s4, s5 nothing; the xs_* and vol_over_mkt findings that filed before no longer file anywhere; nulls 0/3; one failed stage (s2).
- F16 also noted seed 4's lv20 now ends FAILED on calibration + failure_behavior in the gate proof.
You OWN: read-only diagnosis first; then engine/research/evidence.py and feeds.py (FrameStore only) if the cause is there, with tests.
loop.py / quality_gate.py read-only (report defects with a failing test). Never run git. A Test-loop run (livesim_loop2, PID in
Masterstock journal) is running: keep your own processes to <= 2 and never kill others.
Do:
1. From the per-cycle gate histories (every look's verdict + blocking reasons are recorded), explain per seed why the xs_* /
   vol_over_mkt findings stopped filing and why s2/s4/s5 file nothing: which gate blocks, from which look, and which change
   (F14 train_cut, F14 failure floor, F16 frame) moved it - prove it by re-gating the recorded evidence with each change toggled.
2. If a change made the gate WRONGLY stricter (e.g. calibration measured on a window reaching into warm-up data, or the rolling frame
   shrinking unseen evidence), fix it at the source with a test; if the stricter verdict is CORRECT (the earlier filings were weak),
   say so with the evidence - earlier filings are not a target.
3. Report per-seed table (before / after / after-fix), nulls, tests, ruler counts. Nothing VALIDATED by you.
