# F12 full-loop runs (pre-complexity-fix code be1414a816a6c370) - summary, 30 Sep 01:30 MDT

Stopped at ~01:10 MDT by the main session: at 01:04:34 a mistaken `git stash` in the main session reverted these runs' on-disk
compute ledgers, c68 hash chains and checkpoints to the last auto-snapshot while they were running (they then wrote on top of the
reverted files: ledgers lost 70-120 entries, chains 300-650 lines). Every result below is the clean pre-revert state taken from the
stash (the loop_*.json files were restored from it); nothing written after 01:04:34 is used.

| run | world | cycles | reached | knowledge filed (feature, filed_at) | open gates (feature, verdict, looks) | cycles with a failed stage |
|---|---|---|---|---|---|---|
| s0 | default | 118 | 2019-08-09 | - | lv20 NME x7, xs_vol_rank NME x5, xs_range_rank NME x2 | 5 |
| s1 | default | 112 | 2019-06-28 | - | - | 5 |
| s2 | default | 120 | 2019-08-23 | - | - | 6 |
| s3 | default | 113 | 2019-07-05 | lv20 (2019-02-08) | lv20 NME x5 | 6 |
| s4 | default | 111 | 2019-06-21 | - | lv20 NME x6, vol_over_mkt NME x4, xs_atr_rank NME x1 | 1 |
| s5 | default | 115 | 2019-07-19 | xs_atr_rank (2018-04-06), xs_vol_rank (2018-09-07) | xs_range_rank FAILED x1 | 2 |
| null s0 | null | 129 | 2019-10-25 | - | - | 10 |
| null s1 | null | 52 | 2018-05-04 | - | - | 8 |
| null s2 | null | 57 | 2018-06-08 | - | - | 5 |

- Genuine knowledge filed on 2/6 default seeds within ~115 cycles; every filed item is a planted genuine feature; 0 false knowledge.
- No genuine finding was retired (the sequential plan never retired one); 4 seeds still NEEDS_MORE_EVIDENCE at the stop.
- Null worlds: nothing gated or filed (null s1/s2 ended early at 52/57 cycles - cause not investigated).
- Failed stages were `run.experiments: ComputeError: attempt directory ... exists` - a real defect fixed 30 Sep in
  engine/learning/compute.py (attempt folders numbered by attempt_serial across code versions; test
  test_rerun_under_new_code_gets_a_fresh_attempt_folder).
- NEXT: re-run on the final code (complexity fix + attempt fix), 6 default + 3 null seeds, to the end of the planted data.
