# Self-teaching loop: audit and trust gate (h58, 3 Oct 2026)

Owner, 3 Oct 2026 (verbatim): "also it should be able to collect and drill data where it can teach itself efficently i dont know if thats a GPU
job but if it is we need it to be the level where you would trust it to teach itself". Priorities: 1 efficiency, 2 coding, 3 language.

The gate is `creator/selfteach.py` (`python -m creator.selfteach [state] [out_dir]`, registry name `selfteach`). It only reads records.
Reports go to `state/creator/thinking/selfteach_gate.{json,md}`, which is a protected path. `SELF_TEACH_TRUSTED` is true only when all seven
checks pass. Like `thinking.independent`, it does not switch anything on.

## Audit of the current loop (real records, main state, 3 Oct ~23:30 UTC)

Loop steps: collect (git caches, public acquisition and fetch, journal, plan log, research JSON, trace bank) -> drill (walk-forward per source
x variant, select 60% / held-out 40%) -> search (grid, then open-ended proposals, learn-queue remedies, trialerror.react) -> adopt
(`best_variant` = best select Brier) -> trust (`trust_of` on held-out + live) -> learn loop (measure, retention, diagnosis, improve, verify).

| criterion | before h58 | after h58 (branch) |
|---|---|---|
| a grounded | PASS. The only trusted drill (x_git_fixed) uses a deterministic checker on git history. Unverified sources: plan_choice (Nupen predicting its own scheduler) and research_bool (flags written by its own research code). Neither is trusted. Trace bank: 832 rows, all carry model and pulse, 0 are frozen questions. | same |
| b leak-proof | FAIL. (1) Walk-forward sorted resolve-before-create at equal time, so an item was predicted with its own outcome. Canary p 0.020 -> 0.053; plan_choice 890/890 items and research_bool 100% affected; git and journal 0 ties. Same bug in `predictors.walk_forward_model`. (2) 6/40 git_fixed and 6/40 git_churn frozen part-a items had drifted into the select part (moving 60% cut). (3) learnloop diagnosis and trialerror.react read every frozen item and the whole held-out tail to choose the next variants. (4) The items.json hash was only checked against itself. | PASS. Create-before-resolve (reference, numba, predictors; identical results). Frozen subjects are taken out of the select part, and the hash is pinned from outside (`drillsources.PINNED_BENCH_HASH`). Diagnosis and react get the select part only, with no frozen items. |
| c held-out, significant, revert | FAIL. Regression IM-d2e9a3990c (x_git_fixed held-out 0.0259 -> 0.0107) was still the adopted variant of the only trusted drill. There was no auto-revert. Judgment strategies are picked on the subjects they are scored on (none trusted, so this does not block). | PASS. `best_variant` skips hurt_flag_revert variants; `learnloop.record_reverts` logs a `revert` event; the hill-climb never starts from a reverted variant. |
| d no drift | FAIL, only because plan_choice is a self-output source. Outside share of live outcomes in the last 24 h is 0.59 (146 outside / 102 own); 5 public acquisitions in 7 d. | same (follow-up F4) |
| e choosing by gain / CPU-h | FAIL. The filler picks by grid, then learn queue, then round robin. The gain predictor exists and is trusted (fastpred drill_beats_best: n 2388, 1167 live, CI [0.024, 0.037], ECE 0.02) but nothing uses it to choose. 0 of 11 improvements record a predicted gain. | same (follow-up F1) |
| f efficiency tracked | FAIL. 0 of 1300 drill rows from the last day record a cost. | Fixed in code: `compute_row` records `cpu_s` (thread CPU) and `wall_s`. Passes once the swarm runs the merged code. Disk and RAM are reported from the records and resource_samples. |
| g own ruler | FAIL. thinking.py (trust_of), thinkbench, learnloop, drillsources, fastwalk, judgment, fastpred, decide, trialerror and state/creator/thinkbench/** were not protected, so a Nupen package could edit them. focus.THINKING_MODULES lists thinking.py as thinking work. | PASS. All of them, plus selfteach.py and gpuselfteach.py, are in `sandbox.PROTECTED`. |

Gate on the real records with the branch code: failing are d, e and f. The f failure goes away once the merged code runs.

## Past numbers that become invalid (marked, not deleted)

- `drillsources.WALK_VERSION = "w2"`: every digest gets the suffix `.w2`. All rows in `drill_runs.jsonl` without it are SUPERSEDED. They stay in
  the file as history, and `trust_section` never trusts them ("recompute pending"). The swarm re-queues every source automatically because the
  digest changed.
- LEAKY (each item predicted with its own outcome): every plan_choice and research_bool score. That covers held-out gains (plan_choice
  [0.012, 0.042], research_bool [0.031, 0.055]), their progress points and any learn-loop verdict on them.
- Selection saw frozen items: git_fixed / git_churn / journal_persist variant choices. Held-out scores were measured correctly, but which variant
  won may change.
- The learn curve (`progress_curve.json`) drops drill held-out points without a w2 digest. They stay in progress.jsonl.
- thinkbench part a: its git and journal sources have 0 ties, so their walk-forward predictions do not change. Frozen items now stay out of
  variant selection, so a part-a change after merge can come from the variant choice. Bench items, the hash and the scoring are unchanged.

## Bigger gaps, ranked by value for the teacher

1. F1: choose by expected gain per CPU-hour (criterion e). Order the drill filler's candidates by P(beats best) from fastpred, times the
   expected Brier gain, divided by predicted CPU-hours (`cpu_s` of the source's recent rows). The gain predictor is already trusted. Record a
   predicted gain on every learn-loop improvement so verify can score predicted vs actual.
2. F2: make 'helped' significant in `learnloop.verify` (queue_variants). Today it is a point comparison (av > bv); use after > before CI
   upper bound, as the goal-proposal branch does. Real records: 1 helped verdict, and it is significant, so the gate passes today.
   before/after are also unpaired (the held-out tail grows); a paired re-score on the old tail is the honest measure.
3. F3: judgment strategy selection on the scored subjects (winner's curse over 14-15 strategies). Pick on earlier subjects, score on later
   ones, before any judgment topic can be trusted.
4. F4: plan_choice is Nupen predicting its own scheduler. Drop it from `learnloop.DIAG_SOURCES` (no learn-loop goals from it), keep it replay
   only. research_bool: date its outcomes or drop it.
5. F5: the noise stop (`trialerror.select_predicts_heldout`) uses held-out Brier to decide when to stop searching. Mild, but it is a held-out read
   in a decision. Use a split inside the select part.
6. F6: `registry.py` can repoint `thinking` / `drillsources` to another module. Consider protecting the registry entries of measuring modules.
7. F7: gpuday registers a fine-tuned model as servable before any held-out check. Gate adoption on `gpuselfteach.heldout_gate_job` (below).

## GPU role (rented pod)

The statistical drills do not need a GPU. The numba walk-forward does ~55k items in seconds on the CPU. The GPU helps where a model is the
learner or the teacher:

| use | value for self-teaching | exists |
|---|---|---|
| Big-model teacher data: verified worked traces (correct by known answer, outcome-blind) | high: grounded teacher signal at scale | gpupulse pulse 4 `traces` |
| Fine-tunes of small home models (ft1 1.7B/4B on worked traces, ft2 coder) | high, but only behind a held-out gate | gpuday ft1/ft2 + register |
| Held-out gate for a tuned model (paired, time-ordered, CI) | **highest missing piece**: decides whether a model update is kept | NEW `creator/gpuselfteach.py` |
| Bulk drilling / judgment with a stronger model | medium: measures the judge, never grants trust alone | pulses 2/3 |
| Big-model verification of labels | low: labels already come from deterministic checkers | - |

Ready ext_job (`creator.gpuselfteach.HELDOUT_GATE_JOB`; runs after `register_tuned_job` of ft1_17b):

```json
{"name": "heldout_gate_ft1_17b", "call": "creator.gpuselfteach:heldout_gate_job", "model": "Qwen3-1.7B-gpuday-ft1.gguf",
 "minutes": 12, "max_minutes": 25, "low_util_abort_minutes": 0, "args": {"n": 400, "home_model": "Qwen3-1.7B-Q4_K_M.gguf"}}
```

Held-out questions are reasoning questions the home model answered with the 'plain' prompt. They are not in the trace bank (the fine-tune's
data) and are newer than every banked question. Frozen collisions are already removed by `reasondrills.generate`. The tuned model answers the
same prompt. Paired difference: ADOPT only if n >= 50 and the 95% CI lower bound > 0, otherwise KEEP_HOME. The verdict is recorded in
`thinking/gpu_heldout_gate.jsonl`; it never switches anything. Dry run: `tests/test_creator_selfteach.py::test_heldout_gate_dry_run_with_a_stub_model`
and the runner's `parse_job` validation. No pod, server or runner was started or stopped.
