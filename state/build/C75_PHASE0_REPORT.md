# C75 PHASE 0 REPORT — INVENTORY AND BASELINE (30 Sep 2026)

Format: C75 section 22. Every number here comes from a saved artifact under state/build/evidence/c75_phase0/,
state/build/C75_PHASE0_MAPPING.md or the machine checklists.

## What was supposed to happen
- Read every contract and machine checklist, Masterstock, the research mapping, the integration audit and the open-problem list.
- Inspect git.
- Verify hashes and provenance.
- Take baselines for tests, line budget, reachability and type checks.
- Freeze a starting snapshot.
- Map every C62, C66, C67, C68 and benchmark requirement.
- Identify duplicates, missing functionality, and systems that must be extended rather than duplicated.
- No optimisation.

## What actually happened
- **C75 saved:** verbatim (ULTIMATE_MASTER_PROMPT.md, sha256 905d0199), canon C75, hash-locked in provenance, CI and tests.
- **Machine checklist:** 302 items (209 boxes plus 93 section requirements).
- **Baselines taken:**

| Baseline | Result |
|---|---|
| Canon | 75 directives verify |
| verify_integrity | True |
| Reachability | engine.research 63/63 and engine.learning 64/64 reached, 0 unreached |
| Line budget | saved |
| Git | start commit recorded |
| Snapshot tag | `c75-phase0-start` (0b54f2a3d) |
| CI type check (`python -m mypy`) | **RED**: 3 errors |
| Type-check scope | 4 C62 modules masked by a "temporary" ignore_errors override since 29 Sep; engine/research never type-checked (87-91 errors unmasked in learning, 928 across both packages) |
| Full suite | **5,783 passed, 1 failed, 6 skipped**, 1 xfailed, 58 min |

- **The one failure was a real determinism defect.** current_code_hash() depended on which modules were loaded, so identical results carried different code hashes. It is FIXED: the hash now covers the engine tree on disk, with a regression test. Provenance and credit tests: 65 passed.
- **Independent read-only mapping** (F20, Opus, trusting no checklist note) covered every C62, C66, C68, C67 and benchmark requirement → C75_PHASE0_MAPPING.md.
- **330 corrections** were applied through contract_checklist, which enforces its rules. Every checklist now matches the evidence.

| Checklist | Before (JSON) | After (evidenced) |
|---|---|---|
| C62 (191) | 15 VALIDATED / 13 FAILED | **0 VALIDATED**, 154 TESTING, 12 IMPLEMENTED, 12 FAILED, 7 IN_PROGRESS, 6 NOT_STARTED |
| C66 (114) | 10 VALIDATED | **0 VALIDATED**, 79 TESTING, 10 IMPLEMENTED, 18 IN_PROGRESS, 3 FAILED, 4 NOT_STARTED |
| C68 (64) | 1 VALIDATED | 1 VALIDATED (PC03, reachability), 57 TESTING, 1 FAILED, 1 IN_PROGRESS, 4 NOT_STARTED |
| C75 (302) | - | 20 VALIDATED (Phase 0 boxes with evidence), 1 IN_PROGRESS, 281 NOT_STARTED |

- **Why 25 of 26 VALIDATED rows were downgraded:** 18 rested on the survivor-only panel. 7 rested on a report with no code hash that predates later code changes.

## Tests run / passed / failed
- Full suite: 5,783 passed, 1 failed (fixed since), 6 skipped.
- Provenance and credit after the fix: 65 passed.

## Known limitations found (the Phase 1 work list, in C75 order)
1. **CI type check red and masked modules.** F21 (Sonnet) is fixing it.
2. **The default planted world opens 0 positions** in 129/129 cycles of every clean run. So C68's expectation → outcome → error chain, what_changed, research_depth and learned exits never run. F23 (Opus) is fixing it.
3. **Knowledge never reaches the production decision-maker** (adaptive.Session). F23.
4. **Test suite violates Firewall 5.** REFERENCE_SEEDS skips failing seeds 11/17/41, and one test asserts the unguarded infinite-transfer-ratio bug. F22 (Opus).
5. **Persisted health books do not reload** (5 of 6 fail). The failed_lab and waste stages output nothing. The conjunction pre-screen drops XOR pairs. F24 (Sonnet).
6. **The real-vs-noise benchmark** is built but not yet run. F19 (Opus) is running it. There is no cross-world learner yet, so no learning curve is possible by construction.
7. **C67 mover-episode research** is wired as a loop stage but has produced 0 candidates and has never seen real data. The collapse label is missing. The acceleration, close-location and intraday labels are partial.
8. **The research loop has never run on real data.** The C62 learner fails its real-data tests.
9. **Survivor-only panel:** only 77 of 9,250 delisted names have prices. The registry audit and the leak audit are stale. Leak channel 6 is LEAK.
10. **No whole-cycle deterministic-replay test.**
11. **Duplicates:**
    - no single declared mover-model champion;
    - 5 pattern stores;
    - ~7 experiment ledgers, two of them named ExperimentLedger;
    - a HealthLedger name clash;
    - a parallel QuestionEngine;
    - 7 planted-world generators.
12. **Section 4 (29 items):** 4 fixed with a regression test (planted data), 4 partial, 14 open, 6 unmeasured, 1 with no evidence.

## Evidence artifacts
- state/build/evidence/c75_phase0/: start_commit, full_suite, hashes_provenance, reachability, mypy, line_budget, git_status_count.
- state/build/C75_PHASE0_MAPPING.md
- state/build/C75_PHASE0_CORRECTIONS.json
- scripts/apply_corrections.py

## Checklist changes
- The 330 corrections above.
- Phase 0 boxes UM002–UM022 moved to VALIDATED with evidence (UM014 snapshot: tag made).
- UM001 ("read every contract in full") stays open. The auditor did not read every contract end to end, and I will not mark it.

## Remaining blockers before Phase 1 is complete
Items 1–12 above. The Phase 1 builders are running: F21, F22, F23, F24, and F19 (benchmark).

## Next unfinished checklist items
- Phase 1A–1E, per the list above.
- The C67 labels and a real-data mover sweep.
- Duplicate consolidation.
- The whole-cycle replay test.
- A real-data research-loop run.
