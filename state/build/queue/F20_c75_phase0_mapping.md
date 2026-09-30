# F20_c75_phase0_mapping  [model: opus]  (READ-ONLY audit - write only the two output files below)
Canon C75 = the CURRENT master checklist: C:\Users\Peter\weekly7\ULTIMATE_MASTER_PROMPT.md (verbatim). Its machine checklist:
state/build/ULTIMATE_MASTER_CHECKLIST.json (302 items: 209 boxes + 93 section requirements). Read CONTEXT.md rules 1-29.
C75 Phase 0 asks for: every authoritative contract read; every machine checklist read; Masterstock, research mapping, integration
audit and the open-problem list read; git state; hashes; provenance; test / line-budget / reachability / type-check baselines; a
frozen starting snapshot; every C62, C66, C67, C68 and benchmark requirement MAPPED; duplicates; missing functionality; systems that
must be extended rather than duplicated. You do NOT change code, checklists or tests. Never run git except read-only (`git log`,
`git show`, `git status`). Use <= 1 process for anything heavy; a real-data Test loop and the full test suite are running.
Inputs: SELF_LEARNING_CONTRACT.md + state/build/SELF_LEARNING_MASTER_CHECKLIST.json (C62); RESEARCH_BRAIN_CONTRACT.md +
RESEARCH_BRAIN_CHECKLIST.json (C66); canon C67 text; PREDICTION_ERROR_ADDITION.md + PREDICTION_ERROR_CHECKLIST.json (C68);
TEN_HOUR_EXECUTION_CHECKLIST.md + TEN_HOUR_CHECKLIST.json (C69); canon C70-C74; state/build/EXECUTION_LEDGER.md (29 Sep, now stale in
places); state/build/RESEARCH_MAPPING.md, INTEGRATION.md, INTEGRATION_AUDIT.md; Masterstock/MASTERSTOCK.md "START HERE" + JOURNAL.md
since 29 Sep 16:00; state/research/**/SUMMARY*.md; the code (engine/, scripts/, tests/).
Write:
1. state/build/C75_PHASE0_MAPPING.md - for EVERY item of C62 (191), C66 (114), C68 (64), every C67 requirement (from C75 1C and the
   canon text) and every benchmark requirement (C75 3A-3N, 6, 7, 8, 15 + C70-C74): its CURRENT honest state under C75 Firewall 2
   (NOT_STARTED / IMPLEMENTED = code exists / TESTED = tests exist and pass / VALIDATED = evidence proves intended behaviour,
   provenance, data flow, no blocking defect), the code paths, tests, evidence artifact, and what is missing. Flag every place a
   machine checklist's status disagrees with the evidence (stale rows, VALIDATED without evidence, FAILED that has since been fixed).
   Include a section for every item in C75 section 4 ("known failures/open problems"): reproduced? fixed? regression test? current
   evidence? For C67 say concretely whether mover-episode research runs continuously through the research brain or is a disconnected
   report generator.
2. state/build/C75_PHASE0_CORRECTIONS.json - a list of {checklist, id, current_status, evidenced_status, evidence, reason} that the
   main session will apply with scripts/contract_checklist.py (you do not apply them).
Report: counts per checklist per state, the top 20 gaps in C75 order, duplicates found, and anything that blocks Phase 1.
