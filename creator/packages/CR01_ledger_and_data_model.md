# CR01 - Development ledger and data model  [model: opus]
PACKAGE ID: CR01 (C77 sec 63 format). Governing text: C:\Users\Peter\weekly7\CREATOR_MASTER_PROMPT.md (C77, verbatim) - read sections
7, 10, 13, 18, 22, 30, 31, 35, 37, 58, 59, 61, 63, 71; architecture: creator/ARCHITECTURE.md (component K01). CONTEXT.md rules 1-29 apply.
OBJECTIVE: the Creator's single source of truth - an append-only, hash-chained, provenance-stamped development ledger with typed records.
WHY IT EXISTS: every later component records and reads development state through it; C77 forbids silent state mutation (sec 61), requires
traceable changes (22), provenance on every claim (35), durable development memory (37) and computed, not typed, verdicts.
PREREQUISITES: none (first package). INPUTS: engine/provenance.py (stamp, engine_tree_hash, verify_integrity) - reuse, do not duplicate;
look at engine/learning/archive.py chain files and reuse their hash-chain primitive if suitable (say which you reused).
IMPLEMENTATION REQUIREMENTS (package creator/, module creator/ledger.py + creator/model.py):
- Typed frozen records with schema validation and versioned schema: Objective, Requirement, Capability, Gap, ResearchQuestion, Finding,
  DesignOption, WorkPackage (ALL C77 sec 63 fields), ChangeProposal, Experiment, TestRun, Failure, Diagnosis, Repair, Measurement,
  ImprovementClaim, Decision (ADOPT/ROLLBACK/REJECT with reason), StrategyOutcome, CheckpointMarker. Status fields use exactly the C77 sec 7
  states; uncertainty fields use the sec 71 states. Every record: id, parent ids (objective/task lineage), created_by role, provenance
  (engine_tree_hash + git commit + config hash + seed + timestamp), evidence refs (paths + sha256).
- Append-only JSONL store with a hash chain (each record carries prev hash); verify() detects any edit/deletion/reorder; concurrent-writer
  safety (lock file) ; crash-safe append (no torn records accepted on reload).
- Status transitions are themselves records; an illegal transition (e.g. NOT_STARTED -> VALIDATED without TESTED evidence, or VALIDATED
  without an evidence artifact whose hash verifies) is refused (fail closed). Verdict fields (IMPROVEMENT, VALIDATED, ...) can only be set
  by a record that references the computation that produced them.
- Query API: by type, lineage, status, time; "current state" view folded from the event log; export of a snapshot for observability.
INTERFACES: creator.ledger.Ledger(path).append(record) / verify() / view(...) / transition(id, new_state, evidence) ; creator.model types.
DATA FLOW: all later CR packages write/read here only.
TEST REQUIREMENTS (tests/test_creator_ledger.py, < 90 s): schema validation; append/reload; chain tamper detection (edit, delete, reorder,
truncate); torn final line; concurrent appends; illegal transitions refused; verdict without computation refused; lineage queries;
empty ledger; provenance present on every record.
EXPECTED FAILURE MODES: silent overwrite, chain verify that passes on a reordered file, transitions that skip TESTED, hashing non-canonical JSON.
EVIDENCE: test output + a short creator/packages/CR01_REPORT.md (computed numbers only).
ROLLBACK: package is additive (new files only); nothing in engine/ changes.
COMPLETION CRITERIA: all of the above implemented and tested; mypy clean (add creator/ to the [tool.mypy] files list in pyproject.toml);
ruler >= 900 meaningful lines for K01 (ARCHITECTURE sec 7) WITHOUT padding.
ANTI-PREMATURE-COMPLETION: state IMPLEMENTED/TESTED only - never VALIDATED (validation needs later use by the kernel + audit).
MEANINGFUL CODE DEPTH: floor 900 (K01). STATUS: NOT_STARTED. Never run git; <= 1 process.
