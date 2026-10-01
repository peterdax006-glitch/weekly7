# F32_gate_b_depth_audit  [model: opus]  (READ-ONLY: write only the two output files)
Canon C76 = the CURRENT master checkoff list: C:\Users\Peter\weekly7\MASTER_EXECUTION_PROMPT.md (read sections 31-37 Gate B and the
S01-S21 work packages, sections 9-29). Ruler: scripts/contract_lines.py (meaningful lines; the ONLY ruler). Read CONTEXT.md rules 1-29.
Inputs: C62 57-row budget in state/build/SELF_LEARNING_MASTER_CHECKLIST.json ("line_budget"); C66 budget rows in
state/build/RESEARCH_BRAIN_CHECKLIST.json; the C76 B01-B21 ranges (section 33) and S01-S21 minimums (sections 9-29); C67 2,500-4,000;
state/build/C75_PHASE0_MAPPING.md; the code (engine/, scripts/, tests/).
Write:
1. state/build/C76_GATE_B_DEPTH.md - one row per budget item (all 57 C62 rows, every C66 row, B01-B21, S01-S21, C67, C68): the
   authoritative floor/range and its source line, the module(s) honestly owning it (name the mapping rule; never count an unrelated
   module to reach a floor; a module counted for two budgets must be split honestly and say so), the measured meaningful lines (ruler
   output), tests' meaningful lines where the contract counts tests, BELOW / IN RANGE / ABOVE, and for every BELOW row the concrete
   missing CAPABILITIES (from the contract text) that would close the gap - not "add N lines". Flag padding risks you see (duplicated
   blocks, dead code, generated boilerplate) in counted modules.
2. state/build/C76_GATE_B_WORKPLAN.json - the below-floor rows grouped into builder work packages (each: id, rows covered, owning files,
   missing capabilities, floor shortfall, suggested model sonnet|opus, files that must NOT be touched because other packages own them),
   ordered by C76 priority (firewalls/anti-cheating and integration first).
Never edit code, checklists or tests; never run git except read-only; <= 1 process (F28 is running a large benchmark).
Report: counts BELOW / IN RANGE / ABOVE per budget family, the total shortfall in meaningful lines, the top 10 gaps, padding risks.
