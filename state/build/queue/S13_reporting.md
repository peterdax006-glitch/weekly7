# S13_reporting
Contract sections 46 (dashboard), 47, 48, 59, 79 (K01-K15), J15; line minimum "Reporting/dashboard/inspection" 1,500 (ruler; existing reporting code counted so far: 1,166 - build on it).
Read state/build/CONTEXT.md fully (rules 1-19) and SELF_LEARNING_CONTRACT.md sections 46-48, 59, 65, 79, 85. C63: code + unit tests now.
You own: engine/learning/reports.py, scripts/learning_report.py, tests/test_learning_reports.py. You may import every engine/learning
module (read its public API first; do not edit). Build on scripts/run_report.py, scripts/final_report.py, scripts/dashboard_build.py
(read-only) and their conventions (audits fail closed; numbers are computed from artefacts, NEVER typed - memory: prose goes stale).
Build one generator per report K01-K15 (learning curve, transfer, memorisation, identity gap, knowledge health, failure, recovery,
contradiction, missed winner, experiment memory, research priority, meta-learning, promotion/rejection, provenance audit, full
learning-system report) plus the J15 research-priority dashboard data and the section-46 health dashboard data. Each report:
reads its artefacts (state/..., the checklist JSON, module stores), states the section-59 label honestly (IMPLEMENTED — NOT
VALIDATED / FAILED VALIDATION / INSUFFICIENT EVIDENCE / VALIDATED only with evidence), refuses to say "improved" without its
controls (scorecard.gate_improvement_claim), and renders markdown + JSON. Missing artefact -> the report says UNMEASURED (never
blank, never success). scripts/learning_report.py writes all of them to state/research/learning_reports/. Tests: each report on
synthetic stores; the missing-artefact case; a planted false claim ("learning improved" without controls) is refused.
