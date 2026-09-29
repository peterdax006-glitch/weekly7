# Test inventory

41 test files, 613 tests. Level counts: unit 391, integration 32, historical 2, walk-forward 50, blind 10, adversarial 143, reproducibility 23

## Phases

| phase | title | status | tests | levels present | missing |
|---|---|---|---|---|---|
| 0 | BASELINE AND CONTROL SYSTEM | direct | 40 | 1,2,6 | - |
| 1 | POINT-IN-TIME DATA FIREWALL | direct | 38 | 1,4,6 | - |
| 2 | FEATURE/PARITY FIREWALL | direct | 47 | 1,2,4,6,7 | - |
| 3 | PATTERN MINER HARDENING | module | 65 | 1,3,4,6,7 | 2 |
| 4 | PATTERN LIFECYCLE | direct | 45 | 1,4,6,7 | - |
| 5 | LONG-TERM PATTERN BANK | direct | 26 | 1,6,7 | - |
| 6 | PATTERN → FIND VOLATILITY INTEGRATION | direct | 35 | 1,2,4,5,6,7 | - |
| 7 | HEAVY ALGORITHM TESTING | direct | 21 | 1,2,4,5,6,7 | - |
| 8 | ANALOG ENGINE | direct | 38 | 1,2,3,4,6,7 | - |
| 9 | MEMORY SYSTEM | direct | 27 | 1,2,4,6,7 | - |
| 10 | LESSON MEMORY / LEARNING FROM MISTAKES | direct | 19 | 1,4,5,6 | - |
| 11 | RERUN ANTI-MEMORIZATION EXPERIMENT | direct | 19 | 1,4,5,6 | - |
| 12 | PER-STOCK-TYPE TRUST TABLES | direct | 14 | 1,2,4,6,7 | - |
| 13 | DIRECTION ENGINE | direct | 24 | 1,2,6 | - |
| 14 | MISSED-WINNER DETECTOR | module | 25 | 1,2,4,6,7 | - |
| 15 | EXIT LEARNER | direct | 36 | 1,2,3,4,6,7 | - |
| 16 | STOP / LOSS ENGINE | direct | 13 | 1,2,3,4,6 | - |
| 17 | TIMELINE DIAL | direct | 21 | 1,4,6,7 | - |
| 18 | WEEKLY ADAPTER | module | 25 | 1,2,3,4,6,7 | - |
| 19 | TRAIN THE TRAINING BASIS | direct | 15 | 1,2,3,4,6,7 | - |
| 20 | TIERED OBJECTIVE FIREWALL | direct | 54 | 1,2,3,4,6,7 | - |
| 21 | BLIND SIMULATOR HARDENING | direct | 43 | 1,4,5,6,7 | - |
| 22 | BLIND CLOCK | direct | 33 | 1,4,5,6,7 | - |
| 23 | RE-TESTER | direct | 28 | 1,3,4,6,7 | - |
| 24 | WORKER HEALTH | direct | 20 | 1,6 | - |
| 25 | PLANTED-PATTERN CALIBRATION | direct | 65 | 1,3,4,6,7 | - |
| 26 | ANTI-OVERFITTING BATTERY | direct | 31 | 1,2,4,5,6,7 | - |
| 27 | DATA EXPANSION | direct | 31 | 1,4,6,7 | - |
| 28 | LIVE/RESEARCH SEPARATION | direct | 32 | 1,2,5,6 | - |
| 29 | PUBLIC EXPLANATION SYSTEM | none | 0 | - | 1,6 |
| 30 | EXPERIMENT MEMORY | module | 64 | 1,2,6,7 | - |
| 31 | CODE QUALITY FIREWALL | process | 0 | - | - |
| 32 | TEST PYRAMID | direct | 22 | 1,4,6,7 | - |
| 33 | REQUIRED REPRODUCIBILITY | direct | 11 | 1,7 | - |
| 34 | REQUIRED ABLATION FRAMEWORK | direct | 27 | 1,2,4,6,7 | - |
| 35 | CHAMPION / CHALLENGER SYSTEM | module | 22 | 1,6 | - |
| 36 | REQUIRED REPORT AFTER EACH MAJOR RUN | module | 16 | 1,6 | - |
| 37 | CHECKLIST STATE MACHINE | process | 0 | - | - |
| 38 | MASTER CHECKLIST | process | 0 | - | - |
| 39 | DEFINITION OF DONE | process | 0 | - | - |
| 40 | WHAT "KEEP WORKING" MEANS | process | 0 | - | - |
| 41 | NEVER SETTLE FOR A COSMETIC IMPLEMENTATION | process | 0 | - | - |
| 42 | THE STANDARD FOR EVERY CLAIM | process | 0 | - | - |
| 43 | PRIORITY WHEN TIME/COMPUTE IS LIMITED | process | 0 | - | - |
| 44 | RESOURCE MANAGEMENT | process | 0 | - | - |
| 45 | CONTINUOUS RESEARCH LOOP | process | 0 | - | - |
| 46 | FINAL REPORT REQUIREMENT | process | 0 | - | - |

## Checklist items

| id | item | status | tests |
|---|---|---|---|
| A1 | Candles and micro-signals | direct | 22 |
| A2 | Pattern miner | module | 65 |
| A2b | Week-clustered statistics | module | 65 |
| A3 | Heavy tests across eras | direct | 52 |
| A4 | Long-term pattern bank | direct | 26 |
| A5 | Pattern scores → Find Volatility | direct | 35 |
| A6 | Minute collector | none | 0 |
| A7 | Self-tuning Algorithm | direct | 25 |
| A8 | Market analog engine | direct | 38 |
| A8b | Sector analog engine | module | 24 |
| A8c | Stock analog engine | module | 24 |
| A9 | Timeline dial | direct | 21 |
| A10 | Lesson memory | direct | 19 |
| A10b | Rerun anti-memorization test | direct | 19 |
| A11 | Improve-or-discard lifecycle | direct | 45 |
| A12 | Additional data sources | direct | 31 |
| A13 | Planted-pattern calibration | direct | 65 |
| V1 | 95% mover target where sufficient candidates exist | direct | 35 |
| V2 | Direction ≥80% calibrated confidence | direct | 24 |
| V2b | Per-stock-type trust tables | direct | 14 |
| V3 | Exit learner | direct | 36 |
| V4 | Stop/risk learner | direct | 13 |
| V5 | 8-of-10 +10% objective across blind eras | direct | 54 |
| T1 | Archive | none | 0 |
| T2 | Blind loop | direct | 43 |
| T3 | Tiered 7% objective | direct | 54 |
| T4 | Weekly self-adjustment | module | 25 |
| T5 | Insider leak protection | direct | 38 |
| T6 | 13D correctness | direct | 38 |
| T7 | Explorer | direct | 15 |
| T8 | Sensitivity | module | 15 |
| T9 | Re-tester parity | direct | 38 |
| T10 | Future scramble | module | 25 |
| T11 | Time fence | direct | 25 |
| T12 | Worker health | direct | 20 |
| T13 | Reproducibility | direct | 36 |
| T14 | Label permutation | direct | 31 |
| T15 | Feature shuffle | direct | 31 |
| T16 | Ticker permutation | direct | 31 |
| T17 | Planted-pattern calibration | direct | 65 |
| L1 | Only consider upgrade after validated research edg | module | 22 |
| L2 | Verify paper-only safeguards | direct | 32 |
| L3 | Verify trading-hours firewall | direct | 32 |
| L4 | Verify broker safety | direct | 32 |
| L5 | Verify research/live isolation | direct | 32 |

## Gaps

- phases with modules but no test: [29]
- checklist items with no test: ['A6', 'T1']
- checklist items evidenced only by importing the module (no explicit tag): ['A2', 'A2b', 'A8b', 'A8c', 'T4', 'T8', 'T10', 'L1']
- missing pyramid levels by phase: {3: [2], 29: [1, 6]}
- test files matching no phase or item: ['tests/test_audit_registry.py']
- inventory config problems: none
