# F26 gate fixes vs the F19 real-vs-noise benchmark - IMPLEMENTED, NOT VALIDATED

Protocol: F19 final-look (one screen + gate at the world's last date), seeds 0-19 = 14 development + 6 held-out (the F19 baseline set).
Stages are CUMULATIVE; each was run with its code frozen (EvidenceConfig f26_* switches off for the fixes not yet in). Diagnosis and
tuning used development seeds 2, 3 and 9 only (pattern_benchmark.tuning_seeds); held-out worlds were only ever scored inside the frozen
stage runs. Every score comes from the sealed keys + frozen answers (pattern_benchmark.rescore re-verifies each seal; refused = 0).
Answer key, scoring, world generator and detectability floor are untouched (make_world output is byte-identical).
File sha256[:16] at the S4 run: evidence.py, quality_gate.py, pattern_benchmark.py - see the report; per-run code stamps in provenance_part*.json.

## Headline per stage

| stage | split | worlds | real found / detectable | FP / world | FP kinds | precision | null world promotions | s / world | refused |
|---|---|---|---|---|---|---|---|---|---|
| F19 baseline | development | 14 | 2/246 (0.8%) | 5.64 | {'leak': np.int64(78), 'proxy': np.int64(1)} | 2.5% | [8] | 215 | 0 |
| F19 baseline | heldout | 6 | 1/100 (1.0%) | 5.50 | {'leak': np.int64(33)} | 2.9% | [6] | 215 | 0 |
| S1 leak screen + honest multiplicity | development | 14 | 2/246 (0.8%) | 0.21 | {'leak': np.int64(2), 'proxy': np.int64(1)} | 40.0% | [0] | 65 | 0 |
| S1 leak screen + honest multiplicity | heldout | 6 | 1/100 (1.0%) | 0.00 | {} | 100.0% | [0] | 65 | 0 |
| S2 + claim calibration | development | 14 | 10/246 (4.1%) | 0.21 | {'leak': np.int64(2), 'proxy': np.int64(1)} | 76.9% | [0] | 72 | 0 |
| S2 + claim calibration | heldout | 6 | 5/100 (5.0%) | 0.17 | {'proxy': np.int64(1)} | 83.3% | [0] | 72 | 0 |
| S3 + replication power design | development | 14 | 40/246 (16.3%) | 0.43 | {'proxy': np.int64(4), 'leak': np.int64(2)} | 87.0% | [0] | 74 | 0 |
| S3 + replication power design | heldout | 6 | 13/100 (13.0%) | 0.50 | {'proxy': np.int64(3)} | 81.2% | [0] | 74 | 0 |
| S4 + magnitude risk + fold-unit complexity (all fixes) | development | 14 | 71/246 (28.9%) | 0.93 | {'proxy': np.int64(9), 'leak': np.int64(3), 'identity_null': np.int64(1)} | 84.5% | [0] | 69 | 0 |
| S4 + magnitude risk + fold-unit complexity (all fixes) | heldout | 6 | 25/100 (25.0%) | 1.33 | {'proxy': np.int64(7), 'identity_null': np.int64(1)} | 75.8% | [0] | 69 | 0 |

Full protocol (every look 156..208 step 13, all fixes, development seeds 2 and 5): 250 / 259 s per world; gated per look
53 -> 50 -> 21 -> 19 -> 17 (was growing: FAILED candidates never retired); 39 / 45 candidates retired after two consecutive FAILED looks.

## Detail per stage (the final_look block is F19's run restricted to seeds 0-19)

## final_look  (20 worlds; refused 0; mean 215 s/world system time)

### development: 14 worlds
- detectable real found 2/246 (0.8%); all promotions 81; precision 2.5%; false positives 79 = 5.64/world (max 11); null worlds 1 promoting [8]; Brier of confidence 0.1416
- false positives by noise kind: {'leak': 78, 'proxy': 1}
- found / detectable by kind: changing 0/42, conditional 0/42, delayed 0/14, lifecycle 0/33, linear 0/50, rare 0/17, regime 2/19, threshold 0/29
- found / detectable by strength: extremely_subtle 0/10, faint 0/21, moderate 1/78, obvious 1/88, subtle 0/49
- block rate, real detectable not promoted (n=200): {'calibration': 0.955, 'replication': 0.915, 'complexity': 0.52, 'risk': 0.25, 'out_of_sample': 0.2, 'transfer': 0.125, 'identity': 0.02, 'leakage': 0.02}
- block rate, noise gated (n=480): {'calibration': 0.758, 'replication': 0.717, 'complexity': 0.627, 'out_of_sample': 0.556, 'transfer': 0.425, 'risk': 0.267, 'identity': 0.112, 'leakage': 0.112, 'reproducibility': 0.023, 'failure_behavior': 0.019}

### heldout: 6 worlds
- detectable real found 1/100 (1.0%); all promotions 34; precision 2.9%; false positives 33 = 5.50/world (max 13); null worlds 1 promoting [6]; Brier of confidence 0.1323
- false positives by noise kind: {'leak': 33}
- found / detectable by kind: changing 0/15, conditional 1/18, delayed 0/11, interactive 0/1, lifecycle 0/13, linear 0/18, rare 0/6, regime 0/6, threshold 0/12
- found / detectable by strength: extremely_subtle 0/4, faint 0/8, moderate 0/32, obvious 1/38, subtle 0/18
- block rate, real detectable not promoted (n=77): {'calibration': 0.974, 'replication': 0.909, 'complexity': 0.558, 'risk': 0.26, 'transfer': 0.221, 'out_of_sample': 0.169, 'identity': 0.013, 'leakage': 0.013}
- block rate, noise gated (n=197): {'replication': 0.695, 'calibration': 0.69, 'complexity': 0.609, 'out_of_sample': 0.548, 'transfer': 0.421, 'risk': 0.299, 'identity': 0.112, 'leakage': 0.112, 'reproducibility': 0.041, 'failure_behavior': 0.005}

## S1_leak_multiplicity  (20 worlds; refused 0; mean 65 s/world system time)

### development: 14 worlds
- detectable real found 2/246 (0.8%); all promotions 5; precision 40.0%; false positives 3 = 0.21/world (max 2); null worlds 1 promoting [0]; Brier of confidence 0.1416
- false positives by noise kind: {'leak': 2, 'proxy': 1}
- found / detectable by kind: changing 0/42, conditional 0/42, delayed 0/14, lifecycle 0/33, linear 0/50, rare 0/17, regime 2/19, threshold 0/29
- found / detectable by strength: extremely_subtle 0/10, faint 0/21, moderate 1/78, obvious 1/88, subtle 0/49
- block rate, real detectable not promoted (n=200): {'calibration': 0.955, 'replication': 0.915, 'complexity': 0.52, 'out_of_sample': 0.305, 'risk': 0.25, 'transfer': 0.125, 'identity': 0.02, 'leakage': 0.02}
- block rate, noise gated (n=480): {'calibration': 0.758, 'replication': 0.717, 'complexity': 0.627, 'out_of_sample': 0.596, 'transfer': 0.425, 'leakage': 0.402, 'risk': 0.267, 'identity': 0.112, 'reproducibility': 0.023, 'failure_behavior': 0.019}

### heldout: 6 worlds
- detectable real found 1/100 (1.0%); all promotions 1; precision 100.0%; false positives 0 = 0.00/world (max 0); null worlds 1 promoting [0]; Brier of confidence 0.1323
- false positives by noise kind: none
- found / detectable by kind: changing 0/15, conditional 1/18, delayed 0/11, interactive 0/1, lifecycle 0/13, linear 0/18, rare 0/6, regime 0/6, threshold 0/12
- found / detectable by strength: extremely_subtle 0/4, faint 0/8, moderate 0/32, obvious 1/38, subtle 0/18
- block rate, real detectable not promoted (n=77): {'calibration': 0.974, 'replication': 0.909, 'complexity': 0.558, 'out_of_sample': 0.351, 'risk': 0.26, 'transfer': 0.221, 'identity': 0.013, 'leakage': 0.013}
- block rate, noise gated (n=197): {'replication': 0.695, 'calibration': 0.69, 'out_of_sample': 0.619, 'complexity': 0.609, 'transfer': 0.421, 'leakage': 0.416, 'risk': 0.299, 'identity': 0.112, 'reproducibility': 0.041, 'failure_behavior': 0.005}

## S2_calibration  (20 worlds; refused 0; mean 72 s/world system time)

### development: 14 worlds
- detectable real found 10/246 (4.1%); all promotions 13; precision 76.9%; false positives 3 = 0.21/world (max 2); null worlds 1 promoting [0]; Brier of confidence 0.1416
- false positives by noise kind: {'leak': 2, 'proxy': 1}
- found / detectable by kind: changing 2/42, conditional 2/42, delayed 0/14, lifecycle 0/33, linear 3/50, rare 0/17, regime 3/19, threshold 0/29
- found / detectable by strength: extremely_subtle 0/10, faint 0/21, moderate 1/78, obvious 9/88, subtle 0/49
- block rate, real detectable not promoted (n=192): {'replication': 0.953, 'complexity': 0.542, 'out_of_sample': 0.318, 'risk': 0.26, 'transfer': 0.13, 'calibration': 0.078, 'identity': 0.021, 'leakage': 0.021}
- block rate, noise gated (n=480): {'replication': 0.717, 'complexity': 0.627, 'out_of_sample': 0.596, 'transfer': 0.425, 'leakage': 0.402, 'risk': 0.267, 'calibration': 0.14, 'identity': 0.112, 'reproducibility': 0.023, 'failure_behavior': 0.019}

### heldout: 6 worlds
- detectable real found 5/100 (5.0%); all promotions 6; precision 83.3%; false positives 1 = 0.17/world (max 1); null worlds 1 promoting [0]; Brier of confidence 0.1323
- false positives by noise kind: {'proxy': 1}
- found / detectable by kind: changing 1/15, conditional 3/18, delayed 0/11, interactive 0/1, lifecycle 0/13, linear 1/18, rare 0/6, regime 0/6, threshold 0/12
- found / detectable by strength: extremely_subtle 0/4, faint 0/8, moderate 0/32, obvious 5/38, subtle 0/18
- block rate, real detectable not promoted (n=73): {'replication': 0.959, 'complexity': 0.589, 'out_of_sample': 0.37, 'risk': 0.274, 'transfer': 0.233, 'calibration': 0.096, 'identity': 0.014, 'leakage': 0.014}
- block rate, noise gated (n=197): {'replication': 0.695, 'out_of_sample': 0.619, 'complexity': 0.609, 'transfer': 0.421, 'leakage': 0.416, 'risk': 0.299, 'calibration': 0.157, 'identity': 0.112, 'reproducibility': 0.041, 'failure_behavior': 0.005}

## S3_replication  (20 worlds; refused 0; mean 74 s/world system time)

### development: 14 worlds
- detectable real found 40/246 (16.3%); all promotions 46; precision 87.0%; false positives 6 = 0.43/world (max 3); null worlds 1 promoting [0]; Brier of confidence 0.1416
- false positives by noise kind: {'proxy': 4, 'leak': 2}
- found / detectable by kind: changing 6/42, conditional 9/42, delayed 0/14, lifecycle 8/33, linear 9/50, rare 0/17, regime 6/19, threshold 2/29
- found / detectable by strength: extremely_subtle 0/10, faint 0/21, moderate 12/78, obvious 26/88, subtle 2/49
- block rate, real detectable not promoted (n=162): {'replication': 0.759, 'complexity': 0.642, 'out_of_sample': 0.377, 'risk': 0.309, 'transfer': 0.154, 'calibration': 0.093, 'identity': 0.025, 'leakage': 0.025}
- block rate, noise gated (n=480): {'replication': 0.671, 'complexity': 0.627, 'out_of_sample': 0.596, 'transfer': 0.425, 'leakage': 0.402, 'risk': 0.267, 'calibration': 0.14, 'identity': 0.112, 'reproducibility': 0.023, 'failure_behavior': 0.019}

### heldout: 6 worlds
- detectable real found 13/100 (13.0%); all promotions 16; precision 81.2%; false positives 3 = 0.50/world (max 1); null worlds 1 promoting [0]; Brier of confidence 0.1323
- false positives by noise kind: {'proxy': 3}
- found / detectable by kind: changing 2/15, conditional 3/18, delayed 0/11, interactive 0/1, lifecycle 2/13, linear 5/18, rare 0/6, regime 1/6, threshold 0/12
- found / detectable by strength: extremely_subtle 0/4, faint 0/8, moderate 5/32, obvious 7/38, subtle 1/18
- block rate, real detectable not promoted (n=65): {'replication': 0.769, 'complexity': 0.662, 'out_of_sample': 0.415, 'risk': 0.308, 'transfer': 0.262, 'calibration': 0.108, 'identity': 0.015, 'leakage': 0.015}
- block rate, noise gated (n=197): {'replication': 0.624, 'out_of_sample': 0.619, 'complexity': 0.609, 'transfer': 0.421, 'leakage': 0.416, 'risk': 0.299, 'calibration': 0.157, 'identity': 0.112, 'reproducibility': 0.041, 'failure_behavior': 0.005}

## S4_risk_complexity  (20 worlds; refused 0; mean 69 s/world system time)

### development: 14 worlds
- detectable real found 71/246 (28.9%); all promotions 84; precision 84.5%; false positives 13 = 0.93/world (max 4); null worlds 1 promoting [0]; Brier of confidence 0.1416
- false positives by noise kind: {'proxy': 9, 'leak': 3, 'identity_null': 1}
- found / detectable by kind: changing 12/42, conditional 12/42, delayed 0/14, lifecycle 18/33, linear 13/50, rare 0/17, regime 8/19, threshold 8/29
- found / detectable by strength: extremely_subtle 0/10, faint 0/21, moderate 23/78, obvious 45/88, subtle 3/49
- block rate, real detectable not promoted (n=131): {'replication': 0.939, 'out_of_sample': 0.466, 'complexity': 0.435, 'transfer': 0.191, 'calibration': 0.115, 'identity': 0.031, 'leakage': 0.031}
- block rate, noise gated (n=480): {'replication': 0.671, 'out_of_sample': 0.596, 'complexity': 0.488, 'transfer': 0.425, 'leakage': 0.402, 'calibration': 0.14, 'identity': 0.112, 'reproducibility': 0.023, 'failure_behavior': 0.019}

### heldout: 6 worlds
- detectable real found 25/100 (25.0%); all promotions 33; precision 75.8%; false positives 8 = 1.33/world (max 3); null worlds 1 promoting [0]; Brier of confidence 0.1323
- false positives by noise kind: {'proxy': 7, 'identity_null': 1}
- found / detectable by kind: changing 5/15, conditional 5/18, delayed 0/11, interactive 0/1, lifecycle 3/13, linear 8/18, rare 0/6, regime 3/6, threshold 1/12
- found / detectable by strength: extremely_subtle 0/4, faint 0/8, moderate 8/32, obvious 15/38, subtle 2/18
- block rate, real detectable not promoted (n=53): {'replication': 0.943, 'out_of_sample': 0.509, 'complexity': 0.453, 'transfer': 0.321, 'calibration': 0.132, 'identity': 0.019, 'leakage': 0.019}
- block rate, noise gated (n=197): {'replication': 0.624, 'out_of_sample': 0.619, 'complexity': 0.518, 'transfer': 0.421, 'leakage': 0.416, 'calibration': 0.157, 'identity': 0.112, 'reproducibility': 0.041, 'failure_behavior': 0.005}
