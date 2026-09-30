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
