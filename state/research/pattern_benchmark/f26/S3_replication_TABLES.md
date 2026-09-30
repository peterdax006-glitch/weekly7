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
