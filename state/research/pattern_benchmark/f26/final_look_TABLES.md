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
