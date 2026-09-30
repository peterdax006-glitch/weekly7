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
