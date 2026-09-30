## final_look  (40 worlds; refused 0; mean 213 s/world system time)

### development: 31 worlds
- detectable real found 10/535 (1.9%); all promotions 181; precision 5.5%; false positives 171 = 5.52/world (max 16); null worlds 3 promoting [8, 0, 8]; Brier of confidence 0.1444
- false positives by noise kind: {'leak': 169, 'proxy': 2}
- found / detectable by kind: changing 0/84, conditional 1/92, delayed 0/29, interactive 0/2, lifecycle 0/74, linear 1/96, rare 0/37, regime 8/61, threshold 0/59, xor 0/1
- found / detectable by strength: extremely_subtle 0/20, faint 0/47, moderate 3/169, obvious 7/193, subtle 0/106
- block rate, real detectable not promoted (n=414): {'calibration': 0.957, 'replication': 0.928, 'complexity': 0.536, 'risk': 0.258, 'out_of_sample': 0.22, 'transfer': 0.155, 'identity': 0.01, 'leakage': 0.01}
- block rate, noise gated (n=1146): {'calibration': 0.712, 'replication': 0.688, 'complexity': 0.576, 'out_of_sample': 0.512, 'transfer': 0.403, 'risk': 0.32, 'identity': 0.096, 'leakage': 0.096, 'reproducibility': 0.021, 'failure_behavior': 0.014}

### heldout: 9 worlds
- detectable real found 1/168 (0.6%); all promotions 35; precision 2.9%; false positives 34 = 3.78/world (max 13); null worlds 1 promoting [6]; Brier of confidence 0.1371
- false positives by noise kind: {'leak': 34}
- found / detectable by kind: changing 0/24, conditional 1/27, delayed 0/15, interactive 0/2, lifecycle 0/23, linear 0/29, rare 0/11, regime 0/16, threshold 0/21
- found / detectable by strength: extremely_subtle 0/7, faint 0/16, moderate 0/51, obvious 1/60, subtle 0/34
- block rate, real detectable not promoted (n=128): {'calibration': 0.969, 'replication': 0.898, 'complexity': 0.562, 'risk': 0.289, 'out_of_sample': 0.211, 'transfer': 0.211, 'identity': 0.016, 'leakage': 0.016}
- block rate, noise gated (n=311): {'calibration': 0.778, 'replication': 0.698, 'complexity': 0.617, 'out_of_sample': 0.537, 'transfer': 0.395, 'risk': 0.354, 'identity': 0.106, 'leakage': 0.106, 'reproducibility': 0.032, 'failure_behavior': 0.003}
