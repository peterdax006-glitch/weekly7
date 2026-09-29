# Blind gates on real caches

seed 20260928, 12.2 s, sample (16292, 247)

## Seals
40 seals: 0 failing, 0 overlapping pairs.
Per decade: {'1965-1974': 8, '1975-1984': 7, '1985-1994': 5, '1995-2004': 8, '2005-2014': 7, '2015-2025': 5}; per era: {'pre1997': 22, '2001+': 16, '1997-2000': 2}

## Disguise per era
- 1997-2000: {'windows': 1, 'hard_fails': 0, 'dup_paths': 0}
- 2001+: {'windows': 3, 'hard_fails': 0, 'dup_paths': 0}
- pre1997: {'windows': 8, 'hard_fails': 0, 'dup_paths': 0}

## Look-ahead probe
- 1981-06-25: honest passes True, peeker caught True
- 1989-03-20: honest passes True, peeker caught True
- 1996-12-10: honest passes True, peeker caught True
- 2004-09-20: honest passes True, peeker caught True
- 2012-06-22: honest passes True, peeker caught True
- 2020-04-01: honest passes True, peeker caught True

## Re-test
- faithful: PASS: 819 items across 7 components
- one_day_leak: FAIL: 570 items across 7 components; 367 unexplained divergence(s), first: holdings/('2012-03-01', 'CLS') present in only archive
- fill_drift_2pct: FAIL: 819 items across 7 components; 169 unexplained divergence(s), first: trades/2012-03-02/BBGI/buy.price 58.2729 vs 59.4384
- other_code: STALE_CODE: 0 items across 0 components (archive code_hash codeA != replay codeB: stale code, not a parity failure)

## Health
{'OK': 8, 'CRASHED': 1, 'OOM': 1, 'TIMEOUT': 1, 'INVALID': 1, 'STALE_CODE': 1}
