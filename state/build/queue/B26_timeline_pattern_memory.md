# B26_timeline_pattern_memory
Canon C57 + C58 (read them verbatim in canon/CANON.md), C55, C56. Bible phases 4, 5, 9. Estimated code: 1,500-3,000 lines (a requirement).
You own: engine/pattern_memory.py, tests/test_pattern_memory.py, scripts/pattern_memory_demo.py, state/research/pattern_memory/. You may add a thin adapter in engine/pattern_bank.py and engine/memory.py (keep their APIs and tests passing). B22 is concurrently building the repeated-run learning curve in engine/learning_delta.py and will route carried state through your store - coordinate via the API below.

Build a TIMELINE PATTERN MEMORY that only ever improves, never forgets, and can never leak the future:
1. Store: for every pattern (identity-free key: feature/quintile/context conditions - never tickers) a timeline of observations
   (real_obs_date, real_mature_date = obs + label horizon in sessions, effect, n, t, context vector, source run id). Append-only,
   hash-chained, persisted under state/pattern_memory/ (git-ignored if large).
2. Query API used by the trader side: `view(real_now, ctx_now)` returns only observations with real_mature_date < real_now (strictly),
   aggregated per pattern with a RELEVANCE weight computed from the timeline (time distance to real_now, era/context similarity,
   recency of confirmations, number of distinct periods it held) - the real dates themselves are NEVER returned to the trader (C55):
   only pattern keys, pooled stats and weights.
3. Improvement: each run may add new candidate patterns (exploration of the combination space); every candidate ever tried is counted
   in a cumulative family-wise / FDR accounting so repeated search over the same data cannot pass noise (report cumulative tries).
4. Timeline analytics: when each pattern was first noticed, where it held/failed along real time, and its relevance to a given year.
5. Enforcement: a guard that raises if anything with real_mature_date >= real_now reaches the trader; tests with planted leaks
   (later-in-the-year evidence from an earlier run of the same year must be blocked at an earlier simulated date).
6. Tests: accumulation across runs; same-year rerun at an early date sees only matured earlier evidence; relevance weights rank a
   pattern active in similar periods above one active only in a distant era; cumulative multiple-testing blocks best-of-many noise.
Never run git; never kill by name.
