# S19_blind_trader_curator
Canon C64 (THE governing directive - read it verbatim in canon/CANON.md), plus C55, C56, C57, C58, C59. Contract sections 15, 17,
28, 29, 30, 49, 55. Target 1,200-1,800 meaningful lines (ruler) + tests.
Read state/build/CONTEXT.md fully (rules 1-19) and SELF_LEARNING_CONTRACT.md sections 15, 17, 28-30, 49, 55, 85. C63: code + unit
tests now; no real-data runs. You may import every engine/learning module (do not edit them).
You own: engine/learning/curator.py, engine/learning/trader_view.py, tests/test_learning_curator.py.

Owner's design (C64): TWO SIDES.
- TRADER side (blind): the running system never knows the year. It receives information day by day, only as it became available.
  It is never given a real date/year in any object, field, key, name, file path or string.
- CURATOR side (trusted, automatic, invisible to the trader): every memory is FILED under the real year/date it operated in.
  "Relative memory" lives here: the curator knows the year, decides which memories are relevant now (era similarity of market
  state, consistency across periods C59, reliability, C58 matured-before-real-now timeline) and PRIORITISES them - handing the
  trader only weighted, anonymised, date-free memory. The trader cannot see why one memory outranks another.

Build:
1. trader_view.py: frozen, typed trader-side records (TraderMemoryItem, TraderSituation, ...) whose constructors REJECT any field,
   value or string that looks like a real date/year (ISO dates, 4-digit years 1900-2100 in keys/values/ids, datetime objects,
   epoch-like ints in date ranges) - reuse the identity/date regex lessons (\b misses underscores; use lookarounds). A
   `scrub()` that strips/refuses and counts. A static check (AST) that modules on the trader path (engine/livesim.py Feed consumers,
   engine/learning/learner.py decide path, engine/adaptive.py) do not import curator internals or read the curator's store.
2. curator.py: Curator(store_root) on the trusted side. file(item, real_date, real_year, context) appends to the year-filed store
   (reuse archive.py lanes / pattern_memory hash chain - no new store); relevance(real_now, market_state) scores each memory from
   era similarity (on the m_* state, computed trusted-side), C59 consistency, reliability (calibration/temporal), and the C58
   filter (outcome matured strictly before real_now); release(real_now, market_state, k) returns TraderMemoryItems with weights,
   no dates, no years, no explanation of the ranking. An audit trail of every release is kept TRUSTED-side (for reports), never
   passed to the trader.
3. Daily release: Curator.day(real_date) is called by the trusted clock once per simulated day; information grows day by day;
   anything not yet available that day is refused (FirewallBreach) - reuse future_firewall / memory_firewall.
4. Channel-6 measurement switch (for later testing, not wired as default): relative_m_context(frame, lookback) converts each m_*
   input to its position relative to its own trailing history (percentile/z over the past N years, past-only) and drops absolute
   levels; plus a helper that reports how much year-identifiability and how much signal a transform removes (reuse the
   leak_audit fingerprint probe API read-only).
Tests: a planted real date in a memory's payload/id/key is refused or scrubbed; the trader view of a release contains no year even
under adversarial payloads (e.g. "AAPL_2008-09-15", 2008 as a float, a datetime); relevance ranking changes with era without
anything year-like reaching the trader; a memory whose outcome matured after real_now is never released (C58); two identical
situations in different real years get different curator priorities while the trader sees identical-looking items; the static
import check catches a planted trader->curator import; the relative_m_context transform is past-only (truncation invariance).
