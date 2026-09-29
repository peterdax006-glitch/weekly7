# R21_mover_episode_lab  [model: sonnet]
Canon C67 (THE governing directive - read it verbatim in canon/CANON.md) + C66 sections 4, 5, 6, 9, 13, 21, 22, 23.
Target 2,500-4,000 meaningful lines (ruler) + tests. Read state/build/CONTEXT.md fully (rules 1-27), engine/research/core.py,
engine/learning/core.py, RESEARCH_BRAIN_CONTRACT.md sections 0, 4-6, 9, 13, 21-23, 29-31, 44, 53. C63: code + unit tests only.
You own: engine/research/episodes.py, engine/research/episode_paths.py, engine/research/precursors.py,
tests/test_research_episodes.py.
Build on (import, never copy): engine/features.py, engine/candles.py, data/cache daily OHLCV (stocks_open/high/low/close/volume
+ stocks_pre2000_*, via engine.data loaders - read how they load), the minute collector output where it exists (recent only),
engine/fv_pipeline.py, engine/pit.py, engine/learning/situation.py. Parallel builders you must align with (duck-type their
outputs; they are being written now): R04 observer.py/autopsy.py, R07 volatility_lab.py, R10 discovery.py, R16 multiscale.py.

The owner's design (C67): every simulated day, look at HUNDREDS of stocks that moved 5-10% (and >10% as its own band), and at
what they did next; study them continuously ("24/7") across tons of stocks to find even the smallest patterns that could have
been predicted beforehand.
1. Episode detector (streaming year by year - ~38M name-days will not fit in RAM; keep only episode rows): per name-day, the
   move measured several ways from daily OHLC - close-to-close return, open-to-close, intraday range (high-low)/prev close,
   gap (open vs prev close), close location in the range - banded into 5-10% and >10%, up and down. Hundreds per day expected
   on a ~3,000-name universe: record the count per day and per band.
2. Path taxonomy for what happened NEXT (1, 2, 3 and 5 sessions later): CONSOLIDATED (range contracted), EXPANDED (moved way
   more), STOPPED (no follow-through), SPIKED_NEXT_DAY (large next-day move same direction), REVERSED_NEXT_DAY, plus the
   intraday variant "volatile during the day then got way more volatile" (range expansion day over day) and gap-and-fade /
   gap-and-go. Thresholds are parameters relative to the name's own trailing volatility; unlabelled paths stay UNCLASSIFIED.
3. Precursor research: for each episode type and each path class, the point-in-time state BEFORE the move (and before the
   next-day outcome), strictly past-only features (volume expansion, compression, prior ranges, gaps, cross-sectional rank,
   sector/market moves, events/insider/filings with publication lags, candle structure, existing patterns). Contrast
   precursors against matched NON-episodes (same date, same volatility cohort) and against other path classes (e.g. what
   separates reversals from continuations). Every candidate precursor is counted in a cumulative multiple-testing ledger and
   must survive walk-forward, era and shuffled-label controls before it is anything more than a hypothesis. Output candidate
   patterns as research questions / knowledge-object candidates (R10's discovery objects when available, else plain records).
4. Always-on mode: an iterator that keeps sweeping years/universes indefinitely with checkpoints and resumption (a job the
   wave-2 research loop will schedule through compute_manager), recording coverage (which years/names/bands are done) so the
   sweep never redoes finished work and always moves to the least-covered area.
5. Firewall: episodes and path labels are MATURED_RESEARCH_STATE (they use the future by definition); nothing from them
   reaches the trader except through MaturedRecord.gate(now) with maturity strictly before now; no real dates/tickers in
   anything trader-facing.
Tests: planted synthetic panel with a known precursor (e.g. volume spike two days before a 7% move) must be found; a planted
reversal-vs-continuation separator found; a noise-only panel finds nothing after correction; path classifier labels planted
paths exactly; streaming yields identical results to in-memory on a small panel; a future-leaking precursor (uses the move
day's own close) is refused; empty year; resume after interruption does not redo or skip work.
