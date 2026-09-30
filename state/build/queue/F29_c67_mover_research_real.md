# F29_c67_mover_research_real  [model: opus]
C75 Phase 1C (C67 mover-episode foundation) and canon C67 (verbatim in canon/CANON.md). Read CONTEXT.md (rules 1-29) and
state/build/C75_PHASE0_MAPPING.md (the C67 section).
Findings (F20): mover-episode research is wired as the loop stage questions.precursors + an always-on sweep, but in every clean loop it
swept 3 planted year-units on 48 names and went idle - 0 candidates, 0 questions - and it has studied 0 REAL episodes (the loop never
ran with --source real). The COLLAPSE label has no code; acceleration, close-location and intraday-bar labels are partial.
C75 1C requires, for hundreds of stocks per day entering the 5-10% range: consolidation, continuation, acceleration, spike, reversal,
collapse, stagnation, gap behaviour, intraday path, close-to-close, open-to-close, range, close location, next-day and next-week
behaviour - searching for even very small predictable patterns, continuously, THROUGH the research brain (not a report generator).
You OWN: engine/research/episodes.py, episode_paths.py, precursors.py (and the mover-episode lab module R21 built - name it), their
tests, NEW scripts for the real-data sweep. Never edit loop.py, evidence.py, quality_gate.py, pattern_benchmark.py, candidate_forms.py
(F28 is running), feeds.py (report needed hooks). Never run git. <= 2 processes. Data: the real caches under data/cache (RAM >= 2.5 GB
before loading; stream by year). Point-in-time only: an episode's label uses data AFTER the entry day only as the OUTCOME; every
precursor must be known at the entry close (audit it with the F09 truncation method). The price panel is survivor-only: label every
real-data result SURVIVOR_ONLY.
Do: 1. Complete every label in the list above (collapse new; finish acceleration, close location, intraday path where minute/intraday
data exist - say which fields are unavailable and represent them as UNMEASURED, never guessed). 2. Make the sweep keep producing
work: why did it go idle after 3 year-units on the planted world? Fix it (continuous, resumable, budgeted). 3. Run it on REAL data:
>= 5 years of daily history, hundreds of 5-10% movers per day, episodes labelled, precursor candidates raised INTO the research brain's
question/candidate path (show the questions/candidates it created), every candidate counted for multiplicity. 4. Tests: planted
precursor found; null panel yields none beyond its budget; truncation audit clean; resume after kill. Report counts per label, per year,
top precursor candidates with their out-of-sample checks (not gated promotions), runtime, and what is UNMEASURED.
