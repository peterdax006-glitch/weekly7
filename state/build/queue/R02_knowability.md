# R02_knowability  [model: sonnet]
C66 §7 (2,500-4,000), §33 unknown-cause system (1,000-1,800).
Read state/build/CONTEXT.md fully (rules 1-26), engine/research/core.py, engine/learning/core.py, and in RESEARCH_BRAIN_CONTRACT.md sections 0, 1, 29-31, 43, 44, 48, 53 plus the sections below, in full. If state/build/RESEARCH_MAPPING.md exists, read its row for your components first. C63: code + unit tests only.
You own: engine/research/knowability.py, engine/research/unknown_cause.py; tests/test_research_knowability.py.
Build on (import, extend - never copy): engine/learning/failure.py (UNKNOWN never forced), break_detection.py, pattern_reliability.py (unknown-cause rule), engine/edgar.py/insider/events data availability, engine/pit.py (publication lags).

Build: Classify each major move PREDICTABLE / POTENTIALLY / WEAKLY / UNKNOWN / EXTERNALLY_CAUSED / INFORMATIONALLY_UNAVAILABLE / DATA_FAILURE, with each information item tagged KNOWN_BEFORE / ONLY_AFTER / SIMULTANEOUS / UNCERTAIN / UNAVAILABLE by timestamp + provenance. Audit world only (MATURED_RESEARCH). Unknown-cause system: UNKNOWN_CAUSE first-class; rates by regime/volatility/sector/confidence; detect 'unknown rate fell because labels were forced' vs genuine discovery (an explanation needs evidence beating a random-explanation null).
