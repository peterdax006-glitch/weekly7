# B24_generalising_learner
Canon C54/C55/C56 (read them). The owner's product is the LEARNING system. Estimated code: 1,500-3,000 lines (a requirement).
You own: engine/learners.py, scripts/learner_search.py, tests/test_learners.py, state/research/learners/. You may add a learner hook to engine/learning_delta.py (keep its tests passing) and read everything else.

Fact base (measured 2026-09-29 through engine/learning_delta.py, harness self-check VALID):
- memory-bank learner: same-year +0.23%/wk, TRANSFER -0.03%/wk -> memorisation (context fingerprints);
- basis-search learner: same-year +0.09%/wk, TRANSFER -0.002%/wk (CI -0.03..+0.03) -> nothing generalises;
- lessons (B06): -0.19%/wk from other windows; indicator-IC memory: negative walk-forward skill; analogs/dial/exits: no edge;
- the ONE real edge: movement prediction (miner/model IC 0.35-0.38 on |weekly move|); direction on movers ~coin flip.
- 23% of weeks land in the 5-10% band; the tiered objective (engine/objective.py, C38/C39) wants >= 50%.

Task: design and test learners whose improvement TRANSFERS to unseen years (past-only, C56), judged ONLY by the harness's
transfer delta with a 95% CI above zero - same-year gains do not count. Candidates to try (and your own):
(a) a cross-year band-targeting policy: learn from past windows how predicted portfolio movement (from the movement model /
    vol20 / p_move) maps to realised weekly |move|, and choose k / pool / exposure per week to land in the 5-10% band;
(b) regime -> setting mapping learned across many past years with heavy shrinkage (not week-to-week knob chasing);
(c) learning from mistakes aggregated ACROSS years (only lessons that held in >= N distinct past years, tested out of sample);
(d) a learner that improves the movement model's use (e.g. which movers to hold) using past years' outcomes.
Rules: every learner is point-in-time (only windows whose real end precedes the target's real start); run each through the
harness (--pairs >= 12, same memoriser/noise/transfer controls); report per learner: transfer delta per metric with CI,
same-year delta, memorisation gap; adopt NOTHING yourself - report which learners pass. Long real runs DETACHED (PowerShell
Start-Process) with checkpoints; check RAM >= 2.5 GB first. Tests: planted worlds where a generalisable rule exists (the
learner must find it and transfer) and where it does not (the learner must show ~0). Never run git; never kill by name.
