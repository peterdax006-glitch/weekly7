# F27_candidate_representation_recall  [model: opus]
C75 Phase 3A/3B ("search broadly, identify potential patterns, normalise, deduplicate"), 3I (interactions/XOR), canon C70 ("identify ALL
the potential patterns"). Read CONTEXT.md (rules 1-29), F19's report (Masterstock JOURNAL "F19"), state/research/pattern_benchmark/
final_look/SUMMARY.md, engine/research/pattern_benchmark.py (how worlds plant each real kind; read-only for you),
engine/research/discovery.py, discovery_sources.py, targets.py (F24 just added a charged median-split pair screen to
find_conjunctions), interactions.py, volatility_lab.py / vol_hypotheses.py (the screen the benchmark ran).
Benchmark failures you own (F19 ranks 3 and 4):
 3. Representation: 240/900 real patterns can only be found in their TRUE FORM - XOR, interactive (A+B), delayed (lagged), rare
    (sparse events), threshold (non-linear cut) - and the single-feature screen cannot express them.
 4. Screen recall: only 33% of real patterns are ever raised as candidates; faint and subtle ones never.
You OWN: candidate generation - discovery.py, discovery_sources.py, targets.py, interactions.py, and the screen entry the
benchmark calls (name it) - and their tests. evidence.py / quality_gate.py / pattern_benchmark.py belong to F26 (running): if the
benchmark needs a new hook to call your generator, specify the exact call and ask the main session. Never touch engine/patterns.py
(F25), feeds.py/two_stage.py/decision_bridge.py/adaptive.py (F23). Never run git. <= 1 process.
Do:
1. Extend candidate generation so each true form is EXPRESSIBLE: pairwise interactions (AND, XOR via the F24 pair screen), lags
   (a bounded, pre-declared lag set), thresholds (pre-declared quantile cuts), rare-event indicators, and conditional (feature x
   regime/context) forms - with every generated candidate COUNTED in the scanned total so the gate's multiplicity is honest (F26 is
   switching the gate to scanned counts). Normalise and deduplicate equivalent hypotheses (the same pattern reached two ways is one
   candidate).
2. Raise recall without flooding: measure on DEVELOPMENT benchmark seeds only (held-out seeds are forbidden to you) the candidate
   recall by real kind and strength, and the number of candidates per world; report the trade-off curve and choose the operating
   point with the gate's multiplicity cost stated. Never read answer keys to choose candidates - the generator must be blind.
3. Tests: each true form planted alone is raised; a null world's candidate count and false-candidate rate are reported; dedup works.
Report recall before/after by kind and strength, candidates per world, scanned count, runtime, tests, ruler counts.
