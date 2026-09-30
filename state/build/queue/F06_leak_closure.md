# F06_leak_closure  [model: opus]
C69 ledger W-10 and W-11 (anti-cheating, priority 3); C69 sections 21, 22, 30; canon C55, C56, C58, C64, C66 section 30-31.
Read state/build/CONTEXT.md (rules 1-28), state/build/EXECUTION_LEDGER.md (W-10, W-11, section 5 future-leak concerns),
state/research/leak_audit/report.md (computed verdicts), engine/leak_audit.py, scripts/leak_audit.py, scripts/livesim_loop2.py
(train_basis ~line 162/419, BasisLineage, plan_round), engine/learning/curator.py, engine/research/firewall.py.
You OWN: scripts/livesim_loop2.py (train_basis gating only), engine/leak_audit.py and scripts/leak_audit.py (new checks only),
NEW tests/test_leak_closure.py. W02 owns engine/research/loop.py/feeds.py/evidence.py and P06 owns error_loop.py + the C68 modules -
never edit those. Code + tests; the audit re-run on the real caches is allowed (it reads caches, never state/livesim contents).

Do:
1. Gate train_basis: today it trains on all archived windows and only the PLAY gate (lineage basis_for) is proven. Make the training
   call itself refuse any window that ends at/after the earliest window it could be used for (or record per-version the maximum
   window end and make basis_for refuse versions whose training reached the played window) - fail closed, tested with a planted
   future window.
2. Channel 4 residual: META_DEFAULT came from a sensitivity study on real outcomes. Replace it with a data-free neutral default for
   untrained plays (document the choice) or quarantine every play that uses it; test.
3. Channel 6 with the C64 curator path running: re-audit whether anything that can be LOOKED UP by year (curator releases, research
   records, memory) reaches the trader in a disguised replay; add a computed audit part for it.
4. W-11: an integration test that runs a small disguised replay of the SAME real-shaped (synthetic) year twice with research filed
   under that year in between, and proves (a) no research record filed under the replayed year is released during the replay,
   (b) no hindsight label (episode paths, counterfactual classes, what_changed conclusions, error records) reaches the trader side,
   (c) trader_view refuses planted year/date leaks in released items.
5. Re-run scripts/leak_audit.py (detached, RAM >= 2.5 GB) and report the new computed verdict table; a channel is FIXED only when the
   computed verdict says so.
Report ruler counts, test results, and the before/after verdict table.
