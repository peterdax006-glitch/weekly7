# Weekly7: three projects (canon C28)

Attention: **Test** and **Finding volatility** get the effort; **Live** runs on its own and only gets attention if it breaks.

## 1. Live (low attention)
- $1,000 Alpaca paper account traded by GitHub Actions during trading hours only (C6). Conservative v1.2 rules.
- Last seen: equity $1,000.87 (BNY / TEVA / GS / VRSN). Workflow collision fix applied 28 Sep.
- Action only on failure. An upgrade from the simulation findings waits for the owner's decision.

## 2. Test: self-learning blind simulation (high attention)
- Random sealed 12-month windows 1965–2025 (C19), disguised dates and tickers, lockstep clock (C11, C12).
- Pre-season self-training (C17) plus weekly self-adjustment with guards (C15, C16), learning from missed winners (C20).
- Training basis retrained each round; volatility-first toward ±200% years (C21, C22).
- Gates every round: parity, re-tester match, future-scramble (C18).
- Now: archives rebuilding with stable code names and without insider data; loop2 restarts automatically.
- Sub-items: insider-feature leak root cause (sims run without insider data until fixed); SEC filings refresh and 13D fix; Pattern Explorer and Sensitivity pages refreshed from Test results.

## 3. Finding volatility → later: finding which will be positive (high attention)
- Step 1 (now): each week, 10 stocks expected to move ±10%; target 95% correct (C23). Best so far (widest universe): 84.7% average top-10 accuracy (95%+ in 5 of 13 windows); with a 92% confidence bar, 97.0% accuracy at 6.2 picks a week. Future-scramble gate passes.
- Experiments run in parallel at full memory (C25, C26): universe width, model size, stock-type inputs (C27).
- Step 2 (only after 95%): direction, betting only at ≥80% calibrated confidence and skipping unpredictable events (C24), with per-stock-type indicator trust tables (C27).
- Steps 3–4: exact exit and exact stop (losers never worse than −20%, aim −10%). Goal: 8 of 10 finish +10%, consistently in every era, without cheating.
