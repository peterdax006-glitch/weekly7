# Algorithm Blueprint: the self-learning pattern system

Governed by canon C34–C39 (verbatim in `canon/CANON.md`). Priority (C36): **Algorithm > Find volatility > minimal Test changes > Live.**
I build the *system that learns*; the system finds the patterns.

---

## 1. The objective the system learns toward (C31, C38, C39)

A lexicographic objective, applied in this order:

| Tier | Goal | How it is measured | Pass condition |
|---|---|---|---|
| **1** | **7% weekly movement** | Size of the portfolio's weekly move \|r\|. Below 7% is *too low*; far above it is *too risky*. | Most weeks (a majority) land in the band around 7% (\|r\| from about 5% to 10%) |
| **2** | **Minimise risk** (only after tier 1 passes) | Worst-5% week, largest drawdown, and weeks beyond the band | As small as possible without failing tier 1 |
| **3** | **Make the 7% weeks positive** (after tiers 1–2) | Share of in-band weeks that are gains | As high as possible; the end goal is 8 of 10 |

A candidate change is compared on tier 1 first. Tier 2 breaks ties among tier-1 passers, and tier 3 breaks ties among those.

## 2. What it looks at: microscopic, multi-timeframe data

- **Candles on every timeframe available:** daily, weekly (5 sessions), monthly (21 sessions), and minute bars going forward from the intraday collector. *Free minute history does not exist for past decades*, so minute patterns build up from today.
- **Small-print signals only a computer tracks at scale:**
  - momentary reversals (today against yesterday, today against the week);
  - inside and outside days, engulfing candles;
  - up/down streak length;
  - gap size and whether the gap filled;
  - where the close sits in the day, week and month ranges;
  - disagreement between timeframes (weekly vs monthly position, daily vs weekly body).
- **Everything else the engine already has:** momentum, volatility, volume surges, liquidity, earnings timing and reactions, insider purchases, filings, industry, and market regime.
- **More data, always (C38):**
  - price history back to 1962;
  - market-wide macro series (FRED: rates, credit spreads, the yield curve);
  - sector ETFs;
  - the forward intraday collector.
  More data is a standing task: whenever the system is waiting, it acquires more.

## 3. How it finds thousands of patterns

- **Candidate generator:**
  - single conditions ("feature in its top or bottom fifth");
  - pairs;
  - **"A and B, unless C"** exceptions, where a third condition cancels the effect;
  - later, sequences ("reversal *after* a 3-day streak") and cross-timeframe combinations.
- **Every candidate is recorded** in the pattern memory with its test results, whether it passed or not. A rejected pattern is information too.
- **Redundancy pruning:** two patterns that fire on nearly the same stocks on the same days (overlap above 80%) are one idea. The stronger one is kept and the other is marked as a duplicate, so "thousands of patterns" means thousands of *different* patterns.

## 4. Is the pattern real, hallucinated or coincidence? (C37)

Every pattern carries three probabilities, calculated and never guessed:

| Estimate | How it is calculated |
|---|---|
| **P(coincidence)** | A **week-clustered** test. Each week is one observation, because stocks in the same week move together; treating them as independent overstates certainty, which was the flaw found in heavy test 1. It's then corrected for the number of patterns tried (Benjamini–Hochberg q-value). |
| **P(hallucinated)** | A **permutation null**: the outcomes are shuffled across weeks and the whole search rerun. The number of "patterns" found in shuffled data shows how many discoveries the search invents by itself. A pattern's local false-discovery rate comes from comparing its strength with that null distribution. |
| **P(real)** | Confirmation on **later, unseen** data (walk-forward) combined with the two estimates above: P(real) ≈ (1 − local false-discovery rate) × a confirmation factor. Only patterns with P(real) above the threshold reach **active** status. |

- **Effect size:** measured in the week-clustered test, then shrunk toward zero by the evidence behind it. A big effect from little data counts for little.
- **Relevance weighting of memory (C34, C35):** nothing is deleted. Every observation is weighted by:
  - recency (the half-life in years is learned);
  - similarity of market conditions to now;
  - era modernity.
  1962 counts rarely, not never.

## 5. Lifecycle: never stops learning, never learns for no reason

```
candidate -> probation -> active -> decaying -> benched -> (rescoped | retired)
```

- **Never learns for no reason.** A pattern moves from probation to active only if adding it **improves out-of-sample prediction** of the objective beyond what noise would give (a validation-gain test with a complexity penalty). Patterns that don't improve the system are recorded but never used.
- **Never stops.**
  - Every new week of data updates each pattern's evidence.
  - A full re-mine runs on a schedule, and whenever results drift.
  - Memory from earlier windows is re-tested in every new one.
- **Death and its cause.**
  - A decaying pattern (recent effect contradicts the long-run effect) is benched immediately.
  - The system then searches for **why**: a condition under which the pattern still holds.
  - It is re-admitted **only in that narrowed form**, once the cause is found and confirmed. With no cause it stays benched, however good its history.
  - When a pattern dies, the system also tries to grow a successor from its neighbourhood.

## 6. How it plugs in

- **Find volatility:** pattern scores for *movement size* feed the ±10% mover finder.
- **Direction (step 2):** pattern scores for the *sign* of the move.
- **Test:** one input only, the pattern score in each weekly snapshot (C36: minimal direct changes). The self-learning Session weights it like any other input.
- **Anti-cheat (C18, C33):**
  - patterns used in a window come only from data that ended before it;
  - the permutation null guards against the search fooling itself;
  - all fills happen at the next regular-hours open.

## 7. Build order (checklist A-items)

1. Week-clustered statistics, the permutation null (P(hallucinated)), P(real), redundancy pruning, probation, and the validation-gain gate.
2. The tiered 7% objective inside the Test loop and the pattern miner's selection.
3. More data: 1962+ history in the miner, FRED macro series, and the intraday collector.
4. Pattern memory bank across windows (earlier windows only).
5. Pattern score into Find volatility, then direction, then Test.
6. The Algorithm tunes its own settings on held-out eras.
