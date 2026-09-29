# WEEKLY7: MASTER BLUEPRINT

*Version 4.0, 28 September 2026. It is meant to be complete (canon C44): every input, every mechanism, every setting, what is built in, what the system learns by itself, how much risk it takes, and every open question, including designs not yet fully proven. Items not yet built, or whose exact behaviour isn't certain, carry a **confidence tag**: [BUILT], [BUILT, BEING TESTED], [DESIGNED], [PROPOSED] or [UNCERTAIN].*

*Authority: the owner's directives C1–C44 are stored word for word, and hash-locked, in `weekly7/canon/CANON.md`. They outrank this blueprint; this blueprint outranks the code.*

---

# PART I: WHAT WEEKLY7 IS

## 1. Mission

Weekly7 picks US stocks for a simulated $1,000 account and aims for **the whole portfolio to average 7% a week over a year** (C1, C2, C42). A $1,000 paper account trades live on the real market during regular hours (Live). Behind it, a research machine learns to do better:
- **Test:** blind, self-learning replays of random historical years.
- **Find volatility:** finding the stocks that will move a lot, then which way.
- **Algorithm:** a self-learning pattern system with a massive, relevance-weighted memory.

## 2. The target, in numbers

| Quantity | Value | Comment |
|---|---|---|
| Weekly goal | +7% average over a year | Portfolio level, not per stock |
| Compounded equivalent | about 34× a year | No published strategy sustains this; stated for honesty |
| Tier-1 band ("about 7%") | weekly \|move\| 5%–10% | Below is too low, above too risky (C38) |
| Mover definition | touches ±10% within 5 sessions from the next open | C23, C33 |
| Mover accuracy goal | 95% of weekly picks | C23 |
| Direction bet threshold | ≥80% calibrated confidence | C24 |
| End goal | 8 of 10 picks finish +10% | C23, C24 |
| Loss limit per losing pick | never worse than −20%; aim −10% to −15% | C24 |
| Starting capital | $1,000 (simulated) | C1 |

## 3. Success hierarchy (the lexicographic objective)

1. **Tier 1, volatility:** most weeks (≥50%) move about 7% (5–10%), and the yearly average move is near 7%.
2. **Tier 2, risk:** minimise risk (worst-5% week, maximum drawdown, weeks beyond the band) without breaking tier 1.
3. **Tier 3, direction:** raise the share of in-band weeks that are positive, toward 80%.

A change that improves a higher tier always wins over one that only improves a lower tier (C31, C38, C39).

## 4. The four projects and their priority (C28, C35, C36)

| Priority | Project | Role | Attention |
|---|---|---|---|
| 1 | Algorithm | The learning brain: patterns, analogs, lessons, memory | Highest |
| 2 | Find volatility | Weekly movers, then direction, exits and stops | High |
| 3 | Test | The blind self-learning simulator; minimal direct code changes | Minimal |
| 4 | Live | The $1,000 paper account on the real market | Low (fix only if broken) |

**Standing workflow rules:**
- start a new test *first*, then build while it runs;
- after every result, review every project it applies to;
- run many experiments at once and keep memory full;
- close finished work before starting new work;
- record every new directive verbatim in the canon and in memory.

---

# PART II: THE DIRECTIVES (C1–C44), SUMMARISED

The full verbatim text lives in `canon/CANON.md`.

| # | Directive in brief | # | Directive in brief |
|---|---|---|---|
| C1 | Stock-picking site; free tools; aim for 7%/week; $1,000 sim; research first; don't just follow trends | C23 | 10 weekly ±10% movers at 95%; then direction, exit, stop; +10% 8× as often as −10% |
| C2 | 7% is portfolio-level | C24 | Precision over aggression; ≥80% confidence gate; 8/10 positive; losers above −20% |
| C3 | The blueprint must be very advanced | C25 | Run many experiments; use all the memory |
| C4 | Permission to set up GitHub and Alpaca | C26 | Free memory between projects |
| C5 | Self-improvement until perfect | C27 | Per-stock-type learned indicators, after picking the right stocks |
| C6 | Regular trading hours only | C28 | Three projects; little attention on Live |
| C7 | Hundreds of historical tests | C29 | Continuous heavy test / pattern / code cycle |
| C8 | No cheating; keep the 7% focus | C30 | Study picked winners, picked losers and missed winners |
| C9 | Replay a random past year as if live | C31 | 7% first, accuracy second |
| C10 | Random years until 7%/week | C32 | A checkoff list; any free tool; all RAM; don't stop |
| C11 | A blind hidden year, fast live clock, examine, adjust, rerun; no crypto | C33 | No cheating; never after-hours or weekend trading |
| C12 | Make testing fast before bulk testing | C34 | Advanced self-learning with factor-weighted memory |
| C13 | A pattern viewer | C35 | The Algorithm project (thousands of patterns, massive memory); tests while building; verbatim to memory |
| C14 | Which settings matter, how much, how consistently | C36 | Priority: Algorithm > Find volatility > minimal Test |
| C15 | Self-adjusting within a year, mindful of sudden breaks | C37 | Algorithm blueprint; P(real / hallucinated / coincidence); never stops, never learns for no reason |
| C16 | No manual mid-test changes; weekly self-training; train the training basis | C38 | Below 7% too low, above too risky; more data |
| C17 | Pre-season self-training of defaults | C39 | 7% volatility, then risk, then positive share |
| C18 | No way to cheat | C40 | Master blueprint in Downloads, very long |
| C19 | Random start month + 12 months | C41 | Rare analogs; every data point knows its date and relevance |
| C20 | Learn from missed winners | C42 | Yearly 7% average; timeline-aware caution; learn from mistakes by pattern, not memorisation |
| C21 | Below 1%: volatility first | C43 | Never hold a failed pattern: improve or discard |
| C22 | ±200% years | C44 | This blueprint covers every detail |

---

# PART III: ARCHITECTURE

## 5. Components

| Component | File | Role | Status |
|---|---|---|---|
| Universe | `engine/universe.py` | Which stocks exist; exclusions (ETFs, warrants, units, preferreds, SPACs, crypto) | [BUILT] |
| Price data | `engine/data.py` | Daily OHLCV download, repair, cache; partial-bar protection | [BUILT] |
| SEC filings | `engine/edgar.py` | Events (8-K items, offerings, shelves, 13D, late filings), insider purchases, daily refresh | [BUILT]; 13D fix [BUILT, BEING TESTED] |
| Features | `engine/features.py` | 54 point-in-time signals per stock-day | [BUILT] |
| Candles | `engine/candles.py` | 25 multi-timeframe candle and micro signals | [BUILT] |
| Models | `engine/model.py`, `engine/train.py` | LightGBM expected-return regressor, target classifier, walk-forward training, calibration | [BUILT] |
| Policy | `engine/policy.py` | Scoring, eligibility, top-k with hysteresis, sector cap, regime modes | [BUILT] |
| Portfolio math | `engine/portfolio.py` | Scenario simulator and optimiser (kept as a challenger) | [BUILT], unused by champion |
| Session | `engine/adaptive.py` | The shared daily trading loop (Test and re-tester), Adapter (weekly self-learning), missed-winner detector | [BUILT] |
| Memory | `engine/memory.py` | Factor-weighted episodic memory | [BUILT] |
| Pattern miner | `engine/patterns.py` | Thousands of patterns; P(real, hallucinated, coincidence); lifecycle | [BUILT, BEING TESTED] |
| Analog engine | `engine/analogs.py` | Market fingerprints since 1962; nearest past analogs | [BUILT, BEING TESTED] |
| Blind simulator | `engine/livesim.py` | Sealed windows, disguised feed, lockstep clock, parity test | [BUILT] |
| Live | `engine/live.py`, `broker.py`, `tick.py` | Scheduled jobs, Alpaca broker, hours guard | [BUILT] |
| Dashboard | `engine/site_data.py`, `docs/` | Public pages | [BUILT] |
| Test loop | `scripts/livesim_loop2.py` | Rounds of sealed windows, gates, basis training | [BUILT] |
| Find volatility | `scripts/movers.py`, `grid_runner.py` | Mover finder and parallel experiments | [BUILT] |
| Algorithm tests | `scripts/algo_test.py`, `analog_test.py` | Heavy tests of the miner and analogs | [BUILT] |
| Data feeds | `scripts/fetch_macro.py`, `collect_intraday.py` | FRED macro and minute bars | [BUILT] |

## 6. Data flow

1. **Data layer:** prices, filings, insider trades, macro, candles, intraday.
2. **Point-in-time features.** Each value is stamped with the moment it became public.
3. **Algorithm:** pattern miner, analog engine and lesson memory produce scores.
4. **Find volatility:** mover probability, then direction.
5. **Session (policy + Adapter + Memory):** chooses holdings weekly and adjusts itself.
6. **Test gates** decide whether a change is real.
7. **Live** receives only validated changes, by owner decision.

---

# PART IV: DATA, EVERYTHING THE SYSTEM LOOKS AT

## 7. Sources

| Data | Source | Depth | Frequency | Status |
|---|---|---|---|---|
| Daily OHLCV, 5,225 stocks | Yahoo via yfinance | 1962 → today (listed survivors only) | Daily | [BUILT] |
| Market series: SPY, QQQ, IWM, VIX, VIX3M, 11 sector ETFs | Yahoo | ETF inception → | Daily | [BUILT] |
| S&P 500 index, long VIX, rates | Yahoo (^GSPC, ^VIX, ^IRX, ^TNX) | 1962 → | Daily | [BUILT] |
| SEC events | EDGAR submissions API | 2001 → | Daily refresh via daily index | [BUILT] |
| Insider purchases | SEC DERA Form 3/4/5 data sets (2006 →) plus parsed daily Form 4 | 2006 → | Quarterly bulk + daily | [BUILT] |
| Macro (19 series) | FRED public CSV | Varies, 1854 → | Daily to monthly | [BUILT] |
| Intraday | yfinance 1-minute (7-day window) and 5-minute (60-day window), top 400 by liquidity | Forward from 28 Sep 2026 | Collector runs repeatedly | [BUILT]; scheduling [DESIGNED] |
| Options (implied vol, call/put skew, volume) | yfinance option chains | Today only (no free history) | Live decision day | [BUILT] live-only |
| Paper broker | Alpaca paper API | Live | Real time | [BUILT] |

**FRED series (19):**
- **Rates and the curve:** 10-year yield (1962→), 2-year (1976→), 3-month (1981→), 10y−2y curve, 10y−3m curve, Fed funds.
- **Credit and inflation:** high-yield credit spread, investment-grade spread, 10-year breakeven inflation.
- **Markets:** VIX (long history), WTI oil, trade-weighted dollar, gold.
- **Economy:** unemployment, CPI, industrial production, consumer sentiment, recession indicator.
- **Stress:** Chicago Fed financial conditions, St. Louis Fed financial stress.

## 8. Universe rules [BUILT]

- **Listings:** NYSE, Nasdaq, NYSE American and Arca listings, and SEC-registered filers only.
- **Excluded:** ETFs, test issues, warrants, units, rights, preferreds, depositary shares, notes, trust preferreds, SPACs, limited partnerships, and symbols with suffixes.
- **Crypto excluded everywhere (C11):** crypto trusts, miners and exchanges (22 removed by name and ticker list).
- **Tradability filter (Live and research):** price ≥ $3 and a 20-day median dollar volume ≥ $5M.
- **Simulations across eras:** relative filters, excluding the bottom 20% by price and the bottom 40% by dollar volume that day, because split adjustment and old volumes make fixed dollar thresholds meaningless.
- **Volatility finder:** the widest universe, with no price or volume cut (validated as best).

## 9. Feature dictionary: 54 engine signals [BUILT]

*Every feature for day t uses data through t's close. Cross-sectional features are ranked within the day before modelling.*

| Feature | Definition | Idea / literature |
|---|---|---|
| r1, r5, r20, r60, r120 | Log return over 1, 5, 20, 60, 120 sessions | Momentum and reversal at several horizons |
| mom_12_1 | 12-month return skipping the last month | Classic momentum |
| dist_52wh | Close ÷ 52-week high − 1 | George–Hwang 52-week-high effect |
| dist_ma50, dist_ma200 | Close vs 50- and 200-day averages | Trend position |
| vol20 | 20-day standard deviation of daily returns | Volatility level |
| vol_ratio | vol20 ÷ 120-day volatility | Volatility expanding or contracting |
| atr_pct | 14-day average true range ÷ price | Daily range size |
| max20, min20 | Largest and smallest daily return in 20 days | Lottery-stock effect (Bali et al.); crash exposure |
| skew60 | 60-day return skewness | Lottery preference |
| log_dv | Log 20-day median dollar volume | Liquidity and size proxy |
| vol_surge1, vol_surge5 | Volume ÷ 50-day average, today and 5-day | Attention, unusual activity |
| overnight20, intraday20 | 20-day sum of close→open and open→close returns | Overnight vs intraday momentum (Lou–Polk–Skouras) |
| frog | Continuous vs jumpy momentum (Da–Gurun–Warachka) | Steady trends persist more |
| range_compress | 10-day range ÷ 60-day range | Coiling before breakouts |
| gap_today | Today's open vs yesterday's close | Gap behaviour |
| close_loc | Where the close sits in today's range | Intraday buying or selling pressure |
| ind_mom20, ind_mom60 | Average return of the stock's SIC-2 industry | Industry momentum (Moskowitz–Grinblatt) |
| rel_ind20, rel_ind60 | Stock return minus industry return | Stock-specific strength |
| days_since_earn | Sessions since the last earnings 8-K (item 2.02) | Earnings cycle |
| ear | Market-adjusted return around the last earnings release | Post-earnings drift |
| ear_volsurge | Volume surge on the earnings day | Information intensity |
| days_to_earn | Estimated sessions to the next earnings (63-session cadence) | Upcoming event risk |
| earn_in_week | Earnings expected within 6 sessions | Event week flag |
| ev_offering | Equity offering (424B1/4/5/7) in the last 10 sessions | Dilution (negative) |
| ev_shelf | Shelf registration (S-1/S-3/F-1/F-3) in 30 sessions | Future dilution |
| ev_activist, ev_activist_amend | 13D filed / amended in 20 sessions | Activism (currently disabled pending the attribution fix) |
| ev_agreement | Material agreement (8-K 1.01) in 10 sessions | Corporate event |
| ev_red_flag | Restatement, delisting notice, bankruptcy, late filing or auditor change in 60 sessions | Distress |
| news5 | Any filing event in 5 sessions | News vs no-news moves |
| r5_nonews, r5_news | 5-day return split by whether news existed | Chan (2003): news moves persist, no-news moves revert |
| ins_buyers30 | Opportunistic insider buyers in 21 sessions | Cohen–Malloy–Pomorski |
| ins_value30 | Insider buying ÷ 21 days of trading value | Size of conviction |
| ins_officer30 | Officers among buyers | Officer purchases are more informative |
| ins_opportunistic30 | ≥2 opportunistic buyers (cluster) | Clustered buying |
| m_spy_ma50, m_spy_ma200 | Market vs its 50- and 200-day averages | Market trend (regime) |
| m_spy_r5 | Market's last 5 days | Short-term market move |
| m_vix, m_vix_chg5 | Fear level and its 5-day change | Regime |
| m_vix_term | VIX ÷ VIX3M (above 1 = stress) | Fear term structure |
| m_breadth | Share of stocks above their 50-day average | Participation |
| m_dispersion | Cross-sectional return spread (5-day average) | Stock-picking opportunity |

**Insider "routine vs opportunistic" [BUILT]:** a purchase is routine if the same insider bought **the same company** in the same calendar month in each of the prior 3 years, counting only filings public by that day. Keying per company fixed a leak.

## 10. Candle and micro-signal dictionary: 25 signals [BUILT]

| Feature | Definition |
|---|---|
| cd_body, cw_body, cm_body | Candle body (close − open) ÷ range for the day, week (5 sessions) and month (21 sessions): −1 is fully red, +1 fully green |
| cd_upwick, cw_upwick, cm_upwick | Upper wick ÷ range (selling into highs) |
| cd_lowwick, cw_lowwick, cm_lowwick | Lower wick ÷ range (buying into lows) |
| cd_pos, cw_pos, cm_pos | Close position within the candle range |
| cd_range, cw_range, cm_range | Range ÷ price |
| streak | Consecutive up (+) or down (−) days |
| reversal_1d | Today's direction flipped from yesterday (signed) |
| reversal_vs_week | Today against the week's direction (momentary reversal) |
| inside_day / outside_day | Range inside / engulfing yesterday's |
| engulf | Bullish (+1) or bearish (−1) engulfing candle |
| gap | Open vs prior close |
| gap_filled | The gap closed during the day |
| week_vs_month_pos | Weekly close position minus monthly (timeframe disagreement) |
| day_vs_week_body | Daily body minus weekly body |

**Minute candles:** collected forward only (1-minute for 7 days, 5-minute for 60 days per pass). Intraday patterns are planned once enough history exists:
- opening-range breaks;
- VWAP position;
- lunchtime reversals;
- closing-auction pressure;
- intraday volatility regime;
- minute-level momentary reversals.

[DESIGNED]. How much minute history is needed before these become statistically usable is [UNCERTAIN]; the estimate is 6–12 months.

## 11. Signals used only in Live (no free history) [BUILT]

| Signal | Definition |
|---|---|
| vol_spread | Call implied volatility minus put implied volatility at matched strikes (Cremers–Weinbaum) |
| put_skew | Out-of-the-money put IV minus at-the-money call IV (Xing–Zhang–Zhao) |
| cp_volume | Log call ÷ put volume |
| iv_atm | At-the-money IV (a volatility forecast) |

These enter Live as a small tilt. The learning loop measures their live accuracy and adjusts their weight.

## 12. Macro fingerprint inputs (analog engine) [BUILT]

Market drawdown from its 1-year peak; returns over 1, 3, 6 and 12 months; acceleration (6-month minus half the 12-month); distance from the 200-day average; 1-month volatility and its ratio to 1-year; VIX and its term structure; breadth; dispersion; **sector crowding** (Herfindahl concentration of trading value by SIC-2 sector); change in the top sector's share over 3 months; all 19 FRED series; the 1-year change in the 10-year yield; the 3-month change in high-yield spreads.

**Publication lags (point-in-time):**

| Series | Lag (sessions) |
|---|---|
| CPI | 35 |
| Unemployment | 25 |
| Industrial production | 35 |
| Consumer sentiment | 20 |
| Recession flag | 120 |
| Financial condition and stress indices | 7 |
| Daily series | 1 |

## 13. Point-in-time and date rules

- **Features:** a feature for day t uses only data through t's close.
- **Filings:**
  - a filing accepted at or after 15:30 ET counts from the next session;
  - insider filings, whose filing time isn't recorded, count from the next day.
- **Training labels:** training rows must have labels that close at least 5 sessions (volatility finder) or 10 sessions (models) before the prediction date. This is asserted in code.
- **Unfinished bars:** a partial intraday bar never enters daily data, except the deliberate 15:40 snapshot for Live's decision.
- **Every data point keeps its real date** (C41). Blind tests shift all dates equally, so relative ages survive and the absolute year is hidden.

## 14. Known data limits

1. **Survivorship:** only companies listed today, so old eras are thin (one 1960s window had 29 stocks) and flattering.
2. **No free filings before 2001, and no insider data before 2006.**
3. **No free historical minute bars or options history.**
4. **Split-adjusted prices** make old price levels tiny, which is why simulations use relative filters.
5. **Yahoo data** occasionally fails silently. It's repaired by re-fetching empty tickers, but is still a risk.

---

# PART V: HOW DECISIONS ARE MADE

## 15. Scoring

| Score | Formula | Status |
|---|---|---|
| Evidence score | Weighted sum of ranked evidence features (weights: ear 1.0, ins_buyers30 0.8, ins_officer30 0.4, dist_52wh 0.7, ind_mom60 0.6, frog 0.4, mom_12_1 0.3, r5_nonews −0.5, max20 −0.8, ev_offering −1.0, ev_shelf −0.3, ev_red_flag −1.0, skew60 −0.2; ev_activist 0.8 currently off), then ranked | [BUILT] |
| Model score | LightGBM Huber regressor of 5-session excess return (400 trees, 31 leaves, minimum 400 per leaf, learning rate 0.03, 70% row and 60% column sampling, L2 of 5; monotone constraints on max20, ev_offering, ev_red_flag (negative) and ins_buyers30, ear (positive)) | [BUILT] |
| Blend | w_model × model rank + (1 − w_model) × evidence rank. The model alone won blind tests (w_model = 1: +0.32%/week, t = 3.8) | [BUILT] |
| Mover probability (p_move) | LightGBM classifier of touching ±10% in 5 sessions from the next open | [BUILT] |
| Momentum path | Rank of 52-week-high proximity plus rank of last week's return | [BUILT] |
| Pattern score | Sum of the shrunk effects of active patterns matching a stock | [BUILT, BEING TESTED] |
| Analog signal | Similarity-weighted outcomes of the nearest past episodes | [BUILT, BEING TESTED] |
| Missed-winner detector | Online logistic model of "next +7% stock"; earns weight only by predicting later weeks | [BUILT] |

## 16. Eligibility (before ranking)

- **Excluded outright:**
  - red-flag filings in the last 60 sessions;
  - equity offerings in the last 10 sessions for less-liquid names (below the median by dollar volume);
  - crypto.
- **Optional volatility filter:** exclude the top 10% by volatility and by largest daily spike (evidence: those underperform). It's a setting the system can turn off.
- **Liquidity percentile floor:** the liq_q setting.

## 17. Portfolio construction

- **Top-k equal weight,** k from 1 to 16:
  - Live champion: k = 4;
  - Test aggressive basis: k = 2;
  - the system tunes k.
- **Hysteresis:** keep a holding while it stays in the top (1 − exit_q) of eligible names (exit_q from 0.5 to 0.95).
- **Pick mode:**
  - "top" takes the highest scores;
  - "hivol" takes the most volatile names within the top (1 − pool_q) of scores, giving more swing while still carrying an edge.
- **Sector cap:** at most N names per SIC division (Live: 2).
- **Regime modes:**
  - **stress-rebound:** when short-term fear exceeds long-term fear by more than a threshold, switch to a concentrated high-volatility set;
  - **downtrend cash:** when the market is below its 200-day average by more than a threshold, invest only a fraction.
- **Cash buffer:** 1.5% (orders are sized to 98.5% of equity so limit-order reserves never fail).

## 18. Trading mechanics

- **Rebalancing:** on the last session of each week (holiday-aware), or every N weeks. Mid-week, only risk rules act.
- **Fills:**
  - **Live:** marketable limit orders (±0.3% of price), DAY orders, fractional shares; market orders as fallback.
  - **Simulation:** decisions after a close fill at the **next session's open** (C33).
- **Costs:**

  | Venue / era | Cost per side |
  |---|---|
  | Live, liquid (dollar volume above $50M) | 5 bp (research) |
  | Live, illiquid | 30 bp |
  | Simulation, before 1997 (eighth-dollar ticks) | 40 bp |
  | Simulation, 1997–2000 (sixteenths) | 20 bp |
  | Simulation, 2001 onward (decimal) | 10 bp |

- **Trading hours:** Live orders only 09:30–16:00 ET on weekdays; holidays and weekends are skipped by the scheduler; the broker refuses anything else.
- **Pattern-day-trader rule (accounts under $25k):** same-day exits are limited to the remaining day-trade budget. If a stop fires on a position bought today with no budget left, the exit waits until tomorrow.
- **Splits and dividends:** prices are adjusted, so no special handling is needed in research. Live uses the broker's positions.
- **Delisting mid-hold [UNCERTAIN]:** in simulation a delisted stock's last price stays in the valuation. Survivorship means few delistings exist, so this path is barely exercised.

## 19. Risk policy

| Rule | Value | Status | Evidence |
|---|---|---|---|
| Weekly brake | If the week reaches −8%, cut to 1/3 exposure until the week ends (Test explores none / 5% / 8% / 12% / 15% / 20%) | [BUILT] | Helped drawdown in 2017–26; a brake tighter than 5% hurt (sensitivity study) |
| Per-stock stops (ATR-based) | Off | [BUILT, disabled] | Cost $2,695 over 222 stop-outs in backtests |
| Bank the +7% | Off | [BUILT, disabled] | Cost about $850 in backtests |
| Max weight per name | 25% (Live k = 4); tests allow 50–100% (k = 1–2) | [BUILT] | C22 volatility target |
| Sector cap | 2 of 4 names (Live) | [BUILT] | +$3,400 vs no cap in backtest |
| Tier-1 band | Weekly move 5%–10% | [BUILT] | C38 |
| Catastrophe floor | A candidate whose window loses more than 99% is rejected | [BUILT] | C22 |
| Direction confidence gate | Bet only at ≥80% calibrated P(correct direction) | [DESIGNED] | C24 |
| Loser limit | Stop placement so losers never exceed −20%; target −10% to −15% | [DESIGNED] | C24; exact method [UNCERTAIN]. Candidates: volatility-scaled stops chosen per stock type, intraday stop checks, and position sizing by distance to stop |
| Yearly pacing | Aggression up when behind the 7% average, down when ahead, within tier-2 bounds | [DESIGNED] | C42 |
| Unforeseeable events | Skip names with scheduled binary events (FDA dates, trials, court rulings) unless confidence clears the gate | [PROPOSED] | C24. A free source for FDA and trial calendars is [UNCERTAIN] |

---

# PART VI: TEST, THE BLIND SELF-LEARNING SIMULATOR

## 20. Sealed windows

- **The draw:** a random start month (January 1965 – September 2025), then 12 consecutive months (C19).
  - Sealed one at a time before workers start, so parallel draws can't collide.
  - Windows overlapping earlier ones by more than half are avoided.
- **The disguise:**
  - dates are shifted by a secret whole number of weeks (8,000–11,000, landing in the 2100s–2150s), which keeps weekdays and holidays intact;
  - tickers become code names (S0000–S5000), fixed per window by a seeded shuffle.
- **Warm-up:** 6 years before the window, or as much as exists.
- **Reveal:** the real period is revealed only after the round's adjustments are locked in.

## 21. Inside a window

1. **Precompute** the features (the feature service).
2. **Parity test:** recompute the features the slow, strictly-live way on random days; any difference aborts the run.
3. **Train** the models on warm-up rows whose labels closed before the window: the regressor and the p_move finder.
4. **Pre-season study (C17):**
   - a second model stops 18 months earlier;
   - candidate settings (one-step neighbours of the prior) are replayed on those unseen months, split into quarters;
   - a candidate must beat the prior in at least 75% of quarters to replace it.
5. **The clock** releases one session at a time. At each session:
   - fill yesterday's decision at today's open;
   - value the book at today's close;
   - at each week's end, take a snapshot of what the system sees, let the Adapter learn from the week that closed, and decide for tomorrow's open;
   - check the brake at the close.
6. **Blind diagnosis** at the end:
   - performance by holding;
   - costs and turnover;
   - held vs top-20 vs top-20-high-volatility;
   - missed winners.

## 22. Weekly self-adjustment: the Adapter (C15, C16, C34)

- **What it learns each closed week:**
  - one-step counterfactuals for each adaptive knob (w_model, liq_q, k, pool_q, w_move, w_mom), measured against the base it came from;
  - the predictive power of each indicator;
  - missed-winner lessons.
- **Evidence** goes into the factor-weighted Memory (section 29).
- **Switching rule:**
  - at least `min_weeks` of effective evidence;
  - a `cooldown` between switches;
  - the neighbour's shrunk lead must exceed `switch_z` standard errors;
  - one step at a time.
- **Indicator weights:** each weight moves by at most ±100% of its prior, following its measured predictive power. Weights never flip sign and can fade to zero.
- **Fast revert:** if the current deviation from the defaults lost more than `revert_drop` over 2 weeks, return to the defaults.
- **Missed-winner detector:**
  - an online logistic model over 15 inputs (evidence ranks, mu_raw, vol20, max20, log_dv, r5);
  - trained each week on which stocks rose 7% or more (or the top 5% in quiet weeks);
  - judged out-of-sample before each update;
  - gains weight only when its skill is significant (weight 0 before `det_min_weeks`, capped at `det_max`).

## 23. Between rounds: training the training basis (C16)

- **Candidates:** 24 random combinations of the starting settings (CFG_SPACE) and the adaptation and memory settings (META_SPACE).
- **Screening:** each is screened on 10 random archived windows; the top 3 are confirmed on every archived window.
- **Adoption:** the winner by the tiered objective becomes the next basis.
- **Volatility phase (C21):** while the average stays below 1% a week, the aggressive settings dominate the search.

## 24. Gates (fail-closed)

| Gate | Check |
|---|---|
| Parity | Fast features equal the live-computed ones on random days (max difference ≤ 1e-4) |
| Re-tester | Replaying the archived window reproduces the live-clock run (within 0.5%) |
| Future-scramble | Noise injected after a cut changes no decision and no self-adjustment before the cut |
| Time fence | The adaptive code raises an error if handed a price after "today" |
| Worker health | A crashed worker's window is excluded and reported; no stale results are reused |

## 25. Test search spaces (current)

**Starting settings (CFG_SPACE):**

| Setting | Values searched |
|---|---|
| k | 1, 2, 3, 4 |
| exit_q | 0.5, 0.7, 0.8, 0.9 |
| rebalance_weeks | 1, 2 |
| brake | none or 15% |
| max_per_sector | none or 2 |
| w_model | 0.85 or 1.0 |
| pick | hivol or top |
| pool_q | 0.3 – 0.95 |
| liq_q | 0 – 0.2 |
| vol_filter | on / off |
| stress_thr | none, 1.0, 1.05 |
| stress_k | 2 – 4 |
| trend_filter | none or −5% |
| trend_gross | 0 or 0.5 |
| w_move | 0 – 0.7 |
| w_mom | 0 – 0.4 |

**Adaptation and memory settings (META_SPACE):**

| Setting | Values searched |
|---|---|
| half_life | 3, 6, 12 weeks |
| prior_weeks | 4, 8, 16 |
| switch_z | 1.5, 2, 3 |
| min_weeks | 3, 6 |
| cooldown | 2, 4 |
| revert_drop | 2%, 4%, 8% |
| ic_beta | 0 – 2 |
| det_max | 0 – 0.5 |
| det_min_weeks | 4, 8 |
| mem_half_life | 4 – 32 weeks |
| mem_bandwidth | 0.75 – 3 |
| mem_prior_scale | 0 – 0.6 |
| mem_shrink | 2 – 12 |
| mem_shock_k | 1.5 – 4 |
| mem_shock_cut | 0.1 – 0.5 |

---

# PART VII: FIND VOLATILITY

## 26. Step 1: movers [BUILT]

- **Label:** does the stock touch +10% or −10% within the 5 sessions after the next open (high/low vs entry)?
- **Model:**
  - LightGBM classifier (300 trees, 31 leaves, minimum 200 per leaf, learning rate 0.05, 80% row and 70% column sampling);
  - trained on cross-sectional ranks (market readings kept as levels);
  - warm-up week-ends whose label windows closed before the hidden window.
- **Picks:** the 10 highest probabilities each week, or fewer when fewer clear the confidence bar.
- **Results so far (13 blind windows):**
  - 84.7% average accuracy;
  - 95% or better in 5–6 windows;
  - at the 92% bar: about 97% on 5–6 picks a week.
- **Tried, no gain:** a bigger model, stock-type inputs, training on every day (slightly worse).
- **Tried, gain:** the widest universe.
- **In progress:** pattern-miner movement scores, and in-window refitting.

## 27. Steps 2–5 [DESIGNED]

1. **Direction:**
   - a calibrated P(up | it moves) per stock, combining pattern scores for direction, analogs, per-type indicator trust tables (C27) and the missed-winner detector;
   - bet only at ≥80% confidence.
2. **Exit:** learn per stock type whether to exit at +10%, at the week's end, or on a trailing rule. Chosen by out-of-sample results. [UNCERTAIN] which exit family wins.
3. **Stop:** per-type stop distance, chosen so losers average above −15% and never exceed −20%, net of gaps. Overnight gaps can exceed any stop, so limiting losses may also require position sizing and avoiding gap-prone names. [UNCERTAIN] whether −20% can be guaranteed.
4. **Consistency:** 8 of 10 positive at +10%, in every era, via the same blind gates.

**Per-stock-type trust tables (C27) [DESIGNED]:**
- **Types:** sector division × size tercile × volatility tercile × trend state, plus the "cultural trend" of the stock's theme (industry momentum, attention surge).
- **For each type:** each indicator's reliability (week-clustered predictive power, shrunk by sample size).
- **Use:** indicators are weighted per type, and a type with no reliable indicator is skipped.

---

# PART VIII: ALGORITHM, THE SELF-LEARNING SYSTEM

## 28. Pattern miner (C35, C37, C43) [BUILT, BEING TESTED]

- **Inputs:** all 54 engine signals, 25 candle signals and the market context. Each is converted to within-week quintiles.
- **Candidates per run:**
  - singles: 79 features × 5 quintiles;
  - pairs: the top 60 singles paired with each other, plus random pairs, about 4,000 in total;
  - "A & B unless C" exceptions: up to 600;
  - patterns from the long-term bank (always re-tested).
  - About 4,300–4,400 candidates in total.
- **Targets:** direction (excess return) and movement size (\|return\|, relative to the week).
- **Relevance weights per row:** recency (4-year half-life by default) × market-context similarity (bandwidth 1.5 standard deviations) to the moment of use.
- **Statistics:** see section 30.
- **Selection:**
  - P(real) ≥ 80%;
  - not a duplicate (row overlap below 80% with a stronger pattern);
  - passes the validation-gain gate (adding it must raise out-of-sample correlation by more than 0.0005).
- **Scoring:** the sum of the shrunk effects of active patterns matching a stock that day. Improved (rescoped) patterns count only inside their scope.
- **Findings so far:**
  - **Movement patterns are strong:** t up to 15 against a chance level of about 2.5.
  - **Direction patterns are weak** under week-clustered statistics: 0–2 survive per era.
  - Stock-level statistics had overstated them: 1,934 "passes" fell to about 0.

## 29. Memory (C34, C35, C41) [BUILT]

| Factor | Formula | Default | Tuned by |
|---|---|---|---|
| Recency | 0.5^(age ÷ half-life) | 8 weeks (Adapter); 4 years (miner) | Outer loop |
| Market similarity | exp(−distance² ÷ (2 × bandwidth²)) over standardised context | 1.5 | Outer loop |
| Reliability | Shrink the mean by a pseudo-count | 6 | Outer loop |
| Shock | Per-lesson two-sided CUSUM. On a break, evidence before it is multiplied by the shock cut | k = 2.5, cut 0.25 | Outer loop |
| Long-term prior | Lessons from windows that ended before this one began, × prior scale | 0.3 | Outer loop |
| Modernity [DESIGNED] | An era weight for microstructure-sensitive patterns (pre- vs post-decimal, pre- vs post-electronic, pre- vs post-ETF) | Learned | Algorithm |

Nothing is deleted. Relevance decides use.

## 30. Is it real? Three probabilities (C37) [BUILT]

- **P(coincidence):**
  - a week-clustered t-statistic (one observation per week, weighted by relevance);
  - a p-value;
  - a Benjamini–Hochberg correction across every candidate tried (q = 0.05).
- **P(hallucinated):**
  - a permutation null: outcomes are shuffled within each week (2 repetitions over up to 1,500 patterns each);
  - local false-discovery rate = (share of null patterns at least this strong) ÷ (share of real patterns at least this strong).
- **P(real):** (1 − the larger of P(hallucinated) and the corrected P(coincidence)) × Φ(t of confirmation on the later 30% of dates), provided the confirmation keeps the same sign.
- **Effect size:** the week-clustered mean × n_eff ÷ (n_eff + 400).

## 31. Pattern lifecycle (C37, C43) [BUILT]

```
candidate → (rejected | duplicate | no_gain) → recorded, never used
candidate → active → (recent stretch contradicts it) → failed
failed → cause search → improved ("rescoped", consistent through ALL history to date) | discarded
```

**Cause search:** within each context tercile (fear, fear term structure, trend, breadth, dispersion), a failed pattern is re-admitted only if the scoped version holds:
- in the long run (t ≥ 2);
- in the discovery half;
- in the confirmation half;
- in the recent stretch.

Nothing stays benched.

## 32. Rare-analog engine (C41) [BUILT, BEING TESTED]

- **Fingerprints:** one per trading day since 1962 (section 12).
- **Search:**
  - standardise using past data only;
  - distance is feature-weighted (weights learned [DESIGNED]);
  - candidates must end at least 63 sessions before today;
  - one analog per episode (analogs at least 21 sessions apart);
  - top k = 5.
- **Output:**
  - analog dates, distances and ages;
  - a similarity-weighted forecast of next-month return, volatility and drawdown;
  - uniqueness (distance to the nearest);
  - how many close analogs exist.
- **Levels:** market [BUILT]; sector and stock [DESIGNED].
- **Use:** feeds the timeline dial (section 33) and, at stock level, direction. A single analog is allowed, with confidence scaled accordingly.

## 33. Timeline dial and yearly pacing (C42) [DESIGNED]

- **Inputs:** the regime readings, the analog engine's forecast, and year-to-date pacing against the 7% average. No calendar input.
- **Outputs:**
  - exposure (0–100%);
  - k (concentration);
  - pool_q (how far into the volatile tail);
  - brake level.
- **Learning:** the mapping is trained on earlier windows and tested on later ones by the tiered objective. [UNCERTAIN] whether the dial beats a fixed setting. It must prove that in blind windows before it's used.

## 34. Lesson memory: learning from mistakes without memorising (C42) [DESIGNED]

- **Post-mortem per window:** losing decisions and missed winners become lessons (situation fingerprint, features and outcome). **No tickers, no dates.**
- **Use:** a situation resembling a lesson's situation adjusts the decision through memory weights.
- **The rerun experiment:**
  1. play window A;
  2. rerun A under a fresh disguise (new shift, new code names) and measure the improvement;
  3. play unseen window B.
  Lessons are kept only if B isn't hurt.
- [UNCERTAIN] how large the rerun improvement can be without memorisation. Some improvement on A is expected, and it's only real if B holds.

## 35. What is built in vs what the system learns on its own

| Built in (fixed by design) | Learned by the system |
|---|---|
| Anti-cheat rules (point-in-time, next-open fills, disguise, gates) | Which stocks to hold (models, patterns, analogs) |
| The tiered objective (7% band, risk, direction) | Starting settings each window (pre-season study) |
| Regular-hours-only trading | Weekly setting changes (Adapter) |
| Crypto exclusion | Indicator weights (live IC with memory) |
| Costs by era | Which patterns exist, their size, when they apply, when they die |
| The catastrophe floor | How fast memory fades, how much similarity matters, how much to trust earlier windows |
| Candidate families (singles, pairs, unless, sequences) | Which analogs matter; the timeline dial [DESIGNED] |
| Statistical thresholds (P(real) ≥ 0.8, FDR 0.05) [tunable later] | The missed-winner detector |
| Data sources | Per-type indicator trust [DESIGNED] |

## 36. Never stops, never learns for no reason (C37)

- **Never stops:**
  - each closed week updates evidence;
  - the miner re-runs per window and on drift;
  - the bank is re-tested in every window;
  - the outer loop runs rounds continuously.
- **Never for no reason:**
  - a pattern, setting or lesson is used only if it improves out-of-sample results beyond noise (the validation-gain gate, the pre-season quarter test, the confirm-on-every-window rule and the lesson generalisation test);
  - everything else is recorded and unused.

---

# PART IX: OPERATIONS

## 37. Scheduling, compute and resources

- **Live:** GitHub Actions cron (weekdays at :07 and :37, 12:00–21:37 UTC; Saturday 15:17 UTC). The dispatcher decides by New York time what is due:

  | Time (ET) | Job |
  |---|---|
  | 09:45–16:00 | Risk checks |
  | 15:05–16:00 | Decide |
  | from 16:05 | Close |
  | Saturday | Weekly learning |

- **Research:** a local Windows PC (8 cores, 16 GB). A memory-aware queue runs 3–7 jobs at once, down to a small free-memory floor (C25). Idle processes are closed between projects (C26).
- **Speed:** blind windows run at 4–100 ms per trading day, a full round takes minutes, and the feature service is about 65× faster than recomputing.
- **Determinism:** seeded models, fixed code names, stable orderings. Every result can be replayed.

## 38. Security and secrets

- Alpaca keys and the SEC contact email live only in GitHub secrets. The repository is public and contains no secrets.
- Sealed windows (the answers) live only in the git-ignored `state/livesim`.

## 39. Failure handling

- **Yahoo gaps:** repair pass; a missing ticker is excluded that day.
- **SEC slowdowns:** threaded requests at 8 per second and a skip flag for cached filings.
- **Workflow collisions:** the newest run wins on generated files.
- **Memory pressure:** the job queue waits. Jobs killed by the operating system's memory reaper are reported and restarted only by decision.
- **Worker crashes:** excluded, reported, never silently replaced.

## 40. Reporting

- **Public pages:** the dashboard (equity, weekly bars against 7%, baselines, holdings with reasons, ledger, learning loop, backtest), the Pattern Explorer and the Sensitivity page.
- **Records:** the experiment registry (`state/experiments.jsonl`), the checklist (`state/CHECKLIST.md`), the canon, and this blueprint.

---

# PART X: EVIDENCE AND HONEST LIMITS

## 41. Evidence log (selected)

| Date | Finding |
|---|---|
| 28 Sep | The model alone beats the evidence blend (+0.32%/week, t = 3.8) |
| 28 Sep | Stops and "bank +7%" destroyed value; the sector cap and −8% brake helped |
| 28 Sep | Trend-chasing makes +7% weeks but loses money ($1k → $142) |
| 28 Sep | Blind self-learning windows average about +0.2% to +1.0% a week |
| 28 Sep | Movers: 84.7% accuracy; ~97% at the 92% bar on ~5–6 picks a week |
| 28 Sep | Leaks fixed: insider routine keying; same-close fills |
| 28 Sep | Memory made self-adjustment fire (0 → switches and reverts) |
| 28 Sep | Pattern miner: movement strong (t ≈ 15 vs 2.5); direction weak under honest statistics |
| 28 Sep | FRED: 19 series from 1854; intraday: 1.09M 1-minute and 1.87M 5-minute bars collected |

## 42. Open questions and uncertainties (C44: "things you don't know 100% yet")

1. **Is 7% a week reachable at all** with free data and honest simulation? The evidence so far says it's far off (0.2–1%). The system is designed to find out how far precision can go, not to promise the target.
2. **Direction at ≥80% confidence:** the honest statistics say direction patterns are rare. Whether analogs, lessons, per-type trust and missed-winner learning can reach 80% is unproven.
3. **Loser limit:** overnight gaps can exceed any stop. Whether −20% can be *guaranteed* is doubtful; the design aims to make it rare.
4. **Survivorship bias:** it inflates old-era results and thins them. A free source of delisted-company history is [UNCERTAIN] (candidates: SEC filing archives for delisted CIKs combined with price reconstruction; limited).
5. **Minute patterns:** how long until enough minute history exists (estimated 6–12 months) and whether minute effects survive costs.
6. **Pattern count vs overfitting:** thousands of patterns raise false discoveries. The permutation null and gates are the defence; their calibration will be checked by planting fake patterns (a planned test).
7. **The timeline dial and yearly pacing:** these could add risk (chasing the average). They're only allowed if blind tests show tier-2 risk doesn't worsen.
8. **Rerun learning:** separating "learned the lesson" from "memorised the answer" relies on the different-window check. It may prove too strict or too loose.
9. **Market structure change:** decimalisation, electronic trading, ETFs, retail options and 0DTE may make old patterns irrelevant. The modernity weight is meant to learn this; its form is [DESIGNED], not built.
10. **Data vendor risk:** Yahoo's endpoints change without notice, so the system is fragile to that. A backup source (Stooq, Alpaca historical bars) is [PROPOSED].

## 43. The master checklist

**Algorithm (highest priority)**
- [x] A1 Candles and micro-signals
- [x] A2 Pattern miner v1
- [x] A2b Week-clustered statistics, permutation null, P(real), redundancy pruning, validation gate
- [ ] A3 Heavy tests across eras and both targets (in progress)
- [x] A4 Long-term pattern bank
- [ ] A5 Pattern scores into Find volatility (in progress), then direction, then Test
- [x] A6 Minute collector (first pass done; scheduling next)
- [ ] A7 The Algorithm tunes its own settings on held-out eras
- [x] A8 Analog engine (market level) built; heavy test next; sector and stock levels [DESIGNED]
- [ ] A9 Timeline dial and yearly pacing
- [ ] A10 Lesson memory and the rerun experiment
- [x] A11 Improve-or-discard policy
- [ ] A12 More data, always: done FRED and intraday; next sector ETFs, backup price source, delisted history
- [ ] A13 Planted-pattern calibration test for the hallucination estimate

**Find volatility**
- [ ] V1 Movers at 95% where 10 movers exist
- [ ] V2 Direction at ≥80% confidence; per-type trust tables
- [ ] V3 Exit rules
- [ ] V4 Stop rules (losers above −20%)
- [ ] V5 8 of 10 at +10%, every era

**Test (minimal direct changes)**
- [x] T1 Archives rebuilt (move signal, insider data, opening prices)
- [x] T2 Loop running
- [ ] T3 Tiered 7% objective reached
- [~] T4 Self-adjustment fires; aggressiveness tuned by the loop
- [x] T5 Insider leak fixed
- [ ] T6 13D fix (running)
- [ ] T7 Explorer and Sensitivity pages refreshed

**Live**
- [ ] L1 Upgrade decision once a validated edge exists (owner)

## 44. Glossary

- **IC (information coefficient):** the correlation between the ranking and next-period outcomes; above 0 means skill.
- **t-statistic:** standard errors from zero; about 2 or more is unlikely to be chance.
- **Week-clustered:** each week counts as one observation.
- **FDR (false discovery rate):** the expected share of false discoveries.
- **Permutation null:** the search rerun on scrambled outcomes.
- **Walk-forward:** discover early, confirm later.
- **Shrinkage:** pulling noisy estimates toward zero.
- **Hysteresis:** keeping a holding until it falls well down the ranking.
- **Sealed window:** a hidden random 12-month period.
- **Parity test:** proving the fast path equals the live path.
- **Band:** the 5%–10% weekly move range of tier 1.
- **CUSUM:** a cumulative-sum test that detects when a result stream breaks.
- **Herfindahl:** a concentration measure (the sum of squared shares).
