# Weekly7: Master Blueprint (v2, advanced)

A website that picks US stocks and runs a $1,000 simulated portfolio, aiming for the **whole portfolio** to grow +7% each week. Monitoring period: one month.

v2, 28 Sep 2026. It supersedes v1. Built on research into free data sources, the academic literature on 1–5 day returns, goal-seeking portfolio theory, and financial machine-learning methodology. Every signal, method and tool is free.

---

## Part A — The problem, stated precisely

### A1. The objective is not "high return". It is "reach a target by a deadline".

"Grow 7% a week" is a **goal-reaching problem**. That is mathematically different from maximising return or Sharpe ratio.

- **Standard investing** (Markowitz, Kelly) maximises the average or log growth rate.
- **Goal-reaching** maximises the probability of finishing above a line by a deadline: here, P(portfolio ≥ 1.07 × Monday's value by Friday's close).

Theory: Browne (1999), *Reaching goals by a deadline*; Dubins & Savage (1965), *How to Gamble If You Must*.
1. When you have a genuine edge, the best policy is to bet *moderately* and let the edge work.
2. When you have no edge, the best policy is "bold play": concentrate and take more risk the further behind you are and the less time remains.
3. Bold play maximises the chance of hitting the goal, but it greatly raises the chance of large losses.

So Weekly7's objective function is:

```
maximise   P( R_week ≥ +7% )
subject to CVaR_95%( R_week ) ≥ −12%     (the average of the worst 5% of weeks is no worse than −12%)
           E[ log(1 + R_week) ] > 0        (the account must still grow over time)
           position, sector and liquidity limits
```

The rest of the engine exists to estimate the **full probability distribution of next week's portfolio return** as accurately as possible, and then to choose the portfolio that puts the most probability above +7% without breaching the tail limit.

### A2. What is realistic
- +7% a week sustained = about 34× a year. No evidence supports that as an *average*.
- A **good** engine can realistically deliver:
  - +7% in a meaningful share of weeks (roughly 20–35%, from high-volatility portfolios with a positive edge);
  - more good weeks than bad weeks;
  - measurable beating of volatility-matched random portfolios.
- A month is 4 weekly outcomes, which is mostly luck. Part H explains how we measure skill properly anyway, using the tens of thousands of predictions the engine makes each week.

---

## Part B — Signal library (what the engine looks at)

The guiding principle: **price moves backed by information continue; moves backed only by attention reverse.**
- Chan (2003): stocks that move on news drift further in the same direction; stocks that move with no news tend to reverse.
- Barber & Odean; Bali, Cakici & Whitelaw (2011): attention-driven and lottery-like spikes underperform.

This is what separates Weekly7 from a trend follower. For every candidate it asks *why* the stock is moving, not just *whether* it is moving.

Evidence strength: ★★★ well replicated, ★★ solid, ★ weak or conditional.

### B1. Earnings information
| Feature | Evidence | Source (free) |
|---|---|---|
| Earnings-day price reaction relative to the market (EAR), days since | ★★★ Brandt et al. 2008; Ball & Brown | Alpaca bars + earnings calendar |
| Standardised earnings surprise (SUE), actual vs estimate | ★★★ Bernard & Thomas 1989 | Finnhub surprises; EDGAR XBRL |
| Revenue surprise alongside EPS surprise (both beating is stronger) | ★★ Jegadeesh & Livnat 2006 | EDGAR XBRL |
| Pre-earnings run-up (earnings-announcement premium) | ★★ Frazzini & Lamont; Barber et al. 2013 | Earnings calendar |
| Earnings-day abnormal volume (confirms that information arrived) | ★★ | Bars |
| Peer spillover: a same-industry company just beat and this one reports soon | ★★ Thomas & Zhang 2008 | SIC codes (EDGAR) |

### B2. Informed trading
| Feature | Evidence | Source |
|---|---|---|
| Opportunistic insider buys: cluster count, $ size, seniority (CEO/CFO weighted), routine traders excluded | ★★★ Cohen, Malloy & Pomorski 2012 | EDGAR Form 4 |
| Schedule 13D activist stake filed | ★★★ Brav, Jiang, Partnoy & Thomas 2008 (~7% abnormal return around the filing) | EDGAR daily index |
| Call vs put implied-volatility spread (put-call parity deviation) | ★★★ Cremers & Weinbaum 2010 (~50 bp/week) | yfinance chains |
| Change in call implied volatility vs put implied volatility | ★★ An, Ang, Bali & Cakici 2014 | yfinance chains |
| Steep OTM put skew (bearish) | ★★ Xing, Zhang & Zhao 2010 | yfinance chains |
| Signed option volume, call-heavy relative to the stock's own norm | ★★ Pan & Poteshman 2006 | yfinance chains |
| Implied vol well above realised vol **without** earnings pending (something is expected) | ★ | yfinance chains |

### B3. Corporate actions (event flags)
| Feature | Direction | Evidence |
|---|---|---|
| Share offering / shelf takedown (424B, S-3 filings) | **strongly negative**, avoid | ★★★ |
| Buyback authorisation (8-K) | + small | ★★ Ikenberry et al. |
| Material agreement or acquisition target (8-K items 1.01 / 2.01) | + / event | ★ |
| Going-concern warning, auditor change, late filing (NT 10-K) | **negative**, exclude | ★★ |
| Dividend initiation | + | ★★ Michaely et al. 1995 |

### B4. Price and volume structure
| Feature | Evidence |
|---|---|
| Nearness to the 52-week high | ★★★ George & Hwang 2004 |
| Industry momentum (sector/industry ETF 1–3 month return) | ★★★ Moskowitz & Grinblatt 1999 |
| High-volume return premium: unusual volume on a quiet-price day | ★★ Gervais, Kaniel & Mingelgrin 2001 |
| 1-week reversal, **only when there was no news** | ★★ Chan 2003; Jegadeesh 1990 |
| Momentum split into overnight vs intraday parts (the overnight part persists) | ★★ Lou, Polk & Skouras 2019 |
| Frog-in-the-pan: momentum built from many small steady moves beats momentum from one jump | ★★ Da, Gurun & Warachka 2014 |
| Volatility-compression breakout: tight range, then volume expansion | ★ practitioner evidence |

### B5. Penalties (negative weight or exclusion)
| Feature | Evidence |
|---|---|
| MAX: largest single-day return in the last 20 days (lottery stock) | ★★★ Bali, Cakici & Whitelaw 2011 |
| Idiosyncratic volatility extreme (top decile) | ★★★ Ang et al. 2006 |
| Retail attention spike with no fundamental news | ★★ Barber & Odean 2008; Da, Engelberg & Gao 2011 |
| Recent IPO (< 6 months) or lock-up expiry within 10 days | ★★ Field & Hanka 2001 |
| Heavy short interest **plus** a falling price (informed shorts are right) | ★★ Boehmer, Jones & Zhang 2008 |

### B6. Short-squeeze module (small, capped allocation)
Setup: short interest > 20% of float, days-to-cover > 5, a positive information catalyst from B1/B2, and price holding above its 20-day average. Evidence ★. The outcomes are binary, so this module can hold at most one position at 10% of the portfolio.

### B7. Calendar and macro
| Feature | Evidence |
|---|---|
| Pre-FOMC drift (the market rises in the 24 h before Fed announcements) | ★★ Lucca & Moench 2015 |
| Turn-of-month effect | ★★ Ariel 1987; McConnell & Xu 2008 |
| Pre-holiday effect | ★★ |
| CPI / jobs-report days, which raise market-wide variance | ★★ (feeds the risk model) |
| Options-expiry week | ★ |

### B8. Regime features (market-wide)
- SPY vs its 50/200-day averages.
- VIX level, 5-day change and term structure (VIX vs VIX3M).
- Breadth (% of the universe above its 50-day average).
- Credit spreads (FRED high-yield option-adjusted spread).
- 2s10s yield curve (FRED).
- Dispersion (cross-sectional return standard deviation), because stock-picking pays more when dispersion is high.

**About 90 features in total.** Every feature is timestamped with the moment it became public (point-in-time), so the model can never "see" the future.

---

## Part C — Learning: turning signals into forecasts

### C1. Labels that match the goal
We don't just label "5-day return". We use **triple-barrier labels** (López de Prado 2018). For each stock-day, the label records which of these is hit first within 5 trading days:
- an upper barrier (+7%, and a volatility-scaled version);
- a lower barrier (a stop, 2 × ATR);
- the time limit.

This teaches the model the exact event we trade on: **does it reach the target before it hits the stop?**

### C2. Model stack
| Layer | Model | Output |
|---|---|---|
| 1. Cross-sectional ranker | LightGBM **LambdaRank** by date (ranks stocks against each other) | Relative attractiveness today |
| 2. Barrier classifier | LightGBM multiclass on the triple-barrier labels | P(target first), P(stop first), P(neither) |
| 3. Distribution model | LightGBM **quantile regression** (5th … 95th percentiles of the 5-day return) | Full return shape per stock |
| 4. Evidence prior | Fixed-weight z-score composite from Part B (cannot overfit) | A stable anchor |
| 5. Meta-labeler | A small model that predicts *when layers 1–3 are right* (regime, dispersion, signal agreement) | Confidence per pick → sizing |

- **Stacking:** layers 1–4 are combined by a logistic stacker trained on out-of-fold predictions only.
- **Calibration:** isotonic regression, so that a predicted "30%" really happens about 30% of the time.
- **Interval coverage:** conformal prediction wraps the quantile outputs so their intervals have guaranteed coverage.

### C3. Volatility forecast (the scale of the distribution)
Three forecasts blended by recent accuracy:
- **HAR-RV** (Corsi 2009) from daily ranges;
- **option implied volatility**, the market's own forecast, which is usually the best;
- **earnings jump:** if earnings fall inside the week, add the options-implied earnings move (from the straddle around the date) as a separate jump.

Tails use a Student-t distribution (fitted degrees of freedom around 3–5), because stock returns are fat-tailed.

### C4. Training discipline
- Walk-forward training on 2012–2026, refit every week on a rolling 6-year window.
- **Purged k-fold cross-validation with embargo** (López de Prado), so overlapping 5-day labels can't leak between train and test.
- Features are cross-sectionally rank-normalised each day, which removes most market-level drift.
- Feature neutralisation: returns are made partly sector-neutral so the model doesn't just learn "this sector did well".
- Monotonic constraints on features whose direction is well established (for example, MAX can only hurt). This cuts overfitting.

---

## Part D — From stock forecasts to one portfolio forecast

### D1. Joint scenario simulator
Every evening we simulate **20,000 scenarios** of next week's returns for the top ~40 candidates:

```
r_i = β_i,mkt·F_mkt + β_i,sector·F_sector + ε_i + J_i·1[earnings in window]
```

- The factor moves (market, sector) come from a regime-conditional covariance matrix with Ledoit-Wolf shrinkage.
- The stock's own move ε_i is Student-t, scaled by the C3 volatility, and centred on the stacked model's expected return.
- Earnings jumps J_i are sized from options-implied earnings moves.

This matters because **correlation decides whether a +7% week is possible**. Eight stocks that all move together behave like one stock. Eight independent stocks each with a small edge add up to a real edge.

### D2. Optimiser: maximise P(week ≥ +7%) under a tail-risk limit
- Direct optimisation over the scenarios (sample-average approximation). The step function "above 7% or not" is replaced by a smooth S-curve, and the weights are solved by gradient ascent with the CVaR constraint, then checked by exact counting over the scenarios.
- Candidate portfolios of 4–10 names are found by greedy build-up plus local swap search (fast, reliable).
- **Constraints:**
  - max 25% per name;
  - max 40% per sector;
  - max 10% in the squeeze module;
  - at most 2 names holding through earnings;
  - no name with a negative expected return (so the optimiser can't "buy volatility" with no edge);
  - average-week CVaR limit.
- **Output:** target weights, **P(+7% this week)** and the expected worst-5% week, all shown on the dashboard.

### D3. Regime engine
- A 3-state hidden Markov model on SPY returns, VIX, breadth and credit spreads: **Calm-Bull, Choppy, Stress**.
- Each regime has its own covariance matrix, signal weights (for example, reversal works better in Choppy; momentum and 52-week-high in Calm-Bull) and maximum gross exposure (100% / 80% / 40%).
- In Stress, the optimiser may hold cash, because keeping the account alive beats a low-probability shot.

---

## Part E — Within-week control (goal-seeking, capped)

The week is a race to +7% by Friday. A precomputed **policy table** tells the engine how much risk to carry given:
- the return so far this week;
- the trading days left;
- the regime;
- the current edge estimate.

The table is built offline by dynamic programming over the D1 simulator, in the spirit of Browne (1999), with hard caps:

| Situation | Action |
|---|---|
| Week reaches **+7%** | Bank it: cut to ~40% exposure and tighten stops to protect about +6%. What remains keeps some upside open. |
| On track (e.g. +4% by Wednesday) | Hold the plan and keep normal stops |
| Behind, with the edge still positive | Moderate increase, never above 1.25× base risk |
| Behind, with the edge gone negative | **No bold play**: cut risk. Chasing a target with no edge is how accounts die. |
| Week reaches **−8%** | Brake: one-third exposure until Monday |

**Monthly overlay:** if the account falls to −20% from its peak, base risk halves until it recovers half of the drawdown.

---

## Part F — Execution

| Time (ET) | Job |
|---|---|
| 07:30 | Pull EDGAR filings from overnight (Form 4, 13D, 8-K, 424B) and pre-market movers; update event flags |
| 08:45 | Refresh the universe; pre-score; pre-market gap check against news (Chan filter) |
| 09:50 | First risk pass, after the opening volatility settles |
| every 30 min, 10:20–15:20 | Risk only: stops, banking, brakes, and new-filing alerts that can force an exit (e.g. an offering) |
| 15:40 | **Main rebalance** to the optimiser's weights (overnight hold earns the overnight return) |
| 16:15 | Equity snapshot; score every prediction; update the dashboard |
| 20:00 | Evening model run: after-hours earnings, full scenario simulation, tomorrow's plan |
| Sat | Weekly retrain, recalibration, signal health report, weekly scorecard |

**Regular trading hours only (canon C6):** the broker refuses any order outside 9:30–16:00 ET on market days. There is no pre-market or after-hours trading. Positions may be held overnight, but every buy and sell happens during the regular session. Filings that arrive after hours are acted on the next session, from 09:50.

**Options signals are live-only:** free options data has no history, so the B2 options features can't be backtested. They enter the live score as a small, capped tilt, and Part M measures their live IC and promotes or demotes them.

**Order handling:**
- Marketable limit orders (mid + ⅓ spread) rather than market orders.
- Skip any stock with a spread wider than 0.5%.
- Rebalance only when the weight change is greater than 3 percentage points, which avoids churn.
- Keep a pattern-day-trader budget: at most 3 same-day round trips per 5 days, reserved for emergency exits. (Check the current FINRA rule status at build time.)

Every order is reconciled against Alpaca's actual state, so a missed or duplicated cron run can't corrupt the ledger.

---

## Part G — Live learning and self-monitoring

- **Signal health:** every day, for every signal, the cross-sectional information coefficient (IC: how well the signal ranked all ~2,500 stocks). Weights update with Bayesian shrinkage, so they move slowly and only on real evidence. A signal whose IC turns significantly negative is demoted automatically.
- **Drift detection:** a population stability index on feature distributions. When the market changes character, the engine leans back toward the evidence prior.
- **Calibration watch:** a daily Brier score and reliability curve. If "30%" stops meaning 30%, recalibrate.
- **Post-mortems:** every closed position gets an automatic note: what the model expected, what happened, and which signals were right or wrong.

---

## Part H — Proving it works

### H1. Before go-live (backtest, 2012–2026)
1. Point-in-time data. Survivorship bias is measured by comparing against a delisted-inclusive sample.
2. Costs: 5–30 bp per trade depending on liquidity, plus a spread model.
3. **Deflated Sharpe ratio** (Bailey & López de Prado 2014) and **probability of backtest overfitting** (Bailey et al. 2017). Every variant tried is counted, not just the winner.
4. **Placebo tests:** shuffle the labels and the backtest must collapse; shift signals one day into the future (cheating) and it must look suspiciously good. Both confirm the pipeline isn't leaking.
5. Ablation: remove each signal family and see what breaks.
6. Output: the backtest distribution of weekly returns, % of weeks ≥ +7%, % ≤ −7%, and the worst month.

**Ship gate:** it must beat the random-volatility-matched baseline, after costs, at a deflated-Sharpe significance level.

### H2. During the month
Four weekly outcomes can't prove skill. So we measure at three levels:

| Level | Sample in one month | What it proves |
|---|---|---|
| Portfolio vs +7% target | 4 weeks | The headline (mostly luck) |
| Portfolio vs baselines (SPY, Random-Vol, Trend-Chaser) | 4 weeks, paired | Skill vs risk vs trend-following |
| **Universe-wide prediction scoring** (daily IC, calibration of P(target) on every stock) | about 50,000 predictions | **Whether the engine actually forecasts**, with real statistical power |

**Baselines**, each running its own shadow $1,000 portfolio:
- **SPY:** buy and hold the market.
- **Random-Vol:** 8 random stocks at matched volatility, redrawn weekly.
- **Trend-Chaser:** the top 8 stocks by 1-month return plus attention, which is literally "following trending stocks".
- **Evidence-Only:** the Part B composite without machine learning. This shows whether the ML earns its complexity.

---

## Part I — Data and architecture

### I1. Data stack (free, verified late Sep 2026)
| Need | Source |
|---|---|
| Prices, bars, simulated trading | **Alpaca** paper account at $1,000 (IEX real-time, historical bars, 200 requests/min) |
| Backup prices / deep history | yfinance, Stooq |
| Filings: Form 4, 13D, 8-K, 424B, XBRL fundamentals, SIC codes | **SEC EDGAR** (10 requests/sec, contact User-Agent required) |
| Earnings calendar / surprises | Nasdaq calendar endpoint; Finnhub (~60/min, unverified) |
| Options chains / implied vol | yfinance (15-minute delay), top ~60 names per run |
| Short interest | FINRA API (twice monthly) |
| News / attention | Finnhub news, GDELT |
| Macro, credit spreads, yield curve | FRED |

Tiingo and Twelve Data free tiers forbid displaying their data, so they are used only internally, if at all.

### I2. Architecture ($0)
```
GitHub Actions cron ─► Python engine ─► Alpaca paper (orders, fills, equity)
                         │    └─► data cache (Parquet, Actions cache + repo LFS-free chunks)
                         └─► state/*.json ─► GitHub Pages dashboard
```
- **Compute:** a standard runner has 4 CPUs and 16 GB RAM. The weekly retrain on ~8M rows × 90 features in float32 takes about 20–40 minutes; the daily runs take about 2–5 minutes.
- **Reliability:**
  - cron slots are off the hour;
  - a `concurrency` group stops runs overlapping;
  - every run is idempotent;
  - there is a heartbeat file, and the dashboard shows "engine last ran X min ago".
- **Secrets:** Alpaca keys live in GitHub Actions secrets only.

---

## Part J — Dashboard
- **Live strip:** equity, this week's % against the +7% line, **model P(+7% this week)**, expected worst-5% week, regime badge.
- **Weekly scorecard:** bars with the 7% line, and each baseline alongside.
- **Equity curves:** Weekly7 vs SPY vs Random-Vol vs Trend-Chaser vs Evidence-Only.
- **Position cards:** why the stock was picked, shown as SHAP top contributors in plain words ("insider cluster buy, 3 execs, $1.2M"), plus its return distribution, stop, target and days held.
- **Signal health:** the IC trend for each signal family.
- **Calibration chart:** predicted vs actual hit rate.
- **Trade ledger and daily post-mortems.**

---

## Part K — Build phases
1. **Foundation:** repo, Alpaca paper connection, universe, bar cache, EDGAR client.
2. **Features:** the Part B families with point-in-time timestamps and unit tests against known historical events.
3. **Labels and models:** triple-barrier labels, the four-layer stack, purged cross-validation, calibration.
4. **Risk:** volatility blend, scenario simulator, regime model.
5. **Optimiser and policy:** the D2 optimiser and the Part E policy table.
6. **Backtest and validation:** all the Part H1 tests; ship gate.
7. **Execution and ops:** executor, reconciliation, schedule, baselines, dashboard.
8. **Dry run:** 3 trading days, then reset to $1,000 and start the month.

## Part L — Risk register
| Risk | Mitigation |
|---|---|
| Free API breaks (yfinance, Nasdaq endpoint) | Every source has a fallback; the engine degrades to fewer features, never crashes |
| Cron runs dropped | Idempotent reconciliation; heartbeat on the dashboard |
| Overfitting | Evidence prior, monotonic constraints, deflated Sharpe, overfitting probability, placebo tests |
| A single-stock blow-up (fraud, offering, FDA fail) | 25% cap, filing watch every 30 min, CVaR limit, no binary-event names beyond the capped module |
| Chasing the target in a bad week | The no-bold-play-without-edge rule; the −8% brake |
| Short sample misleads | Universe-wide prediction scoring (H2) |

---

## Part M — The self-improvement engine (canon C5)

Weekly7 runs a permanent **diagnose → hypothesise → test → promote** loop, like a research lab that never closes. There is no "perfect system" end state. The loop keeps running, and every change must *prove* it helps before it is allowed to trade.

### M1. Diagnose: find exactly what did not work
Every day and every week, each outcome is broken into **where the result came from** (performance attribution), with a counterfactual for each part:

| Layer | Question | Counterfactual used |
|---|---|---|
| Regime | Was the market-mood call right? | Same picks with a regime-neutral exposure |
| Selection | Which signal families ranked the universe well or badly? | Daily IC per family on all ~2,500 stocks |
| Forecast | Were the probabilities calibrated? Were the ranges too narrow? | Brier score, reliability curve, quantile coverage |
| Construction | Did the optimiser add value over simple equal weights? | Equal-weight version of the same picks |
| Within-week control | Did the bank-at-+7% and −8% brake help or hurt? | Replay of the week with each rule switched off |
| Execution | Slippage vs the decision price; missed runs | Ideal fills at the decision price |
| Data | Stale, missing or late feeds | Feed-health log |

Each failure is tagged from a fixed **failure taxonomy**, for example:
- `SIGNAL_DECAY`, `MISCALIBRATED`, `REGIME_MISS`, `CORRELATION_UNDERESTIMATED`, `EVENT_SHOCK`, `COST_LEAK`, `RULE_HURT`, `DATA_FAULT`.

The tags accumulate. A failure that keeps recurring becomes a **finding** with evidence attached; a one-off bad day does not.

### M2. Hypothesise: a fix for each finding
Every finding produces one or more **challengers**, each a specific, testable change:
- **Automatic challengers,** generated by the engine:
  - re-weight or demote a signal family;
  - retune a threshold (stops, bank level, brake level, exposure caps);
  - recalibrate;
  - change the volatility blend;
  - change the correlation shrinkage;
  - change the training window.
- **Research challengers,** written by me (Claude) during review sessions: a new signal from the literature, a new model, or a new rule. Each is recorded with its reasoning and the finding it answers.

### M3. Test: challenger vs champion, with no self-deception
A challenger must pass **all three** tests:
1. **Historical:** purged walk-forward backtest over 2012–2026 against the current champion, on the same data and costs.
2. **Live shadow:** it runs alongside the champion as a paper-only shadow portfolio for at least 5 trading days, making predictions on the whole universe.
3. **Statistics:**
   - improvement in universe-wide prediction score (IC, Brier score, P(target) calibration), significant after a **multiple-testing correction** (every challenger ever tried is counted, as in the deflated Sharpe ratio);
   - no worse tail risk (CVaR).

Weekly P&L alone can **never** promote a challenger. Four weeks is noise, and chasing it is how systems get worse.

### M4. Promote, monitor, roll back
- A winner becomes the champion. It is versioned (`engine v1.3 → v1.4`) with a change note.
- A **tripwire** watches every new champion for 10 days. If its live prediction score falls below the previous champion's shadow, it rolls back automatically.
- **Guardrails nothing can override:**
  - the canon;
  - the risk caps in Part D2 and Part E;
  - point-in-time data rules;
  - the ban on bold play without an edge.

### M5. Experiment registry (the lab notebook)
`state/experiments.jsonl` records every hypothesis:
- the finding it answers;
- the exact change;
- the test results;
- the decision and the date.

The registry is shown on the dashboard. It also serves as the trial count for the multiple-testing correction, so the system can't quietly try 200 variants and keep the lucky one.

### M6. Cadence
| When | Loop step |
|---|---|
| Daily 16:15 | Score predictions, attribution, failure tags |
| Saturday | Findings, automatic challengers, backtests, start shadows, promote/roll back, weekly improvement report |
| Each review session | I read the report, write research challengers, and update the blueprint when a finding changes the design |

---

## What's needed from you
1. A free **Alpaca** account. Reset the paper account to $1,000 and create API keys.
2. A **GitHub** account, and whether the repo should be public or private.

## Key references
Browne 1999 (Adv. Appl. Prob.); Dubins & Savage 1965; López de Prado 2018 (*Advances in Financial ML*); Bailey & López de Prado 2014; Bailey, Borwein, López de Prado & Zhu 2017; Gu, Kelly & Xiu 2020; Chan 2003; Cohen, Malloy & Pomorski 2012; Brav, Jiang, Partnoy & Thomas 2008; Cremers & Weinbaum 2010; An, Ang, Bali & Cakici 2014; Pan & Poteshman 2006; Xing, Zhang & Zhao 2010; Bernard & Thomas 1989; Brandt et al. 2008; Jegadeesh & Livnat 2006; Frazzini & Lamont; Thomas & Zhang 2008; George & Hwang 2004; Moskowitz & Grinblatt 1999; Gervais, Kaniel & Mingelgrin 2001; Lou, Polk & Skouras 2019; Da, Gurun & Warachka 2014; Bali, Cakici & Whitelaw 2011; Ang, Hodrick, Xing & Zhang 2006; Barber & Odean 2008; Da, Engelberg & Gao 2011; Boehmer, Jones & Zhang 2008; Lucca & Moench 2015; Corsi 2009; Ledoit & Wolf 2004; Novy-Marx & Velikov 2016.

---

## Part N — Evidence log (what the data actually showed)

Measured on 5,247 US stocks, 2013–2026, with walk-forward out-of-sample predictions. Every entry is also in `state/experiments.jsonl`. **Caveat:** price history covers today's listed stocks only (survivorship bias), which flatters every number below slightly.

**28 Sep 2026, first research pass**

| Finding | Number | Action taken |
|---|---|---|
| The evidence prior forecasts next-week returns | IC +0.025, t≈7 | kept |
| The expected-return model forecasts | IC +0.025, t≈8 | kept, now trained on market-relative returns |
| The LambdaRank ranker scored backwards | IC −0.010, t=−2.8 | retired |
| P(touch +7%) is calibrated **but measures volatility, not edge**; its top decile *loses* 0.69%/week | IC −0.021 | removed from the ranking score; kept only as a displayed probability |
| Maximising P(week ≥ +7%) with daily rebalancing turned over 145× the portfolio in 6 months; costs took 28% | $1,000 → $689 in 5 months | weekly rebalancing, a liquidity floor ($20M/day), hysteresis |
| Top-2% picks beat the average stock by ~+0.2%/week in every volatility bucket **except the top 10%, which loses 1%/week** | — | volatility / MAX filter kept |
| Simple top-8, weekly, holdings kept while in the top 20%, after 10 bp costs | $1,000 → $4,069 vs SPY $3,903; 2% of weeks ≥ +7% | this is the realistic bar |
| Concentrated, higher-volatility top picks (4 names) | $3,784; **6.3% of weeks ≥ +7%** (SPY 0.6%); drawdown −57% | how the +7% goal can be chased *with* edge |
| Buying right after a big positive earnings reaction **reverses** since 2015 | −0.53%/week (t=−2.5) | consistent with Martineau 2022 ("PEAD is dead"); the live loop watches it |
| News-driven 5-day drops keep falling (Chan 2003 confirmed) | −0.39%/week, t=−8.8 | supports the news/no-news split |
| Lottery stocks (top 5% MAX) underperform | −0.67%/week, t=−20 | confirmed penalty |
| Share offerings underperform | −0.10%/week, t=−3.8 | confirmed penalty |
| Volume-surge and overnight-strength priors had the wrong sign pre-2017 | t=−2.2 / −0.9 | priors switched off (checked on pre-2017 data only, to avoid tuning on the test years) |
| Activist 13D filings scored negative | −0.47%/week, t=−8.5 | open question: the submissions feed may attribute 13Ds to the *filer* as well as the target; to be fixed before trusting the signal |

**The honest conclusion so far:** published signals on free data give a small, real edge (roughly market-plus-a-little). Weeks of +7% occur only when the portfolio is concentrated in higher-volatility names that still carry an edge, and that also brings deep drawdowns. The engine is built to chase +7% that way and to report plainly how often it happens.

**28 Sep 2026, v1.1 champion selected (supersedes Part D2's optimiser and Part E's bank rule for live trading)**

| Test (2017–2026, out-of-sample, after costs) | $1,000 became | Weeks ≥ +7% | Worst drawdown |
|---|---|---|---|
| v1.1 model: insider features, market-relative regressor, ranker retired → blended score | IC 0.030 (t=9.5), up from 0.022 | — | — |
| Scenario optimiser, P(+7%) objective, 3×ATR stops | $787: stops fired 222 times for −$2,695 | 2.6% | −67% |
| **Top-4 equal weight, kept while in the top 20%, weekly, −8% brake** | **$9,815 (26%/yr)** | **3.9%** | **−45%** |
| Same, plus bank at +7% | $8,724 | 4.1% | −50% |
| Top-3 (would break the 25% single-stock cap) | $12,797 | 4.7% | −42% |
| S&P 500 | $3,903 (15.6%/yr) | 0.6% | — |

**Decision:** the live champion is the simple top-4 constructor (`policy.topk_targets`), rebalanced at the week's last close, with the −8% weekly brake. The scenario optimiser, per-stock stops and the +7% bank rule stay in the code as **challengers**. They return only if they beat the champion in the Part M tests. Top-3 is logged as a challenger that would first need the owner to relax the 25% cap.

**Expectation for the live month, from the backtest distribution:** a +7% week is roughly a 1-in-25 event; the median week is about +0.5%. The dashboard reports the real outcome against this.

**28 Sep 2026, v1.2 sector cap.** The 40% cap (1 name per SIC division at k=4) cut the backtest to $6,458. Two per division (50%) raised it to $13,198 (4.5% of weeks ≥ +7%, drawdown −47%). The cap is now **50% (2 of 4 names)**. Found because the first live portfolio was 75% financials.

**28 Sep 2026, v1.3 from tuning-lab round 1 (canon C7, C8).** 120 configurations × 400 random 4-week months each, drawn from odd years (tuning) and even years (locked exam). A clairvoyant cheat control scored 3.86 +7% weeks/month against the best real config's 0.34, so no leakage. The winner holds 3 names (the most volatile of the top 5% by score), keeps each while it stays in the top 10%, uses the −8% brake, no sector cap, and a $50M/day liquidity floor. On the locked years it made 0.39 +7% weeks/month vs 0.30 for v1.2 (z 3.37 against 3.34 needed). The cost: its average month is +0.9% vs +3.5%, and its 5th-percentile month is −19.5% vs −10.8%. It was promoted because the owner's objective is +7% weeks (C8), and it stayed inside the guards (positive mean month; 5th percentile better than −25%). The single-stock cap is now 34% (3 names). The sector cap from v1.2 was dropped by the lab.
