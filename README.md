# Weekly7

A stock-picking engine and a $1,000 simulation that aims for the **whole portfolio** to gain **+7% a week**. It is measured against honest baselines so that skill can be told apart from luck.

- `canon/CANON.md`: the owner's directives, verbatim and hash-locked. These govern everything.
- `BLUEPRINT.md`: the design, including Part M, the self-improvement loop.
- `engine/`: the pipeline:
  - `universe`, `data`, `edgar`: the stock list, prices and SEC filings;
  - `features`, `model`, `train`: signals and machine learning;
  - `portfolio`: scenarios and the optimiser;
  - `live`, `broker`, `tick`: trading and scheduling;
  - `scoring`, `shadows`, `improve`: evaluation, baselines and the learning loop.
- `docs/`: the dashboard, served by GitHub Pages.
- `state/`: the ledger, predictions, experiments and reports. The engine commits these.

Runs free on GitHub Actions. Paper trading goes through Alpaca when `ALPACA_KEY` / `ALPACA_SECRET` secrets exist; otherwise the local ledger is used.
