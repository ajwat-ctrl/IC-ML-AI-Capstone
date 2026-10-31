# IC-ML-AI-Capstone: Black-Box Optimisation (BBO)

Imperial College ML & AI capstone. The task was to find the maximum of eight unknown functions (2 to 8 inputs each, all inputs scaled to [0, 1)) when each function could only be queried once per round.

**Status:** results for rounds 1 to 4 are included. Round 5 queries are logged but their results were still pending when this was written. Later rounds will be added as they are completed.

## In plain language (non-technical summary)

Imagine searching for the highest point in eight foggy landscapes, but you may only check one spot per landscape each week. I used a statistical model that learns from every earlier check to predict where the highest ground might be, and how unsure it is about each area. Over the first four rounds, seven of the eight landscapes gave better results than my first check, four of them improving again in the latest round, while one stayed flat. With so few checks I cannot be sure I have found any true peak, but the method turns each new result into a better-informed next guess.

## Contents

| Path | What it is |
|------|------------|
| `notebooks/bbo_analysis.ipynb` | Walk-through of the method and results, with outputs. Regenerates every table and plot from `data/` |
| `bbo.py` | Core code: GP fitting, output transforms, acquisition functions (UCB, Expected Improvement), query formatting, CSV helpers |
| `cli.py` | Command line: `suggest` a query per function, `log` a result |
| `config.json` | Per-function settings (acquisition, kappa or xi, transform, Matern nu, noise, local search scale) |
| `data/observations.csv` | Every submitted query and the value the portal returned (rounds 1 to 4) |
| `data/submissions.json` | The same results in the format used during the project |
| `data/rounds/` | Per-round query logs for rounds 2 to 5, with settings and model diagnostics |
| `DATASHEET.md` | Datasheet for the dataset |
| `MODEL_CARD.md` | Model card for the optimisation model |
| `reflections/` | Written reflections |
| `requirements.txt` | Dependencies |
| `LICENSE` | MIT |

## Method

Each round, for every function, the pipeline:

1. **Transforms the outputs** (per function): none, rank (for Function 1, whose outputs are almost all about zero), signed log (Function 4), or log(1+y) (Function 5, which spans orders of magnitude).
2. **Fits a Gaussian process** (scikit-learn) with an ARD Matern kernel (nu = 1.5 for Function 2, 2.5 otherwise) and, except for Function 5, a learned noise term.
3. **Scores 20,000 candidate points** (half near the current best, half uniform) with an acquisition function: Upper Confidence Bound (`mu + kappa * sigma`) for Functions 1, 2, 4, 5, 7, 8 and Expected Improvement for Functions 3 and 6.
4. **Submits the best-scoring candidate** that is not a near-duplicate of an earlier query, formatted to six decimals.
5. **Logs the settings and two leave-one-out checks** (error and calibration) in `data/rounds/`.

Settings were the same in rounds 2 to 5 (see `config.json`), so differences between queries come from new data rather than retuning. The transform definitions in `bbo.py` were inferred from the logged diagnostics and are a reconstruction, not a copy of the original submission code. Round 1 settings were not logged.

## How to reproduce

```bash
git clone https://github.com/ajwat-ctrl/IC-ML-AI-Capstone.git
cd IC-ML-AI-Capstone
pip install -r requirements.txt
jupyter notebook notebooks/bbo_analysis.ipynb
```

`python cli.py suggest --round 6` proposes queries from the points in `data/observations.csv`. The submitted queries were also fitted on the initial data supplied by the capstone (10 to 40 points per function), which is not stored here, so proposals will not match them exactly.

## Data

`data/observations.csv` has 32 rows (8 functions x 4 rounds): `function, round, query, y, rationale`. The data is tiny, so it is stored directly in the repo. See `DATASHEET.md`.

## Results (rounds 1 to 4)

| Function | Inputs | Best submitted result | Round | Best of all points (incl. supplied data)* |
|----------|--------|-----------------------|-------|--------------------------------------------|
| 1 | 2 | about 0 (1.2e-149) | 1 | 7.7e-16 |
| 2 | 2 | 0.702 | 4 | 0.702 |
| 3 | 3 | -0.0489 | 3 | -0.0348 |
| 4 | 4 | -1.136 | 2 | -1.136 |
| 5 | 4 | 8328.7 | 4 | 8328.7 |
| 6 | 5 | -0.5075 | 3 | -0.5075 |
| 7 | 6 | 1.755 | 4 | 1.755 |
| 8 | 8 | 9.956 | 4 | 9.956 |

\* From the model's training set before round 5 (`data/rounds/round_05_queries.json`).

Highlights: Functions 2, 5, 7 and 8 each set a new best in round 4. Function 5 rose from 150 to 1,581 to 6,065 to 8,329 as queries moved towards a corner of the input space. Functions 3 and 6 got worse in round 4 than their round 3 results. Function 4 fell sharply in rounds 3 and 4 (-15.7, then -22.1), showing how steeply that function drops away from its best region. Function 1 returned values indistinguishable from zero in all four rounds.

## Limitations

See `MODEL_CARD.md`. In short: very few observations per function, a model that can be wrong in unexplored regions, and no guarantee of reaching the global maximum.

## Licence

MIT, see `LICENSE`.
