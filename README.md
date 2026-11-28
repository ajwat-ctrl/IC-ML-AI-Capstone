# IC-ML-AI-Capstone: Black-Box Optimisation (BBO)

Imperial College ML & AI capstone. The task was to find the maximum of eight unknown functions (2 to 8 inputs each, all inputs scaled to [0, 1)) when each function could only be queried once per round.

**Status:** complete. Results for all 13 rounds are included. The only gap is the round 6 query log, which I no longer have; its results are included.

## In plain language (non-technical summary)

Imagine searching for the highest point in eight foggy landscapes, but you may only check one spot per landscape each round. I used a statistical model that learns from every earlier check to predict where the highest ground might be, and how unsure it is. Over thirteen rounds every landscape ended higher than my first check, and three of them set their best result in the very last round. Three landscapes were also noisy: checking the same spot twice gave different answers. I cannot be sure I found any true peak, but the method turned each new result into a better-informed next guess.

## Contents

| Path | What it is |
|------|------------|
| `notebooks/bbo_analysis.ipynb` | Walk-through of the method and results, with outputs. Regenerates every table and plot from `data/` |
| `bbo.py` | Core code: GP fitting, output transforms, acquisition functions (UCB, Expected Improvement), query formatting, CSV helpers |
| `cli.py` | Command line: `suggest` a query per function, `log` a result |
| `config.json` | Per-function settings (acquisition, kappa or xi, transform, Matern nu, noise, local search scale) |
| `data/observations.csv` | Every submitted query and the value the portal returned (rounds 1 to 13) |
| `data/submissions.json` | The same results in the format used during the project |
| `data/repeat_queries.csv` | The repeat of round 8's query on all eight functions, used to measure noise |
| `data/rounds/` | Per-round query logs for rounds 2 to 5 and 7 to 13 (round 6 is missing), with settings and model diagnostics |
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

Settings were the same in every logged round, 2 to 13 (see `config.json`), so differences between queries come from new data rather than retuning. The transform definitions in `bbo.py` were inferred from the logged diagnostics and are a reconstruction, not a copy of the original submission code. Round 1 settings were not logged.

## How to reproduce

```bash
git clone https://github.com/ajwat-ctrl/IC-ML-AI-Capstone.git
cd IC-ML-AI-Capstone
pip install -r requirements.txt
jupyter notebook notebooks/bbo_analysis.ipynb
```

`python cli.py suggest --round 14` proposes queries from the points in `data/observations.csv`. The submitted queries were also fitted on the initial data supplied by the capstone (10 to 40 points per function), which is not stored here, so proposals will not match them exactly.

## Data

`data/observations.csv` has 104 rows (8 functions x 13 rounds): `function, round, query, y, rationale`. The data is tiny, so it is stored directly in the repo. `data/repeat_queries.csv` holds one extra evaluation per function (round 8's query submitted again between rounds 8 and 9); it was not used to train the model. See `DATASHEET.md`.

## Results (all 13 rounds)

| Function | Inputs | Round 1 result | Best submitted result | Round of best | Round 13 result |
|----------|--------|----------------|-----------------------|---------------|-----------------|
| 1 | 2 | about 0 (1.2e-149) | about 0 (3.1e-11) | 13 | 3.1e-11 |
| 2 | 2 | 0.437 | 0.765 | 10 | 0.436 |
| 3 | 3 | -0.133 | -0.0266 | 8 | -0.0836 |
| 4 | 4 | -3.617 | 0.295 | 13 | 0.295 |
| 5 | 4 | 150.2 | 8662.4 | 10 | 8642.2 |
| 6 | 5 | -0.603 | -0.212 | 11 | -0.257 |
| 7 | 6 | 1.117 | 1.856 | 9 | 1.593 |
| 8 | 8 | 9.808 | 9.994 | 13 | 9.994 |

The table covers my submitted queries only. The model was also trained on initial data supplied by the capstone, which is not stored here. In the round 13 log, the best point the model had seen for every function matched my best submitted result up to round 12, so the supplied data held no better point.

Highlights:
* **Function 5** rose from 150 to 1,581 to 6,065 to 8,329 in rounds 1 to 4 as queries moved to a high corner of the input space, then stayed within about 4% of that (6,368 to 8,662) for rounds 5 to 13. The best (8,662) came in round 10 at the corner (0.999999 on all four inputs).
* **Function 8** improved in small steps: 9.808 in round 1, 9.956 in round 4, 9.974 in round 11 and 9.994 in round 13, a new best in every round from 9 to 13.
* **Function 4** fell sharply in rounds 3 and 4 (-15.7, then -22.1), then reached 0.041 in round 5, 0.272 in round 12 and 0.295 in round 13 by moving to a narrow region around (0.39, 0.41, 0.39, 0.43).
* **Function 2** peaked at 0.765 in round 10 but returned 0.436 in round 13 at almost the same input, which shows how noisy it is.
* **Functions 3, 6 and 7** set their bests in rounds 8, 11 and 9 and did not beat them afterwards.
* **Function 1** returned values indistinguishable from zero in every round.
* **Noise:** re-submitting round 8's query gave a different value for Functions 2, 3 and 6 (differences about 0.06, 0.005 and 0.065) and identical values for the other five.

## Limitations

See `MODEL_CARD.md`. In short: very few observations per function, a model that can be wrong in unexplored regions, and no guarantee of reaching the global maximum.

## Licence

MIT, see `LICENSE`.
