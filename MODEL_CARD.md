# Model card: GP-based query selector for the BBO capstone

Follows the structure of "Model Cards for Model Reporting" (Mitchell et al.). Items marked `[TODO]` need your own input.

## Model details

- **Type:** Gaussian process regression (scikit-learn `GaussianProcessRegressor`) used as a surrogate model, with an acquisition function that selects the next query.
- **Kernel:** constant x ARD Matern, plus a white-noise term for all functions except Function 5. Hyperparameters fitted by maximum likelihood.
- **Per-function settings** (identical in every logged round, 2 to 13; `config.json`):

| Function | Acquisition | Parameter | Output transform | Matern nu | Noise term |
|----------|-------------|-----------|------------------|-----------|------------|
| 1 | UCB | kappa 3.0 | rank | 2.5 | yes |
| 2 | UCB | kappa 2.0 | none | 1.5 | yes |
| 3 | EI | xi 0.01 | none | 2.5 | yes |
| 4 | UCB | kappa 2.0 | signed log | 2.5 | yes |
| 5 | UCB | kappa 1.5 | log(1+y) | 2.5 | no |
| 6 | EI | xi 0.01 | none | 2.5 | yes |
| 7 | UCB | kappa 2.5 | none | 2.5 | yes |
| 8 | UCB | kappa 1.5 | none | 2.5 | yes |

- **Candidate search:** 20,000 random candidates per function (half drawn near the current best, with local scale 0.05, or 0.03 for Function 8); the best score wins and near-duplicates of earlier queries are removed.
- **Reconstruction note:** `bbo.py` re-creates this pipeline from the logged settings. The transform definitions were inferred from the logs, so it is not a copy of the original submission code.
- **Author and version:** [TODO: your name]; final version, round 13 (2026-11-27).
- **Licence:** MIT.

## Intended use

- **Primary:** best value found per function (the capstone objective). After all 13 rounds, best of my submitted queries: F1 about 0 (3.1e-11), F2 0.765, F3 -0.0266, F4 0.295, F5 8662.4, F6 -0.212, F7 1.856, F8 9.994. Functions 2, 3 and 6 are noisy (round 8's query gave different values when repeated), so their bests may include luck.
- **Model checks logged each round** (leave-one-out, on transformed outputs). Round 13 values (fitted on data up to round 12):

| Function | Points used | LOO MAE | Calibration ratio |
|----------|-------------|---------|-------------------|
| 1 | 22 | 0.262 | 0.99 |
| 2 | 22 | 0.115 | 1.03 |
| 3 | 27 | 0.035 | 0.95 |
| 4 | 42 | 0.410 | 0.86 |
| 5 | 32 | 0.636 | 1.26 |
| 6 | 32 | 0.210 | 1.04 |
| 7 | 42 | 0.172 | 0.71 |
| 8 | 52 | 0.047 | 0.56 |

A calibration ratio near 1 means the model's stated uncertainty matches its real errors; above 1 means overconfident; below 1 underconfident. The exact definition is in the original pipeline, which this repo does not reproduce, so treat the ratio as indicative.

## Training and evaluation data

- Refit from scratch each round on the supplied initial data plus all submitted results (see `DATASHEET.md`). There is no separate test set, because each new query is itself the test.

## Quantitative analysis

- See `notebooks/bbo_analysis.ipynb` for results by round and diagnostics. Highlights over the 13 rounds:
  - Function 5's LOO error fell from 1.54 (round 2 log) to 0.64 (round 13 log), and its best rose from 150 to 8,662, but it stalled after round 10 at the corner of the input space.
  - Function 8's LOO error fell steadily from 0.101 to 0.047 and its result improved in every round from 9 to 13.
  - Function 4's LOO error doubled from 0.14 to 0.30 once its round 2 result was added and stayed between 0.28 and 0.41. Even so, its best went from -3.6 to 0.295, reached in the last round.
  - The calibration ratio for Function 3 rose to 1.58 (overconfident) in round 7, and Functions 7 and 8 stayed below 0.75 (underconfident) throughout, so their uncertainty bands were wider than needed.
- **Repeat-query noise check:** re-submitting round 8's query gave different values for Functions 2, 3 and 6 (differences about 0.06, 0.005 and 0.065) and identical values for 1, 4, 5, 7 and 8 (`data/repeat_queries.csv`).

## Ethical considerations

- No personal data and no direct impact on people.
- Sampling is adaptive and not representative, so statistics from the data should not be generalised.
- Risk of over-trusting the model: predictions look confident near sampled points but can be badly wrong elsewhere. Functions 1, 3 and 5 were overconfident at times (calibration above 1.2; Function 5 was still 1.26 in round 13).

## Caveats and recommendations

- **Few observations:** ARD lengthscales at their upper bound (10) mean an input looked irrelevant or the data could not tell; they should not be read as proof.
- **No guarantee of the global maximum.** Settling on a corner (as Function 5's queries do) may find a boundary optimum or miss a better interior one.
- **Random candidate search** is crude in 6 to 8 dimensions; multi-start gradient-based optimisation of the acquisition function would be better.
- **Fixed settings across rounds:** kappa and xi did not change from round 2 to 13, so exploration was not reduced over time. The learned noise term was on for Functions 4, 7 and 8, which looked deterministic in the repeat test, and off for Function 5, which was consistent with it. Function 5's queries stayed at or next to one corner from round 4 on and its best did not improve after round 10, so more exploration may have helped (or the corner may simply be the optimum).
- **Noise:** Functions 2, 3 and 6 returned different values for the same input. Function 2's round 13 result (0.436) was far below its round 10 result (0.765) at a near-identical input, so its noise may be larger than the single repeat suggests.
- **Missing log:** the round 6 diagnostics are not available; settings for that round are assumed to match the other logged rounds.
- **Possible improvements:** re-tuning settings per function as data grows, repeating queries on noisy functions and averaging, setting the noise term from the repeat evidence, a trust region around the best point, filtering out poor regions before the acquisition step, and multi-start gradient-based optimisation of the acquisition function.
