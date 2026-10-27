# Model card: GP-based query selector for the BBO capstone

Follows the structure of "Model Cards for Model Reporting" (Mitchell et al.). Items marked `[TODO]` need your own input.

## Model details

- **Type:** Gaussian process regression (scikit-learn `GaussianProcessRegressor`) used as a surrogate model, with an acquisition function that selects the next query.
- **Kernel:** constant x ARD Matern, plus a white-noise term for all functions except Function 5. Hyperparameters fitted by maximum likelihood.
- **Per-function settings** (identical in rounds 2 to 4; `config.json`):

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
- **Author and version:** [TODO: your name]; version at round 4 (2026-10-27).
- **Licence:** MIT.

## Intended use

- **Primary:** choosing one query per function per round in the BBO capstone, where evaluations are scarce and the functions are unknown.
- **Out of scope:** safety-critical or high-stakes decisions; problems with many more inputs or evaluations; discrete or constrained search spaces (continuous inputs in [0, 1) are assumed).

## Factors

- **Input dimension:** 2 to 8. The model degrades as dimension rises relative to the number of observations.
- **Function behaviour:** smooth functions suit the model. Function 1 (outputs almost all about zero) and Function 4 (sharp drops) are harder.

## Metrics

- **Primary:** best value found per function (the capstone objective). After round 3: F1 about 0, F2 0.648, F3 -0.0489, F4 -1.136, F5 6065.5, F6 -0.5075, F7 1.521, F8 9.922 (best of submitted queries).
- **Model checks logged each round** (leave-one-out, on transformed outputs). Round 4 values:

| Function | Points used | LOO MAE | Calibration ratio |
|----------|-------------|---------|-------------------|
| 1 | 13 | 0.369 | 1.31 |
| 2 | 13 | 0.117 | 1.02 |
| 3 | 18 | 0.053 | 1.28 |
| 4 | 33 | 0.293 | 0.98 |
| 5 | 23 | 1.000 | 1.07 |
| 6 | 23 | 0.251 | 1.18 |
| 7 | 33 | 0.184 | 0.85 |
| 8 | 43 | 0.087 | 0.72 |

A calibration ratio near 1 means the model's stated uncertainty matches its real errors; above 1 means overconfident; below 1 underconfident. The exact definition is in the original pipeline, which this repo does not reproduce, so treat the ratio as indicative.

## Training and evaluation data

- Refit from scratch each round on the supplied initial data plus all submitted results (see `DATASHEET.md`). There is no separate test set, because each new query is itself the test.

## Quantitative analysis

- See `notebooks/bbo_analysis.ipynb` for results by round and diagnostics. Highlights: Function 5 improved from 150 to 6,065 over three rounds; Function 4's model error rose from 0.143 to 0.297 after its round 3 query landed at -15.7, showing that one extreme point can change how well the model fits.

## Ethical considerations

- No personal data and no direct impact on people.
- Sampling is adaptive and not representative, so statistics from the data should not be generalised.
- Risk of over-trusting the model: predictions look confident near sampled points but can be badly wrong elsewhere. Functions 1, 3 and 5 were overconfident at times (calibration above 1.2).

## Caveats and recommendations

- **Few observations:** ARD lengthscales at their upper bound (10) mean an input looked irrelevant or the data could not tell; they should not be read as proof.
- **No guarantee of the global maximum.** Settling on a corner (as Function 5's queries do) may find a boundary optimum or miss a better interior one.
- **Random candidate search** is crude in 6 to 8 dimensions; multi-start gradient-based optimisation of the acquisition function would be better.
- **Fixed settings across rounds:** kappa and xi did not change from round 2 to 4, so exploration was not reduced over time. [TODO: say whether you plan to change this in later rounds.]
- **Possible improvements:** re-tuning settings per function as data grows, checking noise with repeat queries, a trust region around the best point, and filtering out poor regions before the acquisition step.
