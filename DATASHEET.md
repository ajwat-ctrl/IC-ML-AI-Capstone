# Datasheet: BBO capstone observations

Follows the structure of "Datasheets for Datasets" (Gebru et al.). Items marked `[TODO]` need your own input.

## Motivation

- **Why was the dataset created?** To record every query I submitted in the BBO capstone and the value each returned, so the optimisation can be reproduced and analysed.
- **Who created it, and who funded it?** Created by [TODO: your name] as part of the Imperial College ML & AI programme. No external funding.

## Composition

- **What does each row represent?** One query to one black-box function in one round.
- **Columns:** `function` (1 to 8), `round`, `query` (inputs, hyphen-separated, six decimals, each in [0, 1)), `y` (value returned by the portal), `rationale` (the settings used to choose the query).
- **How many rows?** 24 (8 functions x rounds 1 to 3). Round 4 queries are logged in `data/rounds/round_04_queries.json`; their results were pending.
- **Input dimensions:** F1 2, F2 2, F3 3, F4 4, F5 4, F6 5, F7 6, F8 8.
- **Initial data not included.** The capstone supplied initial points for each function (10, 10, 15, 30, 20, 20, 30 and 40 for functions 1 to 8, inferred from the model's point counts before round 2). The model was trained on these as well as on my submissions, but they are not stored in this repo. [TODO: add them if the course allows.]
- **Is anything missing?** Only submitted queries are recorded. The functions' true formulas and global optima are unknown. Round 1 settings were not logged.
- **Does it contain personal data?** No.

## Collection process

- **How was it collected?** Through the capstone portal, one query per function per round. Round 2 was dated 2026-10-20, round 3 on 2026-10-25 and round 4 on 2026-10-27 [TODO: add the round 1 date].
- **How were query points chosen?** By a Gaussian-process model with an acquisition function (UCB or Expected Improvement, per function). Round 1 queries were chosen before the logged pipeline settings existed [TODO: say how].
- **Is the data noisy?** Function 1 returns values that are numerically indistinguishable from zero (|y| below 1e-18). Functions 2, 3, 4, 6, 7 and 8 were fitted with a learned noise term, and Function 5 without one. Repeat queries were not used to measure noise directly.

## Preprocessing

- Inputs are used as submitted. Outputs are transformed only inside the model (rank for F1, signed log for F4, log(1+y) for F5, none otherwise); stored values are raw.

## Uses

- **Intended use:** studying and reproducing this optimisation run; a teaching example of Bayesian optimisation under a tight query budget.
- **Not suitable for:** drawing conclusions about the true functions beyond the sampled regions, benchmarking optimisers (the sample is tiny and adaptively chosen), or any decision about people.

## Biases and limitations

- **Adaptive sampling bias.** Points were chosen by the optimiser, not at random, so they cluster near promising regions. Averages and correlations from them are not representative of the input space.
- **Small sample.** Three submitted points per function so far, in up to 8 dimensions.
- **Single-run noise.** One evaluation per point, so a lucky value is indistinguishable from a good point.
- **Boundary points.** Several queries sit on or near the edge of the input range (for example Function 5 in round 4), where the model has little information.

## Distribution and maintenance

- **Where?** `data/` in this public repository, under the MIT licence.
- **Maintenance:** I will add results for round 4 and later rounds as they are completed.
- **Contact:** via the GitHub repository.
