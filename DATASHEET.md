# Datasheet: BBO capstone observations

Follows the structure of "Datasheets for Datasets" (Gebru et al.). Items marked `[TODO]` need your own input.

## Motivation

- **Why was the dataset created?** To record every query I submitted in the BBO capstone and the value each returned, so the optimisation can be reproduced and analysed.
- **Who created it, and who funded it?** Created by [TODO: your name] as part of the Imperial College ML & AI programme. No external funding.

## Composition

- **What does each row represent?** One query to one black-box function in one round.
- **Columns:** `function` (1 to 8), `round`, `query` (inputs, hyphen-separated, six decimals, each in [0, 1)), `y` (value returned by the portal), `rationale` (the settings used to choose the query).
- **How many rows?** 104 (8 functions x 13 rounds) in `data/observations.csv`, plus 8 rows in `data/repeat_queries.csv` (round 8's query submitted again; see noise below).
- **Input dimensions:** F1 2, F2 2, F3 3, F4 4, F5 4, F6 5, F7 6, F8 8.
- **Initial data not included.** The capstone supplied initial points for each function (10, 10, 15, 30, 20, 20, 30 and 40 for functions 1 to 8, inferred from the model's point counts before round 2). The model was trained on these as well as on my submissions, but they are not stored in this repo. [TODO: add them if the course allows.]
- **Is anything missing?** Only submitted queries are recorded. The functions' true formulas and global optima are unknown. Round 1 settings were not logged, and the round 6 query log is missing (round 6 queries and results are recorded, but not its model diagnostics).
- **Does it contain personal data?** No.

## Collection process

- **How was it collected?** Through the capstone portal, one query per function per round. Logged dates: round 2 on 2026-10-20, round 3 on 2026-10-25, round 4 on 2026-10-27, round 5 on 2026-10-31, round 7 on 2026-11-03, round 8 on 2026-11-10, round 9 on 2026-11-12, round 10 on 2026-11-14, round 11 on 2026-11-21, round 12 on 2026-11-24 and round 13 on 2026-11-27 [TODO: add the round 1 and round 6 dates, and the date of the repeat query].
- **How were query points chosen?** By a Gaussian-process model with an acquisition function (UCB or Expected Improvement, per function). Round 1 queries were chosen before the logged pipeline settings existed [TODO: say how].
- **Is the data noisy?** Partly. Between rounds 8 and 9 I re-submitted round 8's query for every function (`data/repeat_queries.csv`). Functions 2, 3 and 6 returned different values (0.624 vs 0.563, -0.0266 vs -0.0321, -0.540 vs -0.475). Functions 1, 4, 5, 7 and 8 returned bit-for-bit identical values. One repeat per function is thin evidence, and Function 2's round 13 result (0.436) was far from its round 10 result (0.765) at an almost identical input, which suggests its noise can be larger than the single repeat showed. Function 1 returns values indistinguishable from zero in every round (|y| below 1e-6). The model was fitted with a learned noise term for every function except 5, and the repeat was not used in training.

## Preprocessing

- Inputs are used as submitted. Outputs are transformed only inside the model (rank for F1, signed log for F4, log(1+y) for F5, none otherwise); stored values are raw.

## Uses

- **Intended use:** studying and reproducing this optimisation run; a teaching example of Bayesian optimisation under a tight query budget.
- **Not suitable for:** drawing conclusions about the true functions beyond the sampled regions, benchmarking optimisers (the sample is tiny and adaptively chosen), or any decision about people.

## Biases and limitations

- **Adaptive sampling bias.** Points were chosen by the optimiser, not at random, so they cluster near promising regions. Averages and correlations from them are not representative of the input space.
- **Small sample.** Thirteen submitted points per function, in up to 8 dimensions.
- **Single-run noise.** Almost every point was evaluated once, and Functions 2, 3 and 6 are noisy, so a lucky value is indistinguishable from a good point. The best results for those functions may be overstated.
- **Boundary points.** Several queries sit on or near the edge of the input range (for example Function 5, whose queries from round 4 on sit at or next to the corner 0.999999, and Functions 7 and 8 at 0 or 0.999999 on some inputs), where the model has little information.

## Distribution and maintenance

- **Where?** `data/` in this public repository, under the MIT licence.
- **Maintenance:** the capstone is complete and the dataset covers all 13 rounds. I do not plan further updates.
- **Contact:** via the GitHub repository.
