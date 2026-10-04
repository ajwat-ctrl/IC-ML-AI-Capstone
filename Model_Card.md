# Model Card: GP-Based Bayesian Optimisation for the BBO Capstone

*Following the "Model Cards for Model Reporting" framework (Mitchell et al., 2019) as introduced in Mini-lesson 21.2.*

## 1. Overview

| | |
|---|---|
| **Name** | GP-UCB/EI sequential optimiser |
| **Type** | Sequential model-based (Bayesian) optimisation, one strategy per function |
| **Version** | v1.0 (update this each time settings change; see Change Log) |
| **Code** | `Code/bbo_core.py`, `Code/run_round.py`, `Code/BBO_round_walkthrough.ipynb` |
| **Libraries** | NumPy, SciPy, scikit-learn, Matplotlib |

## 2. Intended Use

**Suitable for**
- Choosing one query per round for each of the eight capstone functions
- Small datasets (roughly 10 to 100 points) in 2 to 8 dimensions with inputs between 0 and 1
- Situations where each evaluation is expensive and the function's formula is unknown

**Not suitable for**
- Real-world decisions in the scenarios the functions are named after
- High-dimensional problems (well beyond 8 inputs) or large datasets, where exact Gaussian Processes become slow and their uncertainty less reliable
- Batch settings where many points are evaluated at once
- Problems with hard constraints beyond simple box bounds

## 3. How It Works

Each round, for each function:

1. **Data:** starter data plus every real portal result logged in `Data/submissions.json`.
2. **Optional output transform:** rank, signed log or log(1 + y), chosen to tame awkward output scales (see Datasheet, section 4).
3. **Surrogate:** a Gaussian Process with an anisotropic Matern kernel (one length scale per input, so the model can learn which inputs matter) plus a white-noise term. It predicts both a value and an uncertainty for any input.
4. **Candidates:** about 30,000 points, of which two thirds are spread uniformly across the space (exploration) and one third are small perturbations around the three best points so far (exploitation). Near-duplicates of existing points are removed.
5. **Acquisition:**
   - *UCB* = predicted value + kappa x uncertainty. A larger kappa means more exploration.
   - *EI* = expected amount by which a point beats the current best.
6. **Safety check:** the winning point must lie in [0, 0.999999] and is formatted to six decimals for the portal.
7. **Logging:** the query, settings, seed and diagnostics are saved to `Results/round_NN_queries.json`.

### Per-function settings (v1.0)

| Fn | Dims | Transform | Kernel | Acquisition | Why |
|---|---|---|---|---|---|
| 1 | 2 | rank | Matern 2.5 | UCB, kappa = 3.0 | Signal only near a hidden source and no source found yet, so exploration is the priority |
| 2 | 2 | none | Matern 1.5 | UCB, kappa = 2.0 | Described as noisy with local optima; a rougher kernel and a noise term suit this |
| 3 | 3 | none | Matern 2.5 | EI, xi = 0.01 | All outputs negative and the best is closest to zero; EI targets improving on the current best |
| 4 | 4 | signed log | Matern 2.5 | UCB, kappa = 2.0 | Many local optima and a wide output range |
| 5 | 4 | log(1 + y) | Matern 2.5, no noise | UCB, kappa = 1.5 | Described as unimodal, so climbing towards the single peak is the priority |
| 6 | 5 | none | Matern 2.5 | EI, xi = 0.01 | Sum of penalties, all negative; push towards zero |
| 7 | 6 | none | Matern 2.5 | UCB, kappa = 2.5 | Higher dimension, so stronger early exploration |
| 8 | 8 | none | Matern 2.5 | UCB, kappa = 1.5, smaller local step | Fairly flat outputs; a strong local maximum is the realistic goal |

### Change Log

| Round | Change | Reason |
|---|---|---|
| 1 | Initial settings above | Based on the official descriptions and starter data |
| *2* | *(record each change here)* | *(what the results showed)* |

## 4. Performance

**Metrics**
- **Best output so far** per function, tracked each round (`Results/progress.png`)
- **Leave-one-out (LOO) mean absolute error**: refit without each point and predict it
- **LOO calibration ratio**: LOO error divided by the GP's average predicted uncertainty. Near 1 means the uncertainty is roughly honest; well above 1 means overconfident.

**Baseline on the starter data (before any of my queries)**

| Fn | Best starter output | LOO MAE* | Calibration ratio |
|---|---|---|---|
| 1 | about 0 | 0.335 | 1.23 |
| 2 | 0.611 | 0.140 | 1.20 |
| 3 | -0.0348 | 0.058 | 1.26 |
| 4 | -4.03 | 0.155 | 0.77 |
| 5 | 1,088.9 | 1.117 | 1.26 |
| 6 | -0.714 | 0.314 | 1.33 |
| 7 | 1.365 | 0.227 | 1.09 |
| 8 | 9.598 | 0.078 | 0.62 |

*Errors are on the transformed scale for Functions 1, 4 and 5, so they are not comparable across functions.

Reading these: most functions sit between 1.1 and 1.3, meaning the GP is slightly overconfident, which argues for keeping some exploration. Functions 4 and 8 are below 1, meaning the GP is somewhat underconfident there.

**Results by round:** *(fill in as rounds come back)*

| Fn | Best after round 1 | Best so far | Round achieved |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |
| 6 | | | |
| 7 | | | |
| 8 | | | |

## 5. Assumptions and Limitations

**Assumptions**
- Each function is reasonably smooth: nearby inputs give similar outputs.
- One kernel shape fits the whole space for each function (stationarity).
- The valid input range is [0, 0.999999], as required by the portal format.
- Outputs are deterministic apart from a small noise term (stronger noise expected for Function 2).
- Leave-one-out results on 10 to 40 points are good enough to compare settings, though not precise.

**Limitations and failure modes**
- **Narrow peaks:** a sharp spike between sampled points (likely for Function 1) can look flat to the GP.
- **Sampling bias:** later queries cluster near the best points so far, leaving edges and corners of 6D and 8D spaces almost unexplored. A better optimum there would go unseen.
- **Deceptive landscapes:** if the best observed region is only a local peak, exploitation will refine the wrong area.
- **Small-sample diagnostics:** LOO numbers shift noticeably when a single point is added.
- **No true held-out test:** each round gives one real result per function, so settings can't be properly A/B tested.
- **Random candidate search:** in 8D, 30,000 random candidates are still sparse, so the true maximiser of the acquisition function may be missed.

## 6. Ethical Considerations

The functions are synthetic, so there is no direct risk of physical, financial or personal harm. Transparency still matters:

- **Reproducibility:** every round saves its query, settings, random seed and diagnostics, and all data is in the repository, so anyone can rerun `run_round.py` and get the same queries.
- **Honest scope:** the scenario names (drug discovery, chemical yield) are illustrative. Anyone reusing this approach on a real problem in those areas would need real measurement noise, safety constraints and domain review, none of which this approach handles.
- **Accurate reporting:** only real portal outputs are logged as results. Predicted values are kept separate in the diagnostics.

## 7. Is This Card Detailed Enough?

The current structure covers what someone needs to understand, rerun and critique the approach: how it works, settings per function with reasons, metrics, and failure modes. The main additions as the project runs will be the Change Log and the results table. Mathematical derivations of the kernels and acquisition functions are deliberately left to the code comments and the notebook, to keep this card readable for non-specialists.
