# IC-ML-AI-Capstone: Black-Box Optimisation Challenge

## Summary

This repository documents my Black-Box Optimisation (BBO) capstone project for the Imperial College London / Emeritus Artificial Intelligence and Machine Learning professional certificate. The challenge: find the input values that maximise eight hidden mathematical functions, using only a handful of real evaluations per round and no knowledge of the functions' internal formulas. Each round, a surrogate model is fitted to all data collected so far, and an acquisition function selects the single most promising next point to test. This repository tracks that strategy, round by round, across all eight functions, and documents the reasoning, trade-offs, and results along the way.

## Project Structure

```
IC-ML-AI-Capstone/
├── README.md                      <- you are here
├── Data/
│   ├── function_1/ ... function_8/
│   │   ├── initial_inputs.npy     <- starting data provided for the challenge
│   │   └── initial_outputs.npy
├── Code/
│   ├── week1_bbo.ipynb            <- one notebook per round, documenting that
│   ├── week2_bbo.ipynb               round's surrogate model, acquisition
│   ├── ...                           strategy, and reasoning
├── Documentation/
│   ├── Datasheet.md               <- data motivation, composition, collection,
│   │                                  preprocessing, distribution and maintenance
│   ├── Model_Card.md              <- approach, performance, assumptions,
│   │                                  limitations, trade-offs, ethics
```

> **Note on data:** The `.npy` files above are small starter files provided for this challenge. No large or sensitive datasets are stored in this repository.

## The Problem

Each of the 8 functions:
- Takes a different number of inputs (2D to 8D)
- Returns a single output value
- Is framed as a **maximisation** problem (even where the real-world analogy sounds like something you'd minimise, the function has already been transformed so that higher is always better)
- Is evaluated through a private portal — one query per function, per round, with results returned at the end of each module

The goal is to find the best input combination for each function within a limited query budget, applying Bayesian Optimisation: fit a surrogate model to observed data, use an acquisition function (e.g. Expected Improvement or Upper Confidence Bound) to balance exploiting known good regions against exploring uncertain ones, submit the chosen query, and repeat.

## Documentation

- **[Datasheet](Documentation/Datasheet.md)** — what the data is, how it was collected, what transformations were applied, and its known limitations.
- **[Model Card](Documentation/Model_Card.md)** — the modelling approach for each function, performance metrics used, assumptions, limitations, and trade-offs made along the way.

## Approach at a Glance

| Function | Dimensions | Surrogate model | Notes |
|---|---|---|---|
| 1 | 2D | *(to add)* | |
| 2 | 2D | *(to add)* | |
| 3 | 3D | *(to add)* | |
| 4 | 4D | *(to add)* | |
| 5 | 4D | *(to add)* | |
| 6 | 5D | *(to add)* | |
| 7 | 6D | *(to add)* | |
| 8 | 8D | *(to add)* | |

*(This table will be filled in and updated as the strategy for each function develops across rounds — see the Model Card for full detail once available.)*

## Results So Far

*(Best output found per function will be summarised here as rounds progress — see individual week notebooks in `/Code` and the Model Card's Performance section for details.)*

## How to Reproduce

1. Clone this repository.
2. Each `Code/weekN_bbo.ipynb` notebook can be run independently; it loads the accumulated data from `/Data`, fits that round's surrogate model(s), and outputs the next query for each function.
3. Required libraries: `numpy`, `scikit-learn`, and (for functions using a neural network surrogate) `torch`. See each notebook's imports for the exact set used that round.

## Status

This repository is actively updated as the capstone progresses through its query-submission rounds. Check the Model Card's version note and the most recent `Code/weekN_bbo.ipynb` file for the current state of the project.
