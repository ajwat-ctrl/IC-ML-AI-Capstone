# IC-ML-AI-Capstone: Black-Box Optimisation (BBO)

Imperial College ML & AI capstone. The task: maximise eight unknown black-box
functions (2D to 8D, inputs in [0, 1)) using one query per function per round.

> **TODO before publishing:** this README must describe what you actually ran.
> Edit the Method and Results sections to match your real rounds, and delete
> anything that is not true of your submissions.

## Method

For each function the pipeline:

1. Fits a Gaussian process (scikit-learn) with an ARD Matern-5/2 kernel and a
   learned noise term, on all observations so far.
2. Scores 20,000 candidate points (half sampled near the current best, half
   uniform over the domain) with an acquisition function, either
   Upper Confidence Bound (`mu + kappa * sigma`) or Expected Improvement.
3. Returns the highest-scoring candidate that is not a near-duplicate of an
   existing query, formatted to six decimals.

Strategy and `kappa` are set per function in `config.json`, so changes to the
exploration/exploitation balance are visible in the git history.

## Repository layout

| Path | Purpose |
|------|---------|
| `bbo.py` | GP fitting, acquisition functions, query formatting, CSV helpers |
| `cli.py` | `suggest` a query per function, `log` a result |
| `config.json` | Per-function acquisition strategy and exploration weight |
| `data/observations.csv` | Every query and the value it returned (long format) |
| `requirements.txt` | Dependencies |

## Data format

`data/observations.csv` has one row per query:

```
function,round,query,y,rationale
3,1,0.412345-0.550000,1.234,"initial space-filling"
```

`query` is hyphen-separated with six decimals, matching the portal format.

## Usage

```bash
pip install -r requirements.txt
python cli.py suggest --round 7                 # one proposal per function
python cli.py suggest --round 7 --function 3
python cli.py log --function 3 --round 7 --query 0.412345-0.550000 --y 1.234 \
                  --rationale "exploit near best, kappa=0.5"
```

## Results

| Function | Dim | Rounds | Best value | Round of best |
|----------|-----|--------|------------|---------------|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |
| 6 | | | | |
| 7 | | | | |
| 8 | | | | |

## Reflections

Add links to your discussion-board reflections here.

## Licence

MIT, see `LICENSE`.
