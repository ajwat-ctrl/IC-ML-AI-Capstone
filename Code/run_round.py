"""
run_round.py
Produce one portal query per function for the next round.

Usage (from the repo root):
    python Code/run_round.py --round 1
    python Code/run_round.py --round 2 --loo      # also run leave-one-out checks (slower)

Before each round after the first, record the previous round's real portal results in
Data/submissions.json, then run this script again.
"""

import argparse
import json
import os
from datetime import date

from bbo_core import REPO_ROOT, load_function_data, load_submissions, propose_query, format_query

# ---------------------------------------------------------------------------
# Per-function settings. Each choice is justified from the official function
# description and what the starter data shows. Update these as your rounds teach
# you more, and record every change in Documentation/Model_Card.md.
# ---------------------------------------------------------------------------
CONFIGS = {
    # F1 2D contamination detection: almost every starter output is ~0 (signal only
    # near a hidden source). Rank transform stops near-zero noise dominating the fit;
    # high kappa because the source has not been found yet, so exploration matters.
    1: dict(transform="rank", nu=2.5, noise=True, acquisition="ucb", kappa=3.0, xi=0.0, local_scale=0.05),
    # F2 2D noisy log-likelihood with local optima: rougher kernel (nu=1.5) and a
    # noise term; moderate exploration to avoid getting stuck on one local peak.
    2: dict(transform="none", nu=1.5, noise=True, acquisition="ucb", kappa=2.0, xi=0.0, local_scale=0.05),
    # F3 3D drug discovery (negative side effects, all outputs < 0): best is closest
    # to zero. EI focuses on beating the current best.
    3: dict(transform="none", nu=2.5, noise=True, acquisition="ei", kappa=0.0, xi=0.01, local_scale=0.05),
    # F4 4D warehouse model, dynamic with many local optima; outputs span -33 to -4.
    # Signed log compresses the range; balanced UCB.
    4: dict(transform="signedlog", nu=2.5, noise=True, acquisition="ucb", kappa=2.0, xi=0.0, local_scale=0.05),
    # F5 4D chemical yield, described as unimodal; outputs span 0.1 to ~1,089, so log1p.
    # Lower kappa: with one peak, climbing towards it is the priority.
    5: dict(transform="log1p", nu=2.5, noise=False, acquisition="ucb", kappa=1.5, xi=0.0, local_scale=0.05),
    # F6 5D recipe score, all outputs negative (sum of penalties): EI towards zero.
    6: dict(transform="none", nu=2.5, noise=True, acquisition="ei", kappa=0.0, xi=0.01, local_scale=0.05),
    # F7 6D ML hyperparameters: higher dimension, so stronger exploration early on.
    7: dict(transform="none", nu=2.5, noise=True, acquisition="ucb", kappa=2.5, xi=0.0, local_scale=0.05),
    # F8 8D black box, outputs fairly flat (5.6 to 9.6): a strong local maximum is the
    # realistic target, so search mostly near the best points with a smaller step.
    8: dict(transform="none", nu=2.5, noise=True, acquisition="ucb", kappa=1.5, xi=0.0, local_scale=0.03),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True, help="round number you are preparing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loo", action="store_true", help="run leave-one-out diagnostics")
    args = parser.parse_args()

    submissions = load_submissions()
    results_dir = os.path.join(REPO_ROOT, "Results")
    os.makedirs(results_dir, exist_ok=True)

    log = {"round": args.round, "date": str(date.today()), "seed": args.seed, "functions": {}}
    print(f"=== Queries for round {args.round} ===\n")

    for fn in range(1, 9):
        X, y, n_init = load_function_data(fn, submissions=submissions)
        cfg = CONFIGS[fn]
        query, diag = propose_query(X, y, cfg, seed=args.seed + fn, run_loo=args.loo)
        q = format_query(query)
        print(f"Function {fn} ({X.shape[1]}D, {len(y)} points, best so far {diag['best_observed']:.6g})")
        print(f"  {q}\n")
        log["functions"][f"function_{fn}"] = {"query": q, "config": cfg, "diagnostics": diag}

    out = os.path.join(results_dir, f"round_{args.round:02d}_queries.json")
    with open(out, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Saved log (queries, settings, seed, diagnostics) to {out}")


if __name__ == "__main__":
    main()
