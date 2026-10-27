"""Command-line interface.

  python cli.py suggest --round 5            # propose a query for every function
  python cli.py suggest --round 5 --function 3
  python cli.py log --function 3 --round 4 --query 0.412345-0.55-0.3 --y -0.04 \
                    --rationale "EI xi=0.01"

Note: `suggest` re-creates the pipeline from config.json using only the points in
data/observations.csv. The original submissions were also fitted on the initial
data provided by the capstone (10 to 40 points per function), which is not in
this repo, so proposals will not match the submitted queries exactly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bbo import append_observation, format_query, load_observations, suggest

ROOT = Path(__file__).parent
DEFAULT_DATA = ROOT / "data" / "observations.csv"
DEFAULT_CONFIG = ROOT / "config.json"
FALLBACK = {"strategy": "ucb", "kappa": 2.0, "xi": 0.0, "transform": "none",
            "nu": 2.5, "noise": True, "local_scale": 0.05}


def load_config(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"default": FALLBACK, "functions": {}}


def cmd_suggest(args) -> None:
    obs = load_observations(args.data)
    cfg = load_config(Path(args.config))
    funcs = [args.function] if args.function else sorted(obs)
    print(f"Round {args.round} proposals (settings from {Path(args.config).name})\n")
    for f in funcs:
        if f not in obs:
            print(f"function {f}: no data yet, skipping")
            continue
        X, y = obs[f]
        p = {**FALLBACK, **cfg.get("default", {}), **cfg.get("functions", {}).get(str(f), {})}
        x, info = suggest(X, y, strategy=p["strategy"], kappa=p["kappa"], xi=p["xi"],
                          transform=p["transform"], nu=p["nu"], noise=p["noise"],
                          local_scale=p["local_scale"], seed=args.seed)
        print(f"function {f} [{p['strategy']}, kappa={p['kappa']}, xi={p['xi']}, "
              f"transform={p['transform']}, nu={p['nu']}, noise={p['noise']}]  "
              f"n={info['n_obs']}  best={info['best_so_far']:.4g}")
        print(f"  query : {format_query(x)}")
        print(f"  pred  : {info['pred_mean_transformed']:.4f} +/- "
              f"{info['pred_std_transformed']:.4f} (transformed units)   "
              f"lengthscales: {info['lengthscales']}\n")


def cmd_log(args) -> None:
    append_observation(args.data, args.function, args.round, args.query, args.y, args.rationale)
    print(f"logged function {args.function}, round {args.round}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("suggest")
    s.add_argument("--round", type=int, required=True)
    s.add_argument("--function", type=int)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_suggest)

    l = sub.add_parser("log")
    l.add_argument("--function", type=int, required=True)
    l.add_argument("--round", type=int, required=True)
    l.add_argument("--query", required=True)
    l.add_argument("--y", type=float, required=True)
    l.add_argument("--rationale", default="")
    l.set_defaults(fn=cmd_log)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
