"""Command-line interface.

  python cli.py suggest --round 7            # propose a query for every function
  python cli.py suggest --round 7 --function 3
  python cli.py log --function 3 --round 7 --query 0.412345-0.55 --y 1.234 \
                    --rationale "exploit near best, kappa=0.5"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bbo import append_observation, format_query, load_observations, suggest

ROOT = Path(__file__).parent
DEFAULT_DATA = ROOT / "data" / "observations.csv"
DEFAULT_CONFIG = ROOT / "config.json"


def load_config(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"default": {"strategy": "ucb", "kappa": 2.0}, "functions": {}}


def cmd_suggest(args) -> None:
    obs = load_observations(args.data)
    cfg = load_config(Path(args.config))
    funcs = [args.function] if args.function else sorted(obs)
    print(f"Round {args.round} proposals (strategy/kappa from {Path(args.config).name})\n")
    for f in funcs:
        if f not in obs:
            print(f"function {f}: no data yet, skipping")
            continue
        X, y = obs[f]
        p = {**cfg["default"], **cfg.get("functions", {}).get(str(f), {})}
        x, info = suggest(X, y, strategy=p["strategy"], kappa=p["kappa"],
                          local_scale=p.get("local_scale", 0.08), seed=args.seed)
        print(f"function {f} [{p['strategy']}, kappa={p['kappa']}]  n={info['n_obs']}  "
              f"best={info['best_so_far']:.4f}")
        print(f"  query : {format_query(x)}")
        print(f"  pred  : {info['pred_mean']:.4f} +/- {info['pred_std']:.4f}   "
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
