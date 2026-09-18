#!/usr/bin/env python3
"""Run any subset of the manuscript experiments from one command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lp_lasso_4vfb.runner import default_run_id, run_selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "paper"), default="smoke")
    parser.add_argument(
        "--experiments",
        default="all",
        help="comma-separated subset: ordinary,exact,perturbed,long,scale (or all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output directory; default: results/runs/<profile>_<UTC timestamp>",
    )
    args = parser.parse_args()
    output = args.output or Path("results") / "runs" / default_run_id(args.profile)
    summary = run_selected(
        profile_name=args.profile, experiments=args.experiments, output=output
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

