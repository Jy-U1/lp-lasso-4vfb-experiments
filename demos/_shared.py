"""Small CLI adapter shared by the standalone demos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lp_lasso_4vfb.runner import default_run_id, run_selected


def run_demo(experiment: str, description: str) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--profile", choices=("smoke", "paper"), default="smoke")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or Path("results") / "runs" / default_run_id(
        f"{args.profile}_{experiment}"
    )
    summary = run_selected(
        profile_name=args.profile, experiments=[experiment], output=output
    )
    print(json.dumps(summary, indent=2, default=str))

