"""Unified experiment dispatcher used by the CLI, demos, and Colab notebook."""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Callable, Iterable

from .experiments import (
    run_exact_saddles,
    run_long_horizon,
    run_ordinary,
    run_perturbed_saddles,
    run_scale_validation,
)
from .experiments.common import ExperimentProfile, get_profile
from .reporting import OutputStore


Experiment = Callable[[ExperimentProfile, Path], dict[str, Any]]

EXPERIMENTS: dict[str, Experiment] = {
    "ordinary": run_ordinary,
    "exact": run_exact_saddles,
    "perturbed": run_perturbed_saddles,
    "long": run_long_horizon,
    "scale": run_scale_validation,
}


def parse_experiment_names(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        names = [item.strip() for item in value.split(",") if item.strip()]
    else:
        names = [str(item).strip() for item in value if str(item).strip()]
    if names == ["all"]:
        return list(EXPERIMENTS)
    unknown = sorted(set(names) - set(EXPERIMENTS))
    if unknown:
        raise ValueError(f"unknown experiments: {', '.join(unknown)}")
    if not names:
        raise ValueError("select at least one experiment")
    return names


def default_run_id(profile: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{profile}_{stamp}"


def run_selected(
    *, profile_name: str, experiments: str | Iterable[str], output: Path
) -> dict[str, Any]:
    """Run selected experiment modules and write one auditable result tree."""
    profile = get_profile(profile_name)
    names = parse_experiment_names(experiments)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    root_store = OutputStore(output)
    root_store.write_environment(output.name, profile_name)
    root_store.json("metadata", "profile", asdict(profile))
    summary: dict[str, Any] = {
        "profile": profile_name,
        "experiments": names,
        "output": output.as_posix(),
        "results": {},
    }
    for name in names:
        print(f"[run] {name}", flush=True)
        summary["results"][name] = EXPERIMENTS[name](profile, output / name)
        print(f"[done] {name}", flush=True)
    (output / "metadata" / "run_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    root_store.checksums()
    return summary
