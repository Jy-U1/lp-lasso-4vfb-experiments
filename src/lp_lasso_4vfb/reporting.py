"""Stable CSV/JSON export and paper-oriented run summaries."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import FourVFBConfig, SolverConfig
from .model import LpLassoProblem, objective
from .results import RunResult


def summarize_run(
    problem: LpLassoProblem,
    result: RunResult,
    run_key: str,
    *,
    initial_objective: float | None = None,
    **context: Any,
) -> dict[str, Any]:
    log = result.verification_log
    row: dict[str, Any] = {
        "run_key": run_key,
        "problem": problem.name,
        "method": result.method,
        "status": result.status,
        "m": problem.m,
        "n": problem.n,
        "p": problem.p,
        "lambda": problem.lam,
        "initial_objective": (
            objective(problem, result.x) if initial_objective is None else initial_objective
        ),
        "iterations": result.iterations,
        "final_objective": result.final_objective,
        "final_residual": result.final_residual,
        "final_raw_residual": result.final_raw_residual,
        "final_normalized_residual": result.final_normalized_residual,
        "support_size": int(result.final_stats["support_size"]),
        "raw_nonzero_count": result.raw_nonzero_count,
        "classification_support_threshold": result.classification_support_threshold,
        "thresholded_tail_norm": result.thresholded_tail_norm,
        "lambda_min": result.final_stats["lambda_min"],
        "relative_lambda_min": result.final_stats["relative_lambda_min"],
        "numerical_locmin": result.passes_nlm(),
        "runtime_total": result.runtime,
        "runtime_algorithm": result.algorithm_runtime,
        "diagnostic_time": result.diagnostic_time,
        "verification_calls": result.verification_calls,
        "scheduled_verifications": result.scheduled_verifications,
        "forced_final_verifications": result.forced_final_verifications,
        "accepted_corrections": result.accepted_corrections,
        "eligible_iterations": result.eligible_iterations,
        "mean_verify_support": float(log.support_size.mean()) if not log.empty else np.nan,
        "max_verify_support": float(log.support_size.max()) if not log.empty else np.nan,
        "eig_time": result.eig_time,
        "probe_time": result.probe_time,
        "verification_time": result.verification_time,
        "verification_time_fraction": (
            result.verification_time / result.runtime if result.runtime > 0.0 else np.nan
        ),
        "gamma": result.gamma,
        "kappa0": result.kappa0,
        "sigma": result.sigma,
        "eta0": result.eta0,
        "guard_mode": result.metadata.get("guard_mode", ""),
        "residual_mode": result.metadata.get("residual_mode", ""),
    }
    if (
        initial_objective is not None
        and np.isfinite(result.sigma)
        and result.sigma > 0.0
        and np.isfinite(result.eta0)
        and np.isfinite(result.prox_gap)
    ):
        eta_lower = min(result.eta0, 0.5 * result.prox_gap)
        row["log10_theoretical_correction_bound"] = (
            math.log10(initial_objective)
            - math.log10(result.sigma)
            - 4.0 * math.log10(eta_lower)
        )
    else:
        row["log10_theoretical_correction_bound"] = np.nan
    row.update(context)
    return row


def attach_verification_context(
    result: RunResult, run_key: str, **context: Any
) -> pd.DataFrame:
    if result.verification_log.empty:
        return pd.DataFrame()
    frame = result.verification_log.copy()
    frame.insert(0, "run_key", run_key)
    frame.insert(1, "method", result.method)
    for key, value in context.items():
        frame[key] = value
    return frame


class OutputStore:
    def __init__(self, root: Path):
        self.root = root
        self.raw = root / "raw"
        self.tables = root / "tables"
        self.figures = root / "figures"
        self.metadata = root / "metadata"
        for directory in (self.raw, self.tables, self.figures, self.metadata):
            directory.mkdir(parents=True, exist_ok=True)

    def csv(self, folder: str, name: str, frame: pd.DataFrame) -> Path:
        destination = getattr(self, folder) / f"{name}.csv"
        frame.to_csv(destination, index=False)
        return destination

    def json(self, folder: str, name: str, value: Any) -> Path:
        destination = getattr(self, folder) / f"{name}.json"
        destination.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
        return destination

    def write_environment(self, run_id: str, profile: str) -> None:
        import matplotlib
        import scipy

        self.json(
            "metadata",
            "environment",
            {
                "run_id": run_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "profile": profile,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "pandas": pd.__version__,
                "matplotlib": matplotlib.__version__,
                "platform": platform.platform(),
                "processor": platform.processor(),
                "cpu_count": __import__("os").cpu_count(),
            },
        )

    def checksums(self) -> Path:
        rows = []
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and path.name != "checksums_sha256.csv":
                rows.append(
                    {
                        "relative_path": path.relative_to(self.root).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
        return self.csv("root", "checksums_sha256", pd.DataFrame(rows))

    @property
    def root_path(self) -> Path:
        return self.root

    def __getattr__(self, name: str) -> Any:
        if name == "root":
            raise AttributeError(name)
        raise AttributeError(name)


def config_record(solver: SolverConfig, four_v: FourVFBConfig) -> dict[str, Any]:
    return {"solver": asdict(solver), "four_v_fb": asdict(four_v)}

