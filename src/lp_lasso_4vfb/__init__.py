"""Reproducible experiments for fourth-order verified forward--backward."""

from .config import DIRLConfig, FourVFBConfig, SolverConfig
from .model import LpLassoProblem

__all__ = ["DIRLConfig", "FourVFBConfig", "LpLassoProblem", "SolverConfig"]
__version__ = "1.0.0"

