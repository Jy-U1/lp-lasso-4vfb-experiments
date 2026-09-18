# 4V-FB experiments for nonconvex \(\ell_p\)-Lasso

This repository contains the modular code and frozen numerical results used by
the manuscript. It is designed to run unchanged in Google Colab or in a local
Python environment (including VS Code).

The experiments are optimization-geometry stress tests, not statistical
recovery benchmarks. They test whether sparse fourth-order verification leaves
ordinary FB trajectories unchanged, rejects constructed non-minimizing fixed
points, remains useful near degenerate geometry, and behaves consistently
under objective rescaling and data perturbations.

## Repository layout

| Path | Purpose |
|---|---|
| `src/lp_lasso_4vfb/algorithms/four_v_fb.py` | Main 4V-FB algorithm |
| `src/lp_lasso_4vfb/algorithms/fb.py` | Plain FB baseline |
| `src/lp_lasso_4vfb/algorithms/dirl.py` | DIRL1 and DIRL2 baselines |
| `src/lp_lasso_4vfb/prox.py` | Safeguarded scalar/vector \(\ell_p\) proximal map |
| `src/lp_lasso_4vfb/problems.py` | Random, exact-saddle, correlated, and perturbed data generators |
| `src/lp_lasso_4vfb/experiments/` | One module per experiment family |
| `demos/` | Five independently runnable demos |
| `run_all.py` | Unified entry point linking all demos |
| `notebooks/colab_runner.ipynb` | Thin Colab front end for this same package |
| `results/reference_20260907T164315Z/` | Frozen legacy-protocol results used in the manuscript |
| `results/scale_validation/` | Frozen scale-equivariant validation results |
| `paper_assets/` | Regenerated LaTeX tables and PDF/PNG figures |
| `scripts/verify_manuscript_results.py` | Automatic data-to-manuscript audit |

The source is intentionally separated from experiment orchestration. Algorithm
files contain no paper-specific grids, while each experiment module receives a
typed profile and writes raw CSV logs plus metadata.

## Install locally (VS Code or terminal)

Python 3.10 or newer is required. From the repository root:

```bash
python -m venv .venv
```

Activate the environment, then install:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`; on macOS or
Linux, use `source .venv/bin/activate`.

Run the fast validation profile:

```bash
python run_all.py --profile smoke --experiments all
```

Run the full manuscript grids:

```bash
python run_all.py --profile paper --experiments all
```

The full command includes the one-million-step control and all 180 paired data
perturbations. It can take several minutes or longer depending on the machine.
New output is written under `results/runs/` and is ignored by Git by default.

## Run one experiment

```bash
python demos/demo_ordinary.py --profile paper
python demos/demo_exact_saddles.py --profile paper
python demos/demo_perturbed_saddles.py --profile paper
python demos/demo_long_horizon.py --profile paper
python demos/demo_scale_validation.py --profile paper
```

The same selection is available through the unified runner, for example:

```bash
python run_all.py --profile paper --experiments exact,perturbed
```

## Google Colab

Open `notebooks/colab_runner.ipynb`, replace `REPO_URL` with the GitHub URL of
your repository, and run the cells from top to bottom. The notebook clones and
installs the package, runs the smoke profile first, and exposes a separate full
paper command. It does not duplicate algorithm code, so Colab and local runs
use exactly the same implementation.

## Verify the manuscript numbers

```bash
python scripts/verify_manuscript_results.py
```

This checks the frozen CSV files against the counts, ranges, timings, residuals,
eigenvalue diagnostics, long-horizon values, and scale-validation table stated
in the current manuscript. The machine-readable report is written to
`audit/manuscript_result_audit.json`. Timing values are properties of the
frozen reference run; rerun times are expected to vary across hardware.

Run numerical unit tests with:

```bash
python -m pytest -q
```

Regenerate the Overleaf-ready assets with:

```bash
python scripts/build_paper_assets.py
```

For a detailed mapping from manuscript claims to code and result files, see
[`docs/EXPERIMENT_MAP.md`](docs/EXPERIMENT_MAP.md). GitHub publication steps
are in [`GITHUB_UPLOAD_GUIDE_CN.md`](GITHUB_UPLOAD_GUIDE_CN.md).

## Two numerical protocols

The current manuscript reports both an archived protocol and a later
scale-equivariant validation. The distinction is explicit in the code:

- `legacy_solver` / `legacy_four_v`: raw FB residual, eligibility \(10^{-7}\),
  additive absolute comparison guard, stopping tolerance \(10^{-11}\).
- `scale_solver` / `scale_four_v`: residual normalized by the proximal output
  gap, eligibility \(10^{-4}\), homogeneous relative comparison guard,
  stopping tolerance \(10^{-11}\).

Do not merge their thresholds when reproducing the tables. Both protocols use
the same zero-at-threshold proximal convention, powers-of-two verification
schedule, and forced final verification.

