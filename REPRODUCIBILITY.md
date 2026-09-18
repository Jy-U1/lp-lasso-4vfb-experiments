# Reproducibility notes

## What is deterministic

Problem data, initialization directions, and perturbations use NumPy's
`default_rng` with seeds fixed in `experiments/common.py`. Paired methods receive
identical data and starting points. The scalar proximal map uses a deterministic
zero-at-threshold convention and safeguarded Newton/bisection. The active
smallest eigenpair is computed by SciPy's dense symmetric eigensolver.

Terminal objectives, residuals, supports, classifications, correction counts,
and iteration counts should reproduce to the displayed precision on standard
binary64 NumPy/SciPy builds. Eigenvectors are sign-indeterminate; the verifier
evaluates both signs, so this does not change the selected correction.

## What is machine dependent

Wall-clock time and its decomposition depend on CPU frequency, BLAS/LAPACK,
thread count, background load, and Python/NumPy/SciPy versions. The manuscript's
timing claims refer to the frozen run in
`results/reference_20260907T164315Z`. A new run should be interpreted as a new
timing measurement, not required to match those seconds.

For formal timing replacement, use one environment, warm the implementation,
run each paired method three times, report medians, preserve the generated
environment JSON, and update both the manuscript and audit expectations.

## Frozen and generated outputs

Raw CSV files are the authoritative output. Derived tables and plots can be
regenerated. The preserved legacy archive retains its original checksum file.
The scale-validation directory has its own protocol and environment metadata.
Every new unified run creates `metadata/environment.json`, `metadata/profile.json`,
`metadata/run_summary.json`, and `checksums_sha256.csv`.

The `smoke` profile is a software test only. It intentionally changes dimensions,
seeds, repetition counts, and horizon length. Only the `paper` profile represents
the full experimental design.

