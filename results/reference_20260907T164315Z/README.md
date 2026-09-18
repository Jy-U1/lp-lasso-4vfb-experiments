# 4V-FB experiment archive

Run ID: 20260907T164315Z
Run mode: paper

Folders:
- figures/: manuscript-ready PNG and PDF figures.
- tables/: compact CSV tables and matching LaTeX tabular files.
- raw/: per-run summaries, verification attempts, histories, and long-horizon checkpoints.
- metadata/: frozen configurations, exact seeds, environment, and baseline citation.
- RESULTS_SUMMARY.md: automatically generated headline findings and capped-run count.

Interpretation:
The experiments test terminal-state verification, not global optimality or recovery quality.
The theoretical correction-count bound is qualitative; measured N_ver, N_corr, eigensolver
time, probe time, and wall time are reported directly.
Runtime ratios use paired medians and IQRs.  Long-horizon objective differences are
re-evaluated at high precision; the figure never replaces numerical zeros by an
artificial plotting floor.
