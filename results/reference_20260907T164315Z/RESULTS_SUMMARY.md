# Numerical results summary

Run ID: 20260907T164315Z
Run mode: paper

- Ordinary paired runs: 60; accepted corrections: 0; capped 4V-FB runs: 0.
- Median paired runtime ratio across (n,p) cells: 1.052. Report the IQR columns rather than interpreting a single Colab timing.
- Schedule ablation: 925 every-eligible calls versus 11 sparse calls.
- Exact constructed saddles: 12/12 4V-FB runs reached numerical local minima.
- Perturbed quartic grid: 4V-FB escaped 120/120; FB/DIRL1/DIRL2 escaped 0/360.
- DIRL robustness configurations: 6 one-factor settings on the canonical quartic instance.
- Long horizon: 1,000,000 requested FB steps; first bitwise machine-fixed iteration: 10.

Interpretation: these experiments test terminal-state verification, not global optimization quality.  The theoretical correction-count bound is qualitative.
