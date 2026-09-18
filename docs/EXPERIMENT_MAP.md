# Manuscript-to-code experiment map

| Manuscript objective | Code entry | Frozen result files | Main diagnostic |
|---|---|---|---|
| Ordinary instances and measured cost | `demos/demo_ordinary.py` / `experiments/ordinary.py` | `reference_20260907T164315Z/raw/ordinary_*`, `schedule_ablation_*` | matched FB endpoints, calls, runtime, eig/probe time |
| Exact strict and quartic fixed points | `demos/demo_exact_saddles.py` / `experiments/exact_saddles.py` | `reference_20260907T164315Z/raw/exact_*` | stationarity, curvature, accepted correction, terminal NLM |
| Perturbed saddles and competitors | `demos/demo_perturbed_saddles.py` / `experiments/perturbed_saddles.py` | `reference_20260907T164315Z/raw/perturbed_*` | escape, NLM, stop/cap, terminal objective |
| Long-horizon FB control | `demos/demo_long_horizon.py` / `experiments/long_horizon.py` | `reference_20260907T164315Z/raw/long_horizon_fb.csv` | machine fixed iteration, distance, high-precision objective change |
| Objective rescaling | `demos/demo_scale_validation.py` / `experiments/scale_validation.py` | `scale_validation/raw/objective_rescaling_*` | normalized endpoint spread and absolute-guard control |
| Correlated zero initialization | same scale demo/module | `scale_validation/raw/correlated_zero_init_*` | NLM and correction counts across \(b/b_c\) |
| Perturbed \(A,y\) | same scale demo/module | `scale_validation/raw/data_perturbation_*` | paired FB/4V-FB NLM counts and iteration ranges |

## Profile definitions

All grids and seeds live in
`src/lp_lasso_4vfb/experiments/common.py` as immutable `ExperimentProfile`
objects. `PAPER_PROFILE` is the manuscript protocol; `SMOKE_PROFILE` is only a
fast implementation check and must not be used for paper tables.

The ordinary grid uses \(n=50,100,200,400\), \(p=0.3,0.5,0.8\), five paired
seeds, \(m=0.8n\), and \(\lambda=0.05\). Constructed cases use \(n=20,50\),
the same three \(p\) values, and \(\lambda=0.1\). The perturbed-data grid uses
three correlations, three perturbation scales, and 20 paired directions per
correlation.

## Result provenance

- `results/reference_20260907T164315Z` is the preserved output archive used for
  the legacy-protocol tables and figures. Its own metadata and SHA-256 manifest
  are retained unchanged.
- `results/scale_validation` was regenerated from the modular implementation
  using the scale-equivariant protocol. Its metadata records the environment
  and complete profile.
- `scripts/verify_manuscript_results.py` audits both result trees together.

Raw CSV files are the source of truth. LaTeX and figure assets are derived
outputs and can be rebuilt by `scripts/build_paper_assets.py`.

