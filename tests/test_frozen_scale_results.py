import json
from pathlib import Path


def test_frozen_scale_summary_matches_manuscript() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "results" / "scale_validation" / "metadata" / "manuscript_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["objective_rescaling"]["homogeneous_nlm"] == 42
    assert summary["objective_rescaling"]["absolute_smallest_scale_corrected"] == 0
    assert summary["correlated_zero_init"]["four_v_nlm"] == 45
    assert summary["data_perturbations"]["four_v_nlm"] == 180
    assert summary["data_perturbations"]["fb_nlm"] == 47
    assert summary["eligibility_pilot"]["corrected"] == 0

