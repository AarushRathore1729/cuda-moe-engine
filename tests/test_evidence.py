from copy import deepcopy
import json
from pathlib import Path
import pytest
from moe_dispatch.evidence import read_results, summarize, check_manifest

RECORDED = Path(__file__).resolve().parents[1] / "benchmarks/a100"


def test_recorded_a100_source_and_samples_are_intact():
    check_manifest(RECORDED)
    result = read_results(RECORDED / "results.json")
    stats = summarize(result)
    assert stats["configurations"] == 12
    assert stats["correctness_checks"] == 36
    assert stats["stages"]["dispatch+combine"]["median_per_configuration_speedup"] == pytest.approx(
        6.3632243524850445
    )


@pytest.mark.parametrize("mutation", ["median", "count", "duplicate", "shape"])
def test_corrupted_evidence_is_rejected(tmp_path, mutation):
    data = deepcopy(json.loads((RECORDED / "results.json").read_text()))
    if mutation == "median":
        data["benchmarks"][0]["median_us"] *= 2
    elif mutation == "count":
        data["configurations"][0]["expert_counts"][0] += 1
    elif mutation == "duplicate":
        data["benchmarks"][1] = deepcopy(data["benchmarks"][0])
    else:
        data["benchmarks"][0]["hidden"] += 1
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        read_results(path)
