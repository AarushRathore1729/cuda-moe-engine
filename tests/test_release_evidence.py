"""Reject damage to the A100 forward evidence that supports release claims."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verify_release", ROOT / "scripts/verify_release.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def recorded():
    return json.loads((ROOT / "benchmarks/a100-block/block_results.json").read_text())


def test_complete_release_evidence():
    summary = verifier.verify()
    assert summary["quick"]["forward_comparisons_passed"] == 24
    assert summary["quick"]["median_forward_ratio"] == pytest.approx(3.1677670384043486)
    assert summary["quick"]["source_hashes_verified"] == 22
    assert summary["profiler_gpu_kernel_events"] > 0
    assert summary["optional"]["graphs"]["graph_replay_checks_passed"] == 24
    assert summary["optional"]["model"]["forward_comparisons_passed"] == 6


@pytest.mark.parametrize("damage", ["missing_status", "missing_timing", "failed_status"])
def test_graph_replay_evidence_must_match_timings(damage):
    data = json.loads((ROOT / "benchmarks/a100-block/graphs/block_results.json").read_text())
    if damage == "missing_status":
        data["validation"][0].pop("graph_status")
    elif damage == "failed_status":
        data["validation"][0]["graph_status"] = "unsupported"
        data["validation"][0]["graph_error"] = "capture unavailable"
    else:
        for index, row in enumerate(data["benchmarks"]):
            if row["stage"] == "forward_graph":
                data["benchmarks"].pop(index)
                break
    with pytest.raises(ValueError):
        verifier.validate_block(data)


@pytest.mark.parametrize("damage", ["median", "p95", "minimum", "nonfinite", "samples",
                                    "duplicate", "missing_stage", "failed_validation",
                                    "missing_validation", "tokens", "source", "backend"])
def test_damaged_forward_evidence_is_rejected(damage):
    data = deepcopy(recorded())
    row = data["benchmarks"][0]
    if damage == "median":
        row["median_us"] *= 2
    elif damage == "p95":
        row["p95_us"] *= 2
    elif damage == "minimum":
        row["min_us"] *= 2
    elif damage == "nonfinite":
        row["samples_us"][0] = float("nan")
    elif damage == "samples":
        row["samples_us"].pop()
    elif damage == "duplicate":
        data["benchmarks"][1] = deepcopy(row)
    elif damage == "missing_stage":
        data["benchmarks"].pop()
    elif damage == "failed_validation":
        data["validation"][0]["passed"] = False
    elif damage == "missing_validation":
        data["validation"].pop()
    elif damage == "tokens":
        data["configurations"][0]["expert_counts"][0] += 1
    elif damage == "source":
        data["source_hashes"]["dispatch.cu"] = "0" * 64
    else:
        data["configurations"][0]["expert_backend"] = "other"
    with pytest.raises(ValueError):
        verifier.validate_block(data)
