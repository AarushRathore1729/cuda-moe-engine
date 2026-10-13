"""Inspect recorded raw samples and source integrity with the Python standard library."""

import hashlib
import json
import math
from pathlib import Path
import statistics


def read_results(path):
    result = json.loads(Path(path).read_text())
    configurations = result.get("configurations", [])
    rows = result.get("benchmarks", [])
    if not configurations or len(rows) != len(configurations) * 9:
        raise ValueError("expected three implementations and three stages per configuration")
    if not result.get("block_demo", {}).get("passed"):
        raise ValueError("MoE block validation is missing or failed")
    if result.get("correctness", {}).get("failed") != 0:
        raise ValueError("correctness failed or was not recorded")
    cases = {c["case_id"]: c for c in configurations}
    if len(cases) != len(configurations):
        raise ValueError("duplicate configuration IDs")
    seen = set()
    implementations = {"PyTorch reference", "CUDA scalar", "CUDA vectorized"}
    stages = {"dispatch", "combine", "dispatch+combine"}
    for row in rows:
        key = (row["case_id"], row["implementation"], row["stage"])
        if key in seen or row["case_id"] not in cases:
            raise ValueError("duplicate or unknown benchmark row")
        seen.add(key)
        if row["implementation"] not in implementations or row["stage"] not in stages:
            raise ValueError("unknown implementation/stage")
        case = cases[row["case_id"]]
        for field in ("tokens", "hidden", "experts", "topk", "pattern", "dtype"):
            if row[field] != case[field]:
                raise ValueError(f"row/configuration mismatch: {field}")
        samples = row["samples_us"]
        if len(samples) != result["method"]["repeats"] or not samples:
            raise ValueError("incorrect raw-sample count")
        if any(not math.isfinite(v) or v <= 0 for v in samples):
            raise ValueError("timings must be finite and positive")
        if not math.isclose(statistics.median(samples), row["median_us"], rel_tol=1e-9):
            raise ValueError("stored median does not match raw samples")
        ordered = sorted(samples)
        position = (len(ordered) - 1) * 0.95
        first, last = math.floor(position), math.ceil(position)
        expected_p95 = ordered[first] + (ordered[last] - ordered[first]) * (position - first)
        if not math.isclose(expected_p95, row["p95_us"], rel_tol=1e-9):
            raise ValueError("stored p95 does not match raw samples")
        if row["workspace_bytes"] <= 0:
            raise ValueError("workspace bytes must be positive")
    for case in configurations:
        counts = case["expert_counts"]
        if len(counts) != case["experts"] or any(c < 0 for c in counts):
            raise ValueError("incorrect expert-count vector")
        if sum(counts) != case["tokens"] * case["topk"]:
            raise ValueError("expert counts do not conserve routed tokens")
    return result


def summarize(result):
    by_key = {(r["case_id"], r["implementation"], r["stage"]): r for r in result["benchmarks"]}
    summary = {
        "gpu": result["hardware"]["gpu"],
        "configurations": len(result["configurations"]),
        "correctness_checks": result["correctness"]["passed"],
        "method": result["method"]["timing"],
        "stages": {},
    }
    for stage in ("dispatch", "combine", "dispatch+combine"):
        ratios, reductions, memory_reductions = [], [], []
        for case in result["configurations"]:
            cid = case["case_id"]
            reference = by_key[cid, "PyTorch reference", stage]
            scalar = by_key[cid, "CUDA scalar", stage]
            vector = by_key[cid, "CUDA vectorized", stage]
            ratios.append(reference["median_us"] / vector["median_us"])
            reductions.append(1 - vector["median_us"] / scalar["median_us"])
            memory_reductions.append(1 - vector["workspace_bytes"] / reference["workspace_bytes"])
        summary["stages"][stage] = {
            "median_per_configuration_speedup": statistics.median(ratios),
            "minimum_speedup": min(ratios),
            "maximum_speedup": max(ratios),
            "maximum_latency_reduction_vs_scalar_cuda": max(reductions),
            "minimum_persistent_workspace_reduction": min(memory_reductions),
            "maximum_persistent_workspace_reduction": max(memory_reductions),
        }
    return summary


def check_manifest(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    for relative, expected in manifest["sha256"].items():
        resolved = (directory / relative).resolve()
        if not resolved.is_relative_to(directory.resolve()):
            raise ValueError("manifest path escapes the evidence directory")
        if hashlib.sha256(resolved.read_bytes()).hexdigest() != expected:
            raise ValueError(f"evidence hash mismatch: {relative}")
    return manifest
