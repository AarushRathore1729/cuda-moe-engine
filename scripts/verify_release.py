"""Verify recorded A100 evidence without importing torch or running a GPU."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from moe_dispatch.evidence import check_manifest, read_results, summarize  # noqa: E402


def validate_block(data, root=ROOT):
    if data.get("kind") != "moe_block_benchmark":
        raise ValueError("not a block benchmark")
    cases = {c["case_id"]: c for c in data["configurations"]}
    if not cases or len(cases) != len(data["configurations"]):
        raise ValueError("empty or duplicate cases")
    for c in cases.values():
        counts = c["expert_counts"]
        if len(counts) != c["experts"] or any(n < 0 for n in counts):
            raise ValueError("invalid expert counts")
        if sum(counts) != c["tokens"] * c["topk"]:
            raise ValueError("routed token conservation failed")
        if c["expert_backend"] != data["method"]["expert_backend"]:
            raise ValueError("expert backend mismatch")
    implementations = {"torch", "cuda", "cuda_fused"}
    expected_validation = {(c, i) for c in cases for i in implementations}
    actual_validation = set()
    for v in data["validation"]:
        key = (v["case_id"], v["implementation"])
        if key in actual_validation or key not in expected_validation or v["passed"] is not True:
            raise ValueError("duplicate, unknown, or failed validation")
        actual_validation.add(key)
        for field in ("max_absolute_error", "relative_l2_error"):
            if not math.isfinite(v[field]) or v[field] < 0:
                raise ValueError("invalid correctness error")
        if v["relative_l2_error"] > v["relative_l2_limit"]:
            raise ValueError("relative error exceeds threshold")
    if actual_validation != expected_validation:
        raise ValueError("missing forward validation")
    stages = {"router", "dispatch", "experts", "combine", "forward"}
    expected = {(c, i, s) for c in cases for i in implementations for s in stages}
    seen, rows = set(), {}
    for r in data["benchmarks"]:
        key = (r["case_id"], r["implementation"], r["stage"])
        if key in seen:
            raise ValueError("duplicate benchmark row")
        seen.add(key)
        if key not in expected and not (
            key[0] in cases and key[1] in implementations and key[2] == "forward_graph"
            and data["method"].get("graphs_requested")
        ):
            raise ValueError("unknown benchmark row")
        samples = r["samples_us"]
        if len(samples) != data["method"]["repeats"] or not samples:
            raise ValueError("sample count mismatch")
        if any(not math.isfinite(v) or v <= 0 for v in samples):
            raise ValueError("invalid sample")
        ordered = sorted(samples)
        pos = (len(ordered) - 1) * 0.95
        low, high = math.floor(pos), math.ceil(pos)
        p95 = ordered[low] + (ordered[high] - ordered[low]) * (pos - low)
        for field, value in (("median_us", statistics.median(samples)),
                             ("p95_us", p95), ("min_us", min(samples))):
            if not math.isclose(r[field], value, rel_tol=1e-9):
                raise ValueError(f"stored {field} differs from raw samples")
        if r["persistent_workspace_bytes"] <= 0:
            raise ValueError("invalid workspace")
        rows[key] = r
    if not expected.issubset(seen):
        raise ValueError("missing benchmark row")
    if data["method"].get("graphs_requested"):
        expected_graphs = set()
        for v in data["validation"]:
            if v.get("graph_status") == "passed":
                expected_graphs.add((v["case_id"], v["implementation"], "forward_graph"))
            elif v.get("graph_status") != "unsupported" or not v.get("graph_error"):
                raise ValueError("graph validation status is missing or invalid")
        if {key for key in seen if key[2] == "forward_graph"} != expected_graphs:
            raise ValueError("graph timings do not match successful replay validation")
    sources = list((root / "src/moe_dispatch").glob("*.py"))
    sources += list((root / "src/moe_dispatch/csrc").glob("*"))
    actual_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    if actual_hashes != data["source_hashes"]:
        raise ValueError("measured source hashes differ from checkout")
    ratios = [rows[c, "torch", "forward"]["median_us"] /
              rows[c, "cuda_fused", "forward"]["median_us"] for c in cases]
    summary = {
        "gpu": data["hardware"]["gpu"], "configurations": len(cases),
        "forward_comparisons_passed": len(actual_validation),
        "source_hashes_verified": len(actual_hashes), "timing_rows_verified": len(seen),
        "median_forward_ratio": statistics.median(ratios),
        "min_forward_ratio": min(ratios), "max_forward_ratio": max(ratios),
        "max_absolute_error": max(v["max_absolute_error"] for v in data["validation"]),
        "max_relative_l2_error": max(v["relative_l2_error"] for v in data["validation"]),
        "expert_backend": data["method"]["expert_backend"],
        "graphs_requested": data["method"]["graphs_requested"],
    }
    if data["method"].get("graphs_requested"):
        successful = [c for c in cases if all(
            (c, i, "forward_graph") in rows for i in implementations
        )]
        graph_ratios = [rows[c, "torch", "forward_graph"]["median_us"] /
                        rows[c, "cuda_fused", "forward_graph"]["median_us"] for c in successful]
        summary["graph_replay_checks_passed"] = len(expected_graphs)
        summary["complete_graph_comparison_cases"] = len(successful)
        if graph_ratios:
            summary.update({"median_graph_ratio": statistics.median(graph_ratios),
                            "min_graph_ratio": min(graph_ratios),
                            "max_graph_ratio": max(graph_ratios)})
    return summary


def verify(root=ROOT):
    block = root / "benchmarks/a100-block"
    check_manifest(root / "benchmarks/a100")
    transport = summarize(read_results(root / "benchmarks/a100/results.json"))
    quick = validate_block(json.loads((block / "block_results.json").read_text()), root)
    smoke = validate_block(json.loads((block / "smoke/block_results.json").read_text()), root)
    trace = json.loads((block / "moe_block_trace.json").read_text())
    kernels = [e for e in trace["traceEvents"] if e.get("cat") == "kernel"]
    if not kernels:
        raise ValueError("profiler trace contains no GPU kernels")
    log = (block / "run.log").read_text()
    for marker in ("81 passed, 39 deselected", "39 passed, 81 deselected",
                   '"passed": 36', '"failed": 0'):
        if marker not in log:
            raise ValueError(f"missing recorded test evidence: {marker}")
    optional = {}
    for name in ("graphs", "model"):
        path = block / name / "block_results.json"
        if path.exists():
            optional[name] = validate_block(json.loads(path.read_text()), root)
    if optional:
        optional_log = (block / "optional_run.log").read_text()
        for marker in ("Saved runs/block-graphs/block_results.json", "Saved runs/model-shaped/block_results.json"):
            if marker not in optional_log:
                raise ValueError("optional execution log missing completion evidence")
    if (block / "manifest.json").exists():
        check_manifest(block)
    return {"quick": quick, "smoke": smoke, "optional": optional, "transport": transport,
            "recorded_tests": {"cpu_passed": 81, "gpu_passed": 39, "transport_passed": 36},
            "profiler_gpu_kernel_events": len(kernels)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = verify()
    content = json.dumps(summary, indent=2) + "\n"
    if args.output:
        args.output.write_text(content)
    print(content)
