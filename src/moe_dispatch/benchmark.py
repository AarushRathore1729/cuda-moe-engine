"""Seeded token-transport benchmarks with reusable workspaces and raw samples."""

import math
import statistics
import subprocess
import platform
import time
import torch
from .workspace import CudaWorkspace
from .reference import TorchWorkspace
from .routing import make_case
from .validation import validate_case


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


@torch.inference_mode()
def measure(function, warmup, repeats):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(repeats)
    ]
    for start, end in events:
        start.record()
        function()
        end.record()
    torch.cuda.synchronize()
    values_us = [start.elapsed_time(end) * 1000.0 for start, end in events]
    return {
        "median_us": statistics.median(values_us),
        "p95_us": percentile(values_us, 0.95),
        "min_us": min(values_us),
        "samples_us": values_us,
    }


@torch.inference_mode()
def benchmark(extension, suite, warmup, repeats):
    if suite == "smoke":
        cases = [(128, 512, 8, "balanced"), (2048, 1024, 16, "skewed")]
    elif suite == "quick":
        cases = [
            (t, h, e, p)
            for t in [128, 2048, 8192]
            for h, e in [(512, 8), (1024, 16)]
            for p in ["balanced", "skewed"]
        ]
    else:
        cases = [
            (t, h, e, p)
            for t in [128, 512, 2048, 8192]
            for h in [512, 1024]
            for e in [8, 16]
            for p in ["balanced", "skewed"]
        ]
    records, configurations = [], []
    for case_id, (tokens, hidden, experts, pattern) in enumerate(cases):
        x, ids32, ids64, weights = make_case(
            tokens, hidden, experts, 2, torch.float16, pattern, 100 + case_id
        )
        workspaces = {
            "PyTorch reference": TorchWorkspace(tokens, hidden, experts, 2, x.dtype),
            "CUDA scalar": CudaWorkspace(
                tokens, hidden, experts, 2, x.dtype, vectorized=False, extension=extension
            ),
            "CUDA vectorized": CudaWorkspace(
                tokens, hidden, experts, 2, x.dtype, vectorized=True, extension=extension
            ),
        }
        configurations.append(
            {
                "case_id": case_id,
                "tokens": tokens,
                "hidden": hidden,
                "experts": experts,
                "topk": 2,
                "pattern": pattern,
                "dtype": "float16",
                "expert_counts": torch.bincount(ids64.reshape(-1), minlength=experts)
                .cpu()
                .tolist(),
            }
        )
        # Every configuration is validated before any result is emitted.
        for workspace in workspaces.values():
            validate_case(workspace, x, ids32, ids64, weights, experts, 100 + case_id)
        # Rotate order between cases to reduce systematic warm-clock/order bias.
        names = list(workspaces)
        rotation = case_id % len(names)
        names = names[rotation:] + names[:rotation]
        for name in names:
            workspace = workspaces[name]
            workspace.dispatch(x, ids32, ids64)

            def dispatch():
                workspace.dispatch(x, ids32, ids64)

            def combine():
                workspace.combine(workspace.packed, weights)

            def transport():
                workspace.dispatch(x, ids32, ids64)
                workspace.combine(workspace.packed, weights)

            for stage, function in [
                ("dispatch", dispatch),
                ("combine", combine),
                ("dispatch+combine", transport),
            ]:
                timing = measure(function, warmup, repeats)
                records.append(
                    {
                        "case_id": case_id,
                        "tokens": tokens,
                        "hidden": hidden,
                        "experts": experts,
                        "topk": 2,
                        "pattern": pattern,
                        "dtype": "float16",
                        "implementation": name,
                        "stage": stage,
                        "workspace_bytes": workspace.workspace_bytes,
                        **timing,
                    }
                )
        print(
            f"Benchmarked {case_id + 1}/{len(cases)}: T={tokens}, H={hidden}, "
            f"E={experts}, {pattern}",
            flush=True,
        )
    return configurations, records


def hardware_metadata():
    properties = torch.cuda.get_device_properties(0)
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        smi = "unavailable"
    return {
        "gpu": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "gpu_memory_bytes": properties.total_memory,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": platform.python_version(),
        "nvidia_smi": smi,
        "utc_run_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
