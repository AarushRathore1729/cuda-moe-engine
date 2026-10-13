"""Optional CPU/CUDA Chrome trace for a warmed token-transport workload."""

from pathlib import Path
import torch
from ._extension import get_extension
from .routing import make_case
from .workspace import CudaWorkspace
from .reference import TorchWorkspace


@torch.inference_mode()
def export_trace(destination):
    extension = get_extension()
    x, ids32, ids64, weights = make_case(2048, 1024, 16, 2, torch.float16, "balanced", 42)
    implementations = {
        "PyTorch reference": TorchWorkspace(2048, 1024, 16, 2, x.dtype),
        "CUDA vectorized": CudaWorkspace(2048, 1024, 16, 2, x.dtype, extension=extension),
    }
    for workspace in implementations.values():
        for _ in range(10):
            workspace.dispatch(x, ids32, ids64)
            workspace.combine(workspace.packed, weights)
    torch.cuda.synchronize()
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=True) as profiler:
        for name, workspace in implementations.items():
            for _ in range(3):
                with torch.autograd.profiler.record_function(name + "/dispatch"):
                    workspace.dispatch(x, ids32, ids64)
                with torch.autograd.profiler.record_function(name + "/combine"):
                    workspace.combine(workspace.packed, weights)
        torch.cuda.synchronize()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(destination))
    return destination
