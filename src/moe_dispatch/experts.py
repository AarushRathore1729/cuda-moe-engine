"""Grouped expert matmuls with a documented backend choice."""

import importlib.util
import torch


def resolve_backend(device, requested="auto"):
    if requested not in ("auto", "torch", "triton"):
        raise ValueError("expert backend must be auto, torch, or triton")
    if torch.device(device).type != "cuda":
        if requested == "triton":
            raise RuntimeError("Triton grouped GEMM requires CUDA")
        return "torch"
    if requested == "auto":
        return "triton" if importlib.util.find_spec("triton") else "torch"
    if requested == "triton" and not importlib.util.find_spec("triton"):
        raise RuntimeError("Triton is missing; install triton or choose --expert-backend torch")
    return requested


def grouped_linear(x, weight, ends, *, backend, output=None):
    """weight is [E,N,K]; transpose views expose logical [E,K,N] to the backend."""
    logical_weight = weight.transpose(-2, -1)
    if backend == "triton":
        from ._triton_gemm import grouped_mm

        return grouped_mm(x, logical_weight, ends, output)
    function = getattr(torch.nn.functional, "grouped_mm", getattr(torch, "_grouped_mm", None))
    if function is None:
        raise RuntimeError("This PyTorch version has no grouped_mm; choose the Triton backend")
    result = function(x, logical_weight, offs=ends)
    if output is not None:
        output.copy_(result)
        return output
    return result


def swiglu_reference(projection):
    gate, up = projection.chunk(2, dim=-1)
    # Match the fused CUDA kernel: FP32 activation/multiply followed by one cast.
    return (torch.nn.functional.silu(gate.float()) * up.float()).to(projection.dtype)
