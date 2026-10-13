"""Convenience API with input validation; benchmarks use prevalidated workspace methods."""

from dataclasses import dataclass
import torch
from .workspace import CudaWorkspace


@dataclass(frozen=True)
class PackedTokens:
    """Views into a workspace. Clone tensors before reusing that workspace if retaining them."""

    packed: torch.Tensor
    inverse: torch.Tensor
    counts: torch.Tensor
    offsets: torch.Tensor
    workspace: CudaWorkspace


def validate_inputs(x, expert_ids, experts):
    if x.ndim != 2 or expert_ids.ndim != 2:
        raise ValueError("expected x[T,H] and expert_ids[T,K]")
    if x.shape[0] != expert_ids.shape[0] or x.shape[1] < 1:
        raise ValueError("token dimensions must match and H must be positive")
    if x.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        raise TypeError("x must be FP16, FP32, or BF16")
    if expert_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("expert_ids must be int32 or int64")
    if x.device != expert_ids.device:
        raise ValueError("x and expert_ids must be on the same device")
    if not x.is_contiguous() or not expert_ids.is_contiguous():
        raise ValueError("inputs must be contiguous")
    if not isinstance(experts, int) or not 1 <= experts <= 64:
        raise ValueError("experts must be an integer in 1..64")
    if expert_ids.shape[1] not in (1, 2):
        raise ValueError("topk must be 1 or 2")
    if x.requires_grad:
        raise ValueError("CUDA operators are forward-only; detach inputs or use inference_mode")
    if expert_ids.numel() and (expert_ids.min().item() < 0 or expert_ids.max().item() >= experts):
        raise ValueError("expert_ids contains an out-of-range expert")


@torch.inference_mode()
def dispatch(x, expert_ids, experts, *, workspace=None, vectorized=True, alignment=1):
    """Pack routed tokens by expert. Bounds checks synchronize; validate once for hot loops."""
    validate_inputs(x, expert_ids, experts)
    if not x.is_cuda:
        raise RuntimeError("dispatch requires CUDA; TorchWorkspace provides the CPU reference")
    expected_shape = (x.shape[0], x.shape[1], experts, expert_ids.shape[1])
    if workspace is None:
        workspace = CudaWorkspace(
            *expected_shape, x.dtype, vectorized=vectorized, device=x.device, alignment=alignment
        )
    if workspace.shape != expected_shape or workspace.dtype != x.dtype:
        raise ValueError("workspace shape/dtype does not match inputs")
    if workspace.packed.device != x.device:
        raise ValueError("workspace device does not match inputs")
    workspace.dispatch(x, expert_ids.to(torch.int32))
    return PackedTokens(
        workspace.packed, workspace.inverse, workspace.counts, workspace.offsets, workspace
    )


@torch.inference_mode()
def combine(expert_output, batch, weights):
    """Weighted gather/reduction; returns a reusable output view and accumulates in FP32."""
    if weights.shape != batch.inverse.shape:
        raise ValueError("weights must have shape [T,K]")
    if expert_output.shape != batch.packed.shape or expert_output.dtype != batch.packed.dtype:
        raise ValueError("expert output shape/dtype must match packed tokens")
    if weights.device != batch.packed.device or expert_output.device != batch.packed.device:
        raise ValueError("all tensors must be on the workspace device")
    if not weights.is_contiguous() or not expert_output.is_contiguous():
        raise ValueError("expert output and weights must be contiguous")
    if not weights.is_floating_point():
        raise TypeError("weights must be floating point")
    batch.workspace.combine(expert_output, weights.to(torch.float32))
    return batch.workspace.output
