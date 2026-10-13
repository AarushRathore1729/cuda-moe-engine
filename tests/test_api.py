import pytest
import torch
from moe_dispatch.ops import validate_inputs
from moe_dispatch.workspace import CudaWorkspace


@pytest.mark.parametrize(
    "bad_ids", [torch.tensor([[0.0, 1.0]]), torch.tensor([[-1, 0]]), torch.tensor([[0, 8]])]
)
def test_invalid_expert_ids_are_rejected(bad_ids):
    with pytest.raises((TypeError, ValueError)):
        validate_inputs(torch.randn(1, 32), bad_ids, 8)


def test_forward_only_contract_is_explicit():
    with pytest.raises(ValueError, match="forward-only"):
        validate_inputs(torch.randn(1, 32, requires_grad=True), torch.tensor([[0, 1]]), 8)


def test_cpu_workspace_does_not_compile_cuda():
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        CudaWorkspace(1, 32, 8, 2, torch.float32, device="cpu")
