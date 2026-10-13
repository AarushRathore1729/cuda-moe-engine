import pytest
import torch
from moe_dispatch.reference import TorchWorkspace
from moe_dispatch.routing import make_case
from moe_dispatch.validation import validate_case


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("alignment", [1, 8, 16])
@pytest.mark.parametrize("shape", [(0, 17, 8, 2), (1, 17, 8, 2), (7, 31, 8, 1), (33, 128, 8, 2)])
def test_packing_and_combine_against_independent_oracle(dtype, alignment, shape):
    tokens, hidden, experts, topk = shape
    x, ids32, ids64, weights = make_case(
        tokens, hidden, experts, topk, dtype, "skewed", 42, device="cpu"
    )
    workspace = TorchWorkspace(*shape, dtype, device="cpu", alignment=alignment)
    validate_case(workspace, x, ids32, ids64, weights, experts, 42)
