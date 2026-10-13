import pytest
import torch
from moe_dispatch._extension import get_extension
from moe_dispatch.workspace import CudaWorkspace
from moe_dispatch.routing import make_case
from moe_dispatch.validation import validate_case
from moe_dispatch.experts import swiglu_reference
from moe_dispatch.layer import MoEBlock, MoEConfig
from moe_dispatch.block_benchmark import check_forward

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required"),
]


@pytest.fixture(scope="module")
def extension():
    return get_extension()


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("alignment", [1, 8])
@pytest.mark.parametrize("tokens", [0, 1, 33, 257])
def test_cuda_transport_empty_skew_and_partial_tiles(extension, dtype, alignment, tokens):
    x, i32, i64, w = make_case(tokens, 32, 8, 2, dtype, "skewed", 42)
    workspace = CudaWorkspace(tokens, 32, 8, 2, dtype, alignment=alignment, extension=extension)
    validate_case(workspace, x, i32, i64, w, 8, 42)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_fused_swiglu_and_unused_rows(extension, dtype):
    projection = torch.randn(37, 34, device="cuda").to(dtype)
    output = torch.empty(37, 17, device="cuda", dtype=dtype)
    rows = torch.tensor([21], device="cuda", dtype=torch.int32)
    extension.swiglu_out(projection, rows, output)
    expected = swiglu_reference(projection)
    tolerance = 0.02 if dtype == torch.bfloat16 else (0.002 if dtype == torch.float16 else 0.00002)
    torch.testing.assert_close(output[:21], expected[:21], rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(output[21:], torch.zeros_like(output[21:]), rtol=0, atol=0)


@pytest.mark.parametrize("implementation", ["torch", "cuda", "cuda_fused"])
def test_full_gpu_block_against_independent_oracle(implementation):
    block = MoEBlock(MoEConfig(32, 64, 8, 2), implementation=implementation)
    block.set_routing_pattern("skewed")
    x = torch.randn(33, 32, device="cuda").bfloat16()
    check_forward(block, x)
