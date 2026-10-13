import pytest
import torch
from moe_dispatch._extension import get_extension

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required"),
]


@pytest.mark.parametrize("experts", [2, 8, 16, 64])
@pytest.mark.parametrize("topk", [1, 2])
def test_fused_router_matches_independent_pytorch(experts, topk):
    generator = torch.Generator(device="cuda").manual_seed(42)
    logits = torch.randn(257, experts, device="cuda", generator=generator)
    bias = torch.randn(experts, device="cuda", generator=generator)
    ids32 = torch.empty(257, topk, device="cuda", dtype=torch.int32)
    ids64 = torch.empty(257, topk, device="cuda", dtype=torch.int64)
    weights = torch.empty(257, topk, device="cuda")
    get_extension().topk_route_out(logits, bias, ids32, ids64, weights)
    selected, expected_ids = torch.topk(logits + bias, topk, dim=-1)
    torch.testing.assert_close(ids64, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(ids32, expected_ids.int(), rtol=0, atol=0)
    torch.testing.assert_close(weights, selected.softmax(-1), rtol=2e-6, atol=2e-6)


def test_fused_router_ties_and_empty_tokens():
    extension = get_extension()
    for tokens in (0, 5):
        logits = torch.zeros(tokens, 64, device="cuda")
        bias = torch.zeros(64, device="cuda")
        ids32 = torch.empty(tokens, 2, device="cuda", dtype=torch.int32)
        ids64 = torch.empty(tokens, 2, device="cuda", dtype=torch.int64)
        weights = torch.empty(tokens, 2, device="cuda")
        extension.topk_route_out(logits, bias, ids32, ids64, weights)
        expected = torch.tensor([0, 1], device="cuda").expand(tokens, 2)
        torch.testing.assert_close(ids64, expected, rtol=0, atol=0)
        torch.testing.assert_close(weights, torch.full_like(weights, 0.5), rtol=0, atol=0)
