import pytest
import torch
from moe_dispatch.layer import MoEBlock, MoEConfig, token_major_oracle


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tokens", [0, 1, 13, 33])
@pytest.mark.parametrize("pattern", ["balanced", "skewed"])
@pytest.mark.parametrize("topk", [1, 2])
def test_complete_cpu_block_matches_token_major_oracle(dtype, tokens, pattern, topk):
    cfg = MoEConfig(hidden=32, intermediate=64, experts=8, topk=topk)
    block = MoEBlock(cfg, device="cpu", dtype=dtype, implementation="torch")
    block.set_routing_pattern(pattern)
    x = torch.randn(tokens, 32, generator=torch.Generator().manual_seed(42)).to(dtype)
    actual = block(x).clone()
    expected = token_major_oracle(block, x)
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)


def test_invalid_block_shapes_fail_early():
    with pytest.raises(ValueError, match="multiples of 8"):
        MoEConfig(hidden=17)
    with pytest.raises(ValueError, match="expert/topk"):
        MoEConfig(experts=1, topk=2)


def test_shared_weights_and_workspace_reuse():
    block = MoEBlock(
        MoEConfig(32, 64, 8, 2), device="cpu", dtype=torch.float32, implementation="torch"
    )
    x = torch.randn(13, 32)
    first = block.workspace(x)
    block(x)
    assert block.workspace(x) is first
