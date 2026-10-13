"""A real top-2 MoE feedforward block; forward validation, not a model speed claim."""

import math
import torch
from .workspace import CudaWorkspace
from .reference import TorchWorkspace


@torch.inference_mode()
def block_demo(extension):
    """Small real MoE feedforward block: forward correctness, not a speed claim."""
    tokens, hidden, intermediate, experts, topk = 64, 128, 256, 8, 2
    generator = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randn(tokens, hidden, device="cuda", generator=generator)
    router = torch.randn(hidden, experts, device="cuda", generator=generator) / math.sqrt(hidden)
    selected, ids64 = torch.topk(x @ router, topk, dim=-1)
    weights = torch.softmax(selected, dim=-1).contiguous()
    ids32 = ids64.to(torch.int32).contiguous()
    w1 = torch.randn(experts, hidden, intermediate, device="cuda", generator=generator) / math.sqrt(
        hidden
    )
    w2 = torch.randn(experts, intermediate, hidden, device="cuda", generator=generator) / math.sqrt(
        intermediate
    )

    def run(workspace):
        workspace.dispatch(x, ids32, ids64)
        expert_output = torch.empty_like(workspace.packed)
        # These shared host offsets are acceptable for the functional demonstration.
        # The block is not included in any latency/speedup benchmark.
        boundaries = workspace.offsets.cpu().tolist()
        for expert in range(experts):
            first, last = boundaries[expert : expert + 2]
            if first < last:
                expert_output[first:last] = (
                    torch.nn.functional.silu(workspace.packed[first:last] @ w1[expert]) @ w2[expert]
                )
        workspace.combine(expert_output, weights)
        return workspace.output.clone()

    reference = run(TorchWorkspace(tokens, hidden, experts, topk, x.dtype))
    actual = run(CudaWorkspace(tokens, hidden, experts, topk, x.dtype, extension=extension))
    torch.testing.assert_close(actual, reference, atol=2e-4, rtol=2e-4)
    # Independent token-major expert evaluation provides a second oracle.
    expected = torch.zeros_like(x)
    for expert in range(experts):
        expert_result = torch.nn.functional.silu(x @ w1[expert]) @ w2[expert]
        gate_weight = (weights * (ids64 == expert)).sum(dim=1)
        expected += expert_result * gate_weight[:, None]
    torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-4)
    result = {
        "passed": True,
        "tokens": tokens,
        "hidden": hidden,
        "experts": experts,
        "topk": topk,
        "output_shape": list(actual.shape),
        "max_absolute_error": float((actual - expected).abs().max()),
        "benchmark_note": "functional MoE block only; no block/model speedup claimed",
    }
    print(
        "MoE block demo: passed against grouped and independent token-major references.", flush=True
    )
    return result
