"""Independent routing, packing, membership, and weighted-combine checks."""

import torch
from .workspace import CudaWorkspace
from .reference import TorchWorkspace
from .routing import make_case


@torch.inference_mode()
def validate_case(workspace, x, ids32, ids64, weights, experts, seed):
    tokens, hidden = x.shape
    topk = ids32.shape[1]
    workspace.dispatch(x, ids32, ids64)
    expected_counts = torch.bincount(ids64.reshape(-1), minlength=experts).to(torch.int32)
    torch.testing.assert_close(workspace.counts, expected_counts, rtol=0, atol=0)
    alignment = getattr(workspace, "alignment", 1)
    expected_spans = (expected_counts + alignment - 1) // alignment * alignment
    torch.testing.assert_close(
        workspace.offsets[1:] - workspace.offsets[:-1], expected_spans, rtol=0, atol=0
    )
    assert int(workspace.offsets[0]) == 0
    assert int(workspace.offsets[-1]) == int(expected_spans.sum())
    inverse64 = workspace.inverse.reshape(-1).long()
    if alignment == 1:
        torch.testing.assert_close(
            torch.sort(inverse64).values,
            torch.arange(tokens * topk, device=x.device),
            rtol=0,
            atol=0,
        )
    else:
        assert inverse64.unique().numel() == tokens * topk
        if inverse64.numel():
            assert int(inverse64.min()) >= 0
            assert int(inverse64.max()) < workspace.packed.shape[0]
    unpacked = workspace.packed.index_select(0, inverse64).view(tokens, topk, hidden)
    torch.testing.assert_close(unpacked, x[:, None, :].expand_as(unpacked), rtol=0, atol=0)
    # Every expert's packed slice must contain exactly its own assignments.
    inferred_experts = torch.bucketize(inverse64, workspace.offsets[1:].long(), right=True)
    torch.testing.assert_close(inferred_experts, ids64.reshape(-1), rtol=0, atol=0)
    if alignment > 1:
        occupied = torch.zeros(workspace.packed.shape[0], dtype=torch.bool, device=x.device)
        occupied[inverse64] = True
        active = torch.arange(workspace.packed.shape[0], device=x.device) < workspace.offsets[-1]
        padding = workspace.packed[active & ~occupied]
        torch.testing.assert_close(padding, torch.zeros_like(padding), rtol=0, atol=0)
    # Arbitrary outputs prevent an identity-only combine test from hiding errors.
    generator = torch.Generator(device=x.device).manual_seed(seed + 501)
    canonical = torch.randn(
        (tokens * topk, hidden), device=x.device, dtype=x.dtype, generator=generator
    )
    expert_output = torch.empty_like(workspace.packed)
    expert_output.index_copy_(0, inverse64, canonical)
    workspace.combine(expert_output, weights)
    expected = (
        (canonical.view(tokens, topk, hidden).float() * weights[:, :, None]).sum(1).to(x.dtype)
    )
    tolerance = 2e-2 if x.dtype == torch.bfloat16 else (2e-3 if x.dtype == torch.float16 else 2e-5)
    torch.testing.assert_close(workspace.output, expected, rtol=tolerance, atol=tolerance)
    difference = (workspace.output.float() - expected.float()).abs()
    return float(difference.max()) if difference.numel() else 0.0


@torch.inference_mode()
def run_tests(extension):
    shapes = [
        (0, 17, 8, 2),
        (1, 17, 8, 2),
        (7, 31, 8, 1),
        (33, 128, 8, 2),
        (129, 512, 16, 2),
        (257, 1024, 16, 2),
    ]
    errors = []
    for dtype in [torch.float32, torch.float16]:
        for index, (tokens, hidden, experts, topk) in enumerate(shapes):
            pattern = "skewed" if index % 2 == 0 else "balanced"
            x, ids32, ids64, weights = make_case(
                tokens, hidden, experts, topk, dtype, pattern, 42 + index
            )
            implementations = [TorchWorkspace(tokens, hidden, experts, topk, dtype)]
            implementations += [
                CudaWorkspace(
                    tokens, hidden, experts, topk, dtype, vectorized=v, extension=extension
                )
                for v in [False, True]
            ]
            for workspace in implementations:
                errors.append(
                    validate_case(workspace, x, ids32, ids64, weights, experts, 42 + index)
                )
    torch.cuda.synchronize()
    summary = {
        "passed": len(errors),
        "failed": 0,
        "max_absolute_error": max(errors, default=0.0),
        "coverage": "FP32/FP16; K=1/2; empty batch; empty experts; skew; odd H",
    }
    print(
        f"Correctness: {summary['passed']} checks passed; "
        f"max absolute error {summary['max_absolute_error']:.3g}",
        flush=True,
    )
    return summary
