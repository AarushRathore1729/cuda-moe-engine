"""Seeded synthetic routed-token inputs, generated outside timed regions."""

import torch


def make_case(tokens, hidden, experts, topk, dtype, pattern, seed, device="cuda"):
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((tokens, hidden), device=device, dtype=dtype, generator=generator)
    # Router is common to both implementations and excluded from transport timings.
    logits = torch.randn((tokens, experts), device=device, generator=generator)
    if pattern == "skewed":
        logits[:, :topk] += 5.0
    elif pattern != "balanced":
        raise ValueError(f"unknown pattern: {pattern}")
    selected_logits, ids64 = torch.topk(logits, topk, dim=-1)
    weights = torch.softmax(selected_logits, dim=-1).contiguous()
    return x, ids64.to(torch.int32).contiguous(), ids64.contiguous(), weights
