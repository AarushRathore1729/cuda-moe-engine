"""Reusable GPU buffers. Use one workspace per independent concurrent invocation."""

import torch
from ._extension import get_extension


class CudaWorkspace:
    def __init__(
        self,
        tokens,
        hidden,
        experts,
        topk,
        dtype,
        vectorized=True,
        device="cuda",
        extension=None,
        alignment=1,
    ):
        if tokens < 0 or hidden < 1 or experts < 1 or experts > 64 or topk not in (1, 2):
            raise ValueError("supports T>=0, H>=1, 1..64 experts, and topk=1/2")
        if tokens * topk > 2**31 - 1 or hidden > 2**31 - 1:
            raise ValueError("shape exceeds int32 indexing capacity")
        if dtype not in (torch.float16, torch.float32, torch.bfloat16):
            raise ValueError("CUDA workspace supports FP16, FP32, and BF16")
        if not 1 <= alignment <= 128:
            raise ValueError("alignment must be 1..128")
        self.alignment = alignment
        n = tokens * topk
        self.capacity_rows = (
            n
            if alignment == 1
            else ((n + experts * (alignment - 1) + 1 + alignment - 1) // alignment * alignment)
        )
        if self.capacity_rows > 2**31 - 1:
            raise ValueError("padded capacity exceeds int32 indexing capacity")
        device = torch.device(device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("CudaWorkspace requires an NVIDIA CUDA GPU")
        self.shape = (tokens, hidden, experts, topk)
        self.dtype = dtype
        self.device = device
        self.vectorized = vectorized
        with torch.cuda.device(device):
            self.extension = extension if extension is not None else get_extension()
            self.counts = torch.empty(experts, device=device, dtype=torch.int32)
            self.offsets = torch.empty(experts + 1, device=device, dtype=torch.int32)
            self.slots = torch.empty(tokens * topk, device=device, dtype=torch.int32)
            self.packed = torch.empty((self.capacity_rows, hidden), device=device, dtype=dtype)
            self.inverse = torch.empty((tokens, topk), device=device, dtype=torch.int32)
            self.output = torch.empty((tokens, hidden), device=device, dtype=dtype)

    def dispatch(self, x, ids32, ids64=None):
        """Hot path: valid in-range int32 IDs and contiguous inputs are required."""
        if self.alignment == 1:
            self.extension.dispatch_out(
                x,
                ids32,
                self.counts,
                self.offsets,
                self.slots,
                self.packed,
                self.inverse,
                self.vectorized,
            )
        else:
            self.extension.dispatch_aligned_out(
                x,
                ids32,
                self.counts,
                self.offsets,
                self.slots,
                self.packed,
                self.inverse,
                self.vectorized,
                self.alignment,
            )

    def combine(self, expert_output, weights):
        self.extension.combine_out(expert_output, self.inverse, weights, self.output)

    @property
    def workspace_bytes(self):
        return sum(
            t.numel() * t.element_size()
            for t in [self.counts, self.offsets, self.slots, self.packed, self.inverse, self.output]
        )
