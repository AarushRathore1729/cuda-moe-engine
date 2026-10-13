"""Preallocated GPU-sorting reference; also supports CPU correctness tests."""

import torch


class TorchWorkspace:
    """GPU sorting reference with reusable outputs and FP32 weighted accumulation."""

    def __init__(self, tokens, hidden, experts, topk, dtype, device="cuda", alignment=1):
        n = tokens * topk
        self.alignment = alignment
        if not 1 <= alignment <= 128:
            raise ValueError("alignment must be 1..128")
        self.capacity_rows = (
            n
            if alignment == 1
            else ((n + experts * (alignment - 1) + 1 + alignment - 1) // alignment * alignment)
        )
        self.tokens, self.topk, self.hidden = tokens, topk, hidden
        self.sorted_ids = torch.empty(n, device=device, dtype=torch.int32)
        self.order = torch.empty(n, device=device, dtype=torch.int64)
        self.token_ids = torch.empty(n, device=device, dtype=torch.int64)
        self.inverse = torch.empty((tokens, topk), device=device, dtype=torch.int64)
        self.arange = torch.arange(n, device=device, dtype=torch.int64)
        self.ones = torch.ones(n, device=device, dtype=torch.int32)
        self.counts = torch.empty(experts, device=device, dtype=torch.int32)
        self.offsets = torch.empty(experts + 1, device=device, dtype=torch.int32)
        self.packed = torch.empty((self.capacity_rows, hidden), device=device, dtype=dtype)
        if alignment > 1:
            self.unpadded_offsets = torch.empty(experts + 1, device=device, dtype=torch.int32)
            self.destinations = torch.empty(n, device=device, dtype=torch.int64)
            self.source_rows = torch.empty((n, hidden), device=device, dtype=dtype)
        self.gathered = torch.empty((n, hidden), device=device, dtype=dtype)
        self.weighted = torch.empty((tokens, topk, hidden), device=device, dtype=torch.float32)
        self.output = torch.empty((tokens, hidden), device=device, dtype=dtype)
        self.accumulated = (
            self.output
            if dtype == torch.float32
            else torch.empty((tokens, hidden), device=device, dtype=torch.float32)
        )

    def dispatch(self, x, ids32, ids64):
        self.counts.zero_()
        self.counts.scatter_add_(0, ids64.reshape(-1), self.ones)
        self.offsets[0:1].zero_()
        torch.cumsum(self.counts, dim=0, dtype=torch.int32, out=self.offsets[1:])
        torch.sort(ids32.reshape(-1), out=(self.sorted_ids, self.order))
        torch.div(self.order, self.topk, rounding_mode="floor", out=self.token_ids)
        if self.alignment == 1:
            self.inverse.reshape(-1).scatter_(0, self.order, self.arange)
            torch.index_select(x, 0, self.token_ids, out=self.packed)
        else:
            self.unpadded_offsets.copy_(self.offsets)
            padded_counts = (self.counts + self.alignment - 1) // self.alignment * self.alignment
            torch.cumsum(padded_counts, dim=0, dtype=torch.int32, out=self.offsets[1:])
            sorted_experts = self.sorted_ids.long()
            self.destinations.copy_(
                self.arange
                - self.unpadded_offsets.index_select(0, sorted_experts).long()
                + self.offsets.index_select(0, sorted_experts).long()
            )
            self.inverse.reshape(-1).scatter_(0, self.order, self.destinations)
            torch.index_select(x, 0, self.token_ids, out=self.source_rows)
            self.packed.zero_()
            self.packed.index_copy_(0, self.destinations, self.source_rows)

    def combine(self, expert_output, weights):
        torch.index_select(expert_output, 0, self.inverse.reshape(-1), out=self.gathered)
        torch.mul(
            self.gathered.view(self.tokens, self.topk, self.hidden),
            weights.unsqueeze(-1),
            out=self.weighted,
        )
        torch.sum(self.weighted, dim=1, out=self.accumulated)
        if self.output is not self.accumulated:
            self.output.copy_(self.accumulated)

    @property
    def workspace_bytes(self):
        values = [v for v in vars(self).values() if isinstance(v, torch.Tensor)]
        unique = {v.data_ptr(): v for v in values}
        return sum(v.numel() * v.element_size() for v in unique.values())
