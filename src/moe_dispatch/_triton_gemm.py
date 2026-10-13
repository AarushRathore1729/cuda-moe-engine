"""Device-scheduled grouped GEMM for shared expert shapes and jagged row counts.

The persistent scheduling pattern is adapted from NVIDIA's MIT-licensed Triton
Grouped GEMM tutorial. Unlike that tutorial, offsets describe packed expert rows;
partial tiles are masked and expert weight tensors use ordinary PyTorch strides.
https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html
Copyright (c) 2023 - 2025 NVIDIA Corporation & Affiliates. All rights reserved.
See THIRD_PARTY_NOTICES.md for the license.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_gemm(
    A,
    B,
    C,
    ENDS,
    K: tl.constexpr,
    N: tl.constexpr,
    E: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    WORKERS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    job = tl.program_id(0)
    consumed = 0
    for expert in range(E):
        start = tl.load(ENDS + expert - 1, mask=expert > 0, other=0)
        end = tl.load(ENDS + expert)
        group_rows = end - start
        n_tiles = tl.cdiv(N, BN)
        jobs = tl.cdiv(group_rows, BM) * n_tiles
        while (job >= consumed) & (job < consumed + jobs):
            relative = job - consumed
            row_tile = relative // n_tiles
            column_tile = relative % n_tiles
            rows = start + row_tile * BM + tl.arange(0, BM)
            columns = column_tile * BN + tl.arange(0, BN)
            inner = tl.arange(0, BK)
            accumulator = tl.zeros((BM, BN), dtype=tl.float32)
            for block in range(tl.cdiv(K, BK)):
                reduction = block * BK + inner
                left = tl.load(
                    A + rows[:, None] * A0 + reduction[None, :] * A1,
                    mask=(rows[:, None] < end) & (reduction[None, :] < K),
                    other=0,
                )
                right = tl.load(
                    B + expert * B0 + reduction[:, None] * B1 + columns[None, :] * B2,
                    mask=(reduction[:, None] < K) & (columns[None, :] < N),
                    other=0,
                )
                accumulator += tl.dot(left, right)
            tl.store(
                C + rows[:, None] * C0 + columns[None, :] * C1,
                accumulator.to(C.dtype.element_ty),
                mask=(rows[:, None] < end) & (columns[None, :] < N),
            )
            job += WORKERS
        consumed += jobs


def grouped_mm(x, weights, ends, output=None):
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("Triton grouped GEMM uses FP16/BF16")
    if x.ndim != 2 or weights.ndim != 3 or x.shape[-1] != weights.shape[-2]:
        raise ValueError("expected x[R,K], weights[E,K,N]")
    if output is None:
        output = torch.empty((x.shape[0], weights.shape[-1]), device=x.device, dtype=x.dtype)
    if output.shape != (x.shape[0], weights.shape[-1]) or output.dtype != x.dtype:
        raise ValueError("incorrect grouped-GEMM output buffer")
    workers = torch.cuda.get_device_properties(x.device).multi_processor_count
    with torch.cuda.device(x.device):
        _grouped_gemm[(workers,)](
            x,
            weights,
            output,
            ends,
            x.shape[-1],
            weights.shape[-1],
            weights.shape[0],
            *x.stride(),
            *weights.stride(),
            *output.stride(),
            workers,
            64,
            64,
            32,
            num_warps=4,
            num_stages=3,
        )
    return output
