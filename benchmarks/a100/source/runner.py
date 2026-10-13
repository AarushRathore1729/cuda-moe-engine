"""CUDA MoE token dispatch/combine experiment; run on an NVIDIA CUDA GPU."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

import torch
from torch.utils.cpp_extension import load_inline


CPP_SOURCE = r'''
#include <torch/extension.h>
void dispatch_out(torch::Tensor x, torch::Tensor ids, torch::Tensor counts,
    torch::Tensor offsets, torch::Tensor slots, torch::Tensor packed,
    torch::Tensor inverse, bool vectorized);
void combine_out(torch::Tensor packed_output, torch::Tensor inverse,
    torch::Tensor weights, torch::Tensor output);
'''

CUDA_SOURCE = r'''
#include <torch/types.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <climits>
#include <type_traits>

__global__ void count_slots_kernel(const int* ids, int* counts, int* slots,
                                  int n, int experts) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < n) {
        int expert = ids[p];
        // Input IDs must be in range. The branch avoids an invalid global write.
        slots[p] = (expert >= 0 && expert < experts)
                 ? atomicAdd(counts + expert, 1) : -1;
    }
}

__global__ void offsets_kernel(const int* counts, int* offsets, int experts) {
    // There are only 8/16 experts in the main benchmark: a tiny serial scan.
    if (threadIdx.x == 0) {
        int running = 0;
        offsets[0] = 0;
        for (int e = 0; e < experts; ++e) {
            running += counts[e];
            offsets[e + 1] = running;
        }
    }
}

template <typename scalar_t>
__global__ void pack_scalar_kernel(const scalar_t* x, const int* ids,
    const int* slots, const int* offsets, scalar_t* packed, int* inverse,
    int n, int hidden, int topk, int experts) {
    int p = blockIdx.x;
    if (p >= n) return;
    int expert = ids[p];
    if (expert < 0 || expert >= experts || slots[p] < 0) return;
    int destination = offsets[expert] + slots[p];
    if (threadIdx.x == 0) inverse[p] = destination;
    const scalar_t* source_row = x + static_cast<int64_t>(p / topk) * hidden;
    scalar_t* target_row = packed + static_cast<int64_t>(destination) * hidden;
    for (int column = threadIdx.x; column < hidden; column += blockDim.x)
        target_row[column] = source_row[column];
}

template <typename scalar_t> struct Vector4;
template <> struct Vector4<float> { using type = float4; };
template <> struct Vector4<at::Half> { using type = uint2; };

template <typename scalar_t>
__global__ void pack_vector_kernel(const scalar_t* x, const int* ids,
    const int* slots, const int* offsets, scalar_t* packed, int* inverse,
    int n, int hidden, int topk, int experts) {
    int p = blockIdx.x;
    if (p >= n) return;
    int expert = ids[p];
    if (expert < 0 || expert >= experts || slots[p] < 0) return;
    int destination = offsets[expert] + slots[p];
    if (threadIdx.x == 0) inverse[p] = destination;
    using vector_t = typename Vector4<scalar_t>::type;
    const vector_t* source = reinterpret_cast<const vector_t*>(
        x + static_cast<int64_t>(p / topk) * hidden);
    vector_t* target = reinterpret_cast<vector_t*>(
        packed + static_cast<int64_t>(destination) * hidden);
    for (int column = threadIdx.x; column < hidden / 4; column += blockDim.x)
        target[column] = source[column];
}

template <typename scalar_t>
__global__ void combine_kernel(const scalar_t* expert_output,
    const int* inverse, const float* weights, scalar_t* output,
    int tokens, int hidden, int topk) {
    int token = blockIdx.x;
    if (token >= tokens) return;
    for (int column = threadIdx.x; column < hidden; column += blockDim.x) {
        float total = 0.0f;
        for (int k = 0; k < topk; ++k) {
            int p = token * topk + k;
            int row = inverse[p];
            if (row >= 0 && row < tokens * topk)
                total += weights[p] * static_cast<float>(
                    expert_output[static_cast<int64_t>(row) * hidden + column]);
        }
        output[static_cast<int64_t>(token) * hidden + column] = scalar_t(total);
    }
}

static void check_cuda_tensor(const torch::Tensor& tensor,
                              const torch::Tensor& anchor) {
    TORCH_CHECK(tensor.is_cuda(), "all tensors must be CUDA tensors");
    TORCH_CHECK(tensor.is_contiguous(), "all tensors must be contiguous");
    TORCH_CHECK(tensor.device() == anchor.device(), "device mismatch");
}

void dispatch_out(torch::Tensor x, torch::Tensor ids, torch::Tensor counts,
    torch::Tensor offsets, torch::Tensor slots, torch::Tensor packed,
    torch::Tensor inverse, bool vectorized) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "x must be contiguous CUDA");
    TORCH_CHECK(x.dim() == 2 && ids.dim() == 2, "expected x[T,H], ids[T,K]");
    TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf,
                "only FP32 and FP16 are supported");
    TORCH_CHECK(ids.scalar_type() == at::kInt, "ids must be int32");
    int64_t tokens64 = x.size(0), hidden64 = x.size(1), topk64 = ids.size(1);
    TORCH_CHECK(ids.size(0) == tokens64 && hidden64 > 0,
                "incompatible token dimensions");
    TORCH_CHECK(topk64 == 1 || topk64 == 2, "topk must be 1 or 2");
    TORCH_CHECK(tokens64 * topk64 <= INT_MAX && hidden64 <= INT_MAX,
                "shape exceeds int32 indexing capacity");
    int tokens = tokens64, hidden = hidden64, topk = topk64;
    int n = tokens * topk, experts = counts.numel();
    TORCH_CHECK(experts > 0 && experts <= 64, "supports 1..64 experts");
    for (const auto& t : {ids, counts, offsets, slots, packed, inverse})
        check_cuda_tensor(t, x);
    TORCH_CHECK(counts.scalar_type() == at::kInt &&
                offsets.scalar_type() == at::kInt &&
                slots.scalar_type() == at::kInt &&
                inverse.scalar_type() == at::kInt, "workspace indices must be int32");
    TORCH_CHECK(offsets.numel() == experts + 1 && slots.numel() == n &&
                inverse.numel() == n, "incorrect index workspace size");
    TORCH_CHECK(packed.dim() == 2 && packed.size(0) == n &&
                packed.size(1) == hidden && packed.scalar_type() == x.scalar_type(),
                "incorrect packed workspace");
    c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaMemsetAsync(counts.data_ptr<int>(), 0,
                                  experts * sizeof(int), stream));
    if (n > 0)
        count_slots_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
            ids.data_ptr<int>(), counts.data_ptr<int>(), slots.data_ptr<int>(), n, experts);
    offsets_kernel<<<1, 1, 0, stream>>>(counts.data_ptr<int>(), offsets.data_ptr<int>(), experts);
    if (n > 0) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "moe_pack", [&] {
            // The API rejects FP64, so it is sufficient to instantiate FP32/FP16.
            if constexpr (std::is_same<scalar_t, float>::value ||
                          std::is_same<scalar_t, at::Half>::value) {
                if (vectorized && hidden % 4 == 0)
                    pack_vector_kernel<scalar_t><<<n, 128, 0, stream>>>(
                        x.data_ptr<scalar_t>(), ids.data_ptr<int>(), slots.data_ptr<int>(),
                        offsets.data_ptr<int>(), packed.data_ptr<scalar_t>(),
                        inverse.data_ptr<int>(), n, hidden, topk, experts);
                else
                    pack_scalar_kernel<scalar_t><<<n, 128, 0, stream>>>(
                        x.data_ptr<scalar_t>(), ids.data_ptr<int>(), slots.data_ptr<int>(),
                        offsets.data_ptr<int>(), packed.data_ptr<scalar_t>(),
                        inverse.data_ptr<int>(), n, hidden, topk, experts);
            }
        });
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void combine_out(torch::Tensor packed_output, torch::Tensor inverse,
    torch::Tensor weights, torch::Tensor output) {
    TORCH_CHECK(packed_output.is_cuda() && packed_output.is_contiguous(),
                "expert output must be contiguous CUDA");
    TORCH_CHECK(output.dim() == 2 && inverse.dim() == 2 && weights.dim() == 2,
                "expected output[T,H], inverse[T,K], weights[T,K]");
    TORCH_CHECK(packed_output.dim() == 2 &&
                (packed_output.scalar_type() == at::kFloat ||
                 packed_output.scalar_type() == at::kHalf), "unsupported output dtype");
    for (const auto& t : {inverse, weights, output}) check_cuda_tensor(t, packed_output);
    TORCH_CHECK(inverse.scalar_type() == at::kInt && weights.scalar_type() == at::kFloat,
                "inverse must be int32; weights must be FP32");
    int64_t tokens64 = output.size(0), hidden64 = output.size(1), topk64 = inverse.size(1);
    TORCH_CHECK(topk64 == 1 || topk64 == 2, "topk must be 1 or 2");
    TORCH_CHECK(tokens64 * topk64 <= INT_MAX && hidden64 <= INT_MAX && hidden64 > 0,
                "invalid output shape");
    TORCH_CHECK(inverse.size(0) == tokens64 && weights.sizes() == inverse.sizes() &&
                packed_output.size(0) == tokens64 * topk64 &&
                packed_output.size(1) == hidden64 &&
                output.scalar_type() == packed_output.scalar_type(), "shape/dtype mismatch");
    c10::cuda::CUDAGuard guard(packed_output.device());
    if (tokens64 > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(packed_output.scalar_type(), "moe_combine", [&] {
            combine_kernel<scalar_t><<<tokens64, 128, 0, stream>>>(
                packed_output.data_ptr<scalar_t>(), inverse.data_ptr<int>(),
                weights.data_ptr<float>(), output.data_ptr<scalar_t>(),
                tokens64, hidden64, topk64);
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}
'''


def build_extension():
    os.environ.setdefault("MAX_JOBS", "2")
    # CUDA_HOME can be imported before a notebook adjusts PATH: detect it here too.
    import torch.utils.cpp_extension as cpp_extension
    if cpp_extension.CUDA_HOME is None:
        for candidate in [os.environ.get("CUDA_HOME"), "/usr/local/cuda"]:
            if candidate and (Path(candidate) / "bin/nvcc").exists():
                cpp_extension.CUDA_HOME = candidate
                break
    source_directory = Path(__file__).resolve().parent / "csrc"
    cpp = (source_directory / "bindings.cpp").read_text() if (source_directory / "bindings.cpp").exists() else CPP_SOURCE
    cuda = (source_directory / "dispatch.cu").read_text() if (source_directory / "dispatch.cu").exists() else CUDA_SOURCE
    digest = hashlib.sha256((cpp + cuda).encode()).hexdigest()[:10]
    print("Compiling C++/CUDA extension (first run only)...", flush=True)
    return load_inline(
        name=f"moe_dispatch_{digest}", cpp_sources=cpp,
        cuda_sources=cuda, functions=["dispatch_out", "combine_out"],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        with_cuda=True, verbose=False,
    )


class CudaWorkspace:
    def __init__(self, extension, tokens, hidden, experts, topk, dtype, vectorized=True):
        self.extension = extension
        self.vectorized = vectorized
        self.counts = torch.empty(experts, device="cuda", dtype=torch.int32)
        self.offsets = torch.empty(experts + 1, device="cuda", dtype=torch.int32)
        self.slots = torch.empty(tokens * topk, device="cuda", dtype=torch.int32)
        self.packed = torch.empty((tokens * topk, hidden), device="cuda", dtype=dtype)
        self.inverse = torch.empty((tokens, topk), device="cuda", dtype=torch.int32)
        self.output = torch.empty((tokens, hidden), device="cuda", dtype=dtype)

    def dispatch(self, x, ids32, ids64=None):
        self.extension.dispatch_out(x, ids32, self.counts, self.offsets, self.slots,
                                    self.packed, self.inverse, self.vectorized)

    def combine(self, expert_output, weights):
        self.extension.combine_out(expert_output, self.inverse, weights, self.output)

    @property
    def workspace_bytes(self):
        return sum(t.numel() * t.element_size() for t in [
            self.counts, self.offsets, self.slots, self.packed, self.inverse, self.output])


class TorchWorkspace:
    """GPU sorting reference with reusable outputs and FP32 weighted accumulation."""
    def __init__(self, tokens, hidden, experts, topk, dtype):
        n = tokens * topk
        self.tokens, self.topk, self.hidden = tokens, topk, hidden
        self.sorted_ids = torch.empty(n, device="cuda", dtype=torch.int32)
        self.order = torch.empty(n, device="cuda", dtype=torch.int64)
        self.token_ids = torch.empty(n, device="cuda", dtype=torch.int64)
        self.inverse = torch.empty((tokens, topk), device="cuda", dtype=torch.int64)
        self.arange = torch.arange(n, device="cuda", dtype=torch.int64)
        self.ones = torch.ones(n, device="cuda", dtype=torch.int32)
        self.counts = torch.empty(experts, device="cuda", dtype=torch.int32)
        self.offsets = torch.empty(experts + 1, device="cuda", dtype=torch.int32)
        self.packed = torch.empty((n, hidden), device="cuda", dtype=dtype)
        self.gathered = torch.empty((n, hidden), device="cuda", dtype=dtype)
        self.weighted = torch.empty((tokens, topk, hidden), device="cuda", dtype=torch.float32)
        self.output = torch.empty((tokens, hidden), device="cuda", dtype=dtype)
        self.accumulated = self.output if dtype == torch.float32 else torch.empty(
            (tokens, hidden), device="cuda", dtype=torch.float32)

    def dispatch(self, x, ids32, ids64):
        self.counts.zero_()
        self.counts.scatter_add_(0, ids64.reshape(-1), self.ones)
        self.offsets[0:1].zero_()
        torch.cumsum(self.counts, dim=0, dtype=torch.int32, out=self.offsets[1:])
        torch.sort(ids32.reshape(-1), out=(self.sorted_ids, self.order))
        torch.div(self.order, self.topk, rounding_mode="floor", out=self.token_ids)
        self.inverse.reshape(-1).scatter_(0, self.order, self.arange)
        torch.index_select(x, 0, self.token_ids, out=self.packed)

    def combine(self, expert_output, weights):
        torch.index_select(expert_output, 0, self.inverse.reshape(-1), out=self.gathered)
        torch.mul(self.gathered.view(self.tokens, self.topk, self.hidden),
                  weights.unsqueeze(-1), out=self.weighted)
        torch.sum(self.weighted, dim=1, out=self.accumulated)
        if self.output is not self.accumulated:
            self.output.copy_(self.accumulated)

    @property
    def workspace_bytes(self):
        values = [v for v in vars(self).values() if isinstance(v, torch.Tensor)]
        unique = {v.data_ptr(): v for v in values}
        return sum(v.numel() * v.element_size() for v in unique.values())


def make_case(tokens, hidden, experts, topk, dtype, pattern, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((tokens, hidden), device="cuda", dtype=dtype, generator=generator)
    # Router is common to both implementations and excluded from transport timings.
    logits = torch.randn((tokens, experts), device="cuda", generator=generator)
    if pattern == "skewed":
        logits[:, :topk] += 5.0
    elif pattern != "balanced":
        raise ValueError(f"unknown pattern: {pattern}")
    selected_logits, ids64 = torch.topk(logits, topk, dim=-1)
    weights = torch.softmax(selected_logits, dim=-1).contiguous()
    return x, ids64.to(torch.int32).contiguous(), ids64.contiguous(), weights


@torch.inference_mode()
def validate_case(workspace, x, ids32, ids64, weights, experts, seed):
    tokens, hidden = x.shape
    topk = ids32.shape[1]
    workspace.dispatch(x, ids32, ids64)
    expected_counts = torch.bincount(ids64.reshape(-1), minlength=experts).to(torch.int32)
    torch.testing.assert_close(workspace.counts, expected_counts, rtol=0, atol=0)
    torch.testing.assert_close(workspace.offsets[1:] - workspace.offsets[:-1],
                               expected_counts, rtol=0, atol=0)
    assert int(workspace.offsets[0]) == 0
    assert int(workspace.offsets[-1]) == tokens * topk
    inverse64 = workspace.inverse.reshape(-1).long()
    torch.testing.assert_close(torch.sort(inverse64).values,
                               torch.arange(tokens * topk, device="cuda"), rtol=0, atol=0)
    unpacked = workspace.packed.index_select(0, inverse64).view(tokens, topk, hidden)
    torch.testing.assert_close(unpacked, x[:, None, :].expand_as(unpacked), rtol=0, atol=0)
    # Every expert's packed slice must contain exactly its own assignments.
    packed_experts = torch.empty(tokens * topk, dtype=torch.int64, device="cuda")
    packed_experts.scatter_(0, inverse64, ids64.reshape(-1))
    expected_experts = torch.repeat_interleave(
        torch.arange(experts, device="cuda"), workspace.counts.long())
    torch.testing.assert_close(packed_experts, expected_experts, rtol=0, atol=0)
    # Arbitrary outputs prevent an identity-only combine test from hiding errors.
    generator = torch.Generator(device="cuda").manual_seed(seed + 501)
    canonical = torch.randn((tokens * topk, hidden), device="cuda",
                             dtype=x.dtype, generator=generator)
    expert_output = torch.empty_like(canonical)
    expert_output.index_copy_(0, inverse64, canonical)
    workspace.combine(expert_output, weights)
    expected = (canonical.view(tokens, topk, hidden).float()
                * weights[:, :, None]).sum(1).to(x.dtype)
    tolerance = 2e-3 if x.dtype == torch.float16 else 2e-5
    torch.testing.assert_close(workspace.output, expected, rtol=tolerance, atol=tolerance)
    difference = (workspace.output.float() - expected.float()).abs()
    return float(difference.max()) if difference.numel() else 0.0


@torch.inference_mode()
def run_tests(extension):
    shapes = [(0, 17, 8, 2), (1, 17, 8, 2), (7, 31, 8, 1),
              (33, 128, 8, 2), (129, 512, 16, 2), (257, 1024, 16, 2)]
    errors = []
    for dtype in [torch.float32, torch.float16]:
        for index, (tokens, hidden, experts, topk) in enumerate(shapes):
            pattern = "skewed" if index % 2 == 0 else "balanced"
            x, ids32, ids64, weights = make_case(
                tokens, hidden, experts, topk, dtype, pattern, 42 + index)
            implementations = [TorchWorkspace(tokens, hidden, experts, topk, dtype)]
            implementations += [CudaWorkspace(extension, tokens, hidden, experts,
                                topk, dtype, vectorized=v) for v in [False, True]]
            for workspace in implementations:
                errors.append(validate_case(workspace, x, ids32, ids64, weights,
                                             experts, 42 + index))
    torch.cuda.synchronize()
    summary = {"passed": len(errors), "failed": 0,
               "max_absolute_error": max(errors, default=0.0),
               "coverage": "FP32/FP16; K=1/2; empty batch; empty experts; skew; odd H"}
    print(f"Correctness: {summary['passed']} checks passed; "
          f"max absolute error {summary['max_absolute_error']:.3g}", flush=True)
    return summary


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


@torch.inference_mode()
def measure(function, warmup, repeats):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(repeats)]
    for start, end in events:
        start.record()
        function()
        end.record()
    torch.cuda.synchronize()
    values_us = [start.elapsed_time(end) * 1000.0 for start, end in events]
    return {"median_us": statistics.median(values_us),
            "p95_us": percentile(values_us, .95),
            "min_us": min(values_us), "samples_us": values_us}


@torch.inference_mode()
def benchmark(extension, suite, warmup, repeats):
    if suite == "smoke":
        cases = [(128, 512, 8, "balanced"), (2048, 1024, 16, "skewed")]
    elif suite == "quick":
        cases = [(t, h, e, p) for t in [128, 2048, 8192]
                 for h, e in [(512, 8), (1024, 16)]
                 for p in ["balanced", "skewed"]]
    else:
        cases = [(t, h, e, p) for t in [128, 512, 2048, 8192]
                 for h in [512, 1024] for e in [8, 16]
                 for p in ["balanced", "skewed"]]
    records, configurations = [], []
    for case_id, (tokens, hidden, experts, pattern) in enumerate(cases):
        x, ids32, ids64, weights = make_case(
            tokens, hidden, experts, 2, torch.float16, pattern, 100 + case_id)
        workspaces = {
            "PyTorch reference": TorchWorkspace(tokens, hidden, experts, 2, x.dtype),
            "CUDA scalar": CudaWorkspace(extension, tokens, hidden, experts, 2,
                                           x.dtype, vectorized=False),
            "CUDA vectorized": CudaWorkspace(extension, tokens, hidden, experts, 2,
                                               x.dtype, vectorized=True),
        }
        configurations.append({"case_id": case_id, "tokens": tokens, "hidden": hidden,
                                "experts": experts, "topk": 2, "pattern": pattern,
                                "dtype": "float16", "expert_counts":
                                torch.bincount(ids64.reshape(-1), minlength=experts).cpu().tolist()})
        # Every configuration is validated before any result is emitted.
        for workspace in workspaces.values():
            validate_case(workspace, x, ids32, ids64, weights, experts, 100 + case_id)
        # Rotate order between cases to reduce systematic warm-clock/order bias.
        names = list(workspaces)
        rotation = case_id % len(names)
        names = names[rotation:] + names[:rotation]
        for name in names:
            workspace = workspaces[name]
            workspace.dispatch(x, ids32, ids64)

            def dispatch():
                workspace.dispatch(x, ids32, ids64)

            def combine():
                workspace.combine(workspace.packed, weights)

            def transport():
                workspace.dispatch(x, ids32, ids64)
                workspace.combine(workspace.packed, weights)

            for stage, function in [("dispatch", dispatch), ("combine", combine),
                                    ("dispatch+combine", transport)]:
                timing = measure(function, warmup, repeats)
                records.append({"case_id": case_id, "tokens": tokens, "hidden": hidden,
                                "experts": experts, "topk": 2, "pattern": pattern,
                                "dtype": "float16", "implementation": name, "stage": stage,
                                "workspace_bytes": workspace.workspace_bytes, **timing})
        print(f"Benchmarked {case_id + 1}/{len(cases)}: T={tokens}, H={hidden}, "
              f"E={experts}, {pattern}", flush=True)
    return configurations, records


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
    w1 = torch.randn(experts, hidden, intermediate, device="cuda",
                     generator=generator) / math.sqrt(hidden)
    w2 = torch.randn(experts, intermediate, hidden, device="cuda",
                     generator=generator) / math.sqrt(intermediate)

    def run(workspace):
        workspace.dispatch(x, ids32, ids64)
        expert_output = torch.empty_like(workspace.packed)
        # These shared host offsets are acceptable for the functional demonstration.
        # The block is not included in any latency/speedup benchmark.
        boundaries = workspace.offsets.cpu().tolist()
        for expert in range(experts):
            first, last = boundaries[expert:expert + 2]
            if first < last:
                expert_output[first:last] = torch.nn.functional.silu(
                    workspace.packed[first:last] @ w1[expert]) @ w2[expert]
        workspace.combine(expert_output, weights)
        return workspace.output.clone()

    reference = run(TorchWorkspace(tokens, hidden, experts, topk, x.dtype))
    actual = run(CudaWorkspace(extension, tokens, hidden, experts, topk, x.dtype))
    torch.testing.assert_close(actual, reference, atol=2e-4, rtol=2e-4)
    # Independent token-major expert evaluation provides a second oracle.
    expected = torch.zeros_like(x)
    for expert in range(experts):
        expert_result = torch.nn.functional.silu(x @ w1[expert]) @ w2[expert]
        gate_weight = (weights * (ids64 == expert)).sum(dim=1)
        expected += expert_result * gate_weight[:, None]
    torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-4)
    result = {"passed": True, "tokens": tokens, "hidden": hidden, "experts": experts,
              "topk": topk, "output_shape": list(actual.shape),
              "max_absolute_error": float((actual - expected).abs().max()),
              "benchmark_note": "functional MoE block only; no block/model speedup claimed"}
    print("MoE block demo: passed against grouped and independent token-major references.", flush=True)
    return result


def hardware_metadata():
    properties = torch.cuda.get_device_properties(0)
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"], text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        smi = "unavailable"
    return {"gpu": properties.name, "compute_capability":
            f"{properties.major}.{properties.minor}", "gpu_memory_bytes": properties.total_memory,
            "torch_version": torch.__version__, "torch_cuda": torch.version.cuda,
            "python": platform.python_version(), "nvidia_smi": smi,
            "utc_run_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def write_report(result, destination):
    records, configurations = result["benchmarks"], result["configurations"]
    rows, panels = [], []
    for case in configurations:
        measurements = {r["implementation"]: r for r in records
                        if r["case_id"] == case["case_id"] and r["stage"] == "dispatch+combine"}
        baseline = measurements["PyTorch reference"]["median_us"]
        vector = measurements["CUDA vectorized"]["median_us"]
        scalar = measurements["CUDA scalar"]["median_us"]
        speedup = baseline / vector
        title = f"T={case['tokens']:,} · H={case['hidden']} · E={case['experts']} · {case['pattern']}"
        bars = []
        largest = max(m["median_us"] for m in measurements.values())
        for index, name in enumerate(["PyTorch reference", "CUDA scalar", "CUDA vectorized"]):
            m = measurements[name]
            width = 100 * m["median_us"] / largest
            bars.append(f'<div class="bar-row"><span>{html.escape(name)}</span>'
                        f'<div class="track"><div class="bar b{index}" style="width:{width:.2f}%"></div></div>'
                        f'<strong>{m["median_us"]:.2f} µs</strong></div>')
        counts = case["expert_counts"]
        maximum_count = max(counts, default=1) or 1
        load_bars = ''.join(f'<div class="load-row"><span>E{e}</span><div class="track">'
                            f'<div class="bar load" style="width:{100*c/maximum_count:.2f}%"></div>'
                            f'</div><strong>{c:,}</strong></div>' for e, c in enumerate(counts))
        panels.append(f'<section class="case"><h2>{html.escape(title)}</h2>'
                      f'<p class="metric">{speedup:.2f}× measured speed ratio '
                      '(PyTorch median / vectorized CUDA median)</p>'
                      + ''.join(bars) + '<details><summary>Expert assignment counts</summary>'
                      + load_bars + '</details></section>')
        rows.append(f'<tr><td>{case["tokens"]:,}</td><td>{case["hidden"]}</td>'
                    f'<td>{case["experts"]}</td><td>{case["pattern"]}</td>'
                    f'<td>{baseline:.2f}</td><td>{scalar:.2f}</td><td>{vector:.2f}</td>'
                    f'<td>{speedup:.2f}×</td></tr>')
    metadata = html.escape(json.dumps(result["hardware"], indent=2))
    checks = result["correctness"]["passed"]
    document = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CUDA MoE Dispatch — GPU Benchmark</title><style>
body{font:15px/1.55 system-ui,sans-serif;background:#f3f5fa;color:#14213b;margin:0}
main{max-width:1060px;margin:36px auto;padding:0 22px}h1{font-size:34px;line-height:1.15}
.eyebrow{font-size:12px;text-transform:uppercase;letter-spacing:2px;color:#3150a1}
.lead{max-width:860px;color:#47546d}.pill{display:inline-block;padding:7px 13px;
border:1px solid #9bb0da;border-radius:18px;background:white;margin:0 8px 8px 0}
.case,.summary{background:white;padding:22px;border:1px solid #d7deed;border-radius:13px;margin:20px 0}
h2{font-size:19px;margin:0 0 8px}.metric{font-weight:650;color:#263c79}
.bar-row{display:grid;grid-template-columns:160px 1fr 105px;align-items:center;gap:12px;margin:9px 0}
.track{background:#edf0f7;height:17px;border-radius:4px;overflow:hidden}.bar{height:100%;min-width:1px}
.b0{background:#71809b}.b1{background:#8a64c5}.b2{background:#3261c7}.load{background:#3261c7}
.load-row{display:grid;grid-template-columns:38px 1fr 70px;gap:10px;margin:5px 0;align-items:center}
details{margin-top:18px}summary{cursor:pointer}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:10px;border-bottom:1px solid #e2e6ee}th{background:#edf1f8}
.scroll{overflow:auto}pre{white-space:pre-wrap;background:#edf1f8;padding:16px;border-radius:7px}
.note{font-size:13px;color:#47546d}.grid{display:grid;grid-template-columns:1fr 1fr;gap:15px}
@media(max-width:700px){.grid{grid-template-columns:1fr}.bar-row{grid-template-columns:120px 1fr 75px;font-size:12px}h1{font-size:27px}}
</style></head><body><main><p class="eyebrow">CUDA / C++ / PyTorch</p>
<h1>MoE Token Dispatch</h1><p class="lead">Measured token packing and weighted output combination on one GPU.
CUDA kernels count expert assignments, calculate offsets, pack token rows, and combine outputs with FP32 accumulation.</p>
'''
    document += f'<span class="pill">{html.escape(result["hardware"]["gpu"])}</span>'
    document += f'<span class="pill">{checks} edge-case checks passed</span>'
    document += f'<span class="pill">{len(configurations)} benchmark configurations</span>'
    document += '<span class="pill">MoE feedforward block verified</span>'
    document += '<div class="summary"><h2>Measurement scope</h2><p class="note">'
    document += html.escape(result["method"]["scope"]) + '</p><p class="note">'
    document += html.escape(result["method"]["timing"]) + '</p></div>'
    document += '<div class="grid">' + ''.join(panels) + '</div>'
    document += '<section class="summary"><h2>Exact transport medians (µs)</h2><div class="scroll"><table>'
    document += '<thead><tr><th>Tokens</th><th>Hidden</th><th>Experts</th><th>Routing</th>'
    document += '<th>PyTorch</th><th>CUDA scalar</th><th>CUDA vectorized</th><th>Ratio</th></tr></thead>'
    document += '<tbody>' + ''.join(rows) + '</tbody></table></div></section>'
    document += '<section class="summary"><h2>Hardware and software</h2><pre>' + metadata + '</pre></section>'
    document += '</main></body></html>'
    destination.write_text(document, encoding="utf-8")


def write_results(result, directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    records = result["benchmarks"]
    if records:
        fields = [key for key in records[0] if key != "samples_us"]
        with (directory / "results.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
    write_report(result, directory / "report.html")
    ratios = []
    for case in result["configurations"]:
        matching = {r["implementation"]: r["median_us"] for r in records
                    if r["case_id"] == case["case_id"] and r["stage"] == "dispatch+combine"}
        ratios.append(matching["PyTorch reference"] / matching["CUDA vectorized"])
    if ratios:
        print(f"Transport speed ratio across configurations: median {statistics.median(ratios):.2f}x; "
              f"range {min(ratios):.2f}x–{max(ratios):.2f}x", flush=True)
    print(f"Saved {directory}/results.json, results.csv, report.html", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["smoke", "quick", "full"], default="quick")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Select an NVIDIA GPU runtime before running this script.")
    if args.repeats < 5 or args.warmup < 1:
        raise ValueError("use repeats>=5 and warmup>=1")
    torch.cuda.set_device(0)
    torch.set_grad_enabled(False)
    hardware = hardware_metadata()
    print(f"GPU: {hardware['gpu']} | PyTorch {hardware['torch_version']} | "
          f"CUDA {hardware['torch_cuda']}", flush=True)
    extension = build_extension()
    correctness = run_tests(extension)
    demo = block_demo(extension)
    configurations, records = benchmark(extension, args.suite, args.warmup, args.repeats)
    result = {"hardware": hardware, "correctness": correctness, "block_demo": demo,
              "method": {"suite": args.suite, "warmup": args.warmup, "repeats": args.repeats,
                         "seed": 42, "timing": "CUDA-event GPU stream elapsed time; per-call samples after warmup. "
                         "Both paths reuse workspaces; count reset is timed. Compilation, input generation, "
                         "router, validation, and expert computation are excluded. Small cases include launch gaps.",
                         "scope": "Dispatch/combine transport only, FP16 inputs with FP32 weighted accumulation. "
                         "The reference uses GPU sorting, indexing, multiplication and reduction; CUDA uses "
                         "atomic slot assignment, a tiny expert scan, row packing and fused weighted gather. "
                         "The separate MoE block checks forward correctness; no model/block speedup is claimed. "
                         "Workspace bytes are explicit persistent buffers, not peak CUDA allocator usage. "
                         "No comparison to NVIDIA/other production MoE libraries has been run."},
              "configurations": configurations, "benchmarks": records}
    write_results(result, args.output)


if __name__ == "__main__":
    main()
