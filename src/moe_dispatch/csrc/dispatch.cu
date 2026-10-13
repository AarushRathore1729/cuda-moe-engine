
#include <torch/types.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <climits>
#include <cmath>
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
template <> struct Vector4<at::BFloat16> { using type = uint2; };

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
    int tokens, int hidden, int topk, int packed_rows) {
    int token = blockIdx.x;
    if (token >= tokens) return;
    for (int column = threadIdx.x; column < hidden; column += blockDim.x) {
        float total = 0.0f;
        for (int k = 0; k < topk; ++k) {
            int p = token * topk + k;
            int row = inverse[p];
            if (row >= 0 && row < packed_rows)
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
    TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
                "only FP32, FP16, and BF16 are supported");
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
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "moe_pack", [&] {
            // The public API supports FP32/FP16/BF16.
            if constexpr (std::is_same<scalar_t, float>::value ||
                          std::is_same<scalar_t, at::Half>::value ||
                          std::is_same<scalar_t, at::BFloat16>::value) {
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
                 packed_output.scalar_type() == at::kHalf ||
                 packed_output.scalar_type() == at::kBFloat16), "unsupported output dtype");
    for (const auto& t : {inverse, weights, output}) check_cuda_tensor(t, packed_output);
    TORCH_CHECK(inverse.scalar_type() == at::kInt && weights.scalar_type() == at::kFloat,
                "inverse must be int32; weights must be FP32");
    int64_t tokens64 = output.size(0), hidden64 = output.size(1), topk64 = inverse.size(1);
    TORCH_CHECK(topk64 == 1 || topk64 == 2, "topk must be 1 or 2");
    TORCH_CHECK(tokens64 * topk64 <= INT_MAX && hidden64 <= INT_MAX && hidden64 > 0,
                "invalid output shape");
    TORCH_CHECK(inverse.size(0) == tokens64 && weights.sizes() == inverse.sizes() &&
                packed_output.size(0) >= tokens64 * topk64 &&
                packed_output.size(1) == hidden64 &&
                output.scalar_type() == packed_output.scalar_type(), "shape/dtype mismatch");
    c10::cuda::CUDAGuard guard(packed_output.device());
    if (tokens64 > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, packed_output.scalar_type(), "moe_combine", [&] {
            combine_kernel<scalar_t><<<tokens64, 128, 0, stream>>>(
                packed_output.data_ptr<scalar_t>(), inverse.data_ptr<int>(),
                weights.data_ptr<float>(), output.data_ptr<scalar_t>(),
                tokens64, hidden64, topk64, packed_output.size(0));
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

__global__ void aligned_offsets_kernel(const int* counts, int* offsets,
                                       int experts, int alignment) {
    if (threadIdx.x == 0) {
        int total = 0;
        offsets[0] = 0;
        for (int e = 0; e < experts; ++e) {
            int padded = ((counts[e] + alignment - 1) / alignment) * alignment;
            total += padded;
            offsets[e + 1] = total;
        }
    }
}

template <typename scalar_t>
__global__ void zero_padding_kernel(scalar_t* packed, const int* counts,
    const int* offsets, int hidden) {
    int e = blockIdx.x;
    int start = offsets[e] + counts[e];
    int end = offsets[e + 1];
    int64_t elements = static_cast<int64_t>(end - start) * hidden;
    for (int64_t i = threadIdx.x; i < elements; i += blockDim.x)
        packed[static_cast<int64_t>(start) * hidden + i] = scalar_t(0.0f);
}

void dispatch_aligned_out(torch::Tensor x, torch::Tensor ids, torch::Tensor counts,
    torch::Tensor offsets, torch::Tensor slots, torch::Tensor packed,
    torch::Tensor inverse, bool vectorized, int64_t alignment) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "x must be contiguous CUDA");
    TORCH_CHECK(x.dim() == 2 && ids.dim() == 2, "expected x[T,H], ids[T,K]");
    TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf ||
                x.scalar_type() == at::kBFloat16, "unsupported dtype");
    TORCH_CHECK(alignment >= 1 && alignment <= 128, "alignment must be 1..128");
    int64_t t64 = x.size(0), h64 = x.size(1), k64 = ids.size(1);
    int64_t e64 = counts.numel(), n64 = t64 * k64;
    TORCH_CHECK(k64 == 1 || k64 == 2, "topk must be 1 or 2");
    TORCH_CHECK(h64 > 0 && e64 > 0 && e64 <= 64 && ids.size(0) == t64,
                "invalid shape");
    int64_t required_rows = ((n64 + e64 * (alignment - 1) + 1 + alignment - 1)
                             / alignment) * alignment;
    TORCH_CHECK(n64 <= INT_MAX && h64 <= INT_MAX && required_rows <= INT_MAX,
                "shape exceeds int32 capacity");
    for (const auto& tensor : {ids, counts, offsets, slots, packed, inverse})
        check_cuda_tensor(tensor, x);
    TORCH_CHECK(ids.scalar_type() == at::kInt && counts.scalar_type() == at::kInt &&
                offsets.scalar_type() == at::kInt && slots.scalar_type() == at::kInt &&
                inverse.scalar_type() == at::kInt, "index tensors must be int32");
    TORCH_CHECK(offsets.numel() == e64 + 1 && slots.numel() == n64 &&
                inverse.numel() == n64 && packed.dim() == 2 &&
                packed.size(0) >= required_rows && packed.size(1) == h64 &&
                packed.scalar_type() == x.scalar_type(), "incorrect aligned workspace");
    c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int n = n64, hidden = h64, topk = k64, experts = e64;
    C10_CUDA_CHECK(cudaMemsetAsync(counts.data_ptr<int>(), 0,
                                  experts * sizeof(int), stream));
    if (n > 0)
        count_slots_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
            ids.data_ptr<int>(), counts.data_ptr<int>(), slots.data_ptr<int>(), n, experts);
    aligned_offsets_kernel<<<1, 1, 0, stream>>>(
        counts.data_ptr<int>(), offsets.data_ptr<int>(), experts, alignment);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "aligned_pack", [&] {
        if constexpr (std::is_same<scalar_t, float>::value ||
                      std::is_same<scalar_t, at::Half>::value ||
                      std::is_same<scalar_t, at::BFloat16>::value) {
            zero_padding_kernel<scalar_t><<<experts, 128, 0, stream>>>(
                packed.data_ptr<scalar_t>(), counts.data_ptr<int>(), offsets.data_ptr<int>(), hidden);
            if (n > 0) {
                if (vectorized && hidden % 4 == 0)
                    pack_vector_kernel<scalar_t><<<n, 128, 0, stream>>>(
                        x.data_ptr<scalar_t>(), ids.data_ptr<int>(), slots.data_ptr<int>(),
                        offsets.data_ptr<int>(), packed.data_ptr<scalar_t>(), inverse.data_ptr<int>(),
                        n, hidden, topk, experts);
                else
                    pack_scalar_kernel<scalar_t><<<n, 128, 0, stream>>>(
                        x.data_ptr<scalar_t>(), ids.data_ptr<int>(), slots.data_ptr<int>(),
                        offsets.data_ptr<int>(), packed.data_ptr<scalar_t>(), inverse.data_ptr<int>(),
                        n, hidden, topk, experts);
            }
        }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
__global__ void swiglu_kernel(const scalar_t* projection, const int* valid_rows,
    scalar_t* output, int rows, int width) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t elements = static_cast<int64_t>(rows) * width;
    if (i >= elements) return;
    int row = i / width, column = i % width;
    if (row >= valid_rows[0]) {
        output[i] = scalar_t(0.0f);
        return;
    }
    int64_t base = static_cast<int64_t>(row) * width * 2 + column;
    float gate = static_cast<float>(projection[base]);
    float up = static_cast<float>(projection[base + width]);
    output[i] = scalar_t((gate / (1.0f + expf(-gate))) * up);
}

void swiglu_out(torch::Tensor projection, torch::Tensor valid_rows, torch::Tensor output) {
    TORCH_CHECK(projection.is_cuda() && projection.is_contiguous(),
                "projection must be contiguous CUDA");
    TORCH_CHECK(projection.dim() == 2 && output.dim() == 2 &&
                projection.size(0) == output.size(0) &&
                projection.size(1) == output.size(1) * 2,
                "expected projection[R,2I], output[R,I]");
    check_cuda_tensor(valid_rows, projection);
    check_cuda_tensor(output, projection);
    TORCH_CHECK(valid_rows.scalar_type() == at::kInt && valid_rows.numel() == 1,
                "valid_rows must be a single int32 CUDA value");
    TORCH_CHECK(output.scalar_type() == projection.scalar_type() &&
                (projection.scalar_type() == at::kFloat ||
                 projection.scalar_type() == at::kHalf ||
                 projection.scalar_type() == at::kBFloat16), "unsupported dtype");
    TORCH_CHECK(output.size(0) <= INT_MAX && output.size(1) <= INT_MAX,
                "shape exceeds index capacity");
    c10::cuda::CUDAGuard guard(projection.device());
    int64_t elements = output.numel();
    if (elements > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16,
            projection.scalar_type(), "swiglu", [&] {
                swiglu_kernel<scalar_t><<<(elements + 255) / 256, 256, 0, stream>>>(
                    projection.data_ptr<scalar_t>(), valid_rows.data_ptr<int>(),
                    output.data_ptr<scalar_t>(), output.size(0), output.size(1));
            });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

// Finite FP32 logits. One warp per token, at most two expert scores per lane.
// Ties use the lower expert index. Only the final selected-score softmax is needed.
__device__ __forceinline__ void warp_argmax(float& value, int& index) {
    for (int shift = 16; shift > 0; shift /= 2) {
        float other_value = __shfl_down_sync(0xffffffff, value, shift);
        int other_index = __shfl_down_sync(0xffffffff, index, shift);
        if (other_value > value || (other_value == value && other_index < index)) {
            value = other_value;
            index = other_index;
        }
    }
}

__global__ void topk_route_kernel(const float* logits, const float* bias,
    int* ids32, int64_t* ids64, float* weights, int experts, int topk) {
    int token = blockIdx.x, lane = threadIdx.x;
    int index0 = lane < experts ? lane : INT_MAX;
    int index1 = lane + 32 < experts ? lane + 32 : INT_MAX;
    float value0 = lane < experts ? logits[token * experts + lane] + bias[lane] : -INFINITY;
    float value1 = lane + 32 < experts
        ? logits[token * experts + lane + 32] + bias[lane + 32] : -INFINITY;
    bool choose1 = value1 > value0 || (value1 == value0 && index1 < index0);
    float best = choose1 ? value1 : value0;
    int best_index = choose1 ? index1 : index0;
    warp_argmax(best, best_index);
    float first_value = __shfl_sync(0xffffffff, best, 0);
    int first_index = __shfl_sync(0xffffffff, best_index, 0);
    if (topk == 1) {
        if (lane == 0) {
            ids32[token] = first_index;
            ids64[token] = first_index;
            weights[token] = 1.0f;
        }
        return;
    }
    if (index0 == first_index) { value0 = -INFINITY; index0 = INT_MAX; }
    if (index1 == first_index) { value1 = -INFINITY; index1 = INT_MAX; }
    choose1 = value1 > value0 || (value1 == value0 && index1 < index0);
    best = choose1 ? value1 : value0;
    best_index = choose1 ? index1 : index0;
    warp_argmax(best, best_index);
    if (lane == 0) {
        float relative = expf(best - first_value);
        float normalization = 1.0f / (1.0f + relative);
        ids32[token * 2] = first_index;
        ids32[token * 2 + 1] = best_index;
        ids64[token * 2] = first_index;
        ids64[token * 2 + 1] = best_index;
        weights[token * 2] = normalization;
        weights[token * 2 + 1] = relative * normalization;
    }
}

void topk_route_out(torch::Tensor logits, torch::Tensor bias, torch::Tensor ids32,
    torch::Tensor ids64, torch::Tensor weights) {
    TORCH_CHECK(logits.is_cuda() && logits.is_contiguous() && logits.dim() == 2 &&
                logits.scalar_type() == at::kFloat, "expected FP32 CUDA logits[T,E]");
    for (const auto& tensor : {bias, ids32, ids64, weights}) check_cuda_tensor(tensor, logits);
    int64_t tokens = logits.size(0), experts = logits.size(1);
    TORCH_CHECK(experts >= 1 && experts <= 64 && tokens <= INT_MAX / experts,
                "routing supports 1..64 experts and int32 indexing");
    TORCH_CHECK(bias.dim() == 1 && bias.numel() == experts && bias.scalar_type() == at::kFloat,
                "expected FP32 bias[E]");
    TORCH_CHECK(ids32.dim() == 2 && ids32.size(0) == tokens &&
                ids32.scalar_type() == at::kInt, "expected int32 ids[T,K]");
    int64_t topk = ids32.size(1);
    TORCH_CHECK((topk == 1 || topk == 2) && topk <= experts &&
                ids64.sizes() == ids32.sizes() && ids64.scalar_type() == at::kLong &&
                weights.sizes() == ids32.sizes() && weights.scalar_type() == at::kFloat,
                "expected top-1/top-2 int64 IDs and FP32 weights");
    c10::cuda::CUDAGuard guard(logits.device());
    if (tokens > 0) {
        topk_route_kernel<<<tokens, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
            logits.data_ptr<float>(), bias.data_ptr<float>(), ids32.data_ptr<int>(),
            ids64.data_ptr<int64_t>(), weights.data_ptr<float>(), experts, topk);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}
