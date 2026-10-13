
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
