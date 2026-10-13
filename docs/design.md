# Design

The public block is a forward-only MoE feedforward module. It uses a common router
and expert weights while switching transport and activation implementations.

## Routing and packing

Inputs have shape `[T,H]`. The router returns `[T,K]` expert IDs and normalized
selected routing scores. Each `(token, selected-expert)` pair receives a local
slot through an atomic increment. A small GPU scan calculates expert offsets.
Packing writes the token row into its expert group and records an inverse map.
The local slot order can vary; the inverse map preserves token identity.

The fused router uses one warp per token for bias, top-1/top-2 selection, and
selected-score softmax. It expects finite FP32 logits and supports up to 64 experts.
Ties choose the lower expert ID; PyTorch's tie order is unspecified. Benchmarks
use seeded continuous scores and correctness checks compare the independent
PyTorch router as well as expert outputs. The router projection remains a matmul.

Compact packing uses exactly `T*K` rows. Full-block packing rounds each expert's
count to an alignment and uses a fixed upper-bound capacity:

`ceil((T*K + E*(alignment-1) + 1) / alignment) * alignment`.

The extra capacity keeps the final dynamic offset below the physical row count.
Padding inside used groups is zeroed on GPU. Extra rows beyond the final offset
are ignored; combine reads only routed rows, and fused activation clears unused rows.
There is no token dropping or per-expert overflow capacity.

## Expert compute

Expert gate/up weights have shape `[E,2I,H]`; down weights have shape `[E,H,I]`.
Transpose views expose logical grouped-matmul shapes. Each backend consumes
device-resident end offsets. The Triton adapter schedules tiles across experts
using a fixed worker grid and masks partial M/N/K tiles. Accumulation is FP32;
matmul outputs are BF16.

SwiGLU evaluates `silu(gate) * up` in FP32 and casts once to the model dtype.
The fused CUDA version performs those operations in one kernel. All benchmark
variants use the same formula and expert backend.

## Buffers and API

`CudaWorkspace` owns counts, offsets, local slots, packed rows, inverse maps,
and the output tensor. A block adds projection/activation/expert-output buffers
and caches them by input token count, dtype, and device. Workspaces are sequential
resources; use separate instances for independent concurrent calls. Returned
tensors are views and are overwritten by the next call.

The convenience `dispatch` API checks IDs and may synchronize for bounds checks.
The workspace methods assume IDs are already valid and avoid those host checks
inside the timed transport loop. Device/shape/dtype checks remain in the C++ binding.

Full-block CUDA inference uses BF16. Standalone transport also supports FP16 and
FP32, top-1/top-2, up to 64 experts, and arbitrary positive hidden widths. Vector
packing falls back to scalar copying for widths not divisible by four.
