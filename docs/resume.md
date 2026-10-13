# Resume evidence

Project title: **CUDA MoE Dispatch Engine**

Suggested three bullets:

- Built a single-GPU C++/CUDA MoE forward engine with fused top-k routing,
  aligned token dispatch, grouped BF16 expert GEMMs, fused SwiGLU, and FP32 combine.
- Measured 3.17× median eager forward speedup across 8 synthetic A100 block
  configurations versus a PyTorch variant sharing the Triton expert backend;
  passed all 24 independent-oracle comparisons.
- Optimized standalone token transport to 6.36× median dispatch/combine speedup
  and 72.7% lower persistent workspace versus a preallocated PyTorch reference;
  passed 39 GPU tests and 36 transport checks.

## Claim boundaries

The 3.17× number is the median of eight per-case `torch / cuda_fused` median
ratios. Range: 1.44–3.35×. It includes router, dispatch, grouped expert GEMMs,
activation, and combine. Inputs and weights are synthetic; all variants share
weights and the Triton expert backend. CUDA-event eager stream timings can include
CPU launch gaps. It is not complete language-model speedup or device-only throughput.

The 6.36× and 72.7% figures belong to the separate 12-case FP16 transport checkpoint.
They exclude router/expert computation. Workspace counts explicit reusable transport
buffers, not weights, backend temporary allocations, or peak GPU memory.

The additional A100 run supports 24 passing CUDA Graph replay checks and a
1.74× median graph-to-graph ratio across 8 cases. The larger preset (H=4096,
I=14336, E=8) passed 6 oracle comparisons and measured 1.06× median eager ratio.
Do not apply the original 3.17× result to that larger preset. No multi-GPU,
training, pretrained-model quality, or production MoE-library superiority claim
is supported. The grouped-GEMM adapter
follows the attributed NVIDIA/Triton tutorial scheduling pattern.

Evidence is in `benchmarks/a100-block` and `benchmarks/a100`; run
`python scripts/verify_release.py` to recompute the ratios and check source hashes.
