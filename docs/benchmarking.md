# Benchmark methodology

## Recorded transport checkpoint

`benchmarks/a100` preserves the exact source, raw JSON samples, CSV, HTML, and hashes
from the first successful run. It used an A100-SXM4-40GB, PyTorch 2.11.0+cu130,
Python 3.13.15, ten warmups, and forty measurements per stage and configuration.
The 12 configurations cover T=128/2048/8192, two H/E pairs, and balanced/skewed routing.
BF16/aligned dispatch, fused routing, and fused SwiGLU were added after this checkpoint.

The aggregate 6.36× speedup is the median of twelve per-configuration ratios:
`PyTorch median transport time / vectorized CUDA median transport time`.
It is not a ratio of pooled samples and is not a full-model inference result.

## Recorded full-block checkpoint

`benchmarks/a100-block` contains the completed v0.2.0 quick and smoke runs,
original HTML/CSV reports, profiler trace, complete log, and evidence hashes.
Quick: 8 configurations, 24 passing oracle comparisons, median forward ratio
3.1677670384×, range 1.4379999383–3.3491803216×. `cuda_fused` is compared to
`torch`; all variants use the same Triton expert backend. Warmup=10, repeats=40.

The 22 package source hashes in both JSON files match this checkout exactly.
`python scripts/verify_release.py` verifies raw samples, ratios, coverage,
source hashes, transport archive, and GPU kernel events without needing a GPU.
The optional A100 execution is archived under `graphs/` and `model/`, with its
separate `optional_run.log`. Graphs: 24 successful replay checks, 1.74× median
graph-to-graph ratio, and 3.08× eager ratio in that rerun. Larger preset: 6 oracle
comparisons passed, 1.06× median eager ratio. These runs share the same 22 source
hashes. Graph replay for the larger preset was not requested.

## Full-block benchmark

The benchmark uses shared weights, identical FP32 router policy, identical BF16
expert backend, identical activation math, and identical output precision across
`torch`, `cuda`, and `cuda_fused` variants. It validates each against independent
token-major expert evaluation before recording any measurements.

`cuda_fused` replaces bias/top-k/selected-score softmax and SwiGLU with CUDA kernels;
the FP32 router projection and grouped-GEMM backend remain common. Finite scores
are required, and the custom router breaks exact ties by the lower expert ID.

`forward` includes routing, dispatch, both expert matmuls, activation, and combine.
The stage measurements reuse prepared intermediate inputs. Stage medians are
independent measurements and need not add up to the `forward` median.

`smoke`: two small configurations. `quick`: eight configurations. `model`: two
larger model-shaped configurations. Synthetic inputs/weights are seeded; model
weights are not downloaded and no task-quality accuracy claim is made.

## Timing limits

CUDA events measure GPU stream elapsed time. Eager execution can include gaps
while Python/driver submits kernels, particularly for tiny workloads. The number
is useful for the measured operator path, but does not isolate device-only kernel
execution or establish hardware throughput. Kernel fusion can reduce those gaps.
Optional graph replay is validated and recorded as `forward_graph`, separately
from eager forward. Graph capture failures are reported rather than converted
into successful measurements.

Persistent workspace bytes count explicit reusable buffers, not peak allocator
usage, backend scratch space, or shared weights. The full-block GEMM backend is
recorded. No comparison to Megatron/Transformer Engine/other production MoE stacks
is implied by a ratio against the included PyTorch reference.

## Reproduction

```bash
moe-dispatch check-env
moe-dispatch block-benchmark --suite smoke --output runs/smoke
moe-dispatch block-benchmark --suite quick --expert-backend triton --output runs/quick
moe-dispatch profile-block --expert-backend triton --output runs/trace.json
```

If the upgraded extension needs diagnosis, set `MOE_DISPATCH_VERBOSE_BUILD=1`.
Keep the entire first error and GPU/PyTorch/toolkit versions. Avoid upgrading a
working CUDA PyTorch installation to fix an unrelated source error.
