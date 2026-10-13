
#include <torch/extension.h>
void dispatch_out(torch::Tensor x, torch::Tensor ids, torch::Tensor counts,
    torch::Tensor offsets, torch::Tensor slots, torch::Tensor packed,
    torch::Tensor inverse, bool vectorized);
void combine_out(torch::Tensor packed_output, torch::Tensor inverse,
    torch::Tensor weights, torch::Tensor output);
void dispatch_aligned_out(torch::Tensor x, torch::Tensor ids, torch::Tensor counts,
    torch::Tensor offsets, torch::Tensor slots, torch::Tensor packed,
    torch::Tensor inverse, bool vectorized, int64_t alignment);
void swiglu_out(torch::Tensor projection, torch::Tensor valid_rows, torch::Tensor output);
void topk_route_out(torch::Tensor logits, torch::Tensor bias, torch::Tensor ids32,
    torch::Tensor ids64, torch::Tensor weights);
