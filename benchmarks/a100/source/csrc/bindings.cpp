
#include <torch/extension.h>
void dispatch_out(torch::Tensor x, torch::Tensor ids, torch::Tensor counts,
    torch::Tensor offsets, torch::Tensor slots, torch::Tensor packed,
    torch::Tensor inverse, bool vectorized);
void combine_out(torch::Tensor packed_output, torch::Tensor inverse,
    torch::Tensor weights, torch::Tensor output);
