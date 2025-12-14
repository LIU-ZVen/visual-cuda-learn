#ifndef CSRC_INCLUDE_KERNELS_H_
#define CSRC_INCLUDE_KERNELS_H_

#include <torch/all.h>

void hello_gpu();

torch::Tensor elementwise_add(const torch::Tensor& a, const torch::Tensor& b);

torch::Tensor relu(const torch::Tensor& input);

#endif  // CSRC_INCLUDE_KERNELS_H_
