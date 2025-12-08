#include <torch/library.h>

#include "kernels.h"
#include "utils.h"

TORCH_LIBRARY_WRAPPER(TORCH_LIBRARY_NAME, ops) {
  ops.def("hello_gpu", &hello_gpu);
  ops.impl("hello_gpu", torch::kCUDA, &hello_gpu);
}

REGISTER_EXTENSION(TORCH_LIBRARY_NAME)
