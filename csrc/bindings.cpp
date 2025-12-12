#include <pybind11/pybind11.h>
#include <torch/library.h>

#include "kernels.h"
#include "tools.h"
#include "utils.h"

TORCH_LIBRARY_WRAPPER(TORCH_LIBRARY_NAME, ops) {
  ops.def("hello_gpu", &hello_gpu);
  ops.impl("hello_gpu", torch::kCUDA, &hello_gpu);
}

// TODO(LIU-ZVen): bind some tools like `get_cuda_info`

REGISTER_EXTENSION(TORCH_LIBRARY_NAME)
