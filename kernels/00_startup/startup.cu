#include <stdio.h>
#include <torch/extension.h>

__global__ void hello_gpu_kernel() {
    printf("Hello from GPU\n");
}

void hello_gpu() {
    hello_gpu_kernel<<<1, 1>>>();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hello_gpu", &hello_gpu, "hello_gpu");
}
