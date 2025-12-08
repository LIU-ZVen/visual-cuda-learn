#include <stdio.h>

#include "kernels.h"

__global__ void hello_gpu_kernel() { printf("Hello from GPU\n"); }

void hello_gpu() { hello_gpu_kernel<<<1, 1>>>(); }
