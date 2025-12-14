#include <cuda_runtime.h>
#include <stdio.h>
#include <torch/extension.h>

#include "kernels.h"

__global__ void relu_kernel(float* input, float* output, int numel) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    output[idx] = fmax(0.0f, input[idx]);
  }
}

__global__ void relu_f32x4_kernel(float* input, float* output, int numel) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < numel) {
    float4 in = reinterpret_cast<float4*>(input)[idx];
    float4 out;
    out.x = fmax(0.0f, in.x);
    out.y = fmax(0.0f, in.y);
    out.z = fmax(0.0f, in.z);
    out.w = fmax(0.0f, in.w);
    reinterpret_cast<float4*>(output)[idx] = out;
  }
}

// NOTE(LIU Ziwen): 这里如果增加打印, 会发现每个 thread 都会打印一遍, block_size
// = 256 就会打印 256 遍; 所以 float4 和 half2 的意义是, 每个 thread
// 内做了更多的运算(但是因为数据类型的问题, 仅 load/save 一次数据, 数据量是普通
// float 的 4 倍和 half 的 2 倍), 减少了 global memory 的访问次数,
// 提升了性能(主要是降低了单位访存延迟).
__global__ void relu_f16x2_kernel(half* input, half* output, int numel) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  printf(
      "dispatch relu_f16x2_kernel: blockIdx.x=%d, blockDim.x=%d, "
      "threadIdx.x=%d, idx=%d\n",
      blockIdx.x, blockDim.x, threadIdx.x, idx);

  if (idx < numel) {
    half2 in = reinterpret_cast<half2*>(input)[idx];
    half2 out = reinterpret_cast<half2*>(output)[idx];
    out.x = __hmax(__float2half(0.0f), in.x);
    out.y = __hmax(__float2half(0.0f), in.y);
    reinterpret_cast<half2*>(output)[idx] = out;
  }
}

torch::Tensor relu(const torch::Tensor& input) {
  auto output = torch::empty_like(input);
  int numel = input.numel();
  int block_size = 256;
  int num_blocks = (numel + block_size - 1) / block_size;

  at::ScalarType dtype = input.scalar_type();

  switch (dtype) {
    case at::kFloat:
      relu_f32x4_kernel<<<num_blocks, block_size>>>(
          input.data_ptr<float>(), output.data_ptr<float>(), numel);
      break;

    case at::kHalf:
      relu_f16x2_kernel<<<num_blocks, block_size>>>(
          reinterpret_cast<half*>(input.data_ptr<at::Half>()),
          reinterpret_cast<half*>(output.data_ptr<at::Half>()), numel);
      break;

    default:
      TORCH_CHECK(false, "unsupported dtype");
  }

  return output;
}
