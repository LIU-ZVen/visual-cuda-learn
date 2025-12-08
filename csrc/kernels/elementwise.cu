#include <cuda_runtime.h>
#include <torch/extension.h>

#include "kernels.h"

#define WARP_SIZE 32 // 32 threads per warp

// FLOAT4 宏：用于向量化内存访问（128-bit aligned load/store）
// float4 类型定义在 <cuda_runtime.h> 中，包含该头文件即可使用
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

/**
 * SM = streaming multiprocessor
 * L2 chache - 所有的 SM 共享
 * L1 chache - SM 内部的所有 warp 共享
 * shared memory - warp 内部的所有 thread 共享
 * local memory - thread 私有的 mem
 */
/**
 * ================================ Warp 介绍
 * ======================================= 查看 GPU 完整信息: nvidia-smi -q,
 * 里面有频率/各层级内存大小/甚至温度, 但是没找到 thread 信息 torch 接口:
 * @code
 * import torch; torch.cuda.get_device_properties(0)    # 0 is device id
 * # output is:
 * # _CudaDeviceProperties(
 * #    name='Tesla T4',
 * #    major=7,
 * #    minor=5,
 * #    total_memory=15948MB,
 * #    multi_processor_count=40,
 * #    uuid=bb20e7f3-6cb7-fbd0-716e-002122814eb4,
 * #    pci_bus_id=0,
 * #    pci_device_id=8,
 * #    pci_domain_id=0,
 * #    L2_cache_size=4MB
 * # )
 * @endcode
 * ================================================================================
 * Per SM 内部的资源 (和 capability 有关, 这里以 Tesla T4 为例, capability 7.5):
 * | Resource       | Amount | Comments           |
 * |----------------|--------|--------------------|
 * | MAX_BLOCK_NUM  | 16     | block_num <= 16    |
 * | WARP_NUM       | 32     |                    |
 * | THREAD_NUM     | 1024   | 32 thread per warp |
 * | 32-bit REG_NUM | 65536  |                    |
 * ================================================================================
 * Per GPU
 * SM 数量: 40, 即上面 torch 接口中显示的 `multi_processor_count=40`
 * ================================================================================
 */

/**
 * 最简单的 elementwise add kernel
 * 每个线程处理一个元素
 */
__global__ void elementwise_add_kernel(const float *a, const float *b, float *c,
                                       int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

/**
 * Elementwise add kernel with FLOAT4 vectorization
 * 每个线程处理 4 个 float（128-bit aligned）
 * 注意：需要确保内存地址是 16 字节对齐的（float4 要求）
 */
__global__ void elementwise_add_f32x4_kernel(float *a, float *b, float *c,
                                             int n) {
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  // 边界检查：确保 idx + 3 < n，即至少能处理 4 个元素
  if (idx + 3 < n) {
    float4 reg_a = FLOAT4(a[idx]);
    float4 reg_b = FLOAT4(b[idx]);
    float4 reg_c;
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    reg_c.z = reg_a.z + reg_b.z;
    reg_c.w = reg_a.w + reg_b.w;
    FLOAT4(c[idx]) = reg_c;
  } else {
    // 处理剩余的不对齐元素（fallback to scalar）
    for (int i = idx; i < n; i++) {
      c[i] = a[i] + b[i];
    }
  }
}

/**
 * Elementwise add 函数
 * 输入两个 torch tensor，输出它们的和
 */
torch::Tensor elementwise_add(const torch::Tensor &a, const torch::Tensor &b) {
  // 检查输入
  TORCH_CHECK(a.device().is_cuda(), "Input tensor a must be on CUDA");
  TORCH_CHECK(b.device().is_cuda(), "Input tensor b must be on CUDA");
  TORCH_CHECK(a.dtype() == torch::kFloat32, "Input tensor a must be float32");
  TORCH_CHECK(b.dtype() == torch::kFloat32, "Input tensor b must be float32");
  TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same shape");

  // 创建输出 tensor
  auto c = torch::empty_like(a);

  // 获取元素数量
  int n = a.numel();

  // 配置 kernel 启动参数
  // 使用 256 个线程的 block（一个 warp 是 32，256 = 8 warps）
  int threads_per_block = 256;
  int num_blocks = (n + threads_per_block - 1) / threads_per_block;

  // 启动 kernel
  elementwise_add_kernel<<<num_blocks, threads_per_block>>>(
      a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);

  return c;
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("elementwise_add", &elementwise_add, "Elementwise add of two
//   tensors");
// }
