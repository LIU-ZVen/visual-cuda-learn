import os
import time

import torch
from torch.utils.cpp_extension import load

# torch.set_grad_enabled(False)

lib = load(
    name="elementwise_lib",
    sources=[
        "elementwise.cu",
    ],
    extra_cuda_cflags=[
        "-O3",
    ],
    extra_cflags=["-std=c++17"],
)


def elementwise_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Elementwise add of two CUDA tensors.

    Args:
        a: First input tensor (float32, CUDA)
        b: Second input tensor (float32, CUDA)

    Returns:
        Elementwise sum of a and b
    """
    return lib.elementwise_add(a, b)


def main(warmup: int = 5) -> None:
    """简单的正确性测试函数（轻量）"""
    # 创建两个 shape 为 8 的 tensor
    a = torch.randn(8, device="cuda:0", dtype=torch.float32)
    b = torch.randn(8, device="cuda:0", dtype=torch.float32)

    # 预热，主要是让 CUDA runtime / JIT 等都就绪
    for _ in range(warmup):
        _ = elementwise_add(a, b)
    torch.cuda.synchronize()

    # 调用我们的 kernel
    c = elementwise_add(a, b)
    torch.cuda.synchronize()

    # 验证结果（与 PyTorch 原生实现对比）
    c_ref = a + b
    print(f"Reference (PyTorch): {c_ref}")
    print(f"Max difference: {(c - c_ref).abs().max().item()}")


def main_for_nsys(
    n: int = 1280 * 256,
    warmup: int = 10,
    iters: int = 10000,
) -> None:
    """专门给 nsys 用的主函数：大量迭代调用 kernel。

    - 不使用 torch.profiler，只保留最干净的 CUDA 时间线
    - 通过 warmup + iters 让 kernel 运行时间足够长，便于分析
    """
    device = "cuda:0"
    dtype = torch.float32

    a = torch.randn(n, dtype=dtype).to(device)
    b = torch.randn(n, dtype=dtype).to(device)

    # 预热：让 context / JIT / cache 等都稳定下来
    for _ in range(warmup):
        _ = elementwise_add(a, b)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        c = elementwise_add(a, b)
    torch.cuda.synchronize()
    end = time.time()

    # 简单打印一下总时间，方便 sanity check
    print(
        f"[nsys] n={n}, warmup={warmup}, iters={iters}, ",
        f"total_time={(end - start)*1000:.3f} ms, ",
        f"each iter time={(end - start)/iters*1000000:.3f} us",
    )
    # 防止编译器 / Python 把 c 优化掉（即使用）
    print(f"[nsys] last elem: {c[-1].item()}")


if __name__ == "__main__":
    torch.manual_seed(3407)
    # 默认跑正确性测试；你可以手动改成 main_for_nsys()
    main_for_nsys(n=320 * 256, warmup=5, iters=10000)
