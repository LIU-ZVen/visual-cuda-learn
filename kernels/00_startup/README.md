# Start Up A CUDA Kernel and Python Binding

## 1. Brief

这个例子中包含了一个最小的 cuda kernel 使用 pybind11 进行绑定并在 python 中进行调用的形式, 后续可在此基础上按需扩展

## 2. misc

简单记录下几个命令吧

### 2.1. 使用对应的卡来进行编译, 防止 nvcc 报错

增加环境变量: `TORCH_CUDA_ARCH_LIST="7.5"`, Tesla T4 属于 "7.5" 的计算能力

### 2.2. nsys 相关

* torch profiler - 查看哪个算子耗时
* nsys profile - 查看系统性能(包括 H2D, D2H, 比较全局的信息)
* nsys compute - 查看 cuda kernel 内部的性能

有一篇讲解 nsys 比较好的文章:
https://zhuanlan.zhihu.com/p/1945304372545291742, 查看 Nsight System 章节

feat: add zven_kernels -> just personal notes