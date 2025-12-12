#include <cuda_runtime.h>

#include <iostream>

#include "tools.h"

void get_cuda_info() {
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);

  std::cout << "CUDA Device Count: " << deviceCount << std::endl;

  for (int i = 0; i < deviceCount; ++i) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);

    std::cout << "\n===== Device " << i << " =====" << std::endl;
    std::cout << "Name: " << prop.name << std::endl;
    std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024)
              << " MB" << std::endl;
    std::cout << "SM Count: " << prop.multiProcessorCount << std::endl;
    std::cout << "Warp Size: " << prop.warpSize << std::endl;

    std::cout << "\n***** Per SM Resources: " << std::endl;
    std::cout << "Max Blocks per SM: " << prop.maxBlocksPerMultiProcessor
              << std::endl;
    std::cout << "Max Threads per SM: " << prop.maxThreadsPerMultiProcessor
              << std::endl;
    std::cout << "Shared Mem per SM: " << prop.sharedMemPerMultiprocessor / 1024
              << " KB" << std::endl;
    std::cout << "Regs per SM: " << prop.regsPerMultiprocessor << std::endl;

    std::cout << "\n***** Per Block Resources: " << std::endl;
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock
              << std::endl;
    std::cout << "Shared Mem per Block: " << prop.sharedMemPerBlock / 1024
              << " KB" << std::endl;
    std::cout << "Regs per Block: " << prop.regsPerBlock << std::endl;

    std::cout << "\nMax Grid Size: " << prop.maxGridSize[0] << " x "
              << prop.maxGridSize[1] << " x " << prop.maxGridSize[2]
              << std::endl;

    std::cout << "Max Threads Dim: " << prop.maxThreadsDim[0] << " x "
              << prop.maxThreadsDim[1] << " x " << prop.maxThreadsDim[2]
              << std::endl;

    std::cout << "Clock Rate: " << prop.clockRate / 1000 << " MHz" << std::endl;
    std::cout << "Memory Clock Rate: " << prop.memoryClockRate / 1000 << " MHz"
              << std::endl;
    std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bit"
              << std::endl;
    std::cout << "====================" << std::endl;
  }
}
