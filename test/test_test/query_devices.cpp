#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    std::cout << "可用的 CUDA 显卡数量: " << deviceCount << std::endl;

    // 遍历每一个 GPU 并显示一些基本信息
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        std::cout << "设备 " << i << ": " << deviceProp.name << std::endl;
        std::cout << "  全局内存大小: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  多处理器数量: " << deviceProp.multiProcessorCount << std::endl;
    }
    
    return 0;
}
