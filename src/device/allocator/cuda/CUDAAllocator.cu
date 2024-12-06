#include "CUDAAllocator.h"
#include <cuda_runtime.h>
#include <iostream>

CUDAAllocator::CUDAAllocator(){ }

void* CUDAAllocator::allocate(size_t size) {
    if (size == 0) {
        std::cerr << "Invalid allocation size: 0 bytes." << std::endl;
        exit(-1);
    }

    cudaError_t lastErr = cudaGetLastError();
    if (lastErr != cudaSuccess) {
        std::cerr << "Previous CUDA error: " << cudaGetErrorString(lastErr) << std::endl;
        cudaDeviceReset();  // 重置设备状态
        exit(-1);
    }

    void* devPtr;
    cudaError_t err = cudaMalloc((void **)&devPtr, size);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed! Error: " << cudaGetErrorString(err) << std::endl;
        // Handle the error (e.g., return, exit, or clean up resources)
        exit(-1);
    }
    return devPtr;
}

void CUDAAllocator::deallocate(void* ptr) {
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
        std::cerr << "cudaFree failed! Error: " << cudaGetErrorString(err) << std::endl;
        // 根据需求处理错误，例如退出程序或记录日志
        exit(EXIT_FAILURE);
    }
    ptr = nullptr;
}