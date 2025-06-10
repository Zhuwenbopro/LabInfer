#include "CUDAMemoryManager.h"
#include <cuda_runtime.h>
#include <iostream>

CUDAMemoryManager::CUDAMemoryManager(){ }

void* CUDAMemoryManager::allocate(size_t bytes) {
    if (bytes == 0) {
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
    cudaError_t err = cudaMalloc((void **)&devPtr, bytes);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed! Error: " << cudaGetErrorString(err) << std::endl;
        // Handle the error (e.g., return, exit, or clean up resources)
        exit(-1);
    }
    return devPtr;
}

void CUDAMemoryManager::deallocate(void* ptr) {
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
        std::cout << " free ptr: "<< ptr << std::endl;
        std::cerr << "cudaFree failed! Error: " << cudaGetErrorString(err) << std::endl;
        std::cerr << "CUDA Error Code: " << err << std::endl;
        exit(EXIT_FAILURE);
    }
    ptr = nullptr;
}