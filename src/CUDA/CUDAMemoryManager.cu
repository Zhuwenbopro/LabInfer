#include "CUDA/CUDAMemoryManager.h"
#include "CUDA/CUDAUtils.h"
// TODO：delelte future
#include <iostream>

CUDAMemoryManager::CUDAMemoryManager(){ }

void* CUDAMemoryManager::allocate(size_t bytes) {
    if (bytes == 0) {
        std::cerr << "Invalid allocation size: 0 bytes." << std::endl;
        exit(-1);
    }
    CUDA_CHECK(cudaGetLastError());
    void* devPtr;
    CUDA_CHECK(cudaMalloc((void **)&devPtr, bytes));
    return devPtr;
}

void CUDAMemoryManager::deallocate(void* ptr) {
    CUDA_CHECK(cudaFree(ptr));
    ptr = nullptr;
}