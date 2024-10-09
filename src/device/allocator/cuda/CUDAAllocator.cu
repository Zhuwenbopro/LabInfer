#include "CUDAAllocator.h"
#include <cuda_runtime.h>

CUDAAllocator::CUDAAllocator(){ }

void* CUDAAllocator::allocate(size_t size) {
    void* devPtr;
    cudaMalloc((void **)&devPtr, size);
    return devPtr;
}

void CUDAAllocator::deallocate(void* ptr) {
    cudaFree(ptr);
    ptr = nullptr;
}