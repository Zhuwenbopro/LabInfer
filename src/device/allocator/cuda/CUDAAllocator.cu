#include "CUDAAllocator.h"
#include <cuda_runtime.h>

CUDAAllocator::CUDAAllocator(){ }

float* CUDAAllocator::allocate(std::size_t size) {
    float* devPtr;
    cudaMalloc((void **)&devPtr, size);
    return devPtr;
}

void CUDAAllocator::deallocate(void* ptr) {
    cudaFree(ptr);
}