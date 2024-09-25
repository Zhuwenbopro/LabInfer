#include "CUDAAllocator.h"
#include <cuda_runtime.h>

CUDAAllocator::CUDAAllocator(){ }

void CUDAAllocator::allocate(void* devPtr, std::size_t size) {
    cudaMalloc((void **)&devPtr, size);
}

void CUDAAllocator::deallocate(void* ptr) {
    cudaFree(ptr);
}