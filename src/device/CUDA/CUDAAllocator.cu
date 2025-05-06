#include "CUDAAllocator.h"
#include <cuda_runtime.h>
#include "CUDAUtils.h"

CUDAAllocator::CUDAAllocator(){ }

std::shared_ptr<void> CUDAAllocator::allocate(size_t bytes) {
    if (bytes == 0) {
        std::cerr << "Invalid allocation size: 0 bytes." << std::endl;
        exit(-1);
    }

    auto deleter = [](void* ptr) {
        if (ptr) {
            CUDA_CHECK(cudaFree(ptr));
            ptr = nullptr;
        }
    };

    void* devPtr;
    CUDA_CHECK(cudaMalloc(&devPtr, bytes));

    return std::shared_ptr<void>(devPtr, deleter);
}
