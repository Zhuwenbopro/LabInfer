#include "allocator/CUDAAllocator.h"
#include "CUDAUtils.h"
#include <stdexcept>


CUDAAllocator::CUDAAllocator(){ }

std::shared_ptr<void> CUDAAllocator::allocate(size_t bytes) {
    if (bytes == 0) {
        throw std::invalid_argument("CUDAAllocator::allocate: size must be greater than 0");
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
