#ifndef CUDAAllocator_H
#define CUDAAllocator_H

#include "Allocator.h"

// 基础款，你之后再搞什么内存池之类的优化吧
class CUDAAllocator : public Allocator {
public:
    CUDAAllocator();

    float* allocate(std::size_t size) override {
        float* devPtr;
        cudaMalloc((void **)&devPtr, size);
        return devPtr;
    }

    void deallocate(void* ptr) override {
        cudaFree(ptr);
    }
}

#endif // CUDAAllocator_H