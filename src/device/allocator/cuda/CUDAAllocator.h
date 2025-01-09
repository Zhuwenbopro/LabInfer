#ifndef CUDAAllocator_H
#define CUDAAllocator_H

#include "Allocator.h"

// 基础款，你之后再搞什么内存池之类的优化吧
class CUDAAllocator : public Allocator {
public:
    CUDAAllocator();

    void* allocate(size_t bytes) override;

    void deallocate(void* ptr) override;
};

#endif // CUDAAllocator_H