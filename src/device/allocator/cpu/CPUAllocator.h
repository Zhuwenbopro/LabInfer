#ifndef CPUAllocator_H
#define CPUAllocator_H

#include "../Allocator.h"

// 基础款，你之后再搞什么内存池之类的优化吧
class CPUAllocator : public Allocator {
public:
    CPUAllocator() {

    }

    float* allocate(std::size_t size) override {

    }

    void deallocate(void* ptr) override {

    }
}

#endif // CPUAllocator_H