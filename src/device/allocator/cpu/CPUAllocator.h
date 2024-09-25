#ifndef CPUAllocator_H
#define CPUAllocator_H

#include "Allocator.h"
#include <cstdlib>
#include <iostream>

// 基础款，你之后再搞什么内存池之类的优化吧
class CPUAllocator : public Allocator {
public:
    CPUAllocator() {

    }

    void* allocate(size_t size) override {
        return malloc(size);
    }

    void deallocate(void* ptr) override {
        free(ptr);
    }
};

#endif // CPUAllocator_H