#ifndef CPUAllocator_H
#define CPUAllocator_H

#include "Allocator.h"
#include "stdio.h"
#include <stdexcept>

// 基础款，你之后再搞什么内存池之类的优化吧
class CPUAllocator : public Allocator {
public:
    CPUAllocator() {

    }

    float* allocate(size_t size) override {
        float* ptr = (float*)malloc(sizeof(float) * size);
        if (ptr == nullptr) {
            throw std::runtime_error("cpu malloc error");
        }
        return ptr;
    }

    void deallocate(void* ptr) override {
        free(ptr);
    }
}

#endif // CPUAllocator_H