#pragma once

#include "Allocator.h"
#include <cstdlib>
#include <stdexcept>

// 基础款，你之后再搞什么内存池之类的优化吧
class CPUAllocator : public Allocator
{
public:
    CPUAllocator() {}

    std::shared_ptr<void> allocate(size_t bytes) override
    {
        if (bytes == 0)
        {
            throw std::invalid_argument("[CPUAllocator] Cannot allocate 0 bytes");
        }

        void *ptr = malloc(bytes);
        if (ptr == nullptr)
        {
            throw std::bad_alloc();
        }

        auto deleter = [](void *ptr)
        {
            if (ptr)
            {
                free(ptr);
            }
        };

        return std::shared_ptr<void>(ptr, deleter);
    }
};