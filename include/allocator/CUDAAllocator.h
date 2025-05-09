#pragma once

#include "Allocator.h"

// 基础款，你之后再搞什么内存池之类的优化吧
class CUDAAllocator : public Allocator {
public:
    CUDAAllocator();

    std::shared_ptr<void> allocate(size_t bytes) override;

};