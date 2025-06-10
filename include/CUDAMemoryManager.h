#pragma once

#include "MemoryManager.h"

// 基础款，你之后再搞什么内存池之类的优化吧
class CUDAMemoryManager : public MemoryManager {
public:
    CUDAMemoryManager();

    void* allocate(size_t bytes) override;

    void deallocate(void* ptr) override;
};