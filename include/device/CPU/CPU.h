#pragma once

#include "../Device.h"
#include "allocator/CPUAllocator.h"
#include <cstring>

class CPU : public Device
{
public:
    // 构造函数
    CPU()
    {
        this->name_ = "cpu";
        this->allocator_ = std::unique_ptr<CPUAllocator>();
        // this->func = create_cpu_function(dtype);
    }

    // 从 CPU 内存中取数据 (此处是个拷贝操作，因为数据已经在 CPU 中)
    void move_in(std::shared_ptr<void> ptr_dev, std::shared_ptr<void> ptr_cpu, size_t bytes) override
    {
        throw std::logic_error("[CPU] device cpu don't need to use move_in, but you used it.");
    }

    // 移除数据到 CPU (此处也是个拷贝操作)
    void move_out(std::shared_ptr<void> ptr_dev, std::shared_ptr<void> ptr_cpu, size_t bytes) override
    {
        throw std::logic_error("[CPU] device cpu don't need to use move_out, but you used it.");
    }

    void copy(std::shared_ptr<void> dst, std::shared_ptr<void> src, size_t bytes) override
    {
        std::memcpy(dst.get(), src.get(), bytes);
    }
};