#ifndef CPU_H
#define CPU_H

#include "Device.h"
#include "allocator/cpu/CPUAllocator.h"
#include "function/cpu/CPUFunction.h"
#include <iostream>

// CPU 不用重载 move_in、move_out
class CPU : public Device {
public:
    // 构造函数
    CPU() {
        device = "cpu";
        allocator = new CPUAllocator();
        F = new CPUFunction();
    }

    // 从 CPU 内存中取数据 (此处是个拷贝操作，因为数据已经在 CPU 中)
    void move_in(float* ptr_dev, float* ptr_cpu, size_t bytes) override { 
        throw std::logic_error("device cpu don't need to use move_in, but you used it.");
    }

    // 移除数据到 CPU (此处也是个拷贝操作)
    void move_out(float* ptr_dev, float* ptr_cpu, size_t bytes) override {
        throw std::logic_error("device cpu don't need to use move_out, but you used it.");
    }

    // 分配内存
    float* allocate(size_t size) override {
        float* ptr = (float*)allocator->allocate(size*sizeof(float));
        return ptr;
    }

    // 回收内存
    void deallocate(float* ptr) override {
        allocator->deallocate((void*)ptr);
    }

    void copy(float* from, float* to, size_t size) override {
        std::copy(from, from + size, to);
    }
};



#endif // CPU_H