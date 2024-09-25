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
        deviceName = "cpu";
        allocator = new CPUAllocator();
        F = new CPUFunction();
    }

    // 从 CPU 内存中取数据 (此处是个拷贝操作，因为数据已经在 CPU 中)
    void move_in(float* ptr_dev, float* ptr_cpu, size_t bytes) override { 
        std::cout << "device cpu don't need to use move in, but you used it." << std::endl;
    }

    // 移除数据到 CPU (此处也是个拷贝操作)
    void move_out(float* ptr_dev, float* ptr_cpu, size_t bytes) override {
        std::cout << "device cpu don't need to use move out, but you used it." << std::endl;
    }

    // 分配内存
    void allocate(float* ptr, size_t size) override {
        allocator->allocate((void*)ptr, size);
    }

    // 回收内存
    void deallocate(float* ptr) override {
        allocator->deallocate((void*)ptr);
    }
};



#endif // CPU_H