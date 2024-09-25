// Device.h
#ifndef DEVICE_H
#define DEVICE_H

#include <string>
#include "Allocator.h"
#include "Function.h"

class Device {
public:
    Device() = default;
    // 虚析构函数，确保派生类的析构函数被调用
    virtual ~Device() = default;

    // 禁止拷贝构造和拷贝赋值，避免浅拷贝问题
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    // 从 CPU 内存中取数据
    virtual void move_in(float* ptr_dev, float* ptr_cpu, size_t size) = 0;
    // 移除数据到 CPU
    virtual void move_out(float* ptr_dev, float* ptr_cpu, size_t size) = 0;
    // 分配内存
    virtual float* allocate(size_t size) = 0;
    // 回收内存
    virtual void deallocate(float* ptr) = 0;

    // 成员变量
    std::string deviceName;
    Function* F;
    // 之所以还要设置allocator，是为了以后在内存管理上做文章，目前还用不到
    Allocator* allocator;
};

#endif // DEVICE_H
