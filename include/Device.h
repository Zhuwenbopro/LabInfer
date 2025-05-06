// Device.h
#ifndef DEVICE_H
#define DEVICE_H

#include <string>
#include "Allocator.h"
#include "Function.h"

class Device {
public:
    Device() = default;
    virtual ~Device() = default;            // 虚析构函数，确保派生类的析构函数被调用

    // 禁止拷贝构造和拷贝赋值，避免浅拷贝问题
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    // 从 CPU 内存中取数据
    virtual void move_in(std::shared_ptr<void> ptr_dev, std::shared_ptr<void> ptr_cpu, size_t bytes) = 0;
    // 移除数据到 CPU
    virtual void move_out(std::shared_ptr<void> ptr_dev, std::shared_ptr<void> ptr_cpu, size_t bytes) = 0;

    virtual void copy(std::shared_ptr<void> dst, std::shared_ptr<void> src, size_t bytes) = 0;
    // 分配内存（自动回收内存）
    virtual std::shared_ptr<void> allocate(size_t bytes) = 0;

    int device_id;
    std::string device_name;
    Function* F;
    Allocator* allocator;
};

#endif // DEVICE_H
