#pragma once
#include <memory>
#include <string>
#include "allocator/Allocator.h"
// #include "IFunction.h"
// #include "DataType.h"

class Device {
public:
    Device() = default;
    virtual ~Device() = default;            // 虚析构函数，确保派生类的析构函数被调用
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&&) = delete;
    Device& operator=(Device&&) = delete;


    virtual void init() = 0;

    virtual void move_in(std::shared_ptr<void> ptr_dev, std::shared_ptr<void> ptr_cpu, size_t bytes) = 0;
    virtual void move_out(std::shared_ptr<void> ptr_dev, std::shared_ptr<void> ptr_cpu, size_t bytes) = 0;
    virtual void copy(std::shared_ptr<void> dst, std::shared_ptr<void> src, size_t bytes) = 0;
    
    // 分配内存（自动回收内存）
    std::shared_ptr<void> allocate(size_t bytes) {
        return allocator_->allocate(bytes);
    }

protected:
    std::string name_;
    std::unique_ptr<Allocator> allocator_;
    // std::unique_ptr<IFunction> func;
};