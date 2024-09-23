#ifndef VARIABLE_H
#define VARIABLE_H

#include "../device/DeviceFactory.h"
#include <iostream>
#include <string>
#include <vector>

/**
    在实现上我应该避免做很多华丽、便捷的接口机制，
    当前主要任务是完成整体流程的运行，因此我仅应该完成最小化的实现。

    1. value、shape、device 初始化
    2. to 设备传输功能，当前仅实现单机多卡之间的设备传输

    TODO：
    1. 当前value成员是一个裸指针（float*），这可能导致内存泄漏或悬挂指针的问题，特别是在复制或移动对象时。
 */

class Variable {
public:
    // 虚析构函数，确保派生类的析构函数被调用
    virtual ~Variable() = default;

    // 禁止拷贝构造和拷贝赋值，避免浅拷贝问题
    Variable(const Variable&) = delete;
    Variable& operator=(const Variable&) = delete;

    // 纯虚函数，用于设备间传输
    virtual void to(const std::string& new_dev) = 0;

    float *Data() const { return value; }
    const std::vector<int>& Shape() const { return shape; }
    size_t Size() const { return size; }
    const std::string& Name() const { return name; }
    void whereami() const { std::cout << dev->getDeviceName() << std::endl; }

protected:
    // 使用智能指针管理内存
    float* value;                             // 数据指针
    std::vector<int> shape;                   // 数据形状
    size_t size;                              // 数据大小（元素个数）
    std::string name;                         // 变量名称
    Device* dev;                              // 设备

    // 构造函数
    Variable(const std::string& var_name, float* var_value, const std::vector<int>& var_shape, 
        const std::string& device) : value(var_value), shape(var_shape), size(1), name(var_name) {

        dev = DeviceFactory::getDevice(device);

        for (const auto& dim : shape) {
            size *= dim;
        }
    }
};

class Tensor : public Variable {
public:
    // 构造函数
    Tensor(const std::string& var_name, float* var_value, const std::vector<int>& var_shape, 
        const std::string& device) : Variable(var_name, var_value, var_shape, device) {
        std::cout << "Tensor constructed: " << name << "\n";
    }

    // 虚析构函数
    ~Tensor() override {
        std::cout << "Tensor destructed: " << name << "\n";
    }

    // 禁止拷贝构造和拷贝赋值
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // 实现设备间传输方法
    void to(const std::string& new_dev) override {
        // 这里应实现实际的设备间数据传输逻辑
        // 例如，使用CUDA API进行GPU间数据拷贝
        std::cout << "Transferring Parameter '" << name << "' from " 
                  << dev << " to " << new_dev << "\n";

        // 假设传输后在新设备上分配新内存，并释放旧内存
        // 这里只是模拟传输，不进行实际的数据拷贝
        dev = DeviceFactory::getDevice(new_dev);
    }
};

class Parameter : public Variable {
public:
    // 构造函数
    Parameter(const std::string& var_name, float* var_value, const std::vector<int>& var_shape, 
        const std::string& device) : Variable(var_name, var_value, var_shape, device) {
        std::cout << "Parameter constructed: " << name << "\n";
    }

    // 虚析构函数
    ~Parameter() override {
        std::cout << "Parameter destructed: " << name << "\n";
    }

    // 禁止拷贝构造和拷贝赋值
    Parameter(const Parameter&) = delete;
    Parameter& operator=(const Parameter&) = delete;

    // 实现设备间传输方法
    void to(const std::string& new_dev) override {
        // 这里应实现实际的设备间数据传输逻辑
        // 例如，使用CUDA API进行GPU间数据拷贝
        std::cout << "Transferring Parameter '" << name << "' from " 
                  << dev << " to " << new_dev << "\n";
        
        // 假设传输后在新设备上分配新内存，并释放旧内存
        // 这里只是模拟传输，不进行实际的数据拷贝
        dev = DeviceFactory::getDevice(new_dev);
    }
};

#endif // VARIABLE_H
