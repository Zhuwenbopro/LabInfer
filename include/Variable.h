#ifndef VARIABLE_H
#define VARIABLE_H

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

    // 隐式转换 Variable 类和 float*
    operator float*() const { return value; }

    float *Data() const { return value; }
    void setData(float* val) { value = val; }
    const std::vector<int>& Shape() const { return shape; }
    size_t Size() const { return size; }
    const std::string& Name() const { return name; }
    const std::string& Device() const { return dev; }

protected:
    // 使用智能指针管理内存
    float* value;                             // 数据指针
    std::vector<int> shape;                   // 数据形状
    size_t size;                              // 数据大小（元素个数）
    std::string name;                         // 变量名称
    std::string dev;                          // 设备

    // 构造函数
    Variable(const std::string& var_name, float* var_value, const std::vector<int>& var_shape, 
        const std::string& device) : value(var_value), shape(var_shape), size(1), name(var_name), dev(device) {

        for (const auto& dim : shape) {
            size *= dim;
        }
    }
};


#endif // VARIABLE_H
