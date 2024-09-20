#include "Variable.h"
#include <cstring>  // 用于 memcpy

// ------------------ Variable 类 ------------------

// 构造函数
Variable::Variable(const std::string& name, const std::string& devName)
    : value(nullptr), size(0), name(name), dev(devName) {}

// 拷贝构造函数
Variable::Variable(const Variable& other)
    : size(other.size), name(other.name), dev(other.dev)
{
    if (other.value && other.size > 0) {
        value = new float[other.size];
        std::memcpy(value, other.value, other.size * sizeof(float));
    } else {
        value = nullptr;
    }
}

// 赋值运算符
Variable& Variable::operator=(const Variable& other)
{
    if (this != &other) {
        delete[] value;  // 释放原有内存

        size = other.size;
        name = other.name;
        dev = other.dev;

        if (other.value && other.size > 0) {
            value = new float[other.size];
            std::memcpy(value, other.value, other.size * sizeof(float));
        } else {
            value = nullptr;
        }
    }
    return *this;
}

// 析构函数
Variable::~Variable()
{
    delete[] value;
}

// ------------------ Tensor 类 ------------------

// 构造函数
Tensor::Tensor(const float* val, size_t size, const std::string& name, const std::string& devName)
    : Variable(name, devName)
{
    this->size = size;
    if (val && size > 0) {
        value = new float[size];
        std::memcpy(value, val, size * sizeof(float));
    } else {
        value = nullptr;
    }
}

// 拷贝构造函数
Tensor::Tensor(const Tensor& other)
    : Variable(other) {}

// 赋值运算符
Tensor& Tensor::operator=(const Tensor& other)
{
    if (this != &other) {
        Variable::operator=(other);
    }
    return *this;
}

// 析构函数
Tensor::~Tensor()
{
    // 基类析构函数已经释放了内存
}

// to() 方法
void Tensor::to(Device &device)
{
    dev = device;
}

// shape() 方法
void Tensor::shape() const
{
    // 示例输出
    std::cout << "Tensor '" << name << "' shape: [" << size << "]" << std::endl;
}

// ------------------ Parameter 类 ------------------

// 构造函数
Parameter::Parameter(const float* val, size_t size, const std::string& name, const std::string& devName)
    : Variable(name, devName)
{
    this->size = size;
    if (val && size > 0) {
        value = new float[size];
        std::memcpy(value, val, size * sizeof(float));
    } else {
        value = nullptr;
    }
}

// 拷贝构造函数
Parameter::Parameter(const Parameter& other)
    : Variable(other) {}

// 赋值运算符
Parameter& Parameter::operator=(const Parameter& other)
{
    if (this != &other) {
        Variable::operator=(other);
    }
    return *this;
}

// 析构函数
Parameter::~Parameter()
{
    // 基类析构函数已经释放了内存
}

// to() 方法
void Parameter::to(Device &device)
{
    dev = device;
}

// shape() 方法
void Parameter::shape() const
{
    // 示例输出
    std::cout << "Parameter '" << name << "' shape: [" << size << "]" << std::endl;
}
