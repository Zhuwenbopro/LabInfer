#ifndef VARIABLE_H
#define VARIABLE_H

#include "Device.h"
#include <iostream>
#include <string>

class Variable {
public:
    Variable(const std::string& name = "", const std::string& devName = "cpu");
    Variable(const Variable& other);                   // 拷贝构造函数
    Variable& operator=(const Variable& other);        // 赋值运算符
    virtual ~Variable();                               // 析构函数

    virtual void to(Device &dev) = 0;
    virtual void shape() const = 0;

    void whereami() { 
        std::cout << dev << std::endl; 
    }

protected:
    float* value;          // 数据指针
    size_t size;           // 数据大小（元素个数）
    std::string name;      // 变量名称
    Device dev;            // 设备
};

class Tensor : public Variable {
public:
    Tensor(const float* val, size_t size, const std::string& name, const std::string& dev = "cpu");
    Tensor(const Tensor& other);                   // 拷贝构造函数
    Tensor& operator=(const Tensor& other);        // 赋值运算符
    ~Tensor();

    void to(Device &dev) override;
    void shape() const override;
};

class Parameter : public Variable {
public:
    Parameter(const float* val, size_t size, const std::string& name, const std::string& dev = "cpu");
    Parameter(const Parameter& other);             // 拷贝构造函数
    Parameter& operator=(const Parameter& other);  // 赋值运算符
    ~Parameter();

    void to(Device &dev) override;
    void shape() const override;
};

#endif // VARIABLE_H
