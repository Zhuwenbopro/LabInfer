#ifndef TENSOR_H
#define TENSOR_H

#include "Variable.h"


class Tensor : public Variable {
public:
    // 构造函数
    Tensor(const std::string& _name, float* _value, const std::vector<size_t>& _shape, 
        const std::string& _device = "cpu") : Variable(_name, _value, _shape, _device) {
    }

     // 拷贝构造函数（浅拷贝）
    Tensor(const Tensor& other) : Variable(other) {
        // 由于希望进行浅拷贝，仅复制指针和基本信息，不创建新的数据副本
    }

    // 拷贝赋值运算符（浅拷贝）
    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            // 调用基类的赋值运算符，进行浅拷贝
            Variable::operator=(other);
        }
        return *this;
    }

    // 深拷贝函数
    Tensor copy() const {
        // 为新 Tensor 分配新的内存
        float* new_value = new float[size];

        // 复制数据到新的内存区域
        std::copy(value, value + size, new_value);

        Tensor res = Tensor(name, new_value, shape, device);
        res.to(device);
        // 创建并返回新的 Tensor 对象
        return res;
    }

    // 虚析构函数
    ~Tensor() override { }

};

#endif