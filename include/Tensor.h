#ifndef TENSOR_H
#define TENSOR_H

#include "Variable.h"


class Tensor : public Variable {
public:
    // 构造函数
    Tensor(const std::string& _name, float* _value, const std::vector<int>& _shape, 
        const std::string& _device) : Variable(_name, _value, _shape, _device) {
        std::cout << "Tensor constructed: " << name << "\n";
    }

    // 虚析构函数
    ~Tensor() override {
        std::cout << "Tensor destructed: " << name << "\n";
    }

    // 禁止拷贝构造和拷贝赋值
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

};

#endif