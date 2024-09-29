#ifndef PARAMETER_H
#define PARAMETER_H

#include "Variable.h"


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

    // parameter 只归属于 layer，它的位置由 layer 负责
};
#endif