#ifndef PARAMETER_H
#define PARAMETER_H

#include "Variable.h"


class Parameter : public Variable {
public:
    // 构造函数
    Parameter(const std::string& var_name, float* var_value, const std::vector<size_t>& var_shape, 
        const std::string& device) : Variable(var_name, var_value, var_shape, device) {
    }

    // 拷贝构造函数（浅拷贝）
    Parameter(const Parameter& other) : Variable(other) {
        // 由于希望进行浅拷贝，仅复制指针和基本信息，不创建新的数据副本
    }

    // 拷贝赋值运算符（浅拷贝）
    Parameter& operator=(const Parameter& other) {
        if (this != &other) {
            // 调用基类的赋值运算符，进行浅拷贝
            Variable::operator=(other);
        }
        return *this;
    }

    // 虚析构函数
    ~Parameter() override {
        std::cout << "Parameter destructed: " << name << "\n";
    }

    // parameter 只归属于 layer，它的位置由 layer 负责
};
#endif