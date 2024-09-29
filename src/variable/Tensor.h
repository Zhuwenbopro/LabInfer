#ifndef TENSOR_H
#define TENSOR_H

#include "Variable.h"


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
    void to(const std::string& new_dev) {
        // 这里应实现实际的设备间数据传输逻辑
        // 例如，使用CUDA API进行GPU间数据拷贝
        std::cout << "Transferring Parameter '" << name << "' from " 
                  << dev << " to " << new_dev << "\n";
    }
};

#endif