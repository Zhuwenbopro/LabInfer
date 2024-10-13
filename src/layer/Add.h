// Add.h

#ifndef ADD_H
#define ADD_H

#include "Layer.h"
#include "Config.h"

class Add : public Layer {
public:
    // 构造函数，初始化线性层的输入和输出尺寸
    // 删除默认构造函数
    Add() = delete;
    // Linear(const Config& config, const std::string& name = "Linear");
    Add::Add(const std::string& _name) : Layer("cpu", _name) {}

    // 覆盖基类的 forward 方法
    void forward(Tensor& y, Tensor& x1, Tensor& x2) override;

    // 虚析构函数
    virtual ~Add() = default;
};


void Add::forward(Tensor& y, Tensor& x1, Tensor& x2)
{
    if(x1.Size() != x2.Size()) {
        throw std::runtime_error("Add Layer x1 size and x2 size don't match.");
    }
    F.get().add(y, x1, x2, x1.Size());
}

#endif // ADD_H