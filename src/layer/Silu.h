// Silu.h
#ifndef SILU_H
#define SILU_H


#include "Layer.h"
#include "Config.h"

class Silu : public Layer {
public:
    // 构造函数，初始化线性层的输入和输出尺寸
    // 删除默认构造函数
    Silu() : Layer("cpu", "Silu") {};
    // Softmax(const Config& config, const std::string& name = "Softmax");
    Silu(const std::string& _name = "Silu") : Layer("cpu", _name) {};

    // 覆盖基类的 forward 方法
    void forward(Tensor& x) override;

    // 虚析构函数
    virtual ~Silu() = default;

private:

};


void Silu::forward(Tensor& x)
{
    F.get().silu(x, x.elemLen(), x.elemNum());
}

#endif