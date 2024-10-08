// Softmax.h
#ifndef SOFTMAX_H
#define SOFTMAX_H



#include "Layer.h"
#include "Config.h"

class Softmax : public Layer {
public:
    // 构造函数，初始化线性层的输入和输出尺寸
    // 删除默认构造函数
    Softmax() = delete;
    // Softmax(const Config& config, const std::string& name = "Softmax");
    Softmax(const size_t _size_in, const std::string& _name = "Softmax") : Layer("cpu", _name), size_in(_size_in) {};

    // 覆盖基类的 forward 方法
    void forward(Tensor& x) override;

    // 虚析构函数
    virtual ~Softmax() = default;

private:
    size_t size_in;
};


void Softmax::forward(Tensor& x)
{
    F.get().softmax(x, size_in);
}

#endif