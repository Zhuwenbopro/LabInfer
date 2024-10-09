// RMSNorm.h
#ifndef RMSNORM_H
#define RMSNORM_H


#include "Layer.h"
#include "Config.h"

class RMSNorm : public Layer {
public:
    // 构造函数，初始化线性层的输入和输出尺寸
    // 删除默认构造函数
    RMSNorm() = delete;
    // Softmax(const Config& config, const std::string& name = "Softmax");
    RMSNorm(const size_t _size_in, const float _epsilon = 1e-5, const std::string& _name = "RMSNorm");

    // 覆盖基类的 forward 方法
    void forward(Tensor& y, Tensor& x) override;

    // 虚析构函数
    virtual ~RMSNorm() = default;

private:
    size_t size_in;
    float epsilon = 1e-5;
};



// 初始化不分配内存，等到load的时候再分配
RMSNorm::RMSNorm(const size_t _size_in, const float _epsilon = 1e-5, const std::string& _name) : Layer("cpu", _name)
{
    size_in = _size_in;
    epsilon = _epsilon;
    params.emplace("W", Parameter("W", nullptr, {size_in}, "cpu"));
}

// 这里写的代码很冗长 是因为 unordered_map 在调用 temps["output"] 时 会调用默认构造函数，
// 但是Tensor和parameter没有默认构造函数 会报错
void RMSNorm::forward(Tensor& y, Tensor& x)
{
    Parameter& W = params.at("W");
    // 使用它们进行运算
    F.get().rmsnorm(y, x, W, epsilon, size_in);

}

#endif