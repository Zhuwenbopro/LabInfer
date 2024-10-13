#ifndef LINEAR_H
#define LINEAR_H

#include "Layer.h"
#include "Config.h"

class Linear : public Layer {
public:
    // 构造函数，初始化线性层的输入和输出尺寸
    // 删除默认构造函数
    Linear() = delete;
    // Linear(const Config& config, const std::string& name = "Linear");
    Linear(const size_t size_in, const size_t size_out, const std::string& _name = "Linear", bool _bias = false);

    // 覆盖基类的 forward 方法
    void forward(Tensor& y, Tensor& x) override;

    // 虚析构函数
    virtual ~Linear() = default;

private:
    size_t input_size;  // 输入向量 x 的维度
    size_t output_size; // 输出向量 y 的维度
    bool bias;
};

// 初始化不分配内存，等到load的时候再分配
Linear::Linear(const size_t size_in, const size_t size_out, const std::string& _name, bool _bias) : Layer("cpu", _name)
{
    input_size = size_in;
    output_size = size_out;
    bias = _bias;
    
    params.emplace("weight", Parameter("weight", {size_in, size_out}, "cpu"));

    if(bias) params.emplace("bias", Parameter("bias", {size_in}, "cpu"));
}

// 这里写的代码很冗长 是因为 unordered_map 在调用 temps["output"] 时 会调用默认构造函数，
// 但是Tensor和parameter没有默认构造函数 会报错
void Linear::forward(Tensor& y, Tensor& x)
{
    Parameter& weight = params.at("weight");
    // 使用它们进行运算
    F.get().matmul(y, x, weight, input_size, output_size, x.Size() / input_size);

}

#endif // LINEAR_H