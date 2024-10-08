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
    Linear(const size_t size_in, const size_t size_out, bool _bias, const std::string& _name = "Linear");

    // 覆盖基类的 forward 方法
    std::vector<Tensor> forward(std::vector<Tensor>& inputs) override;

    // 虚析构函数
    virtual ~Linear() = default;

private:
    size_t input_size;
    size_t output_size;
    bool bias;
};

// 初始化不分配内存，等到load的时候再分配
Linear::Linear(const size_t size_in, const size_t size_out, bool _bias, const std::string& _name) : Layer("cpu", _name)
{

    std::cout << "in Linear Constructor" << std::endl;

    input_size = size_in;
    output_size = size_out;
    bias = _bias;
    
    params.emplace("W", Parameter("W", nullptr, {size_in, size_out}, "cpu"));

    if(bias) params.emplace("b", Parameter("W", nullptr, {size_in}, "cpu"));

    temps.emplace("output", Tensor("output", new float[output_size], {output_size}, device));
}

// 这里写的代码很冗长 是因为 unordered_map 在调用 temps["output"] 时 会调用默认构造函数，
// 但是Tensor和parameter没有默认构造函数 会报错
std::vector<Tensor> Linear::forward(std::vector<Tensor>& inputs)
{
    // 尝试获取 output 和 weight
    Tensor& output = temps.at("output");
    Parameter& weight = params.at("W");

    // 使用它们进行运算
    F.matmul(output, inputs[0], weight, input_size, output_size);

    return { output };
}

#endif // LINEAR_H