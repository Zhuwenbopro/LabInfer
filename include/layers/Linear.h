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
    Tensor<float> forward(Tensor<float>& x) override;

    // 虚析构函数
    virtual ~Linear() = default;

private:
    size_t input_size;  // 输入向量 x 的维度
    size_t output_size; // 输出向量 y 的维度
    bool bias;
};


Linear::Linear(const size_t size_in, const size_t size_out, const std::string& _name, bool _bias) : Layer("cpu", _name)
{
    input_size = size_in;
    output_size = size_out;
    bias = _bias;
    
    params.emplace("weight", Parameter<float>(size_out, size_in, "cpu", "weight"));

    if(bias) params.emplace("bias", Parameter<float>(1, size_in, "cpu", "bias"));
}

Tensor<float> Linear::forward(Tensor<float>& x)
{   
    if(x.ElemLen() != input_size) {
        throw std::runtime_error("Layer " + name + "'s input len not match param len.");
    }

    std::cout << x.ElemLen() << " " << input_size << " " << output_size << "\n";
    std::cout << x.ElemNum() << " " << F << "\n";
    Tensor<float> y(x.ElemNum(), output_size, x.Device(), name + "_output");
    F->matmul(y, x, params.at("weight"), input_size, output_size, x.ElemNum());
    return y;
}

#endif // LINEAR_H