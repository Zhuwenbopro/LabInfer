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
    void forward(InputWarp& inputWarp) override;

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

void Linear::forward(InputWarp& inputWarp)
{   
    if(inputWarp.inter_value.ElemLen() != input_size) {
        std::cout << "input size = " << input_size << "   = " << inputWarp.inter_value.ElemLen() << std::endl;
        throw std::runtime_error("Layer " + name + "'s input len not match param len.");
    }

    Tensor<float> y(inputWarp.inter_value.ElemNum(), output_size, device, name + "_output");
    F->matmul(y, inputWarp.inter_value, params.at("weight"), input_size, output_size, inputWarp.inter_value.ElemNum());
    inputWarp.inter_value = y;
}

#endif // LINEAR_H