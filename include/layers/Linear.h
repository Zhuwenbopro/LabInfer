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
    Tensor forward(Tensor& x) override;

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
    
    params.emplace("weight", Parameter("weight", size_in, size_out, "cpu"));

    if(bias) params.emplace("bias", Parameter("bias", size_in, 1, "cpu"));
}

Tensor Linear::forward(Tensor& x)
{   
    Tensor y(x, output_size);
    F.get().matmul(y, x, params.at("weight"), input_size, output_size, x.elemNum());
    return y;
}

#endif // LINEAR_H