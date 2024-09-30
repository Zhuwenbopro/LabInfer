#ifndef LINEAR_H
#define LINEAR_H

#include "Layer.h"
#include "Config.h"

class Linear : public Layer {
public:
    // 构造函数，初始化线性层的输入和输出尺寸
    Linear(const Config& config, const std::string& name = "Linear");

    // 覆盖基类的 forward 方法
    std::vector<Tensor> forward(std::vector<Tensor>& inputs) override;

    // 虚析构函数
    virtual ~Linear() = default;

private:
    int input_size;
    int output_size;

    // 可以根据需要添加其他成员或方法
};

// 假设 Parameter 可以用尺寸和设备进行初始化
Linear::Linear(const Config& config, const std::string& name) : Layer()
{
    setName(name);

    // 初始化在这里
    // 把input、output的size搞清楚，有没有bias
    // 初始化不分配内存，等到load的时候再分配

}

std::vector<Tensor> Linear::forward(std::vector<Tensor>& inputs)
{
    /*
    // 对于线性层，我们期望一个输入张量
    assert(inputs.size() == 1);
    Tensor input = inputs[0];

    // 获取权重和偏置
    // 假设你有方法通过名称或索引访问 params
    Tensor weight = params[0].getData(); // 形状：[output_size, input_size]
    Tensor bias = params[1].getData(); // 形状：[output_size]

    // 执行线性变换：output = input * weight^T + bias
    // 假设 Tensor 支持矩阵乘法和加法
    Tensor output = input.matmul(weight.transpose()) + bias;

    return { output };
    */
}

#endif // LINEAR_H