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

#endif // LINEAR_H