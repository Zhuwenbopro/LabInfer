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
    RMSNorm(const size_t _hidden_size, const float _epsilon = 1e-5, const std::string& _name = "RMSNorm");

    // 覆盖基类的 forward 方法
    void forward(InputWarp& inputWarp) override;

    // 虚析构函数
    virtual ~RMSNorm() = default;

private:
    size_t hidden_size;
    float epsilon = 1e-5;
};

RMSNorm::RMSNorm(const size_t _hidden_size, const float _epsilon, const std::string& _name) : Layer("cpu", _name), hidden_size(_hidden_size), epsilon(_epsilon)
{
    params.emplace("weight", Parameter<float>(1, hidden_size, "cpu", "weight"));
}

void RMSNorm::forward(InputWarp& inputWarp)
{
    F->rmsnorm(inputWarp.inter_value, params.at("weight"), hidden_size, inputWarp.inter_value.ElemNum(), epsilon);
}

#endif