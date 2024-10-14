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
    RMSNorm(const size_t _dim, const float _epsilon = 1e-5, const std::string& _name = "RMSNorm");

    // 覆盖基类的 forward 方法
    void forward(Tensor& x) override;

    // 虚析构函数
    virtual ~RMSNorm() = default;

private:
    size_t dim;
    float epsilon = 1e-5;
};


// 初始化不分配内存，等到load的时候再分配
RMSNorm::RMSNorm(const size_t _dim, const float _epsilon, const std::string& _name) : Layer("cpu", _name), dim(_dim), epsilon(_epsilon)
{
    params.emplace("weight", Parameter("weight", {dim}, "cpu"));
}

void RMSNorm::forward(Tensor& x)
{
    Parameter& W = params.at("weight");
    // 使用它们进行运算
    F.get().rmsnorm(x, W, dim, x.Size()/dim, epsilon);

}

#endif