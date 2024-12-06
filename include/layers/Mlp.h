#ifndef MLP_H
#define MLP_H

#include "Layer.h"
#include "Linear.h"
#include "RMSNorm.h"
#include "Config.h"

class Mlp : public Layer {
public:
    Mlp() = delete;

    Mlp(Config& config);

    Mlp(const size_t& in_size, const size_t& middle_size);

    // 覆盖基类的 forward 方法
    Tensor forward(Tensor& x) override;

    // 虚析构函数
    virtual ~Mlp() = default;

private:
    size_t middle_size;
    size_t in_size;
};

// 初始化不分配内存，等到load的时候再分配

Mlp::Mlp(Config& config) : Layer("cpu", "mlp")
{
    middle_size = config.get<size_t>("intermediate_size");
    in_size = config.get<size_t>("hidden_size");

    layers.emplace("gate_linear", new Linear(in_size, middle_size, "gate_proj"));
    layers.emplace("up_linear", new Linear(in_size, middle_size, "up_proj"));
    layers.emplace("down_linear", new Linear(middle_size, in_size, "down_proj"));
}

Mlp::Mlp(const size_t& _in_size, const size_t& _middle_size) : Layer("cpu", "mlp")
{
    middle_size = _middle_size;
    in_size = _in_size;

    layers.emplace("gate_linear", new Linear(in_size, middle_size, "gate_proj"));
    layers.emplace("up_linear", new Linear(in_size, middle_size, "up_proj"));
    layers.emplace("down_linear", new Linear(middle_size, in_size, "down_proj"));
}


Tensor Mlp::forward(Tensor& x)
{
    Tensor gate = layers.at("gate_linear")->forward(x);

    F.get().silu(gate, gate.Size());
    
    Tensor up = layers.at("up_linear")->forward(x);
    
    F.get().elem_multiply(gate, gate, up, gate.Size());
    
    Tensor y = layers.at("down_linear")->forward(gate);
    
    return y;
}

#endif // MLP_H