#ifndef MLP_H
#define MLP_H

#include "Layer.h"
#include "Linear.h"
#include "RMSNorm.h"
#include "Config.h"

class Mlp : public Layer {
public:
    // 构造函数，初始化线性层的输入和输出尺寸
    // 删除默认构造函数
    Mlp() = delete;
    // Linear(const Config& config, const std::string& name = "Linear");
    Mlp(const std::string& _name = "mlp");

    // 覆盖基类的 forward 方法
    void forward(Tensor& y, Tensor& x) override;

    // 虚析构函数
    virtual ~Mlp() = default;

private:
    size_t middle_size;
    size_t in_size;
};

// 初始化不分配内存，等到load的时候再分配
Mlp::Mlp(const std::string& _name) : Layer("cpu", _name)
{   
    middle_size = 8192;
    in_size = 2048;
    
    layers.emplace("gate_linear", new Linear(in_size, middle_size, "gate_proj"));
    layers.emplace("up_linear", new Linear(in_size, middle_size, "up_proj"));
    layers.emplace("down_linear", new Linear(middle_size, in_size, "down_proj"));
}

void Mlp::forward(Tensor& y, Tensor& x)
{
    Tensor gate("gate", {1, middle_size}, device, true, {x.elemNum()});
    Tensor up = gate.copy();
  
    layers.at("gate_linear")->forward(gate, x);

    F.get().silu(gate, gate.Size());

    layers.at("up_linear")->forward(up, x);

    F.get().elem_multiply(gate, gate, up, gate.Size());
    layers.at("down_linear")->forward(y, gate);

}

#endif // MLP_H