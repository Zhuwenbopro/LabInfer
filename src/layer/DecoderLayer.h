// DecoderLayer.h

#ifndef DECODERLAYER_H
#define DECODERLAYER_H

#include "Layer.h"
#include "Config.h"
#include "Attention.h"
#include "Mlp.h"

class DecoderLayer : public Layer {
public:
    // 构造函数，初始化线性层的输入和输出尺寸
    // 删除默认构造函数
    DecoderLayer() = delete;
    // Linear(const Config& config, const std::string& name = "Linear");
    DecoderLayer(const std::string& _name="Decoder");

    // 覆盖基类的 forward 方法
    void forward(Tensor& y, Tensor& x, Tensor& pos) override;

    // 虚析构函数
    virtual ~DecoderLayer() = default;
};

DecoderLayer::DecoderLayer(const std::string& _name) : Layer("cpu", _name) {
    size_t hidden_size = 2048;
    float epsilon = 1e-5;

    layers.emplace("input_layernorm", new RMSNorm(hidden_size, epsilon, "input_layernorm"));
    layers.emplace("self_attn", new Attention("self_attn"));
    layers.emplace("post_attention_layernorm", new RMSNorm(hidden_size, epsilon, "post_attention_layernorm"));
    layers.emplace("mlp", new Mlp("mlp"));
}

void DecoderLayer::forward(Tensor& y, Tensor& x, Tensor& pos)
{
    Tensor _x = x.copy();
    
    layers.at("input_layernorm")->forward(x);

    layers.at("self_attn")->forward(x, x, pos);
    F.get().add(x, x, _x, x.Size());

    _x = x.copy();
    layers.at("post_attention_layernorm")->forward(x);

    layers.at("mlp")->forward(x, x);
    F.get().add(y, x, _x, x.Size());
}

#endif // DECODERLAYER_H