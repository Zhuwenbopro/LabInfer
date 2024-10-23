// DecoderLayer.h

#ifndef DECODERLAYER_H
#define DECODERLAYER_H

#include "Layer.h"
#include "Config.h"
#include "Attention.h"
#include "Mlp.h"

class DecoderLayer : public Layer {
public:
    DecoderLayer() = delete;

    DecoderLayer(Config& config);

    // 覆盖基类的 forward 方法
    void forward(Tensor& y, Tensor& x, Tensor& pos) override;

    // 虚析构函数
    virtual ~DecoderLayer() = default;
};

DecoderLayer::DecoderLayer(Config& config) : Layer("cpu", "Decoder") {

    size_t hidden_size = config.get("hidden_size").get<size_t>();
    float epsilon = config.get("rms_norm_eps").get<float>();

    layers.emplace("input_layernorm", new RMSNorm(config));
    layers.at("input_layernorm")->setName("input_layernorm");
    layers.emplace("self_attn", new Attention(config));
    layers.emplace("post_attention_layernorm", new RMSNorm(config));
    layers.at("post_attention_layernorm")->setName("post_attention_layernorm");
    layers.emplace("mlp", new Mlp(config));
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