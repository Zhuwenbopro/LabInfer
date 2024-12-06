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

    DecoderLayer(Config& config, const std::string& name = "decoder_layer");

    // 覆盖基类的 forward 方法
    Tensor forward(Tensor& x) override;

    // 虚析构函数
    virtual ~DecoderLayer() = default;
};

DecoderLayer::DecoderLayer(Config& config, const std::string& _name) : Layer("cpu", _name) {

    size_t hidden_size = config.get<size_t>("hidden_size");
    float epsilon = config.get<float>("rms_norm_eps");

    layers.emplace("input_layernorm", new RMSNorm(config));
    layers.at("input_layernorm")->setName("input_layernorm");
    layers.emplace("self_attn", new Attention(config));
    layers.emplace("post_attention_layernorm", new RMSNorm(config));
    layers.at("post_attention_layernorm")->setName("post_attention_layernorm");
    layers.emplace("mlp", new Mlp(config));
}


Tensor DecoderLayer::forward(Tensor& x)
{
    Tensor _x(x, x.elemLen());
    _x.copy(x);

    x = layers.at("input_layernorm")->forward(x);
    
    x = layers.at("self_attn")->forward(x);
    F.get().add(x, x, _x, x.Size());

    _x.copy(x);
    x = layers.at("post_attention_layernorm")->forward(x);

    x = layers.at("mlp")->forward(x);
    F.get().add(x, x, _x, x.Size());
    return x;
}

#endif // DECODERLAYER_H