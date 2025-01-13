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

    DecoderLayer(
        const size_t attn_head, 
        const size_t kv_head, 
        const size_t hidden_size,
        const size_t intermediate_size,
        const size_t _max_len = 250, 
        const float epsilon = 1e-5
    );

    void forward(InputWarp& inputWarp) override;

    // 虚析构函数
    virtual ~DecoderLayer() = default;
};

DecoderLayer::DecoderLayer(
    const size_t attn_head, 
    const size_t kv_head, 
    const size_t hidden_size,
    const size_t intermediate_size,
    const size_t _max_len, 
    const float epsilon
) : Layer("cpu", "decoder_layer") {

    layers.emplace("input_layernorm", new RMSNorm(hidden_size, epsilon));
    layers.at("input_layernorm")->setName("input_layernorm");
    layers.emplace("self_attn", new Attention(attn_head, kv_head, hidden_size));
    layers.emplace("post_attention_layernorm", new RMSNorm(hidden_size, epsilon));
    layers.at("post_attention_layernorm")->setName("post_attention_layernorm");
    layers.emplace("mlp", new Mlp(hidden_size, intermediate_size));
}


void DecoderLayer::forward(InputWarp& inputWarp)
{
    Tensor<float> x_copy(inputWarp.inter_value.ElemNum(), inputWarp.inter_value.ElemLen(), device, name+"_copy");
    x_copy.copy(0, inputWarp.inter_value, 0, inputWarp.inter_value.Size());

    inputWarp.inter_value = layers.at("input_layernorm")->forward(inputWarp.inter_value);
    
    layers.at("self_attn")->forward(inputWarp);
    F->add(inputWarp.inter_value, inputWarp.inter_value, x_copy, x_copy.Size());

    x_copy.copy(0, inputWarp.inter_value, 0, inputWarp.inter_value.Size());
    inputWarp.inter_value = layers.at("post_attention_layernorm")->forward(inputWarp.inter_value);

    inputWarp.inter_value = layers.at("mlp")->forward(inputWarp.inter_value);
    F->add(inputWarp.inter_value, inputWarp.inter_value, x_copy, x_copy.Size());
}

#endif // DECODERLAYER_H