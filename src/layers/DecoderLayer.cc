#include "layers/DecoderLayer.h"

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

    layers.at("input_layernorm")->forward(inputWarp);
    
    layers.at("self_attn")->forward(inputWarp);
    F->add(inputWarp.inter_value, inputWarp.inter_value, x_copy, x_copy.Size());

    x_copy.copy(0, inputWarp.inter_value, 0, inputWarp.inter_value.Size());
    layers.at("post_attention_layernorm")->forward(inputWarp);

    layers.at("mlp")->forward(inputWarp);
    F->add(inputWarp.inter_value, inputWarp.inter_value, x_copy, x_copy.Size());
}
