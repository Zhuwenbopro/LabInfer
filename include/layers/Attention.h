#ifndef ATTENTION_H
#define ATTENTION_H

#include "Layer.h"
#include "Linear.h"
#include "RoPE.h"
#include "RMSNorm.h"
#include "Config.h"
#include "Cache.h"

class Attention : public Layer {
public:
    Attention() = delete;
    Attention(const size_t attn_head, const size_t kv_head, const size_t hidden_size, const size_t _max_len = 250);

    void forward(InputWarp& inputWarp) override;

    void to(const std::string& _device) override;

    virtual ~Attention() = default;

private:
    size_t head_dim;
    size_t attn_head;
    size_t kv_head;
    size_t hidden_size;
    size_t q_dim;
    size_t kv_dim;

    Cache k_cache;
    Cache v_cache;
    // FIXME：max_len to be dynamic
    size_t max_len = 250;
};

Attention::Attention(
    const size_t _attn_head, 
    const size_t _kv_head, 
    const size_t _hidden_size,
    const size_t _max_len
) : Layer("cpu", "self_attn"), 
    head_dim(_hidden_size/_attn_head), 
    attn_head(_attn_head), 
    kv_head(_kv_head), 
    hidden_size(_hidden_size), 
    k_cache(Cache(_hidden_size, _max_len)),
    v_cache(Cache(_hidden_size, _max_len))
{

    q_dim = head_dim*_attn_head; 
    kv_dim = head_dim*_kv_head;
    layers.emplace("q_linear", new Linear(hidden_size, q_dim, "q_linear"));
    layers.emplace("k_linear", new Linear(hidden_size, kv_dim, "k_linear"));
    layers.emplace("v_linear", new Linear(hidden_size, kv_dim, "v_linear"));
    layers.emplace("o_linear", new Linear(q_dim, hidden_size, "o_linear"));
    layers.emplace("rope", new RoPE(head_dim));
}


void Attention::forward(InputWarp& inputWarp) {
    size_t uid = inputWarp.uid;
    Tensor x = inputWarp.inter_value;

    layers.at("q_linear")->forward(inputWarp);
    layers.at("rope")->forward(inputWarp);
    Tensor q = inputWarp.inter_value;
    inputWarp.inter_value = x;

    layers.at("k_linear")->forward(inputWarp);
    layers.at("rope")->forward(inputWarp);
    Tensor k = inputWarp.inter_value;
    inputWarp.inter_value = x;

    layers.at("v_linear")->forward(inputWarp);
    Tensor v = inputWarp.inter_value;

    int rep = attn_head/kv_head;
    if(rep != 1) {
        Tensor<float> k_exp(x.ElemNum(), q.ElemLen(), device, "k_exp");
        Tensor<float> v_exp(x.ElemNum(), q.ElemLen(), device, "v_exp");
        F->repeat_kv(k_exp, k, head_dim, attn_head/kv_head, k.Size());
        F->repeat_kv(v_exp, v, head_dim, attn_head/kv_head, k.Size());
        k = k_exp;
        v = v_exp;
    }
    k_cache.add(uid, k, inputWarp.start_pos);
    v_cache.add(uid, v, inputWarp.start_pos);

    Tensor<float> o(q.ElemNum(), q.ElemLen(), q.Device(), "attn_output");
    F->masked_attention(o, q, k_cache.get(uid), v_cache.get(uid), nullptr, inputWarp.pos, head_dim, attn_head, q.ElemNum(), k_cache.Len(uid));
    inputWarp.inter_value = o;
    layers.at("o_linear")->forward(inputWarp);
}

void Attention::to(const std::string& _device) {
    if(device == _device) return;

    for (auto& [_name, param] : params) {
        param.to(_device);
    }

    F = DeviceManager::getInstance().getDevice(_device)->F;

    for (auto& [_name, layer] : layers) {
        layer->to(_device);
    }

    device = _device;

    k_cache.to(_device);
    v_cache.to(_device);
}

#endif // ATTENTION_H