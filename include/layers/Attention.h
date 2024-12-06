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
    // 构造函数，初始化线性层的输入和输出尺寸
    // 删除默认构造函数
    Attention() = delete;
    // Linear(const Config& config, const std::string& name = "Linear");
    Attention(Config& config);
    Attention(const size_t head_dim, const size_t attn_head, const size_t kv_head, const size_t hidden_size);

    // 覆盖基类的 forward 方法
    Tensor forward(Tensor& x) override;

    void to(const std::string& new_dev) override;

    // 虚析构函数
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

// 初始化不分配内存，等到load的时候再分配
Attention::Attention(Config& config) : Layer("cpu", "self_attn")
{
    head_dim = config.get<size_t>("head_dim");
    attn_head = config.get<size_t>("num_attention_heads");
    kv_head = config.get<size_t>("num_key_value_heads");
    hidden_size = config.get<size_t>("hidden_size");
    q_dim = head_dim*attn_head;
    kv_dim = head_dim*kv_head;

    layers.emplace("q_linear", new Linear(hidden_size, q_dim, "q_proj"));
    layers.emplace("k_linear", new Linear(hidden_size, kv_dim, "k_proj"));
    layers.emplace("v_linear", new Linear(hidden_size, kv_dim, "v_proj"));
    layers.emplace("o_linear", new Linear(q_dim, hidden_size, "o_proj"));
    layers.emplace("rope", new RoPE(head_dim));

    k_cache = Cache(kv_dim, max_len, "cpu");
    v_cache = Cache(kv_dim, max_len, "cpu");
}

Attention::Attention(const size_t _head_dim, const size_t _attn_head, const size_t _kv_head, const size_t _hidden_size) : Layer("cpu", "self_attn"), 
    head_dim(_head_dim), attn_head(_attn_head), kv_head(_kv_head), hidden_size(_hidden_size), q_dim(head_dim*attn_head), kv_dim(head_dim*kv_head)
{
    layers.emplace("q_linear", new Linear(hidden_size, q_dim, "q_proj"));
    layers.emplace("k_linear", new Linear(hidden_size, kv_dim, "k_proj"));
    layers.emplace("v_linear", new Linear(hidden_size, kv_dim, "v_proj"));
    layers.emplace("o_linear", new Linear(q_dim, hidden_size, "o_proj"));
    layers.emplace("rope", new RoPE(head_dim));

    k_cache = Cache(kv_dim, max_len, "cpu");
    v_cache = Cache(kv_dim, max_len, "cpu");
}

// 进去的 x 会变，y 可以等于 x
Tensor Attention::forward(Tensor& x)
{   
    Tensor q = layers.at("q_linear")->forward(x);
    Tensor k = layers.at("k_linear")->forward(x);
    Tensor v = layers.at("v_linear")->forward(x);


// FIXME: 这里的 pos 在 rope 里需要在 device 上，但在 attention 中需要在 CPU 里
    q = layers.at("rope")->forward(q);
    k = layers.at("rope")->forward(k);

    k_cache.add(k);
    v_cache.add(v);

    //Tensor o("output", x.Shape(), device, true, x.Seq());
    // prefill 阶段用的是 masked attention；decode 阶段用的是 kvcache
    int offset = 0;
    Tensor o(x, x.elemLen());
    for(auto pos : x.Position()) {
        for(int p = 0; p < pos.size(); p++) {
            F.get().maksed_attention(o + offset*hidden_size, q + offset*q_dim, k, v, head_dim, attn_head, kv_head, pos[p]);
            offset += 1;
        }
    }

    Tensor y = layers.at("o_linear")->forward(o);
    return y;
}

void Attention::to(const std::string& new_dev) {
    if(new_dev == device) return;
    
    for (auto& [_name, param] : params) {
        param.to(new_dev);
    }
    
    F = std::ref(Manager::getInstance().getFunction(new_dev));
    device = new_dev;
    
    for (auto& [_name, ptr_layer] : layers) {
        ptr_layer->to(new_dev);
    }

    k_cache.to(new_dev);
    v_cache.to(new_dev);
}

#endif // ATTENTION_H