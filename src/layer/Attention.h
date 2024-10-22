#ifndef ATTENTION_H
#define ATTENTION_H

#include "Layer.h"
#include "Linear.h"
#include "RoPE.h"
#include "RMSNorm.h"
#include "Config.h"

class Attention : public Layer {
public:
    // 构造函数，初始化线性层的输入和输出尺寸
    // 删除默认构造函数
    Attention() = delete;
    // Linear(const Config& config, const std::string& name = "Linear");
    Attention(const std::string& _name = "self_attention");

    // 覆盖基类的 forward 方法
    void forward(Tensor& y, Tensor& x, Tensor& pos) override;

    // 虚析构函数
    virtual ~Attention() = default;

private:
    size_t head_dim;
    size_t attn_head;
    size_t kv_head;
    size_t hidden_size;
    size_t q_dim;
    size_t kv_dim;
};

// 初始化不分配内存，等到load的时候再分配
Attention::Attention(const std::string& _name) : Layer("cpu", _name)
{   
    head_dim = 64;
    attn_head = 32;
    kv_head = 8;
    hidden_size = 2048;
    q_dim = head_dim*attn_head;
    kv_dim = head_dim*kv_head;
    
    layers.emplace("q_linear", new Linear(hidden_size, q_dim, "q_proj"));
    layers.emplace("k_linear", new Linear(hidden_size, kv_dim, "k_proj"));
    layers.emplace("v_linear", new Linear(hidden_size, kv_dim, "v_proj"));
    layers.emplace("o_linear", new Linear(q_dim, hidden_size, "o_proj"));
    layers.emplace("rope", new RoPE(head_dim));
    
}

// 进去的 x 会变，y 可以等于 x
void Attention::forward(Tensor& y, Tensor& x, Tensor& pos)
{   
    Tensor q("query", x.Shape(), device, true, x.Seq());
    Tensor k("key", {x.batchSize(), kv_dim}, device, true, x.Seq());
    Tensor v("query", {x.batchSize(), kv_dim}, device, true, x.Seq());

    layers.at("q_linear")->forward(q, x);
    layers.at("k_linear")->forward(k, x);
    layers.at("v_linear")->forward(v, x);

// FIXME: 这里的 pos 在 rope 里需要在 device 上，但在 attention 中需要在 CPU 里
    pos.to(device);
    layers.at("rope")->forward(q, pos);
    layers.at("rope")->forward(k, pos);
    pos.to("cpu");

    Tensor o("output", x.Shape(), device, true, x.Seq());
    for(int p = 0; p < pos.Size(); p++) {
        float* output = o + p * hidden_size;
        float* query = q + p * hidden_size;
        // 这里的 k、v 从 cache 里搞
        F.get().maksed_attention(output, query, k, v, head_dim, attn_head, kv_head, pos[p]);
    }

    layers.at("o_linear")->forward(y, o);
}

#endif // ATTENTION_H