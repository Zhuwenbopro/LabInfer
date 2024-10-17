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
    
    layers.emplace("rms_norm", new RMSNorm(hidden_size));
    layers.emplace("q_linear", new Linear(hidden_size, q_dim, "q_linear"));
    layers.emplace("k_linear", new Linear(hidden_size, kv_dim, "k_linear"));
    layers.emplace("v_linear", new Linear(hidden_size, kv_dim, "v_linear"));
    layers.emplace("rope", new RoPE(head_dim));
    layers.emplace("o_linear", new Linear(q_dim, hidden_size, "o_linear"));
}

void Attention::forward(Tensor& y, Tensor& x, Tensor& pos)
{
    Layer* rms_norm = layers.at("rms_norm");
    rms_norm->forward(x);
    
    Layer* q_linear = layers.at("q_linear");
    Layer* k_linear = layers.at("k_linear");
    Layer* v_linear = layers.at("v_linear");
    Tensor q("query", x.Shape(), device, true, x.Seq());
    Tensor k("key", {x.Shape()[0], kv_dim}, device, true, x.Seq());
    Tensor v("query", {x.Shape()[0], kv_dim}, device, true, x.Seq());

    q_linear->forward(q, x);
    k_linear->forward(k, x);
    v_linear->forward(v, x);

    Layer* rope = layers.at("rope");
    rope->forward(q, pos);
    rope->forward(k, pos);

    Tensor o("output", x.Shape(), device, true, x.Seq());
    for(int p = 0; p < pos.Size(); p++) {
        float* output = o + p * hidden_size;
        float* query = q + p * hidden_size;
        // 这里的 k、v 从 cache 里搞
        F.get().maksed_attention(output, query, k, v, head_dim, attn_head, kv_head, pos[p]);
    }
    std::cout << "masked attention finsihed" << std::endl;
    Layer* o_linear = layers.at("o_linear");
    o_linear->forward(y, o);
}

#endif // ATTENTION_H