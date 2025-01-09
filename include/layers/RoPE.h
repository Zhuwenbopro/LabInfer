// RoPE.h
#ifndef ROPE_H
#define ROPE_H

#include <cmath>
#include "Layer.h"
#include "Config.h"

class RoPE : public Layer {
public:
    // 构造函数，初始化线性层的输入和输出尺寸
    // 删除默认构造函数
    RoPE() = delete;
    // Linear(const Config& config, const std::string& name = "Linear");
    RoPE(const size_t _head_dim, const std::string& _name = "rotary_positional_embedding");

    // 覆盖基类的 forward 方法
    Tensor<float> forward(Tensor<float>& x, Tensor<int>& pos) override;

    // 虚析构函数
    virtual ~RoPE() = default;

    void load(std::unordered_map<std::string, std::shared_ptr<float>>& state_map) override {};

private:
// FIXME : 这里应该是config里面带的，不能写死
    size_t head_dim;
    float rope_theta = 500000;
    int factor = 32;
    int low_freq_factor = 1;
    int high_freq_factor = 4;
    int old_context_len = 8192;
};


RoPE::RoPE(const size_t _head_dim, const std::string& _name) : Layer("cpu", _name), head_dim(_head_dim)
{
    size_t dim = head_dim / 2;
    params.emplace("inv_freq", Parameter<float>(1, dim, "cpu", "inv"));

    float inv_freq[dim];
    float inv_freq_div_factor[dim];
    for (int k = 0; k < dim; k++) {
        inv_freq[k] = 1.0f / powf(rope_theta, k / (float)dim);
        inv_freq_div_factor[k] = inv_freq[k] / factor;
    }            

    float wavelen[dim];
    for (size_t i = 0; i < dim; ++i)
        wavelen[i] = 2.0f * M_PI / inv_freq[i];
    
    float low_freq_wavelen = old_context_len / low_freq_factor;
    float high_freq_wavelen = old_context_len / high_freq_factor;

    float inv_freq_llama[dim];
    
    for (size_t i = 0; i < dim; ++i) {
        if (wavelen[i] > low_freq_wavelen) {
            params.at("inv_freq")[i] = inv_freq_div_factor[i];
        } else if (wavelen[i] < high_freq_wavelen) {
            params.at("inv_freq")[i] = inv_freq[i];
        } else { // Between high_freq_wavelen and low_freq_wavelen
            float smooth_factor = (old_context_len / wavelen[i] - low_freq_factor) / (high_freq_factor - low_freq_factor);
            float smoothed_inv_freq = (1.0f - smooth_factor) * inv_freq_div_factor[i] + smooth_factor * inv_freq[i];
            params.at("inv_freq")[i] = smoothed_inv_freq;
        }
    }
}

Tensor<float> RoPE::forward(Tensor<float>& x, Tensor<int>& pos)
{
    if(x.ElemNum() != pos.ElemNum()) {
        throw std::logic_error("pos num does not match x.elemNum()"); 
    }

    F->apply_rope(x, pos, params.at("inv_freq"), x.ElemLen(), head_dim, x.ElemNum());

    return x;
}


#endif // ROPE_H