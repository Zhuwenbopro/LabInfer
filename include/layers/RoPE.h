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
    void forward(InputWarp& inputWarp) override;

    // 虚析构函数
    virtual ~RoPE() = default;

    // 不需要 load weights
    void load(std::unordered_map<std::string, std::shared_ptr<float>>& state_map) override { };

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
    params.emplace("inv_freq", Parameter<float>(1, dim, "cpu", "inv", true));
    
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

void RoPE::forward(InputWarp& inputWarp)
{
    F->apply_rope(inputWarp.inter_value, inputWarp.pos, params.at("inv_freq"), inputWarp.inter_value.ElemLen(), head_dim, inputWarp.inter_value.ElemNum());
}


#endif // ROPE_H