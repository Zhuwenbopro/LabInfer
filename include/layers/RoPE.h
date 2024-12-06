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
    Tensor forward(Tensor& x) override;

    // 虚析构函数
    virtual ~RoPE() = default;

private:
// FIXME : 这里应该是config里面带的，不能写死
    static bool init;
    size_t max_pos = 32;
    size_t head_dim;
    float rope_theta = 500000;
    int factor = 32;
    int low_freq_factor = 1;
    int high_freq_factor = 4;
    int old_context_len = 8192;
};

bool RoPE::init = false;


RoPE::RoPE(const size_t _head_dim, const std::string& _name) : Layer("cpu", _name), head_dim(_head_dim)
{
    size_t dim = head_dim / 2;
    Manager& manager = Manager::getInstance();

    if(!init) {
        params.emplace("cos", Parameter("cpu:cos", max_pos, dim, "cpu", true));
        params.emplace("sin", Parameter("cpu:sin", max_pos, dim, "cpu", true));
        Parameter& _cos = params.at("cos");
        Parameter& _sin = params.at("sin");

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
                inv_freq_llama[i] = inv_freq_div_factor[i];
            } else if (wavelen[i] < high_freq_wavelen) {
                inv_freq_llama[i] = inv_freq[i];
            } else { // Between high_freq_wavelen and low_freq_wavelen
                float smooth_factor = (old_context_len / wavelen[i] - low_freq_factor) / (high_freq_factor - low_freq_factor);
                float smoothed_inv_freq = (1.0f - smooth_factor) * inv_freq_div_factor[i] + smooth_factor * inv_freq[i];
                inv_freq_llama[i] = smoothed_inv_freq;
            }
        }

        for(int p = 0; p < max_pos; p++) {
            int No = p * dim;
            for(int d = 0; d < dim; d++) {
                _cos[No + d] = cosf(p * inv_freq_llama[d]);
                _sin[No + d] = sinf(p * inv_freq_llama[d]);
            }
        }

        manager.RegisteMem(device+":cos", _cos.sharedPtr());
        manager.RegisteMem(device+":sin", _sin.sharedPtr());

        init = true;
    }else {
        params.emplace("cos", Parameter(device+":cos", max_pos, dim, "cpu"));
        params.emplace("sin", Parameter(device+":sin", max_pos, dim, "cpu"));
        params.at("cos").setValue(manager.GetMem(device+":cos"));
        params.at("sin").setValue(manager.GetMem(device+":sin"));
    }
    
    params.at("cos").setShared();
    params.at("sin").setShared();
}

Tensor RoPE::forward(Tensor& x)
{
    if(x.elemNum() != x.Pos().elemNum()) {
        throw std::logic_error("pos num does not match x.elemNum()"); 
    }

    F.get().apply_rope(x, x.Pos(), params.at("cos"), params.at("sin"), x.elemLen(), head_dim/2, x.elemNum());

    return x;
}


#endif // ROPE_H