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
    void load(std::unordered_map<std::string, std::shared_ptr<void>>& state_map) override { };

private:
// FIXME : 这里应该是config里面带的，不能写死
    size_t head_dim;
    float rope_theta = 500000;
    int factor = 32;
    int low_freq_factor = 1;
    int high_freq_factor = 4;
    int old_context_len = 8192;
};


#endif // ROPE_H