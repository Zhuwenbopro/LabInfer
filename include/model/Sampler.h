#ifndef SAMPLER_H
#define SAMPLER_H
#include "InputWarp.h"

class Sampler {
private:
    std::string device;
    Function* F;
    bool do_sampling;
    int k;                  // Top-k参数
    float p;                // Top-p (Nucleus) 参数
    float t;      // 温度参数，用于控制采样平滑度

public:
    Sampler(float temperature, int topK, float topP, bool sampling = true) 
        : t(temperature), k(topK), p(topP), do_sampling(sampling), device("cpu") {
        F = DeviceManager::getInstance().getDevice(device)->F;
    }

    ~Sampler() = default;

    void sample(InputWarp& inputWarp) {
        if(do_sampling) {
            F->topK_topP_sampling(inputWarp.output_ids, inputWarp.inter_value, 
                t, k, p, inputWarp.inter_value.ElemLen(), inputWarp.inter_value.ElemNum());
        } else {
            F->max_index(inputWarp.output_ids, inputWarp.inter_value, inputWarp.inter_value.ElemLen(), 1);
        }
        
    }

    void to(const std::string& _device) { 
        device = _device;
        F = DeviceManager::getInstance().getDevice(_device)->F;
    }
};

#endif