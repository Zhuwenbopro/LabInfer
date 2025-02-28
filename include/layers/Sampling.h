#ifndef SAMPLING_H
#define SAMPLING_H

#include "Layer.h"

class Sampling : public Layer {
public:
    // 构造函数，初始化线性层的输入和输出尺寸
    // 删除默认构造函数
    Sampling() = delete;
    Sampling(float temperature, int topK, float topP, bool sampling = true, const std::string& _name = "sampling");

    // 覆盖基类的 forward 方法
    void forward(InputWarp& inputWarp) override;

    // 虚析构函数
    virtual ~Sampling() = default;
    
    // 不需要 load 参数
    void load(std::unordered_map<std::string, std::shared_ptr<void>>& state_map) override { };

private:
    bool do_sampling;
    int k;                  // Top-k参数
    float p;                // Top-p (Nucleus) 参数
    float t;      // 温度参数，用于控制采样平滑度
};

#endif