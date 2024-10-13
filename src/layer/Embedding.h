#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "Layer.h"
#include "Config.h"

class Embedding : public Layer {
public:
    // 构造函数，初始化线性层的输入和输出尺寸
    // 删除默认构造函数
    Embedding() = delete;
    // Linear(const Config& config, const std::string& name = "Linear");
    Embedding(const size_t vocab_size, const size_t hidden_size, const std::string& _name = "embed_tokens");

    // 覆盖基类的 forward 方法
    void forward(Tensor& y, Tensor& x) override;

    // 虚析构函数
    virtual ~Embedding() = default;

private:
    size_t vocab_size;
    size_t hidden_size;
};

// 初始化不分配内存，等到load的时候再分配
Embedding::Embedding(const size_t _vocab_size, const size_t _hidden_size, const std::string& _name) : Layer("cpu", _name)
{
    vocab_size = _vocab_size;
    hidden_size = _hidden_size;
    
    params.emplace("weight", Parameter("weight",{_vocab_size, _hidden_size}, "cpu"));
}

// 这里写的代码很冗长 是因为 unordered_map 在调用 temps["output"] 时 会调用默认构造函数，
// 但是Tensor和parameter没有默认构造函数 会报错
void Embedding::forward(Tensor& y, Tensor& x)
{
    Parameter& weight = params.at("weight");
    // 使用它们进行运算
    F.get().embedding(y, x, weight, hidden_size, x.Size());
}

#endif // EMBEDDING_H