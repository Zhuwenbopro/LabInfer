#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "Layer.h"
#include "Config.h"

class Embedding : public Layer {
public:
    // 构造函数，初始化线性层的输入和输出尺寸
    // 删除默认构造函数
    Embedding() = delete;
    Embedding(const size_t vocab_size, const size_t hidden_size, const std::string& _name = "embed_tokens");

    // 覆盖基类的 forward 方法
    Tensor<float> forward(Tensor<int>& x) override;

    // 虚析构函数
    virtual ~Embedding() = default;

private:
    size_t vocab_size;
    size_t hidden_size;
};


Embedding::Embedding(const size_t _vocab_size, const size_t _hidden_size, const std::string& _name) : Layer("cpu", _name)
{
    vocab_size = _vocab_size;
    hidden_size = _hidden_size;
    
    params.emplace("weight", Parameter<float>(_vocab_size, _hidden_size, "cpu", "weight"));
}

Tensor<float> Embedding::forward(Tensor<int>& x)
{
    Tensor<float> y(x.ElemNum(), hidden_size, x.Device(), name + "_output");
    F->embedding(y, x, params.at("weight"), hidden_size, x.Size());
    return y;
}

#endif // EMBEDDING_H