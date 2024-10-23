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
    Embedding(Config& config);

    Embedding(const size_t vocab_size, const size_t hidden_size, const std::string& _name = "embed_tokens");

    // 覆盖基类的 forward 方法
    void forward(Tensor& y, Tensor& x) override;

    // 虚析构函数
    virtual ~Embedding() = default;

private:
    size_t vocab_size;
    size_t hidden_size;
};

Embedding::Embedding(Config& config) : Layer("cpu", "embed_tokens")
{
    vocab_size = config.get("vocab_size").get<size_t>();
    hidden_size = config.get("hidden_size").get<size_t>();
    
    params.emplace("weight", Parameter("weight",{vocab_size, hidden_size}, "cpu"));
}


Embedding::Embedding(const size_t _vocab_size, const size_t _hidden_size, const std::string& _name) : Layer("cpu", _name)
{
    vocab_size = _vocab_size;
    hidden_size = _hidden_size;
    
    params.emplace("weight", Parameter("weight",{_vocab_size, _hidden_size}, "cpu"));
}

void Embedding::forward(Tensor& y, Tensor& x)
{
    Parameter& weight = params.at("weight");
    // 使用它们进行运算
    F.get().embedding(y, x, weight, hidden_size, x.Size());
}

#endif // EMBEDDING_H