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
    void forward(InputWarp& inputWarp) override;

    // 虚析构函数
    virtual ~Embedding() = default;

private:
    size_t vocab_size;
    size_t hidden_size;
};

#endif // EMBEDDING_H