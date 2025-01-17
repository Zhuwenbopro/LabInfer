// DecoderLayer.h

#ifndef DECODERLAYER_H
#define DECODERLAYER_H

#include "Layer.h"
#include "Config.h"
#include "Attention.h"
#include "Mlp.h"

class DecoderLayer : public Layer {
public:
    DecoderLayer() = delete;

    DecoderLayer(
        const size_t attn_head, 
        const size_t kv_head, 
        const size_t hidden_size,
        const size_t intermediate_size,
        const size_t _max_len = 250, 
        const float epsilon = 1e-5
    );

    void forward(InputWarp& inputWarp) override;

    // 虚析构函数
    virtual ~DecoderLayer() = default;
};

#endif // DECODERLAYER_H