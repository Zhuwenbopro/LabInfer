#ifndef ATTENTION_H
#define ATTENTION_H

#include "Layer.h"
#include "Linear.h"
#include "RoPE.h"
#include "RMSNorm.h"
#include "Config.h"
#include "Cache.h"

class Attention : public Layer {
public:
    Attention() = delete;
    Attention(const size_t attn_head, const size_t kv_head, const size_t hidden_size, const size_t _max_len = 250);

    void forward(InputWarp& inputWarp) override;

    void to(const std::string& _device) override;

    virtual ~Attention() = default;

private:
    size_t head_dim;
    size_t attn_head;
    size_t kv_head;
    size_t hidden_size;
    size_t q_dim;
    size_t kv_dim;

    Cache k_cache;
    Cache v_cache;
    // FIXME：max_len to be dynamic
    size_t max_len = 250;
};
#endif // ATTENTION_H