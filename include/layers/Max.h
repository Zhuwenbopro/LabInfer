#ifndef MAX_LAYER_H
#define MAX_LAYER_H

#include "Layer.h"

class Max : public Layer {
public:
    Max() = delete;
    Max(const size_t _vocab_size);
    void forward(InputWarp& inputWarp) override;
private:
    size_t vocab_size;
};

#endif