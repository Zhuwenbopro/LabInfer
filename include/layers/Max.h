#ifndef MAX_LAYER_H
#define MAX_LAYER_H

#include "Layer.h"

class Max : public Layer {
public:
    void forward(InputWarp& inputWarp) override;
};

void Max::forward(InputWarp& inputWarp) {

}

#endif