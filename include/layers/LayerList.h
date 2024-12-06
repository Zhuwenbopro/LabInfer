#ifndef LAYERLIST_H
#define LAYERLIST_H

#include "Layer.h"

class LayerList : public Layer
{
public:
    LayerList();
    Tensor forward(Tensor& x) override;
    virtual ~LayerList() = default;

    void add_layer(Layer* layer, const std::string& name) override;
private:
    size_t layer_num;
};

Tensor LayerList::forward(Tensor& x) {
    Tensor y = x;
    for(int i = 0; i < layer_num; i++) {
        y = layers.at(std::to_string(i))->forward(y);
    }
    return y;
}

LayerList::LayerList() : Layer("cpu", "layers"), layer_num(0) { }

void LayerList::add_layer(Layer* layer, const std::string& name) {
    layers.emplace(std::to_string(layer_num++), layer);
}
#endif