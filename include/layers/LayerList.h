#ifndef LAYERLIST_H
#define LAYERLIST_H

#include "Layer.h"
#include <vector>

class LayerList : public Layer
{
public:
    LayerList(const std::string& _name = "layers");
    void forward(InputWarp& inputWarp) override;
    virtual ~LayerList() = default;

    void add_layer(Layer* layer, const std::string& name) override;
private:
    std::vector<std::string> layers_name;
};

void LayerList::forward(InputWarp& inputWarp) {
    for(int i = 0; i < layers_name.size(); i++) {
        layers.at(layers_name[i])->forward(inputWarp);
    }
}

LayerList::LayerList(const std::string& _name) : Layer("cpu", _name) { }

void LayerList::add_layer(Layer* layer, const std::string& _name) {
    layers_name.push_back(_name);
    layer->setName(_name);
    layers.emplace(_name, layer);
}
#endif