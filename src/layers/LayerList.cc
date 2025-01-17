#include "layers/LayerList.h"

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