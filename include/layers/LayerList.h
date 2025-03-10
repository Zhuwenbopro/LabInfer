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

    virtual void add_layer(Layer* layer, const std::string& name) override;
    virtual Layer* get_layer(const std::string& name) override;
private:
    std::vector<std::string> layers_name;
};

#endif