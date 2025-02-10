// Layer.h
#ifndef LAYER_H
#define LAYER_H

#include "InputWarp.h"
#include "Parameter.h"
#include "Tensor.h"
#include "Config.h"
#include <stdexcept>
#include <vector>
#include <unordered_map>

class Layer {
public:
    Layer(const std::string& _device, const std::string& _name) : 
            device(_device), name(_name), F(DeviceManager::getInstance().getDevice(_device)->F) { }

    // 虚析构函数，确保派生类的析构函数被调用
    virtual ~Layer() = default; 

    const std::string& Name() const { return name; }
    void setName(const std::string& _name) { name = _name; }
    const std::string& Device() const { return device; }

    virtual void to(const std::string& _device);

    // void load_state(char * filename, bool tie_weights = false);

    // Layer& load_weights(std::unordered_map<std::string, std::shared_ptr<float []>>& weights);

    // Parameter& Param(const std::string& _name) { return params.at(_name); }

    virtual void forward(InputWarp& input);

    virtual void add_layer(Layer* layer, const std::string& name = "x");

protected:
    Function* F;
    // FIXME : 解除与 float 的耦合
    std::unordered_map<std::string, Parameter<float>> params;
    // 用指针才能用多态
    std::unordered_map<std::string, Layer*> layers;
    std::string device;
    std::string name;

private:
    std::unordered_map<std::string, std::shared_ptr<void>> remove_prefix_from_keys(std::unordered_map<std::string, std::shared_ptr<void>>& state_map, const std::string& prefix);

public:
    virtual void load(std::unordered_map<std::string, std::shared_ptr<void>>& state_map);
};


#endif // LAYER_H