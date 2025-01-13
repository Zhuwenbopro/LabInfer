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

    virtual void to(const std::string& _device) {
        if(device == _device) return;

        for (auto& [_name, param] : params) {
            param.to(_device);
        }

        F = DeviceManager::getInstance().getDevice(_device)->F;

        for (auto& [_name, layer] : layers) {
            layer->to(_device);
        }

        device = _device;
    }

    // void load_state(char * filename, bool tie_weights = false);

    // Layer& load_weights(std::unordered_map<std::string, std::shared_ptr<float []>>& weights);

    // Parameter& Param(const std::string& _name) { return params.at(_name); }

    virtual Tensor<float> forward(Tensor<float>& x1, Tensor<float>& x2) {
        throw std::logic_error(name + " Tensor<float> forward(Tensor<float>& x1, Tensor<float>& x2) not implemented.");
    }

    virtual Tensor<float> forward(Tensor<float>& x1, Tensor<int>& x2) {
        throw std::logic_error(name + " Tensor<float> forward(Tensor<float>& x1, Tensor<float>& x2) not implemented.");
    }

    // 把下面这两个函数合并
    virtual Tensor<float> forward(Tensor<float>& x) {
        throw std::logic_error(name + " Tensor<float> forward(Tensor<float>& x) not implemented.");
    }

    virtual Tensor<float> forward(Tensor<int>& x) {
        throw std::logic_error(name + " Tensor<float> forward(Tensor<float>& x) not implemented.");
    }

    virtual void forward(InputWarp& input) {
        throw std::logic_error(name + " Tensor<float> forward(Tensor<float>& x) not implemented.");
    }

    virtual void add_layer(Layer* layer, const std::string& name = "x") { 
        throw std::logic_error(name + " add_layer(Layer* layer) not implemented."); 
    }

protected:
    Function* F;
    // FIXME : 解除与 float 的耦合
    std::unordered_map<std::string, Parameter<float>> params;
    // 用指针才能用多态
    std::unordered_map<std::string, Layer*> layers;
    std::string device;
    std::string name;

private:
    void remove_prefix_from_keys(std::unordered_map<std::string, std::shared_ptr<float>>& state_map, const std::string& prefix);
// FIXME : just test
public:
    // FIXME : float 解耦
    virtual void load(std::unordered_map<std::string, std::shared_ptr<float>>& state_map) {
        remove_prefix_from_keys(state_map, name + ".");

        for (auto& [_name, param] : params) {
            auto it = state_map.find(_name);
            if (it != state_map.end()) {
                param.setValue(it->second);
                state_map.erase(it);
            } else {
                throw std::runtime_error("Layer " + name + "'s param '" + _name + "' not found in weights.");
            }
        }

        for (auto& [name, ptr_layer] : layers) {
            ptr_layer->load(state_map);
        }
    }

    // FIXME : to be delete
    void printParam() { 
        for (auto& [_name, param] : params) {
            std::cout << "\n\nthis is  Layer '" + name << "' in Param '" << _name << "' " << param << "\n\n";
            param.to("cpu");
            for(int i = 1; i <= param.Size(); i++) {
                std::cout << param[i];
                if(i%param.ElemLen() == 0) std::cout << "\n";
                else std::cout << " ";
                if(i > 100) break;
            }
            param.to(device);
        }

        for (auto& [_name, layer] : layers) {
            layer->printParam();
        }
    }
};


#endif // LAYER_H