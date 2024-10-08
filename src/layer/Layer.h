// Layer.h

#ifndef LAYER_H
#define LAYER_H

#include "Manager.h"
#include "Parameter.h"
#include "Tensor.h"
#include <stdexcept>
#include <vector>
#include <unordered_map>
#include <functional> 

class Layer {
public:
    Layer(const std::string& _device, const std::string& _name) : F(Manager::getInstance().getFunction(_device)) { }

    const std::string& Name() const { return name; }
    void setName(const std::string& _name){ name = _name; }

    const std::string& Device() const { return device; }
    void setDevice(const std::string& devName){ device = devName; }

    void to(const std::string& new_dev) {
        if(new_dev == device) return;
        
        for (auto& [name, param] : params) {
            param.to(new_dev);
        }
        
        F = std::ref(Manager::getInstance().getFunction(new_dev));

        device = new_dev;
        
        for (auto& [name, ptr_layer] : layers) {
            ptr_layer->to(new_dev);
        }
    }

    void load_state(std::unordered_map<std::string, float*>& state_map) {
        
        for (auto& [name, param] : params) {
            auto it = state_map.find(param.Name());
            if (it != state_map.end()) {
                param.setData(it->second);
            } else {
                // 这里到时候直接弹出来报错
                std::cout << "Key not found!!!!" << std::endl;
            }
        }

        for (auto& [name, ptr_layer] : layers) {
            ptr_layer->load_state(state_map);
        }
    }

    virtual void forward(Tensor& x) {
        throw std::logic_error("forward(Tensor& x) not implemented.");
    }

    virtual void forward(Tensor& y, Tensor& x) {
        throw std::logic_error("forward(Tensor& y, Tensor& x) not implemented.");
    }

    virtual void forward(Tensor& y, Tensor& x1, Tensor& x2) {
        throw std::logic_error("forward(Tensor& y, Tensor& x1, Tensor& x2) not implemented.");
    }

    // 虚析构函数，确保派生类的析构函数被调用
    virtual ~Layer() = default;

protected:
    std::reference_wrapper<Function> F;
    std::unordered_map<std::string, Parameter> params;
    std::unordered_map<std::string, Layer*> layers;
    std::string device;
    std::string name;

};

#endif // LAYER_H