// Layer.h

#ifndef LAYER_H
#define LAYER_H

#include "Manager.h"
#include "Parameter.h"
#include "Tensor.h"
#include <vector>
#include <unordered_map>

class Layer {
public:
    Layer() : manager(Manager::getInstance()) {

    }

    const std::string& Name() const { return name; }
    void setName(const std::string& _name){ name = _name; }

    const std::string& Device() const { return device; }
    void setDevice(const std::string& devName){ device = devName; }

    void to(const std::string& new_dev) {
        if(new_dev == device) return;

        for(Parameter& param : params) {
            param.to(new_dev);
        }

        for(Layer& layer : layers) {
            layer.to(new_dev);
        }
    }

    void load_state(std::unordered_map<std::string, float*>& state_map) {
        for(Parameter& param : params) {
            auto it = state_map.find(param.Name());
            if (it != state_map.end()) {
                param.setData(it->second);
            } else {
                // 这里到时候直接弹出来报错
                std::cout << "Key not found!!!!" << std::endl;
            }
        }

        for(Layer& layer : layers) {
            layer.load_state(state_map);
        }
    }

    virtual std::vector<Tensor> forward(std::vector<Tensor>& inputs) = 0;

    // 虚析构函数，确保派生类的析构函数被调用
    virtual ~Layer() = default;

    // 禁止拷贝构造和拷贝赋值，避免浅拷贝问题
    Layer(const Layer&) = delete;
    Layer& operator=(const Layer&) = delete;

protected:
    Manager& manager;
    std::vector<Parameter> params;
    std::vector<Layer> layers;
    std::string device;
    std::string name;

};

#endif // LAYER_H