// Layer.h

#ifndef LAYER_H
#define LAYER_H

#include "Manager.h"
#include "Parameter.h"
#include "Tensor.h"
#include "Config.h"
#include <stdexcept>
#include <vector>
#include <unordered_map>
#include <functional> 

class Layer;

class Layer {
public:
    Layer(const std::string& _device, const std::string& _name) : 
            device(_device), name(_name), F(Manager::getInstance().getFunction(_device)) { }

    const std::string& Name() const { return name; }
    void setName(const std::string& _name) { name = _name; }
    const std::string& Device() const { return device; }

    virtual void to(const std::string& new_dev);

    void load_state(char * filename, bool tie_weights = false);

    Layer& load_weights(std::unordered_map<std::string, std::shared_ptr<float []>>& weights);

    Parameter& Param(const std::string& _name) { return params.at(_name); }

    // virtual void forward(Tensor& x) {
    //     throw std::logic_error(name + " forward(Tensor& x) not implemented.");
    // }

    virtual void forward(Tensor& y, Tensor& x) {
        throw std::logic_error(name + " forward(Tensor& y, Tensor& x) not implemented.");
    }

    virtual void forward(Tensor& y, Tensor& x1, Tensor& x2) {
        throw std::logic_error(name + " forward(Tensor& y, Tensor& x1, Tensor& x2) not implemented.");
    }

    virtual Tensor forward(Tensor& x) {
        throw std::logic_error(name + " Tensor& forward(Tensor& x) not implemented.");
    }

    virtual void add_layer(Layer* layer, const std::string& name = "x") { 
        throw std::logic_error(name + " add_layer(Layer* layer) not implemented."); 
    }

    // 虚析构函数，确保派生类的析构函数被调用
    virtual ~Layer() = default;

protected:
    std::reference_wrapper<Function> F;
    std::unordered_map<std::string, Parameter> params;
    // 用指针才能用多态
    // FIXME ： 想办法用引用解决
    std::unordered_map<std::string, Layer*> layers;
    std::string device;
    std::string name;

private:
    void remove_prefix_from_keys(std::unordered_map<std::string, std::shared_ptr<float []>>& state_map, const std::string& prefix);
// FIXME : just test
public:
    void load_state(std::unordered_map<std::string, std::shared_ptr<float []>>& state_map);
};


#endif // LAYER_H