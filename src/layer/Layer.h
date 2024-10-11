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
    Layer(const std::string& _device, const std::string& _name) : 
            device(_device), name(_name), F(Manager::getInstance().getFunction(_device)) { }

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

        remove_prefix_from_keys(state_map, name + "_");

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

private:
    void remove_prefix_from_keys(std::unordered_map<std::string, float*>& state_map, const std::string& prefix) {
        std::unordered_map<std::string, float*> updated_map;

        // 遍历 state_map
        for (auto& pair : state_map) {
            std::string key = pair.first;

            // 检查是否以 prefix 开头
            if (key.rfind(prefix, 0) == 0) {  // rfind 返回0表示从首部开始匹配
                // 去掉 prefix 部分
                std::string new_key = key.substr(prefix.length());

                // 将新的 key 和对应的值插入到 updated_map
                updated_map[new_key] = pair.second;
            } else {
                // 如果不匹配 prefix，保留原来的键值对
                updated_map[key] = pair.second;
            }
        }

        // 用更新后的 map 替换原来的 map
        state_map = std::move(updated_map);
    }
};

#endif // LAYER_H