#include "layers/Layer.h"
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void Layer::forward(InputWarp& input) {
    throw std::logic_error(name + " Tensor<float> forward(Tensor<float>& x) not implemented.");
}

void Layer::add_layer(Layer* layer, const std::string& name) { 
    throw std::logic_error(name + " add_layer(Layer* layer) not implemented."); 
}

Layer* Layer::get_layer(const std::string& name) { 
    throw std::logic_error(name + " get_layer(Layer* layer) not implemented."); 
}

void Layer::to(const std::string& _device) {
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

std::unordered_map<std::string, std::shared_ptr<void>> Layer::remove_prefix_from_keys(std::unordered_map<std::string, std::shared_ptr<void>>& state_map, const std::string& prefix) {
    std::unordered_map<std::string, std::shared_ptr<void>> updated_map;
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
    // state_map = std::move(updated_map);
    return updated_map;
}

void Layer::load(std::unordered_map<std::string, std::shared_ptr<void>>& state_map) {
    std::unordered_map<std::string, std::shared_ptr<void>> update_map = remove_prefix_from_keys(state_map, name + ".");

    // std::cout << name << "  " << std::endl << std::endl;
    // for (auto& [_name, param] : update_map) {
    //     std::cout << _name << std::endl;
    // }
    // std::cout << std::endl << std::endl;

    for (auto& [_name, param] : params) {
        auto it = update_map.find(_name);
        if (it != update_map.end()) {
            param.setValue(it->second);
            update_map.erase(it);
        } else {
            throw std::runtime_error("Layer " + name + "'s param '" + _name + "' not found in weights.");
        }
    }

    for (auto& [name, ptr_layer] : layers) {
        ptr_layer->load(update_map);
    }

}