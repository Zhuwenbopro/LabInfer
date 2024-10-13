#include "Layer.h"

void Layer::to(const std::string& new_dev) {
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

void Layer::load_state(std::unordered_map<std::string, std::shared_ptr<float []>>& state_map) {
    remove_prefix_from_keys(state_map, name + ".");

    for (auto& [name, param] : params) {
        auto it = state_map.find(param.Name());
        if (it != state_map.end()) {
            param.setValue(it->second);
            state_map.erase(it);
        } else {
            // 这里到时候直接弹出来报错
            std::cout << "Key not found!!!!" << std::endl;
        }
    }

    for (auto& [name, ptr_layer] : layers) {
        ptr_layer->load_state(state_map);
    }
}

void Layer::remove_prefix_from_keys(std::unordered_map<std::string, 
            std::shared_ptr<float []>>& state_map, const std::string& prefix) {
    std::unordered_map<std::string, std::shared_ptr<float []>> updated_map;
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