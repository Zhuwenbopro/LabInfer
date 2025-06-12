      
#include "ModelConfig.hpp"
#include <fstream>
#include <iostream>

namespace llm_frame {

bool ModelConfig::loadFromFile(const std::string& filepath) {
    std::ifstream file_stream(filepath);
    if (!file_stream.is_open()) {
        std::cerr << "Error: Cannot open config file: " << filepath << std::endl;
        return false;
    }

    try {
        // 1. 将文件内容解析成一个 nlohmann::json 对象
        nlohmann::json json_obj;
        file_stream >> json_obj;

        // 2. 使用 nlohmann/json 的 from_json 功能将 json 对象转换为 ModelConfig 对象
        //    我们之前定义的 NLOHMANN_DEFINE_TYPE_INTRUSIVE 宏使得这行代码能够工作
        *this = json_obj.get<ModelConfig>();

    } catch (const nlohmann::json::parse_error& e) {
        std::cerr << "Error: JSON parsing error in file " << filepath << ":\n"
                  << e.what() << std::endl;
        return false;
    } catch (const nlohmann::json::exception& e) {
        // 这个异常可能在键不存在或类型不匹配时抛出
        std::cerr << "Error: JSON data validation error in file " << filepath << ":\n"
                  << e.what() << std::endl;
        return false;
    }

    return true;
}

void ModelConfig::printConfig() const {
    std::cout << "--- Model Configuration ---" << std::endl;
    std::cout << "Model Type: " << model_type << std::endl;
    std::cout << "Hidden Size: " << hidden_size << std::endl;
    std::cout << "Number of Hidden Layers: " << num_hidden_layers << std::endl;
    std::cout << "Number of Attention Heads: " << num_attention_heads << std::endl;
    std::cout << "Number of Key-Value Heads: " << num_key_value_heads << std::endl;
    std::cout << "Vocabulary Size: " << vocab_size << std::endl;
    std::cout << "Torch DType: " << torch_dtype << std::endl;
    std::cout << "-------------------------" << std::endl;
}

} // namespace llm_frame

    