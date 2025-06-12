      
#pragma once

#include <string>
#include <vector>
#include "nlohmann/json.hpp" // 包含 nlohmann/json 库

// 定义命名空间来组织框架代码
namespace llm_frame {

class ModelConfig {
public:
    // --- 模型架构和类型 ---
    std::vector<std::string> architectures;
    std::string model_type;
    std::string torch_dtype;
    std::string transformers_version;

    // --- Tokenizer 相关 ---
    int bos_token_id;
    int eos_token_id;
    int vocab_size;

    // --- 模型维度与层数 ---
    int hidden_size;
    int intermediate_size;
    int num_hidden_layers;
    int num_attention_heads;
    int num_key_value_heads; // 用于 Grouped-Query Attention

    // --- 位置编码与Attention ---
    int max_position_embeddings;
    double rope_theta;
    double attention_dropout;

    // --- 归一化与初始化 ---
    double rms_norm_eps;
    double initializer_range;

    // --- KV Cache 与滑动窗口 ---
    bool use_cache;
    bool use_sliding_window;
    int sliding_window;
    int max_window_layers;

    // --- 其他 ---
    bool tie_word_embeddings;
    std::string hidden_act;

public:
    /**
     * @brief 从指定的JSON配置文件加载模型配置
     * @param filepath 配置文件的路径
     * @return 如果加载成功返回 true, 否则返回 false
     */
    bool loadFromFile(const std::string& filepath);

    /**
     * @brief 打印当前加载的配置信息，用于调试
     */
    void printConfig() const;
};

// 使用 nlohmann/json
// 这个宏会自动为 ModelConfig 类生成 from_json 和 to_json 函数。
// 它会根据成员变量的名称去JSON对象中寻找对应的键。
// 注意：这个宏要求所有要序列化的成员都是 public 的。
NLOHMANN_DEFINE_TYPE_INTRUSIVE(ModelConfig,
    architectures,
    attention_dropout,
    bos_token_id,
    eos_token_id,
    hidden_act,
    hidden_size,
    initializer_range,
    intermediate_size,
    max_position_embeddings,
    max_window_layers,
    model_type,
    num_attention_heads,
    num_hidden_layers,
    num_key_value_heads,
    rms_norm_eps,
    rope_theta,
    sliding_window,
    tie_word_embeddings,
    torch_dtype,
    transformers_version,
    use_cache,
    use_sliding_window,
    vocab_size
)

} // namespace llm_frame

    