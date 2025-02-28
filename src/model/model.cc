#include "model/Model.h"
#include <regex>

Model::Model(const std::string& config_file, const std::string& model_file) {
    Config config(config_file);
    size_t hidden_size = config.get<size_t>("hidden_size");
    size_t vocab_size = config.get<size_t>("vocab_size");
    bool tie_weights = config.get<bool>("tie_word_embeddings");
    size_t num_hidden_layers = config.get<size_t>("num_hidden_layers");
    float epsilon = config.get<float>("rms_norm_eps");
    size_t middle_size = config.get<size_t>("intermediate_size");
    size_t head_dim = config.get<size_t>("head_dim");
    size_t attn_head = config.get<size_t>("num_attention_heads");
    size_t kv_head = config.get<size_t>("num_key_value_heads");

    // 下面这一段模型的构建，之后要根据输入 xxx.model 文件创建
    Layer* embed = new LayerList("model");
    embed->add_layer(new Embedding(vocab_size, hidden_size), "embed_tokens");

    Layer* decoders = new LayerList("model.layers");
    for(int i = 0; i < num_hidden_layers; i++) {
        decoders->add_layer(new DecoderLayer(attn_head, kv_head, hidden_size, middle_size, 250, epsilon), std::to_string(i));
    }
    
    Layer* backbone = new LayerList("model");
    backbone->add_layer(new RMSNorm(hidden_size, epsilon), "norm");
    backbone->add_layer(new Linear(hidden_size, vocab_size), "lm_head");
    // Sampling
    backbone->add_layer(new Max(vocab_size), "max");
    
    printf("loading weight...\n");
    // 加载参数
    std::unordered_map<std::string, std::shared_ptr<void>> state_map;
    this->load_state("model.safetensors", state_map, tie_weights);
    embed->load(state_map);
    decoders->load(state_map);
    backbone->load(state_map);
    
    // 读模型配置文件 xxx.model
    printf("queue\n");
    // std::vector<DeviceSection> device_sections = parse_model_file(model_file);
    device_sections.push_back(DeviceSection{std::string("cpu"), embed});
    device_sections.push_back(DeviceSection{std::string("cpu"), decoders});
    device_sections.push_back(DeviceSection{std::string("cpu"), backbone});

    // cpu : embed_tokens
    // cuda:0 : layers 0 layers 1 layers 2 layers 3 layers 4 layers 5 layers 6 layers 7
    // cuda:1 : layers 8 layers 9 layers 10 layers 11 layers 12 layers 13 layers 14 layers 15
    // cpu : norm lm_head
    // 建模型 靠着 DeviceSection.device = [DeviceSection.layer, ...]
    // 加载权重
    // 分配给 worker
    for(int i = 0; i <= device_sections.size(); i++)
        queues.push_back(std::make_shared<SafeQueue<InputWarp>>(std::to_string(i)));

    printf("workers\n");
    workers.push_back(std::make_unique<Worker>("embedding", queues[0], queues[1], embed));
    workers.push_back(std::make_unique<Worker>("decoders", queues[1], queues[2], decoders));
    workers.push_back(std::make_unique<Worker>("backbone", queues[2], queues[3], backbone));

}

void Model::stop() {
    for(int i = 0; i < workers.size(); i++) {
        workers[i].get()->stop();
    }
}

void Model::add_request(InputWarp& inputWarp) {
    if(queues.size() == 0)
        throw std::logic_error("no worker in working."); 
    queues[0]->push(inputWarp);
}

Model::~Model() {
    stop();
}
