#include "model/Model.h"
#include <regex>

Model::Model(const std::string& model_path) {

    std::cout << "initial start...\n";

    Config config(model_path + "/config.json");
    size_t hidden_size = config.get<size_t>("hidden_size");
    size_t vocab_size = config.get<size_t>("vocab_size");
    bool tie_weights = config.get<bool>("tie_word_embeddings");
    size_t num_hidden_layers = config.get<size_t>("num_hidden_layers");
    float epsilon = config.get<float>("rms_norm_eps");
    size_t intermediate_size = config.get<size_t>("intermediate_size");
    size_t head_dim = config.get<size_t>("head_dim");
    size_t attn_head = config.get<size_t>("num_attention_heads");
    size_t kv_head = config.get<size_t>("num_key_value_heads");

    model = new LayerList("model");
    model->add_layer(new Embedding(vocab_size, hidden_size), "embed_tokens");
    model->add_layer(new LayerList("layers"), "layers");
    for(int i = 0; i < num_hidden_layers; i++) {
        model->get_layer("layers")
             ->add_layer(new DecoderLayer(
                            attn_head, 
                            kv_head, 
                            hidden_size, 
                            intermediate_size, 
                            250, 
                            epsilon ), 
                        std::to_string(i));
    }
    model->add_layer(new RMSNorm(hidden_size, epsilon), "norm");
    model->add_layer(new Linear(hidden_size, vocab_size), "lm_head");
    model->add_layer(new Sampling(0.7, 10, 0.9), "sampling");

    paramLoader = ParamLoader(tie_weights);
    // TODO : 不同的模型 命名文件不同
    paramLoader.load_param(model, (model_path + "/model.safetensors").c_str());

    std::cout << "initial finished...\n";
}

std::vector<int> Model::infer(std::vector<int>& input_ids, int max_len) {
    InputWarp inputWarp(input_ids);
    inputWarp.to(device);
    std::vector<int> output_ids = std::vector<int>(max_len, 0);

    for(int i = 0; i < max_len; i++) {
        model->forward(inputWarp);
        inputWarp.input_ids = inputWarp.output_ids;
        Tensor<int> o = inputWarp.output_ids;
        o.to("cpu");
        output_ids[i] = o[0];
    }

    return output_ids;
}

void Model::to(const std::string &_device) {
    device = _device;
    if(model != nullptr) model->to(_device);
}

Model::~Model() { }
