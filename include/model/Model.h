#ifndef MODEL_H
#define MODEL_H

#include "Config.h"
#include "Worker.h"
#include "SafeQueue.h"


struct DeviceSection {
    std::string device;
    std::vector<std::string> layers;
};

std::vector<DeviceSection> parse_model_file(const std::string& filename);

class Model {
private:
    std::vector<std::shared_ptr<SafeQueue<std::string>>> queues;
    std::vector<std::unique_ptr<Worker>> workers;

public:
    Model(Config &config, std::string model_file) {
        size_t hidden_size = config.get<size_t>("hidden_size");
        size_t vocab_size = config.get<size_t>("vocab_size");
        bool tie_weights = config.get<bool>("tie_word_embeddings");
        size_t num_hidden_layers = config.get<size_t>("num_hidden_layers");
        
        float epsilon = config.get<float>("rms_norm_eps");
        size_t middle_size = config.get<size_t>("intermediate_size");
        size_t head_dim = config.get<size_t>("head_dim");
        size_t attn_head = config.get<size_t>("num_attention_heads");
        size_t kv_head = config.get<size_t>("num_key_value_heads");

        LayerList embed("model");
        embed.add_layer(new Embedding(vocab_size, hidden_size), "embed_tokens");

        LayerList decoders("model.layers");
        for(int i = 0; i < num_hidden_layers; i++) {
            decoders.add_layer(new DecoderLayer(attn_head, kv_head, hidden_size, middle_size, 250, epsilon), std::to_string(i));
        }

        LayerList backbone("model");
        backbone.add_layer(new RMSNorm(hidden_size, epsilon), "norm");
        backbone.add_layer(new Linear(hidden_size, vocab_size), "lm_head");


        // 读模型配置文件 xxx.model
        // std::vector<DeviceSection> device_sections = parse_model_file(model_file);

        // // cpu : embed_tokens
        // // cuda:0 : layers 0 layers 1 layers 2 layers 3 layers 4 layers 5 layers 6 layers 7
        // // cuda:1 : layers 8 layers 9 layers 10 layers 11 layers 12 layers 13 layers 14 layers 15
        // // cpu : norm lm_head

        // // 建模型 靠着 DeviceSection.device = [DeviceSection.layer, ...]

        // // 加载权重

        // // 分配给 worker
        // for(int i = 0; i <= device_sections.size(); i++)
        //     queues.push_back(std::make_shared<SafeQueue<std::string>>(std::to_string(i)));

        // for(int i = 0; i < device_sections.size(); i++) {
        //     auto section = device_sections[i];
        //     if(i == 0)
        //         workers.push_back(std::make_unique<Worker>(section.device, queues[i], queues[i+1], FIRST));
        //     else if(i == device_sections.size()-1)
        //         workers.push_back(std::make_unique<Worker>(section.device, queues[i], queues[0], LAST));
        //     else
        //         workers.push_back(std::make_unique<Worker>(section.device, queues[i], queues[i+1]));
        // }
    }

    void run() {

    }

    void add_request(const std::string& msg) {
            // std::vector<int> -> tensor
        if(queues.size() == 0)
            throw std::logic_error("no worker in working."); 
        queues[0]->push(msg);
        // queues[queues.size()-1]->push(msg);
    }
};

#endif