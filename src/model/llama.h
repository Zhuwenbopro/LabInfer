#ifndef LLAMA_H
#define LLAMA_H

#include "layers.h"

class Llama : public Layer {
public:
    Llama() = delete;
    Llama(Config& config);
    void forward(Tensor& y, Tensor& x) override;
    virtual ~Llama() = default;

private:
    size_t hidden_size;
};

Llama::Llama(Config& config) : Layer("cpu", "model") {

    hidden_size = config.get("hidden_size").get<size_t>();
    layers.emplace("embed_tokens", new Embedding(config));
    layers.emplace("layers", new LayerList());
    
    size_t num_hidden_layers = config.get("num_hidden_layers").get<size_t>();
    for(size_t i = 0; i < num_hidden_layers; i++) {
        // 第一个 "i" 是decoderlayer的名字，第二个 "i" 是decoderlayer的层数
        layers.at("layers")->add_layer(new DecoderLayer(config, std::to_string(i)), std::to_string(i));
    }
}

void Llama::forward(Tensor& y, Tensor& x) {
    Tensor pos(x, 1);

    for(int i = 0, index = 0; i < x.batchSize(); i++) {
        for(int j = 0; j < x.Seq()[i]; j++, index++) {
            pos[index] = j;
        }
    }

    layers.at("embed_tokens")->forward(y, x);
    layers.at("layers")->forward(y, y, pos);
}

#endif