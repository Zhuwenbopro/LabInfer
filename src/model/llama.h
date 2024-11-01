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
    size_t vocab_size;
    bool tie_weights;
};

Llama::Llama(Config& config) : Layer("cpu", "model") {

    hidden_size = config.get("hidden_size").get<size_t>();
    vocab_size = config.get("vocab_size").get<size_t>();
    tie_weights = config.get("tie_word_embeddings").get<bool>();
    size_t num_hidden_layers = config.get("num_hidden_layers").get<size_t>();

    layers.emplace("embed_tokens", new Embedding(config));
    layers.emplace("layers", new LayerList());
    for(size_t i = 0; i < num_hidden_layers; i++) {
        // 第一个 "i" 是decoderlayer的名字，第二个 "i" 是decoderlayer的层数
        layers.at("layers")->add_layer(new DecoderLayer(config, std::to_string(i)), std::to_string(i));
    }
    
    layers.emplace("norm", new RMSNorm(config));
    layers.at("norm")->setName("norm");

    layers.emplace("lm_head", new Linear(hidden_size, vocab_size, "lm_head"));
}

void Llama::forward(Tensor& y, Tensor& x) {
    // 根据 tensor 产生 position
    Tensor pos(x, 1);

    for(int i = 0, index = 0; i < x.batchSize(); i++) {
        for(int j = 0; j < x.Seq()[i]; j++, index++) {
            pos[index] = j;
        }
    }

    layers.at("embed_tokens")->forward(y, x);
    layers.at("layers")->forward(y, y, pos);
    layers.at("norm")->forward(y);

    std::cout << "111" << std::endl;
    Tensor x1(1, hidden_size, "cpu", y.Uid());
    for(int i = 0; i < hidden_size; i++) {
        x1[i] = y[(y.elemNum() - 1) * hidden_size + i];
    }
    std::cout << "vec" << std::endl;
    Tensor res(1, vocab_size, "cpu", y.Uid());
    layers.at("lm_head")->forward(res, x1);

    std::cout << "res" << std::endl;
    float index[10];
    F.get().max_index(index, res, vocab_size, 1);
    std::cout << index[0] << std::endl;
}

#endif