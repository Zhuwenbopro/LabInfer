#ifndef LLAMA_H
#define LLAMA_H

#include "layers/layers.h"

class Llama : public Layer {
public:
    Llama() = delete;
    Llama(Config& config);
    Tensor forward(Tensor& x) override;
    virtual ~Llama() = default;

private:
    size_t hidden_size;
    size_t vocab_size;
    bool tie_weights;
};

Llama::Llama(Config& config) : Layer("cpu", "model") {

    hidden_size = config.get<size_t>("hidden_size");
    vocab_size = config.get<size_t>("vocab_size");
    tie_weights = config.get<bool>("tie_word_embeddings");
    size_t num_hidden_layers = config.get<size_t>("num_hidden_layers");

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

Tensor Llama::forward(Tensor& x) {
    
    Tensor y = layers.at("embed_tokens")->forward(x);
    
    y = layers.at("layers")->forward(y);
    
    y = layers.at("norm")->forward(y);

    y = y.tail();
    y = layers.at("lm_head")->forward(y);

    size_t batch_size = y.Uid().size();

    Tensor index(batch_size, 1, x.Device(), y.Uid(), y.SeqLen());
    F.get().max_index(index, y, vocab_size, batch_size);
    return index;
}

#endif