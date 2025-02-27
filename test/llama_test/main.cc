#include "model/ParamLoader.h"
#include "model/Sampler.h"
#include "layers/layers.h"
#include "InputWarp.h"

int main() {
    const size_t batch_size = 6;
    const size_t vocab_size = 128256;
    const size_t attn_head = 32;
    const size_t kv_head = 8;
    const size_t hidden_size = 2048;
    const size_t intermediate_size = 8192;
    const size_t num_hidden_layers = 16;
    const size_t max_len = 250;
    const float epsilon = 1e-5;

    const size_t head_dim = hidden_size / attn_head;
    const size_t q_size  = head_dim * attn_head;
    const size_t kv_size = head_dim * kv_head;

    Layer* model = new LayerList("model");
    model->add_layer(new Embedding(vocab_size, hidden_size), "embed_tokens");
    Layer* decoders = new LayerList("layers");
    for(int i = 0; i < num_hidden_layers; i++) {
        decoders->add_layer(new DecoderLayer(attn_head, kv_head, hidden_size, intermediate_size, 250, epsilon), std::to_string(i));
    }
    model->add_layer(decoders, "layers");
    model->add_layer(new RMSNorm(hidden_size, epsilon), "norm");
    model->add_layer(new Linear(hidden_size, vocab_size), "lm_head");
    printf("loading weight...\n");
    
    ParamLoader loader(true);
    loader.load_param(model, "model.safetensors");
    printf("load finished\n");

    Tensor<int> input_ids(6, 1);
    input_ids[0] = 128000;  input_ids[1] = 791;     input_ids[2] = 1401; 
    input_ids[3] = 311;     input_ids[4] = 2324;    input_ids[5] = 374;

    InputWarp inputWarp(input_ids);
    InputWarp inputWarp1 = inputWarp;
    inputWarp1.to("cuda");
    Sampler sampler(0.7, 2, 0.7);

    printf("forward...\n");
    model->forward(inputWarp);
    printf("sampling...\n");
    sampler.sample(inputWarp);
    std::cout << inputWarp.output_ids[0] << std::endl;
    
    printf("cuda forward...\n");
    model->to("cuda");
    model->forward(inputWarp1);
    printf("cuda sampling...\n");
    sampler.to("cuda");
    sampler.sample(inputWarp1);
    std::cout << inputWarp1.output_ids[0] << std::endl;

    

}