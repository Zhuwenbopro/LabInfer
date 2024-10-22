#include "layers.h"
#include "Tensor.h"
#include <fstream>
#include <memory>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>  // 用于rand函数
#include <ctime>    // 用于时间种子


// ANSI color codes
#define RESET   "\033[0m"
#define RED     "\033[31m"      // Red
#define GREEN   "\033[32m"      // Green

#define N 4096  // 输入向量长度
#define D 4096   // 输出向量长度

std::unordered_map<std::string, std::shared_ptr<float []>> states;

void check_pass(const std::string& message);
void check_error(const std::string& message);
bool compare_results(const float *a, const float *b, int size, float tolerance = 1e-3);

void check_decoder();
void check_transform();

void read_bin(float* ptr, size_t num, const std::string& filename);
void test_layer(Layer& layer, Tensor& result, Tensor& x1, Tensor& x2, Tensor& real_result, float e = 1e-2);
void test_layer(Layer& layer, Tensor& result, Tensor& x, Tensor& real_result, float e = 1e-2);
void test_layer(Layer& layer, Tensor& result, Tensor& real_result, float e = 1e-2);
void load_layer(Layer& layer, const std::string filename, const std::vector<size_t>& weight_shape, const std::string map_name);

int main() {

    // check_transform();
    // check_decoder();
    const size_t batch = 1;
    const size_t seq = 6;
    const size_t vocab_size = 128256;
    const size_t hidden_state = 2048;
    int output_size = 6 * hidden_state;

    Embedding embedding = Embedding(vocab_size, hidden_state, "model.embed_tokens");
    embedding.load_state("./model.safetensors");

    Tensor x("embedding input", {batch, seq}, "cpu", true);
    x[0] = 128000;  x[1] = 791;     x[2] = 1401;
    x[3] = 311;     x[4] = 2324;    x[5] = 374;
    Tensor embedding_tensor("embedding output", {batch, hidden_state}, "cpu", true, {seq});

    Tensor embedding_check("result", {batch, hidden_state}, "cpu", true, {seq});
    read_bin(embedding_check, embedding_check.Size(), "embedding_tensor.bin");

    embedding.forward(embedding_tensor, x);

    if (compare_results(embedding_tensor, embedding_check, embedding_check.Size())) {
        check_pass("[" + embedding.Name() +"] " + embedding.Device() + " results correct.");
    } else {
        check_error("[" + embedding.Name() +"] " + embedding.Device() + " results error!");
    }
    


    Tensor pos("position", {batch, seq}, "cpu", true);
    pos[0] = 0; pos[1] = 1; pos[2] = 2;
    pos[3] = 3; pos[4] = 4; pos[5] = 5;
    DecoderLayer decoder_layer = DecoderLayer("model.layers.0");
    decoder_layer.load_state("./model.safetensors");
    Tensor decoder_check("decoder_check", {batch, hidden_state}, "cpu", true, {seq});
    read_bin(decoder_check, decoder_check.Size(), "layer0_output.bin");

    decoder_layer.forward(embedding_tensor, embedding_tensor, pos);
    
    if (compare_results(embedding_tensor, decoder_check, decoder_check.Size(), 6e-2)) {
        check_pass("[" + decoder_layer.Name() +"] " + decoder_layer.Device() + " results correct.");
    } else {
        check_error("[" + decoder_layer.Name() +"] " + decoder_layer.Device() + " results error!");
    }

    // Attention attention("atten");
    // attention.load_state("./model.safetensors");
    return 0;
}

void check_pass(const std::string&  message){
    std::cout << GREEN << message << RESET << std::endl;
}

void check_error(const std::string&  message){
    std::cout << RED << message << RESET << std::endl;
}

float fabs(float c){
    return c >= 0 ?  c : -c;
}

bool compare_results(const float *a, const float *b, int size, float tolerance) {
    bool flag = true;
    for (int i = 0; i < size; ++i) {
        if (fabs(a[i] - b[i]) > tolerance) {
            std::cout << "Difference at index " << i << ": " << a[i] << " vs " << b[i] << std::endl;
            flag = false;
            // break;
        }
    }
    return flag;
}

// void check_decoder() {
//     const size_t batch = 1;
//     const size_t seq = 6;
//     const size_t vocab_size = 128256;
//     const size_t hidden_state = 2048;
//     const size_t middle_dim = 8192;


//     Parameter embed("weight", {hidden_state, vocab_size}, "cpu", true);
//     Parameter qw("qw", {hidden_state,hidden_state}, "cpu", true);
//     Parameter kw("kw", {hidden_state,512}, "cpu", true);
//     Parameter vw("vw", {hidden_state,512}, "cpu", true);
//     Parameter ow("ow", {hidden_state,hidden_state}, "cpu", true);
//     Parameter rms("rms", {hidden_state}, "cpu", true);
//     Parameter gate_w("gate_w", {hidden_state, middle_dim}, "cpu", true);
//     Parameter up_w("up_w", {hidden_state, middle_dim}, "cpu", true);
//     Parameter down_w("down_w", {middle_dim, hidden_state}, "cpu", true);
//     Parameter rms_post("rms", {hidden_state}, "cpu", true);

//     read_bin(embed, embed.Size(), "model_embed_tokens_weight.bin");
//     read_bin(qw, qw.Size(), "model_layers_0_self_attn_q_proj_weight.bin");
//     read_bin(kw, kw.Size(), "model_layers_0_self_attn_k_proj_weight.bin");
//     read_bin(vw, vw.Size(), "model_layers_0_self_attn_v_proj_weight.bin");
//     read_bin(ow, ow.Size(), "model_layers_0_self_attn_o_proj_weight.bin");
//     read_bin(rms, rms.Size(), "model_layers_0_input_layernorm_weight.bin");
//     read_bin(gate_w, gate_w.Size(), "model_layers_0_mlp_gate_proj_weight.bin");
//     read_bin(up_w, up_w.Size(), "model_layers_0_mlp_up_proj_weight.bin");
//     read_bin(down_w, down_w.Size(), "model_layers_0_mlp_down_proj_weight.bin");
//     read_bin(rms_post, rms_post.Size(), "model_layers_0_post_attention_layernorm_weight.bin");

//     std::unordered_map<std::string, std::shared_ptr<float []>> states;
//     states["Decoder.embed_tokens.weight"] = embed.sharedPtr();
//     states["Decoder.attention.q_linear.weight"] = qw.sharedPtr();
//     states["Decoder.attention.k_linear.weight"] = kw.sharedPtr();
//     states["Decoder.attention.v_linear.weight"] = vw.sharedPtr();
//     states["Decoder.attention.o_linear.weight"] = ow.sharedPtr();
//     states["Decoder.attention.RMSNorm.weight"] = rms.sharedPtr();
//     states["Decoder.mlp.gate_linear.weight"] = gate_w.sharedPtr();
//     states["Decoder.mlp.up_linear.weight"] = up_w.sharedPtr();
//     states["Decoder.mlp.down_linear.weight"] = down_w.sharedPtr();
//     states["Decoder.mlp.rms_post.weight"] = rms_post.sharedPtr();

//     DecoderLayer decoder_layer = DecoderLayer("Decoder");
//     decoder_layer.load_state(states);

//     Tensor embedding_tensor("result", {batch, hidden_state}, "cpu", true, {seq});
//     read_bin(embedding_tensor, embedding_tensor.Size(), "layer0_input.bin");
//     Tensor pos("position", {batch, seq}, "cpu", true);
//     pos[0] = 0; pos[1] = 1; pos[2] = 2;
//     pos[3] = 3; pos[4] = 4; pos[5] = 5;

//     Tensor input_cuda = embedding_tensor.copy();

//     Tensor decoder_check("decoder_check", {batch, hidden_state}, "cpu", true, {seq});
//     read_bin(decoder_check, decoder_check.Size(), "layer0_output.bin");
//     test_layer(decoder_layer, embedding_tensor, embedding_tensor, pos, decoder_check, 1e-1);

//     input_cuda.to("cuda");
//     decoder_layer.to("cuda");
//     test_layer(decoder_layer, input_cuda, input_cuda, pos, decoder_check, 1e-1);
//     return ;
// }

// void check_transform() {
//     std::cout << std::endl << "begin test attention transformer" << std::endl;

//     const size_t batch = 1;
//     const size_t seq = 6;
//     const size_t vocab_size = 128256;
//     const size_t hidden_state = 2048;
//     int output_size = 6 * hidden_state;

//     Parameter embed("weight", {hidden_state, vocab_size}, "cpu", true);
//     Parameter qw("qw", {hidden_state,hidden_state}, "cpu", true);
//     Parameter kw("kw", {hidden_state,512}, "cpu", true);
//     Parameter vw("vw", {hidden_state,512}, "cpu", true);
//     Parameter ow("ow", {hidden_state,hidden_state}, "cpu", true);
//     Parameter rms("rms", {hidden_state}, "cpu", true);
//     read_bin(embed, embed.Size(), "model_embed_tokens_weight.bin");
//     read_bin(qw, qw.Size(), "model_layers_0_self_attn_q_proj_weight.bin");
//     read_bin(kw, kw.Size(), "model_layers_0_self_attn_k_proj_weight.bin");
//     read_bin(vw, vw.Size(), "model_layers_0_self_attn_v_proj_weight.bin");
//     read_bin(ow, ow.Size(), "model_layers_0_self_attn_o_proj_weight.bin");
//     read_bin(rms, rms.Size(), "model_layers_0_input_layernorm_weight.bin");

//     std::unordered_map<std::string, std::shared_ptr<float []>> states;
//     states["embed_tokens.weight"] = embed.sharedPtr();
//     states["attention.q_linear.weight"] = qw.sharedPtr();
//     states["attention.k_linear.weight"] = kw.sharedPtr();
//     states["attention.v_linear.weight"] = vw.sharedPtr();
//     states["attention.o_linear.weight"] = ow.sharedPtr();
//     states["attention.RMSNorm.weight"] = rms.sharedPtr();


// // =============================================== Embedding =================================================
    
//     Embedding embedding = Embedding(vocab_size, hidden_state);
//     embedding.load_state(states);

//     Tensor x("embedding input", {batch, seq}, "cpu", true);
//     x[0] = 128000;  x[1] = 791;     x[2] = 1401;
//     x[3] = 311;     x[4] = 2324;    x[5] = 374;
//     Tensor embedding_tensor("embedding output", {batch, hidden_state}, "cpu", true, {seq});

//     Tensor embedding_check("result", {batch, hidden_state}, "cpu", true, {seq});
//     read_bin(embedding_check, embedding_check.Size(), "embedding_tensor.bin");

//     test_layer(embedding, embedding_tensor, x, embedding_check);

// // =============================================== Attention =================================================

//     Attention attention("attention");    
//     attention.load_state(states);

//     Tensor pos("position", {batch, seq}, "cpu", true);
//     pos[0] = 0; pos[1] = 1; pos[2] = 2;
//     pos[3] = 3; pos[4] = 4; pos[5] = 5;

//     Tensor attn_check = embedding_tensor.copy();
//     read_bin(attn_check, attn_check.Size(), "self_attn.bin");
//     test_layer(attention, embedding_tensor, embedding_tensor, pos, attn_check);

//     Tensor attn_output_cuda = embedding_check.copy();
//     attn_output_cuda.to("cuda");
//     attention.to("cuda");
//     test_layer(attention, attn_output_cuda, attn_output_cuda, pos, attn_check);

//     Manager& manager = Manager::getInstance();
//     Function& F_cpu = manager.getFunction("cpu");

//     F_cpu.add(embedding_tensor, embedding_tensor, embedding_check, embedding_tensor.elemLen(), embedding_tensor.elemNum());
//     read_bin(attn_check, attn_check.Size(), "mlp_input.bin");

//     if (compare_results(attn_check, embedding_tensor, attn_check.Size(), 6e-2)) {
//         check_pass("[residual] CPU  results correct.");
//     } else {
//         check_error("[residual] CPU results error!");
//     }

// // =================================================== MLP ====================================================

//     const size_t middle_dim = 8192;

//     Parameter gate_w("gate_w", {hidden_state, middle_dim}, "cpu", true);
//     Parameter up_w("up_w", {hidden_state, middle_dim}, "cpu", true);
//     Parameter down_w("down_w", {middle_dim, hidden_state}, "cpu", true);
//     Parameter rms_post("rms", {hidden_state}, "cpu", true);
//     read_bin(gate_w, gate_w.Size(), "model_layers_0_mlp_gate_proj_weight.bin");
//     read_bin(up_w, up_w.Size(), "model_layers_0_mlp_up_proj_weight.bin");
//     read_bin(down_w, down_w.Size(), "model_layers_0_mlp_down_proj_weight.bin");
//     read_bin(rms_post, rms_post.Size(), "model_layers_0_post_attention_layernorm_weight.bin");

//     states["mlp.gate_linear.weight"] = gate_w.sharedPtr();
//     states["mlp.up_linear.weight"] = up_w.sharedPtr();
//     states["mlp.down_linear.weight"] = down_w.sharedPtr();
//     states["mlp.rms_post.weight"] = rms_post.sharedPtr();

    
//     Tensor mlp_check("mlp_check", {batch, hidden_state}, "cpu", true, {seq});
//     read_bin(mlp_check, mlp_check.Size(), "down.bin");

//     Mlp mlp = Mlp("mlp");
//     mlp.load_state(states);
//     //mlp.forward(attn_output, attn_output);
//     Tensor attn_o_cuda = embedding_tensor.copy();
//     test_layer(mlp, embedding_tensor, embedding_tensor, mlp_check, 6e-2);

//     mlp.to("cuda");
//     attn_o_cuda.to("cuda");
//     test_layer(mlp, attn_o_cuda, attn_o_cuda, mlp_check, 6e-2);

//     return;
// }

void test_layer(Layer& layer, Tensor& result, Tensor& x1, Tensor& x2, Tensor& real_result, float e) {
    x1.to(layer.Device());
    result.to(layer.Device());

    auto start = std::chrono::high_resolution_clock::now();
    layer.forward(result, x1, x2);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Execution time: " << duration << " microseconds" << std::endl;

    if(layer.Device() != "cpu")
        result.to("cpu");
    
    if (compare_results(result, real_result, result.Size(), e)) {
        check_pass("[" + layer.Name() +"] " + layer.Device() + " results correct.");
    } else {
        check_error("[" + layer.Name() +"] " + layer.Device() + " results error!");
    }
}

void test_layer(Layer& layer, Tensor& result, Tensor& x, Tensor& real_result, float e) {
    x.to(layer.Device());
    result.to(layer.Device());

    auto start = std::chrono::high_resolution_clock::now();
    layer.forward(result, x);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Execution time: " << duration << " microseconds" << std::endl;

    if(layer.Device() != "cpu")
        result.to("cpu");
    
    if (compare_results(result, real_result, result.Size(), e)) {
        check_pass("[" + layer.Name() +"] " + layer.Device() + " results correct.");
    } else {
        check_error("[" + layer.Name() +"] " + layer.Device() + " results error!");
    }
}

void test_layer(Layer& layer, Tensor& result, Tensor& real_result, float e) {
    result.to(layer.Device());

    auto start = std::chrono::high_resolution_clock::now();
    layer.forward(result);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Execution time: " << duration << " microseconds" << std::endl;

    if(layer.Device() != "cpu")
        result.to("cpu");
    
    if (compare_results(result, real_result, result.Size(), e)) {
        check_pass("[" + layer.Name() +"] " + layer.Device() + " results correct.");
    } else {
        check_error("[" + layer.Name() +"] " + layer.Device() + " results error!");
    }
}

// void load_layer(Layer& layer, const std::string filename, const std::vector<size_t>& weight_shape, const std::string map_name) {
//     if(layer.Device() != "cpu"){
        
//     }
//     Parameter weight("weight", weight_shape, "cpu", true);
//     read_bin(weight, weight.Size(), filename);
//     states[map_name] = weight.sharedPtr();
//     layer.load_state(states);
// }

void read_bin(float* ptr, size_t num, const std::string& filename) {
    // 打开二进制文件
    std::ifstream weight_file(filename, std::ios::binary);
    if (!weight_file) {
        std::cerr << "无法打开文件 " << filename << std::endl;
        return;
    }

    // 检查文件大小
    weight_file.seekg(0, std::ios::end);
    std::streamsize file_size = weight_file.tellg();
    weight_file.seekg(0, std::ios::beg);

    if (file_size != static_cast<std::streamsize>(num * sizeof(float))) {
        std::cerr << "文件大小与预期不匹配" << std::endl;
        return;
    }

    // 读取数据
    weight_file.read((char*)ptr, num * sizeof(float));

    if (!weight_file) {
        std::cerr << "读取文件错误，仅读取了 " << weight_file.gcount() << " 字节" << std::endl;
        return;
    }

    weight_file.close();
}
