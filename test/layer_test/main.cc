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

void read_bin(float* ptr, size_t num, const std::string& filename);
void test_layer(Layer& layer, Tensor& result, Tensor& x1, Tensor& x2, Tensor& real_result, float e = 1e-2);
void test_layer(Layer& layer, Tensor& result, Tensor& x, Tensor& real_result, float e = 1e-2);
void test_layer(Layer& layer, Tensor& result, Tensor& real_result, float e = 1e-2);

int main() {

    Config config("config.json");

    const size_t batch = 1;
    const size_t seq = 6;
    const size_t vocab_size = 128256;
    const size_t hidden_state = 2048;
    int output_size = 6 * hidden_state;

    Embedding embedding = Embedding(config);
    embedding.setName("model.embed_tokens");
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
    DecoderLayer decoder_layer = DecoderLayer(config);
    decoder_layer.setName("model.layers.0");
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
