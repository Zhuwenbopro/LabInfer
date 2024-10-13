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

void check_pass(const char* message);
void check_error(const char* message);
bool compare_results(const float *a, const float *b, int size, float tolerance = 1e-3);
void rand_init(float* ptr, int size);
void const_init(float* ptr, int size);

void check_embedding();
void check_linear();
void check_softmax();
void check_attention();

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

int main() {

    check_linear();
    check_softmax();
    check_embedding();
    check_attention();

    return 0;
}


void check_pass(const char*  message){
    std::cout << GREEN << message << RESET << std::endl;
}

void check_error(const char*  message){
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
        }
    }
    return flag;
}

void rand_init(float* ptr, int size){
    // 设置随机数种子
    std::srand(static_cast<unsigned int>(std::time(0)));

    for (int i = 0; i < size; ++i) {
        ptr[i] = static_cast<float>(rand()) / RAND_MAX;
    }    
}

void const_init(float* ptr, int size) {
    for (int i = 0; i < size; ++i) {
        ptr[i] = i;
    }    
}

void check_linear() {
    size_t size_in = 4096;
    size_t size_out = 2048;

    Linear linear = Linear(size_in, size_out);

    Parameter W("Linear.weight", {size_in, size_out}, "cpu", true);
    // 初始化输入数据和权重
    rand_init(W.rawPtr(), size_in * size_out);

    Tensor x("input", {size_in}, "cpu", true);
    Tensor y_cpu("output cpu", {size_out}, "cpu", true);
    rand_init(x.rawPtr(), size_in);

    std::unordered_map<std::string, std::shared_ptr<float []>> states;
    states["Linear.weight"] = W.sharedPtr();
    linear.load_state(states);

    auto start = std::chrono::high_resolution_clock::now();
    linear.forward(y_cpu, x);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Execution time: " << duration << " microseconds" << std::endl;
    Tensor y_cuda = y_cpu.copy();
    y_cuda.setName("op cuda");

    y_cuda.to("cuda");
    x.to("cuda");
    linear.to("cuda");
    start = std::chrono::high_resolution_clock::now();
    linear.forward(y_cuda, x);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Execution time: " << duration << " microseconds" << std::endl;

    y_cuda.to("cpu");
    if (compare_results(y_cuda, y_cpu, size_out, 1e-2)) {
        check_pass("[linear] CUDA and CPU results match.");
    } else {
        check_error("[linear] CUDA and CPU results do not match!");
    }

}

void check_softmax() {
    size_t size_in = 4096;
    Softmax softmax = Softmax(size_in, "Softmax");

    // 初始化输入数据和权重
    Tensor x_cpu("input", {size_in}, "cpu", true);
    rand_init(x_cpu.rawPtr(), size_in);
    Tensor x_cuda = x_cpu.copy();

    auto start = std::chrono::high_resolution_clock::now();
    softmax.forward(x_cpu);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Execution time: " << duration << " microseconds" << std::endl;

    x_cuda.to("cuda");
    softmax.to("cuda");
    start = std::chrono::high_resolution_clock::now();
    softmax.forward(x_cuda);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Execution time: " << duration << " microseconds" << std::endl;

    x_cuda.to("cpu");

    if (compare_results(x_cuda, x_cpu, size_in)) {
        check_pass("[softmax] CUDA and CPU results match.");
    } else {
        check_error("[softmax] CUDA and CPU results do not match!");
    }
}

void check_embedding() {
    std::cout << std::endl << "begin test embedding" << std::endl;
    const size_t dim0 = 128256;
    const size_t dim1 = 2048;
    const size_t total_elements = dim0 * dim1;
    int output_size = 6 * dim1;
    
    // 分配主机内存
    Parameter weight("weight", {dim1, dim0}, "cpu", true);
    read_bin(weight, total_elements, "model_embed_tokens_weight.bin");


    Tensor result("result", {6 * dim1}, "cpu", true);
    read_bin(result, output_size, "embedding_tensor.bin");   

    Tensor x("embedding input", {1, 6}, "cpu", true);
    x[0] = 128000;
    x[1] = 791;
    x[2] = 1401;
    x[3] = 311;
    x[4] = 2324;
    x[5] = 374;

    Embedding embedding = Embedding(dim0, dim1);
    std::unordered_map<std::string, std::shared_ptr<float []>> states;
    states["embed_tokens.weight"] = weight.sharedPtr();
    embedding.load_state(states);

    Tensor y_cpu("embedding output", {6, dim1}, "cpu", true);
    Tensor y_cuda = y_cpu.copy();

    auto start = std::chrono::high_resolution_clock::now();
    embedding.forward(y_cpu, x);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Execution time: " << duration << " microseconds" << std::endl;

    y_cuda.to("cuda");
    y_cuda.setName("y cuda");
    x.to("cuda");
    embedding.to("cuda");

    start = std::chrono::high_resolution_clock::now();
    embedding.forward(y_cuda, x);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Execution time: " << duration << " microseconds" << std::endl;

    y_cuda.to("cpu");


    if (compare_results(result, y_cpu, output_size)) {
        check_pass("[embedding] CPU results correct.");
    } else {
        check_error("[embedding] CPU results error!");
    }

    if (compare_results(y_cuda, result, output_size)) {
        check_pass("[embedding] CUDA results correct.");
    } else {
        check_error("[embedding] CUDA results error!");
    }
}

void check_attention() {
    std::cout << std::endl << "begin test attention" << std::endl;
    const size_t hidden_state = 2048;
    const size_t seq = 6;

    Tensor x_cpu("hidden_state", {seq, hidden_state}, "cpu", true);
    read_bin(x_cpu, seq * hidden_state, "embedding_tensor.bin");
    Tensor x_cuda = x_cpu.copy();

    Tensor result("result", {seq, hidden_state}, "cpu", true);
    read_bin(result, seq * hidden_state, "input_layernorm_output.bin");

    RMSNorm rms_norm = RMSNorm(hidden_state);
    Linear q_linear = Linear(hidden_state, hidden_state, "linear_q");
    Linear k_linear = Linear(hidden_state, 512, "linear_k");
    Linear v_linear = Linear(hidden_state, 512, "linear_v");

    Parameter weight("weight", {hidden_state}, "cpu", true);
    read_bin(weight, hidden_state, "model_layers_0_input_layernorm_weight.bin");
    std::unordered_map<std::string, std::shared_ptr<float []>> states;
    states["RMSNorm.weight"] = weight.sharedPtr();
    rms_norm.load_state(states);

    Parameter qw("weight", {hidden_state, hidden_state}, "cpu", true);
    Parameter kw("weight", {hidden_state, 512}, "cpu", true);
    Parameter vw("weight", {hidden_state, 512}, "cpu", true);
    read_bin(qw, hidden_state*hidden_state, "model_layers_0_self_attn_q_proj_weight.bin");
    read_bin(kw, 512*hidden_state, "model_layers_0_self_attn_k_proj_weight.bin");
    read_bin(vw, 512*hidden_state, "model_layers_0_self_attn_v_proj_weight.bin");


    states["linear_q.weight"] = qw.sharedPtr();
    states["linear_k.weight"] = kw.sharedPtr();
    states["linear_v.weight"] = vw.sharedPtr();

    q_linear.load_state(states);
    k_linear.load_state(states);
    v_linear.load_state(states);

    auto start = std::chrono::high_resolution_clock::now();
    rms_norm.forward(x_cpu);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Execution time: " << duration << " microseconds" << std::endl;
    
    if (compare_results(result, x_cpu, seq * hidden_state, 1e-2)) {
        check_pass("[RMSNorm] CPU results correct.");
    } else {
        check_error("[RMSNorm] CPU results error!");
    }

    Tensor q_result("q_result", {seq, hidden_state}, "cpu", true);
    read_bin(q_result, seq * hidden_state, "query_states.bin");
    Tensor k_result("k_result", {seq, 512}, "cpu", true);
    read_bin(k_result, seq * 512, "key_states.bin");
    Tensor v_result("v_result", {seq, 512}, "cpu", true);
    read_bin(v_result, seq * 512, "value_states.bin");

    Tensor q_result_cpu("q_result_cpu", {seq, hidden_state}, "cpu", true);
    Tensor k_result_cpu("k_result_cpu", {seq, 512}, "cpu", true);
    Tensor v_result_cpu("v_result_cpu", {seq, 512}, "cpu", true);

    Tensor q_result_cuda = q_result_cpu.copy();
    Tensor k_result_cuda = k_result_cpu.copy();
    Tensor v_result_cuda = v_result_cpu.copy();

    start = std::chrono::high_resolution_clock::now();
    q_linear.forward(q_result_cpu, x_cpu);
    k_linear.forward(k_result_cpu, x_cpu);
    v_linear.forward(v_result_cpu, x_cpu);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Execution time: " << duration << " microseconds" << std::endl;

    if (compare_results(q_result_cpu, q_result, seq * hidden_state, 1e-1)) {
        check_pass("[q_linear] CPU results correct.");
    } else {
        check_error("[q_linear] CPU results error!");
    }
    if (compare_results(k_result_cpu, k_result, seq * 512, 1e-1)) {
        check_pass("[k_linear] CPU results correct.");
    } else {
        check_error("[k_linear] CPU results error!");
    }
    if (compare_results(v_result_cpu, v_result, seq * 512, 1e-2)) {
        check_pass("[v_linear] CPU results correct.");
    } else {
        check_error("[v_linear] CPU results error!");
    }



    x_cuda.to("cuda");
    rms_norm.to("cuda");
    
    start = std::chrono::high_resolution_clock::now();
    rms_norm.forward(x_cuda);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Execution time: " << duration << " microseconds" << std::endl;

    q_linear.to("cuda");
    k_linear.to("cuda");
    v_linear.to("cuda");

    q_result_cuda.to("cuda");
    k_result_cuda.to("cuda");
    v_result_cuda.to("cuda");

    start = std::chrono::high_resolution_clock::now();
    q_linear.forward(q_result_cuda, x_cuda);
    k_linear.forward(k_result_cuda, x_cuda);
    v_linear.forward(v_result_cuda, x_cuda);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Execution time: " << duration << " microseconds" << std::endl;

    x_cuda.to("cpu");

    if (compare_results(x_cuda, result, seq * hidden_state, 1e-2)) {
        check_pass("[RMSNorm] CUDA results correct.");
    } else {
        check_error("[RMSNorm] CUDA results error!");
    }

    q_result_cuda.to("cpu");
    k_result_cuda.to("cpu");
    v_result_cuda.to("cpu");

    if (compare_results(q_result_cuda, q_result, seq * hidden_state, 6e-2)) {
        check_pass("[q_linear] CPU results correct.");
    } else {
        check_error("[q_linear] CPU results error!");
    }
    if (compare_results(k_result_cuda, k_result, seq * 512, 5e-2)) {
        check_pass("[k_linear] CPU results correct.");
    } else {
        check_error("[k_linear] CPU results error!");
    }
    if (compare_results(v_result_cuda, v_result, seq * 512, 1e-2)) {
        check_pass("[v_linear] CPU results correct.");
    } else {
        check_error("[v_linear] CPU results error!");
    }
}