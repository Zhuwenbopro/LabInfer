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

    std::cout << std::endl << "begin test embedding" << std::endl;
    const size_t dim0 = 128256;
    const size_t dim1 = 2048;
    const size_t total_elements = dim0 * dim1;
    int output_size = 6 * dim1;
    
    // 分配主机内存
    float* weight = (float*)malloc(total_elements * sizeof(float));
    read_bin(weight, total_elements, "model_embed_tokens_weight.bin");

    float* embedding_tensor = new float[output_size];
    read_bin(embedding_tensor, output_size, "embedding_tensor.bin");   

    float* input = (float*)malloc(total_elements * sizeof(float));
    input[0] = 128000;
    input[1] = 791;
    input[2] = 1401;
    input[3] = 311;
    input[4] = 2324;
    input[5] = 374;
    Tensor x("embedding input", input, {1, 6}, "cpu");

    Embedding embedding = Embedding(dim0, dim1);
    std::unordered_map<std::string, float*> states;
    states["embed_tokens_weight"] = weight;
    embedding.load_state(states);

    float* o = (float*)malloc(output_size * sizeof(float));
    Tensor y_cpu("embedding output", o, {6, dim1}, "cpu");
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


    if (compare_results(embedding_tensor, y_cpu, output_size)) {
        check_pass("[embedding] CPU results correct.");
    } else {
        check_error("[embedding] CPU results error!");
    }

    if (compare_results(y_cuda, embedding_tensor, output_size)) {
        check_pass("[embedding] CUDA results correct.");
    } else {
        check_error("[embedding] CUDA results error!");
    }

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
    for (int i = 0; i < size; ++i) {
        if (fabs(a[i] - b[i]) > tolerance) {
            std::cout << "Difference at index " << i << ": " << a[i] << " vs " << b[i] << std::endl;
            return false;
        }
    }
    return true;
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

    Linear linear = Linear(size_in, size_out, false);
    // load state
    float* W = (float*)malloc(size_in * size_out * sizeof(float));

    // 初始化输入数据和权重
    rand_init(W, size_in * size_out);
    Tensor x("input", nullptr, {size_in}, "cpu", true);
    Tensor y_cpu("output cpu", nullptr, {size_out}, "cpu", true);
    rand_init(x, size_in);

    std::unordered_map<std::string, float*> states;
    states["Linear_W"] = W;
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
    if (compare_results(y_cuda, y_cpu, size_out)) {
        check_pass("[linear] CUDA and CPU results match.");
    } else {
        check_error("[linear] CUDA and CPU results do not match!");
    }

}

void check_softmax() {
    size_t size_in = 4096;

    Softmax softmax = Softmax(size_in, "Softmax");

    float* in = new float[size_in];
    // 初始化输入数据和权重
    rand_init(in, size_in);
    Tensor x_cpu("input", in, {size_in}, "cpu");
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

