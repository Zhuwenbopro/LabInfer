#include "layers.h"
#include "Tensor.h"
#include <iostream>
#include <cstdlib>  // 用于rand函数
#include <ctime>    // 用于时间种子
#include <vector>
#include <chrono>


// ANSI color codes
#define RESET   "\033[0m"
#define RED     "\033[31m"      // Red
#define GREEN   "\033[32m"      // Green

#define N 4096  // 输入向量长度
#define D 4096   // 输出向量长度

void check_pass(const char* message);
void check_error(const char* message);
bool compare_results(const float *a, const float *b, int size, float tolerance);
void rand_init(float* ptr, int size);
void const_init(float* ptr, int size);


void check_linear();
void check_softmax();

int main() {
    /* 要测试 constructor、forward、load_state、to */
    
    check_linear();
    check_softmax();

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

bool compare_results(const float *a, const float *b, int size, float tolerance = 1e-3f) {
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

    Linear linear = Linear(size_in, size_out, false, "Linear1");

    // load state
    float* W = (float*)malloc(size_in * size_out * sizeof(float));

    // 初始化输入数据和权重
    rand_init(W, size_in * size_out);
    Tensor x("input", nullptr, {size_in}, "cpu", true);
    Tensor y_cpu("output cpu", nullptr, {size_out}, "cpu", true);
    rand_init(x, size_in);

    std::unordered_map<std::string, float*> states;
    states["W"] = W;
    linear.load_state(states);

    auto start = std::chrono::high_resolution_clock::now();
    linear.forward(y_cpu, x);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Execution time: " << duration << " microseconds" << std::endl;
    std::cout << "begin copy" << std::endl;
    Tensor y_cuda = y_cpu.copy();
    std::cout << "end copy" << std::endl;
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

    Softmax softmax = Softmax(size_in, "Softmac");

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

