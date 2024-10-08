#include "layers.h"
#include "Tensor.h"
#include <iostream>
#include <cstdlib>  // 用于rand函数
#include <ctime>    // 用于时间种子
#include <vector>

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

int main() {
    /* 要测试 constructor、forward、load_state、to */
    
    size_t size_in = 3;
    size_t size_out = 2;

    Linear linear = Linear(size_in, size_out, false, "Linear1");

    // load state
    float* W = new float[size_in * size_out];
    float* in = new float[size_in];
    // 初始化输入数据和权重
    const_init(in, size_in);
    const_init(W, size_in * size_out);
    Tensor tensor1("input", in, {size_in}, "cpu");
    std::vector<Tensor> inputs = {tensor1};

    std::unordered_map<std::string, float*> states;
    states["W"] = W;

    linear.load_state(states);

    auto res = linear.forward(inputs);

    float* result = res[0];

    for(int i = 0; i < size_in; i++)
        std::cout << in[i] << " ";

    std::cout << std::endl << std::endl;

    for(int i = 0; i < size_in * size_out; i++)
        std::cout << W[i] << " ";

    std::cout << std::endl << std::endl;

    for(int i = 0; i < size_out; i++)
        std::cout << result[i] << " ";

    return 0;
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
