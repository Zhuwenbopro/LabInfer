#include "DeviceManager.h"
#include <iostream>
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
bool compare_results(const float *a, const float *b, int size, float tolerance);
void rand_init(float* ptr, int size);

void check_rmsnorm(Device *cpu, Device *cuda);
void check_matmul(Device *cpu, Device *cuda);
void check_softmax(Device *cpu, Device *cuda);



int main() {

    DeviceManager& manager = DeviceManager::getInstance();

    Device * cpu = manager.getDevice("cpu");
    Device * cuda = manager.getDevice("cuda");

    check_rmsnorm(cpu, cuda);
    check_matmul(cpu, cuda);
    check_softmax(cpu, cuda);
    

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

void const_init(float* ptr, int size, const float cst = 1.0f){
    for (int i = 0; i < size; ++i) {
        ptr[i] = cst;
    }    
}

void check_rmsnorm(Device *cpu, Device *cuda){
    // 分配主机内存
    float *input_cpu = cpu->allocate(N);
    float *weight_cpu = cpu->allocate(N);
    float *output_cpu = cpu->allocate(N);
    float *cuda_to_cpu = cpu->allocate(N);
    float *input_cuda = cuda->allocate(N);
    float *weight_cuda = cuda->allocate(N);
    float *output_cuda = cuda->allocate(N);

    input_cpu[1] = 0.5;
    // 初始化输入数据和权重
    rand_init(input_cpu, N);
    const_init(weight_cpu, N);

    cuda->move_in(input_cuda, input_cpu, N);
    cuda->move_in(weight_cuda, weight_cpu, N);

    // 调用 rmsnorm 函数
    const float epsilon = 1e-5;
    cuda->F->rmsnorm(output_cuda, input_cuda, weight_cuda, epsilon, N);
    cpu->F->rmsnorm(output_cpu, input_cpu, weight_cpu, epsilon, N);

    cuda->move_out(output_cuda, cuda_to_cpu, N);

    // 输出部分结果进行验证
    if (compare_results(cuda_to_cpu, output_cpu, N)) {
        check_pass("[rmsnorm] CUDA and CPU results match.");
    } else {
        check_error("[rmsnorm] CUDA and CPU results do not match!");
    }

    cpu->deallocate(input_cpu);
    cpu->deallocate(weight_cpu);
    cpu->deallocate(output_cpu);
    cpu->deallocate(cuda_to_cpu);
    cuda->deallocate(input_cuda);
    cuda->deallocate(weight_cuda);
    cuda->deallocate(output_cuda);
}

void check_matmul(Device *cpu, Device *cuda){
    // 分配主机内存
    float *x_cpu = cpu->allocate(N);
    float *w_cpu = cpu->allocate(D * N);
    float *xout_cpu = cpu->allocate(D);
    float *cuda_to_cpu = cpu->allocate(D);
    float *x_cuda = cuda->allocate(N);
    float *w_cuda = cuda->allocate(D * N);
    float *xout_cuda = cuda->allocate(D);
    

    // 初始化输入向量 x 和矩阵 w
    rand_init(x_cpu, N);
    rand_init(w_cpu, D * N);

    cuda->move_in(x_cuda, x_cpu, N);
    cuda->move_in(w_cuda, w_cpu, D * N);

    // 计算
    cuda->F->matmul(xout_cuda, x_cuda, w_cuda, N, D);
    cpu->F->matmul(xout_cpu, x_cpu, w_cpu, N, D);

    cuda->move_out(xout_cuda, cuda_to_cpu, D);

    // 比较结果
    if (compare_results(cuda_to_cpu, xout_cpu, D)) {
        check_pass("[matmul] CUDA and CPU results match.");
    } else {
        check_error("[matmul] CUDA and CPU results do not match!");
    }

    // 释放内存
    cpu->deallocate(x_cpu);
    cpu->deallocate(w_cpu);
    cpu->deallocate(xout_cpu);
    cpu->deallocate(cuda_to_cpu);
    cuda->deallocate(x_cuda);
    cuda->deallocate(w_cuda);
    cuda->deallocate(xout_cuda);
}

void check_softmax(Device *cpu, Device *cuda){
    // 分配主机内存
    float *x_cpu = cpu->allocate(N);
    float *cuda_to_cpu = cpu->allocate(N);
    float *x_cuda = cuda->allocate(N);

    rand_init(x_cpu, N);

    cuda->move_in(x_cuda, x_cpu, N);

    cuda->F->softmax(x_cuda, N);
    cpu->F->softmax(x_cpu, N);

    cuda->move_out(x_cuda, cuda_to_cpu, N);

    // 比较结果
    if (compare_results(cuda_to_cpu, x_cpu, D)) {
        check_pass("[softmax] CUDA and CPU results match.");
    } else {
        check_error("[softmax] CUDA and CPU results do not match!");
    }

    cpu->deallocate(x_cpu);
    cpu->deallocate(cuda_to_cpu);
    cuda->deallocate(x_cuda);
}