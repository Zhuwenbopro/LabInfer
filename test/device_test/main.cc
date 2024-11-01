#include "DeviceManager.h"
#include <iostream>
#include <cstdlib>  // 用于rand函数
#include <ctime>    // 用于时间种子
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

void check_rmsnorm(Device *cpu, Device *cuda);
void check_matmul(Device *cpu, Device *cuda);
void check_softmax(Device *cpu, Device *cuda);
void check_silu(Device *cpu, Device *cuda);
void check_add(Device *cpu, Device *cuda);
void check_embedding(Device *cpu, Device *cuda);
void check_elem_multiply(Device *cpu, Device *cuda);
void check_masked_attention(Device *cpu, Device *cuda);
void check_max_index(Device *cpu, Device *cuda);

int main() {

    DeviceManager& manager = DeviceManager::getInstance();

    Device * cpu = manager.getDevice("cpu");
    Device * cuda = manager.getDevice("cuda");

    std::cout << "start testing ..." << std::endl;
    //check_rmsnorm(cpu, cuda);
    check_matmul(cpu, cuda);
    check_softmax(cpu, cuda);
    check_silu(cpu, cuda);
    check_add(cpu, cuda);
    check_embedding(cpu, cuda);
    check_elem_multiply(cpu, cuda);
    check_masked_attention(cpu, cuda);
    check_max_index(cpu, cuda);
    std::cout << "test finished ..." << std::endl;

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
    int batch_size = 5;
    // 分配主机内存
    float *input_cpu = cpu->allocate(N * batch_size);
    float *weight_cpu = cpu->allocate(N * batch_size);
    float *cuda_to_cpu = cpu->allocate(N * batch_size);
    float *input_cuda = cuda->allocate(N * batch_size);
    float *weight_cuda = cuda->allocate(N * batch_size);

    input_cpu[1] = 0.5;
    // 初始化输入数据和权重
    rand_init(input_cpu, N * batch_size);
    const_init(weight_cpu, N * batch_size);

    cuda->move_in(input_cuda, input_cpu, N * batch_size);
    cuda->move_in(weight_cuda, weight_cpu, N * batch_size);

    // 调用 rmsnorm 函数
    const float epsilon = 1e-5;
    cuda->F->rmsnorm(input_cuda, weight_cuda, N, batch_size, epsilon);
    cpu->F->rmsnorm(input_cpu, weight_cpu, N, batch_size, epsilon);

    cuda->move_out(input_cuda, cuda_to_cpu, N * batch_size);

    // 输出部分结果进行验证
    if (compare_results(cuda_to_cpu, input_cpu, N * batch_size)) {
        check_pass("[rmsnorm] CUDA and CPU results match.");
    } else {
        check_error("[rmsnorm] CUDA and CPU results do not match!");
    }

    cpu->deallocate(input_cpu);
    cpu->deallocate(weight_cpu);
    cpu->deallocate(cuda_to_cpu);
    cuda->deallocate(input_cuda);
    cuda->deallocate(weight_cuda);
}



void check_matmul(Device *cpu, Device *cuda){
    // 分配主机内存
    int batch_size = 15;
    float *x_cpu = cpu->allocate(N*batch_size);
    float *w_cpu = cpu->allocate(D * N * batch_size);
    float *xout_cpu = cpu->allocate(D * batch_size);
    float *cuda_to_cpu = cpu->allocate(D * batch_size);
    float *x_cuda = cuda->allocate(N * batch_size);
    float *w_cuda = cuda->allocate(D * N * batch_size);
    float *xout_cuda = cuda->allocate(D * batch_size);
    

    // 初始化输入向量 x 和矩阵 w
    rand_init(x_cpu, N * batch_size);
    rand_init(w_cpu, D * N * batch_size);

    float** x = new float*[batch_size];
    float** y = new float*[batch_size];
    for(int i = 0; i < batch_size; i++) {
        float* x_tmp = cpu->allocate(N);
        for(int j = 0; j < N; j++) {
            x_tmp[j] = x_cpu[i * N + j];
        }
        x[i] = cuda->allocate(N);
        y[i] = cuda->allocate(N);
        cuda->move_in(x[i], x_tmp, N);
        cpu->deallocate(x_tmp);
    }
    

    cuda->move_in(x_cuda, x_cpu, N * batch_size);
    cuda->move_in(w_cuda, w_cpu, D * N * batch_size);

    cuda->F->matmul(y, x, w_cuda, N, D, batch_size);
    cuda->F->matmul(y, x, w_cuda, N, D, batch_size);
    cuda->F->matmul(y, x, w_cuda, N, D, batch_size);
    cuda->F->matmul(xout_cuda, x_cuda, w_cuda, N, D, batch_size);
    cuda->F->matmul(xout_cuda, x_cuda, w_cuda, N, D, batch_size);
    cuda->F->matmul(xout_cuda, x_cuda, w_cuda, N, D, batch_size);

    
    cpu->F->matmul(xout_cpu, x_cpu, w_cpu, N, D, batch_size);
    

    cuda->move_out(xout_cuda, cuda_to_cpu, D * batch_size);

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

void check_embedding(Device *cpu, Device *cuda){
    // 分配主机内存
    int vocal_size = 12000; 
    int dim = 2048;
    int seq = 4;
    float *x_cpu = cpu->allocate(seq);
    float *w_cpu = cpu->allocate(vocal_size * dim);
    float *xout_cpu = cpu->allocate(seq * dim);
    float *cuda_to_cpu = cpu->allocate(seq * dim);
    float *x_cuda = cuda->allocate(seq);
    float *w_cuda = cuda->allocate(vocal_size * dim);
    float *xout_cuda = cuda->allocate(seq * dim);
    

    // 初始化输入向量 x 和矩阵 w
    rand_init(w_cpu, vocal_size * dim);
    x_cpu[0] = 255;
    x_cpu[1] = 3234;
    x_cpu[2] = 44;
    x_cpu[3] = 6326;

    cuda->move_in(x_cuda, x_cpu, seq);
    cuda->move_in(w_cuda, w_cpu, vocal_size * dim);

    // 计算
    cuda->F->embedding(xout_cuda, x_cuda, w_cuda, dim, seq);
    cpu->F->embedding(xout_cpu, x_cpu, w_cpu, dim, seq);

    cuda->move_out(xout_cuda, cuda_to_cpu, seq * dim);

    // 比较结果
    if (compare_results(cuda_to_cpu, xout_cpu, D)) {
        check_pass("[embedding] CUDA and CPU results match.");
    } else {
        check_error("[embedding] CUDA and CPU results do not match!");
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
    int batch_size = 4;
    // 分配主机内存
    float *x_cpu = cpu->allocate(N * batch_size);
    float *cuda_to_cpu = cpu->allocate(N * batch_size);
    float *x_cuda = cuda->allocate(N * batch_size);

    rand_init(x_cpu, N * batch_size);

    cuda->move_in(x_cuda, x_cpu, N * batch_size);

    cuda->F->softmax(x_cuda, N, batch_size);
    cpu->F->softmax(x_cpu, N, batch_size);

    cuda->move_out(x_cuda, cuda_to_cpu, N * batch_size);

    // 比较结果
    if (compare_results(cuda_to_cpu, x_cpu, N * batch_size)) {
        check_pass("[softmax] CUDA and CPU results match.");
    } else {
        check_error("[softmax] CUDA and CPU results do not match!");
    }

    cpu->deallocate(x_cpu);
    cpu->deallocate(cuda_to_cpu);
    cuda->deallocate(x_cuda);
}

void check_silu(Device *cpu, Device *cuda){
    int batch_size = 4;
    // 分配主机内存
    float *x_cpu = cpu->allocate(N * batch_size);
    float *cuda_to_cpu = cpu->allocate(N * batch_size);
    float *x_cuda = cuda->allocate(N * batch_size);

    rand_init(x_cpu, N * batch_size);

    cuda->move_in(x_cuda, x_cpu, N * batch_size);

    // 模拟的这个向量在第 20 的位置
    cuda->F->silu(x_cuda, N, batch_size);
    cpu->F->silu(x_cpu,N, batch_size);

    cuda->move_out(x_cuda, cuda_to_cpu, N * batch_size);

    // 比较结果
    if (compare_results(cuda_to_cpu, x_cpu, N * batch_size)) {
        check_pass("[silu] CUDA and CPU results match.");
    } else {
        check_error("[silu] CUDA and CPU results do not match!");
    }

    cpu->deallocate(x_cpu);
    cpu->deallocate(cuda_to_cpu);
    cuda->deallocate(x_cuda);
}

void check_add(Device *cpu, Device *cuda){
    int batch_size = 5;
    // 分配主机内存
    float *x1_cpu = cpu->allocate(N * batch_size);
    float *x2_cpu = cpu->allocate(N * batch_size);
    float *y_cpu = cpu->allocate(N * batch_size);
    float *cuda_to_cpu = cpu->allocate(N * batch_size);
    float *x1_cuda = cuda->allocate(N * batch_size);
    float *x2_cuda = cuda->allocate(N * batch_size);
    float *y_cuda = cuda->allocate(N * batch_size);

    rand_init(x1_cpu, N * batch_size);
    rand_init(x2_cpu, N * batch_size);

    cuda->move_in(x1_cuda, x1_cpu, N * batch_size);
    cuda->move_in(x2_cuda, x2_cpu, N * batch_size);

    // 模拟的这个向量在第 20 的位置
    cuda->F->add(y_cuda, x1_cuda, x2_cuda, N, batch_size);
    cpu->F->add(y_cpu, x1_cpu, x2_cpu, N, batch_size);

    cuda->move_out(y_cuda, cuda_to_cpu, N * batch_size);

    // 比较结果
    if (compare_results(cuda_to_cpu, y_cpu, N * batch_size)) {
        check_pass("[add] CUDA and CPU results match.");
    } else {
        check_error("[add] CUDA and CPU results do not match!");
    }

    cpu->deallocate(x1_cpu);
    cpu->deallocate(x2_cpu);
    cpu->deallocate(y_cpu);
    cpu->deallocate(cuda_to_cpu);
    cuda->deallocate(x1_cuda);
    cuda->deallocate(x2_cuda);
    cuda->deallocate(y_cuda);
}

void check_elem_multiply(Device *cpu, Device *cuda) {
    int size = 1024; // Define the size of the vectors

    // Allocate host memory
    float *x1_cpu = cpu->allocate(size);
    float *x2_cpu = cpu->allocate(size);
    float *y_cpu = cpu->allocate(size);
    float *cuda_to_cpu = cpu->allocate(size);

    // Allocate device memory
    float *x1_cuda = cuda->allocate(size);
    float *x2_cuda = cuda->allocate(size);
    float *y_cuda = cuda->allocate(size);

    // Initialize input data
    rand_init(x1_cpu, size);
    rand_init(x2_cpu, size);

    // Move data to device
    cuda->move_in(x1_cuda, x1_cpu, size);
    cuda->move_in(x2_cuda, x2_cpu, size);

    // Call the element-wise multiplication function on both devices
    cuda->F->elem_multiply(y_cuda, x1_cuda, x2_cuda, size);
    cpu->F->elem_multiply(y_cpu, x1_cpu, x2_cpu, size);

    // Move result back to host
    cuda->move_out(y_cuda, cuda_to_cpu, size);

    // Compare results
    if (compare_results(cuda_to_cpu, y_cpu, size)) {
        check_pass("[elem_multiply] CUDA and CPU results match.");
    } else {
        check_error("[elem_multiply] CUDA and CPU results do not match!");
    }

    // Clean up
    cpu->deallocate(x1_cpu);
    cpu->deallocate(x2_cpu);
    cpu->deallocate(y_cpu);
    cpu->deallocate(cuda_to_cpu);
    cuda->deallocate(x1_cuda);
    cuda->deallocate(x2_cuda);
    cuda->deallocate(y_cuda);
}

void check_masked_attention(Device *cpu, Device *cuda) {
    // Define parameters
    int dim = 64; // Dimension size
    int q_head = 32; // Number of query heads
    int kv_head = 8; // Number of key/value heads
    int _pos = 10; // Position index

    // Compute sizes
    int q_size = q_head*dim;
    int kv_size = kv_head*dim;

    // Allocate host memory
    float *q_cpu = cpu->allocate(q_size);
    float *k_cpu = cpu->allocate(kv_size*(_pos+1));
    float *v_cpu = cpu->allocate(kv_size*(_pos+1));
    float *y_cpu = cpu->allocate(q_size);
    float *cuda_to_cpu = cpu->allocate(q_size);

    // Allocate device memory
    float *q_cuda = cuda->allocate(q_size);
    float *k_cuda = cuda->allocate(kv_size*(_pos+1));
    float *v_cuda = cuda->allocate(kv_size*(_pos+1));
    float *y_cuda = cuda->allocate(q_size);

    // Initialize input data
    rand_init(q_cpu, q_size);
    rand_init(k_cpu, kv_size*(_pos+1));
    rand_init(v_cpu, kv_size*(_pos+1));

    // Move data to device
    cuda->move_in(q_cuda, q_cpu, q_size);
    cuda->move_in(k_cuda, k_cpu, kv_size*(_pos+1));
    cuda->move_in(v_cuda, v_cpu, kv_size*(_pos+1));


    cpu->F->maksed_attention(y_cpu, q_cpu, k_cpu, v_cpu, dim, q_head, kv_head, _pos);
    cuda->F->maksed_attention(y_cuda, q_cuda, k_cuda, v_cuda, dim, q_head, kv_head, _pos);
    // Call the masked attention function on both devices
    //cuda->F->maksed_attention(y_cuda, q_cuda, k_cuda, v_cuda, dim, q_head, kv_head, _pos);
    //cpu->F->masked_attention(y_cpu, q_cpu, k_cpu, v_cpu, dim, q_head, kv_head, _pos);

    // Move result back to host
    cuda->move_out(y_cuda, cuda_to_cpu, q_size);

    // Compare results
    if (compare_results(cuda_to_cpu, y_cpu, q_size)) {
        check_pass("[masked_attention] CUDA and CPU results match.");
    } else {
        check_error("[masked_attention] CUDA and CPU results do not match!");
    }

    // Clean up
    cpu->deallocate(q_cpu);
    cpu->deallocate(k_cpu);
    cpu->deallocate(v_cpu);
    cpu->deallocate(y_cpu);
    cpu->deallocate(cuda_to_cpu);
    cuda->deallocate(q_cuda);
    cuda->deallocate(k_cuda);
    cuda->deallocate(v_cuda);
    cuda->deallocate(y_cuda);
}

void check_max_index(Device *cpu, Device *cuda) {
    int n = N;             // 每组数据的大小
    int num = 5;           // 组数（批量大小）
    
    // 分配主机内存
    float *x_cpu = cpu->allocate(n * num);
    float *index_cpu = cpu->allocate(num);
    float *cuda_to_cpu = cpu->allocate(num);

    // 分配设备内存
    float *x_cuda = cuda->allocate(n * num);
    float *index_cuda = cuda->allocate(num);

    // 初始化输入数据
    rand_init(x_cpu, n * num);

    // 将输入数据从主机复制到设备
    cuda->move_in(x_cuda, x_cpu, n * num);

    // 在设备和主机上分别调用 max_index 函数
    cuda->F->max_index(index_cuda, x_cuda, n, num);
    cpu->F->max_index(index_cpu, x_cpu, n, num);

    // 将设备上的结果复制回主机
    cuda->move_out(index_cuda, cuda_to_cpu, num);

    // 比较结果
    if (compare_results(cuda_to_cpu, index_cpu, num)) {
        check_pass("[max_index] CUDA and CPU results match.");
    } else {
        check_error("[max_index] CUDA and CPU results do not match!");
    }

    // 释放内存
    cpu->deallocate(x_cpu);
    cpu->deallocate(index_cpu);
    cpu->deallocate(cuda_to_cpu);
    cuda->deallocate(x_cuda);
    cuda->deallocate(index_cuda);
}

//TODO：写 rope 的测试程序