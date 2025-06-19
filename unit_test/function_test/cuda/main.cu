#include "../test.h"
#include "CPU/CPUMemoryManager.h"
#include "CUDA/CUDAMemoryManager.h"


CPUMemoryManager *cpu = new CPUMemoryManager();
CUDAMemoryManager *cuda = new CUDAMemoryManager();

#define N 2048
#define D 2048


void check_matmul();
void check_rope();
void check_softmax();
void check_silu();
void check_addelem();
void check_multiplyelem();
void check_rmsnorm();
void check_maxindex();

int main () {
    check_matmul();
    check_rope();
    check_softmax();
    check_silu();
    check_addelem();
    check_multiplyelem();
    check_rmsnorm();
    check_maxindex();
    return 0;
}

void check_addelem()
{
    Title("check_addelem FP32");
    int batch_size = 10;
    int n = N;

    // 分配内存。这是一个原地操作 (y += x)，所以需要y的副本。
    void* y_cpu = cpu->allocate(n * batch_size * sizeof(float));
    void* x_cpu = cpu->allocate(n * batch_size * sizeof(float));
    void* cuda_to_cpu = cpu->allocate(n * batch_size * sizeof(float));
    void* y_cuda = cuda->allocate(n * batch_size * sizeof(float));
    void* x_cuda = cuda->allocate(n * batch_size * sizeof(float));

    // 初始化数据
    rand_init((float*)y_cpu, n * batch_size);
    rand_init((float*)x_cpu, n * batch_size);

    // 移动数据到设备
    cuda->move_in(y_cuda, y_cpu, n * batch_size * sizeof(float));
    cuda->move_in(x_cuda, x_cpu, n * batch_size * sizeof(float));

    // 声明和执行
    void cpu_fp32_add_elem_exec(void *y, void *x, int n, int num);
    void cuda_fp32_add_elem_exec(void *y, void *x, int n, int num);
    cpu_fp32_add_elem_exec(y_cpu, x_cpu, n, batch_size);
    cuda_fp32_add_elem_exec(y_cuda, x_cuda, n, batch_size);

    // 获取结果并比较
    cuda->move_out(cuda_to_cpu, y_cuda, n * batch_size * sizeof(float));

    if (compare_results((float*)cuda_to_cpu, (float*)y_cpu, n * batch_size)) {
        check_pass("[addelem FP32] CUDA and CPU results match.");
    } else {
        check_error("[addelem FP32] CUDA and CPU results do not match!");
    }

    // 释放内存
    cpu->deallocate(y_cpu);
    cpu->deallocate(x_cpu);
    cpu->deallocate(cuda_to_cpu);
    cuda->deallocate(y_cuda);
    cuda->deallocate(x_cuda);
}

void check_multiplyelem()
{
    Title("check_multiplyelem FP32");
    int batch_size = 10;
    int n = N;

    // 分配内存。这是一个原地操作 (y *= x)，所以需要y的副本。
    void* y_cpu = cpu->allocate(n * batch_size * sizeof(float));
    void* x_cpu = cpu->allocate(n * batch_size * sizeof(float));
    void* cuda_to_cpu = cpu->allocate(n * batch_size * sizeof(float));
    void* y_cuda = cuda->allocate(n * batch_size * sizeof(float));
    void* x_cuda = cuda->allocate(n * batch_size * sizeof(float));

    // 初始化数据
    rand_init((float*)y_cpu, n * batch_size);
    rand_init((float*)x_cpu, n * batch_size);

    // 移动数据到设备
    cuda->move_in(y_cuda, y_cpu, n * batch_size * sizeof(float));
    cuda->move_in(x_cuda, x_cpu, n * batch_size * sizeof(float));

    // 声明和执行
    void cpu_fp32_multiply_elem_exec(void *y, void *x, int n, int num);
    void cuda_fp32_multiply_elem_exec(void *y, void *x, int n, int num);
    cpu_fp32_multiply_elem_exec(y_cpu, x_cpu, n, batch_size);
    cuda_fp32_multiply_elem_exec(y_cuda, x_cuda, n, batch_size);

    // 获取结果并比较
    cuda->move_out(cuda_to_cpu, y_cuda, n * batch_size * sizeof(float));

    if (compare_results((float*)cuda_to_cpu, (float*)y_cpu, n * batch_size)) {
        check_pass("[multiplyelem FP32] CUDA and CPU results match.");
    } else {
        check_error("[multiplyelem FP32] CUDA and CPU results do not match!");
    }

    // 释放内存
    cpu->deallocate(y_cpu);
    cpu->deallocate(x_cpu);
    cpu->deallocate(cuda_to_cpu);
    cuda->deallocate(y_cuda);
    cuda->deallocate(x_cuda);
}

void check_rmsnorm()
{
    Title("check_rmsnorm FP32");
    int batch_size = 10;
    int n = N;
    float epsilon = 1e-5f;

    // 分配内存。RMSNorm是原地操作，需要x的副本。
    void* x_cpu = cpu->allocate(n * batch_size * sizeof(float));
    void* w_cpu = cpu->allocate(n * sizeof(float));
    void* cuda_to_cpu = cpu->allocate(n * batch_size * sizeof(float));
    void* x_cuda = cuda->allocate(n * batch_size * sizeof(float));
    void* w_cuda = cuda->allocate(n * sizeof(float));

    // 初始化数据
    rand_init((float*)x_cpu, n * batch_size);
    rand_init((float*)w_cpu, n);

    // 移动数据到设备
    cuda->move_in(x_cuda, x_cpu, n * batch_size * sizeof(float));
    cuda->move_in(w_cuda, w_cpu, n * sizeof(float));
    
    // 声明和执行
    void cpu_fp32_rmsnorm_exec(void *x, void *w, int n, int num, float e);
    void cuda_fp32_rmsnorm_exec(void *x, void *w, int n, int num, float e);
    cpu_fp32_rmsnorm_exec(x_cpu, w_cpu, n, batch_size, epsilon);
    cuda_fp32_rmsnorm_exec(x_cuda, w_cuda, n, batch_size, epsilon);

    // 获取结果并比较
    cuda->move_out(cuda_to_cpu, x_cuda, n * batch_size * sizeof(float));

    if (compare_results((float*)cuda_to_cpu, (float*)x_cpu, n * batch_size, 1e-5f)) {
        check_pass("[rmsnorm FP32] CUDA and CPU results match.");
    } else {
        check_error("[rmsnorm FP32] CUDA and CPU results do not match!");
    }

    // 释放内存
    cpu->deallocate(x_cpu);
    cpu->deallocate(w_cpu);
    cpu->deallocate(cuda_to_cpu);
    cuda->deallocate(x_cuda);
    cuda->deallocate(w_cuda);
}

void check_maxindex()
{
    Title("check_maxindex FP32");
    int batch_size = 10;
    int n = N;

    // 分配内存。这是一个out-of-place操作。
    void* x_cpu = cpu->allocate(n * batch_size * sizeof(float));
    void* index_cpu = cpu->allocate(batch_size * sizeof(int));
    void* cuda_to_cpu = cpu->allocate(batch_size * sizeof(int));
    void* x_cuda = cuda->allocate(n * batch_size * sizeof(float));
    void* index_cuda = cuda->allocate(batch_size * sizeof(int));

    // 初始化数据
    rand_init((float*)x_cpu, n * batch_size);

    // 移动数据到设备
    cuda->move_in(x_cuda, x_cpu, n * batch_size * sizeof(float));

    // 声明和执行
    void cpu_fp32_max_index_exec(int *index, void *x, int n, int num);
    void cuda_fp32_max_index_exec(int *index, void *x, int n, int num);
    cpu_fp32_max_index_exec((int*)index_cpu, x_cpu, n, batch_size);
    cuda_fp32_max_index_exec((int*)index_cuda, x_cuda, n, batch_size);

    // 获取结果并比较 (输出是整数，需要逐个比较)
    cuda->move_out(cuda_to_cpu, index_cuda, batch_size * sizeof(int));

    bool match = true;
    for (int i = 0; i < batch_size; ++i) {
        if (((int*)cuda_to_cpu)[i] != ((int*)index_cpu)[i]) {
            match = false;
            printf("Mismatch at index %d: CPU_index=%d, CUDA_index=%d\n", i, ((int*)index_cpu)[i], ((int*)cuda_to_cpu)[i]);
            break;
        }
    }

    if (match) {
        check_pass("[maxindex FP32] CUDA and CPU results match.");
    } else {
        check_error("[maxindex FP32] CUDA and CPU results do not match!");
    }
    
    // 释放内存
    cpu->deallocate(x_cpu);
    cpu->deallocate(index_cpu);
    cpu->deallocate(cuda_to_cpu);
    cuda->deallocate(x_cuda);
    cuda->deallocate(index_cuda);
}


void check_silu()
{
    Title("check_silu FP32");
    int batch_size = 10;
    // 分配主机内存
    void *x_cpu = cpu->allocate(N * batch_size * sizeof(float));
    void *cuda_to_cpu = cpu->allocate(N * batch_size * sizeof(float));
    void *x_cuda = cuda->allocate(N * batch_size * sizeof(float));

    rand_init((float*)x_cpu, N * batch_size);

    cuda->move_in(x_cuda, x_cpu, N * batch_size * sizeof(float));

    void cpu_fp32_silu_exec(void *x, int n, int num);
    void cuda_fp32_silu_exec(void *x, int n, int num);
    cpu_fp32_silu_exec(x_cpu, N, batch_size);
    cuda_fp32_silu_exec(x_cuda,N, batch_size);

    cuda->move_out(cuda_to_cpu, x_cuda, N * batch_size * sizeof(float));

    // 比较结果
    if (compare_results((float*)cuda_to_cpu, (float*)x_cpu, N * batch_size)) {
        check_pass("[silu FP32] CUDA and CPU results match.");
    } else {
        check_error("[silu FP32] CUDA and CPU results do not match!");
    }

    cpu->deallocate(x_cpu);
    cpu->deallocate(cuda_to_cpu);
    cuda->deallocate(x_cuda);
}

void check_softmax()
{
    Title("check_softmax FP32");
    int batch_size = 20;
    // 分配主机内存
    void *x_cpu = cpu->allocate(N * batch_size * sizeof(float));
    void *cuda_to_cpu = cpu->allocate(N * batch_size * sizeof(float));
    void *x_cuda = cuda->allocate(N * batch_size * sizeof(float));

    rand_init((float*)x_cpu, N * batch_size);
    cuda->move_in(x_cuda, x_cpu, N * batch_size * sizeof(float));

    void cpu_fp32_softmax_exec(void *x, int n, int num);
    void cuda_fp32_softmax_exec(void *x, int n, int num);
    cuda_fp32_softmax_exec(x_cuda, N, batch_size);
    cpu_fp32_softmax_exec(x_cpu, N, batch_size);

    cuda->move_out(cuda_to_cpu, x_cuda, N * batch_size * sizeof(float));

    // 比较结果
    if (compare_results((float*)cuda_to_cpu, (float*)x_cpu, N * batch_size)) {
        check_pass("[softmax FP32] CUDA and CPU results match.");
    } else {
        check_error("[softmax FP32] CUDA and CPU results do not match!");
    }

    cpu->deallocate(x_cpu);
    cpu->deallocate(cuda_to_cpu);
    cuda->deallocate(x_cuda);
}

void check_matmul()
{
    Title("check_matmul FP32");
    // 分配主机内存
    int batch_size = 200;
    void *x_cpu         =  cpu->allocate(N * batch_size * sizeof(float));
    void *w_cpu         =  cpu->allocate(D * N * sizeof(float));
    void *xout_cpu      =  cpu->allocate(D * batch_size * sizeof(float));
    void *cuda_to_cpu   =  cpu->allocate(D * batch_size * sizeof(float));
    void *x_cuda        = cuda->allocate(N * batch_size * sizeof(float));
    void *w_cuda        = cuda->allocate(D * N * sizeof(float));
    void *xout_cuda     = cuda->allocate(D * batch_size * sizeof(float));
    

    // 初始化输入向量 x 和矩阵 w
    rand_init((float*)x_cpu, N * batch_size);
    rand_init((float*)w_cpu, D * N);
    

    cuda->move_in(x_cuda, x_cpu, N * batch_size * sizeof(float));
    cuda->move_in(w_cuda, w_cpu, D * N * sizeof(float));

    void cuda_fp32_linear_exec(void *y, void *x, void *w, int W_in, int W_out, int num);
    void cpu_fp32_linear_exec(void *y, void *X, void *W, int W_in, int W_out, int num);
    cuda_fp32_linear_exec(xout_cuda, x_cuda, w_cuda, N, D, batch_size);
    cpu_fp32_linear_exec(xout_cpu, x_cpu, w_cpu, N, D, batch_size);
    

    cuda->move_out(cuda_to_cpu, xout_cuda, D * batch_size * sizeof(float));

    // 比较结果
    if (compare_results((float*)cuda_to_cpu, (float*)xout_cpu, D * batch_size, 5e-2)) {
        check_pass("[matmul FP32] CUDA and CPU results match.");
    } else {
        check_error("[matmul FP32] CUDA and CPU results do not match!");
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

void check_rope()
{
    Title("check_rope FP32");

    // 1. 定义 RoPE 参数
    int num = 100;      // 批处理大小 (batch size)
    int n = N;          // 总特征维度
    int head_dim = 128; // 每个头的维度 (必须能被 n 整除)

    if (n % head_dim != 0) {
        check_error("[RoPE FP32] Total dimension 'n' must be divisible by 'head_dim'.");
        return;
    }
    
    // 2. 分配内存
    // CPU 内存
    void* x_cpu = cpu->allocate(num * n * sizeof(float)); // 原始数据，用于CUDA
    void* pos_cpu = cpu->allocate(num * sizeof(int));
    void* inv_freq_cpu = cpu->allocate((head_dim / 2) * sizeof(float));
    void* cuda_to_cpu = cpu->allocate(num * n * sizeof(float));   // 用于存储从CUDA返回的结果

    // CUDA 内存
    void* x_cuda = cuda->allocate(num * n * sizeof(float));
    void* pos_cuda = cuda->allocate(num * sizeof(int));
    void* inv_freq_cuda = cuda->allocate((head_dim / 2) * sizeof(float));

    // 3. 初始化输入数据
    rand_init((float*)x_cpu, num * n);

    // 初始化位置向量
    for (int i = 0; i < num; ++i) {
        ((int*)pos_cpu)[i] = i; // 使用简单的序列位置
    }
    
    // 初始化逆频率向量
    float* inv_freq_ptr = (float*)inv_freq_cpu;
    for (int i = 0; i < head_dim / 2; ++i) {
        inv_freq_ptr[i] = 1.0f / powf(10000.0f, (float)(i * 2) / head_dim);
    }

    // 4. 将数据从主机移动到设备
    cuda->move_in(x_cuda, x_cpu, num * n * sizeof(float));
    cuda->move_in(pos_cuda, pos_cpu, num * sizeof(int));
    cuda->move_in(inv_freq_cuda, inv_freq_cpu, (head_dim / 2) * sizeof(float));

    // 5. 执行 CPU 和 CUDA 函数
    void cuda_fp32_rope_exec(void *x, int *pos, void *inv_freq, int n, int head_dim, int num);
    void cpu_fp32_rope_exec(void *x, int *pos, void *inv_freq, int n, int head_dim, int num);
    
    cuda_fp32_rope_exec(x_cuda, (int*)pos_cuda, inv_freq_cuda, n, head_dim, num);
    cpu_fp32_rope_exec(x_cpu, (int*)pos_cpu, inv_freq_cpu, n, head_dim, num);

    // 6. 将结果从设备移回主机
    cuda->move_out(cuda_to_cpu, x_cuda, num * n * sizeof(float));

    // 7. 比较结果
    // RoPE 涉及三角函数，容忍度可以稍微放宽，但1e-4通常足够
    if (compare_results((float*)cuda_to_cpu, (float*)x_cpu, num * n, 1e-4f)) {
        check_pass("[RoPE FP32] CUDA and CPU results match.");
    } else {
        check_error("[RoPE FP32] CUDA and CPU results do not match!");
    }

    // 8. 释放内存
    cpu->deallocate(x_cpu);
    cpu->deallocate(pos_cpu);
    cpu->deallocate(inv_freq_cpu);
    cpu->deallocate(cuda_to_cpu);
    cuda->deallocate(x_cuda);
    cuda->deallocate(pos_cuda);
    cuda->deallocate(inv_freq_cuda);
}