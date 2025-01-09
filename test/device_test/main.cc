#include "DeviceManager.h"

#include <chrono>
#include "../test.h"


#define N 8192  // 输入向量长度
#define D 8192   // 输出向量长度

DeviceManager& deviceManager = DeviceManager::getInstance();

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

    Device * cpu = deviceManager.getDevice("cpu");
    Device * cuda = deviceManager.getDevice("cuda");

    std::cout << "start testing ..." << std::endl;
    check_rmsnorm(cpu, cuda);
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

void const_init(float* ptr, int size, const float cst = 1.0f){
    for (int i = 0; i < size; ++i) {
        ptr[i] = cst;
    }    
}

void check_rmsnorm(Device *cpu, Device *cuda){
    Title("check_rmsnorm");
    int batch_size = 5;
    // 分配主机内存
    float *input_cpu = (float*)cpu->allocate(N * batch_size * sizeof(float));
    float *weight_cpu = (float*)cpu->allocate(N * batch_size * sizeof(float));
    float *cuda_to_cpu = (float*)cpu->allocate(N * batch_size * sizeof(float));
    float *input_cuda = (float*)cuda->allocate(N * batch_size * sizeof(float));
    float *weight_cuda = (float*)cuda->allocate(N * batch_size * sizeof(float));

    input_cpu[1] = 0.5;
    // 初始化输入数据和权重
    rand_init(input_cpu, N * batch_size);
    const_init(weight_cpu, N * batch_size);

    cuda->move_in(input_cuda, input_cpu, N * batch_size * sizeof(float));
    cuda->move_in(weight_cuda, weight_cpu, N * batch_size * sizeof(float));

    // 调用 rmsnorm 函数
    const float epsilon = 1e-5;
    cuda->F->rmsnorm(input_cuda, weight_cuda, N, batch_size, epsilon);
    cpu->F->rmsnorm(input_cpu, weight_cpu, N, batch_size, epsilon);

    cuda->move_out(input_cuda, cuda_to_cpu, N * batch_size * sizeof(float));

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
    Title("check_matmul");
    // 分配主机内存
    int batch_size = 200;
    float *x_cpu = (float*)cpu->allocate(N*batch_size * sizeof(float));
    float *w_cpu = (float*)cpu->allocate(D * N * batch_size * sizeof(float));
    float *xout_cpu = (float*)cpu->allocate(D * batch_size * sizeof(float));
    float *cuda_to_cpu = (float*)cpu->allocate(D * batch_size * sizeof(float));
    float *x_cuda = (float*)cuda->allocate(N * batch_size * sizeof(float));
    float *w_cuda = (float*)cuda->allocate(D * N * batch_size * sizeof(float));
    float *xout_cuda = (float*)cuda->allocate(D * batch_size * sizeof(float));
    

    // 初始化输入向量 x 和矩阵 w
    rand_init(x_cpu, N * batch_size);
    rand_init(w_cpu, D * N * batch_size);

    // float** x = new float*[batch_size];
    // float** y = new float*[batch_size];
    // for(int i = 0; i < batch_size; i++) {
    //     float* x_tmp = cpu->allocate(N);
    //     for(int j = 0; j < N; j++) {
    //         x_tmp[j] = x_cpu[i * N + j];
    //     }
    //     x[i] = cuda->allocate(N);
    //     y[i] = cuda->allocate(N);
    //     cuda->move_in(x[i], x_tmp, N);
    //     cpu->deallocate(x_tmp);
    // }
    

    cuda->move_in(x_cuda, x_cpu, N * batch_size * sizeof(float));
    cuda->move_in(w_cuda, w_cpu, D * N * batch_size * sizeof(float));

    cuda->F->matmul(xout_cuda, x_cuda, w_cuda, N, D, batch_size);
    cuda->F->matmul(xout_cuda, x_cuda, w_cuda, N, D, batch_size);
    // cuda->F->matmul(xout_cuda, x_cuda, w_cuda, N, D, batch_size);

    // cuda->F->matmul(y, x, w_cuda, N, D, batch_size);
    // cuda->F->matmul(y, x, w_cuda, N, D, batch_size);
    // cuda->F->matmul(y, x, w_cuda, N, D, batch_size);
    
    cpu->F->matmul(xout_cpu, x_cpu, w_cpu, N, D, batch_size);
    

    cuda->move_out(xout_cuda, cuda_to_cpu, D * batch_size * sizeof(float));

    // 比较结果
    if (compare_results(cuda_to_cpu, xout_cpu, D, 5e-2)) {
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
    Title("check_embedding");
    // 分配主机内存
    int vocal_size = 12000; 
    int dim = 2048;
    int seq = 4;
    int *x_cpu = (int*)cpu->allocate(seq * sizeof(int));
    float *w_cpu = (float*)cpu->allocate(vocal_size * dim * sizeof(float));
    float *xout_cpu = (float*)cpu->allocate(seq * dim * sizeof(float));
    float *cuda_to_cpu = (float*)cpu->allocate(seq * dim * sizeof(float));
    int *x_cuda = (int*)cuda->allocate(seq * sizeof(int));
    float *w_cuda = (float*)cuda->allocate(vocal_size * dim * sizeof(float));
    float *xout_cuda = (float*)cuda->allocate(seq * dim * sizeof(float));
    

    // 初始化输入向量 x 和矩阵 w
    rand_init(w_cpu, vocal_size * dim);
    x_cpu[0] = 255;
    x_cpu[1] = 3234;
    x_cpu[2] = 44;
    x_cpu[3] = 6326;

    cuda->move_in(x_cuda, x_cpu, seq * sizeof(float));
    cuda->move_in(w_cuda, w_cpu, vocal_size * dim * sizeof(float));

    // 计算
    cuda->F->embedding(xout_cuda, x_cuda, w_cuda, dim, seq);
    cpu->F->embedding(xout_cpu, x_cpu, w_cpu, dim, seq);

    cuda->move_out(xout_cuda, cuda_to_cpu, seq * dim * sizeof(float));

    // 比较结果
    if (compare_results(cuda_to_cpu, xout_cpu, D, 5e-2)) {
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
    Title("check_softmax");
    int batch_size = 20;
    // 分配主机内存
    float *x_cpu = (float*)cpu->allocate(N * batch_size * sizeof(float));
    float *cuda_to_cpu = (float*)cpu->allocate(N * batch_size * sizeof(float));
    float *x_cuda = (float*)cuda->allocate(N * batch_size * sizeof(float));

    rand_init(x_cpu, N * batch_size);

    cuda->move_in(x_cuda, x_cpu, N * batch_size * sizeof(float));

    // float** x = new float*[batch_size];
    // for(int i = 0; i < batch_size; i++) {
    //     float* x_tmp = cpu->allocate(N);
    //     for(int j = 0; j < N; j++) {
    //         x_tmp[j] = x_cpu[i * N + j];
    //     }
    //     x[i] = cuda->allocate(N);
    //     cuda->move_in(x[i], x_tmp, N);
    //     cpu->deallocate(x_tmp);
    // }


    cuda->F->softmax(x_cuda, N, batch_size);
    // cuda->F->softmax(x_cuda, N, batch_size);
    // cuda->F->softmax(x_cuda, N, batch_size);

    // cuda->F->softmax(x, N, batch_size);
    // cuda->F->softmax(x, N, batch_size);
    // cuda->F->softmax(x, N, batch_size);

    cpu->F->softmax(x_cpu, N, batch_size);

    cuda->move_out(x_cuda, cuda_to_cpu, N * batch_size * sizeof(float));

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
    Title("check_silu");
    int batch_size = 10;
    // 分配主机内存
    float *x_cpu = (float*)cpu->allocate(N * batch_size * sizeof(float));
    float *cuda_to_cpu = (float*)cpu->allocate(N * batch_size * sizeof(float));
    float *x_cuda = (float*)cuda->allocate(N * batch_size * sizeof(float));

    rand_init(x_cpu, N * batch_size);

    cuda->move_in(x_cuda, x_cpu, N * batch_size * sizeof(float));

    // float** x = new float*[batch_size];
    // for(int i = 0; i < batch_size; i++) {
    //     float* x_tmp = cpu->allocate(N);
    //     for(int j = 0; j < N; j++) {
    //         x_tmp[j] = x_cpu[i * N + j];
    //     }
    //     x[i] = cuda->allocate(N);
    //     cuda->move_in(x[i], x_tmp, N);
    //     cpu->deallocate(x_tmp);
    // }

    // 模拟的这个向量在第 20 的位置
    cuda->F->silu(x_cuda, N, batch_size);
    // cuda->F->silu(x_cuda, N, batch_size);
    // cuda->F->silu(x_cuda, N, batch_size);

    // cuda->F->silu(x, N, batch_size);
    // cuda->F->silu(x, N, batch_size);
    // cuda->F->silu(x, N, batch_size);

    // cuda->F->silu(x_cuda, N, batch_size);
    // cuda->F->silu(x_cuda, N, batch_size);
    // cuda->F->silu(x_cuda, N, batch_size);

    cpu->F->silu(x_cpu,N, batch_size);

    cuda->move_out(x_cuda, cuda_to_cpu, N * batch_size * sizeof(float));

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
    Title("check_add");
    int batch_size = 5;
    // 分配主机内存
    float *x1_cpu = (float*)cpu->allocate(N * batch_size * sizeof(float));
    float *x2_cpu = (float*)cpu->allocate(N * batch_size * sizeof(float));
    float *y_cpu = (float*)cpu->allocate(N * batch_size * sizeof(float));
    float *cuda_to_cpu = (float*)cpu->allocate(N * batch_size * sizeof(float));
    float *x1_cuda = (float*)cuda->allocate(N * batch_size * sizeof(float));
    float *x2_cuda = (float*)cuda->allocate(N * batch_size * sizeof(float));
    float *y_cuda = (float*)cuda->allocate(N * batch_size * sizeof(float));

    rand_init(x1_cpu, N * batch_size);
    rand_init(x2_cpu, N * batch_size);

    // float** y = (float**)cpu->allocate(batch_size);
    // for(int i = 0; i < batch_size; i++) {
    //     y[i] = cuda->allocate(N);
    // }

    // float** x1 = (float**)cpu->allocate(batch_size);
    // float** x2 = (float**)cpu->allocate(batch_size);
    // for(int i = 0; i < batch_size; i++) {
    //     float* _tmp = cpu->allocate(N);
    //     for(int j = 0; j < N; j++) {
    //         _tmp[j] = x1_cpu[i * N + j];
    //     }
    //     x1[i] = cuda->allocate(N);
    //     cuda->move_in(x1[i], _tmp, N);
    //     for(int j = 0; j < N; j++) {
    //         _tmp[j] = x2_cpu[i * N + j];
    //     }
    //     x2[i] = cuda->allocate(N);
    //     cuda->move_in(x2[i], _tmp, N);
    //     cpu->deallocate(_tmp);
    // }

    cuda->move_in(x1_cuda, x1_cpu, N * batch_size * sizeof(float));
    cuda->move_in(x2_cuda, x2_cpu, N * batch_size * sizeof(float));

    // 模拟的这个向量在第 20 的位置
    // cuda->F->add(y, x1, x2, N, batch_size);
    // cuda->F->add(y, x1, x2, N, batch_size);
    // cuda->F->add(y, x1, x2, N, batch_size);

    // cuda->F->add(y_cuda, x1_cuda, x2_cuda, N, batch_size);
    // cuda->F->add(y_cuda, x1_cuda, x2_cuda, N, batch_size);
    cuda->F->add(y_cuda, x1_cuda, x2_cuda, N, batch_size);

    cpu->F->add(y_cpu, x1_cpu, x2_cpu, N, batch_size);

    cuda->move_out(y_cuda, cuda_to_cpu, N * batch_size * sizeof(float));

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
    Title("check_elem_multiply");
    int batch_size = 5;
    int size = batch_size*N; // Define the size of the vectors

    // Allocate host memory
    float *x1_cpu = (float*)cpu->allocate(size * sizeof(float));
    float *x2_cpu = (float*)cpu->allocate(size * sizeof(float));
    float *y_cpu = (float*)cpu->allocate(size * sizeof(float));
    float *cuda_to_cpu = (float*)cpu->allocate(size * sizeof(float));

    // Allocate device memory
    float *x1_cuda = (float*)cuda->allocate(size * sizeof(float));
    float *x2_cuda = (float*)cuda->allocate(size * sizeof(float));
    float *y_cuda = (float*)cuda->allocate(size * sizeof(float));

    // Initialize input data
    rand_init(x1_cpu, size);
    rand_init(x2_cpu, size);

    // float** y = (float**)cpu->allocate(batch_size);
    // for(int i = 0; i < batch_size; i++) {
    //     y[i] = cuda->allocate(N);
    // }

    // float** x1 = (float**)cpu->allocate(batch_size);
    // float** x2 = (float**)cpu->allocate(batch_size);
    // for(int i = 0; i < batch_size; i++) {
    //     float* _tmp = cpu->allocate(N);
    //     for(int j = 0; j < N; j++) {
    //         _tmp[j] = x1_cpu[i * N + j];
    //     }
    //     x1[i] = cuda->allocate(N);
    //     cuda->move_in(x1[i], _tmp, N);
    //     for(int j = 0; j < N; j++) {
    //         _tmp[j] = x2_cpu[i * N + j];
    //     }
    //     x2[i] = cuda->allocate(N);
    //     cuda->move_in(x2[i], _tmp, N);
    //     cpu->deallocate(_tmp);
    // }

    // Move data to device
    cuda->move_in(x1_cuda, x1_cpu, size * sizeof(float));
    cuda->move_in(x2_cuda, x2_cpu, size * sizeof(float));

    // Call the element-wise multiplication function on both devices
    // cuda->F->elem_multiply(y, x1, x2, N, batch_size);
    // cuda->F->elem_multiply(y, x1, x2, N, batch_size);
    // cuda->F->elem_multiply(y, x1, x2, N, batch_size);

    // cuda->F->elem_multiply(y_cuda, x1_cuda, x2_cuda, size);
    // cuda->F->elem_multiply(y_cuda, x1_cuda, x2_cuda, size);
    cuda->F->elem_multiply(y_cuda, x1_cuda, x2_cuda, size);

    cpu->F->elem_multiply(y_cpu, x1_cpu, x2_cpu, size);

    // Move result back to host
    cuda->move_out(y_cuda, cuda_to_cpu, size * sizeof(float));

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
    Title("check_masked_attention");
    // Define parameters
    int dim = 64; // Dimension size
    int q_head = 32; // Number of query heads
    int kv_head = 32; // Number of key/value heads
    int position = 10; // Position index

    // Compute sizes
    int q_size  = q_head  * dim;
    int kv_size = kv_head * dim;

    // Allocate host memory
    float *q_cpu  =  (float*)cpu->allocate(q_size * position * sizeof(float));
    float *q_cuda = (float*)cuda->allocate(q_size * position * sizeof(float));

    float *k_cpu  =  (float*)cpu->allocate(kv_size* position * sizeof(float));
    float *k_cuda = (float*)cuda->allocate(kv_size* position * sizeof(float));

    float *v_cpu  =  (float*)cpu->allocate(kv_size* position * sizeof(float));
    float *v_cuda = (float*)cuda->allocate(kv_size* position * sizeof(float));

    float *y_cpu       =  (float*)cpu->allocate(q_size * position * sizeof(float));
    float *y_cuda      = (float*)cuda->allocate(q_size * position * sizeof(float));
    float *cuda_to_cpu =  (float*)cpu->allocate(q_size * position * sizeof(float));

    int *pos_cpu  =  (int*)cpu->allocate(position * sizeof(int));
    int *pos_cuda = (int*)cuda->allocate(position * sizeof(int));
    for(int i = 0; i < position; i++) pos_cpu[i] = i;


    // Initialize input data
    rand_init(q_cpu, q_size  * position);
    rand_init(k_cpu, kv_size * position);
    rand_init(v_cpu, kv_size * position);

    // Move data to device
    cuda->move_in(q_cuda, q_cpu, q_size * position * sizeof(float));
    cuda->move_in(k_cuda, k_cpu, kv_size* position * sizeof(float));
    cuda->move_in(v_cuda, v_cpu, kv_size* position * sizeof(float));
    cuda->move_in(pos_cuda, pos_cpu, position * sizeof(int));


    cpu->F->masked_attention(y_cpu, q_cpu, k_cpu, v_cpu, nullptr, pos_cpu, dim, q_head, position, position);
    cuda->F->masked_attention(y_cuda, q_cuda, k_cuda, v_cuda, nullptr, pos_cuda, dim, q_head, position, position);

    cuda->move_out(y_cuda, cuda_to_cpu, q_size * position * sizeof(float));

    // Compare results
    if (compare_results(cuda_to_cpu, y_cpu, q_size  * position)) {
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
    Title("check_max_index");
    int n = N;             // 每组数据的大小
    int num = 5;           // 组数（批量大小）
    
    // 分配主机内存
    float *x_cpu = (float*)cpu->allocate(n * num * sizeof(float));
    float *index_cpu = (float*)cpu->allocate(num * sizeof(float));
    float *cuda_to_cpu = (float*)cpu->allocate(num * sizeof(float));

    // 分配设备内存
    float *x_cuda = (float*)cuda->allocate(n * num * sizeof(float));
    float *index_cuda = (float*)cuda->allocate(num * sizeof(float));

    // 初始化输入数据
    rand_init(x_cpu, n * num);

    // float** x = new float*[num];
    // for(int i = 0; i < num; i++) {
    //     float* x_tmp = cpu->allocate(N);
    //     for(int j = 0; j < N; j++) {
    //         x_tmp[j] = x_cpu[i * N + j];
    //     }
    //     x[i] = cuda->allocate(N);
    //     cuda->move_in(x[i], x_tmp, N);
    //     cpu->deallocate(x_tmp);
    // }

    // 将输入数据从主机复制到设备
    cuda->move_in(x_cuda, x_cpu, n * num * sizeof(float));

    // 在设备和主机上分别调用 max_index 函数
    // cuda->F->max_index(index_cuda, x, n, num);
    // cuda->F->max_index(index_cuda, x, n, num);
    // cuda->F->max_index(index_cuda, x, n, num);

    // cuda->F->max_index(index_cuda, x_cuda, n, num);
    // cuda->F->max_index(index_cuda, x_cuda, n, num);
    cuda->F->max_index(index_cuda, x_cuda, n, num);
    
    cpu->F->max_index(index_cpu, x_cpu, n, num);

    // 将设备上的结果复制回主机
    cuda->move_out(index_cuda, cuda_to_cpu, num * sizeof(float));

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
//TODO：写 repeat_kv 的测试程序